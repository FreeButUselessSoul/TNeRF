import os
from opt import get_opts
import torch
from torch import nn
from collections import defaultdict
from einops import repeat
from torch.utils.data import DataLoader
from datasets import dataset_dict
from torchvision import transforms as T
# models
# from models.nerf import *
from models.nerf_warp import *
from models.rendering_warp import *
# from models.encoder import *
from models.discriminator import discriminator
from random import randint
# from models.vae_components import EncoderBlock,ResNetEncoder
from torch.nn import functional as F
# optimizer, scheduler, visualization
from utils import *
# losses
from losses import loss_dict
# from stage1.unet_components import Enc
from datasets.ray_sample import FlexGridRaySampler,ImgToPatch
# metrics
from metrics import *
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger

torch.manual_seed(0)

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.discriminators = []
        self.use_D = hparams.use_D
        self.use_contrastive = hparams.contrastive
        self.valid,self.fake = torch.ones(1, 1),torch.zeros(1,1)
        # self.loss = loss_dict['depth'](coef=1)
        self.loss = loss_dict['nerft'](coef=1).to(self.device)
        if self.use_contrastive:
            self.contrastive_loss = loss_dict['contrastive']()
        self.mutex_loss = loss_dict['mutex']()
        self.x_kernel = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]])[None,None,:,:]
        self.models_to_train = []
        self.enc_out_dim=self.hparams.enc_out_dim
        self.latent_dim=self.hparams.latent_dim
        self.img2patch = ImgToPatch(FlexGridRaySampler(hparams.sample_wh,img_wh=hparams.img_wh,scale_anneal=5e-4), hparams.sample_wh)

        if len(self.use_D) > 0:
            self.discriminator = discriminator()
            self.discriminators += [self.discriminator]

        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRFWarp('coarse',
                                in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3,
                                in_channels_f=self.latent_dim)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRFWarp('fine',
                                  in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min,
                                  in_channels_f=self.latent_dim)
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts,proj_mat, full_code_B=None,pixels_i=None, **kwargs): # TODO
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            proj_mat=proj_mat,
                            full_code_B=full_code_B,
                            pixels_i=pixels_i[:,i:i+self.hparams.chunk],
                            **kwargs)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        kwargs['perturbation'] = self.hparams.data_perturb
        kwargs['img_wh'] = tuple(self.hparams.img_wh)
        kwargs['sample_wh'] = self.hparams.sample_wh
        kwargs['img2patch'] = self.img2patch
        kwargs['use_edge'] = self.hparams.use_edge
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)
        if self.train_dataset.use_edge:
            print("Using Recurring Edge Constraint (REC).")
        # self.images_dataset = dataset_dict['wholeImage']
        

    def configure_optimizers(self):
        self.optimizer_G = get_optimizer(self.hparams, self.models_to_train)
        scheduler_G = get_scheduler(self.hparams, self.optimizer_G)
        if len(self.use_D):
            hparams = self.hparams
            hparams.lr=1e-4
            self.optimizer_D = get_optimizer(hparams, self.discriminators)
            scheduler_D = get_scheduler(hparams, self.optimizer_D)
            return [self.optimizer_G,self.optimizer_D], [scheduler_G,scheduler_D]
        return [self.optimizer_G],[scheduler_G]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb,optimizer_idx=0):
        if self.hparams.use_patch:
            rays, rgbs, ts, original = batch['rays'], batch['rgbs'], batch['ts'], batch['original']
            full_code_B = batch['full_code_B']
            pixels_i = batch['pixels_i'].flatten(1,2)[:,:,None,:]
            # depth_gt = batch['depth_gt'].transpose(-1,-2).view(self.hparams.batch_size,1,self.hparams.sample_wh,self.hparams.sample_wh)
            proj_mat = batch['proj_mat']
            if 'edge' in batch.keys():
                edge = batch['edge'].transpose(-1,-2).view(self.hparams.batch_size,3,self.hparams.sample_wh,self.hparams.sample_wh)
            else:
                edge = None
            rays = rays.squeeze()
            rgbs = rgbs.transpose(-1,-2).view(self.hparams.batch_size,3,self.hparams.sample_wh,self.hparams.sample_wh)
            ts = ts.squeeze()
            original = original.transpose(-1,-2).view(self.hparams.batch_size,3,self.hparams.sample_wh,self.hparams.sample_wh)
        results = self(rays, ts,proj_mat,full_code_B,pixels_i=pixels_i)
        ts=ts.view(self.hparams.batch_size,1,self.hparams.sample_wh,self.hparams.sample_wh)
        # if 'B' in self.use_D or (self.use_contrastive and optimizer_idx == 0):
        for key in results.keys():
            shape = results[key].shape
            C = 1 if len(shape)==1 else shape[1]
            if shape[0] == self.hparams.batch_size*self.hparams.sample_wh ** 2:
                results[key] = results[key].t().view(C,self.hparams.batch_size,self.hparams.sample_wh,self.hparams.sample_wh).transpose(0,1)
        img_pred_static = results['_rgb_fine_static']
        img_pred_transient = results['_rgb_fine_transient']
        if optimizer_idx == 0 or len(self.use_D)==0: # Train generator
            loss_d = self.loss(results, rgbs, BG_edge=edge)
            pred = results['rgb_fine']
            loss = sum(l for l in loss_d.values())
            mut_loss = self.mutex_loss(img_pred_static*(1-results['beta']),img_pred_transient*results['beta'],self.x_kernel.to(self.device))
            loss += mut_loss
            self.log(f'mut/',mut_loss,prog_bar=True)

            for k, v in loss_d.items():
                self.log(f'train/{k}', v, prog_bar=True)
            self.log('train/loss', loss)
            with torch.no_grad():
                psnr_ = psnr(pred, original)
                if self.global_step % 250 == 0:
                    stack = torch.cat([img_pred_static,img_pred_transient*results['beta'],pred, original],0) # (3, 3, H, W)
                    self.logger.experiment.add_images('train/Bpred_Rpred_Mixpred_GT', stack, self.global_step)
            self.log('lr', get_learning_rate(self.optimizer_G))
            self.log('train/psnr', psnr_, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, ts, original = batch['rays'], batch['rgbs'], batch['ts'], batch['original']
        full_code_B = batch['full_code_B']
        proj_mat = batch['proj_mat']
        rays = rays[0]
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts[0]
        results = defaultdict(list)
        index_w, index_h = torch.meshgrid([torch.linspace(-1,1,self.hparams.img_wh[0]),
                                         torch.linspace(-1,1,self.hparams.img_wh[1])])
        pixels_i = torch.dstack([index_w,index_h]).flatten(0,1)[None,:,None,:].to(self.device)
        results = self(rays, ts, proj_mat, full_code_B,pixels_i=pixels_i, output_transient=False)
        log = dict()
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        W, H = self.hparams.img_wh
        stack = torch.stack([results[f'rgb_fine'].T.view(3,H,W), original.T.view(3,H,W)]) # (2, 3, H, W)
        self.logger.experiment.add_images('val/pred_GT', stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'].unsqueeze(0), original)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    os.makedirs(f'ckpts/{hparams.exp_name}/',exist_ok=True)
    checkpoint_callback = \
        ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}/',
                                               '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=1,
                        save_last=True)
        # save_top_k = -1:all models; 0:no model; k: top k models on monitor

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus=hparams.gpus,
                      gradient_clip_val=0.5,
                      accelerator='ddp' if (',' in hparams.gpus) else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if len(hparams.gpus)==1 else None)
                    #   profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)