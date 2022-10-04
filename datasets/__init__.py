from datasets.llff_0304 import LLFF0304
from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset
from .blender_patch import BlenderDatasetPatched
from .llff_patch import LLFFPatched
from .glass_synthetic import glass_syn
from .glass0407 import LLFFBerrRllff_G
from .glass0502 import LLFF_realG
from .llff_Bwarp import LLFF_BWarp
from .glass_ablation3 import glass_ablation3
from .llff_warpSpec import LLFF_WarpSpec
from .llff_Spec import LLFF_Spec

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism': PhototourismDataset,
                'blender_p': BlenderDatasetPatched,
                'llff_p': LLFFPatched,
                'glass_syn': glass_syn,
                'llff_Bwarp': LLFF_BWarp,
                'glass': LLFFBerrRllff_G,
                'real': LLFF_realG,
                'real_ravel': glass_ablation3,
                'llff_warpSpec':LLFF_WarpSpec,
                'llff_Spec':LLFF_Spec}