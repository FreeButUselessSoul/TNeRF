# Transmitted NeRF

This is the implementation of NeurIPS 2022 paper "Transsmitted Neural Radiance Fields".

## Quick Start
1. Download the pretrained model of ERRNet at the official repository [link](https://github.com/Vandermode/ERRNet), and move `errnet_060_00463920.pt` to `ckpt_ERRNet/`.

2. Run the recurring edge estimation by `?`. By default, this command generates a folder `edge/` under the specified dataset folder, containing the estimated recurring edge of each viewpoint.

3. 