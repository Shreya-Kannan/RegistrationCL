#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register 
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt 
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in 
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or 
implied. See the License for the specific language governing permissions and limitations under 
the License.
"""

import os
import argparse

# third party
import numpy as np
import nibabel as nib
import torch
import json
import time

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8

# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--json', required=True, help='Json path')
parser.add_argument('--moved', help='warped image output directory')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation directory')
parser.add_argument('--set', required=True, help='Train/val/test set selection')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

# device handling
if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel

if args.set == 'train':
    eval_set = "training_paired_images"
    folder = 'imagesTr'
    label_folder = 'labelsTr'
elif args.set == 'test':
    eval_set = "test_paired_images"
    folder = 'imagesTs'
    label_folder = 'masksTs'
else:
    eval_set = "registration_val"
    folder = 'imagesTr'
    label_folder = 'labelsTr'

with open(args.json) as f:
    pairs = json.load(f)[eval_set]


# keep track of all dice scores
reg_times = []
dice_means = []

for i,pair in enumerate(pairs):
    moving = vxm.py.utils.load_volfile(pair["1"], add_batch_axis=True, add_feat_axis=add_feat_axis)
    fixed, fixed_affine = vxm.py.utils.load_volfile(
        pair["0"], add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
    moving_seg = vxm.py.utils.load_volfile(pair["1"].replace(folder,label_folder), add_batch_axis=True, add_feat_axis=add_feat_axis,normalize=False)
    fixed_seg = vxm.py.utils.load_volfile(pair["0"].replace(folder,label_folder), add_batch_axis=True, add_feat_axis=add_feat_axis,normalize=False)

    inshape = moving.shape[1:-1]
    print(inshape)

    # load and set up model
    model = vxm.networks.VxmDense.load(args.model, device)
    model.to(device)
    model.eval()

    # set up tensors and permute
    input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
    input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
    moving_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
    fixed_seg = torch.from_numpy(fixed_seg).to(device).float().permute(0, 4, 1, 2, 3)
    

    # predict warp and time
    start = time.time()
    moved, warp = model(input_moving, input_fixed, registration=True)
    reg_time = time.time() - start

    if i != 0:
        reg_times.append(reg_time)

    transformer = vxm.layers.SpatialTransformer(inshape)
    warped_seg = transformer(moving_seg, warp)
    #print(np.unique(warped_seg.detach().numpy().astype(int)))

    print("Warped shape:",warped_seg.shape)
    print("fixed_seg:",fixed_seg.shape)

    # compute volume overlap (dice)
    overlap = vxm.py.utils.dice(warped_seg.detach().cpu().numpy().astype(int), fixed_seg.detach().cpu().numpy())
    dice_means.append(np.mean(overlap))
    print('Pair %d    Reg Time: %.4f    Dice: %.4f +/- %.4f' % (i + 1, reg_time,
                                                                np.mean(overlap),
                                                                np.std(overlap)))

    """# save moved image
    if args.moved:
        moved = moved.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

    # save warp
    if args.warp:
        warp = warp.detach().cpu().numpy().squeeze()
        vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)"""
print()
print('Avg Reg Time: %.4f +/- %.4f  (skipping first prediction)' % (np.mean(reg_times),
                                                                    np.std(reg_times)))
print('Avg Dice: %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))
