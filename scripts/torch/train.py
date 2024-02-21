#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

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
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
import random
import argparse
import time
import numpy as np
import torch
import json
from matplotlib import pyplot as plt

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
torch.cuda.empty_cache()
import voxelmorph as vxm  # nopep8

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

#--for RandomfeaturesNet
parser.add_argument('--random-mode', type=bool, default=False,
                    help='Whether to run model with RandomFeaturesNet')
parser.add_argument('--filt-size', type=int, default=5,
                    help='Random Features Filter size(default: 5)')
parser.add_argument('--n-features', type=int, default=8,
                    help='Random Features n features(default: 8)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse , ncc or cl (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
parser.add_argument('--cl-type', default='dv',
                    help='type of contrastive loss')
args = parser.parse_args()

bidir = args.bidir

# load and prepare training data
#train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix, suffix=args.img_suffix)
with open(args.img_list) as f:
    dataset_json = json.load(f)
    train_files = dataset_json["training_paired_images"]
    val_files = dataset_json["registration_val"]

#print(train_files)

assert len(train_files) > 0, 'Could not find any training data.'
assert len(val_files) > 0, 'Could not find any validation data.'

val_steps_per_epoch = len(val_files)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

print(bidir)

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    generator = vxm.generators.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan_custom(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    val_generator = vxm.generators.scan_to_scan_custom(
        val_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape from sampled input
inshape = next(generator)[0][0].shape
print(inshape)
inshape = inshape[1:-1]
print(inshape)

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
#device ='cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
    
    if args.random_mode:
        print("Initalizing RandomFeaturesNet with: ", args.n_features,args.filt_size)
        fnet = vxm.networks.RandomFeatureNet(n_features = args.n_features, filt_size = args.filt_size, device = device).to(device)

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
model.train()

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print("The current loss type chosen is: ", args.image_loss)
# prepare image loss
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
elif args.image_loss == 'cl':
    image_loss_func = vxm.losses.CL(args.cl_type).loss
elif args.image_loss == 'nmi':
    image_loss_func = vxm.losses.MutualInformation()
else:
    raise ValueError('Image loss should be "mse", "ncc" or "cl", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]

loss_log = []
val_loss_log = []

# training loops
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        if args.random_mode:
            torch.save(fnet, os.path.join(model_dir, 'fnet_%04d.pt' % epochs))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    val_epoch_loss = []
    val_epoch_total_loss = []

    model.train()
    for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            if n==0:
                IF,JF = fnet(y_true[n],y_pred[n])
                curr_loss = loss_function(IF,JF) * weights[n]
            else:
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)
    
    model.eval()
    with torch.no_grad():
        for step in range(val_steps_per_epoch):

            # generate inputs (and true outputs) and convert them to tensors
            val_inputs, val_y_true = next(val_generator)
            val_inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in val_inputs]
            val_y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in val_y_true]

            # run inputs through the model to produce a warped image and flow field
            val_y_pred = model(*val_inputs)

            # calculate total loss
            val_loss = 0
            val_loss_list = []
            for n, loss_function in enumerate(losses):
                val_curr_loss = loss_function(val_y_true[n], val_y_pred[n]) * weights[n]
                val_loss_list.append(val_curr_loss.item())
                val_loss += val_curr_loss

            val_epoch_loss.append(val_loss_list)
            val_epoch_total_loss.append(val_loss.item())

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    val_loss_info = 'val loss: %.4e ' % (np.mean(val_epoch_total_loss))
    print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
    loss_log.append(np.mean(epoch_total_loss))
    val_loss_log.append(np.mean(val_epoch_total_loss))

    plt.plot(loss_log, color = 'blue')
    plt.plot(val_loss_log, color = 'green')
    plt.savefig(os.path.join(model_dir, 'loss_log.png'))


# final model save
model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
if args.random_mode:
    torch.save(fnet, os.path.join(model_dir, 'fnet_%04d.pt' % args.epochs))