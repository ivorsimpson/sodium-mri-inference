# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: #Enter feature name here

import torch as t
import numpy as np
from image_functions import get_psf, create_random_images, load_segmentations, transform_images, tissue_mean_priors
import nibabel as nib
from model import SodiumModel, SodiumModelType
from kornia.filters.sobel import spatial_gradient3d

from losses import IMAGE_LOSS_TYPE
from optimizer import SodiumOptimizer
from sklearn.linear_model import LinearRegression
import os


device = SodiumModel.get_device()
np.random.seed(0)

""" Experimental parameters """
# Number of data samples
N_ims = 20

batch_size = 10
no_appearance_updates = 2


transform_lr = 0.002
appearance_lr = 0.01
translate_std = 1e-2
rotate_std = 0.05*1

# Initial Noise level
sigma = 0.04

scalar = 100.0


# Model type
model_type = SodiumModelType.SHARED_TISSUE_VALUE

mask_thresh = 0.01
tissue_prior_weight = 1.0

# Noise model
im_loss_type = IMAGE_LOSS_TYPE.Rice

if im_loss_type == IMAGE_LOSS_TYPE.Gaussian:
    sigma = 0.05

# Use the PSF
use_psf = True
mask_nll = False

base_dir = 'data/'
base_op_dir = 'results/'

subjects = ['CISC24691', 'CISC24300', 'CISC24937', 'CISC24975']
types = ['sleep', 'awake']
seg_names = ['rCISC24691_20191031_001_008_MPRAGE.nii', 'rCISC24300_20191011_001_008_MPRAGE.nii',
             'rCISC24937_20191125_002_009_MPRAGE.nii','rCISC24975_20191126_001_008_MPRAGE.nii']

subj_idx = 0
type_idx = 0
folder = subjects[subj_idx]+'_'+types[type_idx]+'/'
seg_name = seg_names[subj_idx]

if type_idx == 1:
    rotate_std = rotate_std / 2
    translate_std = translate_std / 2
    sigma = sigma * 1

path = base_dir+folder
op_dir = base_op_dir + folder
if not os.path.exists(op_dir):
    os.makedirs(op_dir)

real_im = nib.load(path +'sodium4d.nii.gz')


old_header = None
if real_im is not None:
    old_header = real_im.header.copy()
    real_im = real_im.get_fdata()
    if False and type_idx == 0:
        real_im = real_im[:,:,:,:16]
    N_ims = real_im.shape[-1]

else:
    old_header = nib.load('data/c' + str(1) + 'rCISC24691_20191031_001_008_MPRAGE.nii').header.copy()


# Create the random images and get the parameters
images, true_m, true_angles, true_translations, true_tissue_means = create_random_images(path, seg_name,
                                                                                         N_ims, use_psf=use_psf,
                                                                                         rice_sigma=sigma,
                                                                                         translation_std_dev=2e-2,
                                                                                         angle_std_dev=1e-1)
if real_im is not None:
    images = np.clip(real_im, 1e-4, 1e3)
    images[np.isnan(images)] = 1e-4


true_tissue_means = t.Tensor(true_tissue_means).to(device)

true_m = t.Tensor(true_m).to(device)
true_translations = true_translations.to(device)
true_angle = true_angles.to(device)

images = t.Tensor(images).to(device)
images = t.reshape(images, (-1, N_ims))
D = 64 * 64 * 64
X = np.ones((D, N_ims, 1))
X = np.concatenate([X, X * np.reshape(np.arange(N_ims, dtype=np.float32), (1, N_ims, 1))], axis=-1)

if use_psf:
    psf = t.Tensor(get_psf(5, True)).to(device)
else:
    psf = None

segs = t.Tensor(load_segmentations(path=path, name=seg_name, n_tissues=3, mask_thresh=mask_thresh)).to(device)
brain_mask = segs[-1]

phantom_segs = t.Tensor(load_segmentations(path=path, name=seg_name, n_tissues=5, load_phantoms=True, mask_thresh=mask_thresh)).to(device)
orig_images = (images).clone().to(device)
images = t.clip(images, 1e-4, 1e3)



phantom_conc = (30/scalar, 50.0/scalar, 75.0/scalar,  120/scalar)

phantom_segs = phantom_segs[-4 -1:-1]

median_phantom_val = []
for phantom_idx in range(4):
    median_phantom_val.append(t.median(t.flatten(t.mean(images,-1))[t.reshape(phantom_segs, (4, -1))[phantom_idx]>0.1]).cpu().numpy())
    #mean_phantom_val = t.sum((t.reshape(t.mean(images,-1), (1, -1)) * t.reshape(phantom_segs, (4, -1))))/t.sum(t.reshape(phantom_segs, (4, -1)), -1)

median_phantom_val = np.array(median_phantom_val)
print(median_phantom_val)
order = np.argsort(median_phantom_val)
print(order)
ordered_phantom_conc = [phantom_conc[np.argmax(order==x)] for x in range(len(order))]
print('phantom ordering', median_phantom_val, ordered_phantom_conc, np.take_along_axis(np.array(phantom_conc), order,0))
lr = LinearRegression(fit_intercept=False)
lr.fit(np.array([ordered_phantom_conc]).reshape(-1, 1), median_phantom_val, sample_weight=(np.array([ordered_phantom_conc]).reshape(-1)-0.1))
print(lr.predict(np.array([ordered_phantom_conc]).reshape(-1, 1)))
sig_means = lr.predict(np.array([40, 30, 140, 60, 60]).reshape(-1, 1)/scalar)

std_concentrations = lr.predict(np.array([4, 4, 6, 10, 10]).reshape(-1, 1)/scalar)
tissue_mean_priors = (sig_means, std_concentrations)
print(tissue_mean_priors)


Y_t = images.to(device)

model = SodiumModel(segs, N_ims, tissue_mean_priors, sigma, model_type, im_loss_type, psf,
                           translate_std=translate_std,
                           rotate_std=rotate_std)
optimiser = SodiumOptimizer(model, transform_lr, appearance_lr, mask_nll)
print("Translations error:", t.mean(t.abs(true_translations - t.cat(model.translations,dim=0)[:, :3])))
print("Translation magnitude:", t.mean(t.abs(t.cat(model.translations,dim=0)[:, :3])))
print("Rotation magnitude:", t.mean(t.abs(t.cat(model.angles, dim=0)[:, 3])))
print('tissue means', model.tissue_distributions[0, :].cpu().detach().numpy())

init_appearance_updates = 20
if type_idx == 1:
    init_appearance_updates = 10
for k in range(init_appearance_updates):
    optimiser.update_appearance(Y_t, batch_size)
print('tissue means', model.tissue_distributions[0, :].cpu().detach().numpy())
for k in range(301):
    no_appearance_updates = 4
    if type_idx == 1:
        no_appearance_updates = 2
    optimiser.update_parameters(Y_t, batch_size, no_appearance_updates=no_appearance_updates)

    if k in []:#[10, 50, 90]:
        optimiser.sweep_rotations(Y_t, np.linspace(-0.15, 0.15, 10))

    with t.no_grad():
        if k % 20 == 0:


            print("Translation magnitude:", t.mean(t.abs(t.cat(model.translations, dim=0)[:, :3])).cpu().numpy())
            print("Rotation magnitude:", t.mean(t.abs(t.cat(model.angles, dim=0)[:, 3])).cpu().numpy())
            print('tissue means', model.tissue_distributions[0, :].cpu().detach().numpy())


            inv_rot = t.cat(model.angles, dim=0)[:, :4] * t.tensor(np.array([[1.0, 1.0, 1.0, -1.0]]), device=Y_t.device, dtype=t.float32)
            trans_ims = transform_images(orig_images, t.cat(model.angles, dim=0)[:, :4],
                                            t.cat(model.translations,dim=0)[:, :3],
                                            1.0 + t.tanh(model._scale)*0.1, (64, 64, 64), inverse=True,
                                            final_resample=True)
            print('std: ', 1e4*t.mean(t.std(trans_ims, 0)[0] * brain_mask))

            spatial_grad = spatial_gradient3d(t.mean(trans_ims, 0, keepdim=True))

            print('sharpness:', 1e4*t.sum(t.abs(spatial_grad * t.reshape(brain_mask, (1, 1, 1, 64, 64, 64))))/t.sum(brain_mask))

            spatial_grad = spatial_gradient3d(t.mean(trans_ims, 0, keepdim=True), order=2)

            print('sharpness:',
                  1e4 * t.sum(t.square(spatial_grad * t.reshape(brain_mask, (1, 1, 1, 64, 64, 64)))) / t.sum(brain_mask))

            trans_ims =  t.stack(t.split(trans_ims[:, 0,...], 1, 0), -1)

            array_img = nib.Nifti1Image(trans_ims[0].cpu().detach().numpy(), None, header=old_header)

            nib.save(array_img, op_dir+"reg_output.nii.gz")

            predictions = model.forward_model(np.arange(trans_ims.shape[-1]))
            array_img = nib.Nifti1Image(predictions.reshape(64, 64, 64, trans_ims.shape[-1]).cpu().detach().numpy(), None, header=old_header)
            nib.save(array_img, op_dir+"fwd_output.nii.gz")

            np.savez(op_dir+'transforms', **{'angles': t.cat(model.angles, dim=0)[:, :4].cpu().detach().numpy(),
                      'translations': t.cat(model.translations, dim=0)[:, :3].cpu().detach().numpy(),
                      'scale': 1.0 + t.tanh(model._scale).cpu().detach().numpy()*0.1,
                      'tissue_means': model.tissue_distributions[0, :].cpu().detach().numpy(),
                      'noise': model.noise_parameter.cpu().detach().numpy()
                      })