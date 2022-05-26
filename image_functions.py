# Author: Ivor Simpson, University of Sussex (i.simpson@sussex.ac.uk)
# Purpose: Image manipulation functions
import torch as t
import numpy as np
import nibabel as nib


def get_psf(psf_width=5, as_tensor=False):
    from scipy.io import loadmat

    mat = loadmat('./psf/FID64844_floret_8_noIR_4mm_TR120_1_rho1_filt_ham_170914_psf.mat')
    _psf = np.array(mat['rho1_g1'])

    psf_var = (psf_width - 1) // 2
    psf = _psf[31 - psf_var:32 + psf_var, 31 - psf_var: 32 + psf_var, 31 - psf_var: 32 + psf_var]
    psf = psf / np.sum(psf)
    if as_tensor:
        psf = t.reshape(t.tensor(psf, dtype=t.float32), (1, 1, psf_width, psf_width, psf_width))
    return psf


def transform_images(data, angles, translations, scale, im_shape, inverse=False, final_resample=False):
    """
    Transform the image data as a rigid body
    :param data: the original image data (N_SAMPLES, D, N)
    :param angles: A tensor of an axis-angle representation (N, (x,y,z,theta))
    :param translations: A tensor of the 3d translations (N, (x,y))
    :return: the transformed images (N, N_SAMPLES, im_shape)
    """
    N = data.shape[-1]
    D = data.shape[-2]

    data = t.reshape(data, (-1, D, N))
    N_SAMPLES = data.shape[0]

    # Move the batch to the front
    split_samples = t.split(data, 1, -1)
    reshaped_data = t.stack([x for x in split_samples], dim=0)

    angles[:, 0:3] = angles[:, 0:3]
    angle_sum = t.sqrt(t.sum(t.square(angles[:, 0:3]) + 1e-4, dim=-1))
    z = angles[:, 2] / angle_sum
    y = angles[:, 1] / angle_sum
    x = angles[:, 0] / angle_sum

    theta = angles[:, 3:4] / 1.0
    zeros = t.zeros_like(z)
    axis = t.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1)
    matrix = t.reshape(axis * theta, (-1, 3, 3))

    matrix = t.matrix_exp(matrix)

    translations = t.reshape(translations, (-1, 3, 1))
    if inverse:
        matrix = t.transpose(matrix, 1, 2) * scale
        translations = t.matmul(matrix, -translations)
    else:
        matrix = matrix / scale
    matrix = t.cat([matrix, translations], dim=-1)

    img = t.reshape(reshaped_data, (N * N_SAMPLES, 1) + im_shape)
    warped_grid = t.nn.functional.affine_grid(matrix, img.shape)
    if final_resample:
        out = t.nn.functional.grid_sample(img, warped_grid, padding_mode='border', align_corners=False, mode='nearest')
    else:
        out = t.nn.functional.grid_sample(img, warped_grid, padding_mode='border', align_corners=False)

    return out

def build_voxelwise_image(segs, tissue_means):
    """"
    Recreate voxelwise priors based on current tissue distribution
    """

    voxelwise_image = t.zeros((64 * 64 * 64)).to(tissue_means.device).detach().requires_grad_(False)
    print(tissue_means.shape, len(segs))
    for i in range(len(segs) - 1):
        voxelwise_image = voxelwise_image + segs[i].flatten() * tissue_means[0, i]

    return voxelwise_image

def build_voxelwise_priors(segs, tissue_mean_dist, slope_var=1e-4):
    """"
    Recreate voxelwise priors based on current tissue distribution.
    Assume signal is a weighted sum of normal random variables, rather than a mixture model.
    args: segs: a list of tensors, one for each tissue class, the final being a brain mask.
    args: tissue_mean_dist: the distribution (means and log_variances) of the tissue means for each type. stored in a tensor of shape [2, 3]
    returns: a voxelwise map of the mean and variances (not log variances) for the tissue means ([:,:,0]) and slopes ([:,:,1])
    """

    voxelwise_priors = t.zeros((2, 64 * 64 * 64, 1)).to(tissue_mean_dist.device).detach().requires_grad_(False)

    for i in range(len(segs) - 1):
        voxelwise_priors[0, :, 0] = voxelwise_priors[0, :, 0] + segs[i].flatten() * tissue_mean_dist[0, i]
        voxelwise_priors[1, :, 0] = voxelwise_priors[1, :, 0] + segs[i].flatten() * t.exp(tissue_mean_dist[1, i])

    return voxelwise_priors


def load_segmentations(path='data/CISC24691_sleep/', n_tissues=5, name='rCISC24691_20191031_001_008_MPRAGE.nii',
                       load_phantoms=False, mask_thresh=0.01):
    prob_maps = []
    for i in range(1, n_tissues+1):
        im = nib.load(path+'c' + str(i) + name)
        im_data = im.get_fdata()
        im_data = np.where(im_data > mask_thresh, im_data, np.zeros_like(im_data))
        prob_maps.append(im_data)

    non_phantom = np.ones_like(prob_maps[-1]>0)
    if load_phantoms:
        for i in range(1, 5):
            im = nib.load(path + str(i) + '_phant.nii')
            prob_maps.append(im.get_fdata())

            non_phantom = np.logical_and(non_phantom, prob_maps[-1] < mask_thresh)

    for i in range(n_tissues):
        prob_maps[i] = np.where(non_phantom, prob_maps[i], np.zeros_like(prob_maps[i]))

    # Create a mask of the unlikely voxels
    brain_mask = np.zeros_like(prob_maps[0])
    for m in prob_maps:
        brain_mask = brain_mask + (m > mask_thresh)

    brain_mask = np.float32(brain_mask > mask_thresh)

    prob_maps = [x*brain_mask for x in prob_maps]
    norm = np.sum(np.stack(prob_maps,0),0)+1e-6
    prob_maps = [x/norm for x in prob_maps]

    prob_maps.append(brain_mask)

    return prob_maps


def create_image_parameters(path, seg_name, slope_std_dev=0.0):
    """ Create the parameters of the synthetic images
    """
    # Get the tissue prior information
    # GM, WM, CSF stats have a mean and std-dev across the population
    # The std variability is the std-dev between voxels
    mean_concentrations, std_concentrations, std_variability = tissue_mean_priors()
    prob_maps = load_segmentations(path=path, name=seg_name)
    print(len(prob_maps))

    # Create the tissue means for an example
    tissue_means = mean_concentrations + np.random.randn(len(prob_maps) - 1) * std_concentrations

    # TODO: Think about slopes
    concentration_slopes = np.random.randn(*prob_maps[0].shape) * slope_std_dev

    # The mean at a given voxel is a sum (as given below)
    concentration_means = np.zeros(prob_maps[0].shape)

    for i in range(len(prob_maps) - 1):
        # For each voxel draw a random sample according to the tissue variability
        voxel_tissue_values = np.random.randn(*prob_maps[0].shape) * std_variability[i] + tissue_means[i]
        # Multiply by the class probability at each voxel
        concentration_means += voxel_tissue_values * prob_maps[i]

    return np.float32(concentration_means), np.float32(concentration_slopes), np.float32(tissue_means)


def create_random_images(path, seg_name, N=10, use_psf=False, rice_sigma=10, translation_std_dev=0.0,
                         angle_std_dev=0.0):
    """
    Create some synthetic images
    :param N: The number of images to create
    :param use_psf: Include the point-spread function
    :param rice_sigma: what sigma should be used for the rice distribution
    :return:
    """
    import torch as t
    from scipy.stats import rice

    concentration_means, concentration_slopes, tissue_means = create_image_parameters(path, seg_name)
    angles = []
    translations = []
    images = []

    for n in range(N):
        im_n = t.tensor(concentration_means + concentration_slopes * n, dtype=t.float32)
        # TODO: use the full foward model
        # Apply abs to make sure it's positive (could switch to softplus)
        # im_n = t.nn.Softplus()(im_n)
        im_n = t.abs(im_n)

        # Presume the first 2 images have no motion
        if n > 2:
            # Transform using torch code
            # TODO: Make the transforms non zero
            im_n = t.reshape(im_n, (1, -1, 1))
            rot_axis_angle = t.randn(1, 4)
            rot_axis_angle[0, -1] = t.randn(1, 1) * angle_std_dev
            angles.append(rot_axis_angle)

            translations.append(t.randn(1, 3) * translation_std_dev)
            im_n = transform_images(im_n, angles[-1], translations[-1], 1.0, (64, 64, 64))
        else:
            angles.append(t.zeros(1, 4))
            translations.append(t.zeros((1, 3)))

        im_n = t.reshape(im_n, (1, 1, 64, 64, 64))

        if use_psf:
            # Apply 3D convolutions of the PSF with the images
            psf = get_psf(5, True)
            im_n = t.conv3d(t.nn.functional.pad(im_n, (2, 2, 2, 2, 2, 2)), psf, stride=1)

        # Convert back to numpy
        im_n = im_n.detach().numpy()

        # Add random noise
        # TODO: Select a better scale of the noise
        scale = rice_sigma
        im_n_noisy = rice.rvs(im_n / scale, scale=scale)

        im_n_noisy = np.reshape(im_n_noisy, (64, 64, 64))
        images.append(im_n_noisy)

    images = np.stack(images, -1)
    angles = t.cat(angles, dim=0)
    translations = t.cat(translations, dim=0)

    true_means_slopes = np.stack([concentration_means.flatten(), concentration_slopes.flatten()], axis=-1)

    return images, true_means_slopes, angles, translations, tissue_means


def tissue_mean_priors(intensity_scalar=1000.0, means=(40, 30, 80, 50, 50), std_concentrations=(4, 4, )):
    """
    Define the tissue
    :return:
    """
    # 40, 30, 140
    mean_concentrations = np.array(means) / intensity_scalar
    std_concentrations = np.array([2, 2, 4, 5, 5]) / intensity_scalar # (np.array([0.5, 0.5, 1.5, 2.5, 2.5]) * 2.0) / intensity_scalar # np.array([2, 2, 4, 5, 5]) / intensity_scalar
    std_variability = np.array([2.5, 2.5, 4, 10, 10]) / intensity_scalar
    return mean_concentrations, std_concentrations, std_variability