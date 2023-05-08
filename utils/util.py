from __future__ import print_function

import math
import os
import random

import numba
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from einops import rearrange
from scipy.interpolate import RegularGridInterpolator as rgi
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # return image_numpy.astype(imtype)

    n_img = min(image_tensor.shape[0], 16)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # if image_tensor.shape[1] == 4:
    # import pdb; pdb.set_trace()

    image_tensor = vutils.make_grid(image_tensor, nrow=4)

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.
    return image_numpy.astype(imtype)


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def interp3(V, xi, yi, zi, fill_value=0):
    x = np.arange(V.shape[0])
    y = np.arange(V.shape[1])
    z = np.arange(V.shape[2])
    interp_func = rgi((x, y, z), V, 'linear', False, fill_value)
    return interp_func(np.array([xi, yi, zi]).T)


def mesh_grid(input_lr, output_size):
    x_min, x_max, y_min, y_max, z_min, z_max = input_lr
    length = max(max(x_max - x_min, y_max - y_min), z_max - z_min)
    center = np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / 2.
    x = np.linspace(center[0] - length / 2, center[0] + length / 2, output_size[0])
    y = np.linspace(center[1] - length / 2, center[1] + length / 2, output_size[1])
    z = np.linspace(center[2] - length / 2, center[2] + length / 2, output_size[2])
    return np.meshgrid(x, y, z)


def thresholding(V, threshold):
    """
    return the original voxel in its bounding box and bounding box coordinates.
    """
    if V.max() < threshold:
        return np.zeros((2,2,2)), 0, 1, 0, 1, 0, 1
    V_bin = (V >= threshold)
    x_sum = np.sum(np.sum(V_bin, axis=2), axis=1)
    y_sum = np.sum(np.sum(V_bin, axis=2), axis=0)
    z_sum = np.sum(np.sum(V_bin, axis=1), axis=0)

    x_min = x_sum.nonzero()[0].min()
    y_min = y_sum.nonzero()[0].min()
    z_min = z_sum.nonzero()[0].min()
    x_max = x_sum.nonzero()[0].max()
    y_max = y_sum.nonzero()[0].max()
    z_max = z_sum.nonzero()[0].max()
    return V[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1], x_min, x_max, y_min, y_max, z_min, z_max


def downsample(vox_in, times, use_max=True):
    global downsample_uneven_warned
    if vox_in.shape[0] % times != 0 and not downsample_uneven_warned:
        print('WARNING: not dividing the space evenly.')
        downsample_uneven_warned = True
    return _downsample(vox_in, times, use_max=use_max)


@numba.jit(nopython=True, cache=True)
def _downsample(vox_in, times, use_max=True):
    dim = vox_in.shape[0] // times
    vox_out = np.zeros((dim, dim, dim))
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                subx = x * times
                suby = y * times
                subz = z * times
                subvox = vox_in[subx:subx + times,
                                suby:suby + times, subz:subz + times]
                if use_max:
                    vox_out[x, y, z] = np.max(subvox)
                else:
                    vox_out[x, y, z] = np.mean(subvox)
    return vox_out


def downsample_voxel(voxel, threshold, output_size, resample=True):
    if voxel.shape[0] > 100:
        assert output_size[0] in (32, 128)
        # downsample to 32 before finding bounding box
        if output_size[0] == 32:
            voxel = downsample(voxel, 4, use_max=True)
    if not resample:
        return voxel

    voxel, x_min, x_max, y_min, y_max, z_min, z_max = thresholding(
        voxel, threshold)
    x_mesh, y_mesh, z_mesh = mesh_grid(
        (x_min, x_max, y_min, y_max, z_min, z_max), output_size)
    x_mesh = np.reshape(np.transpose(x_mesh, (1, 0, 2)), (-1))
    y_mesh = np.reshape(np.transpose(y_mesh, (1, 0, 2)), (-1))
    z_mesh = np.reshape(z_mesh, (-1))

    fill_value = 0
    voxel_d = np.reshape(interp3(voxel, x_mesh, y_mesh, z_mesh, fill_value),
                         (output_size[0], output_size[1], output_size[2]))
    return voxel_d


def voxelize(x, thres=0.0):
    x_gt = x.clone().cpu()
    x_gt[x > thres] = 0.
    x_gt[x <= thres] = 1.
    return x_gt.numpy()


def iou_vox(x_gt_mask, x_mask):
    # x_gt_mask = x_gt.clone().detach()
    # x_mask = x.clone().detach()

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b d h w -> b (d h w)')
    union = rearrange(union, 'b d h w -> b (d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou
    # return torch.max(iou).item()


def iou(x_gt, x, thres):
    thres_gt = 0.0

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
    x_gt_mask[x_gt > thres_gt] = 0.
    x_gt_mask[x_gt <= thres_gt] = 1.

    x_mask = x.clone().detach()
    x_mask[x > thres] = 0.
    x_mask[x <= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    if len(x.shape) == 4:
        inter = rearrange(inter, 'b d h w -> b (d h w)')
        union = rearrange(union, 'b d h w -> b (d h w)')
    else:
        inter = rearrange(inter, 'b c d h w -> b (c d h w)')
        union = rearrange(union, 'b c d h w -> b (c d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    # return torch.max(iou).item()

    return iou


# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

# import torch
# import torch.nn.functional as F
def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


# Noam Learning rate schedule.
# From https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py
class NoamLR(_LRScheduler):

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
