import os

import imageio
import numpy as np
import torch
import torchvision.utils as vutils
from pytorch3d import structures

from .util_3d import sdf_to_mesh, render_mesh, rotate_mesh_360


def make_batch(data, B=16):
    x = data['sdf']
    x_idx = data['idx']
    z_q = data['z_q']
    bs = x.shape[1]
    if bs > B:
        return data

    data['sdf'] = x.repeat(B // bs, 1, 1, 1, 1)
    data['idx'] = x_idx.repeat(B // bs, 1, 1, 1)
    data['z_q'] = z_q.repeat(B // bs, 1, 1, 1, 1)
    return data


def get_partial_shape_by_range(sdf, input_range, thres=0.2):
    sdf = torch.clamp(sdf, min=-thres, max=thres)

    min_x, max_x = input_range['x1'], input_range['x2']
    min_y, max_y = input_range['y1'], input_range['y2']
    min_z, max_z = input_range['z1'], input_range['z2']

    bins_x = np.linspace(-1, 1, num=9)
    bins_y = np.linspace(-1, 1, num=9)
    bins_z = np.linspace(-1, 1, num=9)

    # -1: 1, 1: 9
    # find cube idx
    x_inds = np.digitize([min_x, max_x], bins_x)
    y_inds = np.digitize([min_y, max_y], bins_y)
    z_inds = np.digitize([min_z, max_z], bins_z)

    x_inds -= 1
    y_inds -= 1
    z_inds -= 1

    cube_x1, cube_x2 = x_inds
    cube_y1, cube_y2 = y_inds
    cube_z1, cube_z2 = z_inds

    x1, x2 = cube_x1 * 8, (cube_x2) * 8
    y1, y2 = cube_y1 * 8, (cube_y2) * 8
    z1, z2 = cube_z1 * 8, (cube_z2) * 8

    # clone sdf
    x = sdf.clone()
    x_missing = sdf.clone()
    gen_order = torch.arange(512).cuda()
    gen_order = gen_order.view(8, 8, 8)

    x[:, :, :x1, :, :] = 0.2
    gen_order[:cube_x1, :, :] = -1
    x[:, :, x2:, :, :] = 0.2
    gen_order[cube_x2:, :, :] = -1

    x[:, :, :, :y1, :] = 0.2
    gen_order[:, :cube_y1, :] = -1
    x[:, :, :, y2:, :] = 0.2
    gen_order[:, cube_y2:, :] = -1

    x[:, :, :, :, :z1] = 0.2
    gen_order[:, :, :cube_z1] = -1
    x[:, :, :, :, z2:] = 0.2
    gen_order[:, :, cube_z2:] = -1

    x_missing[:, :, x1:x2, y1:y2, z1:z2] = 0.2

    gen_order = gen_order[gen_order != -1]
    gen_order = gen_order.view(-1)

    return {'sdf': x, 'sdf_missing': x_missing, 'gen_order': gen_order}


def get_partial_shape_by_range_d(sdf, input_range, thres=0.2):
    sdf = torch.clamp(sdf, min=-thres, max=thres)

    min_x, max_x = input_range['x1'], input_range['x2']
    min_y, max_y = input_range['y1'], input_range['y2']
    min_z, max_z = input_range['z1'], input_range['z2']

    bins_x = np.linspace(-1, 1, num=9)
    bins_y = np.linspace(-1, 1, num=9)
    bins_z = np.linspace(-1, 1, num=9)

    # -1: 1, 1: 9
    # find cube idx
    x_inds = np.digitize([min_x, max_x], bins_x)
    y_inds = np.digitize([min_y, max_y], bins_y)
    z_inds = np.digitize([min_z, max_z], bins_z)

    x_inds -= 1
    y_inds -= 1
    z_inds -= 1

    cube_x1, cube_x2 = x_inds
    cube_y1, cube_y2 = y_inds
    cube_z1, cube_z2 = z_inds

    x1, x2 = cube_x1 * 8, (cube_x2) * 8
    y1, y2 = cube_y1 * 8, (cube_y2) * 8
    z1, z2 = cube_z1 * 8, (cube_z2) * 8

    # clone sdf
    x = sdf.clone()
    x_missing = sdf.clone()
    gen_order = torch.zeros(512).to(sdf.device)
    gen_order = gen_order.view(8, 8, 8)
    raw_mask = torch.zeros(64 ** 3).to(sdf.device)
    raw_mask = raw_mask.view(64, 64, 64)

    x[:, :, :x1, :, :] = 0.2
    raw_mask[:x1, :, :] = 1
    gen_order[:cube_x1, :, :] = 1
    x[:, :, x2:, :, :] = 0.2
    raw_mask[x2:, :, :] = 1
    gen_order[cube_x2:, :, :] = 1

    x[:, :, :, :y1, :] = 0.2
    raw_mask[:, :y1, :] = 1
    gen_order[:, :cube_y1, :] = 1
    x[:, :, :, y2:, :] = 0.2
    raw_mask[:, y2:, :] = 1
    gen_order[:, cube_y2:, :] = 1

    x[:, :, :, :, :z1] = 0.2
    raw_mask[:, :, :z1] = 1
    gen_order[:, :, :cube_z1] = 1
    x[:, :, :, :, z2:] = 0.2
    raw_mask[:, :, z2:] = 1
    gen_order[:, :, cube_z2:] = 1

    # gen_order = -gen_order * 2.0 - 1.0
    x = x.squeeze(1)
    gen_order = gen_order.unsqueeze(0)
    raw_mask = raw_mask.unsqueeze(0)

    x_missing[:, :, x1:x2, y1:y2, z1:z2] = 0.2

    # gen_order = gen_order[gen_order != -1]
    # gen_order = gen_order.view(-1)

    return {'masked_image': x, 'sdf_missing': x_missing, 'mask': gen_order, 'raw_mask': raw_mask}


def sample_missing(sdf, input_range):
    min_x, max_x = input_range['x1'], input_range['x2']
    min_y, max_y = input_range['y1'], input_range['y2']
    min_z, max_z = input_range['z1'], input_range['z2']

    bins_x = np.linspace(-1, 1, num=9)
    bins_y = np.linspace(-1, 1, num=9)
    bins_z = np.linspace(-1, 1, num=9)

    x_inds = np.digitize([min_x, max_x], bins_x)
    y_inds = np.digitize([min_y, max_y], bins_y)
    z_inds = np.digitize([min_z, max_z], bins_z)

    x_inds -= 1
    y_inds -= 1
    z_inds -= 1

    cube_x1, cube_x2 = x_inds
    cube_y1, cube_y2 = y_inds
    cube_z1, cube_z2 = z_inds

    x1, x2 = cube_x1 * 8, (cube_x2) * 8
    y1, y2 = cube_y1 * 8, (cube_y2) * 8
    z1, z2 = cube_z1 * 8, (cube_z2) * 8

    # clone sdf
    x = sdf.clone()
    x[:, :, x1:x2, y1:y2, z1:z2] = 0.2
    return x


def get_shape_comp_input_mesh(sdf_partial, sdf_missing):
    ############################################
    ## make red cuboid for the partial shapes ##
    ############################################
    # x_p = test_comp_data['sdf'].clone()
    # x_res = test_comp_data['sdf_res'].clone()
    x_p = sdf_partial
    x_res = sdf_missing

    mesh_part = sdf_to_mesh(x_p[:1])
    mesh_res = sdf_to_mesh(x_res, color=[1, .6, .6])

    # combine
    mesh_comb = structures.join_meshes_as_scene([mesh_part, mesh_res])

    return mesh_comb


def save_mesh_as_gif(mesh_renderer, mesh, rot=True, nrow=3, out_name='1.gif'):
    """ save batch of mesh into gif """

    if not rot:
        img_comb = render_mesh(mesh_renderer, mesh, norm=False)
        _, C, H, W = img_comb.shape
        img_comb = vutils.make_grid(img_comb, nrow=nrow)
        img_comb = img_comb.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        with imageio.get_writer(out_name, mode='i') as writer:
            writer.append_data(img_comb)
    # rotate
    else:
        rot_comb = rotate_mesh_360(mesh_renderer, mesh)  # save the first one

        # gather img into batches
        nimgs = len(rot_comb)
        nrots = len(rot_comb[0])
        H, W, C = rot_comb[0][0].shape
        rot_comb_img = []
        for i in range(nrots):
            img_grid_i = torch.zeros(nimgs, H, W, C)
            for j in range(nimgs):
                img_grid_i[j] = torch.from_numpy(rot_comb[j][i])

            img_grid_i = img_grid_i.permute(0, 3, 1, 2)
            img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
            img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)

            rot_comb_img.append(img_grid_i)

        with imageio.get_writer(out_name, mode='I', duration=.08) as writer:

            # combine them according to nrow
            for rot in rot_comb_img:
                writer.append_data(rot)


def save_mesh_as_pngs(mesh_renderer, mesh, out_dir='1.gif'):
    """ save batch of mesh into gif """
    rot_comb = rotate_mesh_360(mesh_renderer, mesh, dg=5)  # save the first one

    # gather img into batches
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])

        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=1)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)

        rot_comb_img.append(img_grid_i)

    for i, rot in enumerate(rot_comb_img):
        with imageio.get_writer(os.path.join(out_dir, '%01d.png' % i), mode='i') as writer:
            writer.append_data(rot)
