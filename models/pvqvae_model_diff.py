import os
from collections import OrderedDict

import omegaconf
from termcolor import colored
from einops import rearrange
from tqdm import tqdm

import torch
from torch import nn, optim

from models.voxdiff.util import instantiate_from_config

from models.base_model import BaseModel
from models.networks.pvqvae_networks.auto_encoder import  PVQVAE_diff

import utils.util
from utils.util_3d import init_mesh_renderer, render_sdf


class PVQVAEModel(BaseModel):
    def name(self):
        return 'PVQVAE-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()

        # -------------------------------
        # Define Networks
        # -------------------------------

        # model
        assert opt.vq_cfg is not None
        configs = omegaconf.OmegaConf.load(opt.vq_cfg)
        mparam = configs.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        lossconfig = mparam.lossconfig

        n_down = len(ddconfig.ch_mult) - 1

        self.loss = instantiate_from_config(lossconfig).to(opt.device)

        self.vqvae = PVQVAE_diff(ddconfig, n_embed, embed_dim)
        self.vqvae.to(opt.device)

        if opt.dist_train:
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print('use {} gpus!'.format(num_gpus))
                self.vqvae = nn.parallel.DistributedDataParallel(self.vqvae, device_ids=[opt.local_rank],
                                                                 output_device=opt.local_rank,
                                                                 find_unused_parameters=True)
                self.loss = nn.parallel.DistributedDataParallel(self.loss, device_ids=[opt.local_rank],
                                                                output_device=opt.local_rank,
                                                                find_unused_parameters=True)
            self.gpu_ids = [opt.local_rank]

        if self.isTrain:
            # ----------------------------------
            # define loss functions
            # ----------------------------------
            # lossconfig = configs.lossconfig
            # lossparams = lossconfig.params
            # self.loss_vq = VQLoss(**lossparams).to(opt.device)

            # ---------------------------------
            # initialize optimizers
            # ---------------------------------
            self.optimizer = optim.Adam(self.vqvae.parameters(), lr=opt.lr, betas=(0.5, 0.9))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 10 if opt.dataset_mode == 'imagenet' else 30,
                                                       0.5, )
            # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        resolution = configs.model.params.ddconfig['resolution']
        self.resolution = resolution

        # setup hyper-params 
        nC = resolution
        self.cube_size = 2 ** n_down  # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        assert nC == 64, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

        # setup renderer
        dist, elev, azim = 1.7, 20, 20
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # for saving best ckpt
        self.best_iou = -1e12

    @staticmethod
    # def unfold_to_cubes(self, x, cube_size=8, stride=8):
    def unfold_to_cubes(x, cube_size=8, stride=8):
        """ 
            assume x.shape: b, c, d, h, w 
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

    @staticmethod
    # def fold_to_voxels(self, x_cubes, batch_size, ncubes_per_dim):
    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size)
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                      p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def set_input(self, input):
        '''Samples at training time'''
        # import pdb; pdb.set_trace()
        x = input['sdf']
        self.x = x
        self.cur_bs = x.shape[0]  # to handle last batch

        self.x_cubes = self.unfold_to_cubes(x, self.cube_size, self.stride)
        vars_list = ['x', 'x_cubes']

        self.tocuda(var_names=vars_list)

    def forward(self):
        # qloss: codebook loss
        self.dec, self.posterior = self.vqvae(self.x_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)
        # if self.opt.dist_train:
        #     self.zq_cubes, self.qloss, _ = self.vqvae.module.encode(self.x_cubes)  # zq_cubes: ncubes X zdim X 1 X 1 X 1
        #     self.zq_voxels = self.fold_to_voxels(self.zq_cubes, batch_size=self.cur_bs,
        #                                          ncubes_per_dim=self.ncubes_per_dim)  # zq_voxels: bs X zdim X ncubes_per_dim X ncubes_per_dim X ncubes_per_dim
        #     self.x_recon = self.vqvae.module.decode(self.zq_voxels)
        # else:
        #     self.dec, self.posterior = self.vqvae(self.x)
        # self.zq_cubes, self.qloss, _ = self.vqvae.encode(self.x) # zq_cubes: ncubes X zdim X 1 X 1 X 1
        # self.zq_voxels = self.fold_to_voxels(self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim) # zq_voxels: bs X zdim X ncubes_per_dim X ncubes_per_dim X ncubes_per_dim
        # self.x_recon = self.vqvae.decode(self.zq_voxels)

    def inference(self, data, should_render=False, verbose=False):
        self.vqvae.eval()
        self.set_input(data)

        # make sure it has the same name as forward
        with torch.no_grad():
            self.x_recon, _ = self.vqvae(self.x_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)
            # if self.opt.dist_train:
            #     self.zq_cubes, _, self.info = self.vqvae.module.encode(self.x_cubes)
            #     self.zq_voxels = self.fold_to_voxels(self.zq_cubes, batch_size=self.cur_bs,
            #                                          ncubes_per_dim=self.ncubes_per_dim)
            #     self.x_recon = self.vqvae.module.decode(self.zq_voxels)
            #     # _, _, quant_ix = info
            #     #
            # else:
            #     # self.zq_cubes, _, self.info = self.vqvae.encode(self.x_cubes)
            #     # self.zq_voxels = self.fold_to_voxels(self.zq_cubes, batch_size=self.cur_bs,
            #     #                                      ncubes_per_dim=self.ncubes_per_dim)
            #     # self.x_recon = self.vqvae.decode(self.zq_voxels)
            #     self.x_recon, _ = self.vqvae(self.x)

            if should_render:
                self.image = render_sdf(self.renderer, self.x)
                self.image_recon = render_sdf(self.renderer, self.x_recon)

        self.vqvae.train()

    def set_input_generate(self, input=None):

        self.x = input['sdf']
        self.x_idx = input['idx']
        self.z_q = input['z_q']
        self.z_shape = self.z_q.shape

        if self.opt.dataset_mode in ['pix3d_img', 'snet_img']:
            self.gt_vox = input['gt_vox']

        self.x_idx_seq = rearrange(self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous()  # to (T, B)
        self.x_idx = self.x_idx_seq.clone()

        # prepare input for transformer
        # T, B = self.x_idx.shape[:2]

        # if gen_order is None:
        #     self.gen_order = self.get_gen_order(T, self.opt.device)
        #     self.context_len = -1  # will be specified in inference
        # else:
        #     if len(gen_order) != T:
        #
        #         self.context_len = len(gen_order)
        #         # pad the remaining
        #         remain = torch.tensor([i for i in range(T) if i not in gen_order]).to(gen_order)
        #         remain = remain[torch.randperm(len(remain))]
        #         self.gen_order = torch.cat([gen_order, remain])
        #     else:
        #         self.gen_order = gen_order

        # x_idx_seq_shuf = self.x_idx_seq
        # x_seq_shuffled = torch.cat([torch.LongTensor(1, bs).fill_(self.sos), x_idx_seq_shuf], dim=0)  # T+1
        # pos_shuffled = torch.cat([self.grid_table[:1], self.grid_table[1:][self.gen_order]],
        #                          dim=0)  # T+1, <sos> should always at start.

        # self.inp = x_seq_shuffled[:-1].clone()
        # self.tgt = x_seq_shuffled[1:].clone()
        # self.inp_pos = pos_shuffled[:-1].clone()
        # self.tgt_pos = pos_shuffled[1:].clone()

        # self.counter += 1

        vars_list = ['x_idx', 'x_idx_seq', 'z_q', 'x']

        self.tocuda(var_names=vars_list)

    def decode_idx(self, data, codex):
        self.set_input_generate(data)
        with torch.no_grad():
            x_recon_tf = self.vqvae.decode_enc_idices(codex, z_spatial_dim=8)
        return x_recon_tf

    def test_iou(self, data, thres=0.0):
        """
            thres: threshold to consider a voxel to be free space or occupied space.
        """
        # self.set_input(data)

        self.vqvae.eval()
        self.inference(data, should_render=False)
        self.vqvae.train()

        x = self.x
        x_recon = self.x_recon

        iou = utils.util.iou(x, x_recon, thres)

        return iou

    def eval_metrics(self, dataloader, thres=0.0):
        self.eval()

        iou_list = []
        with torch.no_grad():
            for ix, test_data in tqdm(enumerate(dataloader), total=len(dataloader)):
                iou = self.test_iou(test_data, thres=thres)
                iou_list.append(iou.detach())

        iou = torch.cat(iou_list)
        iou_mean, iou_std = iou.mean(), iou.std()

        ret = OrderedDict([
            ('iou', iou_mean.data),
            ('iou_std', iou_std.data),
        ])

        # check whether to save best epoch
        if ret['iou'] > self.best_iou:
            self.best_iou = ret['iou']
            save_name = f'epoch-best'
            self.save(save_name)

        self.train()
        return ret

    def backward(self, optimizer_idx, total_steps):
        '''backward pass for the generator in training the unsupervised model'''
        # aeloss, log_dict_ae = self.loss_vq(self.qloss, self.x, self.x_recon)
        # if optimizer_idx == 0:
        if self.opt.dist_train:
            aeloss, log_dict_ae = self.loss(self.x, self.dec, self.posterior, optimizer_idx, total_steps,
                                            last_layer=self.vqvae.module.get_last_layer(), split="train")
        else:
            aeloss, log_dict_ae = self.loss(self.x, self.dec, self.posterior, optimizer_idx, total_steps,
                                            last_layer=self.vqvae.get_last_layer(), split="train")
        self.loss_ae = aeloss
        # self.loss_codebook = log_dict_ae['loss_codebook']
        self.loss_nll = log_dict_ae['train/nll_loss']
        self.loss_rec = log_dict_ae['train/rec_loss']
        self.loss_kl = log_dict_ae['train/kl_loss']
        self.loss_ae.backward()

        # if optimizer_idx == 1:
        #     if self.opt.dist_train:
        #         discloss, log_dict_disc = self.loss(self.x, self.dec, self.posterior, optimizer_idx, total_steps,
        #                                             last_layer=self.vqvae.module.get_last_layer(), split="train")
        #     else:
        #         discloss, log_dict_disc = self.loss(self.x, self.dec, self.posterior, optimizer_idx, total_steps,
        #                                             last_layer=self.vqvae.get_last_layer(), split="train")
        #     self.loss_disc = discloss
        # self.loss_codebook = log_dict_ae['loss_codebook']
        # self.loss_real = log_dict_disc['train/logits_real']
        # self.loss_fake = log_dict_disc['train/logits_fake']
        # self.loss_disc.backward()

    def optimize_parameters(self, total_steps):

        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward(optimizer_idx=0, total_steps=total_steps)
        self.optimizer.step()

        # self.forward()
        # self.optimizer.zero_grad(set_to_none=True)
        # self.backward(optimizer_idx=1, total_steps=total_steps)
        # self.optimizer.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret

    def get_current_errors(self):

        ret = OrderedDict([
            ('nll', self.loss_nll.data),
            ('rec', self.loss_rec.data),
            ('kl', self.loss_kl.data),
            # ('real', self.loss_real.data),
            # ('fake', self.loss_fake.data)
        ])

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.image = render_sdf(self.renderer, self.x)
            self.image_recon = render_sdf(self.renderer, self.x_recon)

        vis_tensor_names = [
            'image',
            'image_recon',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        # vis_tensor_names = ['%s/%s' % (phase, n) for n in vis_tensor_names]
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label):
        save_filename = 'vqvae_%s.pth' % (label)
        save_path = os.path.join(self.save_dir, save_filename)
        if self.opt.dist_train:
            save_obj = {
                'model': self.vqvae.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        else:
            save_obj = {
                'model': self.vqvae.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        torch.save(save_obj, save_path)
        # torch.save(self.vqvae.module.state_dict(), save_path)
        # if len(self.gpu_ids) and torch.cuda.is_available():
        #     torch.save(self.vqvae.module.cpu().state_dict(), save_path)
        #     self.vqvae.cuda(self.gpu_ids[0])
        # else:
        #     torch.save(self.vqvae.cpu().state_dict(), save_path)

    def get_codebook_weight(self):
        ret = self.vqvae.quantize.embedding.cpu().state_dict()
        self.vqvae.quantize.embedding.cuda()
        return ret

    def load_ckpt(self, ckpt):
        if type(ckpt) == str:
            state_dict = torch.load(ckpt)['model']
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict)
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))
