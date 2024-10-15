import os
import matplotlib
import matplotlib.pyplot as plt
from typing import Sequence
matplotlib.use('Agg')

import torch
from torch import nn, autograd  
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np

from torch.autograd import grad as torch_grad

from model import Generator, Discriminator, Editor
from dataloader import get_loader

import time
import clip
from monai.losses import DiceCELoss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
        

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class Trainier:
    def __init__(self, 
                args=None):
        
        self.args = args
        self.global_step = 0
        
        self.trainer_initialized = False
        self.device = 'cuda'

    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model()
        self.init_optimizer()
        self.init_loss()
        self.config_wirter()
        
        self.trainer_initialized = True
        
    def config_dataset(self):
        self.train_dataloader = get_loader(self.args.train_json,
                                self.args.batch_size,
                                int(self.args.workers),
                                phase='train')
        
        self.val_dataloader = get_loader(self.args.val_json,
                                        self.args.batch_size,
                                        int(self.args.workers),
                                        phase='val')
    
    def init_model(self):
        self.G = Editor(     
                in_c=self.args.in_c,
                filters=self.args.filters,
                style_dim=self.args.style_dim,
                out_c=self.args.out_c       
            ).to(self.device).train()
        
        self.D = Discriminator().to(self.device).train()
        self.clip, _ = clip.load("ViT-B/16")
        self.sigmoid = torch.nn.Sigmoid()

        pytorch_encoder_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)
        print("Transformer parameters count", pytorch_encoder_params)
        
        pytorch_encoder_params = sum(p.numel() for p in self.D.parameters() if p.requires_grad)
        print("Transformer parameters count", pytorch_encoder_params)
        
    def init_optimizer(self):
        named_param = list(self.G.encoder.named_parameters())
        encoder_params = [p for n, p in named_param if p.requires_grad]

        named_param = list(self.G.decoder.named_parameters())
        decoder_params = [p for n, p in named_param if p.requires_grad]
        
        self.G_optimizer = torch.optim.Adam(
            [
                {"params": encoder_params},
                {"params": decoder_params}
            ],
            lr=self.args.optim_lr, betas=(0.5, 0.9)
        )

        self.apply_gradient_penalty = True
        
        
        self.D_optimizer = torch.optim.Adam(list(self.D.parameters()),
                                            lr=self.args.optim_lr,
                                            betas=(0.5, 0.9))
        self.scheduler = None
        
    def init_loss(self):
        self.dice_loss = DiceCELoss(include_background=True, batch=True, sigmoid=True)

    def config_wirter(self):
        self.writer = None
        if self.args.logdir is not None:
            self.writer = SummaryWriter(log_dir=f'{self.args.logdir}/tensorboard')
            print("Writing Tensorboard logs to ", f'{self.args.logdir}/tensorboard')
        print("Save model to ", self.args.logdir)
        
    def train_one_epoch(self):
        self.G.train()
        self.clip.eval()
        self.D.train()
        
        run_g_loss = AverageMeter()
        run_dice_loss = AverageMeter()
        run_d_loss = AverageMeter()
        run_l1_loss = AverageMeter()
        
        for idx, batch_data in enumerate(self.train_dataloader):

            gli_0, gli_1, token = batch_data['gli_0'].to('cuda'), batch_data['gli_1'].to('cuda'), batch_data['token'].to('cuda')

            w = self.clip.encode_text(token.to(torch.int32).squeeze(1)).to(torch.float32)

            self.G_optimizer.zero_grad()

            y_hat = self.G(gli_0, w)
            y_hat_ = self.sigmoid(y_hat)
            y_hat_ = torch.gt(y_hat, 0.5).to(torch.float32)

            # adversarial loss
            if self.args.adv_lambda > 0: 
                d_loss_dict = self.train_discriminator(real_img=gli_1, fake_img=y_hat_)

            loss, loss_dict = self.calc_loss(y=gli_1, y_hat=y_hat)
            
            loss.backward()
            self.G_optimizer.step()

            run_g_loss.update(loss_dict['loss_g'], n=self.args.batch_size)
            run_dice_loss.update(loss_dict['loss_dice'], n=self.args.batch_size)
            run_d_loss.update(d_loss_dict['loss_d'], n=self.args.batch_size)

        return run_g_loss.avg, run_dice_loss.avg, run_d_loss.avg

    def validate(self):
        self.G.eval()
        
        g_loss = AverageMeter()
        dice_loss = AverageMeter()
        d_loss = AverageMeter()
        
        for idx, batch_data in enumerate(self.val_dataloader):
            gli_0, gli_1, token = batch_data['gli_0'].to('cuda'), batch_data['gli_1'].to('cuda'), batch_data['token'].to('cuda')
            w = self.clip.encode_text(token.to(torch.int32).squeeze(1)).to(torch.float32)
            
            with torch.no_grad():
                y_hat = self.G(gli_0, w)
                y_hat_ = self.sigmoid(y_hat)
                y_hat_ = torch.gt(y_hat, 0.5).to(torch.float32)
                
                loss, loss_dict = self.calc_loss(y=gli_1, y_hat=y_hat)
                
            d_loss_dict = self.validate_discriminator(real_img=gli_1, fake_img=y_hat_)

            g_loss.update(loss_dict['loss_g'], n=self.args.batch_size)
            dice_loss.update(loss_dict['loss_dice'], n=self.args.batch_size)
            d_loss.update(d_loss_dict['loss_d'], n=self.args.batch_size)

        self.G.train()
        return g_loss.avg, dice_loss.avg, d_loss.avg
        
    def train(self):
        if self.trainer_initialized is False:
            self.initial_trainer()
            
        G_loss_min = 10.0
    
        for epoch in range(self.args.start_epoch, self.args.max_epochs):

            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            run_g_loss, run_dice_loss, run_d_loss = self.train_one_epoch()

            print(
                "Train Epoch  {}/{}".format(epoch, self.args.max_epochs - 1),
                "g_loss: {:.4f}".format(run_g_loss),
                "dice_loss: {:.4f}".format(run_dice_loss),
                "d_loss: {:.4f}".format(run_d_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            self.writer.add_scalar("g_loss", run_g_loss, epoch)
            self.writer.add_scalar("dice_loss", run_dice_loss, epoch)
            self.writer.add_scalar("d_loss", run_d_loss, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:

                epoch_time = time.time()
                val_g_loss, val_dice_loss, val_d_loss = self.validate()

                print(
                    "Final validation stats {}/{}".format(epoch, self.args.max_epochs - 1),
                    ", val_g_loss | val_dice_loss | val_d_loss:",
                    val_g_loss, val_dice_loss, val_d_loss,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if self.writer is not None:
                    self.writer.add_scalar("Mean_Val_G_Loss", np.mean(val_g_loss), epoch)

                val_dice_loss = np.mean(val_dice_loss)

                if val_dice_loss < G_loss_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(G_loss_min, val_dice_loss))
                    G_loss_min = val_dice_loss

                    self.save_checkpoint(file_name="ckpt_best.pt", epoch=epoch, best_acc=G_loss_min)
                    self.save_G(file_name="gli_gen.pt")

                self.save_checkpoint(file_name="ckpt_final.pt", epoch=epoch, best_acc=G_loss_min)

        print("Training Finished !, Best Accuracy: ", G_loss_min)
        return G_loss_min

    def save_checkpoint(self, file_name, epoch, best_acc):
        state_dict = self.G.state_dict() 
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if self.G_optimizer is not None:
            save_dict["optimizer"] = self.G_optimizer.state_dict()
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()

        file_name = os.path.join(self.args.logdir, file_name)
        torch.save(save_dict, file_name)
        print("Saving checkpoint", file_name)
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        self.model.load_state_dict(new_state_dict, strict=False)
        print("=> loaded model checkpoint")

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "optimizer" in checkpoint.keys():
            for k, v in checkpoint["optimizer"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            self.G_optimizer.load_state_dict(new_state_dict)
            for state in self.G_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()      
            print("=> loaded optimizer checkpoint")
        if "scheduler" in checkpoint.keys():
            for k, v in checkpoint["scheduler"].items():
                new_state_dict[k.replace("backbone.", "")] = v
            self.scheduler.load_state_dict(new_state_dict)
            self.scheduler.step(epoch=start_epoch)
            print("=> loaded scheduler checkpoint")
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(self.args.checkpoint, start_epoch, best_acc))
    
    def save_G(self, file_name):
        file_name = os.path.join(self.args.logdir, file_name)
        torch.save(self.G, file_name)
        
    def load_G(self):
        encoder = torch.load(self.args.G, map_location=torch.device('cpu'))
        return encoder

    def calc_loss(self, y, y_hat):
        loss_dict = {}
        loss = 0.0
        
        if self.args.dice_lambda > 0:
            loss_dice = self.dice_loss(y_hat, y)
            loss_dict['loss_dice'] = float(loss_dice)
            loss += loss_dice * self.args.dice_lambda
        if self.args.adv_lambda > 0:  
            loss_g = F.softplus(-self.D(y_hat)).mean()
            loss_dict['loss_g'] = float(loss_g)
            loss += loss_g * self.args.adv_lambda

        loss_dict['loss'] = float(loss)
        return loss, loss_dict
    
    ##### modified
    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['loss_d_real'] = float(real_loss)
        loss_dict['loss_d_fake'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, real_img, fake_img):
        loss_dict = {}
        self.requires_grad(self.D, True)

        real_pred = self.D(real_img)
        fake_pred = self.D(fake_img.detach())
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['loss_d'] = float(loss)
        loss = loss * self.args.adv_lambda 
        
        # if self.apply_gradient_penalty:
        #     gp = gradient_penalty(real_img.requires_grad_(), real_pred)
        #     self.last_gp_loss = gp.clone().detach().item()
        #     self.track(self.last_gp_loss, 'GP')
        #     disc_loss = disc_loss + gp

        self.D_optimizer.zero_grad()
        loss.backward()
        self.D_optimizer.step()
    
        # Reset to previous state
        self.requires_grad(self.D, False)

        return loss_dict
    
    def validate_discriminator(self, real_img, fake_img):
        with torch.no_grad():
            loss_dict = {}
            real_pred = self.D(real_img)
            fake_pred = self.D(fake_img.detach())
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['loss_d'] = float(loss)
            loss = loss * self.args.adv_lambda 
            return loss_dict