import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from dataloader import get_loader
from model import Editor
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from loss import Loss


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
        
        
class Trainier():
    def __init__(self, args) -> None:
        self.args = args
        
        if self.args.amp:
            self.scaler = GradScaler()
            
        self.trainer_initialized = False
        
        self.start_epoch = self.args.start_epoch
        self.device = 'cuda'
            
    def initial_trainer(self):
        self.trainer_initialized = False
        self.config_dataset()
        self.init_model()
        self.config_optimizer()
        self.config_losses_and_metrics()
        self.config_wirter()
        
        self.trainer_initialized = True
        
    def config_dataset(self):
        self.train_loader = get_loader(
            datalist_json=self.args.train_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            args=None,
            phase='train'
        )
    
        self.val_loader = get_loader(
            datalist_json=self.args.val_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            args=None,
            phase='val'
        )
        
        self.test_loader = get_loader(
            datalist_json=self.args.test_json,
            batch_size=self.args.batch_size,
            num_works=self.args.workers,
            args=None,
            phase='test'
        )
        print('Train data number: {} | Val data number: {} | Test data number: {}'.format(
            len(self.train_loader) * self.args.batch_size,
            len(self.val_loader) * self.args.batch_size,
            len(self.test_loader) * self.args.batch_size))
        
    def config_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.optim_lr, weight_decay=self.args.decay)

        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.args.max_epochs)
        print("Config optimier")
        
    def config_losses_and_metrics(self):
        self.loss = Loss()
        print("Config losses and metrics")
        
    def config_wirter(self):
        self.writer = None
        if self.args.logdir is not None:
            self.writer = SummaryWriter(log_dir=f'{self.args.logdir}/tensorboard')
            print("Writing Tensorboard logs to ", f'{self.args.logdir}/tensorboard')
        print("Save model to ", self.args.logdir)

    def init_model(self):
        self.model = Editor(
            in_c=self.args.in_c,
            filters=self.args.filters,
            style_dim=self.args.style_dim,
            out_c=self.args.out_c  
        ).to('cuda')
        
        pytorch_model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Model parameters count", pytorch_model_params)

    def save_checkpoint(self, file_name, epoch, best_acc):
        state_dict = self.model.state_dict() 
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if self.optimizer is not None:
            save_dict["optimizer"] = self.optimizer.state_dict()
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
            self.optimizer.load_state_dict(new_state_dict)
            for state in self.optimizer.state.values():
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
    
    def save_encoder(self, file_name):
        file_name = os.path.join(self.args.logdir, file_name)
        torch.save(self.model, file_name)
        
    def load_encoder(self):
        encoder = torch.load(self.args.encoder, map_location=torch.device('cpu'))
        return encoder
    
    def train_one_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        run_loss = AverageMeter()
        for idx, batch_data in enumerate(self.train_loader):

            img, label, gli_0, gli_1 = batch_data["mri_0"].cuda(), batch_data['mri_1'].cuda(), batch_data['gli_0'].cuda(), batch_data['gli_1'].cuda()

            for param in self.model.parameters():
                param.grad = None
                
            y, mask = self.model(img, gli_0, gli_1)
            y = y * mask
            label = label * mask
            loss, _ = self.loss(y, label)    

            loss.backward()
            self.optimizer.step()

            run_loss.update(loss.item(), n=self.args.batch_size)

        for param in self.model.parameters():
            param.grad = None
        return run_loss.avg
    
    def validata(self):
        self.model.eval()
        run_loss = AverageMeter()

        with torch.no_grad():
            for idx, batch_data in enumerate(self.val_loader):
                img, label, gli_0, gli_1 = batch_data["mri_0"].cuda(), batch_data['mri_1'].cuda(), batch_data['gli_0'].cuda(), batch_data['gli_1'].cuda()

                y, mask = self.model(img, gli_0, gli_1)

                y = y * mask
                label = label * mask
                loss, _ = self.loss(y, label)
                    
                run_loss.update(loss.item(), n=self.args.batch_size)

        return run_loss.avg
    
    def train(self):
        if self.trainer_initialized is False:
            self.initial_trainer()
            
            val_loss_min = 10.0
    
        for epoch in range(self.args.start_epoch, self.args.max_epochs):

            print(time.ctime(), "Epoch:", epoch)
            epoch_time = time.time()
            train_loss = self.train_one_epoch()

            print(
                "Train Epoch  {}/{}".format(epoch, self.args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )

            self.writer.add_scalar("train_loss", train_loss, epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:

                epoch_time = time.time()
                val_acc = self.validata()

                print(
                    "Final validation stats {}/{}".format(epoch, self.args.max_epochs - 1),
                    ", loss:",
                    val_acc,
                    ", time {:.2f}s".format(time.time() - epoch_time),
                )

                if self.writer is not None:
                    self.writer.add_scalar("Mean_Val_Dice", np.mean(val_acc), epoch)

                val_avg_acc = np.mean(val_acc)

                if val_avg_acc < val_loss_min:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_loss_min, val_avg_acc))
                    val_loss_min = val_avg_acc

                    self.save_checkpoint(file_name="ckpt_best.pt", epoch=epoch, best_acc=val_loss_min)
                    self.save_encoder(file_name="encoder_best.pt")

                self.save_checkpoint(file_name="ckpt_final.pt", epoch=epoch, best_acc=val_loss_min)

        print("Training Finished !, Best Accuracy: ", val_loss_min)
        return val_loss_min
    



