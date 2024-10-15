import argparse

from trainer import Trainier
from eva import Evaluator
import os


parser = argparse.ArgumentParser()

# Train
parser.add_argument("--optim_lr", default=1e-3, type=float, help="optimization learning rate")
parser.add_argument("--decay", default=0.1, type=float)
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=0, type=int, help="number of warmup epochs")
parser.add_argument("--start_epoch", default=0, type=int)
parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")

# Data
parser.add_argument("--train_json", default='/Brats2023/ag_2d/ag_t1_edit_train.json', type=str, help="dataset json file")
parser.add_argument("--val_json", default='/Brats2023/ag_2d/ag_t1_edit_val.json', type=str, help="dataset json file")
parser.add_argument("--test_json", default='/Brats2023/ag_2d/ag_t1_edit_test.json', type=str, help="dataset json file")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--batch_size", default=32, type=int, help="number of batch size")

# Model
parser.add_argument("--in_c", default=1, type=int)
parser.add_argument("--filters", default=16, type=list)
parser.add_argument("--style_dim", default=512, type=list)
parser.add_argument("--out_c", default=1, type=list)

# Checkpoint
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint, including model, "
                                                       "optimizer, scheduler")

parser.add_argument("--logdir", default="/train_log", type=str, help="directory to save the tensorboard logs")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    trainer = Trainier(args)
    train_acc = trainer.train()
    
    return train_acc


if __name__ == "__main__":
    main()