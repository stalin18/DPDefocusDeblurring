import argparse

import datetime
import os
import torch.multiprocessing as mp

from trainer import Trainer

parser = argparse.ArgumentParser('Dual-Pixel based Defocus Deblurring')

parser.add_argument('--training_system', default='mlp',
                      help= 'Training machine or system to handle checkpoint and log paths properly')
parser.add_argument('--ngpu', default=2, type=int, help='Number of GPUs for training')

parser.add_argument('--exp_id', default='DPDD', help='Experiment identifier used for naming each experiment folder')
parser.add_argument('--exp_name', help='Experiment identifier for uniquely naming each experiment folder')
parser .add_argument('--model', default='dpdnet', help= 'Model to use for training, default is DPDNet')

parser.add_argument('--data_dir',
                    default='/kunal.swami/Workspaces/Datasets/DefocusDeblurning/dataset/DPDNet',
                    help= 'Dataset root directory')
parser.add_argument('--checkpointdir',
                    default='/kunal.swami/Workspaces/Datasets/DefocusDebLurcing/checkpoints',
                    help='Folder for saving model checkpoints, logs, results, etc.')
parser .add_argument ('--checkpoint',
                      default=None,
                      heLp= 'Finetune from given checkpoint')

parser. add_argument('--batch_size', default=16, type=int, help='Batch size')
parser.add_argument('--num_workers', default=12, type=int, help='Number of threads for dataloader')

parser.add_argument('--num_epoch', type=int, default=200, help='Total epochs for train')
parser .add_argument('--lr', default=2e-5, type=float, help='Learning rate')
parser .add_argument('--Lr_decay_step', default=60, type=int, help='Learning rate decay step')
parser.add_argument('--lr_decav_factor', default=0.5, type=float, help= 'Learning rate decay factor')

parser.add_argument('--vis_after', default=40, type=int, help='Visualize training images, learnina rate, etc. after these many steps')
parser.add_argument('--save_after', default=1, type=int, help='Save model checkpoint after these many epochs')
parser.add_argument('--validate_after', default=1, type=int, help='Perform validation after these many epochs')
parser.add_argument('--check_validation', default=False, type=bool, help='Perform validation code check')

parser.add_argument('--dropout_rate', default=0.4, type=float, heLp='Dropout rate for some models')
parser.add_argument('--Loss_type', default=None, type=str, help= 'Which type of Loss to use, default (None) is MSE, '
                                                                  'other options are Charbonnier, Mix (Charbonnier + MSSSIM)')
parser.add_argument('--loss_wt1', default=50, type=float, help='Weight for 1st loss')
parser. add_argument('--loss_wt2', default=200, type=float, help='Weight for 2nd loss')


def create_data_directories(args):
    if not os.path.exists(args.checkpointdir):
        os.makedirs(args.checkpointdir)

    if not os.path.exists(os.path.join(args.checkpointdir, args.exp_name)):
        os.makedirs(args.checkpointdir, args.exp_name)

    if not os.path.exists(os.path.join(args.checkpointdir, args.exp_name, 'logs')):
        os.makedirs(os.path.join(args.checkpointdir, args.exp_name, 'logs'))

    if not os.path.exists(os.path.join(args.checkpointdir, args.exp_name, 'results')):
        os.makedirs(os.path.join(args.checkpointdir, args.exp_name, 'results'))


def train(rank, world_size, args):
    trainer = Trainer(args, rank, world_size)
    trainer.run_training()


def main():
    args = parser.parse_args()

    # create experiment name
    exp_name = str(datetime.date.today()) + '_' + args.model + '_' + args.exp_id
    args.exp_name = exp_name

    # print all args for summary
    print(args)

    # create all relevant directories
    create_data_directories()

    """ MultiGPU training """
    # num GPUs for training
    world_size = args.ngpu
    # below IP and port can remain same becuase using a different VM for every training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, args=(world_size, args), nprocs=world_size)


if __name__ == '__main__':
    main()


