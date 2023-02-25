"""Argument parser"""

import argparse


def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='/apdcephfs/share_1313228/home/haiwendiao',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='/apdcephfs/share_1313228/home/haiwendiao/SGRAF-master/vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='/apdcephfs/share_1313228/home/haiwendiao/SGRAF-master/runs/f30k_SGR/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='/apdcephfs/share_1313228/home/haiwendiao/SGRAF-master/runs/f30k_SGR/log',
                        help='Path to save Tensorboard log.')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=40, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=30, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=5, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=50, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin or parameter for GCE.')
    parser.add_argument('--noise_rate', default=0.2, type=float,
                        help='Noise rate.')
    parser.add_argument('--tau', default=0.05, type=float,
                        help='Temperature')
    parser.add_argument('--ratio', default=-1, type=float,
                        help='ratio')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--warm_step', default=0, type=int,
                        help='Cross-Entropy warm step number.')

    # ------------------------- model setting -----------------------#
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SGR', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')
    parser.add_argument('--resume', default='',
                        help='Train from checkpoint.')
    parser.add_argument('--func', default='CNL', help='{NCL, CL, NL, TR, TR-HN}')
    parser.add_argument('--loss', default='log', help='{log, exp, gce, max-margin}')
    opt = parser.parse_args()

    if opt.loss != 'max-margin' and opt.ratio >= 0:
        opt.margin = opt.ratio

    if opt.module_name == 'SGRAF':
        opt.model_path = [opt.model_name + '/' + ('%s_%s_model_best_%g_%g_%s_%g.pth.tar' % (opt.data_name, module_name, opt.noise_rate, opt.tau, opt.loss, opt.margin)) for module_name in ['SAF', 'SGR']]
        opt.module_names = ['SAF', 'SGR', 'SGRAF']
    else:
        opt.model_path = [opt.model_name + '/' + ('%s_%s_model_best_%g_%g_%s_%g.pth.tar' % (opt.data_name, opt.module_name, opt.noise_rate, opt.tau, opt.loss, opt.margin))]
        opt.module_names = [opt.module_name]
        opt.best_model_filename = ('%s_%s_model_best_%g_%g_%s_%g.pth.tar' % (opt.data_name, opt.module_name, opt.noise_rate, opt.tau, opt.loss, opt.margin))

    print(opt)
    return opt

# python train.py --data_name coco_precomp --num_epochs 20 --lr_update 10 --module_name SGR --log_step 200 --data_path ../SCAN/data/data --vocab_path ../SCAN/data/vocab --model_name runs_new/coco/checkpoint --logger_name runs_new/coco/log  --noise_rate 0.8 --margin 0.6 --resume
