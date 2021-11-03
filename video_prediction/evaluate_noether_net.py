import scipy.stats as st
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import argparse
import os
import copy
import random
from torch.utils.data import DataLoader
import utils
import numpy as np

import higher
from lpips_pytorch import LPIPS

from models.forward import tailor_many_steps
from models.cn import replace_cn_layers
from utils import svg_crit

# PID
print(f'PID: {os.getpid()}')

torch.cuda.set_device(0)


# NOTE: deterministic for debugging
print(f'torch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}')
torch.backends.cudnn.deterministic = True
print(f'torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--baseline_model_path', default='', help='path to model')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--dataset', default='bair', help='dataset to train with')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--use_action', type=int, default=0, help='if true, train action-conditional model')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--tailor', action='store_true', help='if true, perform tailoring')
parser.add_argument('--num_inner_steps', type=int, default=1, help='how many tailoring steps?')
parser.add_argument('--num_train_batch', type=int, default=-1, help='if -1, do all of them')
parser.add_argument('--num_val_batch', type=int, default=-1, help='if -1, do all of them')
parser.add_argument('--train_set_length', type=int, default=256, help='size of training set')
parser.add_argument('--test_set_length', type=int, default=-1, help='size of test set')
parser.add_argument('--learn_inner_lr', action='store_true', help='optimize inner LR in outer loop?')
parser.add_argument('--n_trials', type=int, default=7, help='number of trials to average over')
parser.add_argument('--emb_dim', type=int, default=8, help='dim for Emb')
parser.add_argument('--inner_crit_mode', default='mse', help='mse or cosine')
parser.add_argument('--inner_lr', type=float, default=-1, help='learning rate for inner optimizer')
parser.add_argument('--val_inner_lr', type=float, default=-1, help='val. LR for inner opt (if -1, use orig.)')
parser.add_argument('--svg_loss_kl_weight', type=float, default=0.0001, help='weighting factor for KL loss')
parser.add_argument('--reuse_lstm_eps', action='store_true', help='correlated eps samples for prior & posterior?')
parser.add_argument('--only_tailor_on_improvement', action='store_true', help='no outer update if no inner improvement')
parser.add_argument('--stack_frames', action='store_true', help='stack every 2 frames channel-wise')
parser.add_argument('--only_twenty_degree', action='store_true', help='for Phys101 ramp, only 20 degree setting?')
parser.add_argument('--center_crop', type=int, default=1080, help='center crop param (phys101)')
parser.add_argument('--crop_upper_right', type=int, default=1080, help='upper right crop param (phys101)')
parser.add_argument('--frame_step', type=int, default=2, help='controls frame rate for Phys101')
parser.add_argument('--num_emb_frames', type=int, default=1, help='number of frames to pass to the embedding')
parser.add_argument('--horiz_flip', action='store_true', help='randomly flip phys101 sequences horizontally (p=.5)?')
parser.add_argument('--save_warmstart_dataset', action='store_true', help='save_warmstart_dataset')
parser.add_argument('--inner_opt_all_model_weights', action='store_true', help='optimize non-CN model weights in inner loop?')
parser.add_argument('--adam_inner_opt', action='store_true', help='use Adam in inner loop?')


opt = parser.parse_args()
os.makedirs('eval_metrics', exist_ok=True)

track_gen = True

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor
device = torch.device('cuda')


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

if (opt.num_train_batch == -1) or (len(train_data) // opt.batch_size < opt.num_train_batch):
    opt.num_train_batch = len(train_data) // opt.batch_size
if (opt.num_val_batch == -1) or (len(test_data) // opt.batch_size < opt.num_val_batch):
    opt.num_val_batch = len(test_data) // opt.batch_size

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)


def get_batch_generator(data_loader):
    while True:
        for sequence in data_loader:
            if not opt.use_action:
                batch = utils.normalize_data(opt, dtype, sequence)
                yield batch
            else:
                images, actions = sequence
                images = utils.normalize_data(opt, dtype, images)
                actions = utils.sequence_input(actions.transpose_(0, 1), dtype)
                yield images, actions

training_batch_generator = get_batch_generator(train_loader)
testing_batch_generator = get_batch_generator(test_loader)

print('\nDatasets loaded!')


# ---------------- plotting util fns --------------------------
def combine_dims(a, start=2, count=2):
    """ Reshapes numpy array a by combining count dimensions,
        starting at dimension index start """
    s = a.transpose((0,2,1,3,4)).shape
    return np.reshape(a.transpose((0,2,1,3,4)), s[:start] + (-1,) + s[start+count:])


def conf_int(data, alpha=0.95, dist='t'):
    if dist == 't':
        return st.t.interval(alpha, data.shape[0]-1, loc=data.mean(axis=0), scale=st.sem(data, axis=0))
    elif dist == 'norm':
        return st.norm.interval(alpha, loc=data.mean(axis=0), scale=st.sem(data, axis=0))
    elif dist == 'sem':
        return data.mean(axis=0) - st.sem(data, axis=0), data.mean(axis=0) + st.sem(data, axis=0)
    raise NotImplementedError


ckpt = torch.load(opt.model_path)
_opt = ckpt['opt']

# ---------------- set the options ----------------------------
if hasattr(_opt, 'num_emb_frames'):
    opt.num_emb_frames = _opt.num_emb_frames
opt.dataset = _opt.dataset
opt.last_frame_skip = _opt.last_frame_skip
opt.channels = _opt.channels
opt.image_width = _opt.image_width
if hasattr(_opt, 'inner_crit_compare_to'):
    opt.inner_crit_compare_to = _opt.inner_crit_compare_to

# ---------------- load the models ----------------------------
if 'svg_model' in ckpt.keys():
    if opt.inner_lr == -1:
        opt.inner_lr = _opt.inner_lr
    try:
        if opt.val_inner_lr == -1:
            opt.val_inner_lr = _opt.val_inner_lr
    except:
        pass

    opt.inner_crit_mode = _opt.inner_crit_mode
    opt.svg_loss_kl_weight = _opt.svg_loss_kl_weight
    svg_model = ckpt['svg_model']
    print('\nSVG model with pre-trained weights loaded!')
else:
    svg_model = utils.modernize_model(opt.model_path, opt)
    print('\nOld SVG model with pre-trained weights loaded and modernized!')
val_inner_lr = opt.inner_lr
if opt.val_inner_lr != -1:
    val_inner_lr = opt.val_inner_lr
replace_cn_layers(svg_model.encoder, batch_size=opt.batch_size)
replace_cn_layers(svg_model.decoder, batch_size=opt.batch_size)
svg_model.frame_predictor.batch_size = opt.batch_size
svg_model.posterior.batch_size = opt.batch_size
svg_model.prior.batch_size = opt.batch_size

svg_model.cuda()
emb = svg_model.emb
svg_model.eval()


norm_trnfm = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])

lpips = LPIPS(
    net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
).cuda()

print(opt)

opt.only_cn_decoder = False


# filename for saved metrics
num_epochs = int(opt.model_path.split('/')[-1].split('_')[-1].split('.')[0])
experiment_id = opt.model_path.split('/')[-2]
baseline_fname = f'eval_metrics/cached_metrics_id{experiment_id}-ep{int(num_epochs)}-trials{opt.n_trials}'
if val_inner_lr > 0 and opt.num_inner_steps > 0:
    baseline_fname += f'-lr{val_inner_lr}'
    baseline_fname += f'-steps{opt.num_inner_steps}'
    if opt.adam_inner_opt:
        baseline_fname += '-adam'
baseline_fname += '.npz'
print(f'fname for save:\n\t{baseline_fname}')


print('starting eval loop...')
all_tailor_ssims = []
all_tailor_psnrs = []
all_tailor_mses = []
all_tailor_lpips = []
all_val_inner_losses = []
all_val_svg_losses = []
all_gen = []
all_outer_losses = []


# perform a bunch of trials, like in the SVG paper, and keep track of avg and best
for trial_num in range(opt.n_trials):
    print(f'TRIAL {trial_num}')

    tailor_ssims = []
    tailor_psnrs = []
    tailor_mses = []
    tailor_lpips = []
    val_inner_losses = []
    val_svg_losses = []
    base_ssims = []
    base_psnrs = []
    base_mses = []

    val_outer_loss = 0.
    baseline_outer_loss = 0.

    for batch_num in tqdm(range(opt.num_val_batch)):
        batch = next(testing_batch_generator)

        # tailoring pass
        gen_seq, mus, logvars, mu_ps, logvar_ps = tailor_many_steps(
            svg_model, batch, opt=opt, track_higher_grads=False,
            mode='eval',
            # extra kwargs
            inner_crit_mode=opt.inner_crit_mode,
            reuse_lstm_eps=opt.reuse_lstm_eps,
            val_inner_lr=val_inner_lr,
            svg_losses=val_svg_losses,
            tailor_losses=val_inner_losses,
            tailor_ssims=tailor_ssims,
            tailor_psnrs=tailor_psnrs,
            tailor_mses=tailor_mses,
            only_cn_decoder=opt.only_cn_decoder,
            adam_inner_opt=opt.adam_inner_opt,
        )

        if track_gen:
            all_gen.append([f.detach().cpu() for f in gen_seq])

        # LPIPS
        with torch.no_grad():
            lpips_scores = [[lpips(norm_trnfm(b[_idx]), norm_trnfm(g[_idx])).detach().cpu().item() \
                             for b, g in zip(batch[opt.n_past:], gen_seq[opt.n_past:])] \
                            for _idx in range(batch[0].shape[0])]
            tailor_lpips.append(lpips_scores)

        with torch.no_grad():
            outer_loss = svg_crit(gen_seq, batch, mus, logvars, mu_ps, logvar_ps, opt)

        val_outer_loss += outer_loss.detach().cpu().numpy().item()

    all_val_inner_losses.append([sum(x) / (opt.num_val_batch) for x in zip(*val_inner_losses)])
    all_val_svg_losses.append([sum(x) / (opt.num_val_batch) for x in zip(*val_svg_losses)])
    all_tailor_ssims.append(copy.deepcopy(tailor_ssims))
    all_tailor_psnrs.append(copy.deepcopy(tailor_psnrs))
    all_tailor_mses.append(copy.deepcopy(tailor_mses))
    all_tailor_lpips.append(copy.deepcopy(tailor_lpips))
    all_outer_losses.append(val_outer_loss / (opt.num_val_batch))

    print(f'Model {trial_num}:')
    print(np.array(all_val_svg_losses[-1]).shape)
    print(f'\tOuter SVG loss:   {np.array(all_val_svg_losses[-1]).mean(axis=(0))}')
    print(f'\tInner VAL loss:   {np.array(all_val_inner_losses[-1]).mean(axis=(0))}')
    print(f'\tOuter VAL loss:   {all_outer_losses[-1]}')
    print(f'\tOuter SSIM:       {np.array(all_tailor_ssims[-1]).mean(axis=(0,1,-2))}\n\t\tmean: {np.array(all_tailor_ssims[-1]).mean()}')
    print(f'\tOuter PSNR:       {np.array(all_tailor_psnrs[-1]).mean(axis=(0,1,-2))}\n\t\tmean: {np.array(all_tailor_psnrs[-1]).mean()}')
    print(f'\tOuter MSE:        {np.array(all_tailor_mses[-1]).mean(axis=(0,1,-2))}\n\t\tmean: {np.array(all_tailor_mses[-1]).mean()}')


all_tailor_ssims = np.array(all_tailor_ssims)
all_tailor_psnrs = np.array(all_tailor_psnrs)
all_tailor_mses = np.array(all_tailor_mses)
all_tailor_lpips = np.array(all_tailor_lpips)[:,:,None,:,:]
all_val_inner_losses = np.array(all_val_inner_losses)
all_val_svg_losses = np.array(all_val_svg_losses)
all_outer_losses = np.array(all_outer_losses)

np.savez(
    baseline_fname,
    all_base_tailor_ssims=all_tailor_ssims,
    all_base_tailor_psnrs=all_tailor_psnrs,
    all_base_tailor_mses=all_tailor_mses,
    all_base_tailor_lpips=all_tailor_lpips,
    all_val_inner_losses=all_val_inner_losses,
    all_val_svg_losses=all_val_svg_losses,
)
print(f'saved metrics to {baseline_fname}')
