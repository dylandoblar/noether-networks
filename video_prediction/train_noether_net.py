from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import random
from torch.utils.data import DataLoader
import utils
from utils import svg_crit
import itertools
import numpy as np
import copy
import higher

from models.forward import predict_many_steps, tailor_many_steps
from models.cn import replace_cn_layers
from models.svg import SVGModel
from models.basic_model import BasicModel
from models.embedding import ConservedEmbedding, EncoderEmbedding, ConvConservedEmbedding
import models.lstm as lstm_models


# NOTE: deterministic for debugging
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--plot_train', type=int, default=0, help='if true, also predict training data')
parser.add_argument('--use_action', type=int, default=0, help='if true, train action-conditional model')
parser.add_argument('--dataset', default='bair', help='dataset to train with')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--n_epochs', type=int, default=100, help='how many epochs to train for')
parser.add_argument('--ckpt_every', type=int, default=5, help='how many epochs per checkpoint save')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--num_epochs_per_val', type=int, default=1, help='perform validation every _ epochs')
parser.add_argument('--tailor', action='store_true', help='if true, perform tailoring')
parser.add_argument('--num_inner_steps', type=int, default=1, help='how many tailoring steps?')
parser.add_argument('--num_jump_steps', type=int, default=0, help='how many tailoring steps?')
parser.add_argument('--num_train_batch', type=int, default=-1, help='if -1, do all of them')
parser.add_argument('--num_val_batch', type=int, default=-1, help='if -1, do all of them')
parser.add_argument('--inner_lr', type=float, default=0.0001, help='learning rate for inner optimizer')
parser.add_argument('--val_inner_lr', type=float, default=-1, help='val. LR for inner opt (if -1, use orig.)')
parser.add_argument('--outer_lr', type=float, default=0.0001, help='learning rate for outer optimizer')
parser.add_argument('--svg_loss_kl_weight', type=float, default=0.0001, help='weighting factor for KL loss')
parser.add_argument('--emb_dim', type=int, default=4, help='dimensionality of convserved embedding')
parser.add_argument('--last_frame_skip', action='store_true', help='skip connection config')
parser.add_argument('--num_trials', type=int, default=5, help='how many times to run training procedure')
parser.add_argument('--inner_crit_mode', default='mse', help='"mse" or "cosine"')
parser.add_argument('--enc_dec_type', default='basic', help='"basic" or "less_basic" or "vgg"')
parser.add_argument('--emb_type', default='basic', help='"basic" or "conserved"')
parser.add_argument('--random_weights', action='store_true', help='randomly init SVG weights?')
parser.add_argument('--outer_opt_model_weights', action='store_true', help='optimize SVG weights in outer loop?')
parser.add_argument('--learn_inner_lr', action='store_true', help='optimize inner LR in outer loop?')
parser.add_argument('--reuse_lstm_eps', action='store_true', help='correlated eps samples for prior & posterior?')
parser.add_argument('--only_tailor_on_improvement', action='store_true', help='no outer update if no inner improvement')
parser.add_argument('--only_cn_decoder', action='store_true', help='CN layers in just decoder or encoder as well?')
parser.add_argument('--stack_frames', action='store_true', help='stack every 2 frames channel-wise')
parser.add_argument('--only_twenty_degree', action='store_true', help='for Phys101 ramp, only 20 degree setting?')
parser.add_argument('--center_crop', type=int, default=1080, help='center crop param (phys101)')
parser.add_argument('--crop_upper_right', type=int, default=1080, help='upper right crop param (phys101)')
parser.add_argument('--frame_step', type=int, default=2, help='controls frame rate for Phys101')
parser.add_argument('--num_emb_frames', type=int, default=2, help='number of frames to pass to the embedding')
parser.add_argument('--horiz_flip', action='store_true', help='randomly flip phys101 sequences horizontally (p=.5)?')
parser.add_argument('--train_set_length', type=int, default=256, help='size of training set')
parser.add_argument('--test_set_length', type=int, default=-1, help='size of test set')
parser.add_argument('--baseline', action='store_true', help='evaluate baseline as well?')
parser.add_argument('--stop_grad', action='store_true', help='perform stop grad?')
parser.add_argument('--inner_crit_compare_to', default='prev', help='zero or prev')
parser.add_argument('--encoder_emb', action='store_true', help='use EncoderEmbedding or ConservedEmbedding?')
parser.add_argument('--optimize_emb_enc_params', action='store_true', help='optimize emb.encoder as well as enc.linear?')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--a_dim', type=int, default=8, help='dimensionality of action, or a_t')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--prior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--posterior_rnn_layers', type=int, default=1, help='number of layers')
parser.add_argument('--predictor_rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--no_teacher_force', action='store_true', help='whether or not to do teacher forcing')
parser.add_argument('--add_inner_to_outer_loss', action='store_true', help='optimize inner loss term in outer loop?')
parser.add_argument('--inner_opt_all_model_weights', action='store_true', help='optimize non-CN model weights in inner loop?')
parser.add_argument('--batch_norm_to_group_norm', action='store_true', help='replace BN layers with GN layers')
parser.add_argument('--conv_emb', action='store_true', help='use fully-convolutional embedding?')
parser.add_argument('--warmstart_emb_path', default='', help='path to pretrained embedding weights')


opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

if opt.image_width == 64:
    import models.vgg_64 as model
elif opt.image_width == 128:
    import models.vgg_128 as model
else:
    raise ValueError('image width must be 64 or 128')

val_inner_lr = opt.inner_lr
if opt.val_inner_lr != -1:
    val_inner_lr = opt.val_inner_lr

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# --------- tensorboard configs -------------------------------
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(os.path.join(opt.log_dir), 'tensorboard')

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

if opt.stack_frames:
    assert opt.n_past % 2 == 0 and opt.n_future % 2 == 0
    opt.channels *= 2
    opt.n_past = opt.n_past // 2
    opt.n_future = opt.n_future // 2
opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

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

print(opt)
print('\nDatasets loaded!')

print(f'train_data length: {len(train_data)}')
print(f'num_train_batch: {opt.num_train_batch}')
print(f'test_data length: {len(test_data)}')
print(f'num_val_batch: {opt.num_val_batch}')


# --------- init losses ------------------------------------
def inner_crit(fmodel, gen_seq, mode='mse', num_emb_frames=1, compare_to='prev'):
    # compute embeddings for sequence
    if num_emb_frames == 1:
        embs = [fmodel(frame, mode='emb') for frame in gen_seq]
    elif num_emb_frames == 2:
        stacked_gen_seq = []
        for i in range(1, len(gen_seq)):
            stacked_gen_seq.append(torch.cat((gen_seq[i-1], gen_seq[i]), dim=1))
        embs = [fmodel(frame, mode='emb') for frame in stacked_gen_seq]  # len(embs) = len(gen_seq) - 1
    else:
        raise ValueError
    if mode == 'mse':
        # we penalize the pairwise losses
        if compare_to == 'prev':
            pairwise_inner_losses = torch.stack([F.mse_loss(embs[t-1], embs[t], reduction='none') for t in range(1, len(embs))]).mean(dim=2)
        elif compare_to == 'zero':
            pairwise_inner_losses = torch.stack([F.mse_loss(embs[0], embs[t], reduction='none') for t in range(1, len(embs))]).mean(dim=2)
        elif compare_to == 'zero_and_prev':
            pairwise_inner_losses = torch.stack([F.mse_loss(embs[t-1], embs[t], reduction='none') for t in range(1, len(embs))] + [F.mse_loss(embs[0], embs[t], reduction='none') for t in range(1, len(embs))]).mean(dim=2)
        else:
            raise ValueError('must choose prev or zero or zero_and_prev')
#     elif mode == 'cosine':
#         # cosine distance is 1 minus cosine similarity
#         pairwise_inner_losses = torch.stack([1 - F.cosine_similarity(embs[t-1], embs[t]) for t in range(1, len(embs))])
    else:
        raise NotImplementedError('please use either "mse" or "cosine" mode')
    # total inner loss is just the sum of pairwise losses
    return torch.sum(pairwise_inner_losses, dim=0)


# --------- tracking metrics ------------------------------------
all_outer_losses = []
all_inner_losses = []
all_baseline_outer_losses = []
all_val_outer_losses = []
all_val_inner_losses = []
train_fstate_dict = []  # We'll write higher's fmodel.state_dict() here to compare
val_fstate_dict = []  # We'll write higher's fmodel.state_dict() here to compare
all_emb_weights = []
all_emb_biases = []
all_param_grads = []
all_grad_norms = []
all_emb_norms = []
all_inner_lr_scales = []


# --------- meta-training loop ------------------------------------
# usually only do one trial -- a trial is basically a run through the
# entire meta-training loop
for trial_num in range(opt.num_trials):
    start_epoch = 0

    print(f'TRIAL {trial_num}')
    if opt.random_weights:
        print('initializing model with random weights')
        opt.a_dim = 0 if not opt.use_action else opt.a_dim
        frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim+opt.a_dim, opt.g_dim, opt.rnn_size, opt.predictor_rnn_layers, opt.batch_size)
        posterior = lstm_models.gaussian_lstm(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.posterior_rnn_layers, opt.batch_size)
        prior = lstm_models.gaussian_lstm(opt.g_dim+opt.a_dim, opt.z_dim, opt.rnn_size, opt.prior_rnn_layers, opt.batch_size)
        frame_predictor.apply(utils.init_weights)
        frame_predictor.apply(utils.init_forget_bias_to_one)
        posterior.apply(utils.init_weights)
        prior.apply(utils.init_weights)
        encoder = model.encoder(opt.g_dim, opt.channels, use_cn_layers=True, batch_size=opt.batch_size)
        decoder = model.decoder(opt.g_dim, opt.channels, use_cn_layers=True, batch_size=opt.batch_size)
        encoder.apply(utils.init_weights)
        decoder.apply(utils.init_weights)
        if opt.encoder_emb:
            embedding = EncoderEmbedding(encoder, opt.emb_dim, opt.image_width,
                                         nc=opt.channels, num_emb_frames=opt.num_emb_frames,
                                         batch_size=opt.batch_size)
        elif opt.conv_emb:
            embedding = ConvConservedEmbedding(image_width=opt.image_width,
                                               nc=opt.num_emb_frames * opt.channels)
            print('initialized ConvConservedEmbedding')
        else:
            embedding = ConservedEmbedding(emb_dim=opt.emb_dim, image_width=opt.image_width,
                                           nc=opt.num_emb_frames * opt.channels)
        svg_model = SVGModel(encoder, frame_predictor, decoder, prior, posterior, embedding).cuda()

    # load the model from ckpt
    else:
        start_epoch = int(opt.model_path.split('_')[-1].split('.')[0]) + 1
        print(f'loading model from checkpoint (trained for {start_epoch-1} epochs)')
        ckpt = torch.load(opt.model_path)
        if 'svg_model' in ckpt.keys():
            # this is usually the setting we care about, starting from pre-trained weights
            svg_model = ckpt['svg_model']
            if not hasattr(ckpt['opt'], 'num_emb_frames') or not opt.num_emb_frames == ckpt['opt'].num_emb_frames \
                        or (not hasattr(ckpt['opt'], 'encoder_emb') and opt.encoder_emb) \
                        or (hasattr(ckpt['opt'], 'encoder_emb') and ckpt['opt'].encoder_emb != opt.encoder_emb):
                print(f're-initializing embedding to take {opt.num_emb_frames} frames')
                if opt.encoder_emb:
                    print('\tInitializing new EncoderEmbedding')
                    svg_model.emb = EncoderEmbedding(svg_model.encoder, opt.emb_dim, opt.image_width,
                                                     nc=opt.channels, num_emb_frames=opt.num_emb_frames,
                                                     batch_size=opt.batch_size)
                else:
                    print('\tInitializing new ConservedEmbedding')
                    svg_model.emb = ConservedEmbedding(emb_dim=opt.emb_dim, image_width=opt.image_width,
                                                       nc=opt.num_emb_frames * opt.channels).cuda()
            else:
                print(f'using embedding from {opt.model_path}')
            print('SVG model with pre-trained weights loaded!')
        else:
            svg_model = utils.modernize_model(opt.model_path, opt)
            print('\nOld SVG model with pre-trained weights loaded and modernized!')

    replace_cn_layers(svg_model.encoder, batch_size=opt.batch_size)
    replace_cn_layers(svg_model.decoder, batch_size=opt.batch_size)
    svg_model.frame_predictor.batch_size = opt.batch_size
    svg_model.posterior.batch_size = opt.batch_size
    svg_model.prior.batch_size = opt.batch_size

    if opt.batch_norm_to_group_norm:
        print('replacing batch norm layers with group norm')
        svg_model = utils.batch_norm_to_group_norm(svg_model)

    if opt.warmstart_emb_path != '':
        emb_ckpt = torch.load(opt.warmstart_emb_path)
        if 'emb' in emb_ckpt.keys():
            svg_model.emb = emb_ckpt['emb']
            print(f'loading pretrained embedding from {opt.warmstart_emb_path}')
        else:
            print(f'embedding ckpt path given but no embedding found...')
    svg_model.cuda()
    print(svg_model)


    # For comparing later
    old_state_dict = copy.deepcopy(svg_model.state_dict())
    if opt.baseline:
        # only useful for tailoring or meta-tailoring from a checkpoint
        baseline_svg_model = copy.deepcopy(svg_model)
        for param in baseline_svg_model.parameters():
            param.requires_grad = False
        baseline_svg_model.cuda()
        baseline_svg_model.eval()

    # Init outer optimizer
    emb_params = [p[1] for p in svg_model.emb.named_parameters() if not ('gamma' in p[0] or 'beta' in p[0])]
    if opt.encoder_emb and not opt.optimize_emb_enc_params:
        emb_params = list(svg_model.emb.linear.parameters())
    if opt.outer_opt_model_weights:
        # optimize the non-CN model weights in the outer loop, as well as emb params
        non_cn_params = [p[1] for p in list(svg_model.encoder.named_parameters()) + \
                     list(svg_model.decoder.named_parameters()) \
                     if not ('gamma' in p[0] or 'beta' in p[0])]
        outer_params = non_cn_params + emb_params + \
                       list(svg_model.prior.parameters()) + \
                       list(svg_model.posterior.parameters()) + \
                       list(svg_model.frame_predictor.parameters())
    else:
        outer_params = emb_params

    if opt.learn_inner_lr:
        svg_model.inner_lr_scale = torch.nn.Parameter(torch.tensor(0.))
        outer_params.append(svg_model.inner_lr_scale)

    outer_opt = optim.Adam(outer_params, lr=opt.outer_lr)

    baseline_outer_losses = []
    outer_losses = []
    svg_losses = []
    val_svg_losses = []
    inner_losses = []
    val_outer_losses = []
    val_inner_losses = []
    emb_weights = []
    emb_biases = []
    all_gen = None
    param_grads = []
    grad_norms =  []
    emb_norms = []

    print(f'starting at epoch {start_epoch}')
    for epoch in range(start_epoch, opt.n_epochs):

        print(f'Epoch {epoch} of {opt.n_epochs}')
        train_outer_loss = 0.
        grad_norm_sum = 0.
        emb_norm_sum = 0.
        epoch_inner_losses = []
        epoch_val_inner_losses = []
        epoch_svg_losses = []
        epoch_val_svg_losses = []
        svg_model.eval()

        # validation
        if epoch % opt.num_epochs_per_val == 0:
            print(f'validation {epoch}')
            val_outer_loss = 0.
            baseline_outer_loss = 0.

            svg_model.eval()

            for batch_num in tqdm(range(opt.num_val_batch)):
                batch = next(testing_batch_generator)

                with torch.no_grad():
                    # we optionally evaluate a baseline (untailored) model for comparison
                    prior_epses=[]
                    posterior_epses=[]
                    if opt.baseline:
                        base_gen_seq, base_mus, base_logvars, base_mu_ps, base_logvar_ps = \
                                    predict_many_steps(baseline_svg_model, batch, opt, mode='eval',
                                                      prior_epses=prior_epses, posterior_epses=posterior_epses)
                        base_outer_loss = svg_crit(base_gen_seq, batch, base_mus, base_logvars,
                                                   base_mu_ps, base_logvar_ps, opt).mean()

                # tailoring pass
                val_cached_cn = [None]  # cached cn params
                val_batch_inner_losses = []
                val_batch_svg_losses = []
                for batch_step in range(opt.num_jump_steps + 1):
                    # jump steps are effectively inner steps that have a single higher innerloop_ctx per
                    # iteration, which allows for many inner steps during training without running
                    # into memory issues due to storing the whole dynamic computational graph
                    # associated with unrolling the sequence in the inner loop for many steps
                    gen_seq, mus, logvars, mu_ps, logvar_ps = tailor_many_steps(
                        svg_model, batch, opt=opt, track_higher_grads=False,  # no need for higher grads in val
                        mode='eval',
                        # extra kwargs
                        tailor_losses=val_batch_inner_losses,
                        inner_crit_mode=opt.inner_crit_mode,
                        reuse_lstm_eps=opt.reuse_lstm_eps,
                        val_inner_lr=val_inner_lr,
                        svg_losses=val_batch_svg_losses,
                        only_cn_decoder=opt.only_cn_decoder,
                        # fstate_dict=val_fstate_dict,
                        cached_cn=val_cached_cn,
                        load_cached_cn=(batch_step != 0),
                    )

                    with torch.no_grad():
                        outer_loss = svg_crit(gen_seq, batch, mus, logvars, mu_ps, logvar_ps, opt).mean()

                    val_outer_loss += outer_loss.detach().cpu().numpy().item()
                    if opt.baseline:
                        baseline_outer_loss += base_outer_loss.detach().cpu().numpy().item()

                if opt.num_inner_steps > 0 or opt.num_jump_steps > 0:
                    # fix the inner losses to account for jump step
                    # after the zeroth, take the tailored inner loss (index 1)
                    val_batch_svg_losses = [val_batch_svg_losses[0][0]] + [l[1] for l in val_batch_svg_losses]
                    val_batch_inner_losses = [val_batch_inner_losses[0][0]] + [l[1] for l in val_batch_inner_losses]
                else:
                    val_batch_svg_losses = [val_batch_svg_losses[0][0]]
                    val_batch_inner_losses = [val_batch_inner_losses[0][0]]

                epoch_val_inner_losses.append(val_batch_inner_losses)
                epoch_val_svg_losses.append(val_batch_svg_losses)

            val_inner_losses.append([sum(x) / (opt.num_val_batch) for x in zip(*epoch_val_inner_losses)])
            val_svg_losses.append([sum(x) / (opt.num_val_batch) for x in zip(*epoch_val_svg_losses)])
            val_outer_losses.append(val_outer_loss / (opt.num_val_batch))
            if opt.baseline:
                baseline_outer_losses.append(baseline_outer_loss / (opt.num_val_batch))

            writer.add_scalar('Outer Loss/val', val_outer_losses[-1],
                              (epoch + 1))
            if opt.baseline:
                writer.add_scalar('Outer Loss/baseline', baseline_outer_losses[-1],
                                  (epoch + 1))
                print(f'\tOuter BASE loss:  {baseline_outer_losses[-1]}')
            writer.add_scalars('Inner Loss/val', {f'{i} steps': v for i, v in enumerate(val_inner_losses[-1])},
                               (epoch + 1))
            print(f'\tInner VAL loss:   {val_inner_losses[-1]}')
            writer.add_scalars('SVG Loss/val', {f'{i} steps': v for i, v in enumerate(val_svg_losses[-1])},
                               (epoch + 1))
            print(f'\tSVG VAL loss:     {val_svg_losses[-1]}')
            writer.flush()


        # Training
        print(f'training {epoch}')

        for batch_num in tqdm(range(opt.num_train_batch)):
            batch = next(training_batch_generator)

            train_mode = 'eval' if opt.no_teacher_force else 'train'
            # tailoring pass
            cached_cn = [None]  # cached cn params
            batch_inner_losses = []
            batch_svg_losses = []
            for batch_step in range(opt.num_jump_steps + 1):

                gen_seq, mus, logvars, mu_ps, logvar_ps = tailor_many_steps(
                    svg_model, batch, opt=opt, track_higher_grads=True,
                    mode=train_mode,
                    # extra kwargs
                    tailor_losses=batch_inner_losses,
                    inner_crit_mode=opt.inner_crit_mode,
                    reuse_lstm_eps=opt.reuse_lstm_eps,
                    svg_losses=batch_svg_losses,
                    only_cn_decoder=opt.only_cn_decoder,
                    # fstate_dict=train_fstate_dict,
                    cached_cn=cached_cn,
                    load_cached_cn=(batch_step != 0),
                )

                outer_loss = svg_crit(gen_seq, batch, mus, logvars, mu_ps, logvar_ps, opt).mean()
                if opt.add_inner_to_outer_loss:
                    inner_loss_component = inner_crit(svg_model, gen_seq, mode='mse',
                                                      num_emb_frames=opt.num_emb_frames,
                                                      compare_to=opt.inner_crit_compare_to).mean()
                    print(f'outer_loss = {outer_loss.detach().cpu().numpy().item()}')
                    print(f'inner_loss = {inner_loss_component.detach().cpu().numpy().item()}')
                    outer_loss += inner_loss_component

                train_outer_loss += outer_loss.detach().cpu().numpy().item()

                outer_loss.backward()
                if opt.num_inner_steps > 0:
                    # gradient clipping, and tracking the grad norms
                    param_grads.append([-1. if p.grad is None else torch.norm(p.grad).item() for p in svg_model.parameters()])
                    grad_norm = nn.utils.clip_grad_norm_(svg_model.emb.parameters(), 1000)
                    grad_norm_sum += grad_norm.item()
                    emb_norm_sum += torch.norm(torch.stack(
                        [torch.norm(p.detach()) for p in svg_model.emb.parameters()]
                    )).item()

            if opt.num_inner_steps > 0 or opt.num_jump_steps > 0:
                # fix the inner losses to account for jump step
                batch_inner_losses = [batch_inner_losses[0][0]] + [l[1] for l in batch_inner_losses]
                batch_svg_losses = [batch_svg_losses[0][0]] + [l[1] for l in batch_svg_losses]
            else:
                batch_inner_losses = [batch_inner_losses[0][0]]
                batch_svg_losses = [batch_svg_losses[0][0]]

            epoch_inner_losses.append(batch_inner_losses)
            epoch_svg_losses.append(batch_svg_losses)

            outer_opt.step()
            svg_model.zero_grad(set_to_none=True)

        svg_losses.append([sum(x) / (opt.num_train_batch) for x in zip(*epoch_svg_losses)])
        inner_losses.append([sum(x) / (opt.num_train_batch) for x in zip(*epoch_inner_losses)])
        grad_norms.append(grad_norm_sum / opt.num_train_batch)
        emb_norms.append(emb_norm_sum / opt.num_train_batch)

        outer_losses.append(train_outer_loss / (opt.num_train_batch))
        writer.add_scalar('Outer Loss/train', outer_losses[-1],
                          (epoch + 1))
        writer.add_scalars('Inner Loss/train', {f'{i} steps': v for i, v in enumerate(inner_losses[-1])},
                           (epoch + 1))
        print(f'\tInner TRAIN loss: {inner_losses[-1]}')
        writer.add_scalars('SVG Loss/train', {f'{i} steps': v for i, v in enumerate(svg_losses[-1])},
                           (epoch + 1))
        print(f'\tSVG TRAIN loss: {svg_losses[-1]}')
        writer.add_scalar('Embedding/grad norm', grad_norms[-1],
                          (epoch + 1))
        writer.add_scalar('Embedding/param norm', emb_norms[-1],
                          (epoch + 1))
        if opt.learn_inner_lr:
            writer.add_scalar('Embedding/inner LR scale factor', svg_model.inner_lr_scale,
                              (epoch + 1))
            all_inner_lr_scales.append(svg_model.inner_lr_scale.detach().cpu().item())

        # checkpointing
        if epoch % opt.ckpt_every == 0:
            torch.save({'svg_model': svg_model, 'opt': opt},
                       '%s/model_%d.pth' % (opt.log_dir, epoch))

    all_outer_losses.append(copy.deepcopy(svg_losses))
    all_inner_losses.append(copy.deepcopy(inner_losses))
    all_val_outer_losses.append(copy.deepcopy(val_svg_losses))
    all_baseline_outer_losses.append(copy.deepcopy(baseline_outer_losses))
    all_val_inner_losses.append(copy.deepcopy(val_inner_losses))
    all_emb_weights.append(copy.deepcopy(emb_weights))
    all_emb_biases.append(copy.deepcopy(emb_biases))
    all_param_grads.append(copy.deepcopy(param_grads))
    all_grad_norms.append(copy.deepcopy(grad_norms))
    all_emb_norms.append(copy.deepcopy(emb_norms))


print(5*'\n')
print('all_inner_losses')
print(all_inner_losses)
print('all_val_inner_losses')
print(all_val_inner_losses)
print('all_outer_losses')
print(all_outer_losses)
print('all_val_outer_losses')
print(all_val_outer_losses)
print('all_baseline_outer_losses')
print(all_baseline_outer_losses)
print('all_grad_norms')
print(all_grad_norms)
