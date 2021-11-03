import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import utils
import numpy as np
import copy
import contextlib

import higher
from torchviz import make_dot

from models.cn import replace_cn_layers, load_cached_cn_modules, cache_cn_modules
from utils import svg_crit


def inner_crit(fmodel, gen_seq, mode='mse', num_emb_frames=1, compare_to='prev'):
    # compute embeddings for sequence
    if num_emb_frames == 1:
        embs = [fmodel(frame, mode='emb') for frame in gen_seq]
    elif num_emb_frames == 2:
        # TODO: verify exact number of frames coming in 
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
    elif mode == 'cosine':
        # cosine distance is 1 minus cosine similarity
        pairwise_inner_losses = torch.stack([1 - F.cosine_similarity(embs[t-1], embs[t]) for t in range(1, len(embs))])
    else:
        raise NotImplementedError('please use either "mse" or "cosine" mode')
    # total inner loss is just the sum of pairwise losses
    return torch.sum(pairwise_inner_losses, dim=0)


def predict_many_steps(func_model, gt_seq, opt, mode='eval', prior_epses=[], posterior_epses=[]):
    mus, logvars, mu_ps, logvar_ps = [], [], [], []
    if 'Basic' not in type(func_model).__name__:
        func_model.frame_predictor.hidden = func_model.frame_predictor.init_hidden()
        func_model.posterior.hidden = func_model.posterior.init_hidden()
        func_model.prior.hidden = func_model.prior.init_hidden()

#     print(f'predict_many_steps - prior_epses: {prior_epses}')
    
    gen_seq = [gt_seq[0]]

    # skip connections for this prediction sequence: always take the latest GT one
    #     (after opt.n_past time steps, this will be the last GT frame)
    skip = [None]

    for i in range(1, opt.n_eval):
        # TODO: different mode for training, where we get frames for more than just conditioning?
        if mode == 'eval':
            gt = None if i >= opt.n_eval else gt_seq[i]
            # TODO: the following line causes issues, is there an elegant way to do stop grad?
            # x_in = gen_seq[-1].clone().detach()
            # this one seems to work, but is super hacky
            if hasattr(opt, 'stop_grad') and opt.stop_grad:
                x_in = torch.tensor(gen_seq[-1].clone().detach().cpu().numpy()).cuda()
            else:
                # and this one doesn't do stop grad at all
                x_in = gen_seq[-1]
        elif mode == 'train':
            gt = gt_seq[i]
            x_in = gt_seq[i-1]
        else:
            raise NotImplementedError
#         gt = None if i >= opt.n_past else gt_seq[i]

        prior_eps = [None]
        posterior_eps = [None]
#         print(f"i: {i}")
        if i-1 < len(prior_epses) and i-1 < len(posterior_epses):
            prior_eps = prior_epses[i-1]
            posterior_eps = posterior_epses[i-1]
#             print('re-using eps')
#         else:
#             print('sampling eps')
        x_hat, mu, logvar, mu_p, logvar_p, skip = func_model(
            x_in,
            gt, skip, opt,
            i=i, mode=mode,
            prior_eps=prior_eps,
            posterior_eps=posterior_eps,
        )
#         print(f'prior_eps[0,0] after func_model:  {prior_eps[0][0,0]}')
        
        
        if not (i-1 < len(prior_epses) and i-1 < len(posterior_epses)):
#             print('appending to lstm_eps')
            prior_epses.append([prior_eps[0].detach()])
            posterior_epses.append([posterior_eps[0].detach()])


        if i < opt.n_past:
            gen_seq.append(gt_seq[i])
        else:
            gen_seq.append(x_hat)
        # track statistics from prior and posterior for KL divergence loss term
        mus.append(mu)
        logvars.append(logvar)
        mu_ps.append(mu_p)
        logvar_ps.append(logvar_p)

    return gen_seq, mus, logvars, mu_ps, logvar_ps



def tailor_many_steps(svg_model, x, opt, track_higher_grads=True, mode='eval', **kwargs):
    '''
    Perform a round of tailoring.
    '''
    if not hasattr(opt, 'num_emb_frames'):
        opt.num_emb_frames = 1  # number of frames to pass to the embedding
    # re-initialize CN params
    # TODO: uncomment
    replace_cn_layers(svg_model.encoder)
    replace_cn_layers(svg_model.decoder)
    if 'load_cached_cn' in kwargs and kwargs['load_cached_cn'] and \
                            'cached_cn' in kwargs and kwargs['cached_cn'][0] is not None:
        load_cached_cn_modules(svg_model, kwargs['cached_cn'][0])
    # TODO: investigate the effect of not replacing these after jump step zero
    # _cn_beta = list(filter(lambda p: 'beta' in p[0], svg_model.decoder.named_parameters()))
    # print(f'CN layer gamma: {_cn_beta[1]}')

    cn_module_params = list(svg_model.decoder.named_parameters())
    if not 'only_cn_decoder' in kwargs or not kwargs['only_cn_decoder']:
         cn_module_params += list(svg_model.encoder.named_parameters())
    cn_params = [p[1] for p in cn_module_params if ('gamma' in p[0] or 'beta' in p[0])]

    if 'Basic' in type(svg_model).__name__:
        cn_params = list(svg_model.encoder.parameters()) + list(svg_model.decoder.parameters())

    elif opt.inner_opt_all_model_weights:
        # TODO: try with ALL modules, not just enc and dec
        cn_params = list(svg_model.encoder.parameters()) + list(svg_model.decoder.parameters())
#                     + list(svg_model.prior.parameters()) + list(svg_model.posterior.parameters()) + \
#                        list(svg_model.frame_predictor.parameters())

    inner_lr = opt.inner_lr
    if 'val_inner_lr' in kwargs:
        inner_lr = kwargs['val_inner_lr']
        
    
    if not hasattr(opt, 'inner_crit_compare_to'):
        opt.inner_crit_compare_to = 'prev'

    inner_opt = optim.SGD(cn_params, lr=inner_lr)
    if 'adam_inner_opt' in kwargs and kwargs['adam_inner_opt']:
        inner_opt = optim.Adam(cn_params, lr=inner_lr)

    inner_crit_mode = 'mse'
    if 'inner_crit_mode' in kwargs:
        inner_crit_mode = kwargs['inner_crit_mode']

    tailor_losses = []
    svg_losses = []
    ssims = []
    psnrs = []
    mses = []
    epsilons = []
    
    orig_gen_seq = None
    orig_tailor_loss = None

    with higher.innerloop_ctx(
        svg_model,
        inner_opt,
        track_higher_grads=track_higher_grads,
        copy_initial_weights=False
        # copy_initial_weights=True  # grads are zero if we copy initial weights!!
    ) as (fmodel, diffopt):
        prior_epses = []
        posterior_epses = []
    
        # TODO: set requires_grad=False for the outer params
        
        for inner_step in range(opt.num_inner_steps):
            if 'reuse_lstm_eps' not in kwargs or not kwargs['reuse_lstm_eps']:
#                 print('not re-use lstm eps')
                prior_epses = []
                posterior_epses = []
            # inner step: make a prediction, compute inner loss, backprop wrt inner loss
            
#             print(f'beginning of step {inner_step} of tailoring loop: prior_epses = {prior_epses}')
            
            gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps(fmodel, x, opt, mode=mode,
                                                                         prior_epses=prior_epses,
                                                                         posterior_epses=posterior_epses,
                                                                        )

            tailor_loss = inner_crit(fmodel, gen_seq, mode=inner_crit_mode,
                                     num_emb_frames=opt.num_emb_frames,
                                     compare_to=opt.inner_crit_compare_to)
            
            if inner_step == 0:
#                 print('writing orig_gen_seq and orig_tailor_loss')
                orig_gen_seq = [f.detach() for f in gen_seq]
                orig_tailor_loss = tailor_loss.detach()
            
            loss = tailor_loss.mean()
            if opt.learn_inner_lr:
                # we optionally meta-learn the inner learning rate (log scale)
                loss *= torch.exp(list(filter(lambda x: x.size() == torch.Size([]),
                                              [param for param in svg_model.parameters()]))[0])
            diffopt.step(loss)
            
            # cache CN params
            if 'cached_cn' in kwargs:
                kwargs['cached_cn'][0] = cache_cn_modules(fmodel)
            
            # TODO: test outer opt pass in inner loop
            if 'svg_crit' in kwargs:
                outer_loss = kwargs['svg_crit'](gen_seq, x, mus, logvars, mu_ps, logvar_ps, opt).mean()
#                 print(f'outer_loss in inner loop: {outer_loss}')
                outer_loss.backward()

            # track metrics
            # TODO: also compute outer loss at each step for plotting
            tailor_losses.append(tailor_loss.mean().detach().cpu().numpy().item())

            if 'tailor_ssims' in kwargs:
                # compute SSIM for gen_seq batch
                mse, ssim, psnr = utils.eval_seq([f.detach().cpu().numpy() for f in x[opt.n_past:]],
                                                 [f.detach().cpu().numpy() for f in gen_seq[opt.n_past:]])
                ssims.append(copy.deepcopy(ssim))
                psnrs.append(copy.deepcopy(psnr))
                mses.append(copy.deepcopy(mse))
 
            svg_loss = svg_crit(gen_seq, x, mus, logvars, mu_ps, logvar_ps, opt).detach().cpu().numpy().item()
            svg_losses.append(copy.deepcopy(svg_loss))

#         # TODO: remove next two lines
#         _cn_beta = list(filter(lambda p: 'beta' in p[0], fmodel.decoder.named_parameters()))
#         print(f'CN layer beta: {_cn_beta[1]}')

        if 'reuse_lstm_eps' not in kwargs or not kwargs['reuse_lstm_eps']:
#             print('not re-use lstm eps')
            prior_epses = []
            posterior_epses = []

        # generate the final model prediction with the tailored weights
        final_gen_seq, mus, logvars, mu_ps, logvar_ps = predict_many_steps(fmodel, x, opt, mode=mode,
                                                                           prior_epses=prior_epses,
                                                                           posterior_epses=posterior_epses,
                                                                          )

        # track metrics
        tailor_loss = inner_crit(fmodel, final_gen_seq, mode=inner_crit_mode,
                                 num_emb_frames=opt.num_emb_frames,
                                 compare_to=opt.inner_crit_compare_to).detach()
        tailor_losses.append(tailor_loss.mean().detach().cpu().numpy().item())

        svg_loss = svg_crit(final_gen_seq, x, mus, logvars, mu_ps, logvar_ps, opt).detach().cpu().numpy().item()
        svg_losses.append(copy.deepcopy(svg_loss))

        if 'tailor_ssims' in kwargs:
            # compute SSIM for gen_seq batch
            mse, ssim, psnr = utils.eval_seq([f.detach().cpu().numpy() for f in x[opt.n_past:]],
                                             [f.detach().cpu().numpy() for f in final_gen_seq[opt.n_past:]])
            ssims.append(copy.deepcopy(ssim))
            psnrs.append(copy.deepcopy(psnr))
            mses.append(copy.deepcopy(mse))

        if opt.only_tailor_on_improvement and orig_gen_seq is not None and orig_tailor_loss is not None:
            
#             print(f'orig_tailor_loss > tailor_loss: {orig_tailor_loss > tailor_loss}')
            
            # per-batch basis
#             final_gen_seq = orig_gen_seq
            # per-sequence basis
#             print(f'fin.shape: {final_gen_seq[0].shape}')
            mask = (orig_tailor_loss > tailor_loss).detach().view(-1, 1, 1, 1)
            print(f'percent of sequences in batch improved by tailoring: {mask.float().mean()}')
#             print(f'mask shape: {mask.shape}')

            final_gen_seq = [torch.where(mask, fin, orig)
                             for fin, orig in zip(final_gen_seq, orig_gen_seq)]

            svg_loss = svg_crit(final_gen_seq, x, mus, logvars, mu_ps, logvar_ps, opt).detach().cpu().numpy().item()
            svg_losses.append(copy.deepcopy(svg_loss))
            
            tailor_loss = inner_crit(fmodel, final_gen_seq, mode=inner_crit_mode,
                                     num_emb_frames=opt.num_emb_frames,
                                     compare_to=opt.inner_crit_compare_to).detach()
            tailor_losses.append(tailor_loss.mean().detach().cpu().numpy().item())


            if 'tailor_ssims' in kwargs:
                # compute SSIM for gen_seq batch
                mse, ssim, psnr = utils.eval_seq([f.detach().cpu().numpy() for f in x[opt.n_past:]],
                                                 [f.detach().cpu().numpy() for f in final_gen_seq[opt.n_past:]])
                ssims.append(copy.deepcopy(ssim))
                psnrs.append(copy.deepcopy(psnr))
                mses.append(copy.deepcopy(mse))

#     print(f'    avg INNER losses: {sum(tailor_losses) / len(tailor_losses)}')
    # track metrics
    if 'tailor_losses' in kwargs:
        kwargs['tailor_losses'].append(tailor_losses)

    if all(m in kwargs for m in ('tailor_ssims', 'tailor_psnrs', 'tailor_mses')):
        kwargs['tailor_ssims'].append(ssims)
        kwargs['tailor_psnrs'].append(psnrs)
        kwargs['tailor_mses'].append(mses)

    if 'svg_losses' in kwargs:
        kwargs['svg_losses'].append(svg_losses)

    # we need the first and second order statistics of the posterior and prior for outer (SVG) loss
    return final_gen_seq, mus, logvars, mu_ps, logvar_ps
