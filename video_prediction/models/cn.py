from copy import deepcopy
import torch
import torch.nn as nn

class CNLayer(nn.Module):
    """
    nn.Module wrapper implementing conditional normalization layer
    """
    def __init__(self, shape):
        super(CNLayer, self).__init__()
        assert len(shape) == 4, "CN layer must have 2-dimensional shape"
        self.shape = shape
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x * self.gamma + self.beta

    def extra_repr(self):
        return f'shape={self.shape}'

    def reinitialize_params(self, new_shape=None):
        if new_shape is not None:
            self.shape = new_shape
            self.gamma = nn.Parameter(torch.ones(new_shape).cuda())
            self.beta = nn.Parameter(torch.zeros(new_shape).cuda())
        else:
            self.gamma = nn.Parameter(torch.ones(self.shape).cuda())
            self.beta = nn.Parameter(torch.zeros(self.shape).cuda())

            # maybe bad form to modify .data; don't do that anymore
            # self.gamma.data = torch.ones_like(self.gamma)
            # self.beta.data = torch.zeros_like(self.beta)


def replace_cn_layers(model, batch_size=None):
    for module in model.modules():
        if 'CNLayer' in type(module).__name__:
            if batch_size is not None:
                module.reinitialize_params(new_shape=(batch_size,) + module.shape[1:])
            else:
                module.reinitialize_params()
                
                
def cache_cn_modules(model):
    copied_modules = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'gamma' in name or 'beta' in name:
                copied_modules[name] = param.detach().clone()
            
        # copy modules not named params:
#         for idx, module in enumerate(model.modules()):
#             if 'CNLayer' in type(module).__name__:
#                 copied_modules[idx] = {
#                     'beta': deepcopy(module.beta),
#                     'gamma': deepcopy(module.gamma),
#                 }
    return copied_modules


def load_cached_cn_modules(model, cached_modules):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in cached_modules:
                param.copy_(cached_modules[name])
        model.cuda()
        
#         for idx, module in enumerate(model.modules()):
#             if 'CNLayer' in type(module).__name__:
#                 module.gamma.copy_(cached_modules[idx]['gamma'])
#                 module.beta.copy_(cached_modules[idx]['beta'])
