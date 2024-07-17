import copy

import torch
from process.c_get_models import get_models

models = get_models(in_channels=1, dataset='esc10')

model_list = []
for name, model in models.items():
    if name == 'shufflenet_cgss':
        state_dic = torch.load('bestmodel/shufflenet_cgss_esc10_coc_out.pth')
        model_clone = copy.deepcopy(model)
        model_list.append(model_clone.load_state_dict(state_dic))



