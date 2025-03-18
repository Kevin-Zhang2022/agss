import timm
import torch.nn as nn
import torchvision
import torchvision.models as models
from SimpleAuditory.model import harmolearn, none, Hebbian, SOM, CT, AT, negative, tsliding, fsliding, tfsliding, ShuffleBasic,agss,cgss
    # ShuffleNetEnsemble, ShuffleNet28ensemble, ShuffleNetbncp, ShuffleBasic, ShuffleNetHierESC10, ShuffleNetHierUS8K
from SimpleAuditory.net import tt, fc
# from torchsummary import summary
from SimpleAuditory.model import Framework
import sys


def adjust_io(models_dict, num_classes):
    channels = 1  # 1
    adjusted_models = {}
    for model_name, model in models_dict.items():
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.features[0][0] = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'resnext':
            model = models.resnext50_32x4d(pretrained=True)
            model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'shufflenet':
            model = models.shufflenet_v2_x1_0(pretrained=True)
            model.conv1[0] = nn.Conv2d(channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        adjusted_models[model_name] = model

    return adjusted_models


def add_layers(model_dic, layers):
    # new_dic = model_dic.copy()
    new_dic = {}

    for layer in layers:
        for name, model in model_dic.items():
            # new_dic[name+'_'+layer.name] = nn.Sequential(layer, model)
            new_dic[name+'_'+layer.name] = Framework(layer, model)
    return new_dic


def get_models(in_channels=1, dataset='esc10', **kwargs):
    # 示例模型字典
    data_class_dic = {'esc10-all': 10, 'us8k-all': 10, 'pump-all': 3, 'engine-all': 4,
                      'esc10-bn': 2, 'esc10-b2': 2, 'esc10-n8': 8, 'esc10-ncp': 2, 'esc10-nc3': 3, 'esc10-np5': 5,
                      'esc10-A01': 2, 'esc10-A1B01': 2, 'esc10-A0B2': 2, 'esc10-A1B0C5': 5, 'esc10-A1B1C3': 3,
                      'us8k-A01': 2, 'us8k-A1B0C2': 2, 'us8k-A0B3': 3, 'us8k-A1B01': 2, 'us8k-A1B1C5': 5}
    postfix = dataset.split('-')[-1]
    num_classes = data_class_dic[dataset]


    models = {
        "shufflenet": torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        # "shufflenet_ensemble_bncp": ShuffleNetEnsemble(num_models=1, num_classes=num_classes),
        # f"ShufffleBasic": ShuffleBasic(num_classes=num_classes),  # do not need stack
        # f"shufflenet": ShuffleNetEnsemble(num_models=1, num_classes=num_classes),  # need stack
        # "shufflenet_n2": ShuffleNetEnsemble(num_models=1, num_classes=num_classes),
        # "shufflenet_n8": ShuffleNetEnsemble(num_models=1, num_classes=num_classes)# basically good
        # "shufflenet2": ShuffleNetEnsemble(num_models=2),  # basically good
        # "shufflenet3": ShuffleNetEnsemble(num_models=3),  # basically good
        # "shufflenet4": ShuffleNetEnsemble(num_models=4),  # basically good
        # "shufflenet5": ShuffleNetEnsemble(num_models=5),  # basically good
        # "shufflenet_bn": torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        # "resnext": torchvision.models.resnext50_32x4d(pretrained=True),  # basically good
        # 'resnet50': torchvision.models.resnet50(pretrained=True),  # basically good
        # 'efficientnet_b0': torchvision.models.efficientnet_b0(pretrained=True),  # basically good esc10 45 74
    }

    # data_class_dic = {'esc10': 10, 'us8k': 10}
    # num_classes = data_class_dic[dataset]
    #
    # # 调整各模型输入层
    models = adjust_io(models, num_classes=num_classes)
    # models = add_layers(models, [agss()])  #

    if kwargs:
        method = kwargs['method']
        if method == 'agss':
            models = add_layers(models, [agss()])
        elif method == 'cgss':
            models = add_layers(models, [cgss()])
    else:
        models = add_layers(models, [agss()])
    return models


# def get_ensemble28model(in_channels=1, dataset='esc10'):
#     # 示例模型字典
#     models= [get_models(dataset='esc10-bn'), get_models(dataset='esc10-b2'), get_models(dataset='esc10-n8')]
#     ensemble28 = ShuffleNet28ensemble(models)
#
#     return ensemble28

# def get_ensemblebncp(dataset='esc10-all'):
#     # 示例模型字典
#
#     if dataset == 'esc10-all':
#         models = [get_models(dataset='esc10-bn'), get_models(dataset='esc10-b2'), get_models(dataset='esc10-ncp'),
#                  get_models(dataset='esc10-nc3'), get_models(dataset='esc10-np5')]
#         expert_ensemble = ShuffleNetHierESC10(models)
#     else:
#         models = [get_models(dataset='us8k-A01'), get_models(dataset='us8k-A1B0C2'), get_models(dataset='us8k-A0B3'),
#                  get_models(dataset='us8k-A1B01'), get_models(dataset='us8k-A1B1C5')]
#         expert_ensemble = ShuffleNetHierUS8K(models)
#     return expert_ensemble

if __name__ == '__main__':
    a =10
    models = get_models(in_channels=1, dataset='us8k')
    a=10




