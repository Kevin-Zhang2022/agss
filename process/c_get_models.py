import timm
import torch.nn as nn
import torchvision
import torchvision.models as models
from SimpleAuditory.net import tt, fc
from SimpleAuditory.model import agss, cgss
from torchsummary import summary

def adjust_io(models_dict, num_classes):

    adjusted_models = {}
    for model_name, model in models_dict.items():
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=True)
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == 'resnext':
            model = models.resnext50_32x4d(pretrained=True)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'shufflenet':
            model = models.shufflenet_v2_x1_0(pretrained=True)
            model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)

        adjusted_models[model_name] = model

    return adjusted_models


def add_layers(model_dic, layers):
    new_dic = model_dic.copy()
    # new_dic = {}
    for layer in layers:
        for name, model in model_dic.items():
            new_dic[name+'_'+layer.name] = nn.Sequential(layer, model)
    return new_dic


def get_models(in_channels=1, dataset='us8k'):
    # 示例模型字典
    data_class_dic = {'esc10': 10, 'us8k': 10}
    num_classes = data_class_dic[dataset]
    models = {
        # 'fc': fc(in_features=224, out_features=num_classes),
        "resnext": torchvision.models.resnext50_32x4d(pretrained=True),  # basically good
        "shufflenet": torchvision.models.shufflenet_v2_x1_0(pretrained=True),  # basically good
        'resnet50': torchvision.models.resnet50(pretrained=True),  # basically good
        'efficientnet_b0': torchvision.models.efficientnet_b0(pretrained=True),  # basically good
        # 'mobilenet_v2': torchvision.models.mobilenet_v2(pretrained=True),  # basically good esc10 45 74
    }

    # data_class_dic = {'esc10': 10, 'us8k': 10}
    # num_classes = data_class_dic[dataset]
    #
    # # 调整各模型输入层
    models = adjust_io(models, num_classes=num_classes)
    models = add_layers(models, [cgss(), agss()])

    return models


if __name__ == '__main__':
    a =10
    models = get_models(in_channels=1, dataset='us8k')
    a=10




