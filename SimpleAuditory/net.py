import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import snntorch as snn


class tt(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(tt, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.input_projection = nn.Linear(input_dim, model_dim)

    def forward(self, x):
        # x: [batch_size, input_dim, seq_len]
        x = x.permute(2, 0, 1)  # Transformer expects [b(0),c(1),t(2)] ->[t(2), b(0), c(1)]
        x = self.input_projection(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # Global average pooling over the sequence length
        x = self.fc(x)
        return x

# 定义全连接层模型
class fc(nn.Module):
    def __init__(self, in_features, out_features):
        super(fc, self).__init__()
        self.fc0 = nn.Linear(in_features, out_features)
        self.sequence = None

    def forward(self, x):
        x = self.fc0(x)
        return x


# class Base(nn.Module):
#     def __init__(self):
#         super(Base, self).__init__()

def gaussian_pdf(x, mean, sigma):
    out = torch.exp(-0.5 * ((x - mean) / sigma) ** 2)
    out /= (out.sum()+1e-5)
    return out


def generate_gaussian_pdf_matrix(sigma):
    channels = sigma.shape[-1]
    x_values = torch.linspace(0, channels - 1, channels)  # 根据 sigma 选择适当的范围

    if len(sigma.shape) == 1:
        matrix = torch.zeros((channels, channels))
        for i in range(channels):
            matrix[i, :] = gaussian_pdf(x_values, i, sigma[i])
    else:
        matrix = torch.zeros((sigma.shape[0], channels, channels))
        for b in range(sigma.shape[0]):
            for i in range(channels):
                matrix[b, i, :] = gaussian_pdf(x_values, i, sigma[b, i])

    return matrix
    plt.imshow(matrix[0,:,:].detach().numpy())
    plt.show()




if __name__ == "__main__":
    pass
    # x = torch.ones((20, 224, 224))
    # matplotlib.use('tkagg')
    # model = mobile_v2_a(channels=224, num_class=10, sigma_range=(0, 10))
    # model(x)
# 参数设置
#     input_dim = 224  # 输入向量的维度
#     model_dim = 128  # Transformer的维度
#     num_heads = 8  # 多头注意力的头数
#     num_layers = 4  # Transformer层数
#     num_classes = 10  # 类别数

    # 实例化模型
    # model = tt(input_dim, model_dim, num_heads, num_layers, num_classes)

    # 选择损失函数和优化器
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

