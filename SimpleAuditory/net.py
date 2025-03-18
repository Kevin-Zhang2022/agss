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


class Base(nn.Module):
    def __init__(self, channels, num_class, sigma_range=(1,20)):
        super(Base, self).__init__()
        # 加载预训练的 ResNeXt 模型
        self.linear = nn.Linear(channels, channels)  # 2048是ResNeXt的输入尺寸
        self.num_class = num_class
        self.sleaky = snn.Leaky(beta=0.95)
        self.sigma = nn.Parameter(torch.ones(1)*sigma_range.mean())


    def gaussian_pdf(self, x, mean, sigma):
        return (1.0 / (sigma * torch.sqrt(torch.tensor(2) * torch.pi))) * torch.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def generate_gaussian_pdf_matrix(self, sigma):
        matrix = torch.zeros((self.channels, self.channels))
        x_values = torch.linspace(0, self.channels - 1, self.channels)  # 根据 sigma 选择适当的范围
        for i in range(self.channels):
            matrix[i, :] = self.gaussian_pdf(x_values, i, sigma)
        return matrix


class my_mobile(Base):
    def __init__(self, channels, num_class, sigma_range):
        super(my_mobile, self).__init__(channels, num_class, sigma_range)
        # 添加一个线性层

        self.mobile = models.mobilenet_v2(pretrained=True)
        self.sleaky = snn.Leaky(beta=0.95)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    def forward(self, inp):
        mem = self.sleaky.init_leaky()
        spks = []
        for t in range(inp.shape[1]):
            x = inp[:, :, t]
            w = self.generate_gaussian_pdf_matrix(self.sigma)
            I = x - torch.matmul(w, x)
            spk, mem = self.sleaky(I,mem)
            spks.append(spk)
        spks = torch.stack(spks, dim=2)
        spks = self.resnext(spks)
        return spks


if __name__ == "__main__":
# 参数设置
    input_dim = 224  # 输入向量的维度
    model_dim = 128  # Transformer的维度
    num_heads = 8  # 多头注意力的头数
    num_layers = 4  # Transformer层数
    num_classes = 10  # 类别数

    # 实例化模型
    model = tt(input_dim, model_dim, num_heads, num_layers, num_classes)

    # 选择损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

