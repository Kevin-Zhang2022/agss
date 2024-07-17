import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from numpy import (pi, cos, exp, log10, sqrt)
import snntorch as snn
import pyfilterbank.gammatone as gt
import copy
import scipy

class innerhaircell(nn.Module):
    def __init__(
            self,
    ):
        super(innerhaircell, self).__init__()

    def forward(self, x):
        out = (x > 0) * x
        return out

class auditorynerve(nn.Module):
    def __init__(self,
                 uth
    ):
        super(auditorynerve, self).__init__()
        self.uth = uth
        self.sleaky = snn.Leaky(beta=0.95, threshold=uth)

    def forward(self, inp):
        mem = self.sleaky.init_leaky()
        inp = torch.tensor(inp, dtype=torch.float32)
        spikes = []
        for t in range(inp.size(1)):
            x = inp[:, t]
            spk, mem = self.sleaky(x, mem)
            spikes.append(spk)
        spikes = torch.stack(spikes, dim=1)
        return spikes.detach().numpy()




class stellate(nn.Module):
    def __init__(
            self,
            channels=224,
            thr=0.8,
            sigma=10,
    ):
        super(stellate, self).__init__()
        self.channels = channels
        self.w = torch.tensor(self.get_w(sigma=sigma), dtype=torch.float32)
        self.sleaky = snn.Leaky(beta=0.95, threshold=thr)

    def forward(self, inp):
        inp = torch.tensor(inp, dtype=torch.float32)
        spks = []
        mem = self.sleaky.init_leaky()
        for t in range(inp.shape[1]):
            x = inp[:, t]
            I = torch.matmul(self.w, x.unsqueeze(-1)).squeeze(-1)
            spk, mem = self.sleaky(x-I, mem)
            spks.append(spk)
        spks = torch.stack(spks, dim=1)
        return spks.detach().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(inp)
        # plt.imshow(spks)

    def get_w(self, sigma, radius=6):
        x = np.linspace(0, self.channels-1, self.channels)
        out = []
        for i in range(self.channels):
            p = np.exp(-0.5 * ((x - i) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            out.append(p)
            # # 只保留均值左右6个数的数值，其余置为0
            # mask = np.zeros_like(p)
            # start = max(0, i - radius)
            # end = min(self.channels, i + radius+1)
            # mask[start:end] = 1
            # p *= mask
        out = np.stack(out)
        return out


class agss(nn.Module):
    def __init__(
            self,
            channels=224,
            thr=0.5,
            sigma_range=[1, 10],
    ):
        super(agss, self).__init__()
        self.name = 'agss'
        self.channels = channels
        self.sigma_range = torch.tensor(sigma_range, dtype=torch.float32)
        self.linear = nn.Linear(channels, 1, bias=False)
        self.sleaky = snn.Leaky(beta=0.95, threshold=thr)

    def forward(self, inp, **kwargs):
        inp = inp.squeeze(1)
        spks = []
        mem = self.sleaky.init_leaky()
        for t in range(inp.shape[1]):
            x = inp[:, :, t]
            w = (self.sigma_range.mean()+self.linear(x)).clamp(self.sigma_range[0], self.sigma_range[1])
            # print(self.w[0][::100].detach().numpy())
            kernels = self.gaussian_kernel(w.unsqueeze(-1))
            x_mean = self.gaussian_mean(x, kernels)
            spk, mem = self.sleaky(x-x_mean, mem)
            spks.append(spk)
            if len(kwargs):
                kwargs['w_rec'].append(w)
        spks = torch.stack(spks, dim=2).unsqueeze(1)
        if len(kwargs):
            w_rec = torch.stack(kwargs['w_rec'], dim=-1)
            return spks, w_rec
        else:
            return spks


    def gaussian_kernel(self, sigma: torch.Tensor, size=13):
        kernel_range = torch.linspace(-(size // 2), size // 2, size).reshape(1, -1)
        sigma_tensor = sigma.reshape(-1, 1)
        kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
        return kernel_1d

    def gaussian_mean(self, images, kernels):
        out = []
        for i in range(len(kernels)):
            out.append(torch.nn.functional.conv2d(images[i].unsqueeze(0).unsqueeze(0),
                                                  kernels[i].unsqueeze(0).unsqueeze(0).unsqueeze(0), padding='same'))
        out = torch.stack(out, dim=0).squeeze(1).squeeze(1)
        return out


class cgss(nn.Module):
    def __init__(
            self,
            channels=224,
            thr=0.5,
            sigma_range=[1, 10],
    ):
        super(cgss, self).__init__()
        self.name = 'cgss'
        self.channels = channels
        self.sigma_range = torch.tensor(sigma_range, dtype=torch.float32)
        self.w = nn.Parameter(self.sigma_range.mean())
        self.linear = nn.Linear(channels, 1, bias=False)
        self.sleaky = snn.Leaky(beta=0.95, threshold=thr)

    def forward(self, inp,  **kwargs):
        inp = torch.tensor(inp, dtype=torch.float32).squeeze(1)
        spks = []
        mem = self.sleaky.init_leaky()
        for t in range(inp.shape[1]):
            x = inp[:, :, t]
            w = self.sigma_range.mean() + self.w.clamp(self.sigma_range[0], self.sigma_range[1]).repeat(inp.shape[0])

            kernels = self.gaussian_kernel(w.unsqueeze(-1))
            x_mean = self.gaussian_mean(x, kernels)
            spk, mem = self.sleaky(x-x_mean, mem)
            spks.append(spk)
        # print(self.w.detach().numpy())
        spks = torch.stack(spks, dim=2).unsqueeze(1)
        return spks

    def gaussian_kernel(self, sigma: torch.Tensor, size=13):
        kernel_range = torch.linspace(-(size // 2), size // 2, size).reshape(1, -1)
        sigma_tensor = sigma.reshape(-1, 1)
        kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
        return kernel_1d

    def gaussian_mean(self, images, kernels):
        out = []
        for i in range(len(kernels)):
            out.append(torch.nn.functional.conv2d(images[i].unsqueeze(0).unsqueeze(0),
                                                  kernels[i].unsqueeze(0).unsqueeze(0).unsqueeze(0), padding='same'))
        out = torch.stack(out, dim=0).squeeze(1).squeeze(1)
        return out

class cochlea(nn.Module):
    def __init__(self,
                 channels,
                 window,
                 frequency_range=(20, 20000),
                 bandwidth_factor=0.05,
                 sample_rate=44100,
                 ):
        super(cochlea, self).__init__()
        start_band = gt.hertz_to_erbscale(frequency_range[0])
        end_band = gt.hertz_to_erbscale(frequency_range[1])
        self.gtfb = gt.GammatoneFilterbank(samplerate=sample_rate, bandwidth_factor=bandwidth_factor,
                                           startband=start_band, endband=end_band, channels=channels)
        self.cf = self.gtfb.centerfrequencies
        self.window = window
        self.hop = window

    def forward(self, inp):
        out = []

        results = self.gtfb.analyze(inp)
        for (band, state) in results:
            out.append(np.real(band))
        out = np.array(out)

        out = (out > 0)*out
        temp = []
        for t in range(0, out.shape[1]-self.window, self.hop):
            temp.append(np.mean(out[:, t:t+self.window], axis=1))
        out = np.stack(temp, axis=1)
        # np.stach(out.append(temp))

        return out

class harmolearn(nn.Module):
    def __init__(
            self,
            channels=224,
            window=16,
            w_range=(-1, 3)
    ):
        super(harmolearn, self).__init__()
        self.channels = channels
        self.window = window
        self.w = np.ones(channels)
        self.w_range = w_range
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        # plt.plot(self.w_1[20,:])

    def forward(self, inp):
        out = copy.deepcopy(inp)
        pad_size = self.window//2
        inp_padded = np.pad(inp, pad_width=((0, 0), (pad_size, pad_size)))

        mem = self.sleaky.init_leaky()
        w_rec = []
        B_rec = []
        for t in range(inp.shape[1]-self.window):
            B = inp_padded[:, t:t+self.window]
            # B += 1e-5*np.random.random(B.shape)  # box
            B_ram = np.mean(B @ B.T, axis=1)  # row absolute mean
            # B_ram = (B_ram-B_ram.mean())/(B_ram.std()+1e-5)
            B_rec.append(B_ram)

            self.w = 0.5*self.w + 0.5*B_ram
            self.w = np.clip(self.w, self.w_range[0], self.w_range[1])
            w_rec.append(self.w)
            x = inp_padded[:, t]
            spk, mem = self.sleaky(torch.tensor(self.w*x), mem)
            out[:, t] = spk
            # print(self.w[25])
        w_rec = np.stack(w_rec, axis=1)
        B_rec = np.stack(B_rec, axis=1)
        return out

        # self.window = 8
        # self.w_range = (-1.5, 1.5)
        # fig, axes = plt.subplots(2,2,figsize=(8,8))
        # axes[0,0].imshow(inp)
        # axes[0, 0].set_title('inp')
        # axes[0, 1].imshow(out)
        # axes[0, 1].set_title('out')
        # axes[1, 0].imshow(B_rec)
        # axes[1, 0].set_title('B_rec')
        # axes[1, 1].imshow(w_rec)
        # axes[1, 1].set_title('w_rec')


def compute_kurtosis(vector, radius=8):
    # Pad the vector with zeros on both sides
    extended_vector = np.pad(vector, (radius, radius), 'constant', constant_values=(0, 0))
    # Initialize a list to store the kurtosis values
    kurtosis_values = []
    # Compute the kurtosis for each point in the original vector
    for i in range(radius, len(vector) + radius):
        window = extended_vector[i - radius:i + radius + 1]
        kurtosis_values.append(scipy.stats.kurtosis(window))

    return np.array(kurtosis_values)


class kurtosislearn(nn.Module):
    def __init__(
            self,
            channels=224,
            window=16,
            radius=8,
            w_range=(0, 10)
    ):
        super(kurtosislearn, self).__init__()
        self.channels = channels
        self.window = window
        self.radius = radius
        self.w = np.ones(channels)*w_range.mean()
        self.w_range = w_range
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        # plt.plot(self.w_1[20,:])

    def forward(self, inp):
        out = copy.deepcopy(inp)
        pad_size = self.window//2
        inp_padded = np.pad(inp, pad_width=((0, 0), (pad_size, pad_size)))

        mem = self.sleaky.init_leaky()
        w_rec = []
        B_rec = []
        kurtosis_rec = []
        for t in range(inp.shape[1]-self.window):
            vector = np.sum(inp_padded[:, t:t+self.window],dim=1)
            kurtosis = compute_kurtosis(vector)
            kurtosis_rec.append(kurtosis)

            self.w = 0.5*self.w + 0.5*kurtosis
            self.w = np.clip(self.w, self.w_range[0], self.w_range[1])
            w_rec.append(self.w)
            x = inp_padded[:, t]
            spk, mem = self.sleaky(torch.tensor(self.w*x), mem)
            out[:, t] = spk
            # print(self.w[25])
        w_rec = np.stack(w_rec, axis=1)
        B_rec = np.stack(B_rec, axis=1)
        kurtosis_rec = np.stack(kurtosis_rec, axis=1)
        return out

        # self.window = 8
        # self.w_range = (-1.5, 1.5)
        # fig, axes = plt.subplots(2,2,figsize=(8,8))
        # axes[0,0].imshow(inp)
        # axes[0, 0].set_title('inp')
        # axes[0, 1].imshow(out)
        # axes[0, 1].set_title('out')
        # axes[1, 0].imshow(B_rec)
        # axes[1, 0].set_title('B_rec')
        # axes[1, 1].imshow(w_rec)
        # axes[1, 1].set_title('w_rec')




if __name__ == "__main__":
    pass
    # sigma = torch.rand((20))
    # kernels = gaussian_kernel(sigma)
    # kernels = kernels.view(20, 1, 1, 13)
    #
    # images = torch.rand((20, 1, 1, 224))
    # kernels_expanded = kernels.view(20, 1, 1, 13)

    # kernels = kernels.unsqueeze(1).unsqueeze(1).permute(1,0,2,3)

