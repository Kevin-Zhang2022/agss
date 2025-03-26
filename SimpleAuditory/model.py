import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from numpy import (pi, cos, exp, log10, sqrt)
import snntorch as snn
import pyfilterbank.gammatone as gt
import copy
import torch.nn.functional as F

import torchvision
import torchvision.models as models

class Framework(nn.Module):
    def __init__(
            self,
            prefix,
            cnn
    ):
        super(Framework, self).__init__()
        self.a = prefix
        self.b = cnn

    def forward(self, x, **kwargs):
        x = self.a(x, **kwargs)
        x = self.b(x)
        return x

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
        # plt.plot(self.w_1[20,:])

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
            w_range=(-1, 1)
    ):
        super(harmolearn, self).__init__()
        self.channels = channels
        self.window = window
        self.w = np.random.normal(loc=0.8, scale=0.05, size=(channels,1))
        # self.w = 0.8*np.ones((channels, 1)) + np.random
        self.w_range = w_range
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        self.name = 'hl'
        # plt.plot(self.w_1[20,:])

    def forward(self, inp, **kwargs):
        inp = inp.detach().numpy()
        batch, h, w = inp.shape
        out = np.zeros((batch, 1, self.channels, self.channels))
        cc_rec = np.zeros((batch, self.channels, self.channels))
        w_rec = np.zeros((batch, self.channels, self.channels))
        for b in range(batch):
            inp_b = inp[b, :, :]
            pad_size = self.window
            inp_p = np.pad(inp_b, pad_width=((0, 0), (pad_size, 0)), mode='constant', constant_values=0)
            mem = self.sleaky.init_leaky()
            for t in range(self.window, inp_p.shape[1]):
                B = inp_p[:, t-self.window:t]
                cc_mat = np.corrcoef(B + 1e-5 * np.random.random(B.shape))
                cc_mat = np.expand_dims(cc_mat, axis=-1)
                cc_vec = np.mean(abs(cc_mat), axis=-2)
                # plt.imshow(cc_mat[:,:,0], aspect='auto')
                # plt.figure()
                # plt.plot(cc_vec)
                cc_vec = (cc_vec-cc_vec.min())/(cc_vec.max()-cc_vec.min())
                cc_rec[b, :, t-self.window] = cc_vec.squeeze(-1)

                self.w = 0.1*self.w + 0.9*cc_vec
                w = self.w.clip(self.w_range[0], self.w_range[1])
                w_rec[b, :, t-self.window] = w.squeeze(-1)
                x = np.expand_dims(inp_p[:, t], axis=-1)
                I = self.w*x
                # I_nor = (I-I.mean())/(I.std()+1e-5)
                # 224 1
                spk, mem = self.sleaky(torch.tensor(I, dtype=torch.float32), mem)
                out[b, :, :, t-self.window] = spk.squeeze(-1).detach().numpy()
        if 'w_rec' in kwargs:
            kwargs['w_rec'] = np.stack(w_rec, axis=-1)
            kwargs['h_rec'] = np.stack(cc_rec, axis=-1)
            return out, kwargs['w_rec'], kwargs['h_rec']
        else:
            return torch.tensor(out, dtype=torch.float32)


        # row = 2
        # col = 5
        # fig, axes = plt.subplots(row, col, figsize=(8, 4))
        # for i in range(row):
        #     for j in range(col):
        #         axes[i, j].imshow(inp[i*col+j, :, :], aspect='auto')
        #         axes[i, j].set_title(f"{i*col+j:d}", fontsize=6)
        # plt.tight_layout()
        #
        # fig, axes = plt.subplots(row, col, figsize=(8, 4))
        # for i in range(row):
        #     for j in range(col):
        #         axes[i, j].imshow(out[i*col+j, 0, :, :], aspect='auto')
        #         axes[i, j].set_title(f"{i*col+j:d}", fontsize=6)
        # plt.tight_layout()
        #
        # cc_stack = np.stack(cc_rec, axis=0)
        # fig, axes = plt.subplots(row, col, figsize=(8, 4))
        # for i in range(row):
        #     for j in range(col):
        #         axes[i, j].imshow(cc_stack[i*col+j, :, :], aspect='auto')
        #         axes[i, j].set_title(f"{i*col+j:d}", fontsize=6)
        # plt.title('cc')
        # plt.tight_layout()
        #
        #
        # w_stack = np.stack(w_rec, axis=0)
        # fig, axes = plt.subplots(row, col, figsize=(8, 4))
        # for i in range(row):
        #     for j in range(col):
        #         axes[i, j].imshow(w_stack[i*col+j, :, :], aspect='auto')
        #         axes[i, j].set_title(f"{i*col+j:d}", fontsize=6)
        # plt.title('w')
        # plt.tight_layout()
        # self.window =
        # self.w_range = (-1.5, 1.5)

        # row = 7
        # col = 8
        # # start = 100
        # for p in range(4):
        #     fig, axes = plt.subplots(row,col, figsize=(12,8))
        #     for i in range(row):
        #         for j in range(col):
        #             axes[i, j].imshow(out[p*56+i*7+j], aspect='auto')
        #             axes[i, j].set_title(f"{p*56+i*7+j:d}", fontsize=6)
        #     plt.tight_layout()


        # axes[0, 0].imshow(inp[:, self.window:])
        # axes[0, 0].set_title('inp')
        # axes[0, 1].imshow(out)
        # axes[0, 1].set_title('out')
        # axes[1, 0].imshow(cc_stack)
        # axes[1, 0].set_title('cc_rec')
        # axes[1, 1].imshow(w_stack)
        # axes[1, 1].set_title('w_rec')


# agss new
class agss(nn.Module):
    def __init__(
            self,
            channels=224,
            # thr=0.7,
            sigma_range=[0.4, 10],
    ):
        super(agss, self).__init__()
        self.name = 'agss'
        self.channels = channels
        self.sigma_range = torch.tensor(sigma_range, dtype=torch.float32)
        self.linear = nn.Linear(channels, 1, bias=False)
        self.kernel_size = 13
        # self.linear2 = nn.Linear(channels, 1, bias=False)
        # self.w = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)

    def forward(self, inp, **kwargs):
        inp = inp.squeeze(1)  # Ensure correct input shape
        B, C, T = inp.shape  # Batch, Channels, Time

        # Precompute constants
        sigma_mean = self.sigma_range.mean()
        sigma_min, sigma_max = self.sigma_range[0], self.sigma_range[1]

        # Initialize memory
        mem = self.sleaky.init_leaky()
        # if len(kwargs):
        #     kwargs['w_rec'] = []  # Ensure correct storage

        # Compute `w` in a batch-efficient way
        x_flat = inp.permute(0, 2, 1).reshape(B * T, C)  # Flatten batch & time
        w_flat = (sigma_mean+self.linear(x_flat)*(sigma_max-sigma_min)).clamp(sigma_min, sigma_max)
        # w_flat = 0.4 * torch.ones(x_flat.shape[0], 1)


        # Correct the reshape to [B, T, 1]
        w = w_flat.view(B, T, 1)
        # plt.plot(w[31, :, 0].detach().numpy())
        # plt.show()
        # Compute kernels and means
        kernels = self.gaussian_kernel(w.unsqueeze(-1), size=self.kernel_size)  # Ensure correct input shape
        kernels = kernels.view(B, T, self.kernel_size)
        # plt.plot(kernels[0, 1, :].detach().numpy())
        x_mean = self.gaussian_mean(inp.permute(0, 2, 1), kernels)
        # Process through the loop
        # spk, mem = self.sleaky(inp - x_mean, mem)
        I = torch.empty((B, C, T), device=inp.device)
        spk = torch.empty((B, C, T), device=inp.device)  # Pre-allocate tensor
        for t in range(T):
            I[:,:,t] = inp[:, :, t] - torch.relu(inp[:, :, t] - x_mean[:, :, t])

            spk[:, :, t], mem = self.sleaky(I[:,:,t], mem)
            # spk[:, :, t], mem = self.sleaky(inp[:, :, t] - x_mean[:, :, t], mem.clamp(min=0.0))
            # spk[:, :, t], mem = self.sleaky(inp[:, :, t] - x_mean[:, :, t], mem.clamp(min=0.0))
            # print(mem[0, 24])
        if 'w_rec' in kwargs:
            return spk.squeeze(0).detach().numpy(), w.squeeze(0).squeeze(-1).detach().numpy()

        return spk.unsqueeze(1)

        # plt.figure()
        # plt.imshow(inp[0,:,:].detach().numpy())
        # plt.figure()
        # plt.imshow(spk[0,:,:].detach().numpy())
        # plt.figure()
        # plt.imshow(I[0,:,:].detach().numpy())
        # plt.show()
        # plt.figure()
        # plt.imshow(x_mean[0,:,:].detach().numpy())
        # plt.figure()
        # plt.imshow(torch.relu(inp[0,:,:]-x_mean[0,:,:]).detach().numpy())
        # plt.figure()
        # plt.plot(w[0,:,0].detach().numpy())
        # plt.figure()


    def gaussian_kernel(self, sigma: torch.Tensor, size=13):
        kernel_range = torch.linspace(-(size // 2), size // 2, size).reshape(1, -1)
        kernel_range_all = torch.linspace(-(size // 2) * 10, size // 2 * 10, size * 10).reshape(1, -1)
        sigma_tensor = sigma.reshape(-1, 1)
        # kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)

        kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)
        kernel_1d_all = torch.exp(-0.5 * (kernel_range_all / sigma_tensor) ** 2)
        # kernel_1d = torch.exp(-kernel_range ** 2 / (2 * sigma_tensor ** 2))

        kernel_1d = kernel_1d / kernel_1d_all.sum(1, keepdim=True)
        # kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
        # kernel_1d[:,(size // 2)]=0
        return kernel_1d
    # plt.plot(kernel_1d[0, :].detach().numpy())

    def gaussian_mean(self, input_tensor, kernels):
        # out = []
        # for i in range(len(kernels)):
        #     out.append(torch.nn.functional.conv2d(input_tensor[i].unsqueeze(0).unsqueeze(0),
        #                                           kernels[i].unsqueeze(0).unsqueeze(0).unsqueeze(0), padding='same'))
        # out = torch.stack(out, dim=0).squeeze(1).squeeze(1)
        # return out
        B, H, W = input_tensor.shape
        _, _, K = kernels.shape  # Kernel size in last dimension

        # Compute required padding
        pad_size = (K - 1) // 2  # Symmetric padding for proper dot product

        # Pad input along last dimension
        padded_input = F.pad(input_tensor, (pad_size, pad_size), mode="constant", value=0)  # [B, H, W + 2*pad_size]

        # Apply dot product using unfolding (sliding window approach)
        unfolded = padded_input.unfold(dimension=2, size=K, step=1)  # Shape: [B, H, W, K]

        # Compute dot product along last axis
        result = (unfolded * kernels.unsqueeze(2)).sum(dim=-1)  # [B, H, W]

        return result.permute(0, 2, 1)


# cgss new
class cgss(nn.Module):
    def __init__(
            self,
            channels=224,
            # thr=0.7,
            sigma_range=[0.4, 10],
    ):
        super(cgss, self).__init__()
        self.name = 'cgss'
        self.channels = channels
        self.sigma_range = torch.tensor(sigma_range, dtype=torch.float32)
        self.w = nn.Parameter(torch.tensor((sigma_range[0]+sigma_range[1])/2, dtype=torch.float32))
        self.linear = nn.Linear(channels, 1, bias=False)
        self.sleaky = snn.Leaky(beta=0.97, threshold=1)
        self.kernel_size = 13

    def forward(self, inp,  **kwargs):
        inp = inp.squeeze(1)  # Ensure correct dtype and shape
        B, C, T = inp.shape  # Batch, Channels, Time

        # Initialize memory and spike storage

        sigma_mean = self.sigma_range.mean()
        sigma_min, sigma_max = self.sigma_range[0], self.sigma_range[1]

        # Compute `w` once and expand for all time steps
        w = self.w.clamp(sigma_min, sigma_max)
        # w = (sigma_mean + -0.3 * (sigma_max - sigma_min)).clamp(sigma_min, sigma_max)
        # w = torch.tensor(10)

        w = w.repeat(B).unsqueeze(-1)  # Expand across batch

        # Compute kernels once for all time steps
        kernels = self.gaussian_kernel(w.unsqueeze(-1), size=self.kernel_size)

        # Compute x_mean for all time steps at once
        x_mean = self.gaussian_mean(inp.permute(0, 2, 1), kernels.unsqueeze(1).repeat(1, 224, 1))  # [B, C, T]

        # plt.figure()
        # plt.imshow(x_mean[2,:,:].detach().numpy())
        # plt.figure()
        # plt.imshow(inp[2,:,:].detach().numpy())
        # Compute spikes in a batch-wise manner
        # spk, _ = self.sleaky(inp - x_mean, mem)
        mem = self.sleaky.init_leaky()
        spk = torch.empty((B, C, T), device=inp.device)  # Pre-allocate output tensor
        I= torch.empty((B, C, T), device=inp.device)
        for t in range(T):
            I[:,:,t] = inp[:, :, t] - torch.relu(inp[:, :, t] - x_mean[:, :, t])

            spk[:, :, t], mem = self.sleaky(I[:,:,t], mem)

        if 'w_rec' in kwargs:
            return spk.squeeze(0).detach().numpy(), w.squeeze(0).detach().numpy()*np.ones(224)

        return spk.unsqueeze(1)

        # plt.figure()
        # plt.imshow(inp[0, :, :].detach().numpy())
        # plt.figure()
        # plt.imshow(spk[0, :, :].detach().numpy())
        # plt.figure()
        # plt.imshow(I[0, :, :].detach().numpy())

        # plt.show()


    def gaussian_kernel(self, sigma: torch.Tensor, size=13):
        kernel_range = torch.linspace(-(size // 2), size // 2, size).reshape(1, -1)
        kernel_range_all = torch.linspace(-(size // 2) * 10, size // 2 * 10, size * 10).reshape(1, -1)
        sigma_tensor = sigma.reshape(-1, 1)
        # kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)

        kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)
        kernel_1d_all = torch.exp(-0.5 * (kernel_range_all / sigma_tensor) ** 2)
        # kernel_1d = torch.exp(-kernel_range ** 2 / (2 * sigma_tensor ** 2))

        kernel_1d = kernel_1d / kernel_1d_all.sum(1, keepdim=True)
        # kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
        # kernel_1d[:,(size // 2)]=0
        return kernel_1d

        # plt.plot(kernel_1d[0,:])
        # kernel_1d[0].sum()

    def gaussian_mean(self, input_tensor, kernel):
        # out = []
        # for i in range(len(kernels)):
        #     out.append(torch.nn.functional.conv2d(images[i].unsqueeze(0).unsqueeze(0),
        #                                           kernels[i].unsqueeze(0).unsqueeze(0).unsqueeze(0), padding='same'))
        # out = torch.stack(out, dim=0).squeeze(1).squeeze(1)
        # return out
        B, H, W = input_tensor.shape
        _, _, K = kernel.shape  # Kernel size in last dimension

        # Compute required padding
        pad_size = (K - 1) // 2  # Symmetric padding for proper dot product

        # Pad input along last dimension
        padded_input = F.pad(input_tensor, (pad_size, pad_size), mode="constant", value=0)  # [B, H, W + 2*pad_size]

        # Apply dot product using unfolding (sliding window approach)
        unfolded = padded_input.unfold(dimension=2, size=K, step=1)  # Shape: [B, H, W, K]

        # Compute dot product along last axis
        result = (unfolded * kernel.unsqueeze(2)).sum(dim=-1)  # [B, H, W]


        return result.permute(0,2,1)





# class agss(nn.Module):
#     def __init__(
#             self,
#             channels=224,
#             thr=0.5,
#             sigma_range=[1, 10],
#     ):
#         super(agss, self).__init__()
#         self.name = 'agss'
#         self.channels = channels
#         self.sigma_range = torch.tensor(sigma_range, dtype=torch.float32)
#         self.linear = nn.Linear(channels, 1, bias=False)
#         self.sleaky = snn.Leaky(beta=0.95, threshold=thr)
#
#     def forward(self, inp, **kwargs):
#         inp = inp.squeeze(1)
#         spks = []
#         mem = self.sleaky.init_leaky()
#         mem = torch.zeros_like(inp[:, :, 0])
#         w_rec = []
#         for t in range(inp.shape[1]):
#             x = inp[:, :, t]
#             w = (self.sigma_range.mean()+self.linear(x)).clamp(self.sigma_range[0], self.sigma_range[1])
#             # print(w[0][::100].detach().numpy())
#             kernels = self.gaussian_kernel(w.unsqueeze(-1))
#             m_mean = self.gaussian_mean(mem, kernels)
#             spk, mem = self.sleaky(x, mem-m_mean)
#             spks.append(spk)
#             w_rec.append(w)
#         spks = torch.stack(spks, dim=2).unsqueeze(1)
#         if 'w_rec' in kwargs:
#             # w_rec = torch.stack(kwargs['w_rec'], dim=-1)
#             return spks, w_rec
#         else:
#             return spks
#
#
#     def gaussian_kernel(self, sigma: torch.Tensor, size=13):
#         kernel_range = torch.linspace(-(size // 2), size // 2, size).reshape(1, -1)
#         sigma_tensor = sigma.reshape(-1, 1)
#         kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)
#         kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
#         return kernel_1d
#
#     def gaussian_mean(self, images, kernels):
#         out = []
#         for i in range(len(kernels)):
#             out.append(torch.nn.functional.conv2d(images[i].unsqueeze(0).unsqueeze(0),
#                                                   kernels[i].unsqueeze(0).unsqueeze(0).unsqueeze(0), padding='same'))
#         out = torch.stack(out, dim=0).squeeze(1).squeeze(1)
#         return out


# class cgss(nn.Module):
#     def __init__(
#             self,
#             channels=224,
#             thr=0.5,
#             sigma_range=[1, 10],
#     ):
#         super(cgss, self).__init__()
#         self.name = 'cgss'
#         self.channels = channels
#         self.sigma_range = torch.tensor(sigma_range, dtype=torch.float32)
#         self.w = nn.Parameter(self.sigma_range.mean())
#         self.linear = nn.Linear(channels, 1, bias=False)
#         self.sleaky = snn.Leaky(beta=0.95, threshold=thr)
#
#     def forward(self, inp,  **kwargs):
#         inp = torch.tensor(inp, dtype=torch.float32).squeeze(1)
#         spks = []
#         mem = self.sleaky.init_leaky()
#         for t in range(inp.shape[1]):
#             x = inp[:, :, t]
#             w = self.sigma_range.mean() + self.w.clamp(self.sigma_range[0], self.sigma_range[1]).repeat(inp.shape[0])
#
#             kernels = self.gaussian_kernel(w.unsqueeze(-1))
#             x_mean = self.gaussian_mean(x, kernels)
#             spk, mem = self.sleaky(x-x_mean, mem)
#             spks.append(spk)
#         # print(self.w.detach().numpy())
#         spks = torch.stack(spks, dim=2).unsqueeze(1)
#         return spks
#
#     def gaussian_kernel(self, sigma: torch.Tensor, size=13):
#         kernel_range = torch.linspace(-(size // 2), size // 2, size).reshape(1, -1)
#         sigma_tensor = sigma.reshape(-1, 1)
#         kernel_1d = torch.exp(-0.5 * (kernel_range / sigma_tensor) ** 2)
#         kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
#         return kernel_1d
#
#     def gaussian_mean(self, images, kernels):
#         out = []
#         for i in range(len(kernels)):
#             out.append(torch.nn.functional.conv2d(images[i].unsqueeze(0).unsqueeze(0),
#                                                   kernels[i].unsqueeze(0).unsqueeze(0).unsqueeze(0), padding='same'))
#         out = torch.stack(out, dim=0).squeeze(1).squeeze(1)
#         return out


class none(nn.Module):
    def __init__(self):
        super(none, self).__init__()
        self.name = 'none'

    def forward(self, inp, **kwargs):
        inp = inp.unsqueeze(1)
        return inp


class tsliding(nn.Module):
    def __init__(self,
                 portion):
        super(tsliding, self).__init__()
        self.name = 'ts' + str(portion).replace(".", "")
        self.portion = portion

    def forward(self, inp, **kwargs):
        inp = inp.unsqueeze(1)
        roll_amount = torch.randint(-int(112*self.portion), int(112*self.portion), (1,)).item()
        # # Perform the roll operation
        inp = torch.roll(inp, shifts=roll_amount, dims=-1)
        return inp


class fsliding(nn.Module):
    def __init__(self,
                 portion):
        super(fsliding, self).__init__()
        self.name = 'fs' + str(portion).replace(".", "")
        self.portion = portion

    def forward(self, inp, **kwargs):
        inp = inp.unsqueeze(1)
        roll_amount = torch.randint(-int(112*self.portion), int(112*self.portion), (1,)).item()
        # # Perform the roll operation
        inp = torch.roll(inp, shifts=roll_amount, dims=-2)
        return inp


class tfsliding(nn.Module):
    def __init__(self,
                 portion):
        super(tfsliding, self).__init__()
        self.name = 'tfs' + str(portion).replace(".", "")
        self.portion = portion

    def forward(self, inp, **kwargs):
        inp = inp.unsqueeze(1)
        roll_amount = torch.randint(-int(112*self.portion), int(112*self.portion), (1,)).item()
        inp = torch.roll(inp, shifts=roll_amount, dims=-1)
        roll_amount = torch.randint(-int(112*self.portion), int(112*self.portion), (1,)).item()
        inp = torch.roll(inp, shifts=roll_amount, dims=-2)
        return inp


class negative(nn.Module):
    def __init__(self,
                 portion):
        super(negative, self).__init__()
        self.name = 'negative' + str(portion).replace(".", "")
        self.portion = portion

    def forward(self, inp, **kwargs):
        # inp = inp.unsqueeze(1)
        # batch_size = inp.shape[0]
        # num_to_flip = int(batch_size * self.portion)
        # indices = torch.randperm(batch_size)[:num_to_flip]
        # # temp =
        #   # Clone to avoid modifying the original tensor
        # inp = inp.clone()  # Clone to ensure no modification of the original tensor
        # temp = torch.abs(1 - inp[indices])  # Compute the modified values
        # inp = inp.scatter(0, indices.unsqueeze(1), temp)  # Out-of-place update
        # inp[indices] = abs(1-inp[indices])

        # plt.imshow(inp[indices[0], 0, :, :].detach().numpy(), aspect='auto')
        # plt.show()
        # = torch.tensor(not(),dtype=torch.floatt32)

        # inp = torch.roll(inp, shifts=roll_amount, dims=3)

        inp = inp.unsqueeze(1)  # ✅ Keep unsqueeze at the beginning (batch_size, 1, H, W)
        batch_size, _, H, W = inp.shape  # Get shape
        num_to_flip = int(batch_size * self.portion)

        # Select indices to modify
        indices = torch.randperm(batch_size)[:num_to_flip].unsqueeze(1).unsqueeze(2).unsqueeze(
            3)  # Shape (num_to_flip, 1, 1, 1)

        # Expand indices to match input shape (num_to_flip, 1, H, W)
        indices = indices.expand(-1, 1, H, W)  # Expand to match input dimensions

        # Clone input to ensure no in-place modification
        inp = inp.clone()

        # Generate mask for selected indices
        mask = torch.zeros_like(inp, dtype=torch.bool)  # Create a boolean mask
        mask.scatter_(0, indices, True)  # Mark selected indices as True

        # Compute modified values (torch.abs(1 - inp) ensures differentiability)
        modified_values = torch.abs(1 - inp)

        # Apply out-of-place modification using torch.where()
        inp = torch.where(mask, modified_values, inp)

        return inp

        # plt.imshow(inp[0,0,:,:].detach().numpy())


class Hebbian(nn.Module):
    def __init__(
            self,
            channels=224,
            initial_mean=0,
            initial_std=np.sqrt(2. / 224)
    ):
        super(Hebbian, self).__init__()
        self.w = np.random.normal(initial_mean, initial_std, (channels, channels))
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        self.name = 'heb'
        self.channels = channels

    def forward(self, inp,  **kwargs):
        inp = inp.detach().numpy()
        batch, h, w = inp.shape

        out = np.zeros((batch, 1, self.channels, self.channels))
        # cc_rec = np.zeros((batch, self.channels, self.channels))
        # w_rec = np.zeros((batch, self.channels, self.channels))
        mem = self.sleaky.init_leaky()
        for t in range(inp.shape[-1]):
            x = inp[:, :, t]
            I = np.matmul(self.w, np.expand_dims(x, axis=-1))
            spk, mem = self.sleaky(torch.tensor(I, dtype=torch.float32), mem)
            out[:, 0, :, t] = spk.squeeze(-1).detach().numpy()
        # ind = 1
        # plt.figure()
        # plt.imshow(out[ind, 0, :, :],aspect='auto')
        # plt.figure()
        # plt.imshow(inp[ind, :, :], aspect='auto')
        if 'lr' in kwargs:
            lr = kwargs['lr']
            # self.w = self.update_weights(inp, out.squeeze(1), self.w, A_plus=lr, A_minus=lr, sf=1e-3)
            self.w = self.update_weights_normal(inp, out.squeeze(1), self.w, lr=lr, sf=1e-3)
        return torch.tensor(out, dtype=torch.float32)

    def update_weights(self, pre_spikes_matrix, post_spikes_matrix, weights,
                       A_plus, A_minus, sf, tau_plus=20.0, tau_minus=20.0):

        batch, N_pre, T = pre_spikes_matrix.shape
        _, N_post, _ = post_spikes_matrix.shape

        delta_w = np.zeros_like(weights)
        time_diff_matrix = np.arange(T).reshape(1, T) - np.arange(T).reshape(T, 1)
        # Compute the exponential decay based on time differences
        LTP_matrix = A_plus * np.exp(-time_diff_matrix / tau_plus) * (time_diff_matrix > 0)
        LTD_matrix = -A_minus * np.exp(time_diff_matrix / tau_minus) * (time_diff_matrix < 0)
        # Combine LTP and LTD matrices
        STDP_matrix = LTP_matrix + LTD_matrix
        for b in range(batch):
            # Compute the pairwise time differences (t_post - t_pre)
            delta_w += post_spikes_matrix[b] @ STDP_matrix @ pre_spikes_matrix[b].T
            # update_weights
        # Update the weights
        updated_weights = np.clip(weights + sf*delta_w, -1, 1)
        # plt.imshow(delta_w)
        # updated_weights = weights, -1, 1)
        return updated_weights

    def update_weights_normal(self, pre_spikes_matrix, post_spikes_matrix, weights, lr, sf):

        batch, N_pre, T = pre_spikes_matrix.shape
        _, N_post, _ = post_spikes_matrix.shape

        delta_w = np.zeros_like(weights)

        for b in range(batch):
            # Compute the pairwise time differences (t_post - t_pre)
            delta_w += post_spikes_matrix[b] @ pre_spikes_matrix[b].T
        delta_w = (delta_w-delta_w.mean())/(1e-5+delta_w.std())
        updated_weights = np.clip(weights + sf*lr*delta_w, -1, 1)
        # plt.imshow(delta_w)
        # updated_weights = weights, -1, 1)
        return updated_weights

class SOM(nn.Module):
    def __init__(
            self,
            channels=224,
            initial_mean=0,
            initial_std=np.sqrt(2. / 224)
    ):
        super(SOM, self).__init__()
        self.w = np.random.normal(initial_mean, initial_std, (channels, channels))
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        self.name = 'som'
        self.channels = channels

    def forward(self, inp,  **kwargs):
        inp = inp.detach().numpy()
        batch, h, w = inp.shape
        out = np.zeros((batch, 1, self.channels, self.channels))
        mem = self.sleaky.init_leaky()
        for t in range(inp.shape[-1]):
            x = inp[:, :, t]
            # I = np.matmul(self.w, np.expand_dims(x, axis=-1))
            # spk, mem = self.sleaky(torch.tensor(I, dtype=torch.float32), mem)
            error = self.w - x[:, np.newaxis, :]
            eudistance = np.sum(error ** 2, axis=-1)
            min_ind = np.argmin(eudistance, axis=-1)
            # min_value = np.min(eudistance, axis=0)
            out[np.arange(batch), 0, min_ind, t] = 1

        if 'lr' in kwargs:
            self.w = self.update_weights(inp, self.w, kwargs['lr'], sf=1e-3)
        # ind=0
        # plt.figure()
        # plt.imshow(inp[ind, :, :], aspect='auto')
        # plt.figure()
        # plt.imshow(out[ind, 0, :, :], aspect='auto')
        return torch.tensor(out, dtype=torch.float32)

    def find_bmu(self, weights, input_spikes):
        """Find the BMU for each time step in the input spikes."""
        # Calculate the distance for all time steps in one go
        distances = np.sum((weights[:, :] - input_spikes.T[:, np.newaxis, :]) ** 2, axis=2)

        # Find the BMU indices for each time step
        bmu_indices = np.argmin(distances, axis=-1)
        return bmu_indices


    def update_weights(self, input_spikes, weights, lr, sf):

        batch, _, _ = input_spikes.shape

        delta_w = np.zeros_like(weights)
        for b in range(batch):
            bmu_indices = self.find_bmu(weights, input_spikes[b])
            # update_matrix = np.zeros_like(weights)
            np.add.at(delta_w, bmu_indices, lr * sf * (input_spikes[b].T - weights[bmu_indices]))
        updated_weights = np.clip(weights + delta_w, -1, 1)

        return updated_weights


class CT(nn.Module):
    def __init__(
            self,
            channels=224,
            initial_mean=1,
            initial_std=np.sqrt(2. / 224)
    ):
        super(CT, self).__init__()
        self.t = nn.Parameter(torch.normal(mean=3, std=initial_std, size=(1,)))
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        # self.msleaky = my_sleaky()
        self.name = 'ct'
        self.channels = channels

    def forward(self, inp,  **kwargs):
        inp = inp.unsqueeze(1)
        roll_amount = torch.randint(0, 224, (1,)).item()
        # # Perform the roll operation
        inp = torch.roll(inp, shifts=roll_amount, dims=3)
        inp = inp.squeeze(1)

        batch, h, w = inp.size()

        out = torch.zeros((batch, 1, self.channels, self.channels))
        mem = self.sleaky.init_leaky()
        # w =
        mem_rec = []
        T = self.t.clamp(1, 5)
        self.sleaky.threshold = T
        # spk = torch.zeros((batch, h))
        # mem = torch.zeros((batch, h))
        for t in range(inp.shape[-1]):
            x = inp[:, :, t]
            spk, mem = self.sleaky(x, mem)
            # spk, mem = self.msleaky(spk, mem, x, T)

            mem_rec.append(mem)
            out[:, 0, :, t] = spk.squeeze(-1)
        # mem_map = torch.stack(mem_rec,dim=-1)
        # plt.figure()
        # plt.imshow(mem_map[0,:,:].detach().numpy())
        # plt.figure()
        # plt.imshow(out[0,0,:,:].detach().numpy())

        return out
        # plt.figure()
        # plt.imshow(inp[0,:,:],aspect='auto')
        # plt.figure()
        # plt.imshow(out[0,0,:,:].detach().numpy(),aspect='auto')

class AT(nn.Module):
    def __init__(
            self,
            channels=224,
            initial_mean=0,
            initial_std=np.sqrt(2. / 224)
    ):
        super(AT, self).__init__()
        self.w = nn.Parameter(torch.normal(mean=0, std=initial_std, size=(channels,)))
        # mid = 1024
        # self.linear1 = nn.Linear(1, mid)
        # self.linear2 = nn.Linear(mid, 1)
        self.sleaky = snn.Leaky(beta=0.95, threshold=1)
        self.name = 'at'
        self.channels = channels

    def forward(self, inp,  **kwargs):
        inp = inp.unsqueeze(1)
        roll_amount = torch.randint(0, 224, (1,)).item()
        # # Perform the roll operation
        inp = torch.roll(inp, shifts=roll_amount, dims=3)
        inp = inp.squeeze(1)

        batch, h, w = inp.size()

        out = torch.zeros((batch, 1, self.channels, self.channels))
        mem = self.sleaky.init_leaky()
        t_rec = []
        T=0
        for t in range(inp.shape[-1]):
            x = inp[:, :, t]
            # w = self.w.clamp(min=0)
            # temp = self.linear2(torch.relu(self.linear1(torch.sum(x, dim=-1).unsqueeze(-1))))
            temp = 3+torch.tanh(torch.matmul(self.w, x.unsqueeze(-1)).squeeze(-1))*2
            T = temp.clip(1, 5).unsqueeze(-1).repeat((1, h))
            self.sleaky.threshold = T
            t_rec.append(T)
            # I = x-T+1
            spk, mem = self.sleaky(x, mem)
            out[:, 0, :, t] = spk.squeeze(-1)
        if 'w_rec' in kwargs:
            return out, t_rec
        else:
            return out
        # t_map = torch.stack(t_rec, dim=-1)
        # plt.figure()
        # plt.imshow(t_map[0,:,:].detach().numpy())
        # plt.figure()
        # plt.imshow(out[0,0,:,:].detach().numpy())

        # ind=4
        # plt.figure()
        # plt.imshow(inp[ind,:,:],aspect='auto')
        # plt.figure()
        # plt.imshow(out[ind,0,:,:].detach().numpy(),aspect='auto')


class ShuffleNetEnsemble(nn.Module):
    def __init__(self, num_models, num_classes=10, input_channels=1):
        super(ShuffleNetEnsemble, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes

        # Create n ShuffleNet models
        self.shufflenets = nn.ModuleList([
            self.create_shufflenet(input_channels, num_classes) for _ in range(num_models)
        ])

        # Linear layer to map 10*n outputs to 10 classes
        # self.fc = nn.Linear(num_models * num_classes, num_classes)

    def create_shufflenet(self, input_channels, num_classes):
        shufflenet = models.shufflenet_v2_x1_0(weights=True)
        shufflenet.conv1[0] = nn.Conv2d(input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        shufflenet.fc = nn.Linear(shufflenet.fc.in_features, num_classes)

        return shufflenet

    def forward(self, x):
        # Collect outputs from all ShuffleNet models
        outputs = [model(x) for model in self.shufflenets]

        # Concatenate outputs along the feature dimension
        sum_outputs = torch.sum(torch.stack(outputs, dim=1), dim=1)  # Shape: [batch_size, 10 * num_models]

        # Pass through the final linear layer
        # final_output = self.fc(sum_outputs)  # Shape: [batch_size, num_classes]
        return sum_outputs

class ShuffleNet28ensemble(nn.Module):
    def __init__(self, models, num_classes=10, input_channels=1):
        super(ShuffleNet28ensemble, self).__init__()
        self.num_classes = num_classes
        key = next(iter(models[0]))
        self.bn = copy.deepcopy(models[0])[key]
        self.bn.load_state_dict(torch.load(f'bestmodel/shufflenet_bn_stack_esc10-bn_coc_out.pth'), strict=False)
        self.b2 = copy.deepcopy(models[1])[key]
        self.b2.load_state_dict(torch.load(f'bestmodel/shufflenet_b2_stack_esc10-b2_coc_out.pth'), strict=False)
        self.n8 = copy.deepcopy(models[2])[key]
        self.n8.load_state_dict(torch.load(f'bestmodel/shufflenet_n8_stack_esc10-n8_coc_out.pth'), strict=False)


    def forward(self, x):
        # Collect outputs from all ShuffleNet models
        output = torch.zeros(x.size(0), self.num_classes)
        output_bn = self.bn(x)
        for i in range(x.size(0)):
            if output_bn[i, 0] > output_bn[i, 1]:
                output_b = self.b2(x[i].unsqueeze(0))
                output[i,6] = output_b[0,0]
                output[i,8] = output_b[0,1]
            else:
                output_n = self.n8(x[i].unsqueeze(0))
                output[i,0] = output_n[0,0]
                output[i,1] = output_n[0,1]
                output[i,2] = output_n[0,2]
                output[i,3] = output_n[0,3]
                output[i,4] = output_n[0,4]
                output[i,5] = output_n[0,5]
                output[i,7] = output_n[0,6]
                output[i,9] = output_n[0,7]

        return output


class ShuffleNetbncp(nn.Module):
    def __init__(self, models, num_classes=10, input_channels=1):
        super(ShuffleNetbncp, self).__init__()
        self.num_classes = num_classes
        key = next(iter(models[0]))
        self.bn = copy.deepcopy(models[0])[key]
        self.bn.load_state_dict(torch.load(f'bestmodel/shufflenet_bn_stack_esc10-bn_coc_out.pth'), strict=False)
        key = next(iter(models[1]))
        self.b2 = copy.deepcopy(models[1])[key]
        self.b2.load_state_dict(torch.load(f'bestmodel/shufflenet_b2_stack_esc10-b2_coc_out.pth'), strict=False)
        key = next(iter(models[2]))
        self.ncp = copy.deepcopy(models[2])[key]
        self.ncp.load_state_dict(torch.load(f'bestmodel/shufflenet_ncp_stack_esc10-ncp_coc_out.pth'), strict=False)
        key = next(iter(models[3]))
        self.nc3 = copy.deepcopy(models[3])[key]
        self.nc3.load_state_dict(torch.load(f'bestmodel/shufflenet_nc3_stack_esc10-nc3_coc_out.pth'), strict=False)
        key = next(iter(models[4]))
        self.np5 = copy.deepcopy(models[4])[key]
        self.np5.load_state_dict(torch.load(f'bestmodel/shufflenet_np5_stack_esc10-np5_coc_out.pth'), strict=False)

    def forward(self, x):
        # Collect outputs from all ShuffleNet models
        output = torch.zeros(x.size(0), self.num_classes)
        output_bn = self.bn(x)
        for i in range(x.size(0)):
            if output_bn[i, 0] > output_bn[i, 1]:
                output_b = self.b2(x[i].unsqueeze(0))
                output[i, 6] = output_b[0, 0]
                output[i, 8] = output_b[0, 1]
            else:
                output_ncp = self.ncp(x[i].unsqueeze(0))
                if output_ncp[0,0] > output_ncp[0,1]:
                    output_nc3 = self.nc3(x[i].unsqueeze(0))
                    output[i, 0] = output_nc3[0, 0]
                    output[i, 5] = output_nc3[0, 1]
                    output[i, 7] = output_nc3[0, 2]
                else:
                    output_np5 = self.np5(x[i].unsqueeze(0))
                    output[i, 1] = output_np5[0, 0]
                    output[i, 2] = output_np5[0, 1]
                    output[i, 3] = output_np5[0, 2]
                    output[i, 4] = output_np5[0, 3]
                    output[i, 9] = output_np5[0, 4]

        return output

class ShuffleNetHierESC10(nn.Module):
    def __init__(self, models, num_classes=10, input_channels=1):
        super(ShuffleNetHierESC10, self).__init__()
        self.num_classes = num_classes
        key = next(iter(models[0]))
        self.bn = copy.deepcopy(models[0])[key]
        self.bn.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_esc10-A01.pth'), strict=False)
        key = next(iter(models[1]))
        self.b2 = copy.deepcopy(models[1])[key]
        self.b2.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_esc10-A0B2.pth'), strict=False)
        key = next(iter(models[2]))
        self.ncp = copy.deepcopy(models[2])[key]
        self.ncp.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_esc10-A1B01.pth'), strict=False)
        key = next(iter(models[3]))
        self.nc3 = copy.deepcopy(models[3])[key]
        self.nc3.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_esc10-A1B1C3.pth'), strict=False)
        key = next(iter(models[4]))
        self.np5 = copy.deepcopy(models[4])[key]
        self.np5.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_esc10-A1B0C5.pth'), strict=False)

    def forward(self, x):
        # Collect outputs from all ShuffleNet models
        output = torch.zeros(x.size(0), self.num_classes)
        output_bn = self.bn(x)
        for i in range(x.size(0)):
            if output_bn[i, 0] > output_bn[i, 1]:
                output_b = self.b2(x[i].unsqueeze(0))
                output[i, 6] = output_b[0, 0]
                output[i, 8] = output_b[0, 1]
            else:
                output_ncp = self.ncp(x[i].unsqueeze(0))
                if output_ncp[0, 1] > output_ncp[0, 0]:
                    output_nc3 = self.nc3(x[i].unsqueeze(0))
                    output[i, 0] = output_nc3[0, 0]
                    output[i, 2] = output_nc3[0, 1]
                    output[i, 5] = output_nc3[0, 2]
                else:
                    output_np5 = self.np5(x[i].unsqueeze(0))
                    output[i, 1] = output_np5[0, 0]
                    output[i, 3] = output_np5[0, 1]
                    output[i, 4] = output_np5[0, 2]
                    output[i, 7] = output_np5[0, 3]
                    output[i, 9] = output_np5[0, 4]

        return output


class ShuffleNetHierUS8K(nn.Module):
    def __init__(self, models, num_classes=10, input_channels=1):
        super(ShuffleNetHierUS8K, self).__init__()
        # models = [get_models(dataset='us8k-A01'), get_models(dataset='us8k-A1B0C2'), get_models(dataset='us8k-A0B3'),
        #          get_models(dataset='us8k-A1B01'), get_models(dataset='us8k-A1B1C5')]

        self.num_classes = num_classes
        key = next(iter(models[0]))
        self.A01 = copy.deepcopy(models[0])[key]
        self.A01.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_us8k-A01.pth'), strict=False)
        key = next(iter(models[1]))
        self.A1B0C2 = copy.deepcopy(models[1])[key]
        self.A1B0C2.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_us8k-A1B0C2.pth'), strict=False)
        key = next(iter(models[2]))
        self.A0B3 = copy.deepcopy(models[2])[key]
        self.A0B3.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_us8k-A0B3.pth'), strict=False)
        key = next(iter(models[3]))
        self.A1B01 = copy.deepcopy(models[3])[key]
        self.A1B01.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_us8k-A1B01.pth'), strict=False)
        key = next(iter(models[4]))
        self.A1B1C5 = copy.deepcopy(models[4])[key]
        self.A1B1C5.load_state_dict(torch.load(f'bestmodel/shufflenet_stack_us8k-A1B1C5.pth'), strict=False)

    def forward(self, x):
        # Collect outputs from all ShuffleNet models
        output = torch.zeros(x.size(0), self.num_classes)
        output_bn = self.A01(x)
        for i in range(x.size(0)):
            if output_bn[i, 0] > output_bn[i, 1]:
                output_b = self.A0B3(x[i].unsqueeze(0))
                output[i, 1] = output_b[0, 0]
                output[i, 3] = output_b[0, 1]
                output[i, 6] = output_b[0, 1]
            else:
                output_ncp = self.A1B01(x[i].unsqueeze(0))
                if output_ncp[0, 0] > output_ncp[0, 1]:
                    output_nc3 = self.A1B0C2(x[i].unsqueeze(0))
                    output[i, 4] = output_nc3[0, 0]
                    output[i, 7] = output_nc3[0, 1]
                else:
                    output_np5 = self.A1B1C5(x[i].unsqueeze(0))
                    output[i, 0] = output_np5[0, 0]
                    output[i, 2] = output_np5[0, 1]
                    output[i, 5] = output_np5[0, 2]
                    output[i, 8] = output_np5[0, 3]
                    output[i, 9] = output_np5[0, 4]

        return output


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.shape
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x


class ShuffleV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if stride == 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels,
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels,
                          bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        self.shuffle = ChannelShuffle(2)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x2 = self.branch2(x2)
            out = torch.cat((x1, x2), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return self.shuffle(out)


class ShuffleBasic(nn.Module):
    def __init__(self, num_classes=1000, model_size='1.0x'):
        super().__init__()
        cfgs = {
            '0.1x': (24, [24, 48, 96, 1024]),
            '0.5x': (24, [48, 96, 192, 1024]),
            '1.0x': (24, [116, 232, 464, 1024]),
            '1.5x': (24, [176, 352, 704, 1024]),
            '2.0x': (24, [244, 488, 976, 2048])
        }
        assert model_size in cfgs, "Invalid model size"
        # out_channels = cfgs[model_size]
        in_channels, out_channels = cfgs[model_size]

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage2 = self._make_stage(in_channels, out_channels[0], 4)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 8)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 4)

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(out_channels[2], out_channels[3], kernel_size=1, bias=False),
        #     nn.BatchNorm2d(out_channels[3]),
        #     nn.ReLU(inplace=True)
        # )
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )

        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2, num_classes, bias=False)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [ShuffleV2Block(in_channels, out_channels, stride=2)]
        for _ in range(num_blocks - 1):
            layers.append(ShuffleV2Block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        # # Perform the roll operation
        x = torch.roll(x, shifts=torch.randint(0, 224, (1,)).item(), dims=3)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x)
        x = torch.flatten(x, 1)
        if kwargs:
            return x, self.fc(x)
        else:
            return self.fc(x)

        # a = x.detach().numpy()
        # plt.scatter(x= a[:,0], y =a[:, 1])
        # plt.plot([0, self.fc.weight.detach().numpy()[0][0]],
        #          [0, self.fc.weight.detach().numpy()[0][1]], marker='o', linestyle='-', color='b', label='(3,4)')
        # plt.plot([0, self.fc.weight.detach().numpy()[1][0]],
        #          [0, self.fc.weight.detach().numpy()[1][1]], marker='o',
        #          linestyle='-', color='b', label='(3,4)')
# Define the CNN architecture







if __name__ == "__main__":
    a=10
    import torch
    a = 10*torch.rand(10)
    b= torch.clip(a, 2, 5)
    c = torch.clamp(a, 2, 5)




