import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, input_shape):
        out = self.conv(torch.ones(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.noisylayers = [nn.NoisyLinear(conv_out_size, 512), \
                            nn.NoisyLinear(512, n_actions)]
        self.fc = nn.Sequential(
            self.noisylayers[0],
            nn.ReLU(),
            self.noisylayers[1],
        )

    def _get_conv_out(self, input_shape):
        out = self.conv(torch.ones(1, *input_shape))
        return int(np.prod(out.size()))


    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def noisylayer_snr(self):
        snr = [((l.weight**2).mean().sqrt()/((l.sigma_weight**2).mean().sqrt())).item() for l in self.noisylayers]
        return snr

class NoisyLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, sigma_init=0.017, bias=True, *args, **kwargs):
        super().__init__(in_channels, out_channels, bias=bias, *args, **kwargs):
        self.in_channels = in_channels
        sigma_mat = torch.full((out_channels, in_channels), sigma_init)
        self.sigma_weight = nn.Parameter(sigma_mat)
        z = torch.zeros(out_channels, in_channels)
        self.register_buffer("weight_noise", z) # this will be generated in fwd pass and scaled with sigma
        if bias:
            self.sigma_bias= nn.Parameter(torch.full((out_channels,), sigma_init))
            z = torch.zeros(out_channels)
            self.register_buffer("bias_noise", z) # this will be generated in fwd pass and scaled with sigma
        self.reset_parameters()

    def  reset_parameters(self):
        std = math.sqrt(3/self.in_channels)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, inp):
        # sample the noise for weight and bias
        self.weight_noise.normal_()
        bias = self.bias
        if bias:
            self.bias_noise.normal_()
            bias = bias + self.bias_noise.data * self.sigma_bias

        weight = self.weight + self.weight_noise.data * self.sigma_weight

        return F.linear(inp, w, bias)
