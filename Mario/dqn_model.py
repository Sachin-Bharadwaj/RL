import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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
        super(NoisyDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.noisylayers = [NoisyLinear(conv_out_size, 512), \
                            NoisyLinear(512, n_actions)]
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
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True, *args, **kwargs):
        super().__init__(in_features, out_features, bias=bias, *args, **kwargs)
        sigma_mat = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(sigma_mat)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("weight_noise", z) # this will be generated in fwd pass and scaled with sigma
        if bias:
            self.sigma_bias= nn.Parameter(torch.full((out_features,), sigma_init))
            z = torch.zeros(out_features)
            self.register_buffer("bias_noise", z) # this will be generated in fwd pass and scaled with sigma
        self.reset_parameters()

    def  reset_parameters(self):
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, inp):
        # sample the noise for weight and bias
        self.weight_noise.normal_()
        bias = self.bias
        if bias is not None:
            self.bias_noise.normal_()
            bias = bias + self.bias_noise.data * self.sigma_bias

        weight = self.weight + self.weight_noise.data * self.sigma_weight

        return F.linear(inp, weight, bias)

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        '''

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, input_shape):
        out = self.conv(torch.ones(1, *input_shape))
        return int(np.prod(out.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))
