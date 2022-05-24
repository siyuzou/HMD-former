import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# dropout = dropout_layer
dropout = nn.Dropout


class Mlp(nn.Module):
    """ Borrowed from PoseFormer, further modified.
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DuplexMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            negative_slope=0.2
    ):
        assert negative_slope > 0

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.negative_slope = negative_slope

        self.fc1 = DuplexLinear(in_features, hidden_features)
        self.act = nn.LeakyReLU(negative_slope=negative_slope)
        self.fc2 = DuplexLinear(hidden_features, out_features)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, front, x):
        if front:
            return self.forward_front(x)
        else:
            return self.forward_back(x)

    def forward_front(self, x, detach=None):
        x = self.fc1.forward_front(x, detach=detach)
        x = self.act(x)
        x = self.fc2.forward_front(x, detach=detach)
        return x

    def forward_back(self, x, detach=None):
        x = self.fc2.forward_back(x, detach=detach)
        x = (x > 0).float() * x + (x <= 0).float() * x / self.negative_slope
        x = self.fc1.forward_back(x, detach=detach)
        return x


class DuplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs) -> None:
        super(DuplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 不知道为啥，Linear自带的init效果相对很好
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        # self.weight.data.normal_(mean=0.0, std=0.02)
        # if self.bias is not None:
        #     self.bias.data.zero_()

    def forward(self, front, x):
        if front:
            return self.forward_front(x)
        else:
            return self.forward_back(x)

    def forward_front(self, x, detach=False):
        if detach:
            return F.linear(x, self.weight.detach(), self.bias.detach())
        else:
            return F.linear(x, self.weight, self.bias)

    def forward_back(self, x, detach=False):
        w = self.weight.T  # (3, 512)
        b = self.bias  # (512, )

        if detach:
            w = w.detach()
            b = b.detach()

        w_inv_r = w.T @ torch.inverse(w @ w.T)  # (512, 3)
        return (x - b.unsqueeze(0).unsqueeze(0)) @ w_inv_r
