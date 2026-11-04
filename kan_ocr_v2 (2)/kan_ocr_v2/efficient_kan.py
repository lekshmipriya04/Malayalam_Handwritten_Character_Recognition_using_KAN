import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """A lightweight, fully-vectorized B-spline KAN layer.

    Deliberately reimplemented instead of using the `pykan` package: pykan's
    MultKAN carries a symbolic-regression branch and full activation caching
    on every forward pass (built for interpretability workflows, not
    minibatch training), and its `update_grid_from_samples` crashes on CUDA
    for any near-degenerate batch (bare `except` around a `lstsq` call that
    never sets its return value). This layer has none of that: a fixed grid,
    one spline path plus one residual "base" path (as in the standard
    efficient-kan formulation), computed as ordinary batched tensor ops.
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            grid_points = self.grid.T[self.spline_order : -self.spline_order]
            self.spline_weight.data.copy_(self.scale_spline * self.curve2coeff(grid_points, noise))

    def b_splines(self, x):
        # x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def curve2coeff(self, x, y):
        # x: (n_points, in_features), y: (n_points, in_features, out_features)
        A = self.b_splines(x).transpose(0, 1)  # (in_features, n_points, coeff)
        B = y.transpose(0, 1)  # (in_features, n_points, out_features)
        solution = torch.linalg.lstsq(A, B).solution  # (in_features, coeff, out_features)
        return solution.permute(2, 0, 1).contiguous()  # (out_features, in_features, coeff)

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class KAN(nn.Module):
    """A small stack of KANLinear layers, e.g. width=[256, 128, num_classes]."""

    def __init__(self, width, grid_size=5, spline_order=3, grid_range=(-1, 1)):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                KANLinear(
                    width[i],
                    width[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range=grid_range,
                )
                for i in range(len(width) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
