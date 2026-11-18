import torch.nn as nn

from efficient_kan import KAN


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel attention — cheap, and useful here
    since discriminating between visually similar conjunct characters is
    mostly about which stroke regions matter."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return out + self.shortcut(x)


class CNNBackbone(nn.Module):
    def __init__(self, in_ch=1, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
        )
        self.stage1 = ResBlock(widths[0], widths[1], stride=2)
        self.stage2 = ResBlock(widths[1], widths[2], stride=2)
        self.stage3 = ResBlock(widths[2], widths[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[3]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return x


class OCRHead(nn.Module):
    """BatchNorm1d -> Dropout -> Tanh bounds backbone features into the KAN
    spline domain ([-1, 1]) and keeps that calibration stable across
    training — validated necessary in the prior CNN+KAN debugging pass:
    without a stable, bounded feature distribution the spline grid drifts
    out of range and outputs diverge to NaN within a few optimizer steps."""

    def __init__(
        self,
        in_dim,
        num_classes,
        head_type="kan2",
        kan_hidden=128,
        kan_grid=8,
        kan_order=3,
        dropout=0.3,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(dropout),
            nn.Tanh(),
        )
        if head_type == "linear":
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, kan_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(kan_hidden, num_classes),
            )
        elif head_type == "kan1":
            self.classifier = KAN([in_dim, num_classes], grid_size=kan_grid, spline_order=kan_order)
        elif head_type == "kan2":
            self.classifier = KAN(
                [in_dim, kan_hidden, num_classes], grid_size=kan_grid, spline_order=kan_order
            )
        else:
            raise ValueError(f"unknown head_type: {head_type}")

    def forward(self, x):
        x = self.pre(x)
        return self.classifier(x)


class MalayalamOCRNet(nn.Module):
    def __init__(
        self,
        num_classes,
        head_type="kan2",
        widths=(32, 64, 128, 256),
        kan_hidden=128,
        kan_grid=8,
        kan_order=3,
        dropout=0.3,
    ):
        super().__init__()
        self.backbone = CNNBackbone(widths=widths)
        self.head = OCRHead(
            self.backbone.out_dim,
            num_classes,
            head_type=head_type,
            kan_hidden=kan_hidden,
            kan_grid=kan_grid,
            kan_order=kan_order,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
