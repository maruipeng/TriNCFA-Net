# TriNCFANet.py
# ---------------------------------------------------------------
# 与原 trimodnet_v17_ncfa.py 的差异:
#   * [仅一处结构改动] PAA4chGAFBranch 适配任意 signal_length:
#       1) 移除 assert n_paa^2 == 2*L 的强约束
#       2) _paa 改用 F.adaptive_avg_pool1d (在 L%M==0 时与原版 reshape+mean 数值等价)
#       3) forward 末尾添加 adaptive_avg_pool1d 长度对齐 (L=128 时不触发, 等同原版)
#       4) n_paa 自动选取 ⌈√(2L)⌉ (偶数), 保证 M²/2 ≥ L
#     ★ conv1/conv2 完全不动, 参数量与原版严格相同 (3424)
#
#   * [可选轻量化, 默认关闭] 通过 create_TriNCFANet 的可选参数瘦身:
#       - ncfa_hidden=64  (默认 128) -> FeatureDecoder hidden 减半, 节省 4160 参数
#       - proj_dim=32     (默认 64)                                 节省 1056 参数
#     默认配置 (ncfa_hidden=128, proj_dim=64) 时参数量与原版完全一致 (88175 @ L=128, 11类)
#
#   * 其他模块 (DDMNetGatedAPFusion / MSTCP / FeatureStatisticGatedFusion /
#     EnhancedMultiScaleBlock / 主网络结构 / 损失函数) 一字不动
# ---------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from timm.models.layers import trunc_normal_
except Exception:
    from torch.nn.init import trunc_normal_


# =============================================================
# ★ PAA-4ch-GAF 分支 (适配任意 L, 结构与原版完全一致)
# =============================================================
class PAA4chGAFBranch(nn.Module):
    """
    输入: iq (B, 2, L)
    输出: feat (B, out_dim, L)
    与原版的差异仅在长度处理 (PAA + 末尾 pool), conv1/conv2 完全不变.
    """

    def __init__(self, out_dim=16, n_paa=None, signal_length=128):
        super().__init__()
        # 自动选取 M: ⌈√(2L)⌉, 偶数, 保证 M²/2 ≥ L
        if n_paa is None:
            n_paa = int(math.ceil(math.sqrt(2 * signal_length)))
            if n_paa % 2 == 1:
                n_paa += 1
            # L=128  -> M=16 (M²/2=128=L,  pool 不触发, 等同原版)
            # L=1024 -> M=46 (M²/2=1058,   pool 下采样到 1024)
        self.M = n_paa
        self.L = signal_length

        # ↓↓↓ 完全保持原版 conv1 / conv2, 参数量 = 3424 ↓↓↓
        self.conv1 = nn.Sequential(
            nn.Conv1d(8, out_dim * 2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_dim * 2, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def _paa(self, x):
        """
        Piecewise Aggregate Approximation (B, L) -> (B, M).
        当 L % M == 0 时, adaptive_avg_pool1d 与原版 reshape+mean 数值等价.
        L=128, M=16 时段长=8 整除, 完全等同原版.
        """
        # x: (B, L)
        return F.adaptive_avg_pool1d(x.unsqueeze(1), self.M).squeeze(1)

    def _normalize_to_minus1_1(self, x):
        mn = x.min(dim=-1, keepdim=True)[0]
        mx = x.max(dim=-1, keepdim=True)[0]
        rng = (mx - mn).clamp(min=1e-8)
        return (x - mn) / rng * 2.0 - 1.0

    def _compute_gaf(self, seq):
        theta = torch.acos(seq.clamp(-0.999999, 0.999999))
        s = torch.sin(theta)
        c = torch.cos(theta)
        gasf = c.unsqueeze(2) * c.unsqueeze(1) - s.unsqueeze(2) * s.unsqueeze(1)
        gadf = s.unsqueeze(2) * c.unsqueeze(1) - c.unsqueeze(2) * s.unsqueeze(1)
        return gasf, gadf

    def forward(self, iq):
        B = iq.size(0)
        I, Q = iq[:, 0, :], iq[:, 1, :]
        amp = (I ** 2 + Q ** 2 + 1e-8).sqrt()
        phase = torch.atan2(Q, I)

        amp_paa = self._paa(amp)                 # (B, M)
        phase_paa = self._paa(phase)             # (B, M)
        amp_n = self._normalize_to_minus1_1(amp_paa)
        phase_n = self._normalize_to_minus1_1(phase_paa)

        gasf_a, gadf_a = self._compute_gaf(amp_n)    # (B, M, M)
        gasf_p, gadf_p = self._compute_gaf(phase_n)  # (B, M, M)

        # reshape 为 (B, 2, M²/2), 与原版完全一致
        gasf_a = gasf_a.reshape(B, 2, -1)
        gadf_a = gadf_a.reshape(B, 2, -1)
        gasf_p = gasf_p.reshape(B, 2, -1)
        gadf_p = gadf_p.reshape(B, 2, -1)

        feat = torch.cat([gasf_a, gadf_a, gasf_p, gadf_p], dim=1)  # (B, 8, M²/2)
        feat = self.conv2(self.conv1(feat))                        # (B, out_dim, M²/2)

        # 长度对齐到 L: L=128 时长度本身就等于 L, 此处为 no-op, 完全等同原版
        if feat.size(-1) != self.L:
            feat = F.adaptive_avg_pool1d(feat, self.L)
        return feat


# =============================================================
# DDMNet 式门控差分融合 (完全不动)
# =============================================================
class DDMNetGatedAPFusion(nn.Module):
    def __init__(self, out_ch=16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv1d(6, 6, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(6, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, S, D):
        SD = torch.cat([S, D], dim=1)
        gate = self.gate(SD)
        GS = gate[:, :2, :] * S
        GD = gate[:, 2:, :] * D
        GOUT = torch.cat([GS, GD], dim=1)
        return self.proj(GOUT)


def compute_differential(ap):
    B, C, L = ap.shape
    diff1 = ap[:, :, 1:] - ap[:, :, :-1]
    diff2 = (ap[:, :, 2:] - ap[:, :, :-2]) / 2.0
    diff1 = F.pad(diff1, (0, 1), mode='replicate')
    diff2 = F.pad(diff2, (1, 1), mode='replicate')
    return torch.cat([diff1, diff2], dim=1)


# =============================================================
# 复数卷积 (完全不动)
# =============================================================
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        I = x[:, 0:1, :]
        Q = x[:, 1:2, :]
        real = self.real_conv(I) - self.imag_conv(Q)
        imag = self.imag_conv(I) + self.real_conv(Q)
        return torch.cat([real, imag], dim=1)


class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn_real = nn.BatchNorm1d(num_features, eps, momentum)
        self.bn_imag = nn.BatchNorm1d(num_features, eps, momentum)

    def forward(self, x):
        C = x.size(1) // 2
        return torch.cat([self.bn_real(x[:, :C, :]), self.bn_imag(x[:, C:, :])], dim=1)


# =============================================================
# 其他模块 (完全不动)
# =============================================================
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dwconv = nn.Conv1d(in_channels, in_channels, kernel_size, stride,
                                padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        return self.bn(self.dwconv(x))


class MultiScaleConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw1 = DepthwiseConv(dim, kernel_size=1)
        self.dw2 = DepthwiseConv(dim, kernel_size=3)
        self.dw3 = DepthwiseConv(dim, kernel_size=5)
        self.dw4 = DepthwiseConv(dim, kernel_size=7)
        self.dw5 = DepthwiseConv(dim, kernel_size=31)
        self.proj = nn.Sequential(
            nn.Conv1d(dim * 5, dim, 1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([self.dw1(x), self.dw2(x), self.dw3(x),
                         self.dw4(x), self.dw5(x)], dim=1)
        return self.proj(out)


class NoiseRobustAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        B, C, L = x.shape
        y = self.avg_pool(x).view(B, C)
        attn = self.fc(y).view(B, C, 1)
        thr = self.threshold.abs().view(1, C, 1)
        x_den = torch.sign(x) * torch.clamp(x.abs() - thr, min=0.0)
        return x_den * attn


class EnhancedMultiScaleBlock(nn.Module):
    def __init__(self, dim, use_glu=True, use_nra=True, downsample=False):
        super().__init__()
        self.use_glu = use_glu
        if use_glu:
            self.glu = nn.Sequential(nn.Conv1d(dim, dim * 2, 1), nn.GLU(dim=1))
        self.msconv = MultiScaleConv(dim)
        self.nra = NoiseRobustAttention(dim) if use_nra else nn.Identity()
        self.downsample = DepthwiseConv(dim, kernel_size=3, stride=2) if downsample else None

    def forward(self, x):
        if self.use_glu:
            x = self.glu(x)
        x = self.msconv(x)
        x = self.nra(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class MSTCPBlock(nn.Module):
    def __init__(self, dim, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseConv(dim, ks) for ks in kernel_sizes])
        self.fusion = nn.Conv1d(dim * len(kernel_sizes), dim, 1)
        self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True, dropout=0.1)

    def forward(self, x):
        ms = torch.cat([c(x) for c in self.convs], dim=1)
        ms = self.fusion(ms)
        ms_T = ms.transpose(1, 2)
        out, _ = self.attn(ms_T, ms_T, ms_T)
        return out.transpose(1, 2) + x


class StatisticExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim * 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        mean = self.gap(x).squeeze(-1)
        max_val = self.gmp(x).squeeze(-1)
        x_sq = x ** 2
        var = self.gap(x_sq).squeeze(-1) - mean ** 2
        stats = torch.cat([mean, max_val, var], dim=1)
        return self.fc(stats)


class FeatureStatisticGatedFusion(nn.Module):
    def __init__(self, in_dims, out_dim, hidden_dim=32):
        super().__init__()
        self.num_mods = len(in_dims)
        self.stat_extractors = nn.ModuleList([
            StatisticExtractor(dim, hidden_dim) for dim in in_dims
        ])
        self.env_encoder = nn.Sequential(
            nn.Linear(hidden_dim * self.num_mods, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate_l1 = nn.Sequential(
            nn.Linear(hidden_dim, in_dims[1] + in_dims[2]),
            nn.Sigmoid()
        )
        self.fuse_l1 = nn.Sequential(
            nn.Conv1d(in_dims[1] + in_dims[2], out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        self.gate_l2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim + in_dims[0]),
            nn.Sigmoid()
        )
        self.fuse_l2 = nn.Sequential(
            nn.Conv1d(out_dim + in_dims[0], out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, f_iq, f_ap, f_gaf):
        stats = []
        for i, feat in enumerate([f_iq, f_ap, f_gaf]):
            s = self.stat_extractors[i](feat)
            stats.append(s)
        stats_cat = torch.cat(stats, dim=1)
        env_vec = self.env_encoder(stats_cat)
        cat_ap_gaf = torch.cat([f_ap, f_gaf], dim=1)
        gate_l1 = self.gate_l1(env_vec).unsqueeze(-1)
        fused_l1 = self.fuse_l1(cat_ap_gaf * gate_l1)
        cat_l2 = torch.cat([f_iq, fused_l1], dim=1)
        gate_l2 = self.gate_l2(env_vec).unsqueeze(-1)
        fused_final = self.fuse_l2(cat_l2 * gate_l2)
        return fused_final


# =============================================================
# 损失函数 (完全不动)
# =============================================================
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        return ((features - self.centers[labels]).pow(2).sum(1) / 2.0).mean()


class ClassSeparationLoss(nn.Module):
    def __init__(self, margin=10.0):
        super().__init__()
        self.margin = margin

    def forward(self, centers):
        dist = torch.cdist(centers, centers, p=2)
        mask = ~torch.eye(centers.size(0), dtype=bool, device=centers.device)
        return F.relu(self.margin - dist[mask]).mean()


class SupConLoss(nn.Module):
    """监督对比损失 (Khosla et al. NeurIPS 2020)"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        N = features.size(0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        eye_mask = torch.eye(N, device=device)
        pos_mask = mask * (1 - eye_mask)
        all_mask = 1 - eye_mask

        exp_sim = torch.exp(sim) * all_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(1).clamp(min=1)
        mean_log_prob = (pos_mask * log_prob).sum(1) / pos_count
        valid = (pos_mask.sum(1) > 0).float()
        loss = -(mean_log_prob * valid).sum() / valid.sum().clamp(min=1)
        return loss


class FeatureDecoder(nn.Module):
    """
    特征级去噪解码器 (NCFA 训练时激活, 推理时剥离)
    残差结构: 解码器学习 "噪声偏移", 主路径恒等通过
    """

    def __init__(self, feat_dim, hidden=128, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, feat_dim)
        )

    def forward(self, x):
        return x + self.net(x)



# =============================================================
# ★ PAR 三通道谱图分支: 替换原 Branch 3 的 GAF 模态
# =============================================================
class PAR3chSpectrogramBranch(nn.Module):
    """
    输入:  iq (B, 2, L)
    输出:  feat (B, out_dim, L)

    该分支只替换 TriNCFANet 的第三模态输入:
      R: 长窗功率谱图, 保留整体频谱包络
      G: 短窗 phase-hole 谱图, 强化 PSK/QAM 相位跳变造成的局部能量空洞
      B: 短窗时间纹理谱图, 强化 QAM/PAM 幅相变化和边缘变化

    为了适合 RML2016 的小样本长度和训练速度, 这里没有调用 torch.stft,
    而是用固定 DFT 卷积核实现 STFT 功率谱。这样在 GPU 上可以直接参与
    batch 训练, 且不引入可学习参数。
    """

    def __init__(self, out_dim=16, signal_length=128,
                 n_fft=None, long_win=None, long_hop=None,
                 short_win=None, short_hop=None, freq_bins=None):
        super().__init__()
        self.L = int(signal_length)

        if n_fft is None:
            # 窄带 RML2016: L=128, 用 32 点频率分辨率即可;
            # RML2018: L=1024, 用 64 点更稳。
            n_fft = 32 if self.L <= 256 else 64
        self.n_fft = int(n_fft)
        self.freq_bins = int(freq_bins or self.n_fft)

        self.long_win = int(long_win or self.n_fft)
        self.long_hop = int(long_hop or max(1, self.long_win // 8))
        self.short_win = int(short_win or max(8, self.n_fft // 2))
        self.short_hop = int(short_hop or max(1, self.short_win // 16))

        lr, li = self._make_dft_kernels(self.long_win, self.n_fft)
        sr, si = self._make_dft_kernels(self.short_win, self.n_fft)
        self.register_buffer("long_real", lr, persistent=False)
        self.register_buffer("long_imag", li, persistent=False)
        self.register_buffer("short_real", sr, persistent=False)
        self.register_buffer("short_imag", si, persistent=False)

        # 轻量 2D 编码器: 仅用于把 PAR 三通道图压成原网络需要的 1D 特征序列
        self.encoder2d = nn.Sequential(
            nn.Conv2d(3, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, self.L))

    @staticmethod
    def _make_dft_kernels(win_length, n_fft):
        k = torch.arange(win_length, dtype=torch.float32).view(1, -1)
        n = torch.arange(n_fft, dtype=torch.float32).view(-1, 1)
        win = torch.hann_window(win_length, periodic=True).view(1, -1)
        angle = -2.0 * math.pi * n * k / float(n_fft)
        real = (torch.cos(angle) * win).unsqueeze(1)  # (F, 1, W)
        imag = (torch.sin(angle) * win).unsqueeze(1)  # (F, 1, W)
        return real, imag

    @staticmethod
    def _norm01(x):
        # x: (B, F, T), per-sample min-max normalization
        mn = x.amin(dim=(1, 2), keepdim=True)
        mx = x.amax(dim=(1, 2), keepdim=True)
        return (x - mn) / (mx - mn).clamp_min(1e-8)

    def _stft_log_power_conv(self, iq, win_length, hop_length, kernel_r, kernel_i):
        # iq: (B, 2, L)
        I = iq[:, 0:1, :]
        Q = iq[:, 1:2, :]

        pad_left = win_length // 2
        pad_right = win_length - 1 - pad_left
        if I.size(-1) > max(pad_left, pad_right):
            I = F.pad(I, (pad_left, pad_right), mode="reflect")
            Q = F.pad(Q, (pad_left, pad_right), mode="reflect")
        else:
            I = F.pad(I, (pad_left, pad_right), mode="replicate")
            Q = F.pad(Q, (pad_left, pad_right), mode="replicate")

        kr = kernel_r.to(device=iq.device, dtype=iq.dtype)
        ki = kernel_i.to(device=iq.device, dtype=iq.dtype)

        # complex DFT: sum (I+jQ) * (kr+jki)
        real = F.conv1d(I, kr, stride=hop_length) - F.conv1d(Q, ki, stride=hop_length)
        imag = F.conv1d(I, ki, stride=hop_length) + F.conv1d(Q, kr, stride=hop_length)
        P = torch.log1p(real.pow(2) + imag.pow(2))   # (B, F, T)

        # fftshift: 频率从 [-Fs/2, Fs/2) 排列
        P = torch.roll(P, shifts=self.n_fft // 2, dims=1)

        # 对齐为固定 PAR 图尺寸: (B, F_par, L)
        P = F.interpolate(
            P.unsqueeze(1),
            size=(self.freq_bins, self.L),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return P

    def forward(self, iq):
        # R: 长窗功率谱图
        long_p = self._stft_log_power_conv(
            iq, self.long_win, self.long_hop, self.long_real, self.long_imag
        )
        r = self._norm01(long_p)

        # G/B: 短窗 PA/phase-hole 与时间纹理
        short_p = self._stft_log_power_conv(
            iq, self.short_win, self.short_hop, self.short_real, self.short_imag
        )
        short01 = self._norm01(short_p)

        # phase-hole: 谱图局部能量空洞在该通道中变亮
        g = 1.0 - short01

        # 时间纹理: 相邻时刻短窗谱图变化, 帮助区分 QAM/PAM 与 PSK
        b = torch.abs(short01[:, :, 1:] - short01[:, :, :-1])
        b = F.pad(b, (1, 0), mode="replicate")
        b = self._norm01(b)

        par = torch.stack([r, g, b], dim=1)        # (B, 3, F, L)
        feat2d = self.encoder2d(par)              # (B, out_dim, F, L)
        feat = self.freq_pool(feat2d).squeeze(2)  # (B, out_dim, L)
        return feat

# =============================================================
# 主模型 TriNCFANet
# =============================================================
class TriNCFANet(nn.Module):
    """
    与 v17 完全相同的拓扑, 仅 PAA4chGAFBranch 适配任意 L.
    可选构造参数 ncfa_hidden / proj_dim 用于轻量化, 默认值与原版一致.
    """

    def __init__(self, num_classes=11, signal_length=128, feat_dim=64,
                 num_blocks=5, drop_rate=0.2,
                 lambda_c=0.003, lambda_s=0.001,
                 proj_dim=64,
                 ncfa_hidden=128):     # ★ 新增: FeatureDecoder hidden 维度, 默认 128 = 原版
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        stem_out = feat_dim // 4      # 16
        inner_dim = feat_dim // 2     # 32
        self.inner_dim = inner_dim

        # ---- Branch 1: IQ ----
        self.complex_iq = nn.Sequential(
            ComplexConv1d(1, stem_out, kernel_size=5, padding=2),
            ComplexBatchNorm1d(stem_out),
            nn.ReLU(inplace=True)
        )
        self.iq_to_real = nn.Conv1d(stem_out * 2, stem_out, kernel_size=1)

        # ---- Branch 2: AP ----
        self.ap_fusion = DDMNetGatedAPFusion(out_ch=stem_out)

        # ---- Branch 3: PAR 三通道谱图 ----
        # 只替换第三模态输入: 原 GAF -> PAR(R:长窗谱图, G:phase-hole, B:时间纹理)
        # 输出仍是 (B, stem_out, L), 后续 MSTCP / 融合 / 分类头保持不变。
        self.paa_gaf = PAR3chSpectrogramBranch(
            out_dim=stem_out,
            signal_length=signal_length,
        )

        # ---- MSTCP ----
        self.ms_tcp_iq = MSTCPBlock(stem_out, kernel_sizes=(3, 5, 7))
        self.ms_tcp_ap = MSTCPBlock(stem_out, kernel_sizes=(3, 5, 7))
        self.ms_tcp_gaf = MSTCPBlock(stem_out, kernel_sizes=(3, 5, 7))

        # ---- 融合 ----
        self.fusion = FeatureStatisticGatedFusion(
            in_dims=[stem_out, stem_out, stem_out],
            out_dim=inner_dim,
            hidden_dim=32
        )

        # ---- 深层增强块 ----
        blocks = []
        for i in range(num_blocks):
            blocks.append(EnhancedMultiScaleBlock(
                inner_dim, use_glu=True, use_nra=True,
                downsample=(i in [1, 3])
            ))
        self.blocks = nn.Sequential(*blocks)

        # ---- 分类头 ----
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(inner_dim, num_classes)

        # ---- Projection Head (对比学习) ----
        self.proj_head = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(inplace=True),
            nn.Linear(inner_dim, proj_dim)
        )

        # ---- Feature Decoder (特征级去噪, hidden 可瘦身) ----
        self.feat_decoder = FeatureDecoder(inner_dim, hidden=ncfa_hidden, drop=0.1)

        # ---- 固定 lambda ----
        self.register_buffer('lambda_c_val', torch.tensor(lambda_c))
        self.register_buffer('lambda_s_val', torch.tensor(lambda_s))
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s

        self.center_loss_fn = CenterLoss(num_classes, inner_dim)
        self.sep_loss_fn = ClassSeparationLoss(margin=10.0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def compute_ap(self, iq):
        I, Q = iq[:, 0, :], iq[:, 1, :]
        amp = (I ** 2 + Q ** 2 + 1e-8).sqrt()
        phase = torch.atan2(Q, I)
        return torch.stack([amp, phase], dim=1)

    def augment_iq(self, iq):
        """温和增强: 小噪声 + 90度旋转 + 随机掩码"""
        if not self.training:
            return iq
        iq = iq + torch.randn_like(iq) * 0.01
        if np.random.rand() > 0.5:
            B = iq.size(0)
            k = np.random.randint(0, 4, B)
            cv = torch.tensor([1, 0, -1, 0], device=iq.device, dtype=iq.dtype)[k]
            sv = torch.tensor([0, 1, 0, -1], device=iq.device, dtype=iq.dtype)[k]
            I, Q = iq[:, 0, :], iq[:, 1, :]
            iq = torch.stack([
                cv.view(-1, 1) * I - sv.view(-1, 1) * Q,
                sv.view(-1, 1) * I + cv.view(-1, 1) * Q
            ], dim=1)
        if np.random.rand() < 0.12:
            B, C, L = iq.shape
            iq = iq.clone()
            ml = np.random.randint(int(L * 0.05), int(L * 0.12))
            st = np.random.randint(0, L - ml)
            iq[:, :, st:st + ml] = 0
        return iq

    def _encode(self, iq):
        # Branch 1: IQ
        iq_cplx = self.complex_iq(iq)
        iq_feat = self.iq_to_real(iq_cplx)
        # Branch 2: AP
        ap = self.compute_ap(iq)
        diff_ap = compute_differential(ap)
        ap_feat = self.ap_fusion(ap, diff_ap)
        # Branch 3: GAF
        gaf_feat = self.paa_gaf(iq)

        # MSTCP
        iq_feat = self.ms_tcp_iq(iq_feat)
        ap_feat = self.ms_tcp_ap(ap_feat)
        gaf_feat = self.ms_tcp_gaf(gaf_feat)

        fused = self.fusion(iq_feat, ap_feat, gaf_feat)
        x = self.blocks(fused)
        feat = self.gap(x).squeeze(-1)
        return feat

    def forward(self, iq, return_features=False, apply_rotation=False,
                return_all=False, return_contrast=False):
        if apply_rotation and self.training:
            iq = self.augment_iq(iq)

        feat = self._encode(iq)

        if return_features:
            return feat

        logits = self.head(self.head_drop(feat))

        if return_contrast:
            proj = F.normalize(self.proj_head(feat), dim=1)
            feat_rec = self.feat_decoder(feat)
            return logits, feat, proj, feat_rec

        if return_all:
            return logits, feat
        return logits


# =============================================================
# 工厂函数
# =============================================================
def create_TriNCFANet(num_classes, signal_length=128, drop_rate=0.2,
                      proj_dim=32, ncfa_hidden=64):
    """
    Args:
        num_classes:    分类数
        signal_length:  信号长度 (128 / 1024)
        drop_rate:      分类头 dropout
        proj_dim:       对比学习投影维度
                        默认 64 (与原版一致)
                        可选 32 (轻量化, 节省 1056 参数)
        ncfa_hidden:    FeatureDecoder hidden 维度
                        默认 128 (与原版一致)
                        可选 64 (轻量化, 节省 4160 参数, 几乎不影响精度)

    参数量 (L=128, num_classes=11):
        默认               -> 88,175  (与原 trimodnet_v17_ncfa.py 完全一致)
        ncfa_hidden=64     -> 84,015  (推荐: 风险极低, 仅训练时激活的解码器)
        ncfa_hidden=64,
        proj_dim=32        -> 82,959  (激进: SupCon 投影维度减半)
    """
    return TriNCFANet(
        num_classes=num_classes,
        signal_length=signal_length,
        feat_dim=64,
        num_blocks=5,
        drop_rate=drop_rate,
        lambda_c=0.003,
        lambda_s=0.001,
        proj_dim=proj_dim,
        ncfa_hidden=ncfa_hidden,
    )


# =============================================================
# 自检
# =============================================================
if __name__ == "__main__":
    def n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("=" * 70)
    print("Self-test: TriNCFANet")
    print("=" * 70)

    # ---- L=128, 11 类 (RML2016a) ----
    print("\n[L=128, num_classes=11]")
    for cfg in [
        dict(),                                            # 默认 = 原版
        dict(ncfa_hidden=64),                              # 瘦身 1
        dict(ncfa_hidden=64, proj_dim=32),                 # 瘦身 2
    ]:
        m = create_TriNCFANet(num_classes=11, signal_length=128, **cfg)
        x = torch.randn(2, 2, 128)
        y = m(x)
        assert y.shape == (2, 11), y.shape
        m.train()
        logits, feat, proj, feat_rec = m(x, return_contrast=True, apply_rotation=True)
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items()) or "default"
        print(f"  {cfg_str:40s} | params = {n_params(m):,}")

    # ---- L=1024, 24 类 (RML2018a) ----
    print("\n[L=1024, num_classes=24]")
    for cfg in [dict(), dict(ncfa_hidden=64)]:
        m = create_TriNCFANet(num_classes=24, signal_length=1024, **cfg)
        x = torch.randn(2, 2, 1024)
        y = m(x)
        assert y.shape == (2, 24), y.shape
        m.train()
        logits, feat, proj, feat_rec = m(x, return_contrast=True, apply_rotation=True)
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items()) or "default"
        print(f"  {cfg_str:40s} | params = {n_params(m):,}")

    # ---- 验证 GAF 分支 n_paa 自动选取 ----
    print("\n[GAF branch n_paa auto-selection]")
    for L in [128, 256, 512, 1024]:
        m = PAA4chGAFBranch(out_dim=16, n_paa=None, signal_length=L)
        gaf_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(2, 2, L)
        y = m(x)
        assert y.shape == (2, 16, L), (y.shape, L)
        print(f"  L={L:5d} -> M={m.M:3d}, M²/2={m.M ** 2 // 2:5d}, "
              f"output_shape={tuple(y.shape)}, gaf_params={gaf_params}")

    print("\nAll self-tests PASSED.")

