import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import DropPath, trunc_normal_

# ---------- 基础模块（与官方 IQFormer.py 完全一致）----------
def stemIQ(in_chs, out_chs):
    """I/Q 信号嵌入层，输出 (B, out_chs//2, T)"""
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs//2, kernel_size=5, stride=1, padding=2, groups=in_chs),
        nn.BatchNorm1d(out_chs//2),
    )

def stemSTFT(f, in_chs, out_chs):
    """
    STFT 时频图嵌入层，输出 (B, out_chs//2, 1, T)
    f: 频率维度大小，卷积核 (f, 1) 直接压缩频率维到 1
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs//2, kernel_size=(f, 1), stride=1, groups=in_chs),
        nn.BatchNorm2d(out_chs//2),
        nn.ReLU()
    )

class Embedding(nn.Module):
    """1D 卷积块下采样"""
    def __init__(self, patch_size=3, stride=1, padding=1, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm1d):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class ConvEncoder_IQ(nn.Module):
    """卷积编码器（深度可分离卷积 + 残差）"""
    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x

class FCN(nn.Module):
    """全连接卷积网络（1x1 卷积）"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EfficientAdditiveAttnetion(nn.Module):
    """高效加性注意力模块"""
    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        query_weight = query @ self.w_g  # BxNx1
        A = query_weight * self.scale_factor
        A = F.normalize(A, dim=1)
        G = torch.sum(A * query, dim=1)  # BxD
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])
        out = self.Proj(G * key) + query
        out = self.final(out)
        return out

class LocalRepresentation(nn.Module):
    """局部表示模块（深度可分离卷积 + 残差）"""
    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x

class Fusion(nn.Module):
    """
    多模态融合模块（对齐官方逻辑）：
    输入：concat(x, stft) 通道数为 input_channel
    输出：input_channel * 2 通道
    """
    def __init__(self, input_channel, drop):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, input_channel * 2, 1),
            nn.BatchNorm1d(input_channel * 2),
            nn.GELU(),
            nn.Conv1d(input_channel * 2, input_channel * 2, 1),  # 输出为 input_channel*2
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, stft):
        concat = torch.cat((x, stft), dim=1)
        fusion = self.conv(concat)
        return self.drop(fusion)

class IQFormer_Encoder(nn.Module):
    """IQFormer 编码器块"""
    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.local_representation = LocalRepresentation(dim=dim, kernel_size=3, drop_path=0., use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = FCN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.attn(x.permute(0,2,1)).permute(0,2,1))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(self.attn(x.permute(0,2,1)).permute(0,2,1))
            x = x + self.drop_path(self.linear(x))
        return x

def Stage(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, drop_rate=0., drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if layers[index] - block_idx <= vit_num:
            blocks.append(IQFormer_Encoder(dim, mlp_ratio=mlp_ratio, act_layer=act_layer,
                                            drop_path=block_dpr, use_layer_scale=use_layer_scale,
                                            layer_scale_init_value=layer_scale_init_value))
        else:
            blocks.append(ConvEncoder_IQ(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))
    return nn.Sequential(*blocks)

class IQFormer(nn.Module):
    def __init__(self,
                 num_classes=11,
                 signal_length=128,
                 layers=[1, 2, 1],
                 embed_dims=[80, 80, 80],   # 保持官方 80 维
                 mlp_ratios=4,
                 down_patch_size=5,
                 down_stride=3,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 vit_num=1):
        super().__init__()
        self.signal_length = signal_length

        # 固定 STFT 参数，使频率维度为 32（对齐官方）
        self.n_fft = 62          # n_fft = 2*freq_dim - 2, freq_dim=32
        self.win_length = 61
        self.hop_length = 1
        self.window = 'blackman'
        self.freq_dim = self.n_fft // 2 + 1  # 32

        # 归一化层
        self.BN = nn.BatchNorm1d(2)
        self.BN_stft = nn.BatchNorm2d(1)

        # 嵌入层：输出通道数 = embed_dims[0]//8
        self.patch_embedIQ = stemIQ(2, embed_dims[0] // 4)                   # (B, C1, T)  C1=embed_dims[0]//8
        self.patch_embedSTFT = stemSTFT(self.freq_dim, 1, embed_dims[0] // 4) # (B, C1, 1, T_out)

        # 融合模块：输入 concat 通道数为 2*C1 = embed_dims[0]//4，输出 embed_dims[0]//2
        self.fusion = Fusion(embed_dims[0] // 4, drop_rate)

        # Bi-LSTM：输入 embed_dims[0]//2，隐藏层大小 embed_dims[0]//2，双向后输出 embed_dims[0]
        self.patch_LSTM = nn.LSTM(input_size=embed_dims[0] // 2,
                                  hidden_size=embed_dims[0] // 2,
                                  bidirectional=True,
                                  batch_first=True,
                                  num_layers=1,
                                  dropout=drop_rate)

        # 多阶段网络
        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers,
                          mlp_ratio=mlp_ratios,
                          act_layer=nn.GELU,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    Embedding(patch_size=down_patch_size, stride=down_stride, padding=down_pad,
                              in_chans=embed_dims[i], embed_dim=embed_dims[i + 1])
                )
        self.network = nn.ModuleList(network)

        # 分类头
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)
        self.globalavgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x):
        # x: (B, 2, T)
        x = self.BN(x)

        # 计算 STFT 幅度谱（n_fft=62，频率维度=32，窗口=61，hop_length=1）
        complex_signal = torch.complex(x[:, 0, :], x[:, 1, :])
        window = torch.blackman_window(self.win_length, periodic=True).to(x.device)
        stft_complex = torch.stft(complex_signal,
                                  n_fft=self.n_fft,
                                  hop_length=self.hop_length,
                                  win_length=self.win_length,
                                  window=window,
                                  center=True,
                                  return_complex=True)
        magnitude = torch.abs(stft_complex).unsqueeze(1)  # (B, 1, freq_dim, T_out)
        magnitude = self.BN_stft(magnitude)

        # 嵌入
        iq_tokens = self.patch_embedIQ(x)                     # (B, C1, T)
        stft_tokens = self.patch_embedSTFT(magnitude)  # (B, C, 1, T_out)
        # 强制压缩频率维度到 1
        stft_tokens = F.adaptive_avg_pool2d(stft_tokens, (1, stft_tokens.size(-1)))
        stft_tokens = stft_tokens.squeeze(2)              # (B, C1, T_out)

        # 对齐时间维度（若不等长）
        if stft_tokens.size(-1) != iq_tokens.size(-1):
            stft_tokens = F.adaptive_avg_pool1d(stft_tokens, iq_tokens.size(-1))

        # 融合 → 输出 (B, embed_dims[0]//2, T)
        fused = self.fusion(iq_tokens, stft_tokens)

        # LSTM
        fused = fused.permute(0, 2, 1)                        # (B, T, C_lstm)
        lstm_out, _ = self.patch_LSTM(fused)                   # (B, T, 2*C_lstm) = (B, T, embed_dims[0])
        lstm_out = lstm_out.permute(0, 2, 1)                   # (B, embed_dims[0], T)

        # 多阶段编码
        tokens = self.forward_tokens(lstm_out)                 # (B, embed_dims[-1], T)
        tokens = self.norm(tokens)
        pooled = self.globalavgpool(tokens)                    # (B, embed_dims[-1])
        logits = self.head(pooled)
        return logits