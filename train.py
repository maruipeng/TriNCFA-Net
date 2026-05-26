import copy
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.manifold import TSNE
import argparse
import os
import sys
import json
import random
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from viz_metrics import (
    setup_plot_style,
    measure_inference_time,
    plot_snr_curve,
    plot_confusion,
    plot_tsne_per_snr,
    plot_tsne_all,
    plot_train_curves,
)

sys.path.append('./models')

# ============================================================================
# 全局调制类型表 (动态根据 args.dataset 赋值)
# ============================================================================
MODULATION_TYPES_2016A = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK",
                          "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

MODULATION_TYPES_2016B = ["8PSK", "AM-DSB", "BPSK", "CPFSK", "GFSK",
                          "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]

MODULATION_TYPES_2018A = ["OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK",
                          "16PSK", "32PSK", "16APSK", "32APSK", "64APSK", "128APSK",
                          "16QAM", "32QAM", "64QAM", "128QAM", "256QAM",
                          "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
                          "FM", "GMSK", "OQPSK"]

# 这些会在 main() 里根据 args.dataset 赋值
MODULATION_TYPES = MODULATION_TYPES_2016A
NUM_CLASSES = len(MODULATION_TYPES)


# ============================================================================
# 模型注册（按 sig_size 自适应选择网络版本）
# ============================================================================
def import_model(model_name, sig_size):
    """动态导入模型并返回工厂函数"""
    if model_name == 'AMCNet':
        from models.AMC_Net.amcnet import amcnet
        return lambda num_classes, sig_size: amcnet(num_classes, sig_size)
    elif model_name == 'MCLDNN':
        from models.MCLDNN.mcldnn import MCLDNN
        return lambda num_classes, sig_size: MCLDNN(num_classes, sig_size)
    elif model_name == 'PETCGDNN':
        from models.PETCGDNN.petcgdnn import PETCGDNN
        return lambda num_classes, sig_size: PETCGDNN(num_classes, sig_size)
    elif model_name == 'FEA_T':
        # 根据 sig_size 选择对应版本
        if sig_size == 128:
            from models.FEA_T.fea_t import fea_t_128
            return lambda num_classes, sig_size: fea_t_128(num_classes, sig_size)
        else:
            from models.FEA_T.fea_t import fea_t_1024
            return lambda num_classes, sig_size: fea_t_1024(num_classes, sig_size)
    elif model_name == 'MCFormer':
        from models.MCFormer.MCformer import MCformer
        return lambda num_classes, sig_size: MCformer(num_classes=num_classes, signal_length=sig_size)
    elif model_name == 'IQFormer':
        from models.IQFormer.iqformer import IQFormer
        return lambda num_classes, sig_size: IQFormer(num_classes=num_classes, signal_length=sig_size)
    elif model_name == 'SMT':
        if sig_size == 128:
            from models.SMT.smt import smt_128
            return lambda num_classes, sig_size: smt_128(num_classes, sig_size)
        else:
            from models.SMT.smt import smt_1024
            return lambda num_classes, sig_size: smt_1024(num_classes, sig_size)
    elif model_name == 'TriNCFA_Net':
        from models.TriNCFANet.TriNCFANet import create_TriNCFANet
        return lambda num_classes, sig_size: create_TriNCFANet(num_classes, signal_length=sig_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# ============================================================================
# 数据集加载
# ============================================================================
class RML2016aDataset(Dataset):
    """加载 RML2016.10a (pickle), 输出 (2,128) 信号、标签、SNR"""

    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')

        all_iq, all_labels, all_snr = [], [], []
        # 调制名称映射
        mod_mapping = {
            'BPSK': 'BPSK', 'QPSK': 'QPSK', '8PSK': '8PSK',
            'QAM16': 'QAM16', '16QAM': 'QAM16',
            'QAM64': 'QAM64', '64QAM': 'QAM64',
            'PAM4': 'PAM4', 'GFSK': 'GFSK', 'CPFSK': 'CPFSK',
            'WBFM': 'WBFM', 'AM-DSB': 'AM-DSB', 'AM-SSB': 'AM-SSB'
        }
        mod_to_idx = {mod: i for i, mod in enumerate(MODULATION_TYPES_2016A)}

        for (mod, snr), samples in raw_data.items():
            std_mod = mod_mapping.get(mod, mod)
            if std_mod not in mod_to_idx:
                continue
            label = mod_to_idx[std_mod]
            if samples.ndim == 3:
                if samples.shape[1] == 2 and samples.shape[2] == 128:
                    iq_data = samples
                elif samples.shape[1] == 128 and samples.shape[2] == 2:
                    iq_data = np.transpose(samples, (0, 2, 1))
                else:
                    raise ValueError(f"Unexpected shape: {samples.shape}")
            else:
                raise ValueError(f"Expected 3D, got {samples.ndim}D")
            for i in range(iq_data.shape[0]):
                sig = iq_data[i].astype(np.float32)
                power = np.mean(sig ** 2)
                if power > 0:
                    sig = sig / np.sqrt(power)
                all_iq.append(sig)
                all_labels.append(label)
                all_snr.append(snr)

        self.IQ = torch.tensor(np.array(all_iq), dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        self.snr = torch.tensor(all_snr, dtype=torch.float32)
        print(f"[RML2016a] Loaded {len(self.labels)} samples, "
              f"signal_length={self.IQ.shape[-1]}, classes={len(MODULATION_TYPES_2016A)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.IQ[idx], self.labels[idx], self.snr[idx]


class RML2016bDataset(Dataset):
    """加载 RML2016.10b (.dat 实际为 pickle 格式), 10类, 长度128"""

    def __init__(self, dat_path):
        with open(dat_path, 'rb') as f:
            raw_data = pickle.load(f, encoding='latin1')

        all_iq, all_labels, all_snr = [], [], []
        mod_mapping = {
            'BPSK': 'BPSK', 'QPSK': 'QPSK', '8PSK': '8PSK',
            'QAM16': 'QAM16', '16QAM': 'QAM16',
            'QAM64': 'QAM64', '64QAM': 'QAM64',
            'PAM4': 'PAM4', 'GFSK': 'GFSK', 'CPFSK': 'CPFSK',
            'WBFM': 'WBFM', 'AM-DSB': 'AM-DSB'
        }
        mod_to_idx = {mod: i for i, mod in enumerate(MODULATION_TYPES_2016B)}

        for (mod, snr), samples in raw_data.items():
            std_mod = mod_mapping.get(mod, mod)
            if std_mod not in mod_to_idx:
                continue
            label = mod_to_idx[std_mod]
            if samples.ndim == 3:
                if samples.shape[1] == 2 and samples.shape[2] == 128:
                    iq_data = samples
                elif samples.shape[1] == 128 and samples.shape[2] == 2:
                    iq_data = np.transpose(samples, (0, 2, 1))
                else:
                    raise ValueError(f"Unexpected shape: {samples.shape}")
            else:
                raise ValueError(f"Expected 3D, got {samples.ndim}D")
            for i in range(iq_data.shape[0]):
                sig = iq_data[i].astype(np.float32)
                power = np.mean(sig ** 2)
                if power > 0:
                    sig = sig / np.sqrt(power)
                all_iq.append(sig)
                all_labels.append(label)
                all_snr.append(snr)

        self.IQ = torch.tensor(np.array(all_iq), dtype=torch.float32)
        self.labels = torch.tensor(all_labels, dtype=torch.long)
        self.snr = torch.tensor(all_snr, dtype=torch.float32)
        print(f"[RML2016b] Loaded {len(self.labels)} samples, "
              f"signal_length={self.IQ.shape[-1]}, classes={len(MODULATION_TYPES_2016B)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.IQ[idx], self.labels[idx], self.snr[idx]


class RML2018aDataset(Dataset):
    """
    加载 RML2018.01a (HDF5), 24类, 长度1024
    支持每类每SNR子采样, 避免内存爆炸 (全量约 20+ GB)
    """

    def __init__(self, h5_path, sample_per_class_per_snr=512, snr_range=None):
        with h5py.File(h5_path, 'r') as f:
            Y = f['Y'][:]                       # (N, 24) one-hot
            Z = f['Z'][:]                       # (N, 1) SNR
            labels_all = np.argmax(Y, axis=1).astype(np.int64)
            snr_all = Z.flatten().astype(np.float32)

            unique_snrs = np.unique(snr_all)
            if snr_range is not None:
                unique_snrs = unique_snrs[
                    (unique_snrs >= snr_range[0]) & (unique_snrs <= snr_range[1])]

            # 分层子采样
            rng = np.random.RandomState(42)
            indices = []
            for cls in range(len(MODULATION_TYPES_2018A)):
                for snr_val in unique_snrs:
                    mask = (labels_all == cls) & (snr_all == snr_val)
                    idx = np.where(mask)[0]
                    if sample_per_class_per_snr is not None and len(idx) > sample_per_class_per_snr:
                        idx = rng.choice(idx, sample_per_class_per_snr, replace=False)
                    indices.extend(idx.tolist())
            indices = np.array(sorted(indices))
            print(f"[RML2018a] Selected indices: {len(indices)} from total {len(labels_all)}")

            # h5py fancy indexing 必须用排序后的 unique 索引
            X_sel = f['X'][indices, :, :]       # (N_sel, 1024, 2)

        # 转为 (N, 2, 1024) 并按功率归一化
        X = np.transpose(X_sel, (0, 2, 1)).astype(np.float32)
        power = (X ** 2).mean(axis=(1, 2), keepdims=True)
        power = np.where(power > 0, power, 1.0)
        X = X / np.sqrt(power)

        self.IQ = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(labels_all[indices], dtype=torch.long)
        self.snr = torch.tensor(snr_all[indices], dtype=torch.float32)
        print(f"[RML2018a] Loaded {len(self.labels)} samples, "
              f"signal_length={self.IQ.shape[-1]}, classes={len(MODULATION_TYPES_2018A)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.IQ[idx], self.labels[idx], self.snr[idx]


def get_dataset(args):
    """统一数据集入口"""
    global MODULATION_TYPES, NUM_CLASSES
    if args.dataset == "RML2016a":
        MODULATION_TYPES = MODULATION_TYPES_2016A
        NUM_CLASSES = len(MODULATION_TYPES)
        return RML2016aDataset(args.dataset_path)
    elif args.dataset == "RML2016b":
        MODULATION_TYPES = MODULATION_TYPES_2016B
        NUM_CLASSES = len(MODULATION_TYPES)
        return RML2016bDataset(args.dataset_path)
    elif args.dataset == "RML2018a":
        MODULATION_TYPES = MODULATION_TYPES_2018A
        NUM_CLASSES = len(MODULATION_TYPES)
        return RML2018aDataset(args.dataset_path,
                               sample_per_class_per_snr=args.samples_per_class_snr)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


# ============================================================================
# 损失/工具函数
# ============================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def snr_aware_noise_pair(iq, snr_range=(-10.0, 10.0)):
    """对 iq 加随机 SNR 高斯噪声, 返回噪声视图"""
    B = iq.size(0)
    sig_pow = iq.pow(2).mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)
    snr_db = torch.empty(B, 1, 1, device=iq.device).uniform_(snr_range[0], snr_range[1])
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr_lin
    noise = torch.randn_like(iq) * noise_pow.sqrt()
    return iq + noise


# ============================================================================
# TriNCFA 训练 / 验证 / 评估
# ============================================================================
def train_one_epoch_trimodnet_v17_ncfa(model, dataloader, optimizer,
                                       criterion_ce, supcon_loss_fn, device, epoch,
                                       total_epochs=60,
                                       lambda_sup_max=0.5, lambda_rec_max=0.1,
                                       warmup_ncfa=10, ramp_ncfa=20):
    model.train()
    total_loss = total_ce = total_center = total_sep = total_sup = total_rec = 0
    correct = total = 0

    if epoch <= warmup_ncfa:
        w_sup, w_rec = 0.0, 0.0
        noise_range = (-5.0, 10.0)
    elif epoch <= warmup_ncfa + ramp_ncfa:
        r = (epoch - warmup_ncfa) / float(ramp_ncfa)
        w_sup = lambda_sup_max * r
        w_rec = lambda_rec_max * r
        noise_range = (-10.0, 10.0)
    else:
        w_sup = lambda_sup_max
        w_rec = lambda_rec_max
        noise_range = (-15.0, 10.0)

    for x, y, _ in tqdm(dataloader,
                        desc=f'Train V17-NCFA [w_sup={w_sup:.2f} w_rec={w_rec:.2f}]',
                        leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if w_sup > 0 or w_rec > 0:
            if 15 < epoch < total_epochs - 5 and np.random.rand() < 0.3:
                lam = np.random.beta(0.1, 0.1)
                idx_mix = torch.randperm(x.size(0), device=device)
                x_mix = lam * x + (1 - lam) * x[idx_mix]
                y_a, y_b = y, y[idx_mix]
            else:
                x_mix, y_a, y_b, lam = x, y, None, 1.0

            x_noisy = snr_aware_noise_pair(x_mix, snr_range=noise_range)
            x_cat = torch.cat([x_mix, x_noisy], dim=0)
            y_cat = torch.cat([y_a, y_a], dim=0)

            logits, feat, proj, feat_rec = model(x_cat, apply_rotation=True, return_contrast=True)

            if lam < 1.0:
                y_b_cat = torch.cat([y_b, y_b], dim=0)
                loss_ce = lam * criterion_ce(logits, y_cat) + \
                          (1 - lam) * criterion_ce(logits, y_b_cat)
            else:
                loss_ce = criterion_ce(logits, y_cat)

            loss_center = model.center_loss_fn(feat, y_cat) * model.lambda_c
            loss_sep = model.sep_loss_fn(model.center_loss_fn.centers) * model.lambda_s
            loss_sup = supcon_loss_fn(proj, y_cat)
            N = x_mix.size(0)
            feat_clean = feat[:N].detach()
            feat_noisy_rec = feat_rec[N:]
            loss_rec = F.mse_loss(feat_noisy_rec, feat_clean)
            loss = loss_ce + loss_center + loss_sep + w_sup * loss_sup + w_rec * loss_rec

            pred = logits[:N].argmax(dim=1)
            correct += (pred == y_a).sum().item()
            bs = x.size(0)
            total += bs
        else:
            logits, feat = model(x, apply_rotation=True, return_all=True)
            loss_ce = criterion_ce(logits, y)
            loss_center = model.center_loss_fn(feat, y) * model.lambda_c
            loss_sep = model.sep_loss_fn(model.center_loss_fn.centers) * model.lambda_s
            loss_sup = torch.tensor(0.0, device=device)
            loss_rec = torch.tensor(0.0, device=device)
            loss = loss_ce + loss_center + loss_sep
            pred = logits.argmax(dim=1)
            bs = x.size(0)
            correct += (pred == y).sum().item()
            total += bs

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_center += loss_center.item() * bs
        total_sep += loss_sep.item() * bs
        total_sup += loss_sup.item() * bs
        total_rec += loss_rec.item() * bs

    return (total_loss / total, correct / total,
            total_ce / total, total_center / total, total_sep / total,
            total_sup / total, total_rec / total)


def validate_trimodnet_v17_ncfa(model, dataloader, criterion_ce, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x, apply_rotation=False)
            loss = criterion_ce(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total



def snr_aware_noise_pair_safe(iq, snr=None, snr_range=(12.0, 24.0), only_snr_ge=0.0):
    """
    给高/中 SNR 样本构造轻度噪声视图。
    不再对 RML2018a 的低 SNR 样本继续强行加噪。
    """
    B = iq.size(0)
    sig_pow = iq.pow(2).mean(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    snr_db = torch.empty(B, 1, 1, device=iq.device).uniform_(snr_range[0], snr_range[1])
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pow = sig_pow / snr_lin
    noisy = iq + torch.randn_like(iq) * noise_pow.sqrt()

    if snr is not None:
        mask = (snr >= only_snr_ge).float().view(B, 1, 1).to(iq.device)
        noisy = noisy * mask + iq * (1.0 - mask)

    return noisy


def train_one_epoch_trincfa_rml2018(
    model, dataloader, optimizer, criterion_ce, supcon_loss_fn,
    device, epoch, total_epochs=100,
    lambda_sup_max=0.05,
    lambda_rec_max=0.02,
    warmup_ncfa=25,
    ramp_ncfa=25,
    ce_noisy_weight=0.0,
):
    model.train()

    total_loss = total_ce = total_center = total_sep = total_sup = total_rec = 0.0
    correct = total = 0

    # RML2018a: NCFA 延后、减弱
    if epoch <= warmup_ncfa:
        w_sup, w_rec = 0.0, 0.0
        noise_range = (20.0, 30.0)
    elif epoch <= warmup_ncfa + ramp_ncfa:
        r = (epoch - warmup_ncfa) / float(ramp_ncfa)
        w_sup = lambda_sup_max * r
        w_rec = lambda_rec_max * r
        noise_range = (16.0, 28.0)
    else:
        w_sup = lambda_sup_max
        w_rec = lambda_rec_max
        noise_range = (12.0, 24.0)

    for x, y, snr in tqdm(
        dataloader,
        desc=f'Train TriNCFA-RML2018 [w_sup={w_sup:.3f} w_rec={w_rec:.3f}]',
        leave=False
    ):
        x = x.to(device)
        y = y.to(device)
        snr = snr.to(device)

        optimizer.zero_grad()

        if w_sup > 0.0 or w_rec > 0.0:
            x_noisy = snr_aware_noise_pair_safe(
                x, snr=snr,
                snr_range=noise_range,
                only_snr_ge=0.0
            )

            x_cat = torch.cat([x, x_noisy], dim=0)
            y_cat = torch.cat([y, y], dim=0)

            logits, feat, proj, feat_rec = model(
                x_cat,
                apply_rotation=True,
                return_contrast=True
            )

            N = x.size(0)
            logits_clean = logits[:N]
            logits_noisy = logits[N:]
            feat_clean = feat[:N]
            feat_noisy_rec = feat_rec[N:]

            # CE 主要约束 clean 样本，避免 noisy 样本过强干扰分类头
            loss_ce = criterion_ce(logits_clean, y)
            if ce_noisy_weight > 0:
                loss_ce = loss_ce + ce_noisy_weight * criterion_ce(logits_noisy, y)

            # CenterLoss 只用 clean 特征，避免噪声视图把中心拉散
            loss_center = model.center_loss_fn(feat_clean, y) * model.lambda_c
            loss_sep = model.sep_loss_fn(model.center_loss_fn.centers) * model.lambda_s

            # SupCon 用 clean + noisy 成对特征，但权重要小
            loss_sup = supcon_loss_fn(proj, y_cat)

            # noisy 重构对齐 clean
            loss_rec = F.mse_loss(feat_noisy_rec, feat_clean.detach())

            loss = loss_ce + loss_center + loss_sep + w_sup * loss_sup + w_rec * loss_rec

            pred = logits_clean.argmax(dim=1)
            correct += (pred == y).sum().item()
            bs = x.size(0)
            total += bs

        else:
            logits, feat = model(x, apply_rotation=True, return_all=True)

            loss_ce = criterion_ce(logits, y)
            loss_center = model.center_loss_fn(feat, y) * model.lambda_c
            loss_sep = model.sep_loss_fn(model.center_loss_fn.centers) * model.lambda_s
            loss_sup = torch.tensor(0.0, device=device)
            loss_rec = torch.tensor(0.0, device=device)

            loss = loss_ce + loss_center + loss_sep

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            bs = x.size(0)
            total += bs

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_center += loss_center.item() * bs
        total_sep += loss_sep.item() * bs
        total_sup += loss_sup.item() * bs
        total_rec += loss_rec.item() * bs

    return (
        total_loss / total,
        correct / total,
        total_ce / total,
        total_center / total,
        total_sep / total,
        total_sup / total,
        total_rec / total,
    )



# ============================================================================
# 通用训练 / 验证
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for x, y, _ in tqdm(dataloader, desc='Training', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


# ============================================================================
# 评估: 按 SNR 分组准确率, 同时收集特征用于 t-SNE
# ============================================================================
@torch.no_grad()
def evaluate_with_features(model, dataset, device, model_name, batch_size=256):
    """
    返回:
      snr_vals : list of unique snr
      acc_vals : 各 snr 准确率
      preds, labels : 全量预测/标签 (np.ndarray)
      snrs : 全量 snr (np.ndarray)
      features : 全量特征 (np.ndarray, [N, D])
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels, all_snr = [], [], []
    all_features = []

    for x, y, snr in loader:
        x = x.to(device)
        # 前向 + 特征提取
        if model_name == 'TriNCFA_Net':
            feat = model(x, apply_rotation=False, return_features=True)
            logits = model(x, apply_rotation=False)
        else:
            feat = extract_feature_generic(model, x, model_name)
            logits = model(x)
        pred = logits.argmax(dim=1)
        all_preds.append(pred.cpu().numpy())
        all_labels.append(y.numpy())
        all_snr.append(snr.numpy())
        all_features.append(feat.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    snrs = np.concatenate(all_snr)
    features = np.concatenate(all_features, axis=0)

    snr_vals = sorted(np.unique(snrs).tolist())
    acc_vals = []
    for s in snr_vals:
        m = snrs == s
        acc_vals.append(float((preds[m] == labels[m]).mean()) if m.sum() > 0 else 0.0)
    return snr_vals, acc_vals, preds, labels, snrs, features


def extract_feature_generic(model, x, model_name):
    """
    通用特征提取: 取分类头之前的特征向量。
    实现方式: 对每个模型 hook 倒数第二层, 失败时回退为 logits。
    """
    feats_holder = {}

    # 找最后一个 Linear 层 (分类头)
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m

    if last_linear is None:
        with torch.no_grad():
            return model(x)

    def hook(module, input, output):
        feats_holder['x'] = input[0].detach()

    handle = last_linear.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(x)
        feat = feats_holder.get('x', None)
        if feat is None:
            feat = model(x)
    finally:
        handle.remove()
    return feat


def compute_metrics(all_preds, all_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    accuracy = (all_preds == all_labels).mean()
    return accuracy, precision, recall, f1


# ============================================================================
# 模型复杂度: Params / FLOPs / Inference time
# ============================================================================
def compute_complexity(model, sig_size, device):
    """计算参数量 (M) 和 FLOPs (M)"""
    n_params = sum(p.numel() for p in model.parameters())
    flops = 0
    try:
        from thop import profile
        model.eval()
        dummy = torch.randn(1, 2, sig_size).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception as e:
        print(f"[WARN] FLOPs (thop) failed: {e}, FLOPs=0. (pip install thop)")
    return n_params, macs




# ============================================================================
# 主程序
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    setup_plot_style()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Dataset: {args.dataset}, Model: {args.model_name}, sig_size: {args.sig_size}')

    os.makedirs('weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # -------- 1. 数据集 --------
    source_dataset = get_dataset(args)
    class_names = MODULATION_TYPES
    print(f'Total samples: {len(source_dataset)}, classes: {NUM_CLASSES}')

    labels = source_dataset.labels.numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(sss.split(np.zeros(len(labels)), labels))
    val_idx_rel, test_idx_rel = next(
        StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42).split(
            np.zeros(len(temp_idx)), labels[temp_idx]))
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    train_dataset = Subset(source_dataset, train_idx)
    val_dataset = Subset(source_dataset, val_idx)
    test_dataset = Subset(source_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # -------- 2. 模型 --------
    factory = import_model(args.model_name, args.sig_size)
    model = factory(NUM_CLASSES, args.sig_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    # 只有显式 resume 时才加载旧权重
    if getattr(args, "resume", False):
        if os.path.exists(args.save_model):
            print(f"[RESUME] Loading checkpoint: {args.save_model}")
            model.load_state_dict(torch.load(args.save_model, map_location=device))
        else:
            print(f"[WARN] --resume is set but checkpoint not found: {args.save_model}")
    else:
        print("[INFO] Training from scratch.")

    # -------- 3. 训练 --------
    if args.model_name == 'TriNCFA_Net':
        from models.TriNCFANet.TriNCFANet import SupConLoss

        is_2018 = args.dataset == "RML2018a"

        criterion_ce = LabelSmoothingCrossEntropy(
            smoothing=0.02 if is_2018 else 0.05
        )
        supcon_loss_fn = SupConLoss(temperature=0.1)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=5e-5 if is_2018 else 1e-4
        )

        # RML2018a 不要 10 轮低学习率 warmup
        if is_2018:
            warmup_epochs = 2
        else:
            warmup_epochs = 10

        if warmup_epochs > 0:
            scheduler_warmup = LinearLR(
                optimizer,
                start_factor=0.5 if is_2018 else 0.1,
                total_iters=warmup_epochs
            )
            scheduler_cosine = CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - warmup_epochs
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                [scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_losses, val_losses, val_accs = [], [], []
        best_val_acc = 0.0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, args.epochs + 1):
            if is_2018:
                train_out = train_one_epoch_trincfa_rml2018(
                    model, train_loader, optimizer, criterion_ce, supcon_loss_fn,
                    device, epoch,
                    total_epochs=args.epochs,
                    lambda_sup_max=0.05,
                    lambda_rec_max=0.02,
                    warmup_ncfa=25,
                    ramp_ncfa=25,
                    ce_noisy_weight=0.0,
                )
            else:
                train_out = train_one_epoch_trimodnet_v17_ncfa(
                    model, train_loader, optimizer, criterion_ce, supcon_loss_fn,
                    device, epoch, total_epochs=args.epochs,
                    lambda_sup_max=0.5,
                    lambda_rec_max=0.1,
                    warmup_ncfa=10,
                    ramp_ncfa=20,
                )

            train_loss, train_acc, train_ce, train_center, train_sep, train_sup, train_rec = train_out

            val_loss, val_acc = validate_trimodnet_v17_ncfa(
                model, val_loader, criterion_ce, device
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f'Epoch {epoch:3d}: TrLoss {train_loss:.4f} '
                  f'(CE:{train_ce:.4f} Cen:{train_center:.4f} Sep:{train_sep:.4f} '
                  f'Sup:{train_sup:.4f} Rec:{train_rec:.4f}) '
                  f'TrAcc {train_acc:.4f} | VaLoss {val_loss:.4f} VaAcc {val_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), args.save_model)
                print(f'  -> Best model saved (val_acc={val_acc:.4f})')

            scheduler.step()


    else:
        criterion_ce = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        train_losses, val_losses, val_accs = [], [], []
        best_val_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion_ce, device)
            val_loss, val_acc = validate(model, val_loader, criterion_ce, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f'Epoch {epoch:3d}: TrLoss {train_loss:.4f} TrAcc {train_acc:.4f} | '
                  f'VaLoss {val_loss:.4f} VaAcc {val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.save_model)
                print(f'  -> Best model saved')
            scheduler.step()

    # -------- 4. 训练曲线 --------
    name_prefix = f'{args.dataset}_{args.model_name}'
    plot_train_curves(train_losses, val_losses, val_accs,
                      name_prefix=name_prefix, results_dir='results')

    # -------- 5. 加载最佳模型并评估 --------
    model.load_state_dict(torch.load(args.save_model, map_location=device))
    print('\n=== Evaluating on test set ===')

    snr_vals, acc_vals, preds, labels_arr, snrs_arr, features = evaluate_with_features(
        model, test_dataset, device, args.model_name, batch_size=args.batch_size)

    acc, prec, rec, f1 = compute_metrics(preds, labels_arr)
    print(f'Test  Acc:{acc:.4f}  Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1:.4f}')

    # -------- 6. 标准 SNR 表 (-20..高端) --------
    if args.dataset == "RML2018a":
        snr_full = list(range(-20, 32, 2))
    else:
        snr_full = list(range(-20, 19, 2))
    acc_dict = dict(zip(snr_vals, acc_vals))
    source_acc_list = [acc_dict.get(s, 0.0) for s in snr_full]

    # -------- 7. 复杂度指标 --------
    print('\n=== Computing complexity ===')
    n_params, flops = compute_complexity(model, args.sig_size, device)
    inf_dict = measure_inference_time(
        model, args.sig_size, device,
        batch_size=128, n_warmup=30, n_runs=100,
    )
    inf_time = inf_dict["per_sample_ms"]  # 论文常用指标
    latency_1f = inf_dict["latency_1f_ms"]
    fps = inf_dict["throughput_fps"]
    print(f'  Params: {n_params / 1e6:.4f} M, FLOPs: {flops / 1e6:.2f} M')
    print(f'  PerSample(b=128): {inf_time:.4f} ms  | '
          f'Latency(b=1): {latency_1f:.4f} ms | Throughput: {fps:.1f} fps')

    # -------- 8. low_snr / high_snr 区间平均 --------
    def avg_in_range(snr_lo, snr_hi):
        vals = [a for s, a in zip(snr_vals, acc_vals) if snr_lo <= s <= snr_hi]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    low_snr_acc = avg_in_range(-20, 0)        # -20..0
    if args.dataset == "RML2018a":
        high_snr_acc = avg_in_range(0, 30)    # 0..30
    else:
        high_snr_acc = avg_in_range(0, 18)    # 0..18

    # -------- 9. 可视化 --------
    plot_snr_curve(snr_vals, acc_vals,
                   f'{name_prefix} SNR-Accuracy',
                   f'results/{name_prefix}_snr_acc.png')
    plot_confusion(labels_arr, preds,
                   f'{name_prefix} Confusion Matrix (All SNR)',
                   f'results/{name_prefix}_confusion.png',
                   class_names=class_names)

    # 不同 SNR 的混淆矩阵: 选 -10, 0, 10 (RML2018a 还增加 18)
    cm_snr_levels = [-10, 0, 10]
    if args.dataset == "RML2018a":
        cm_snr_levels.append(18)
    for snr_target in cm_snr_levels:
        mask = snrs_arr == snr_target
        if mask.sum() < 50:
            continue
        plot_confusion(labels_arr[mask], preds[mask],
                       f'{name_prefix} CM @ SNR={int(snr_target)} dB',
                       f'results/{name_prefix}_confusion_snr{int(snr_target):+d}dB.png',
                       class_names=class_names)

    # t-SNE 可视化
    print('\n=== Running t-SNE per SNR ===')
    if args.dataset == "RML2018a":
        tsne_snr_levels = [-10, -4, 0, 6, 10, 18]
    else:
        tsne_snr_levels = [-10, -4, 0, 6, 10]
    plot_tsne_per_snr(features, labels_arr, snrs_arr,
                      class_names=class_names,
                      save_path_prefix=f'results/{name_prefix}_tsne',
                      snr_levels=tsne_snr_levels)

    # 全量 t-SNE (混合 SNR)
    try:
        plot_tsne_all(
            features, labels_arr, class_names=class_names,
            save_path=f'results/{name_prefix}_tsne_all.png',
            max_samples=3000,
            title=f'{name_prefix} t-SNE (All SNR)',
        )
    except Exception as e:
        print(f"[WARN] all-snr t-SNE failed: {e}")

    # -------- 10. 保存 JSON --------
    result_dict = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "signal_length": args.sig_size,
        "num_classes": NUM_CLASSES,
        "complexity": {
            "Parameters(M)": float(n_params / 1e6),
            "FLOPs(M)": float(flops / 1e6),
            "per_sample_ms": float(inf_time),  # b=128 amortized, 论文常用
            "latency_1f_ms": float(latency_1f),  # b=1 单帧延迟
            "throughput_fps": float(fps),
            # 旧字段保留, 兼容已生成的可视化
            "inference_time(ms/sample)": float(inf_time),
        },
        "source": {
            "snr_full": snr_full,
            "snr": source_acc_list,
            "metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "low_snr": float(low_snr_acc),
                "high_snr": float(high_snr_acc),
            }
        }
    }

    json_path = f'results/{name_prefix}_baseline_best.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)

    print(f'\n=== Summary ===')
    print(f'  Overall Acc : {acc:.4f}')
    print(f'  Low  SNR (-20..0)  : {low_snr_acc:.4f}')
    print(f'  High SNR ({"0..30" if args.dataset == "RML2018a" else "0..18"}) : {high_snr_acc:.4f}')
    print(f'  Params(M)     : {n_params/1e6:.4f}')
    print(f'  FLOPs(M)      : {flops/1e6:.2f}')
    print(f'  PerSample(ms) : {inf_time:.4f}  (b=128 amortized, 论文常用)')
    print(f'  Latency_1f(ms): {latency_1f:.4f}  (b=1 单帧, 实时性参考)')
    print(f'  Throughput    : {fps:.1f} fps')
    print(f'\nResults saved to {json_path}')
    print('All done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="RML2016a",
        choices=["RML2016a", "RML2016b", "RML2018a"],
        help="The dataset to be used for Auto Modulation Classification",
    )
    parser.add_argument('--model_name', type=str, default='TriNCFA_Net',
                        help='AMCNet, MCLDNN, PETCGDNN, FEA_T, MCFormer, IQFormer, SMT, TriNCFA_Net')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='路径; 若为 None 会按 --dataset 自动设置')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--sig_size', type=int, default=None,
                        help='信号长度; 若为 None 按 --dataset 自动设置 (128 or 1024)')
    parser.add_argument('--save_model', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--samples_per_class_snr', type=int, default=None,
                        help='RML2018a 每类每 SNR 子采样数 (None 表示全量)')
    parser.add_argument('--resume', action='store_true',default=False,
                        help='resume training from args.save_model')
    args = parser.parse_args()

    # ---- 数据集路径与 sig_size 自动适配 ----
    default_paths = {
        "RML2016a": "dataset/RML2016.10a_dict.pkl",
        "RML2016b": "dataset/RML2016.10b.dat",
        "RML2018a": "dataset/GOLD_XYZ_OSC.0001_1024.hdf5",
    }
    default_sig = {"RML2016a": 128, "RML2016b": 128, "RML2018a": 1024}

    if args.dataset_path is None:
        args.dataset_path = default_paths[args.dataset]
    if args.sig_size is None:
        args.sig_size = default_sig[args.dataset]
    if args.save_model is None:
        args.save_model = f'weights/{args.dataset}_{args.model_name}_baseline_best.pth'
    else:
        # 若用户传入但没带 dataset 前缀, 强制加上以避免冲突
        base = os.path.basename(args.save_model)
        if args.dataset not in base:
            args.save_model = f'weights/{args.dataset}_{args.model_name}_baseline_best.pth'

    main(args)

