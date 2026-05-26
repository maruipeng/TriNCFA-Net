# viz_metrics.py
# ---------------------------------------------------------------
# 统一管理:
#   1. SCI 论文级绘图风格 (Times New Roman, 高清, 合理字号)
#   2. 修复版 measure_inference_time (大 batch amortize)
#   3. 修复版 plot_snr_curve / plot_confusion / plot_tsne_per_snr
# 替换 train.py 中对应的旧函数即可
# ---------------------------------------------------------------
import os
import time
import warnings

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle, ConnectionPatch


# ============================================================================
# 1. SCI 论文级绘图风格
# ============================================================================
def setup_plot_style(font_family="Times New Roman", base_size=13):
    """
    SCI 论文级 matplotlib 风格.
    自动检测系统是否安装了 Times New Roman, 若无则退化到 'serif' 并给出告警.

    Args:
        font_family: 主字体, 默认 Times New Roman
        base_size: 基础字号, 标签/标题会在此基础上微调
    """
    # 检测字体可用性
    available = {f.name for f in font_manager.fontManager.ttflist}
    if font_family not in available:
        warnings.warn(
            f"[viz_metrics] '{font_family}' not found in system fonts. "
            f"Falling back to generic 'serif'. "
            f"Tip: place Times New Roman .ttf into matplotlib's font dir, "
            f"then run `font_manager._rebuild()`.",
            stacklevel=2,
        )
        font_family = "serif"

    matplotlib.rcParams.update({
        # 字体
        "font.family":      font_family,
        "font.serif":       [font_family, "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",          # 数学字体匹配 Times
        "axes.unicode_minus": False,         # 负号正常显示
        # 字号
        "font.size":        base_size,
        "axes.titlesize":   base_size + 1,
        "axes.labelsize":   base_size + 1,
        "xtick.labelsize":  base_size - 1,
        "ytick.labelsize":  base_size - 1,
        "legend.fontsize":  base_size - 2,
        # 线宽 / 标记
        "axes.linewidth":   1.2,
        "lines.linewidth":  1.8,
        "lines.markersize": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size":  4.0,
        "ytick.major.size":  4.0,
        # 输出
        "figure.dpi":       110,
        "savefig.dpi":      600,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.04,
        "savefig.transparent": False,
        # 网格 (子图按需开启)
        "grid.linestyle":  "--",
        "grid.linewidth":  0.6,
        "grid.alpha":      0.5,
    })


def _save_fig(fig, save_path, also_pdf=True):
    """统一保存: PNG 600dpi + 同名 PDF (矢量, 论文首选)"""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    if also_pdf:
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ============================================================================
# 2. 修复版 inference time
# ============================================================================
def measure_inference_time(model, sig_size, device,
                           batch_size=128, n_warmup=30, n_runs=100):
    """
    返回 dict:
      {
        "per_sample_ms":   batch=128 amortized 每样本时间   (论文常用)
        "latency_1f_ms":   batch=1 单帧延迟                (实时性参考)
        "throughput_fps":  batch=128 时每秒处理样本数
      }

    背景:
      原版用 batch=1 测量, 主要被 GPU kernel launch overhead 主导
      (单次约 1-5ms), 对所有轻量模型都得到 5-6ms 级数字, 失真严重.
      论文中 0.x ms/sample 通常是大 batch 吞吐量除以 batch_size.
    """
    model.eval()
    results = {}

    # ---- 1) 大 batch amortized per-sample ----
    dummy_b = torch.randn(batch_size, 2, sig_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_b)

    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_b)
        e.record()
        torch.cuda.synchronize()
        total_ms = s.elapsed_time(e)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(dummy_b)
        total_ms = (time.perf_counter() - t0) * 1000.0

    per_sample = total_ms / (n_runs * batch_size)
    results["per_sample_ms"]  = float(per_sample)
    results["throughput_fps"] = float(1000.0 / per_sample)

    # ---- 2) batch=1 单帧延迟 ----
    dummy_1 = torch.randn(1, 2, sig_size, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_1)

    n_runs_1 = n_runs * 2  # batch=1 时多跑些
    if device.type == "cuda":
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            for _ in range(n_runs_1):
                _ = model(dummy_1)
        e.record()
        torch.cuda.synchronize()
        total_ms_1 = s.elapsed_time(e)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs_1):
                _ = model(dummy_1)
        total_ms_1 = (time.perf_counter() - t0) * 1000.0

    results["latency_1f_ms"] = float(total_ms_1 / n_runs_1)
    return results


# ============================================================================
# 3. 高清绘图函数 (替换 train.py 中对应的旧函数)
# ============================================================================
def plot_snr_curve(snr_vals, acc_vals, title, save_path,
                   ylim=(0.0, 1.0), color="#1f77b4"):
    """单模型 SNR-Accuracy 曲线 (论文级)"""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(snr_vals, acc_vals, marker="o", color=color, linewidth=1.2,
            markerfacecolor="white", markeredgewidth=1.6, label="Accuracy")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    _save_fig(fig, save_path)


# def plot_snr_curves_multi(snr_vals_dict, title, save_path,
#                           ylim=(0.0, 1.0), highlight_key=None):
#     """
#     多模型 SNR-Accuracy 对比曲线
#     snr_vals_dict: {model_name: (snr_list, acc_list)}
#     highlight_key: 高亮线 (本文方法), 用稍粗实线 + 实心圆
#     横轴固定 -20 到 20 dB，主刻度每5dB；纵轴主刻度每0.1
#     图例单列放置
#     """
#     # 协调的宽高比：宽度略大于高度
#     fig, ax = plt.subplots(figsize=(5.5, 5.0))
#     markers = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "<", ">"]
#     cmap = plt.get_cmap("tab10")
#     for i, (name, (snrs, accs)) in enumerate(snr_vals_dict.items()):
#         is_h = (name == highlight_key)
#         ax.plot(
#             snrs, accs,
#             marker=markers[i % len(markers)],
#             linewidth=1.5 if is_h else 1.0,
#             markersize=5 if is_h else 3.5,
#             color=cmap(i),
#             label=name,
#             markerfacecolor=cmap(i) if is_h else "white",
#             markeredgecolor=cmap(i),
#             markeredgewidth=1.0 if is_h else 0.6,
#         )
#     ax.set_xlabel("SNR (dB)")
#     ax.set_ylabel("Accuracy")
#     ax.set_xlim(-20, 20)
#     ax.set_ylim(*ylim)
#     # 横轴刻度：-20 到 20，步长5
#     # "RML2016a", "RML2016b",(-20, 21, 5)
#     # "RML2018a" (-20, 31, 5)
#
#     ax.set_xticks(np.arange(-20, 21, 5))
#     # 纵轴刻度：根据 ylim 范围每0.1
#     y_min, y_max = ylim
#     ax.set_yticks(np.arange(y_min, y_max + 0.01, 0.1))
#
#     ax.set_title(title)
#     ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
#     # 图例单列，放右下方
#     ax.legend(loc="lower right", ncol=1, frameon=True, framealpha=0.9,
#               edgecolor="0.8", fontsize=12)
#     fig.tight_layout()
#     _save_fig(fig, save_path)



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch


def plot_snr_curves_multi(snr_vals_dict, title, save_path,
                          ylim=(0.0, 1.0), highlight_key=None,
                          zoom_xrange=None, inset=True,
                          inset_position=None):
    """
    多模型 SNR-Accuracy 对比曲线，带局部放大框。

    功能：
        1. RML2016a / RML2016b 自动放大 -4 dB 到 2 dB
        2. RML2018a 自动放大 4 dB 到 6 dB
        3. 图例固定在左上角
        4. 放大框放在放大区间右下侧
        5. 主图只标出真实放大区域矩形，避免与 inset 重叠
        6. 手动绘制连接线，避免连接线穿过放大框内部

    参数：
        snr_vals_dict:
            {
                model_name: (snr_list, acc_list)
            }

        title:
            图标题，一般包含 RML2016a / RML2016b / RML2018a

        save_path:
            保存路径

        ylim:
            y 轴范围，默认 (0.0, 1.0)

        highlight_key:
            需要高亮的模型名称

        zoom_xrange:
            手动指定放大区间，例如 (-4, 2) 或 (4, 6)

        inset:
            是否启用放大框

        inset_position:
            放大框位置，[left, bottom, width, height]，使用 axes fraction 坐标
    """

    if not snr_vals_dict:
        raise ValueError("snr_vals_dict is empty.")

    # ------------------------------------------------------------------
    # 1. 根据 title / save_path 自动判断数据集类型
    # ------------------------------------------------------------------
    dataset_hint = (str(title) + " " + str(save_path)).lower()

    is_rml2018 = "rml2018" in dataset_hint
    is_rml2016 = ("rml2016a" in dataset_hint) or ("rml2016b" in dataset_hint)

    if zoom_xrange is None:
        if is_rml2018:
            zoom_xrange = (0, 10)
        elif is_rml2016:
            zoom_xrange = (-4, 6)
        else:
            zoom_xrange = None

    # ------------------------------------------------------------------
    # 2. 自动确定横轴范围
    # ------------------------------------------------------------------
    all_snrs = []
    for snrs, _ in snr_vals_dict.values():
        all_snrs.extend(np.asarray(snrs, dtype=float).tolist())

    max_snr = max(all_snrs) if all_snrs else 20

    if is_rml2018 or max_snr > 20:
        xlim = (-20, 30)
        xticks = np.arange(-20, 31, 5)
    else:
        xlim = (-20, 20)
        xticks = np.arange(-20, 21, 5)

    # ------------------------------------------------------------------
    # 3. 主图
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    n_models = len(snr_vals_dict)

    markers = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "<", ">", "p"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
    cmap = plt.get_cmap("tab20", max(n_models, 10))

    line_specs = {}

    for i, (name, (snrs, accs)) in enumerate(snr_vals_dict.items()):
        is_h = (name == highlight_key)

        color = cmap(i)
        marker = markers[i % len(markers)]
        linestyle = "-" if is_h else linestyles[i % len(linestyles)]

        line_specs[name] = {
            "color": color,
            "marker": marker,
            "linestyle": linestyle,
            "is_highlight": is_h,
        }

        ax.plot(
            snrs, accs,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0 if is_h else 1.2,
            markersize=5.2 if is_h else 3.8,
            color=color,
            label=name,
            markerfacecolor=color if is_h else "white",
            markeredgecolor=color,
            markeredgewidth=1.1 if is_h else 0.7,
            zorder=4 if is_h else 3,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xticks(xticks)

    y_min, y_max = ylim
    ax.set_yticks(np.arange(y_min, y_max + 0.001, 0.1))

    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)

    # ------------------------------------------------------------------
    # 4. 局部放大框
    # ------------------------------------------------------------------
    if inset and zoom_xrange is not None:
        zx0, zx1 = float(zoom_xrange[0]), float(zoom_xrange[1])

        if zx0 > zx1:
            zx0, zx1 = zx1, zx0

        # --------------------------------------------------------------
        # 4.1 获取放大区间内的 y 范围
        # --------------------------------------------------------------
        zoom_y_values = []

        for _, (snrs, accs) in snr_vals_dict.items():
            snrs_np = np.asarray(snrs, dtype=float)
            accs_np = np.asarray(accs, dtype=float)

            mask = (snrs_np >= zx0) & (snrs_np <= zx1)

            if np.any(mask):
                zoom_y_values.extend(accs_np[mask].tolist())

        if len(zoom_y_values) > 0:
            zy_min = max(y_min, float(np.nanmin(zoom_y_values)))
            zy_max = min(y_max, float(np.nanmax(zoom_y_values)))

            # y 方向 padding，让局部曲线不要贴边
            pad = max(0.025, 0.18 * max(zy_max - zy_min, 1e-3))

            zy_min = max(y_min, zy_min - pad)
            zy_max = min(y_max, zy_max + pad)

            # 防止 y 范围太窄
            if zy_max - zy_min < 0.08:
                mid = 0.5 * (zy_min + zy_max)
                zy_min = max(y_min, mid - 0.04)
                zy_max = min(y_max, mid + 0.04)

            # ----------------------------------------------------------
            # 4.2 放大框位置
            # ----------------------------------------------------------
            if inset_position is None:
                # 右下角，尺寸较小，尽量不遮挡主图曲线
                # [left, bottom, width, height]
                inset_position = [0.485, 0.165, 0.405, 0.335]

                if is_rml2018:
                    inset_position = [0.485, 0.105, 0.405, 0.335]
                else:
                    inset_position = [0.485, 0.165, 0.405, 0.335]

            axins = ax.inset_axes(inset_position)

            # ----------------------------------------------------------
            # 4.3 绘制 inset 内部曲线
            # ----------------------------------------------------------
            for name, (snrs, accs) in snr_vals_dict.items():
                spec = line_specs[name]
                is_h = spec["is_highlight"]

                axins.plot(
                    snrs, accs,
                    marker=spec["marker"],
                    linestyle=spec["linestyle"],
                    linewidth=1.35 if is_h else 0.85,
                    markersize=3.2 if is_h else 2.35,
                    color=spec["color"],
                    markerfacecolor=spec["color"] if is_h else "white",
                    markeredgecolor=spec["color"],
                    markeredgewidth=0.70 if is_h else 0.45,
                    zorder=4 if is_h else 3,
                )

            axins.set_xlim(zx0, zx1)
            axins.set_ylim(zy_min, zy_max)

            # 放大区间通常只有 2~6 dB，刻度不要太密
            if zx1 - zx0 <= 3:
                axins.set_xticks(np.arange(zx0, zx1 + 0.001, 1.0))
            else:
                axins.set_xticks(np.arange(zx0, zx1 + 0.001, 2.0))

            axins.set_yticks(np.linspace(zy_min, zy_max, 3))

            axins.tick_params(
                axis="both",
                labelsize=7.5,
                width=0.75,
                length=2.5
            )

            axins.grid(True, linestyle="--", linewidth=0.40, alpha=0.50)

            axins.set_title(
                f"{zx0:g} to {zx1:g} dB",
                fontsize=8.2,
                pad=1.2
            )

            # inset 边框略加粗
            for spine in axins.spines.values():
                spine.set_linewidth(1.0)
                spine.set_edgecolor("0.20")

            # ----------------------------------------------------------
            # 4.4 主图中只画真实 zoom 区域，不画整条 axvspan
            # ----------------------------------------------------------
            zoom_rect_bg = Rectangle(
                (zx0, zy_min),
                zx1 - zx0,
                zy_max - zy_min,
                fill=True,
                facecolor="0.85",
                edgecolor="none",
                alpha=0.18,
                zorder=1,
            )
            ax.add_patch(zoom_rect_bg)

            zoom_rect = Rectangle(
                (zx0, zy_min),
                zx1 - zx0,
                zy_max - zy_min,
                fill=False,
                edgecolor="0.25",
                linewidth=0.9,
                linestyle="-",
                zorder=5,
            )
            ax.add_patch(zoom_rect)

            # ----------------------------------------------------------
            # 4.5 手动画连接线，避免连接线穿过放大框内部
            # ----------------------------------------------------------
            # 连接策略：
            #   主图 zoom 框右下角 -> inset 左上角
            #   主图 zoom 框右上角 -> inset 左下角
            #
            # 由于 inset 放在右下角，这种连接方式通常不会穿过 inset 内部。
            # ----------------------------------------------------------
            con1 = ConnectionPatch(
                xyA=(zx0, zy_min),
                coordsA=ax.transData,
                xyB=(0.0, 0.0),
                coordsB=axins.transAxes,
                axesA=ax,
                axesB=axins,
                color="0.35",
                linewidth=0.75,
                linestyle="-",
                alpha=0.85,
                zorder=2,
                clip_on=False,
            )

            con2 = ConnectionPatch(
                xyA=(zx1, zy_max),
                coordsA=ax.transData,
                xyB=(1.0, 1.0),
                coordsB=axins.transAxes,
                axesA=ax,
                axesB=axins,
                color="0.35",
                linewidth=0.75,
                linestyle="-",
                alpha=0.85,
                zorder=2,
                clip_on=False,
            )

            ax.add_artist(con1)
            ax.add_artist(con2)

    # ------------------------------------------------------------------
    # 5. 图例固定左上角
    # ------------------------------------------------------------------
    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.8",
        fontsize=11,
        handlelength=2.2,
        handletextpad=0.5,
        borderpad=0.35,
    )

    fig.tight_layout()
    _save_fig(fig, save_path)






from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_confusion(all_labels, all_preds, title, save_path, class_names,
                   normalize="true"):
    """混淆矩阵 (Times New Roman, 适配类别数, colorbar 高度与主图一致)"""
    cm = confusion_matrix(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        normalize=normalize,   # 'true', 'pred', 'all', None
    )
    n_cls = len(class_names)
    # 自适应画布大小
    side = max(6.5, min(12.0, 0.5 * n_cls + 2.2))
    fig, ax = plt.subplots(figsize=(side, side * 0.9))

    # 绘制混淆矩阵
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    # 设置坐标轴刻度与标签
    ax.set_xticks(np.arange(n_cls))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)

    # 添加格子数值
    fmt = ".2f" if normalize else "d"
    text_fontsize = max(7, 11 - n_cls // 4)
    # 根据背景色深浅自动切换文字颜色
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n_cls):
        for j in range(n_cls):
            val = cm[i, j]
            if normalize:
                # 处理全零行导致的 NaN
                if np.isnan(val):
                    text = "0.00"
                else:
                    text = f"{val:.2f}"
            else:
                text = f"{int(val):d}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val > threshold else "black",
                    fontsize=text_fontsize)

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # 关键：用 divider 创建与主图等高的 colorbar 轴
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    if normalize:
        cbar.set_label("Proportion")
    else:
        cbar.set_label("Counts")

    fig.tight_layout()
    _save_fig(fig, save_path)


def plot_tsne_per_snr(features, labels, snrs, class_names,
                      save_path_prefix, snr_levels=None, max_samples=2000):
    """对不同 SNR 分别做 t-SNE 并保存 (高清)"""
    if snr_levels is None:
        snr_levels = [-10, -4, 0, 6, 10]
    rng = np.random.RandomState(42)
    saved_paths = []

    for snr_target in snr_levels:
        m = (snrs == snr_target)
        if m.sum() < 30:
            continue
        feat_sub = features[m]
        lab_sub = labels[m]
        if len(feat_sub) > max_samples:
            sel = rng.choice(len(feat_sub), max_samples, replace=False)
            feat_sub = feat_sub[sel]
            lab_sub = lab_sub[sel]
        try:
            tsne = TSNE(
                n_components=2, random_state=42,
                perplexity=min(30, max(5, len(feat_sub) // 5)),
                init="pca", learning_rate="auto",
            )
            emb = tsne.fit_transform(feat_sub)
        except Exception as ex:
            print(f"[WARN] t-SNE @ SNR={snr_target} dB failed: {ex}")
            continue

        n_cls = len(class_names)
        fig, ax = plt.subplots(figsize=(7.5, 6.0))
        cmap = plt.get_cmap("tab20" if n_cls > 10 else "tab10", n_cls)
        for cls in np.unique(lab_sub):
            sel = lab_sub == cls
            ax.scatter(
                emb[sel, 0], emb[sel, 1],
                s=14, color=cmap(int(cls)),
                label=class_names[int(cls)],
                alpha=0.75, edgecolors="none",
            )
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.set_title(f"t-SNE @ SNR = {int(snr_target)} dB")
        ax.tick_params(axis="both", which="both", length=0)  # 去掉刻度
        ax.set_xticklabels([]); ax.set_yticklabels([])

        # 图例放图内，字体缩小至 8，列数根据类别数自适应
        ncol = 1 if n_cls <= 12 else 2
        ax.legend(
            loc="best",
            ncol=ncol,
            frameon=True, framealpha=0.9, edgecolor="0.8",
            fontsize=8,
            handletextpad=0.5,
        )
        sp = f"{save_path_prefix}_snr{int(snr_target):+d}dB.png"
        _save_fig(fig, sp)
        saved_paths.append(sp)
    return saved_paths


def plot_tsne_all(features, labels, class_names, save_path,
                  max_samples=3000, title="t-SNE (All SNR)"):
    """全量混合 SNR 的 t-SNE"""
    rng = np.random.RandomState(0)
    sel = rng.choice(len(features), min(max_samples, len(features)), replace=False)
    try:
        tsne = TSNE(n_components=2, random_state=42, init="pca",
                    learning_rate="auto")
        emb = tsne.fit_transform(features[sel])
    except Exception as ex:
        print(f"[WARN] all-snr t-SNE failed: {ex}")
        return

    n_cls = len(class_names)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    cmap = plt.get_cmap("tab20" if n_cls > 10 else "tab10", n_cls)
    for cls in np.unique(labels[sel]):
        m = labels[sel] == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=12,
                   color=cmap(int(cls)), label=class_names[int(cls)],
                   alpha=0.75, edgecolors="none")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(title)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([]); ax.set_yticklabels([])

    # 图例放图内，字体缩小至 8，列数自适应
    ncol = 1 if n_cls <= 12 else 2
    ax.legend(
        loc="best",
        ncol=ncol,
        frameon=True, framealpha=0.9, edgecolor="0.8",
        fontsize=8,
        handletextpad=0.5,
    )
    _save_fig(fig, save_path)


def plot_train_curves(train_losses, val_losses, val_accs,
                      name_prefix, results_dir="results"):
    """训练 loss + val_acc 曲线 (论文级)"""
    # loss
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(train_losses, label="Train Loss", color="#1f77b4")
    ax.plot(val_losses, label="Val Loss", color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{name_prefix} Loss Curve")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    _save_fig(fig, f"{results_dir}/{name_prefix}_loss.png")

    # val acc
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(val_accs, color="#2ca02c", marker="o", markersize=3,
            label="Val Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{name_prefix} Validation Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout()
    _save_fig(fig, f"{results_dir}/{name_prefix}_val_acc.png")


# ============================================================================
# 4. 自检
# ============================================================================
if __name__ == "__main__":
    setup_plot_style()
    # 模拟数据
    snr = list(range(-20, 19, 2))
    acc1 = [0.1 + 0.04 * i + 0.01 * np.random.randn() for i in range(len(snr))]
    acc2 = [0.1 + 0.045 * i + 0.01 * np.random.randn() for i in range(len(snr))]
    plot_snr_curve(snr, acc1, "Demo SNR-Acc", "_demo_snr.png")
    plot_snr_curves_multi(
        {"A": (snr, acc1), "B": (snr, acc2)},
        "Demo Multi", "_demo_multi.png", highlight_key="B"
    )
    print("Self-test OK. Files: _demo_snr.png, _demo_multi.png")
