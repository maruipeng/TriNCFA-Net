# plot_snr_comparison.py
"""
绘制指定数据集上所有模型的 SNR-Accuracy 对比曲线。
用法：
    python plot_snr_comparison.py --dataset RML2016a
    python plot_snr_comparison.py --dataset RML2018a --highlight TriNCFA_Net --output results/RML2018a_snr_compare.png
"""
import argparse
import json
import glob
import os
import sys

# 确保能导入同目录下的 viz_metrics
sys.path.insert(0, os.path.dirname(__file__))

from viz_metrics import setup_plot_style, plot_snr_curves_multi


def find_json_files(dataset, results_dir="results"):
    """按数据集通配符查找所有 JSON 结果文件"""
    pattern = os.path.join(results_dir, f"{dataset}_*_baseline_best.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No JSON files found for dataset '{dataset}' "
                                f"in '{results_dir}/' (pattern: {pattern})")
    return files


def load_snr_data(json_path):
    """
    从 JSON 中提取模型名、snr_full、snr_acc。
    返回 (model_name, snr_full, snr_acc) 或 None（解析失败）
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None

    model_name = data.get("model_name")
    source = data.get("source", {})
    snr_full = source.get("snr_full")
    snr_acc = source.get("snr")
    if not model_name or snr_full is None or snr_acc is None:
        print(f"[WARN] Incomplete data in {json_path}, skipping")
        return None
    if len(snr_full) != len(snr_acc):
        print(f"[WARN] Length mismatch in {json_path} (snr_full: {len(snr_full)}, "
              f"snr_acc: {len(snr_acc)}), skipping")
        return None
    return model_name, snr_full, snr_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot SNR-Accuracy comparison for all models on a given dataset"
    )
    parser.add_argument("--dataset", type=str, default="RML2016b",
                        choices=["RML2016a", "RML2016b", "RML2018a"],
                        help="Dataset name (RML2016a, RML2016b, RML2018a)")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing *_baseline_best.json files")
    parser.add_argument("--highlight", type=str, default="TriNCFA_Net",
                        help="Model to highlight (bold line)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: results/<dataset>_snr_compare.png)")
    parser.add_argument("--ylim", type=float, nargs=2, default=(0.0, 1.0),
                        help="y-axis limits (default: 0.0 1.0)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 设置论文风格
    setup_plot_style()

    # 2. 查找所有 JSON 文件
    json_files = find_json_files(args.dataset, args.results_dir)
    print(f"Found {len(json_files)} JSON files for dataset '{args.dataset}':")
    for f in json_files:
        print(f"  - {f}")

    # 3. 加载数据，构建字典
    snr_dict = {}
    for jf in json_files:
        res = load_snr_data(jf)
        if res is None:
            continue
        model_name, snr_full, snr_acc = res
        # 若同一模型出现了多次（例如不同版本），后面的会覆盖前面的；可以加提示
        if model_name in snr_dict:
            print(f"[WARN] Duplicate model '{model_name}' detected, "
                  f"overwriting previous curve with {jf}")
        snr_dict[model_name] = (snr_full, snr_acc)

    if not snr_dict:
        print("[ERROR] No valid SNR data loaded, exiting.")
        sys.exit(1)

    print(f"Loaded SNR curves for models: {list(snr_dict.keys())}")

    # 4. 确定输出路径
    if args.output is None:
        os.makedirs(args.results_dir, exist_ok=True)
        output_path = os.path.join(args.results_dir,
                                   f"{args.dataset}_snr_compare.png")
    else:
        output_path = args.output

    # 5. 绘图
    title = f"{args.dataset}"
    plot_snr_curves_multi(
        snr_vals_dict=snr_dict,
        title=title,
        save_path=output_path,
        ylim=args.ylim,
        highlight_key=args.highlight,
    )
    print(f"Plot saved to: {output_path} (and corresponding .pdf)")


if __name__ == "__main__":
    main()