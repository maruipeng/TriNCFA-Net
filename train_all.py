# train_all.py
# ---------------------------------------------------------------
# 一次性训练并评估指定数据集上的 8 个模型, 自动汇总结果
#
# 用法:
#   python train_all.py --dataset RML2016a --epochs 100
#   python train_all.py --dataset RML2018a --epochs 80 --skip MCFormer FEA_T
#   python train_all.py --dataset RML2016b --models AMCNet MCLDNN TriNCFA_Net
#
# 产物:
#   weights/<dataset>_<model>_baseline_best.pth      逐模型权重
#   results/<dataset>_<model>_baseline_best.json     逐模型指标
#   results/<dataset>_<model>_*.png/.pdf             逐模型可视化
#   results/<dataset>_summary.json                   汇总 JSON
#   results/<dataset>_summary.md                     汇总 Markdown 表
#   results/<dataset>_summary.csv                    汇总 CSV
#   results/<dataset>_snr_compare.png/.pdf           各模型 SNR 曲线对比图
# ---------------------------------------------------------------
import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime

# 默认全量模型列表 (与 train.py 中 import_model 对齐)
ALL_MODELS = [
    "AMCNet", "MCLDNN", "PETCGDNN",
    "FEA_T", "MCFormer", "IQFormer",
    "SMT", "TriNCFA_Net",
]


# -------------------------------------------------------------
# 解析参数
# -------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Batch trainer for all AMR baselines")
    p.add_argument("--dataset", type=str, default="RML2016a",
                   choices=["RML2016a", "RML2016b", "RML2018a"])
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--models", nargs="+", default=None,
                   help="指定要训练的模型子集; 默认全部 8 个")
    p.add_argument("--skip", nargs="+", default=[],
                   help="跳过的模型 (在 --models 基础上排除)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples_per_class_snr", type=int, default=512)
    p.add_argument("--sig_size", type=int, default=None)
    p.add_argument("--retrain", action="store_true",
                   help="忽略已存在 JSON, 强制重训")
    p.add_argument("--resume", action="store_true",default=False,
                   help="是否从已有权重继续训练；默认从头训练")
    return p.parse_args()


# -------------------------------------------------------------
# 构造 train.py main() 所需的 args
# -------------------------------------------------------------
def make_train_args(parent, model_name):
    """根据用户的批量参数, 拼出 train.py 单次运行的 args (Namespace)"""
    default_paths = {
        "RML2016a": "dataset/RML2016.10a_dict.pkl",
        "RML2016b": "dataset/RML2016.10b.dat",
        "RML2018a": "dataset/GOLD_XYZ_OSC.0001_1024.hdf5",
    }
    default_sig = {"RML2016a": 128, "RML2016b": 128, "RML2018a": 1024}

    a = argparse.Namespace()
    a.dataset       = parent.dataset
    a.dataset_path  = parent.dataset_path or default_paths[parent.dataset]
    a.sig_size      = parent.sig_size or default_sig[parent.dataset]
    a.model_name    = model_name
    a.batch_size    = parent.batch_size
    a.lr            = parent.lr
    a.epochs        = parent.epochs
    a.num_workers   = parent.num_workers
    a.seed          = parent.seed
    a.samples_per_class_snr = parent.samples_per_class_snr
    a.save_model    = f"weights/{a.dataset}_{model_name}_baseline_best.pth"
    a.resume = parent.resume
    return a


# -------------------------------------------------------------
# 单模型训练 (调用 train.py 的 main)
# -------------------------------------------------------------
def run_one_model(parent, model_name):
    """
    成功 -> 返回 (True, json_path)
    失败 -> 返回 (False, traceback_str)
    """
    # 兼容: 若 JSON 已存在且未 --retrain, 直接复用
    json_path = f"results/{parent.dataset}_{model_name}_baseline_best.json"
    if os.path.exists(json_path) and not parent.retrain:
        print(f"  [SKIP] {model_name}: JSON exists -> reuse {json_path}")
        return True, json_path

    args = make_train_args(parent, model_name)
    try:
        # 延迟 import: 避免一开始就把所有模型加载进来
        from train import main as train_main
        train_main(args)
        return True, json_path
    except Exception:
        return False, traceback.format_exc()


# -------------------------------------------------------------
# 汇总: JSON / CSV / Markdown / 多模型 SNR 曲线对比
# -------------------------------------------------------------
def collect_results(dataset, model_list):
    """读各模型 JSON, 汇总成 list of dict"""
    rows = []
    for m in model_list:
        jp = f"results/{dataset}_{m}_baseline_best.json"
        if not os.path.exists(jp):
            print(f"  [WARN] missing {jp}")
            continue
        with open(jp, "r", encoding="utf-8") as f:
            d = json.load(f)
        c = d.get("complexity", {})
        s = d.get("source", {}).get("metrics", {})
        rows.append({
            "Model": m,
            "Params(M)": c.get("Parameters(M)", 0.0),
            "FLOPs(M)":  c.get("FLOPs(M)", 0.0),
            # 兼容新版 (per_sample_ms / latency_1f_ms) 与旧版 (inference_time(ms/sample))
            "PerSample(ms)": c.get("per_sample_ms",
                                   c.get("inference_time(ms/sample)", 0.0)),
            "Latency_1f(ms)": c.get("latency_1f_ms", 0.0),
            "Throughput(fps)": c.get("throughput_fps", 0.0),
            "Acc":      s.get("accuracy", 0.0),
            "F1":       s.get("f1", 0.0),
            "LowSNR":   s.get("low_snr", 0.0),
            "HighSNR":  s.get("high_snr", 0.0),
            # 取 SNR 曲线最高点
            "Peak":     max(d.get("source", {}).get("snr", [0.0])) if d.get("source", {}).get("snr") else 0.0,
            "_snr_full": d.get("source", {}).get("snr_full", []),
            "_snr_acc":  d.get("source", {}).get("snr",       []),
        })
    return rows


def write_markdown(rows, dataset, save_path):
    headers = [
        "Model", "Params(M)", "FLOPs(M)",
        "PerSample(ms)", "Latency_1f(ms)", "Throughput(fps)",
        "Acc", "F1", "LowSNR(-20~0)", "HighSNR", "Peak",
    ]
    lines = [
        f"# Summary on {dataset}",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| {Model} | {p:.4f} | {f:.2f} | {ps:.4f} | {l1:.4f} "
                     "| {tp:.1f} | {a:.4f} | {f1:.4f} | {lo:.4f} | {hi:.4f} | {pk:.4f} |".format(
            Model=r["Model"],
            p=r["Params(M)"], f=r["FLOPs(M)"],
            ps=r["PerSample(ms)"], l1=r["Latency_1f(ms)"], tp=r["Throughput(fps)"],
            a=r["Acc"], f1=r["F1"],
            lo=r["LowSNR"], hi=r["HighSNR"], pk=r["Peak"],
        ))
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(rows, save_path):
    import csv
    cols = ["Model", "Params(M)", "FLOPs(M)",
            "PerSample(ms)", "Latency_1f(ms)", "Throughput(fps)",
            "Acc", "F1", "LowSNR", "HighSNR", "Peak"]
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def plot_snr_compare(rows, dataset, save_path, highlight="TriNCFA_Net"):
    """所有模型的 SNR-Accuracy 曲线对比图"""
    try:
        from viz_metrics import setup_plot_style, plot_snr_curves_multi
    except ImportError:
        print("[WARN] viz_metrics.py not found, skip SNR compare plot")
        return
    setup_plot_style()
    snr_dict = {}
    for r in rows:
        if r["_snr_full"] and r["_snr_acc"]:
            snr_dict[r["Model"]] = (r["_snr_full"], r["_snr_acc"])
    if len(snr_dict) == 0:
        return
    plot_snr_curves_multi(
        snr_dict,
        title=f"SNR-Accuracy on {dataset}",
        save_path=save_path,
        highlight_key=highlight,
    )


# -------------------------------------------------------------
# 主流程
# -------------------------------------------------------------
def main():
    parent = parse_args()

    # 1) 决定要跑哪些模型
    target = list(parent.models) if parent.models else list(ALL_MODELS)
    target = [m for m in target if m not in parent.skip]
    invalid = [m for m in target if m not in ALL_MODELS]
    if invalid:
        print(f"[ERR] Unknown models: {invalid}")
        print(f"     Available: {ALL_MODELS}")
        sys.exit(1)

    print("=" * 64)
    print(f"  Dataset : {parent.dataset}")
    print(f"  Models  : {target}")
    print(f"  Epochs  : {parent.epochs}")
    print(f"  Retrain : {parent.retrain}")
    print("=" * 64)

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 2) 顺序训练
    log = []
    t_all0 = time.time()
    for i, m in enumerate(target, 1):
        print(f"\n[{i}/{len(target)}] === Train {m} on {parent.dataset} ===")
        t0 = time.time()
        ok, info = run_one_model(parent, m)
        elapsed = time.time() - t0
        if ok:
            print(f"  [OK]  {m} done in {elapsed/60:.1f} min  -> {info}")
            log.append({"model": m, "ok": True, "elapsed_min": elapsed / 60.0,
                        "json": info})
        else:
            print(f"  [FAIL] {m} after {elapsed/60:.1f} min")
            print(info)
            log.append({"model": m, "ok": False, "elapsed_min": elapsed / 60.0,
                        "error": info})
    total_min = (time.time() - t_all0) / 60.0
    print(f"\nAll training finished in {total_min:.1f} min.")

    # 3) 汇总
    print("\n=== Aggregating results ===")
    rows = collect_results(parent.dataset, [m for m in target
                                            if any(l["model"] == m and l["ok"]
                                                   for l in log)])
    summary = {
        "dataset":   parent.dataset,
        "epochs":    parent.epochs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_minutes": total_min,
        "log":  log,
        "rows": [{k: v for k, v in r.items()
                  if not k.startswith("_")} for r in rows],
    }
    sum_json = f"results/{parent.dataset}_summary.json"
    sum_md   = f"results/{parent.dataset}_summary.md"
    sum_csv  = f"results/{parent.dataset}_summary.csv"
    sum_png  = f"results/{parent.dataset}_snr_compare.png"

    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_markdown(rows, parent.dataset, sum_md)
    write_csv(rows, sum_csv)
    plot_snr_compare(rows, parent.dataset, sum_png, highlight="TriNCFA_Net")

    print(f"\nSummary saved:")
    print(f"  - {sum_json}")
    print(f"  - {sum_md}")
    print(f"  - {sum_csv}")
    print(f"  - {sum_png} (and .pdf)")

    # 4) 控制台打印 markdown
    print("\n" + "=" * 64)
    with open(sum_md, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
