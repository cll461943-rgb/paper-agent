#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-1 双模型对比测试脚本
用法:
    python scripts/compare_models.py                # 默认 cases 1,2
    python scripts/compare_models.py --cases "1,3" --time-budget 90
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def load_dotenv(env_file: Path) -> None:
    """加载 .env 文件到当前进程环境"""
    if not env_file.exists():
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            os.environ.setdefault(k, v)


def run_eval(
    python: str,
    config: str,
    dataset: str,
    cases: str,
    time_budget: float,
    out_json: str,
    log_file: str,
    label: str,
) -> dict:
    """运行单次评估，返回指标字典"""
    cmd = [
        python, "evaluate.py",
        "--mode", "live",
        "--config", config,
        "-f", dataset,
        "-c", cases,
        "--time-budget", str(time_budget),
        "-o", out_json,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"  Config : {config}")
    print(f"  Output : {out_json}")
    print(f"  Cmd    : {' '.join(cmd)}")
    print(f"{'='*60}")

    with open(log_file, "w", encoding="utf-8") as lf:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            encoding="utf-8", errors="replace"
        )
        lf.write(result.stdout)

    # 实时打印日志
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)

    exit_code = result.returncode
    if exit_code != 0 or not Path(out_json).exists():
        print(f"  [ERROR] exit_code={exit_code}")
        return {"status": f"FAILED (exit={exit_code})", "f1": None, "recall": None, "precision": None}

    try:
        data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        return {
            "status": "OK",
            "f1":        round(float(summary.get("avg_f1_final", 0)), 4),
            "recall":    round(float(summary.get("avg_recall_final", 0)), 4),
            "precision": round(float(summary.get("avg_precision_final", 0)), 4),
            "wall_time": round(float(summary.get("avg_wall_time", 0)), 1),
        }
    except Exception as e:
        return {"status": f"PARSE_ERROR: {e}", "f1": None, "recall": None, "precision": None}


def fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "N/A"


def delta(a, b) -> str:
    if a is None or b is None:
        return "N/A"
    d = b - a
    return f"+{d:.4f}" if d >= 0 else f"{d:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 双模型对比测试")
    parser.add_argument("--cases", default="1,2", help="案例编号，如 '1,2' 或 '1,3'")
    parser.add_argument("--time-budget", type=float, default=90.0, help="每个 case 时间预算(秒)")
    parser.add_argument("--dataset", default="data/benchmarks/AutoScholarQuery_train.jsonl",
                        help="测试集路径")
    args = parser.parse_args()

    # 切到 paper-agent 根目录
    root = Path(__file__).parent.parent
    os.chdir(root)

    # 加载 .env
    load_dotenv(root / ".env")
    print(f"Working dir: {root}")

    # 验证测试集存在
    dataset = args.dataset
    if not Path(dataset).exists():
        print(f"[ERROR] Dataset not found: {dataset}", file=sys.stderr)
        sys.exit(1)

    # Python 解释器路径
    python = str(root / ".venv" / "Scripts" / "python.exe")
    if not Path(python).exists():
        python = sys.executable

    # 输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"outputs/compare_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 模型配置
    models = [
        {
            "label":    "DeepSeek (high_recall)",
            "config":   "configs/high_recall.yaml",
            "out_json": str(out_dir / "result_deepseek.json"),
            "log_file": str(out_dir / "run_deepseek.log"),
            "tag":      "deepseek",
        },
        {
            "label":    "GPT-5.5 (gpt55)",
            "config":   "configs/gpt55.yaml",
            "out_json": str(out_dir / "result_gpt55.json"),
            "log_file": str(out_dir / "run_gpt55.log"),
            "tag":      "gpt55",
        },
    ]

    # 顺序运行两个模型
    results: dict[str, dict] = {}
    for m in models:
        res = run_eval(
            python=python,
            config=m["config"],
            dataset=dataset,
            cases=args.cases,
            time_budget=args.time_budget,
            out_json=m["out_json"],
            log_file=m["log_file"],
            label=m["label"],
        )
        res["label"] = m["label"]
        results[m["tag"]] = res

    # 打印对比报告
    ds = results.get("deepseek", {})
    gp = results.get("gpt55", {})

    report_lines = [
        "# Phase-1 双模型对比报告",
        f"- 测试时间: {ts}",
        f"- 数据集: {dataset}",
        f"- Cases: {args.cases}",
        f"- Time Budget: {args.time_budget}s/case",
        "",
        "| 指标 | DeepSeek (high_recall) | GPT-5.5 (gpt55) | Delta (GPT-DS) |",
        "| :--- | :---: | :---: | :---: |",
        f"| F1        | {fmt(ds.get('f1'))} | {fmt(gp.get('f1'))} | {delta(ds.get('f1'), gp.get('f1'))} |",
        f"| Recall    | {fmt(ds.get('recall'))} | {fmt(gp.get('recall'))} | {delta(ds.get('recall'), gp.get('recall'))} |",
        f"| Precision | {fmt(ds.get('precision'))} | {fmt(gp.get('precision'))} | {delta(ds.get('precision'), gp.get('precision'))} |",
        f"| Wall Time | {fmt(ds.get('wall_time'))}s | {fmt(gp.get('wall_time'))}s | {delta(ds.get('wall_time'), gp.get('wall_time'))}s |",
        "",
        "## 运行状态",
        f"- DeepSeek : {ds.get('status', 'N/A')}",
        f"- GPT-5.5  : {gp.get('status', 'N/A')}",
        "",
        "## 日志文件",
        f"- DeepSeek : {out_dir}/run_deepseek.log",
        f"- GPT-5.5  : {out_dir}/run_gpt55.log",
    ]

    report_str = "\n".join(report_lines)
    report_path = out_dir / "compare_report.md"
    report_path.write_text(report_str, encoding="utf-8")

    print("\n" + "="*60)
    print("PHASE-1 COMPARISON REPORT")
    print("="*60)
    print(report_str)
    print(f"\nReport saved: {report_path}")
    print("="*60)


if __name__ == "__main__":
    main()
