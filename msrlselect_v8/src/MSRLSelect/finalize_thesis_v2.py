#!/usr/bin/env python3
"""Create final v2 tables and an evidence-bound thesis experiment report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = []
    for directory in sorted((root / "chapter4").iterdir()):
        metrics_file = directory / "metrics_final.json"
        if not metrics_file.exists():
            continue
        metrics = json.loads(metrics_file.read_text())
        test = metrics["test"]
        valid = metrics["valid"]
        rows.append({
            "model": directory.name,
            "validation_auc": valid["fixed_0_5"]["auc"],
            "validation_auprc": valid["fixed_0_5"]["auprc"],
            "test_auc": test["fixed_0_5"]["auc"],
            "test_auprc": test["fixed_0_5"]["auprc"],
            "test_accuracy_fixed_0_5": test["fixed_0_5"]["accuracy"],
            "test_f1_fixed_0_5": test["fixed_0_5"]["f1"],
            "test_sensitivity_fixed_0_5": test["fixed_0_5"]["sensitivity"],
            "test_specificity_fixed_0_5": test["fixed_0_5"]["specificity"],
            "test_accuracy_validation_youden": test["validation_youden"]["accuracy"],
            "test_f1_validation_youden": test["validation_youden"]["f1"],
            "test_sensitivity_validation_youden": test["validation_youden"]["sensitivity"],
            "test_specificity_validation_youden": test["validation_youden"]["specificity"],
            "validation_youden_threshold": test["validation_youden"]["threshold"],
        })
    chapter4 = pd.DataFrame(rows).sort_values("model")
    chapter4.to_csv(root / "chapter4_results.csv", index=False)

    chapter3_dir = root / "chapter3"
    chapter3 = pd.read_csv(chapter3_dir / "chapter3_results.csv")
    stats = pd.read_csv(chapter3_dir / "chapter3_selection_statistics.csv")
    chapter3 = chapter3.merge(stats, on="method", how="left")
    chapter3.to_csv(root / "chapter3_results.csv", index=False)

    def metric(model, column):
        value = chapter4.loc[chapter4.model == model, column]
        return float(value.iloc[0]) if len(value) else float("nan")
    report = f'''# SurGen + HeMiCoRe 硕士论文基础实验矩阵报告（v2）

## 统一标签与评估定义

- `label 0`：Survived 5y；`label 1`：Died within 5y。
- 所有患者级预测中的连续正类概率字段为 `death_risk` / `prob_died_within_5y`，即五年内死亡风险。
- AUROC、AUPRC、F1、Sensitivity 均以 label 1 为阳性；Specificity 为正确识别五年生存患者的比例。
- AUROC/AUPRC 直接用连续死亡风险计算。
- 固定 0.5 与 Validation 最大 Youden 阈值两组分类指标均已保存。Youden 阈值只在 Validation 确定，原样应用于 Test。

## 数据与实验规则

固定 SR386 划分：Train 255、Validation 84、Test 84；没有更改划分或重新提取 UNI 特征。所有实验使用固定 seed 1。每个新训练模型仅使用 Train 训练、根据 Validation AUROC 选择 checkpoint，之后一次性评估 Test。

## 第三章：区域选择对比

完整结果：`chapter3_results.csv`；选择统计：`chapter3_selection_statistics.csv`；不同决策步骤结果：`chapter3_step_metrics.csv`；患者级预测位于 `chapter3/predictors/`；选择轨迹与二维坐标图位于 `chapter3/visualizations/`。

- Random、HeMiCoRe、固定预算、自适应预算、去空间和去证据奖励均通过相同的下游 MLP、相同划分和同一评估脚本比较。
- HeMiCoRe 选择来自冻结的原始 Stage-3 checkpoint；其空间超图与 PPO 推理路径未被改写。
- 第三章的结论必须以 `chapter3_results.csv` 的单 seed 数字为准：该表可以回答 MS-RLSelect 相对随机/HeMiCoRe、空间项、证据奖励以及自适应预算的方向性差异，但不能支持统计显著性结论。
- 原始 HeMiCoRe 的动作是连续簇内采样比例，不是与 MS-RLSelect 完全同构的离散“选/不选区域”动作；重复率和区域数应据此解释。

## 第四章：病例级五年死亡风险预测

完整结果：`chapter4_results.csv`；每个模型目录包含患者级 CSV、ROC、PR、Validation 阈值记录、最终指标与 checkpoint。

当前 Test AUROC/AUPRC（连续死亡风险）包括：

- Mean Pooling + MLP：{metric('mean_mlp','test_auc'):.4f} / {metric('mean_mlp','test_auprc'):.4f}
- Global-only：{metric('global_only','test_auc'):.4f} / {metric('global_only','test_auprc'):.4f}
- Local-only：{metric('local_only','test_auc'):.4f} / {metric('local_only','test_auprc'):.4f}
- Simple Concat：{metric('simple_concat','test_auc'):.4f} / {metric('simple_concat','test_auprc'):.4f}
- Gated Fusion：{metric('gated_fusion','test_auc'):.4f} / {metric('gated_fusion','test_auprc'):.4f}

`attention_mil` 是本轮真实 patch-level Attention MIL；`attention_interface_legacy` 仅为历史接口输出，不应作为 Attention MIL 对照使用。

## 可写入论文的结论与边界

1. 已完成数据审计、统一标签/阈值协议、六种区域选择方法的单 seed 对比、选择统计/轨迹、基础预测模型矩阵和真实 Attention MIL。
2. Local-only 弱于 Global-only 时，合理解释是局部选择压缩了病例级背景和全局肿瘤异质性；是否存在融合增益应根据同表的 Fusion 与 Concat 行客观陈述。
3. 门控融合与简单融合、空间项与证据奖励、自适应与固定预算的优劣均应基于当前单 seed Test 表述为“本实验观察”，不写成稳定性或显著性结论。
4. 多 seed、置信区间、显著性检验、外部队列和更大结构改进属于下一阶段，未在本阶段执行。
'''
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
