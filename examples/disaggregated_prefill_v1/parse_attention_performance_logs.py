#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention性能日志解析脚本

该脚本用于解析vllm日志文件中的attention性能数据，
生成JSON格式的性能报告和热力图。

使用方法:
    python parse_attention_performance_logs.py <log_file>
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any
from collections import defaultdict

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("警告: 未安装matplotlib/seaborn/numpy，将跳过热力图生成")


def parse_log_file(log_file: str) -> List[Dict[str, Any]]:
    """解析vllm日志文件，提取性能数据"""
    performance_data = []

    # vllm日志中性能数据的正则表达式模式
    # 支持带ANSI颜色代码和进程信息的日志格式
    log_pattern = re.compile(
        r'.*INFO\s+(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}).*\[ATTENTION_PERF\]\|'
        r'dp_rank=(\d+)\|layer_index=(\d+)\|step=(\d+)\|'
        r'execution_time_ms=([\d.]+)\|timestamp=([\d.]+)'
    )

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '[ATTENTION_PERF]' not in line:
                    continue

                match = log_pattern.match(line)
                if match:
                    log_time, dp_rank, layer_index, step, exec_time, timestamp = match.groups()

                    data_point = {
                        "dp_rank": int(dp_rank),
                        "layer_index": int(layer_index),
                        "step": int(step),
                        "execution_time_ms": float(exec_time),
                        "timestamp": float(timestamp),
                        "log_time": log_time
                    }
                    performance_data.append(data_point)

    except Exception as e:
        print(f"错误: 解析文件 {log_file} 失败: {e}")

    return performance_data


def save_as_json(data: List[Dict[str, Any]], output_file: str) -> None:
    """保存为JSON格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存性能数据到 {output_file}, 共 {len(data)} 条记录")


def create_heatmap(data: List[Dict[str, Any]], output_file: str = "attention_performance_heatmap.png") -> None:
    """创建性能热力图"""
    if not HAS_PLOTTING:
        print("跳过热力图生成: 缺少必要的绘图库 (numpy, matplotlib, seaborn)")
        return

    # 按dp_rank和step分组，计算每个组合的总执行时间
    step_rank_times = defaultdict(lambda: defaultdict(float))

    for record in data:
        dp_rank = record["dp_rank"]
        step = record["step"]
        exec_time = record["execution_time_ms"]
        step_rank_times[step][dp_rank] += exec_time

    if not step_rank_times:
        print("没有数据可以绘制热力图")
        return

    # 获取所有的step和dp_rank
    all_steps = sorted(step_rank_times.keys())
    all_ranks = sorted(set(rank for step_data in step_rank_times.values() for rank in step_data.keys()))

    # 创建热力图数据矩阵
    heatmap_data = []
    for rank in all_ranks:
        row = []
        for step in all_steps:
            total_time = step_rank_times[step].get(rank, 0)
            row.append(total_time)
        heatmap_data.append(row)

    # 转换为numpy数组
    heatmap_data = np.array(heatmap_data)

    # 创建热力图
    plt.figure(figsize=(max(8, len(all_steps) * 0.8), max(6, len(all_ranks) * 0.5)))

    # 使用seaborn绘制热力图
    sns.heatmap(
        heatmap_data,
        xticklabels=[f"Step {s}" for s in all_steps],
        yticklabels=[f"DP Rank {r}" for r in all_ranks],
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Total Execution Time (ms)'}
    )

    plt.title('Attention Performance Heatmap\n(Total execution time per DP rank per step)')
    plt.xlabel('Step')
    plt.ylabel('DP Rank')
    plt.tight_layout()

    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"已生成热力图: {output_file}")
    print(f"热力图数据: {len(all_ranks)} 个DP ranks, {len(all_steps)} 个steps")


def main():
    parser = argparse.ArgumentParser(
        description="解析vllm日志中的Attention性能数据并生成热力图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("log_file", help="vllm日志文件路径")
    parser.add_argument("--output", "-o", default="attention_performance.json",
                       help="输出JSON文件路径 (默认: attention_performance.json)")
    parser.add_argument("--heatmap", default="attention_performance_heatmap.png",
                       help="热力图输出文件路径 (默认: attention_performance_heatmap.png)")

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.log_file):
        print(f"错误: 日志文件不存在: {args.log_file}")
        return

    # 解析日志
    print(f"开始解析日志文件: {args.log_file}")
    data = parse_log_file(args.log_file)

    if not data:
        print("未找到任何性能数据")
        print("请确认:")
        print("1. 设置了环境变量 VLLM_ATTENTION_PERF_MONITOR=true")
        print("2. 程序已运行并执行了attention层的forward方法")
        print("3. 日志文件中包含 [ATTENTION_PERF] 标记的行")
        return

    print(f"解析完成，共找到 {len(data)} 条性能记录")

    # 保存JSON数据
    save_as_json(data, args.output)

    # 生成热力图
    create_heatmap(data, args.heatmap)

    print("处理完成!")


if __name__ == "__main__":
    main()
