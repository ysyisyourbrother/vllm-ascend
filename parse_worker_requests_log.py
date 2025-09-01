#!/usr/bin/env python3
"""
Parse worker request logs and generate stacked bar chart

Usage:
    python parse_worker_requests_log.py <log_file_path> [--output output.png]
"""

import re
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List
import numpy as np

def parse_log_file(log_file_path: str) -> Dict[int, List[int]]:
    """Parse log file and extract worker request information"""
    worker_requests = defaultdict(list)

    log_pattern = re.compile(r'\[WORKER_KV_SORTED\] Worker_(\d+): (.+)')

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = log_pattern.search(line)
                if match:
                    worker_id = int(match.group(1))
                    requests_str = match.group(2).strip()

                    if requests_str == "No requests found":
                        continue

                    # Parse request pairs: req_id1:length1,req_id2:length2,...
                    request_pairs = requests_str.split(',')
                    kv_lengths = []
                    for pair in request_pairs:
                        if ':' in pair:
                            _, kv_length_str = pair.split(':', 1)
                            try:
                                kv_lengths.append(int(kv_length_str))
                            except ValueError:
                                continue

                    # Sort by KV cache length (largest first, as they appear in log)
                    worker_requests[worker_id] = kv_lengths

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file_path}")
        return {}
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return {}

    return dict(worker_requests)

def create_stacked_bar_chart(worker_requests: Dict[int, List[int]], output_path: str):
    """Create stacked bar chart and save to file"""
    if not worker_requests:
        print("No worker request data found")
        return

    worker_ids = sorted(worker_requests.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.6
    x_positions = np.arange(len(worker_ids))

    # Simple color scheme
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', '#ff9ff3']

    # Draw stacked bars for each worker
    for i, worker_id in enumerate(worker_ids):
        kv_lengths = worker_requests[worker_id]
        bottom = 0

        for j, kv_length in enumerate(kv_lengths):
            color = colors[j % len(colors)]
            ax.bar(x_positions[i], kv_length, bar_width,
                  bottom=bottom, color=color,
                  edgecolor='white', linewidth=1,
                  alpha=0.8)
            bottom += kv_length

    # Set chart properties
    ax.set_xlabel('Worker ID')
    ax.set_ylabel('KV Cache Length (tokens)')
    ax.set_title('Worker Request Distribution (Bottom to Top: Large to Small KV Cache)')

    # Set x-axis labels - just show numbers
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(wid) for wid in worker_ids])

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Parse worker request logs and generate stacked bar chart')
    parser.add_argument('log_file', help='Log file path')
    parser.add_argument('--output', '-o', default='worker_requests_chart.png', help='Output image path (default: worker_requests_chart.png)')

    args = parser.parse_args()

    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    worker_requests = parse_log_file(args.log_file)

    if not worker_requests:
        print("No valid worker request data found")
        return

    # Print basic stats
    total_requests = sum(len(requests) for requests in worker_requests.values())
    print(f"Found {len(worker_requests)} workers with {total_requests} total requests")

    # Generate chart
    print("Generating stacked bar chart...")
    create_stacked_bar_chart(worker_requests, args.output)

if __name__ == '__main__':
    main()
