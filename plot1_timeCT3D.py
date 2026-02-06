#!/usr/bin/env python3
"""
plot1_time_CT3D.py: Timing and quality visualization for 3D Coordinated Turn computation time comparison
from main1_timeCT3D.py. Handles 5 filters (EKF + 4 DR methods).
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
FILTERS_ORDER = ['EKF', 'DR_EKF_CDC', 'DR_EKF_CDC_FW', 'DR_EKF_TAC', 'DR_EKF_TAC_FW']

DISPLAY_NAMES = {
    'EKF': 'EKF',
    'DR_EKF_CDC': 'CDC-MOSEK',
    'DR_EKF_CDC_FW': 'CDC-FW',
    'DR_EKF_TAC': 'TAC-MOSEK',
    'DR_EKF_TAC_FW': 'TAC-FW',
}

COLORS = {
    'EKF': '#1f77b4',
    'DR_EKF_CDC': '#2ca02c',
    'DR_EKF_CDC_FW': '#ff7f0e',
    'DR_EKF_TAC': '#d62728',
    'DR_EKF_TAC_FW': '#9467bd',
}

MARKERS = {
    'EKF': 'D',
    'DR_EKF_CDC': 'o',
    'DR_EKF_CDC_FW': 's',
    'DR_EKF_TAC': 'v',
    'DR_EKF_TAC_FW': '^',
}

RESULTS_DIR = os.path.join("results", "timing_comparison_CT3D_cpp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def display_name(filter_name):
    return DISPLAY_NAMES.get(filter_name, filter_name)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_timing_data(results_path, dist):
    """Load saved timing results from main1_timeCT3D.py."""
    quality_file = os.path.join(results_path, f'timing_quality_results_all_theta_{dist}.pkl')
    timing_file = os.path.join(results_path, f'timing_results_all_theta_{dist}.pkl')

    if os.path.exists(quality_file):
        print(f"Loading quality results from: {quality_file}")
        with open(quality_file, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(timing_file):
        print(f"Loading timing results from: {timing_file}")
        with open(timing_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No results file found in {results_path}")


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_timing_vs_theta(all_results):
    """Extract timing data organised by filter for plotting."""
    timing_data = {filt: {
        'theta': [], 'time_mean': [], 'time_std': [],
        'freq_mean': [], 'freq_std': []
    } for filt in FILTERS_ORDER}

    theta_vals = sorted(all_results.keys())

    for theta in theta_vals:
        theta_results = all_results[theta]
        for filt in FILTERS_ORDER:
            if filt in theta_results:
                timing_data[filt]['theta'].append(theta)
                timing_data[filt]['time_mean'].append(theta_results[filt]['avg_time_per_step_ms'])
                timing_data[filt]['time_std'].append(theta_results[filt]['std_time_per_step_ms'])
                timing_data[filt]['freq_mean'].append(theta_results[filt]['frequency_hz'])

                time_s = theta_results[filt]['avg_time_per_step_ms'] / 1000.0
                freq_std = (theta_results[filt]['std_time_per_step_ms'] / 1000.0) / (time_s ** 2) if time_s > 0 else 0
                timing_data[filt]['freq_std'].append(freq_std)

    return timing_data, theta_vals


def extract_rmse_vs_theta(all_results):
    """Extract RMSE data organised by filter for plotting."""
    rmse_data = {filt: {
        'theta': [], 'rmse_mean': [], 'rmse_std': []
    } for filt in FILTERS_ORDER}

    theta_vals = sorted(all_results.keys())

    for theta in theta_vals:
        theta_results = all_results[theta]
        for filt in FILTERS_ORDER:
            if filt in theta_results and 'avg_rmse' in theta_results[filt]:
                rmse_data[filt]['theta'].append(theta)
                rmse_data[filt]['rmse_mean'].append(theta_results[filt]['avg_rmse'])
                rmse_data[filt]['rmse_std'].append(theta_results[filt]['std_rmse'])

    return rmse_data, theta_vals


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def create_timing_plot(timing_data, dist):
    """Single plot showing computation time vs theta for all filters."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for filt in FILTERS_ORDER:
        data = timing_data[filt]
        if len(data['theta']) > 0:
            ax.errorbar(data['theta'], data['time_mean'], yerr=data['time_std'],
                       marker=MARKERS[filt], linewidth=2, markersize=8, capsize=4,
                       color=COLORS[filt], label=display_name(filt))

    ax.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax.set_ylabel('Computation Time (ms/step)', fontsize=14)
    ax.set_title('Computation Time vs Robustness Parameter (3D CT)', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    _ensure_dir(RESULTS_DIR)
    save_path = os.path.join(RESULTS_DIR, f"timing_vs_theta_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Timing plot saved to: {save_path}")


def create_frequency_plot(timing_data, dist):
    """Single plot showing update frequency vs theta for all filters."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for filt in FILTERS_ORDER:
        data = timing_data[filt]
        if len(data['theta']) > 0:
            ax.errorbar(data['theta'], data['freq_mean'], yerr=data['freq_std'],
                       marker=MARKERS[filt], linewidth=2, markersize=8, capsize=4,
                       color=COLORS[filt], label=display_name(filt))

    ax.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax.set_ylabel('Update Frequency (Hz)', fontsize=14)
    ax.set_title('Update Frequency vs Robustness Parameter (3D CT)', fontsize=16)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    _ensure_dir(RESULTS_DIR)
    save_path = os.path.join(RESULTS_DIR, f"frequency_vs_theta_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Frequency plot saved to: {save_path}")


def create_rmse_plot(rmse_data, dist):
    """Single plot showing RMSE vs theta for all filters."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for filt in FILTERS_ORDER:
        data = rmse_data[filt]
        if len(data['theta']) > 0:
            ax.errorbar(data['theta'], data['rmse_mean'], yerr=data['rmse_std'],
                       marker=MARKERS[filt], linewidth=2, markersize=8, capsize=4,
                       color=COLORS[filt], label=display_name(filt))

    ax.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_title('State Estimation RMSE vs Robustness Parameter (3D CT)', fontsize=16)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    _ensure_dir(RESULTS_DIR)
    save_path = os.path.join(RESULTS_DIR, f"rmse_vs_theta_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"RMSE plot saved to: {save_path}")


def create_speedup_bar_chart(all_results, dist):
    """Bar chart: speedup of each DR filter relative to EKF."""
    theta_vals = sorted(all_results.keys())
    dr_filters = [f for f in FILTERS_ORDER if f != 'EKF']

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(theta_vals))
    width = 0.18

    for i, filt in enumerate(dr_filters):
        speedups = []
        for theta in theta_vals:
            ekf_time = all_results[theta].get('EKF', {}).get('avg_time_per_step_ms', 0)
            filt_time = all_results[theta].get(filt, {}).get('avg_time_per_step_ms', 0)
            speedups.append(ekf_time / filt_time if filt_time > 0 else 0)

        bars = ax.bar(x + i * width, speedups, width, label=display_name(filt), color=COLORS[filt])

        for bar, sp in zip(bars, speedups):
            if sp > 0:
                ax.annotate(f'{sp:.2f}x',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 2), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Robustness Parameter θ', fontsize=14)
    ax.set_ylabel('Speedup vs EKF (×)', fontsize=14)
    ax.set_title('Computation Speedup Relative to EKF (3D CT)', fontsize=16)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{t}' for t in theta_vals], fontsize=12)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    _ensure_dir(RESULTS_DIR)
    save_path = os.path.join(RESULTS_DIR, f"speedup_bar_chart_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Speedup bar chart saved to: {save_path}")


def create_time_vs_rmse_scatter(all_results, dist):
    """Scatter plot: Time vs RMSE trade-off at middle theta."""
    theta_vals = sorted(all_results.keys())
    mid_theta = theta_vals[len(theta_vals) // 2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f'Time vs RMSE Trade-off (θ = {mid_theta})', fontsize=16)
    ax.set_xlabel('Computation Time (ms/step)', fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    for filt in FILTERS_ORDER:
        fdata = all_results[mid_theta].get(filt, {})
        if 'avg_rmse' in fdata:
            ax.scatter(fdata['avg_time_per_step_ms'], fdata['avg_rmse'],
                      s=200, marker=MARKERS[filt],
                      color=COLORS[filt], label=display_name(filt), zorder=5)

    ax.legend(fontsize=12, loc='best')
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    _ensure_dir(RESULTS_DIR)
    save_path = os.path.join(RESULTS_DIR, f"time_vs_rmse_tradeoff_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Time-vs-RMSE scatter saved to: {save_path}")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def print_timing_summary(all_results):
    """Print summary table of timing results."""
    theta_vals = sorted(all_results.keys())

    print(f"\n{'=' * 120}")
    print("COMPUTATION TIME (ms/step)")
    print(f"{'=' * 120}")
    header = f"{'theta':<10}"
    for filt in FILTERS_ORDER:
        header += f" {display_name(filt):<22}"
    print(header)
    print("-" * 120)

    for theta in theta_vals:
        row = f"{theta:<10.2f}"
        for filt in FILTERS_ORDER:
            d = all_results[theta].get(filt, {})
            if d:
                row += f" {d['avg_time_per_step_ms']:>8.3f}±{d['std_time_per_step_ms']:<8.3f}    "
            else:
                row += f" {'N/A':<22}"
        print(row)
    print(f"{'=' * 120}")


def print_rmse_summary(all_results):
    """Print RMSE summary."""
    theta_vals = sorted(all_results.keys())
    sample_theta = theta_vals[0]
    if 'avg_rmse' not in all_results[sample_theta].get('EKF', {}):
        print("\nRMSE data not available in results")
        return

    print(f"\n{'=' * 120}")
    print("RMSE")
    print(f"{'=' * 120}")
    header = f"{'theta':<10}"
    for filt in FILTERS_ORDER:
        header += f" {display_name(filt):<22}"
    print(header)
    print("-" * 120)

    for theta in theta_vals:
        row = f"{theta:<10.2f}"
        for filt in FILTERS_ORDER:
            d = all_results[theta].get(filt, {})
            if d and 'avg_rmse' in d:
                row += f" {d['avg_rmse']:>8.4f}±{d['std_rmse']:<8.4f}    "
            else:
                row += f" {'N/A':<22}"
        print(row)
    print(f"{'=' * 120}")


def print_speedup_summary(all_results):
    """Print speedup vs EKF."""
    theta_vals = sorted(all_results.keys())
    dr_filters = [f for f in FILTERS_ORDER if f != 'EKF']

    print(f"\n{'=' * 100}")
    print("SPEEDUP vs EKF")
    print(f"{'=' * 100}")
    header = f"{'theta':<10}"
    for filt in dr_filters:
        header += f" {display_name(filt):<22}"
    print(header)
    print("-" * 100)

    for theta in theta_vals:
        row = f"{theta:<10.2f}"
        ekf_time = all_results[theta].get('EKF', {}).get('avg_time_per_step_ms', 0)
        for filt in dr_filters:
            d = all_results[theta].get(filt, {})
            if d and ekf_time > 0:
                sp = ekf_time / d['avg_time_per_step_ms']
                row += f" {sp:>20.2f}x "
            else:
                row += f" {'N/A':<22}"
        print(row)
    print(f"{'=' * 100}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Main plotting routine."""
    parser = argparse.ArgumentParser(description='Plot timing comparison results (C++ accelerated)')
    parser.add_argument('--dist', type=str, default='normal',
                       help='Distribution type (normal or quadratic)')
    args = parser.parse_args()

    results_path = f"./{RESULTS_DIR}/"

    print("=" * 80)
    print("TIMING VISUALIZATION (3D Coordinated Turn) - C++ Accelerated")
    print("=" * 80)
    print(f"Distribution: {args.dist}")
    print(f"Results path: {results_path}")
    print(f"Filters: {FILTERS_ORDER}")
    print("=" * 80)

    # Load
    print("\nLoading results data...")
    all_results = load_timing_data(results_path, args.dist)

    # Extract
    print("Extracting timing data...")
    timing_data, theta_vals = extract_timing_vs_theta(all_results)
    print(f"Found {len(theta_vals)} theta values: {theta_vals}")

    print("Extracting RMSE data...")
    rmse_data, _ = extract_rmse_vs_theta(all_results)

    # Summary tables
    print_timing_summary(all_results)
    print_rmse_summary(all_results)
    print_speedup_summary(all_results)

    # Plots
    print("\nCreating plots...")
    create_timing_plot(timing_data, args.dist)
    create_frequency_plot(timing_data, args.dist)
    create_rmse_plot(rmse_data, args.dist)
    create_speedup_bar_chart(all_results, args.dist)
    create_time_vs_rmse_scatter(all_results, args.dist)

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {RESULTS_DIR}/")
    print("Generated files:")
    print(f"  - timing_vs_theta_{args.dist}.pdf")
    print(f"  - frequency_vs_theta_{args.dist}.pdf")
    print(f"  - rmse_vs_theta_{args.dist}.pdf")
    print(f"  - speedup_bar_chart_{args.dist}.pdf")
    print(f"  - time_vs_rmse_tradeoff_{args.dist}.pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
