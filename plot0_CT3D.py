#!/usr/bin/env python3
"""
plot0_CT3D.py: Visualization for EKF vs DR-EKF comparison (3D CT)
from main0_CT3D.py. Handles 4 filters (EKF + 3 DR methods).
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FILTERS_ORDER = ['EKF', 'DR_EKF_CDC', 'DR_EKF_CDC_FW', 'DR_EKF_TAC', 'DR_EKF_TAC_FW']
RESULTS_DIR = os.path.join("results", "EKF_comparison_CT3D_cpp")

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def display_name(filter_name):
    return DISPLAY_NAMES.get(filter_name, filter_name)


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(results_path, dist):
    """Load optimal_results and all_results dicts."""
    optimal_file = os.path.join(results_path, f'optimal_results_{dist}.pkl')
    all_results_file = os.path.join(results_path, f'all_results_{dist}.pkl')
    if not os.path.exists(optimal_file):
        raise FileNotFoundError(f"Not found: {optimal_file}")
    with open(optimal_file, 'rb') as f:
        optimal_results = pickle.load(f)
    all_results = None
    if os.path.exists(all_results_file):
        with open(all_results_file, 'rb') as f:
            all_results = pickle.load(f)
    return optimal_results, all_results


def load_detailed_results_for_filter(results_path, filter_name, theta_vals, dist):
    """Load detailed trajectory data for *filter_name* with its optimal theta."""
    if filter_name == 'EKF':
        filename = f'detailed_results_{filter_name}_{dist}.pkl'
    elif filter_name.startswith('DR_EKF_CDC'):
        tx, tv = theta_vals['theta_x'], theta_vals['theta_v']
        filename = f'detailed_results_{filter_name}_tx{tx}_tv{tv}_{dist}.pkl'
    elif filter_name.startswith('DR_EKF_TAC'):
        tx, tv, tw = theta_vals['theta_x'], theta_vals['theta_v'], theta_vals['theta_w']
        filename = f'detailed_results_{filter_name}_tx{tx}_tv{tv}_tw{tw}_{dist}.pkl'
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    path = os.path.join(results_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------
def extract_trajectory_data(optimal_results, results_path, dist):
    """Return {filter_name: {...trajectory stats...}} for every available filter."""
    trajectory_data = {}

    for filt, stats in optimal_results.items():
        if filt == 'EKF':
            theta_vals, theta_str = {}, "N/A"
        elif filt.startswith('DR_EKF_TAC'):
            theta_vals = {'theta_x': stats['theta_x'], 'theta_v': stats['theta_v'],
                          'theta_w': stats['theta_w']}
            theta_str = (f"\u03b8_x={stats['theta_x']}, \u03b8_v={stats['theta_v']}, "
                         f"\u03b8_w={stats['theta_w']}")
        elif filt.startswith('DR_EKF_CDC'):
            theta_vals = {'theta_x': stats['theta_x'], 'theta_v': stats['theta_v']}
            theta_str = f"\u03b8_x={stats['theta_x']}, \u03b8_v={stats['theta_v']}"
        else:
            continue

        try:
            detailed = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)
            if filt not in detailed:
                continue
            sim_results = detailed[filt]['results']

            est = [np.squeeze(r['est_state_traj'], axis=-1) for r in sim_results]
            true = [np.squeeze(r['state_traj'], axis=-1) for r in sim_results]

            if est:
                ea, ta = np.array(est), np.array(true)
                trajectory_data[filt] = {
                    'mean': np.mean(ea, axis=0),
                    'std': np.std(ea, axis=0),
                    'theta_vals': theta_vals,
                    'theta_str': theta_str,
                    'true_mean': np.mean(ta, axis=0),
                    'num_sims': len(est),
                }
                print(f"  Loaded {len(est)} trajectories for {filt}")
        except Exception as e:
            print(f"  Could not load data for {filt}: {e}")

    return trajectory_data


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_single_trajectory_3d(ax, tdata, color, label, linestyle='-', lw=2.5,
                                show_true=True):
    """Plot estimated 3D trajectory on *ax*."""
    mean = tdata['mean']
    x_m, y_m, z_m = mean[:, 0], mean[:, 1], mean[:, 2]
    ax.plot(x_m, y_m, z_m, linestyle, color=color, linewidth=lw, label=label)
    if show_true and 'true_mean' in tdata:
        tm = tdata['true_mean']
        ax.plot(tm[:, 0], tm[:, 1], tm[:, 2], ':', color='red', linewidth=1.5, alpha=0.6)
        ax.scatter(tm[0, 0], tm[0, 1], tm[0, 2], marker='X', s=60, color='red', zorder=5)
        ax.scatter(tm[-1, 0], tm[-1, 1], tm[-1, 2], marker='X', s=60, color='red', zorder=5)


def _plot_single_trajectory_2d(ax, tdata, color, label, linestyle='-', lw=2.5,
                                show_true=True):
    """Plot estimated 2D (XY projection) trajectory on *ax*."""
    mean = tdata['mean']
    x_m, y_m = mean[:, 0], mean[:, 1]
    ax.plot(x_m, y_m, linestyle, color=color, linewidth=lw, label=label)
    if show_true and 'true_mean' in tdata:
        tm = tdata['true_mean']
        ax.plot(tm[:, 0], tm[:, 1], ':', color='red', linewidth=1.5, alpha=0.6)
        ax.scatter(tm[0, 0], tm[0, 1], marker='X', s=60, color='red', zorder=5)
        ax.scatter(tm[-1, 0], tm[-1, 1], marker='X', s=60, color='red', zorder=5)


# ---------------------------------------------------------------------------
# 1) Individual trajectory plots (3D)
# ---------------------------------------------------------------------------
def plot_individual_trajectories(trajectory_data, dist):
    """One PDF per filter - 3D trajectory."""
    _ensure_dir(RESULTS_DIR)
    saved = []
    for filt, tdata in trajectory_data.items():
        color = COLORS.get(filt, '#888888')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        _plot_single_trajectory_3d(ax, tdata, color, display_name(filt))

        ax.set_xlabel('X position', fontsize=16)
        ax.set_ylabel('Y position', fontsize=16)
        ax.set_zlabel('Z position', fontsize=16)
        title = display_name(filt)
        if tdata['theta_str'] != "N/A":
            title += f"\n({tdata['theta_str']})"
        ax.set_title(title, fontsize=18, pad=12)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=14, loc='best')
        plt.tight_layout()

        fname = f"traj_3d_{filt}_{dist}.pdf"
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300, bbox_inches='tight')
        saved.append(fname)
        plt.close(fig)

    print(f"Saved {len(saved)} individual 3D trajectory plots to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# 2) Combined trajectory subplot (all filters in one figure, 3D)
# ---------------------------------------------------------------------------
def plot_combined_trajectories(trajectory_data, dist):
    """All filters in one figure with 3D subplots."""
    _ensure_dir(RESULTS_DIR)

    available_filters = [f for f in FILTERS_ORDER if f in trajectory_data]
    if not available_filters:
        print("No trajectory data available - skipping combined plot.")
        return

    n_filters = len(available_filters)
    n_cols = min(2, n_filters)
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(8 * n_cols, 7 * n_rows))

    for idx, filt in enumerate(available_filters):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        tdata = trajectory_data[filt]
        color = COLORS.get(filt, '#888888')

        _plot_single_trajectory_3d(ax, tdata, color, display_name(filt), show_true=True)

        ax.set_title(display_name(filt), fontsize=14, pad=8)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)
        ax.tick_params(labelsize=9)

    # Shared legend
    legend_els = [
        plt.Line2D([0], [0], color='red', ls=':', lw=1.5, label='True trajectory'),
        plt.Line2D([0], [0], color='gray', lw=2, label='Estimated'),
    ]
    fig.legend(handles=legend_els, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    fname = f"traj_3d_combined_{dist}.pdf"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined 3D trajectory figure: {RESULTS_DIR}/{fname}")


# ---------------------------------------------------------------------------
# 2b) Combined 2D XY-projection trajectory subplot
# ---------------------------------------------------------------------------
def plot_combined_trajectories_2d(trajectory_data, dist):
    """All filters in one figure with 2D XY-projection subplots."""
    _ensure_dir(RESULTS_DIR)

    available_filters = [f for f in FILTERS_ORDER if f in trajectory_data]
    if not available_filters:
        print("No trajectory data available - skipping combined 2D plot.")
        return

    n_filters = len(available_filters)
    n_cols = min(3, n_filters)
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    for idx, filt in enumerate(available_filters):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        tdata = trajectory_data[filt]
        color = COLORS.get(filt, '#888888')

        _plot_single_trajectory_2d(ax, tdata, color, display_name(filt), show_true=True)

        ax.set_title(display_name(filt), fontsize=14, pad=8)
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots
    for idx in range(n_filters, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    legend_els = [
        plt.Line2D([0], [0], color='red', ls=':', lw=1.5, label='True trajectory'),
        plt.Line2D([0], [0], color='gray', lw=2, label='Estimated'),
    ]
    fig.legend(handles=legend_els, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=2, fontsize=12, frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    fname = f"traj_2d_combined_{dist}.pdf"
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined 2D trajectory figure: {RESULTS_DIR}/{fname}")


# ---------------------------------------------------------------------------
# 3) MSE heatmaps for CDC filters (theta_x vs theta_v)
# ---------------------------------------------------------------------------
def plot_mse_heatmaps(all_results, dist):
    """Create heatmap figures for CDC filters."""
    if all_results is None:
        print("No all_results - skipping heatmaps.")
        return
    _ensure_dir(RESULTS_DIR)

    # CDC-based filters (theta_x x theta_v)
    cdc_filters = [f for f in FILTERS_ORDER if f.startswith('DR_EKF_CDC')]

    for filt in cdc_filters:
        if filt not in all_results or not all_results[filt]:
            continue

        data = {}
        for theta_key, res in all_results[filt].items():
            if isinstance(theta_key, tuple) and len(theta_key) == 2:
                data[theta_key] = res['mse_mean']

        if not data:
            continue

        tx_vals = sorted(set(k[0] for k in data.keys()))
        tv_vals = sorted(set(k[1] for k in data.keys()))

        mat = np.full((len(tv_vals), len(tx_vals)), np.nan)
        for i, tv in enumerate(tv_vals):
            for j, tx in enumerate(tx_vals):
                mat[i, j] = data.get((tx, tv), np.nan)

        valid = mat[~np.isnan(mat)]
        if len(valid) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower',
                       norm=LogNorm(vmin=max(valid.min(), 1e-8), vmax=valid.max()))
        ax.set_xticks(np.arange(len(tx_vals)))
        ax.set_yticks(np.arange(len(tv_vals)))
        ax.set_xticklabels([f'{v:.2f}' for v in tx_vals], fontsize=11)
        ax.set_yticklabels([f'{v:.2f}' for v in tv_vals], fontsize=11)
        ax.set_xlabel('\u03b8_x', fontsize=14)
        ax.set_ylabel('\u03b8_v', fontsize=14)
        ax.set_title(f'MSE Heatmap: {display_name(filt)}', fontsize=16, pad=10)

        for i in range(len(tv_vals)):
            for j in range(len(tx_vals)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                            color='w', fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"mse_heatmap_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved heatmap: {save_path}")

    # TAC-based filters (theta_v x theta_w, theta_x fixed)
    tac_filters = [f for f in FILTERS_ORDER if f.startswith('DR_EKF_TAC')]

    for filt in tac_filters:
        if filt not in all_results or not all_results[filt]:
            continue

        data = {}
        fixed_tx = None
        for theta_key, res in all_results[filt].items():
            if isinstance(theta_key, tuple) and len(theta_key) == 3:
                tx, tv, tw = theta_key
                if fixed_tx is None:
                    fixed_tx = tx
                data[(tv, tw)] = res['mse_mean']

        if not data:
            continue

        tv_vals = sorted(set(k[0] for k in data.keys()))
        tw_vals = sorted(set(k[1] for k in data.keys()))

        mat = np.full((len(tw_vals), len(tv_vals)), np.nan)
        for i, tw in enumerate(tw_vals):
            for j, tv in enumerate(tv_vals):
                mat[i, j] = data.get((tv, tw), np.nan)

        valid = mat[~np.isnan(mat)]
        if len(valid) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, aspect='auto', cmap='viridis', origin='lower',
                       norm=LogNorm(vmin=max(valid.min(), 1e-8), vmax=valid.max()))
        ax.set_xticks(np.arange(len(tv_vals)))
        ax.set_yticks(np.arange(len(tw_vals)))
        ax.set_xticklabels([f'{v:.2f}' for v in tv_vals], fontsize=11)
        ax.set_yticklabels([f'{v:.2f}' for v in tw_vals], fontsize=11)
        ax.set_xlabel('\u03b8_v', fontsize=14)
        ax.set_ylabel('\u03b8_w', fontsize=14)
        ax.set_title(f'MSE Heatmap: {display_name(filt)} (\u03b8_x={fixed_tx} fixed)', fontsize=16, pad=10)

        for i in range(len(tw_vals)):
            for j in range(len(tv_vals)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                            color='w', fontsize=10)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"mse_heatmap_{filt}_{dist}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved heatmap: {save_path}")


# ---------------------------------------------------------------------------
# 4) Violin plots - MSE distribution at optimal parameters
# ---------------------------------------------------------------------------
def create_violin_plots(optimal_results, results_path, dist):
    """Violin plots of MSE for all filters at optimal parameters."""
    _ensure_dir(RESULTS_DIR)
    print("Creating violin plots...")

    v_data = []
    v_labels = []
    v_colors = []

    for filt in FILTERS_ORDER:
        if filt not in optimal_results:
            continue

        stats = optimal_results[filt]
        if filt == 'EKF':
            theta_vals = {}
        elif filt.startswith('DR_EKF_TAC'):
            theta_vals = {'theta_x': stats['theta_x'], 'theta_v': stats['theta_v'],
                          'theta_w': stats['theta_w']}
        elif filt.startswith('DR_EKF_CDC'):
            theta_vals = {'theta_x': stats['theta_x'], 'theta_v': stats['theta_v']}
        else:
            continue

        try:
            detailed = load_detailed_results_for_filter(results_path, filt, theta_vals, dist)
            if filt not in detailed:
                continue
            raw = [np.mean(r['mse']) for r in detailed[filt]['results']]
            if not raw:
                continue
        except Exception:
            continue

        v_data.append(raw)
        v_labels.append(display_name(filt))
        v_colors.append(COLORS.get(filt, '#888888'))

    if not v_data:
        print("No data for violin plots - skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    positions = np.arange(1, len(v_data) + 1)

    parts = ax.violinplot(v_data, positions=positions, showmeans=True, showmedians=True, widths=0.75)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(v_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.8)
    for pname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if pname in parts:
            parts[pname].set_edgecolor('black')
            parts[pname].set_linewidth(1.2)

    ax.set_xticks(positions)
    ax.set_xticklabels(v_labels, fontsize=12, rotation=15, ha='right')
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_title(f'MSE Distribution at Optimal Parameters - 3D CT ({dist.title()})', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, which='major', ls='--', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"violin_plot_mse_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved violin plot: {save_path}")


# ---------------------------------------------------------------------------
# 5) MSE bar chart comparing all filters
# ---------------------------------------------------------------------------
def plot_mse_bar_chart(optimal_results, dist):
    """Bar chart comparing MSE across all filters at optimal parameters."""
    _ensure_dir(RESULTS_DIR)

    filters_with_data = [f for f in FILTERS_ORDER if f in optimal_results]
    if not filters_with_data:
        return

    mse_vals = [optimal_results[f]['mse_mean'] for f in filters_with_data]
    colors = [COLORS.get(f, '#888888') for f in filters_with_data]
    labels = [display_name(f) for f in filters_with_data]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(filters_with_data))
    bars = ax.bar(x, mse_vals, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, rotation=15, ha='right')
    ax.set_ylabel('MSE (at optimal \u03b8)', fontsize=14)
    ax.set_title(f'MSE Comparison at Optimal Parameters - 3D CT ({dist.title()})', fontsize=16)
    ax.grid(True, axis='y', ls='--', alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f"mse_bar_chart_{dist}.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved MSE bar chart: {save_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_optimal_results_summary(optimal_results):
    print("\nOptimal Results Summary:")
    print("=" * 100)
    print(f"{'Filter':<20} {'Optimal Theta':<50} {'MSE':>15}")
    print("-" * 100)
    for filt in FILTERS_ORDER:
        if filt not in optimal_results:
            continue
        res = optimal_results[filt]
        mse_mean = res.get('mse_mean', float('nan'))
        mse_std = res.get('mse_std', float('nan'))
        if filt == 'EKF':
            theta_str = "N/A"
        elif filt.startswith('DR_EKF_TAC'):
            theta_str = (f"\u03b8_x={res.get('theta_x','?')}, "
                         f"\u03b8_v={res.get('theta_v','?')}, "
                         f"\u03b8_w={res.get('theta_w','?')}")
        elif filt.startswith('DR_EKF_CDC'):
            theta_str = (f"\u03b8_x={res.get('theta_x','?')}, "
                         f"\u03b8_v={res.get('theta_v','?')}")
        else:
            theta_str = "?"
        print(f"{display_name(filt):<20} {theta_str:<50} {mse_mean:.6f}\u00b1{mse_std:.6f}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Visualise EKF vs DR-EKF comparison results (C++ accelerated, 3D CT)')
    parser.add_argument('--dist', default='normal', choices=['normal', 'quadratic'])
    args = parser.parse_args()

    results_path = f"./{RESULTS_DIR}/"

    print("=" * 80)
    print("VISUALIZATION (3D Coordinated Turn) - C++ Accelerated")
    print("=" * 80)
    print(f"Distribution: {args.dist}")
    print(f"Results path: {results_path}")
    print(f"Filters: {FILTERS_ORDER}")
    print("=" * 80)

    try:
        optimal_results, all_results = load_data(results_path, args.dist)
        print_optimal_results_summary(optimal_results)

        # Heatmaps
        print("\nCreating MSE heatmaps...")
        plot_mse_heatmaps(all_results, args.dist)

        # Violin plots
        print("\nCreating violin plots...")
        create_violin_plots(optimal_results, results_path, args.dist)

        # MSE bar chart
        print("\nCreating MSE bar chart...")
        plot_mse_bar_chart(optimal_results, args.dist)

        # Trajectory plots
        print("\nExtracting trajectory data...")
        trajectory_data = extract_trajectory_data(optimal_results, results_path, args.dist)

        if trajectory_data:
            print("\nCreating individual 3D trajectory plots...")
            plot_individual_trajectories(trajectory_data, args.dist)

            print("\nCreating combined 3D trajectory figure...")
            plot_combined_trajectories(trajectory_data, args.dist)

            print("\nCreating combined 2D (XY) trajectory figure...")
            plot_combined_trajectories_2d(trajectory_data, args.dist)
        else:
            print("No trajectory data found.")

        print("\n" + "=" * 80)
        print("PLOTTING COMPLETE")
        print("=" * 80)
        print(f"All plots saved to: {RESULTS_DIR}/")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Run main0_CT3D.py with --dist {args.dist} first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
