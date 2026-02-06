#!/usr/bin/env python3
"""
Computation time comparison for 3D Coordinated Turn (CT) dynamics with radar measurements.
Sequential execution to measure pure estimator computation time.
"""

import numpy as np
import time
import os

# C++ accelerated implementations
from estimator.EKF import EKF_CPP
from estimator.DR_EKF_CDC import DR_EKF_CDC_CPP
from estimator.DR_EKF_TAC import DR_EKF_TAC_CPP
from common_utils import save_data, enforce_positive_definiteness, wrap_angle
from estimator.base_filter import BaseFilter

DISPLAY_NAMES = {
    'EKF': 'EKF',
    'DR_EKF_CDC': 'CDC-MOSEK',
    'DR_EKF_CDC_FW': 'CDC-FW',
    'DR_EKF_TAC': 'TAC-MOSEK',
    'DR_EKF_TAC_FW': 'TAC-FW',
}


def display_name(filter_name):
    """Human-readable display name for a filter."""
    return DISPLAY_NAMES.get(filter_name, filter_name)


def build_filters_to_test():
    """List of filters to test (no multi-rate)."""
    return ['EKF', 'DR_EKF_CDC', 'DR_EKF_CDC_FW', 'DR_EKF_TAC', 'DR_EKF_TAC_FW']

# Helper for sampling functions
_temp_A, _temp_C = np.eye(7), np.eye(3, 7)
_temp_params = np.zeros((7, 1)), np.eye(7)
_temp_params_v = np.zeros((3, 1)), np.eye(3)
_sampler = BaseFilter(1, 'normal', 'normal', (_temp_A, _temp_C), np.eye(7, 3),
                     *_temp_params, *_temp_params, *_temp_params_v,
                     *_temp_params, *_temp_params, *_temp_params_v)

def sample_from_distribution(mu, Sigma, dist_type, max_val=None, min_val=None, scale=None, N=1):
    """Sample from specified distribution type"""
    if dist_type == "normal":
        return _sampler.normal(mu, Sigma, N)
    elif dist_type == "quadratic":
        return _sampler.quadratic(max_val, min_val, N)
    elif dist_type == "laplace":
        return _sampler.laplace(mu, scale, N)
    else:
        raise ValueError(f"Unsupported distribution: {dist_type}")

# --- 3D Coordinated Turn (CT) Dynamics Implementation ---
def ct_dynamics(x, u, dt=0.2, omega_eps=1e-4):
    """3D CT dynamics: x = [px, py, pz, vx, vy, vz, omega]^T (u is unused, kept for compatibility)"""
    px, py, pz, vx, vy, vz, omega = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0], x[5, 0], x[6, 0]

    phi = omega * dt

    if abs(omega) >= omega_eps:
        # Turn rate is significant - use CT model
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        A = sin_phi / omega
        B = (1 - cos_phi) / omega
        C = (cos_phi - 1) / omega
        D = sin_phi / omega

        px_next = px + A * vx + B * vy
        py_next = py + C * vx + D * vy
        pz_next = pz + vz * dt
        vx_next = cos_phi * vx - sin_phi * vy
        vy_next = sin_phi * vx + cos_phi * vy
        vz_next = vz
        omega_next = omega
    else:
        # Straight-line approximation (constant velocity)
        px_next = px + vx * dt
        py_next = py + vy * dt
        pz_next = pz + vz * dt
        vx_next = vx
        vy_next = vy
        vz_next = vz
        omega_next = omega

    return np.array([[px_next], [py_next], [pz_next], [vx_next], [vy_next], [vz_next], [omega_next]])

def ct_jacobian(x, u, dt=0.2, omega_eps=1e-4):
    """Jacobian of 3D CT dynamics w.r.t. state (7x7)"""
    px, py, pz, vx, vy, vz, omega = x[0, 0], x[1, 0], x[2, 0], x[3, 0], x[4, 0], x[5, 0], x[6, 0]

    phi = omega * dt

    if abs(omega) >= omega_eps:
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        A = sin_phi / omega
        B = (1 - cos_phi) / omega
        C = (cos_phi - 1) / omega
        D = sin_phi / omega

        dA_domega = (dt * cos_phi) / omega - sin_phi / (omega**2)
        dB_domega = (dt * sin_phi) / omega - (1 - cos_phi) / (omega**2)
        dC_domega = (-dt * sin_phi) / omega - (cos_phi - 1) / (omega**2)
        dD_domega = (dt * cos_phi) / omega - sin_phi / (omega**2)

        dcos_phi_domega = -dt * sin_phi
        dsin_phi_domega = dt * cos_phi

        F = np.array([
            [1, 0, 0, A, B, 0, vx * dA_domega + vy * dB_domega],
            [0, 1, 0, C, D, 0, vx * dC_domega + vy * dD_domega],
            [0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, cos_phi, -sin_phi, 0, vx * dcos_phi_domega + vy * (-dsin_phi_domega)],
            [0, 0, 0, sin_phi, cos_phi, 0, vx * dsin_phi_domega + vy * dcos_phi_domega],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
    else:
        F = np.array([
            [1, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

    return F

def radar_observation_function(x, sensor_pos=(0, 0, 0)):
    """3D Radar observation function: y = [range, azimuth, elevation]^T"""
    px, py, pz = x[0, 0], x[1, 0], x[2, 0]
    sx, sy, sz = sensor_pos

    dx = px - sx
    dy = py - sy
    dz = pz - sz

    rho = np.sqrt(dx**2 + dy**2)
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    range_val = r
    azimuth = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, rho)

    return np.array([[range_val], [azimuth], [elevation]])

def radar_observation_jacobian(x, sensor_pos=(0, 0, 0), range_eps=1e-6, rho_min=1e-2):
    """3D Radar observation Jacobian H = dh/dx (3x7) with angle gating"""
    px, py, pz = x[0, 0], x[1, 0], x[2, 0]
    sx, sy, sz = sensor_pos

    dx = px - sx
    dy = py - sy
    dz = pz - sz

    rho = np.sqrt(dx**2 + dy**2)
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    r = max(r, range_eps)

    dr_dpx = dx / r
    dr_dpy = dy / r
    dr_dpz = dz / r

    if rho < rho_min:
        H = np.array([
            [dr_dpx, dr_dpy, dr_dpz, 0, 0, 0, 0],
            [0,      0,      0,      0, 0, 0, 0],
            [0,      0,      0,      0, 0, 0, 0]
        ])
    else:
        rho_safe = max(rho, range_eps)

        daz_dpx = -dy / (rho_safe**2)
        daz_dpy = dx / (rho_safe**2)
        daz_dpz = 0

        del_dpx = -dx * dz / (rho_safe * r**2)
        del_dpy = -dy * dz / (rho_safe * r**2)
        del_dpz = rho_safe / (r**2)

        H = np.array([
            [dr_dpx,  dr_dpy,  dr_dpz,  0, 0, 0, 0],
            [daz_dpx, daz_dpy, daz_dpz, 0, 0, 0, 0],
            [del_dpx, del_dpy, del_dpz, 0, 0, 0, 0]
        ])

    return H

def wrap_angle_measurement(y_measured, y_predicted):
    """Wrap angle measurements to be consistent with predicted angles for 3D radar."""
    y_wrapped = y_measured.copy()
    if y_wrapped.shape[0] >= 3:
        # Wrap azimuth
        azimuth_diff = y_measured[1, 0] - y_predicted[1, 0]
        wrapped_diff = wrap_angle(azimuth_diff)
        y_wrapped[1, 0] = y_predicted[1, 0] + wrapped_diff

        # Wrap elevation
        elevation_diff = y_measured[2, 0] - y_predicted[2, 0]
        wrapped_diff_el = wrap_angle(elevation_diff)
        y_wrapped[2, 0] = y_predicted[2, 0] + wrapped_diff_el

    return y_wrapped

def run_timing_simulation(estimator, T, dt):
    """Run CT simulation and measure estimator computation time"""
    nx, ny = estimator.nx, estimator.ny
    nu = 2  # Dummy control dimension

    # Allocate arrays
    x = np.zeros((T+1, nx, 1))
    y = np.zeros((T+1, ny, 1))
    x_est = np.zeros((T+1, nx, 1))
    u_traj = np.zeros((T, nu, 1))

    # Store timing results
    estimator_times = []

    # Store covariance traces
    P_trace = []
    wc_post_trace = []
    wc_v_trace = []

    # Reset noise index
    estimator._noise_index = 0

    # Initialization
    x[0] = estimator.sample_initial_state()
    x_est[0] = estimator.nominal_x0_mean.copy()

    # First measurement and update - TIME THIS
    v0 = estimator.sample_measurement_noise()
    y_raw0 = radar_observation_function(x[0]) + v0

    # Wrap angles for initial measurement
    if hasattr(estimator, 'h'):
        y_pred0 = estimator.h(x_est[0]) + estimator.nominal_mu_v
        y[0] = wrap_angle_measurement(y_raw0, y_pred0)
    else:
        y[0] = y_raw0

    start_time = time.perf_counter()
    x_est[0] = estimator._initial_update(x_est[0], y[0])
    end_time = time.perf_counter()
    estimator_times.append(end_time - start_time)

    # Collect covariance traces after initial update
    if hasattr(estimator, "_P") and estimator._P is not None:
        P_trace.append(float(np.trace(estimator._P)))
    else:
        P_trace.append(np.nan)
    if hasattr(estimator, "_last_wc_Xpost") and estimator._last_wc_Xpost is not None:
        wc_post_trace.append(float(np.trace(estimator._last_wc_Xpost)))
    else:
        wc_post_trace.append(np.nan)
    if hasattr(estimator, "_last_wc_Sigma_v") and estimator._last_wc_Sigma_v is not None:
        wc_v_trace.append(float(np.trace(estimator._last_wc_Sigma_v)))
    else:
        wc_v_trace.append(np.nan)

    # Main simulation loop
    for t in range(T):
        # No controller - CT model is autonomous
        u = np.zeros((nu, 1))
        u_traj[t] = u.copy()

        # True state propagation using CT dynamics (NOT TIMED)
        w = estimator.sample_process_noise()
        x[t+1] = ct_dynamics(x[t], u, dt=dt) + w
        estimator._noise_index += 1

        # Measurement using radar observation function (NOT TIMED)
        v = estimator.sample_measurement_noise()
        y_raw = radar_observation_function(x[t+1]) + v

        # Wrap bearing measurements (NOT TIMED)
        if hasattr(estimator, 'h'):
            x_pred = estimator.f(x_est[t], u) + estimator.nominal_mu_w
            y_pred = estimator.h(x_pred) + estimator.nominal_mu_v
            y[t+1] = wrap_angle_measurement(y_raw, y_pred)
        else:
            y[t+1] = y_raw

        # State estimation update - TIME ONLY THIS
        start_time = time.perf_counter()
        x_est[t+1] = estimator.update_step(x_est[t], y[t+1], t+1, u)
        end_time = time.perf_counter()
        estimator_times.append(end_time - start_time)

        # Collect covariance traces after each update
        if hasattr(estimator, "_P") and estimator._P is not None:
            P_trace.append(float(np.trace(estimator._P)))
        else:
            P_trace.append(np.nan)
        if hasattr(estimator, "_last_wc_Xpost") and estimator._last_wc_Xpost is not None:
            wc_post_trace.append(float(np.trace(estimator._last_wc_Xpost)))
        else:
            wc_post_trace.append(np.nan)
        if hasattr(estimator, "_last_wc_Sigma_v") and estimator._last_wc_Sigma_v is not None:
            wc_v_trace.append(float(np.trace(estimator._last_wc_Sigma_v)))
        else:
            wc_v_trace.append(np.nan)

    # Compute RMSE
    err = x - x_est
    rmse = np.sqrt(np.mean(np.sum(err[:, :, 0]**2, axis=1)))

    # Collect DR solve statistics
    dr_solve_count = getattr(estimator, '_dr_solve_count', 0)
    dr_step_count = getattr(estimator, '_dr_step_count', 0)
    dr_update_period = getattr(estimator, 'dr_update_period', 1)

    return {
        "times": estimator_times,
        "x_true": x,
        "x_est": x_est,
        "y": y,
        "u": u_traj,
        "err": err,
        "rmse": float(rmse),
        "P_trace": np.array(P_trace),
        "wc_post_trace": np.array(wc_post_trace),
        "wc_v_trace": np.array(wc_v_trace),
        "dr_solve_count": dr_solve_count,
        "dr_step_count": dr_step_count,
        "dr_update_period": dr_update_period,
    }

def estimate_nominal_parameters(true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                              dist, num_samples, x0_max=None, x0_min=None, w_max=None, w_min=None,
                              v_max=None, v_min=None, x0_scale=None, w_scale=None, v_scale=None):
    """Estimate nominal parameters using samples from true distributions"""
    nx, ny = true_x0_mean.shape[0], true_mu_v.shape[0]

    if dist == "normal":
        x0_samples = sample_from_distribution(true_x0_mean, true_x0_cov, "normal", N=num_samples)
        w_samples = sample_from_distribution(true_mu_w, true_Sigma_w, "normal", N=num_samples)
        v_samples = sample_from_distribution(true_mu_v, true_Sigma_v, "normal", N=num_samples)
    elif dist == "quadratic":
        x0_samples = sample_from_distribution(None, None, "quadratic", x0_max, x0_min, N=num_samples)
        w_samples = sample_from_distribution(None, None, "quadratic", w_max, w_min, N=num_samples)
        v_samples = sample_from_distribution(None, None, "quadratic", v_max, v_min, N=num_samples)
    elif dist == "laplace":
        x0_samples = sample_from_distribution(true_x0_mean, None, "laplace", scale=x0_scale, N=num_samples)
        w_samples = sample_from_distribution(true_mu_w, None, "laplace", scale=w_scale, N=num_samples)
        v_samples = sample_from_distribution(true_mu_v, None, "laplace", scale=v_scale, N=num_samples)

    nominal_x0_mean = np.mean(x0_samples, axis=1).reshape(-1, 1)
    nominal_x0_cov = np.cov(x0_samples)

    nominal_mu_w = np.mean(w_samples, axis=1).reshape(-1, 1)
    nominal_Sigma_w = np.cov(w_samples)

    nominal_mu_v = np.mean(v_samples, axis=1).reshape(-1, 1)
    nominal_Sigma_v = np.cov(v_samples)

    nominal_x0_cov = enforce_positive_definiteness(nominal_x0_cov)
    nominal_Sigma_w = enforce_positive_definiteness(nominal_Sigma_w)
    nominal_Sigma_v = enforce_positive_definiteness(nominal_Sigma_v)

    return (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w,
            nominal_mu_v, nominal_Sigma_v)

def run_timing_experiment(dist='normal', T_total=10.0, num_samples=100, robust_val=0.1, num_trials=10):
    """Run timing comparison for all filters with CT3D dynamics"""

    dt = 0.2
    T = int(T_total / dt)
    nx, ny, nu = 7, 3, 2

    # System matrices for DR-EKF
    A = np.eye(nx)  # Placeholder - will use jacobians online
    B = np.zeros((nx, nu))  # Placeholder - CT dynamics are autonomous
    C = np.array([[1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0]])
    system_data = (A, C)

    # Initial state: [px, py, pz, vx, vy, vz, omega]
    x0_mean = np.array([[0.0], [0.0], [0.0], [2.0], [0.0], [0.0], [0.10]])
    x0_cov = np.diag([0.5**2, 0.5**2, 0.5**2, 0.5**2, 0.5**2, 0.5**2, 0.05**2])

    if dist == "normal":
        mu_w = np.zeros((nx, 1))
        mu_v = np.zeros((ny, 1))

        # Process noise
        sigma_px = sigma_py = sigma_pz = 0.01
        sigma_vx = sigma_vy = sigma_vz = 0.02
        sigma_omega = 0.015
        Sigma_w = np.diag([sigma_px**2, sigma_py**2, sigma_pz**2,
                          sigma_vx**2, sigma_vy**2, sigma_vz**2, sigma_omega**2])

        # Measurement noise: [range, azimuth, elevation]
        sigma_range = 0.01
        sigma_azimuth = np.deg2rad(0.02)
        sigma_elevation = np.deg2rad(0.02)
        Sigma_v = np.diag([sigma_range**2, sigma_azimuth**2, sigma_elevation**2])

        v_max = v_min = w_max = w_min = x0_max = x0_min = None
        x0_scale = w_scale = v_scale = None
    else:  # quadratic
        x0_max = np.array([0.5, 0.5, 0.5, 2.5, 0.5, 0.5, 0.2])
        x0_min = np.array([-0.5, -0.5, -0.5, 1.5, -0.5, -0.5, 0.1])
        x0_mean = (0.5 * (x0_max + x0_min)).reshape(-1, 1)
        x0_cov = 3.0/20.0 * np.diag((x0_max - x0_min)**2)

        w_max = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.1, 0.03])
        w_min = -w_max
        mu_w = np.zeros((nx, 1))
        Sigma_w = 3.0/20.0 * np.diag((w_max - w_min)**2)

        v_max = np.array([0.02, np.deg2rad(0.1), np.deg2rad(0.1)])
        v_min = -v_max
        mu_v = np.zeros((ny, 1))
        Sigma_v = 3.0/20.0 * np.diag((v_max - v_min)**2)

        x0_scale = w_scale = v_scale = None

    # Generate nominal parameters
    np.random.seed(2024)
    nominal_params = estimate_nominal_parameters(
        x0_mean, x0_cov, mu_w, Sigma_w, mu_v, Sigma_v, dist, num_samples,
        x0_max, x0_min, w_max, w_min, v_max, v_min, x0_scale, w_scale, v_scale)

    (nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w,
     nominal_mu_v, nominal_Sigma_v) = nominal_params

    # Filters to test (generated programmatically)
    filters_to_test = build_filters_to_test()

    # Common kwargs shared by all estimators
    common_kwargs = dict(
        T=T, dist=dist, noise_dist=dist, system_data=system_data, B=B,
        true_x0_mean=x0_mean, true_x0_cov=x0_cov,
        true_mu_w=mu_w, true_Sigma_w=Sigma_w,
        true_mu_v=mu_v, true_Sigma_v=Sigma_v,
        nominal_x0_mean=nominal_x0_mean, nominal_x0_cov=nominal_x0_cov,
        nominal_mu_w=nominal_mu_w, nominal_Sigma_w=nominal_Sigma_w,
        nominal_mu_v=nominal_mu_v, nominal_Sigma_v=nominal_Sigma_v,
        nonlinear_dynamics=ct_dynamics,
        dynamics_jacobian=ct_jacobian,
        observation_function=radar_observation_function,
        observation_jacobian=radar_observation_jacobian,
        x0_max=x0_max, x0_min=x0_min, w_max=w_max, w_min=w_min,
        v_max=v_max, v_min=v_min, x0_scale=x0_scale, w_scale=w_scale, v_scale=v_scale,
    )

    # Store all trial data
    all_trial_data = {filter_name: [] for filter_name in filters_to_test}

    print(f"Running timing comparison for {num_trials} trials...")
    print(f"Distribution: {dist}, Trajectory length: {T} steps, Robust parameter: {robust_val}")
    print(f"Filters: {len(filters_to_test)} ({', '.join(filters_to_test)})")
    print("-" * 80)

    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}:")

        for filter_name in filters_to_test:
            np.random.seed(2024 + trial)

            if filter_name == 'EKF':
                # Use C++ EKF
                estimator = EKF_CPP(**common_kwargs, dynamics_type="ct3d", dt=dt)
            elif filter_name == 'DR_EKF_TAC':
                estimator = DR_EKF_TAC_CPP(
                    **common_kwargs,
                    theta_x=robust_val, theta_v=robust_val, theta_w=robust_val,
                    dynamics_type="ct3d", dt=dt)
            elif filter_name == 'DR_EKF_CDC':
                estimator = DR_EKF_CDC_CPP(
                    **common_kwargs,
                    theta_x=robust_val, theta_v=robust_val,
                    solver='mosek',
                    dynamics_type="ct3d", dt=dt)
            elif filter_name == 'DR_EKF_CDC_FW':
                # Use C++ FW solver
                estimator = DR_EKF_CDC_CPP(
                    **common_kwargs,
                    theta_x=robust_val, theta_v=robust_val,
                    solver='fw',
                    dynamics_type="ct3d", dt=dt)
            elif filter_name == 'DR_EKF_TAC_FW':
                estimator = DR_EKF_TAC_CPP(
                    **common_kwargs,
                    theta_x=robust_val, theta_v=robust_val, theta_w=robust_val,
                    solver='fw',
                    dynamics_type="ct3d", dt=dt)
            else:
                raise ValueError(f"Unknown filter: {filter_name}")

            # Run timing simulation
            out = run_timing_simulation(estimator, T, dt)
            all_trial_data[filter_name].append(out)

            avg_time_per_step = np.mean(out["times"])
            frequency_hz = 1.0 / avg_time_per_step if avg_time_per_step > 0 else float('inf')
            dr_frac = out["dr_solve_count"] / max(out["dr_step_count"], 1)

            print(f"  {display_name(filter_name):<30}: {avg_time_per_step*1000:.4f} ms/step, {frequency_hz:.1f} Hz, "
                  f"RMSE: {out['rmse']:.4f}, DR-frac: {dr_frac:.2f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS - Effective computation time per step (Averaged over all trials):")
    print("=" * 80)

    final_results = {}

    for filter_name in filters_to_test:
        all_times_flat = [t_val for trial in all_trial_data[filter_name] for t_val in trial["times"]]
        all_rmse = [trial["rmse"] for trial in all_trial_data[filter_name]]
        all_dr_frac = [trial["dr_solve_count"] / max(trial["dr_step_count"], 1)
                       for trial in all_trial_data[filter_name]]

        avg_time_per_step = np.mean(all_times_flat)
        std_time_per_step = np.std(all_times_flat)
        frequency_hz = 1.0 / avg_time_per_step if avg_time_per_step > 0 else float('inf')

        avg_rmse = np.mean(all_rmse)
        std_rmse = np.std(all_rmse)

        final_results[filter_name] = {
            'avg_time_per_step_ms': avg_time_per_step * 1000,
            'std_time_per_step_ms': std_time_per_step * 1000,
            'frequency_hz': frequency_hz,
            'avg_rmse': avg_rmse,
            'std_rmse': std_rmse,
            'trials': all_trial_data[filter_name],
        }

        print(f"{display_name(filter_name):<20}: {avg_time_per_step*1000:.4f}+/-{std_time_per_step*1000:.4f} ms/step, "
              f"{frequency_hz:.1f} Hz, RMSE: {avg_rmse:.4f}+/-{std_rmse:.4f}")

    return final_results

def main():
    """Main timing comparison routine for CT3D (C++ accelerated)"""

    # Configuration
    dist = 'normal'
    T_total = 5.0   # 5 seconds simulation
    num_samples = 100
    robust_vals = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    num_trials = 5

    filters_to_test = build_filters_to_test()

    print("3D Coordinated Turn - EKF vs DR-EKF Computation Time Comparison (C++ Accelerated)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Distribution: {dist}")
    print(f"  Simulation time: {T_total} seconds")
    print(f"  Time step: 0.2 seconds ({int(T_total/0.2)} steps)")
    print(f"  State: [px, py, pz, vx, vy, vz, omega] (7D)")
    print(f"  Observation: [range, azimuth, elevation] (3D radar)")
    print(f"  Robustness parameters (theta): {robust_vals}")
    print(f"  Number of trials per theta: {num_trials}")
    print(f"  Nominal parameter samples: {num_samples}")
    print(f"  Filters: {filters_to_test}")
    print("=" * 80)

    all_results = {}

    for robust_val in robust_vals:
        print("\n" + "=" * 80)
        print(f"Testing theta = {robust_val}")
        print("=" * 80)

        results = run_timing_experiment(dist, T_total, num_samples, robust_val, num_trials)
        all_results[robust_val] = results

        print("\n" + "-" * 80)
        print(f"SUMMARY (theta = {robust_val}):")
        print("-" * 80)

        ekf_time = results['EKF']['avg_time_per_step_ms']
        ekf_rmse = results['EKF']['avg_rmse']

        for filter_name, data in results.items():
            if filter_name != 'EKF':
                speedup_factor = ekf_time / data['avg_time_per_step_ms'] if data['avg_time_per_step_ms'] > 0 else float('inf')
                rmse_ratio = data['avg_rmse'] / ekf_rmse if ekf_rmse > 0 else float('inf')
                print(f"{display_name(filter_name):<15}: {speedup_factor:.2f}x {'faster' if speedup_factor > 1 else 'slower'} than EKF, "
                      f"RMSE ratio: {rmse_ratio:.3f}")

        print("-" * 80)

    # Save all results (C++ accelerated version)
    results_path = "./results/timing_comparison_CT3D_cpp/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_data(os.path.join(results_path, f'timing_quality_results_all_theta_{dist}.pkl'), all_results)
    save_data(os.path.join(results_path, f'timing_results_all_theta_{dist}.pkl'), all_results)

    print("\n" + "=" * 80)
    print("ALL RESULTS SAVED")
    print("=" * 80)
    print(f"Results saved to: {results_path}")
    print(f"Theta values tested: {robust_vals}")
    print(f"Filters tested: {filters_to_test}")
    print("=" * 80)

    # Print summary table
    print(f"\n{'=' * 100}")
    print("COMPUTATION TIME (ms/step)")
    print(f"{'=' * 100}")
    header = f"{'theta':<8}"
    for fn in filters_to_test:
        header += f" {display_name(fn):<18}"
    print(header)
    print("-" * 100)

    for theta in robust_vals:
        row = f"{theta:<8.2f}"
        for fn in filters_to_test:
            data = all_results[theta][fn]
            row += f" {data['avg_time_per_step_ms']:6.3f}+/-{data['std_time_per_step_ms']:5.3f}  "
        print(row)
    print(f"{'=' * 100}")

    print(f"\n{'=' * 100}")
    print("RMSE")
    print(f"{'=' * 100}")
    header = f"{'theta':<8}"
    for fn in filters_to_test:
        header += f" {display_name(fn):<18}"
    print(header)
    print("-" * 100)

    for theta in robust_vals:
        row = f"{theta:<8.2f}"
        for fn in filters_to_test:
            data = all_results[theta][fn]
            row += f" {data['avg_rmse']:6.4f}+/-{data['std_rmse']:5.4f}  "
        print(row)
    print(f"{'=' * 100}")

    return all_results

if __name__ == "__main__":
    main()
