#!/usr/bin/env python3
"""
C++ accelerated DR-EKF CDC (MOSEK, Frank-Wolfe, and FW-Exact solvers).
Thin wrapper that inherits BaseFilter and delegates computation to C++ via pybind11.
"""

import numpy as np
from .base_filter import BaseFilter


class DR_EKF_CDC_CPP(BaseFilter):
    def __init__(self, T, dist, noise_dist, system_data, B,
                 true_x0_mean, true_x0_cov,
                 true_mu_w, true_Sigma_w,
                 true_mu_v, true_Sigma_v,
                 nominal_x0_mean, nominal_x0_cov,
                 nominal_mu_w, nominal_Sigma_w,
                 nominal_mu_v, nominal_Sigma_v,
                 nonlinear_dynamics=None,
                 dynamics_jacobian=None,
                 observation_function=None,
                 observation_jacobian=None,
                 x0_max=None, x0_min=None, w_max=None, w_min=None, v_max=None, v_min=None,
                 x0_scale=None, w_scale=None, v_scale=None,
                 theta_x=None, theta_v=None, shared_noise_sequences=None,
                 input_lower_bound=None, input_upper_bound=None,
                 solver="fw_exact",
                 fw_beta_minus1=1.0,
                 fw_tau=2.0,
                 fw_zeta=2.0,
                 fw_delta=0.05,
                 fw_max_iters=20,
                 fw_gap_tol=1e-4,
                 fw_bisect_max_iters=30,
                 fw_bisect_tol=1e-6,
                 fw_exact_iters=8,
                 fw_exact_gap_tol=1e-6,
                 oracle_exact_tol=1e-8,
                 oracle_exact_max_iters=60,
                 dr_update_period=1,
                 dynamics_type="ct2d", dt=0.2, sensor_pos=None):

        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, shared_noise_sequences,
                        input_lower_bound, input_upper_bound)

        if solver not in ("fw", "fw_exact", "mosek"):
            raise ValueError(f"DR_EKF_CDC_CPP only supports solver='fw', 'fw_exact', or 'mosek', got '{solver}'.")

        import dr_ekf_cpp
        from .EKF import _create_dynamics

        self._cpp_dynamics = _create_dynamics(dynamics_type, dt, sensor_pos)

        self._cpp = dr_ekf_cpp.DR_EKF_CDC(
            self._cpp_dynamics,
            np.ascontiguousarray(nominal_x0_cov, dtype=np.float64),
            np.ascontiguousarray(nominal_Sigma_w, dtype=np.float64),
            np.ascontiguousarray(nominal_Sigma_v, dtype=np.float64),
            np.ascontiguousarray(nominal_mu_w.flatten(), dtype=np.float64),
            np.ascontiguousarray(nominal_mu_v.flatten(), dtype=np.float64),
            theta_x,
            theta_v,
            solver,
            dr_update_period,
            fw_beta_minus1,
            fw_tau,
            fw_zeta,
            fw_delta,
            fw_max_iters,
            fw_gap_tol,
            fw_bisect_max_iters,
            fw_bisect_tol,
            fw_exact_iters,
            fw_exact_gap_tol,
            oracle_exact_tol,
            oracle_exact_max_iters,
        )

        # Store for compatibility
        self.f = nonlinear_dynamics
        self.F_jacobian = dynamics_jacobian
        self.h = observation_function
        self.C_jacobian = observation_jacobian
        self.solver = solver
        self.theta_x = theta_x
        self.theta_v = theta_v
        self.dr_update_period = dr_update_period
        self._P = None
        self._dr_solve_count = 0
        self._dr_step_count = 0

    def _initial_update(self, x_est_init, y0):
        result = self._cpp.initial_update(
            x_est_init.flatten(),
            y0.flatten()
        )
        self._P = self._cpp.get_P()
        self._dr_solve_count = self._cpp.get_dr_solve_count()
        self._dr_step_count = self._cpp.get_dr_step_count()
        return result.reshape(-1, 1)

    def update_step(self, x_est_prev, y_curr, t, u_prev):
        result = self._cpp.update_step(
            x_est_prev.flatten(),
            y_curr.flatten(),
            t,
            u_prev.flatten()
        )
        self._P = self._cpp.get_P()
        self._dr_solve_count = self._cpp.get_dr_solve_count()
        self._dr_step_count = self._cpp.get_dr_step_count()
        return result.reshape(-1, 1)

    def forward(self):
        raise NotImplementedError("Use run_single_simulation in main scripts")

    def forward_track(self, desired_trajectory):
        raise NotImplementedError("Use run_single_simulation in main scripts")
