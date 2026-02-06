#!/usr/bin/env python3
"""
C++ accelerated Extended Kalman Filter (EKF).
Thin wrapper that inherits BaseFilter and delegates computation to C++ via pybind11.
"""

import os
import sys

# MOSEK DLL directory must be added before importing dr_ekf_cpp on Windows
if sys.platform == "win32":
    _mosek_bin = os.path.join(os.environ.get("MOSEK_DIR", "C:/mosek/mosek/10.1/tools/platform/win64x86"), "bin")
    if os.path.isdir(_mosek_bin):
        os.add_dll_directory(_mosek_bin)

import numpy as np
from .base_filter import BaseFilter


class EKF_CPP(BaseFilter):
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
                 input_lower_bound=None, input_upper_bound=None,
                 dynamics_type="ct2d", dt=0.2, sensor_pos=None):
        super().__init__(T, dist, noise_dist, system_data, B,
                        true_x0_mean, true_x0_cov, true_mu_w, true_Sigma_w, true_mu_v, true_Sigma_v,
                        nominal_x0_mean, nominal_x0_cov, nominal_mu_w, nominal_Sigma_w, nominal_mu_v, nominal_Sigma_v,
                        x0_max, x0_min, w_max, w_min, v_max, v_min,
                        x0_scale, w_scale, v_scale, None,
                        input_lower_bound, input_upper_bound)

        import dr_ekf_cpp

        # Create C++ dynamics object
        self._cpp_dynamics = _create_dynamics(dynamics_type, dt, sensor_pos)

        # Create C++ EKF
        self._cpp = dr_ekf_cpp.EKF(
            self._cpp_dynamics,
            np.ascontiguousarray(nominal_x0_cov, dtype=np.float64),
            np.ascontiguousarray(nominal_Sigma_w, dtype=np.float64),
            np.ascontiguousarray(nominal_Sigma_v, dtype=np.float64),
            np.ascontiguousarray(nominal_mu_w.flatten(), dtype=np.float64),
            np.ascontiguousarray(nominal_mu_v.flatten(), dtype=np.float64),
        )

        # Keep Python function refs for compatibility (not used by C++ path)
        self.f = nonlinear_dynamics
        self.F_jacobian = dynamics_jacobian
        self.h = observation_function
        self.C_jacobian = observation_jacobian
        self._P = None

    def _initial_update(self, x_est_init, y0):
        result = self._cpp.initial_update(
            x_est_init.flatten(),
            y0.flatten()
        )
        self._P = self._cpp.get_P()
        return result.reshape(-1, 1)

    def update_step(self, x_est_prev, y_curr, t, u_prev):
        result = self._cpp.update_step(
            x_est_prev.flatten(),
            y_curr.flatten(),
            t,
            u_prev.flatten()
        )
        self._P = self._cpp.get_P()
        return result.reshape(-1, 1)

    def forward(self):
        raise NotImplementedError("Use run_single_simulation in main scripts")

    def forward_track(self, desired_trajectory):
        raise NotImplementedError("Use run_single_simulation in main scripts")


def _create_dynamics(dynamics_type, dt=0.2, sensor_pos=None):
    """Create C++ dynamics object based on type string."""
    import dr_ekf_cpp

    if dynamics_type == "ct2d":
        sx, sy = (0.0, 0.0) if sensor_pos is None else sensor_pos[:2]
        return dr_ekf_cpp.CT2D(dt=dt, sensor_x=sx, sensor_y=sy)
    elif dynamics_type == "ct3d":
        sx, sy, sz = (0.0, 0.0, 0.0) if sensor_pos is None else sensor_pos[:3]
        return dr_ekf_cpp.CT3D(dt=dt, sensor_x=sx, sensor_y=sy, sensor_z=sz)
    # --- TEMPLATE: Add your dynamics here ---
    # elif dynamics_type == "my_dynamics":
    #     return dr_ekf_cpp.MyDynamics(dt=dt /*, ... */)
    else:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type}. Use 'ct2d' or 'ct3d'.")
