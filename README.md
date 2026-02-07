# Distributionally Robust Extended Kalman Filter

C++ accelerated DR-EKF implementation with two solver backends: a Frank-Wolfe first-order method (pure C++/Eigen, embeddable) and MOSEK semidefinite programming. Python bindings via pybind11.

## Directory Structure

```
DREKF2026/
├── estimator/                  # Python wrappers and C++ source
│   ├── EKF.py                  # Standard EKF wrapper
│   ├── DR_EKF_CDC.py           # DR-EKF CDC wrapper (mosek/fw/fw_exact solvers)
│   ├── DR_EKF_TAC.py           # DR-EKF TAC wrapper (mosek/fw/fw_exact solvers)
│   ├── base_filter.py          # Base filter class
│   └── cpp/                    # C++ core
│       ├── CMakeLists.txt
│       ├── include/dr_ekf/     # Headers
│       │   ├── mosek_sdp_solver.h   # MOSEK SDP solver (CDC + TAC formulations)
│       │   ├── dr_ekf_cdc.h         # DR-EKF CDC filter
│       │   ├── dr_ekf_tac.h         # DR-EKF TAC filter
│       │   ├── ekf.h, dynamics.h, kalman_utils.h, fw_oracle.h, types.h
│       └── src/                # Implementations + pybind11 bindings
├── main0_CT3D.py               # 3D coordinated turn — grid search over theta
├── main0_CT.py                 # 2D coordinated turn — grid search over theta
├── main1_timeCT3D.py           # 3D coordinated turn — timing comparison
├── plot0_CT3D.py               # Plot results from main0_CT3D
├── plot0_CT.py                 # Plot results from main0_CT
├── plot1_timeCT3D.py          # Plot timing results
├── video0_CT3D.py              # Animate 3D CT trajectories
├── common_utils.py             # Shared utilities (EM estimation, grid search, etc.)
├── old_files/                  # Legacy CVXPY-based implementations
└── results/                    # Experiment output (auto-created)
```

## Requirements

- Python 3.11
- numpy, scipy, matplotlib, joblib
- MOSEK 10.1 SDK (with valid license at `~/mosek/mosek.lic`)
- pybind11, Eigen3, CMake
- A C++17 compiler: MinGW (Windows), GCC (Linux), or Clang (macOS)

## Building the C++ Module

```bash
mkdir -p estimator/cpp/build && cd estimator/cpp/build
```

**Windows (MinGW):**
```bash
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="<path-to-pybind11>/share/cmake/pybind11" \
  -DMOSEK_DIR="C:/mosek/mosek/10.1/tools/platform/win64x86"
cmake --build . --config Release
```

**Linux:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DMOSEK_DIR="$HOME/mosek/mosek/10.1/tools/platform/linux64x86"
cmake --build . --config Release
```

**macOS:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DMOSEK_DIR="$HOME/mosek/mosek/10.1/tools/platform/osxaarch64"  # Apple Silicon
# For Intel Mac, use osx64x86 instead
cmake --build . --config Release
```

The built module (`.pyd` on Windows, `.so` on Linux/macOS) must be placed in the project root and/or `estimator/`.

> **Note:** The `-Dpybind11_DIR` flag is only needed if CMake cannot find pybind11 automatically. If you installed it via `pip install pybind11`, CMake should detect it.

**Rebuild after C++ changes** (headers, source, or defaults):
```bash
cd estimator/cpp/build && cmake --build .
```

## Running Experiments

**Grid search** (3D coordinated turn):
```bash
py main0_CT3D.py
```

**Grid search** (2D coordinated turn):
```bash
py main0_CT.py
```

Arguments: `--dist` (normal/quadratic), `--num_sim`, `--num_exp`, `--T_total`, `--T_em`, `--num_samples`

**Timing comparison** (no CLI args, edit script to configure):
```bash
py main1_timeCT3D.py
```

## Plotting Results

Run the corresponding plot script after an experiment completes:
```bash
py plot0_CT3D.py
py plot0_CT.py
py plot1_timeCT3D.py
```

## Generating Videos

Animated 3D tracking videos showing the true trajectory (rendered as an airplane) alongside filter estimates. Requires experiment results to be generated first.

```bash
py video0_CT3D.py          # uses results from main0_CT3D.py
```

Arguments: `--dist` (normal/quadratic), `--fps` (default 15), `--duration` (seconds, default full trajectory), `--format` (mp4/gif, default mp4)

Each script generates both normal and zoomed views for mean trajectories and up to 5 single simulation instances. MP4 output requires [ffmpeg](https://ffmpeg.org/); GIF uses pillow.

## Porting to Embedded C++

The core algorithm lives in `estimator/cpp/` and can be deployed without Python.

**Use the Frank-Wolfe solver only.** MOSEK is a desktop-only commercial solver and cannot run on embedded targets. The FW solver (`fw` or `fw_exact`) is pure C++ with only an Eigen dependency.

### Files needed

From `estimator/cpp/`:
- **Headers**: `types.h`, `dynamics.h`, `ekf.h`, `dr_ekf_cdc.h`, `dr_ekf_tac.h`, `kalman_utils.h`, `fw_oracle.h`
- **Sources**: `dynamics.cpp`, `ekf.cpp`, `dr_ekf_cdc.cpp`, `dr_ekf_tac.cpp`, `kalman_utils.cpp`, `fw_oracle.cpp`

Skip: `mosek_sdp_solver.*` (MOSEK dependency), `bindings.cpp` (pybind11 glue).

### Implementing your own dynamics

Subclass `DynamicsInterface` in `dynamics.h`. You need to provide four functions:

```cpp
#include "dr_ekf/dynamics.h"

class MyDynamics : public dr_ekf::DynamicsInterface {
public:
    double dt;
    MyDynamics(double dt) : dt(dt) {}

    int nx() const override { return /* state dimension */; }
    int ny() const override { return /* observation dimension */; }

    // x_{t+1} = f(x_t, u_t)
    VecXd f(const VecXd& x, const VecXd& u) const override {
        VecXd x_next(nx());
        // TODO: implement state transition
        return x_next;
    }

    // Jacobian df/dx
    MatXd F_jac(const VecXd& x, const VecXd& u) const override {
        MatXd F = MatXd::Zero(nx(), nx());
        // TODO: implement Jacobian of f w.r.t. x
        return F;
    }

    // y_t = h(x_t)
    VecXd h(const VecXd& x) const override {
        VecXd y(ny());
        // TODO: implement observation model
        return y;
    }

    // Jacobian dh/dx
    MatXd H_jac(const VecXd& x) const override {
        MatXd H = MatXd::Zero(ny(), nx());
        // TODO: implement Jacobian of h w.r.t. x
        return H;
    }
};
```

See `CT2D` and `CT3D` in `dynamics.cpp` for reference implementations.

### Usage

```cpp
MyDynamics dynamics(dt);
dr_ekf::DR_EKF_CDC filter(dynamics, x0_cov, Sigma_w, Sigma_v,
                           mu_w, mu_v, theta_x, theta_v, "fw_exact");

x_est = filter.initial_update(x_est, y0);
for (int t = 1; t <= T; t++) {
    x_est = filter.update_step(x_est, y[t], t, u[t-1]);
}
```
