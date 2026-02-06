#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "dr_ekf/types.h"
#include "dr_ekf/dynamics.h"
#include "dr_ekf/ekf.h"
#include "dr_ekf/dr_ekf_cdc.h"
#include "dr_ekf/dr_ekf_tac.h"
#include "dr_ekf/kalman_utils.h"
#include "dr_ekf/fw_oracle.h"

namespace py = pybind11;
using namespace dr_ekf;

PYBIND11_MODULE(dr_ekf_cpp, m) {
    m.doc() = "C++ implementation of DR-EKF estimators (EKF, CDC-FW, CDC-FW-Exact, CDC-MOSEK, TAC-MOSEK, TAC-FW, TAC-FW-Exact)";

    // --- Dynamics Interface ---
    py::class_<DynamicsInterface>(m, "DynamicsInterface")
        .def("f", &DynamicsInterface::f)
        .def("F_jac", &DynamicsInterface::F_jac)
        .def("h", &DynamicsInterface::h)
        .def("H_jac", &DynamicsInterface::H_jac)
        .def("nx", &DynamicsInterface::nx)
        .def("ny", &DynamicsInterface::ny);

    py::class_<CT2D, DynamicsInterface>(m, "CT2D")
        .def(py::init<double, double, double, double, double>(),
             py::arg("dt") = 0.2,
             py::arg("omega_eps") = 1e-4,
             py::arg("sensor_x") = 0.0,
             py::arg("sensor_y") = 0.0,
             py::arg("range_eps") = 1e-6)
        .def("f", &CT2D::f)
        .def("F_jac", &CT2D::F_jac)
        .def("h", &CT2D::h)
        .def("H_jac", &CT2D::H_jac)
        .def("nx", &CT2D::nx)
        .def("ny", &CT2D::ny);

    py::class_<CT3D, DynamicsInterface>(m, "CT3D")
        .def(py::init<double, double, double, double, double, double, double>(),
             py::arg("dt") = 0.2,
             py::arg("omega_eps") = 1e-4,
             py::arg("sensor_x") = 0.0,
             py::arg("sensor_y") = 0.0,
             py::arg("sensor_z") = 0.0,
             py::arg("range_eps") = 1e-6,
             py::arg("rho_min") = 1e-2)
        .def("f", &CT3D::f)
        .def("F_jac", &CT3D::F_jac)
        .def("h", &CT3D::h)
        .def("H_jac", &CT3D::H_jac)
        .def("nx", &CT3D::nx)
        .def("ny", &CT3D::ny);

    // --- TEMPLATE: Binding for a new dynamics model ---
    // Uncomment after declaring and implementing your class.
    //
    // py::class_<MyDynamics, DynamicsInterface>(m, "MyDynamics")
    //     .def(py::init<double /*, ... */>(),
    //          py::arg("dt") = 0.1 /*, ... */)
    //     .def("f", &MyDynamics::f)
    //     .def("F_jac", &MyDynamics::F_jac)
    //     .def("h", &MyDynamics::h)
    //     .def("H_jac", &MyDynamics::H_jac)
    //     .def("nx", &MyDynamics::nx)
    //     .def("ny", &MyDynamics::ny);

    // --- EKF ---
    py::class_<EKF>(m, "EKF")
        .def(py::init<const DynamicsInterface&, const MatXd&, const MatXd&, const MatXd&, const VecXd&, const VecXd&>(),
             py::arg("dynamics"),
             py::arg("nominal_x0_cov"),
             py::arg("nominal_Sigma_w"),
             py::arg("nominal_Sigma_v"),
             py::arg("nominal_mu_w"),
             py::arg("nominal_mu_v"),
             py::keep_alive<1, 2>())  // Keep dynamics alive as long as EKF exists
        .def("initial_update", &EKF::initial_update)
        .def("update_step", &EKF::update_step)
        .def("get_P", &EKF::get_P)
        .def("get_nx", &EKF::get_nx)
        .def("get_ny", &EKF::get_ny);

    // --- DR_EKF_CDC ---
    py::class_<DR_EKF_CDC>(m, "DR_EKF_CDC")
        .def(py::init<const DynamicsInterface&, const MatXd&, const MatXd&, const MatXd&,
                       const VecXd&, const VecXd&,
                       double, double, const std::string&, int,
                       double, double, double, double, int, double, int, double,
                       int, double, double, int>(),
             py::arg("dynamics"),
             py::arg("nominal_x0_cov"),
             py::arg("nominal_Sigma_w"),
             py::arg("nominal_Sigma_v"),
             py::arg("nominal_mu_w"),
             py::arg("nominal_mu_v"),
             py::arg("theta_x"),
             py::arg("theta_v"),
             py::arg("solver") = "fw_exact",
             py::arg("dr_update_period") = 1,
             py::arg("fw_beta_minus1") = 1.0,
             py::arg("fw_tau") = 2.0,
             py::arg("fw_zeta") = 2.0,
             py::arg("fw_delta") = 0.05,
             py::arg("fw_max_iters") = 50,
             py::arg("fw_gap_tol") = 1e-4,
             py::arg("fw_bisect_max_iters") = 30,
             py::arg("fw_bisect_tol") = 1e-6,
             py::arg("fw_exact_iters") = 8,
             py::arg("fw_exact_gap_tol") = 1e-6,
             py::arg("oracle_exact_tol") = 1e-8,
             py::arg("oracle_exact_max_iters") = 60,
             py::keep_alive<1, 2>())  // Keep dynamics alive
        .def("initial_update", &DR_EKF_CDC::initial_update)
        .def("update_step", &DR_EKF_CDC::update_step)
        .def("get_P", &DR_EKF_CDC::get_P)
        .def("get_nx", &DR_EKF_CDC::get_nx)
        .def("get_ny", &DR_EKF_CDC::get_ny)
        .def("get_dr_solve_count", &DR_EKF_CDC::get_dr_solve_count)
        .def("get_dr_step_count", &DR_EKF_CDC::get_dr_step_count);

    // --- DR_EKF_TAC ---
    py::class_<DR_EKF_TAC>(m, "DR_EKF_TAC")
        .def(py::init<const DynamicsInterface&, const MatXd&, const MatXd&, const MatXd&,
                       const VecXd&, const VecXd&,
                       double, double, double, const std::string&, int,
                       double, double, double, double, int, double, int, double,
                       int, double, double, int>(),
             py::arg("dynamics"),
             py::arg("nominal_x0_cov"),
             py::arg("nominal_Sigma_w"),
             py::arg("nominal_Sigma_v"),
             py::arg("nominal_mu_w"),
             py::arg("nominal_mu_v"),
             py::arg("theta_x"),
             py::arg("theta_v"),
             py::arg("theta_w"),
             py::arg("solver") = "mosek",
             py::arg("dr_update_period") = 1,
             py::arg("fw_beta_minus1") = 1.0,
             py::arg("fw_tau") = 2.0,
             py::arg("fw_zeta") = 2.0,
             py::arg("fw_delta") = 0.05,
             py::arg("fw_max_iters") = 50,
             py::arg("fw_gap_tol") = 1e-4,
             py::arg("fw_bisect_max_iters") = 30,
             py::arg("fw_bisect_tol") = 1e-6,
             py::arg("fw_exact_iters") = 8,
             py::arg("fw_exact_gap_tol") = 1e-6,
             py::arg("oracle_exact_tol") = 1e-8,
             py::arg("oracle_exact_max_iters") = 60,
             py::keep_alive<1, 2>())  // Keep dynamics alive
        .def("initial_update", &DR_EKF_TAC::initial_update)
        .def("update_step", &DR_EKF_TAC::update_step)
        .def("get_P", &DR_EKF_TAC::get_P)
        .def("get_nx", &DR_EKF_TAC::get_nx)
        .def("get_ny", &DR_EKF_TAC::get_ny)
        .def("get_dr_solve_count", &DR_EKF_TAC::get_dr_solve_count)
        .def("get_dr_step_count", &DR_EKF_TAC::get_dr_step_count);

    // --- Standalone utility functions (for testing) ---
    m.def("posterior_cov", &posterior_cov, "Compute posterior covariance");
    m.def("kalman_stats", [](const MatXd& X_pred, const MatXd& Sigma_v, const MatXd& C_t) {
        auto result = kalman_stats(X_pred, Sigma_v, C_t);
        return py::make_tuple(result.trace_post, result.D_x, result.D_v);
    }, "Compute Kalman statistics (trace_post, D_x, D_v)");
    m.def("oracle_bisection_fast", &oracle_bisection_fast,
          py::arg("Sigma_hat"), py::arg("rho"),
          py::arg("Sigma_ref"), py::arg("D"), py::arg("delta"),
          py::arg("bisect_max_iters") = 30, py::arg("bisect_tol") = 1e-6);
    m.def("oracle_exact_fast", &oracle_exact_fast,
          py::arg("Sigma_hat"), py::arg("rho"), py::arg("D"),
          py::arg("tol") = 1e-8, py::arg("max_iters") = 60);
}
