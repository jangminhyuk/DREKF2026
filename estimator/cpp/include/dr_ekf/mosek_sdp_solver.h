#pragma once

#include "types.h"
#include <mosek.h>
#include <string>
#include <stdexcept>

namespace dr_ekf {
namespace mosek_solver {

struct SDPResultCDC {
    MatXd wc_Sigma_v, wc_Xprior, wc_Xpost;
    bool success;
};

struct SDPResultTAC {
    MatXd wc_Sigma_v, wc_Sigma_w, wc_Xprior, wc_Xpost;
    bool success;
};

// Check MOSEK return code and throw on error
inline void check_res(MSKrescodee r, const char* msg) {
    if (r != MSK_RES_OK) {
        char symname[MSK_MAX_STR_LEN];
        char desc[MSK_MAX_STR_LEN];
        MSK_getcodedesc(r, symname, desc);
        throw std::runtime_error(std::string(msg) + ": " + symname + " - " + desc);
    }
}

// Solver for CDC initial SDP (also used for TAC t=0)
class CDCSolver {
public:
    CDCSolver(int nx, int ny);
    ~CDCSolver();

    SDPResultCDC solve(const MatXd& X_pred_hat, const MatXd& Sigma_v_hat,
                       double theta_x, double theta_v, const MatXd& C_t);

private:
    int nx_, ny_;
    MSKenv_t env_;
};

// Solver for TAC regular SDP (t > 0)
class TACSolver {
public:
    TACSolver(int nx, int ny);
    ~TACSolver();

    SDPResultTAC solve(const MatXd& X_post_prev, const MatXd& A_t, const MatXd& C_t,
                       const MatXd& Sigma_w_hat, const MatXd& Sigma_v_hat,
                       double theta_w, double theta_v);

private:
    int nx_, ny_;
    MSKenv_t env_;
};

} // namespace mosek_solver
} // namespace dr_ekf
