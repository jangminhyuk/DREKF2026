/*
 * MOSEK C API-based SDP solver for DR-EKF CDC and TAC formulations.
 *
 * Uses bar variables (semidefinite variables) for symmetric PSD matrices,
 * linear constraints via putbarablocktriplet, and affine conic constraints
 * (ACC) with SVEC PSD cone domains for LMI constraints.
 *
 * MOSEK barvar storage: lower-triangular, column-major.
 * barx[j] returns d*(d+1)/2 values for a d×d bar variable.
 * The mapping: for column c=0..d-1, row r=c..d-1:
 *   index = c*d - c*(c-1)/2 + (r - c)
 * This is the UNSCALED lower triangle. To get the matrix:
 *   M(r,c) = M(c,r) = barx[index]
 */

#include "dr_ekf/mosek_sdp_solver.h"
#include "dr_ekf/kalman_utils.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace dr_ekf {
namespace mosek_solver {

static const double SQRT2 = std::sqrt(2.0);

// barvar lower-triangle index for entry (r,c) with r >= c, in a dim x dim matrix
static inline MSKint32t bar_idx(int r, int c, int dim) {
    // column-major lower triangle
    return c * dim - c * (c - 1) / 2 + (r - c);
}

// svec dimension for n x n symmetric matrix
static inline int svec_dim(int n) { return n * (n + 1) / 2; }

// svec index for entry (r,c) with r >= c (same formula as bar_idx)
static inline int svec_idx(int r, int c, int n) {
    return c * n - c * (c - 1) / 2 + (r - c);
}

// Extract symmetric matrix from barx solution (unscaled lower triangle)
static MatXd barx_to_mat(const double* barx, int dim) {
    MatXd M(dim, dim);
    int idx = 0;
    for (int c = 0; c < dim; ++c) {
        for (int r = c; r < dim; ++r) {
            M(r, c) = barx[idx];
            M(c, r) = barx[idx];
            ++idx;
        }
    }
    return M;
}

// ============================================================
// CDC Solver
// ============================================================

CDCSolver::CDCSolver(int nx, int ny) : nx_(nx), ny_(ny), env_(nullptr) {
    check_res(MSK_makeenv(&env_, nullptr), "MSK_makeenv");
}

CDCSolver::~CDCSolver() {
    if (env_) MSK_deleteenv(&env_);
}

SDPResultCDC CDCSolver::solve(const MatXd& X_pred_hat, const MatXd& Sigma_v_hat,
                               double theta_x, double theta_v, const MatXd& C_t) {
    /*
     * CDC SDP (maximize trace(X)):
     *
     * Bar variables (symmetric PSD):
     *   0: X         (nx × nx) — posterior covariance
     *   1: X_pred    (nx × nx) — prior covariance
     *   2: Sigma_v   (ny × ny) — measurement noise covariance
     *   3: Y         (nx × nx) — Wasserstein auxiliary for X_pred
     *   4: Z         (ny × ny) — Wasserstein auxiliary for Sigma_v
     *
     * Objective: maximize trace(X)
     *
     * Linear constraints (on bar variables):
     *   L0: trace(X_pred) + trace(Y) <= theta_x^2 + trace(X_pred_hat) - 2*trace(Y) ...
     *       Actually: trace(X_pred_hat + X_pred - 2*Y) <= theta_x^2
     *       => trace(X_pred) - 2*trace(Y) <= theta_x^2 - trace(X_pred_hat)
     *   L1: trace(Sigma_v) - 2*trace(Z) <= theta_v^2 - trace(Sigma_v_hat)
     *
     * PSD cone constraints (via ACC + SVEC_PSD):
     *   S0: Schur complement ((nx+ny) × (nx+ny)):
     *       [[X_pred - X,   X_pred*C^T   ],
     *        [C*X_pred,     C*X_pred*C^T + Sigma_v]] >> 0
     *
     *   S1: Wasserstein LMI for X_pred (2*nx × 2*nx):
     *       [[X_pred_hat, Y   ],
     *        [Y^T,        X_pred]] >> 0
     *
     *   S2: Wasserstein LMI for Sigma_v (2*ny × 2*ny):
     *       [[Sigma_v_hat, Z   ],
     *        [Z^T,         Sigma_v]] >> 0
     *
     *   S3: X >> epsilon*I (nx × nx)
     *   S4: X_pred >> lambda_x*I (nx × nx)
     *   S5: Sigma_v >> lambda_v*I (ny × ny)
     */

    MSKtask_t task = nullptr;
    MSKrescodee r;

    try {
        r = MSK_maketask(env_, 0, 0, &task);
        check_res(r, "MSK_maketask");

        // Compute bounds
        Eigen::SelfAdjointEigenSolver<MatXd> eig_x(X_pred_hat);
        double lam_x = std::max(eig_x.eigenvalues()(0), 1e-10);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_v(Sigma_v_hat);
        double lam_v = std::max(eig_v.eigenvalues()(0), 1e-10);
        double eps_pd = 1e-10;

        // Append 5 bar variables: X(nx), X_pred(nx), Sigma_v(ny), Y(nx), Z(ny)
        MSKint32t bar_dims[] = {(MSKint32t)nx_, (MSKint32t)nx_, (MSKint32t)ny_,
                                (MSKint32t)nx_, (MSKint32t)ny_};
        check_res(MSK_appendbarvars(task, 5, bar_dims), "appendbarvars");

        // Bar variable indices
        const int BAR_X = 0, BAR_XPRED = 1, BAR_SV = 2, BAR_Y = 3, BAR_Z = 4;

        // --- Objective: maximize trace(X) ---
        // putbarcblocktriplet: for each diagonal entry of X, add coefficient 1.0
        {
            std::vector<MSKint32t> obj_j(nx_, BAR_X);
            std::vector<MSKint32t> obj_k(nx_), obj_l(nx_);
            std::vector<double> obj_v(nx_, 1.0);
            for (int i = 0; i < nx_; ++i) {
                obj_k[i] = i;
                obj_l[i] = i;
            }
            check_res(MSK_putbarcblocktriplet(task, nx_, obj_j.data(), obj_k.data(),
                                              obj_l.data(), obj_v.data()), "putbarcblocktriplet obj");
        }
        check_res(MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE), "putobjsense");

        // --- Linear constraints ---
        // 2 constraints: Wasserstein trace inequalities
        check_res(MSK_appendcons(task, 2), "appendcons");

        // L0: trace(X_pred) - 2*trace(Y) <= theta_x^2 - trace(X_pred_hat)
        double rhs0 = theta_x * theta_x - X_pred_hat.trace();
        check_res(MSK_putconbound(task, 0, MSK_BK_UP, -MSK_INFINITY, rhs0), "putconbound L0");

        // L1: trace(Sigma_v) - 2*trace(Z) <= theta_v^2 - trace(Sigma_v_hat)
        double rhs1 = theta_v * theta_v - Sigma_v_hat.trace();
        check_res(MSK_putconbound(task, 1, MSK_BK_UP, -MSK_INFINITY, rhs1), "putconbound L1");

        // Build triplets for linear constraint coefficients on bar variables
        {
            std::vector<MSKint32t> ai, aj, ak, al;
            std::vector<double> av;

            // L0: +trace(X_pred) => +1 on diagonal of bar var 1
            for (int i = 0; i < nx_; ++i) {
                ai.push_back(0); aj.push_back(BAR_XPRED); ak.push_back(i); al.push_back(i); av.push_back(1.0);
            }
            // L0: -2*trace(Y) => -2 on diagonal of bar var 3
            for (int i = 0; i < nx_; ++i) {
                ai.push_back(0); aj.push_back(BAR_Y); ak.push_back(i); al.push_back(i); av.push_back(-2.0);
            }
            // L1: +trace(Sigma_v) => +1 on diagonal of bar var 2
            for (int i = 0; i < ny_; ++i) {
                ai.push_back(1); aj.push_back(BAR_SV); ak.push_back(i); al.push_back(i); av.push_back(1.0);
            }
            // L1: -2*trace(Z) => -2 on diagonal of bar var 4
            for (int i = 0; i < ny_; ++i) {
                ai.push_back(1); aj.push_back(BAR_Z); ak.push_back(i); al.push_back(i); av.push_back(-2.0);
            }

            check_res(MSK_putbarablocktriplet(task, (MSKint32t)ai.size(),
                                              ai.data(), aj.data(), ak.data(), al.data(), av.data()),
                      "putbarablocktriplet linear");
        }

        // --- PSD cone constraints via ACC ---

        // Compute total AFE rows needed
        int schur_n = nx_ + ny_;
        int svec_schur = svec_dim(schur_n);
        int svec_wass_x = svec_dim(2 * nx_);
        int svec_wass_v = svec_dim(2 * ny_);
        int svec_X_pd = svec_dim(nx_);
        int svec_Xp_pd = svec_dim(nx_);
        int svec_Sv_pd = svec_dim(ny_);

        MSKint64t total_afe = svec_schur + svec_wass_x + svec_wass_v + svec_X_pd + svec_Xp_pd + svec_Sv_pd;
        check_res(MSK_appendafes(task, total_afe), "appendafes");

        // AFE row offsets for each PSD cone
        MSKint64t afe_schur = 0;
        MSKint64t afe_wass_x = afe_schur + svec_schur;
        MSKint64t afe_wass_v = afe_wass_x + svec_wass_x;
        MSKint64t afe_X_pd = afe_wass_v + svec_wass_v;
        MSKint64t afe_Xp_pd = afe_X_pd + svec_X_pd;
        MSKint64t afe_Sv_pd = afe_Xp_pd + svec_Xp_pd;

        // Build all barF triplets
        // barF connects bar variables to AFE rows
        // Format: (afe_index, barvar_index, row_in_barvar, col_in_barvar, value)
        // The value should include the svec scaling: sqrt(2) for off-diagonal in the PSD cone svec
        std::vector<MSKint64t> bf_afe;
        std::vector<MSKint32t> bf_bar, bf_k, bf_l;
        std::vector<double> bf_v;

        auto add_barF = [&](MSKint64t afe, MSKint32t barvar, MSKint32t k, MSKint32t l, double val) {
            if (std::abs(val) < 1e-16) return;
            // Ensure k >= l for lower triangle
            if (k < l) std::swap(k, l);
            bf_afe.push_back(afe);
            bf_bar.push_back(barvar);
            bf_k.push_back(k);
            bf_l.push_back(l);
            bf_v.push_back(val);
        };

        // Also collect constant terms (g vector for AFE)
        std::vector<MSKint64t> g_idx;
        std::vector<double> g_val;

        auto add_g = [&](MSKint64t afe, double val) {
            if (std::abs(val) < 1e-16) return;
            g_idx.push_back(afe);
            g_val.push_back(val);
        };

        // ============================================================
        // S0: Schur complement LMI ((nx+ny) × (nx+ny))
        // [[X_pred - X,   X_pred*C^T   ],
        //  [C*X_pred,     C*X_pred*C^T + Sigma_v]] >> 0
        //
        // The AFE for svec entry (a,b) with a >= b:
        //   svec_scale = (a==b) ? 1.0 : sqrt(2)
        // ============================================================

        // Top-left block (0:nx, 0:nx): X_pred - X
        for (int c = 0; c < nx_; ++c) {
            for (int r = c; r < nx_; ++r) {
                MSKint64t afe = afe_schur + svec_idx(r, c, schur_n);
                double sc = (r == c) ? 1.0 : SQRT2;
                double bar_scale = (r == c) ? 1.0 : 0.5;
                // +X_pred(r,c) with svec scaling and bar_scale compensation
                add_barF(afe, BAR_XPRED, r, c, sc * bar_scale);
                // -X(r,c) with svec scaling and bar_scale compensation
                add_barF(afe, BAR_X, r, c, -sc * bar_scale);
            }
        }

        // Off-diagonal block: position (nx+j_b, i) for i in [0,nx), j_b in [0,ny)
        // This stores X_pred * C^T at row i, col j_b of the off-diagonal
        // In the Schur block: entry (nx+j_b, i), always off-diagonal (nx+j_b > i when nx >= 1)
        // (X_pred * C^T)(i, j_b) = sum_k X_pred(i,k) * C(j_b, k)
        for (int j_b = 0; j_b < ny_; ++j_b) {
            for (int i = 0; i < nx_; ++i) {
                int a = nx_ + j_b;  // row in schur block
                int b = i;          // col in schur block, a > b always
                MSKint64t afe = afe_schur + svec_idx(a, b, schur_n);
                double sc = SQRT2;  // always off-diagonal

                // (X_pred * C^T)(i, j_b) = sum_k X_pred(i,k) * C(j_b,k)
                // X_pred(i,k) maps to barvar entry (max(i,k), min(i,k))
                // with coefficient 1 if i==k, 2 if i!=k (because barvar stores lower tri,
                // and <E_ij, X> = X(i,j) for i==j, but <E_ij, X> = X(i,j)+X(j,i) = 2*X(i,j) for i!=j
                // when E_ij has 1 at both (i,j) and (j,i))
                // Wait: putafebarfblocktriplet sets coefficient for the (k,l) entry of the barvar.
                // The inner product is: sum over lower triangle: coeff(r,c) * barx(r,c) * (r==c ? 1 : 2)
                // So if we want to pick up barX(i,k) and i!=k, we set coeff at (max(i,k),min(i,k))
                // and the effective contribution is coeff * 2 * barX(max,min) = coeff * 2 * X(i,k)
                // To get exactly C(j_b,k) * X(i,k), we need coeff = C(j_b,k) / 2 for off-diag,
                // and coeff = C(j_b,k) for diagonal.
                // Then multiply by svec scaling sc.

                for (int k = 0; k < nx_; ++k) {
                    double C_jk = C_t(j_b, k);
                    if (std::abs(C_jk) < 1e-15) continue;

                    int rr = std::max(i, k);
                    int cc = std::min(i, k);
                    double bar_scale = (rr == cc) ? 1.0 : 0.5;  // compensate for the 2x in inner product
                    add_barF(afe, BAR_XPRED, rr, cc, sc * C_jk * bar_scale);
                }
            }
        }

        // Bottom-right block (nx:nx+ny, nx:nx+ny): C*X_pred*C^T + Sigma_v
        for (int a = 0; a < ny_; ++a) {
            for (int b = 0; b <= a; ++b) {
                MSKint64t afe = afe_schur + svec_idx(nx_ + a, nx_ + b, schur_n);
                double sc = (a == b) ? 1.0 : SQRT2;

                // +Sigma_v(a,b)
                {
                    int rr = std::max(a, b);
                    int cc = std::min(a, b);
                    double bar_scale = (rr == cc) ? 1.0 : 0.5;
                    add_barF(afe, BAR_SV, rr, cc, sc * bar_scale);
                }

                // +C*X_pred*C^T (a,b) = sum_{I,J} C(a,I)*C(b,J)*X_pred(I,J)
                for (int I = 0; I < nx_; ++I) {
                    for (int J = 0; J <= I; ++J) {
                        double coeff = C_t(a, I) * C_t(b, J);
                        if (I != J) coeff += C_t(a, J) * C_t(b, I);
                        if (std::abs(coeff) < 1e-15) continue;

                        double bar_scale = (I == J) ? 1.0 : 0.5;
                        add_barF(afe, BAR_XPRED, I, J, sc * coeff * bar_scale);
                    }
                }
            }
        }

        // ============================================================
        // S1: Wasserstein LMI for X_pred (2*nx × 2*nx)
        // [[X_pred_hat, Y   ],
        //  [Y^T,        X_pred]] >> 0
        // ============================================================
        int wass_x_n = 2 * nx_;

        // Top-left (0:nx, 0:nx): X_pred_hat (constant)
        for (int c = 0; c < nx_; ++c) {
            for (int r = c; r < nx_; ++r) {
                MSKint64t afe = afe_wass_x + svec_idx(r, c, wass_x_n);
                double sc = (r == c) ? 1.0 : SQRT2;
                add_g(afe, sc * X_pred_hat(r, c));
            }
        }

        // Bottom-right (nx:2nx, nx:2nx): X_pred (bar variable)
        for (int c = 0; c < nx_; ++c) {
            for (int r = c; r < nx_; ++r) {
                MSKint64t afe = afe_wass_x + svec_idx(nx_ + r, nx_ + c, wass_x_n);
                double sc = (r == c) ? 1.0 : SQRT2;
                double bar_scale = (r == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_XPRED, r, c, sc * bar_scale);
            }
        }

        // Off-diagonal: Y at position (nx+a, i) for a,i in [0,nx)
        // Y is bar variable 3 (nx x nx), entry Y(i, a) maps to position (nx+a, i) in the block
        // Since nx+a > i always, this is lower triangle.
        for (int a = 0; a < nx_; ++a) {
            for (int i = 0; i < nx_; ++i) {
                int block_r = nx_ + a;
                int block_c = i;
                // block_r > block_c always since nx >= 1
                MSKint64t afe = afe_wass_x + svec_idx(block_r, block_c, wass_x_n);
                double sc = SQRT2;  // always off-diagonal

                // Y(i, a): in the barvar, Y is stored as symmetric, so Y(i,a) = barY(max(i,a),min(i,a))
                int rr = std::max(i, a);
                int cc = std::min(i, a);
                double bar_scale = (rr == cc) ? 1.0 : 0.5;
                add_barF(afe, BAR_Y, rr, cc, sc * bar_scale);
            }
        }

        // ============================================================
        // S2: Wasserstein LMI for Sigma_v (2*ny × 2*ny)
        // [[Sigma_v_hat, Z   ],
        //  [Z^T,         Sigma_v]] >> 0
        // ============================================================
        int wass_v_n = 2 * ny_;

        // Top-left: Sigma_v_hat (constant)
        for (int c = 0; c < ny_; ++c) {
            for (int r = c; r < ny_; ++r) {
                MSKint64t afe = afe_wass_v + svec_idx(r, c, wass_v_n);
                double sc = (r == c) ? 1.0 : SQRT2;
                add_g(afe, sc * Sigma_v_hat(r, c));
            }
        }

        // Bottom-right: Sigma_v
        for (int c = 0; c < ny_; ++c) {
            for (int r = c; r < ny_; ++r) {
                MSKint64t afe = afe_wass_v + svec_idx(ny_ + r, ny_ + c, wass_v_n);
                double sc = (r == c) ? 1.0 : SQRT2;
                double bar_scale = (r == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_SV, r, c, sc * bar_scale);
            }
        }

        // Off-diagonal: Z at position (ny+a, i)
        for (int a = 0; a < ny_; ++a) {
            for (int i = 0; i < ny_; ++i) {
                int block_r = ny_ + a;
                int block_c = i;
                MSKint64t afe = afe_wass_v + svec_idx(block_r, block_c, wass_v_n);
                double sc = SQRT2;

                int rr = std::max(i, a);
                int cc = std::min(i, a);
                double bar_scale = (rr == cc) ? 1.0 : 0.5;
                add_barF(afe, BAR_Z, rr, cc, sc * bar_scale);
            }
        }

        // ============================================================
        // S3: X >> eps*I  =>  X - eps*I >> 0
        // ============================================================
        for (int c = 0; c < nx_; ++c) {
            for (int r = c; r < nx_; ++r) {
                MSKint64t afe = afe_X_pd + svec_idx(r, c, nx_);
                double sc = (r == c) ? 1.0 : SQRT2;
                double bar_scale = (r == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_X, r, c, sc * bar_scale);
                if (r == c) {
                    add_g(afe, -eps_pd);
                }
            }
        }

        // ============================================================
        // S4: X_pred >> lam_x*I
        // ============================================================
        for (int c = 0; c < nx_; ++c) {
            for (int r = c; r < nx_; ++r) {
                MSKint64t afe = afe_Xp_pd + svec_idx(r, c, nx_);
                double sc = (r == c) ? 1.0 : SQRT2;
                double bar_scale = (r == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_XPRED, r, c, sc * bar_scale);
                if (r == c) {
                    add_g(afe, -lam_x);
                }
            }
        }

        // ============================================================
        // S5: Sigma_v >> lam_v*I
        // ============================================================
        for (int c = 0; c < ny_; ++c) {
            for (int r = c; r < ny_; ++r) {
                MSKint64t afe = afe_Sv_pd + svec_idx(r, c, ny_);
                double sc = (r == c) ? 1.0 : SQRT2;
                double bar_scale = (r == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_SV, r, c, sc * bar_scale);
                if (r == c) {
                    add_g(afe, -lam_v);
                }
            }
        }

        // Submit barF triplets
        if (!bf_afe.empty()) {
            check_res(MSK_putafebarfblocktriplet(task, (MSKint64t)bf_afe.size(),
                                                  bf_afe.data(), bf_bar.data(),
                                                  bf_k.data(), bf_l.data(), bf_v.data()),
                      "putafebarfblocktriplet");
        }

        // Submit g values
        for (size_t i = 0; i < g_idx.size(); ++i) {
            check_res(MSK_putafeg(task, g_idx[i], g_val[i]), "putafeg");
        }

        // Create SVEC PSD cone domains and ACC
        // MSK_appendsvecpsdconedomain takes the svec dimension n*(n+1)/2, not the matrix dimension n
        auto append_psd_acc = [&](MSKint64t afe_start, int mat_dim) {
            MSKint64t domidx;
            int svec_d = svec_dim(mat_dim);
            check_res(MSK_appendsvecpsdconedomain(task, svec_d, &domidx), "appendsvecpsdconedomain");
            std::vector<MSKint64t> afe_list(svec_d);
            for (int i = 0; i < svec_d; ++i) afe_list[i] = afe_start + i;
            check_res(MSK_appendacc(task, domidx, svec_d, afe_list.data(), nullptr), "appendacc");
        };

        append_psd_acc(afe_schur, schur_n);
        append_psd_acc(afe_wass_x, wass_x_n);
        append_psd_acc(afe_wass_v, wass_v_n);
        append_psd_acc(afe_X_pd, nx_);
        append_psd_acc(afe_Xp_pd, nx_);
        append_psd_acc(afe_Sv_pd, ny_);

        // Solve
        MSKrescodee trmcode;
        check_res(MSK_optimizetrm(task, &trmcode), "optimizetrm");

        // Check solution status
        MSKsolstae solsta;
        check_res(MSK_getsolsta(task, MSK_SOL_ITR, &solsta), "getsolsta");

        if (solsta != MSK_SOL_STA_OPTIMAL && solsta != MSK_SOL_STA_PRIM_AND_DUAL_FEAS) {
            MSK_deletetask(&task);
            return {MatXd::Zero(ny_, ny_), MatXd::Zero(nx_, nx_), MatXd::Zero(nx_, nx_), false};
        }

        // Extract solution
        int len_X = svec_dim(nx_);
        int len_Sv = svec_dim(ny_);

        std::vector<double> barx_X(len_X), barx_Xp(len_X), barx_Sv(len_Sv);
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_X, barx_X.data()), "getbarxj X");
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_XPRED, barx_Xp.data()), "getbarxj Xpred");
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_SV, barx_Sv.data()), "getbarxj Sv");

        MatXd wc_Xpost = sym(barx_to_mat(barx_X.data(), nx_));
        MatXd wc_Xprior = sym(barx_to_mat(barx_Xp.data(), nx_));
        MatXd wc_Sigma_v = sym(barx_to_mat(barx_Sv.data(), ny_));

        MSK_deletetask(&task);
        return {wc_Sigma_v, wc_Xprior, wc_Xpost, true};

    } catch (const std::exception& e) {
        if (task) MSK_deletetask(&task);
        std::cerr << "MOSEK CDC solve error: " << e.what() << std::endl;
        return {MatXd::Zero(ny_, ny_), MatXd::Zero(nx_, nx_), MatXd::Zero(nx_, nx_), false};
    }
}

// ============================================================
// TAC Solver
// ============================================================

TACSolver::TACSolver(int nx, int ny) : nx_(nx), ny_(ny), env_(nullptr) {
    check_res(MSK_makeenv(&env_, nullptr), "MSK_makeenv");
}

TACSolver::~TACSolver() {
    if (env_) MSK_deleteenv(&env_);
}

SDPResultTAC TACSolver::solve(const MatXd& X_post_prev, const MatXd& A_t, const MatXd& C_t,
                               const MatXd& Sigma_w_hat, const MatXd& Sigma_v_hat,
                               double theta_w, double theta_v) {
    /*
     * TAC regular SDP (t > 0):
     *
     * Bar variables (symmetric PSD):
     *   0: X         (nx × nx) — posterior covariance
     *   1: X_pred    (nx × nx) — prior covariance
     *   2: Sigma_v   (ny × ny) — measurement noise covariance
     *   3: Sigma_w   (nx × nx) — process noise covariance
     *   4: W         (nx × nx) — Wasserstein auxiliary for Sigma_w
     *   5: Z         (ny × ny) — Wasserstein auxiliary for Sigma_v
     *
     * Objective: maximize trace(X)
     *
     * Linear constraints:
     *   E0 (equality): X_pred = A * X_post_prev * A^T + Sigma_w
     *       => X_pred(r,c) - Sigma_w(r,c) = (A*X_post_prev*A^T)(r,c)  for all (r,c) in lower triangle
     *       One constraint per lower-triangle entry of nx x nx matrix
     *   L0: trace(Sigma_w) - 2*trace(W) <= theta_w^2 - trace(Sigma_w_hat)
     *   L1: trace(Sigma_v) - 2*trace(Z) <= theta_v^2 - trace(Sigma_v_hat)
     *
     * PSD cones (same structure as CDC but with Sigma_w additions):
     *   S0: Schur complement ((nx+ny) × (nx+ny))
     *   S1: Wasserstein LMI for Sigma_w (2*nx × 2*nx)
     *   S2: Wasserstein LMI for Sigma_v (2*ny × 2*ny)
     *   S3: X >> eps*I
     *   S4: Sigma_v >> lam_v*I
     *   S5: Sigma_w >> lam_w*I
     */

    MSKtask_t task = nullptr;
    MSKrescodee r;

    try {
        r = MSK_maketask(env_, 0, 0, &task);
        check_res(r, "MSK_maketask");

        // Compute bounds
        Eigen::SelfAdjointEigenSolver<MatXd> eig_v(Sigma_v_hat);
        double lam_v = std::max(eig_v.eigenvalues()(0), 1e-10);
        Eigen::SelfAdjointEigenSolver<MatXd> eig_w(Sigma_w_hat);
        double lam_w = std::max(eig_w.eigenvalues()(0), 1e-10);
        double eps_pd = 1e-10;

        // Precompute A * X_post_prev * A^T
        MatXd AXpA = sym(A_t * X_post_prev * A_t.transpose());

        // Append 6 bar variables
        MSKint32t bar_dims[] = {(MSKint32t)nx_, (MSKint32t)nx_, (MSKint32t)ny_,
                                (MSKint32t)nx_, (MSKint32t)nx_, (MSKint32t)ny_};
        check_res(MSK_appendbarvars(task, 6, bar_dims), "appendbarvars");

        const int BAR_X = 0, BAR_XPRED = 1, BAR_SV = 2, BAR_SW = 3, BAR_W = 4, BAR_Z = 5;

        // --- Objective: maximize trace(X) ---
        {
            std::vector<MSKint32t> obj_j(nx_, BAR_X);
            std::vector<MSKint32t> obj_k(nx_), obj_l(nx_);
            std::vector<double> obj_v(nx_, 1.0);
            for (int i = 0; i < nx_; ++i) {
                obj_k[i] = i;
                obj_l[i] = i;
            }
            check_res(MSK_putbarcblocktriplet(task, nx_, obj_j.data(), obj_k.data(),
                                              obj_l.data(), obj_v.data()), "putbarcblocktriplet obj");
        }
        check_res(MSK_putobjsense(task, MSK_OBJECTIVE_SENSE_MAXIMIZE), "putobjsense");

        // --- Linear constraints ---
        // Equality: X_pred - Sigma_w = AXpA  (one constraint per lower-triangle entry)
        int n_eq = svec_dim(nx_);
        // Total constraints: n_eq equality + 2 inequality
        int n_cons = n_eq + 2;
        check_res(MSK_appendcons(task, n_cons), "appendcons");

        // Equality constraint bounds
        {
            int idx = 0;
            for (int c = 0; c < nx_; ++c) {
                for (int r_idx = c; r_idx < nx_; ++r_idx) {
                    double rhs = AXpA(r_idx, c);
                    check_res(MSK_putconbound(task, idx, MSK_BK_FX, rhs, rhs), "putconbound eq");
                    ++idx;
                }
            }
        }

        // Wasserstein trace inequalities
        int con_L0 = n_eq;
        int con_L1 = n_eq + 1;
        double rhs_w = theta_w * theta_w - Sigma_w_hat.trace();
        double rhs_v = theta_v * theta_v - Sigma_v_hat.trace();
        check_res(MSK_putconbound(task, con_L0, MSK_BK_UP, -MSK_INFINITY, rhs_w), "putconbound L0");
        check_res(MSK_putconbound(task, con_L1, MSK_BK_UP, -MSK_INFINITY, rhs_v), "putconbound L1");

        // Build triplets for all linear constraints
        {
            std::vector<MSKint32t> ai, aj, ak, al;
            std::vector<double> av;

            // Equality: X_pred(r,c) - Sigma_w(r,c) = AXpA(r,c)
            // For bar variable inner product:
            //   <E_rc, X> = X(r,c) if r==c, else X(r,c) + X(c,r) = 2*X(r,c) if r!=c
            // We want coefficient 1 on X_pred(r,c), so:
            //   coeff_bar = 1 for diagonal, 0.5 for off-diagonal
            {
                int idx = 0;
                for (int c = 0; c < nx_; ++c) {
                    for (int r_idx = c; r_idx < nx_; ++r_idx) {
                        double bar_scale = (r_idx == c) ? 1.0 : 0.5;
                        // +X_pred
                        ai.push_back(idx); aj.push_back(BAR_XPRED); ak.push_back(r_idx); al.push_back(c); av.push_back(bar_scale);
                        // -Sigma_w
                        ai.push_back(idx); aj.push_back(BAR_SW); ak.push_back(r_idx); al.push_back(c); av.push_back(-bar_scale);
                        ++idx;
                    }
                }
            }

            // L0: trace(Sigma_w) - 2*trace(W)
            for (int i = 0; i < nx_; ++i) {
                ai.push_back(con_L0); aj.push_back(BAR_SW); ak.push_back(i); al.push_back(i); av.push_back(1.0);
            }
            for (int i = 0; i < nx_; ++i) {
                ai.push_back(con_L0); aj.push_back(BAR_W); ak.push_back(i); al.push_back(i); av.push_back(-2.0);
            }

            // L1: trace(Sigma_v) - 2*trace(Z)
            for (int i = 0; i < ny_; ++i) {
                ai.push_back(con_L1); aj.push_back(BAR_SV); ak.push_back(i); al.push_back(i); av.push_back(1.0);
            }
            for (int i = 0; i < ny_; ++i) {
                ai.push_back(con_L1); aj.push_back(BAR_Z); ak.push_back(i); al.push_back(i); av.push_back(-2.0);
            }

            check_res(MSK_putbarablocktriplet(task, (MSKint32t)ai.size(),
                                              ai.data(), aj.data(), ak.data(), al.data(), av.data()),
                      "putbarablocktriplet linear");
        }

        // --- PSD cone constraints via ACC ---

        int schur_n = nx_ + ny_;
        int svec_schur = svec_dim(schur_n);
        int wass_w_n = 2 * nx_;
        int svec_wass_w = svec_dim(wass_w_n);
        int wass_v_n = 2 * ny_;
        int svec_wass_v = svec_dim(wass_v_n);
        int svec_X_pd = svec_dim(nx_);
        int svec_Xp_pd = svec_dim(nx_);  // X_pred >> 0
        int svec_Sv_pd = svec_dim(ny_);
        int svec_Sw_pd = svec_dim(nx_);

        MSKint64t total_afe = svec_schur + svec_wass_w + svec_wass_v + svec_X_pd + svec_Xp_pd + svec_Sv_pd + svec_Sw_pd;
        check_res(MSK_appendafes(task, total_afe), "appendafes");

        MSKint64t afe_schur = 0;
        MSKint64t afe_wass_w = afe_schur + svec_schur;
        MSKint64t afe_wass_v = afe_wass_w + svec_wass_w;
        MSKint64t afe_X_pd = afe_wass_v + svec_wass_v;
        MSKint64t afe_Xp_pd = afe_X_pd + svec_X_pd;
        MSKint64t afe_Sv_pd = afe_Xp_pd + svec_Xp_pd;
        MSKint64t afe_Sw_pd = afe_Sv_pd + svec_Sv_pd;

        std::vector<MSKint64t> bf_afe;
        std::vector<MSKint32t> bf_bar, bf_k, bf_l;
        std::vector<double> bf_v;

        auto add_barF = [&](MSKint64t afe, MSKint32t barvar, MSKint32t k, MSKint32t l, double val) {
            if (std::abs(val) < 1e-16) return;
            if (k < l) std::swap(k, l);
            bf_afe.push_back(afe);
            bf_bar.push_back(barvar);
            bf_k.push_back(k);
            bf_l.push_back(l);
            bf_v.push_back(val);
        };

        std::vector<MSKint64t> g_idx;
        std::vector<double> g_val;

        auto add_g = [&](MSKint64t afe, double val) {
            if (std::abs(val) < 1e-16) return;
            g_idx.push_back(afe);
            g_val.push_back(val);
        };

        // ============================================================
        // S0: Schur complement LMI (same as CDC)
        // ============================================================

        // Top-left: X_pred - X
        for (int c = 0; c < nx_; ++c) {
            for (int r_i = c; r_i < nx_; ++r_i) {
                MSKint64t afe = afe_schur + svec_idx(r_i, c, schur_n);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_XPRED, r_i, c, sc * bar_scale);
                add_barF(afe, BAR_X, r_i, c, -sc * bar_scale);
            }
        }

        // Off-diagonal: X_pred * C^T
        for (int j_b = 0; j_b < ny_; ++j_b) {
            for (int i = 0; i < nx_; ++i) {
                int a = nx_ + j_b;
                int b = i;
                MSKint64t afe = afe_schur + svec_idx(a, b, schur_n);
                double sc = SQRT2;

                for (int k = 0; k < nx_; ++k) {
                    double C_jk = C_t(j_b, k);
                    if (std::abs(C_jk) < 1e-15) continue;
                    int rr = std::max(i, k);
                    int cc = std::min(i, k);
                    double bar_scale = (rr == cc) ? 1.0 : 0.5;
                    add_barF(afe, BAR_XPRED, rr, cc, sc * C_jk * bar_scale);
                }
            }
        }

        // Bottom-right: C*X_pred*C^T + Sigma_v
        for (int a = 0; a < ny_; ++a) {
            for (int b = 0; b <= a; ++b) {
                MSKint64t afe = afe_schur + svec_idx(nx_ + a, nx_ + b, schur_n);
                double sc = (a == b) ? 1.0 : SQRT2;

                // Sigma_v
                {
                    int rr = std::max(a, b);
                    int cc = std::min(a, b);
                    double bar_scale = (rr == cc) ? 1.0 : 0.5;
                    add_barF(afe, BAR_SV, rr, cc, sc * bar_scale);
                }

                // C*X_pred*C^T
                for (int I = 0; I < nx_; ++I) {
                    for (int J = 0; J <= I; ++J) {
                        double coeff = C_t(a, I) * C_t(b, J);
                        if (I != J) coeff += C_t(a, J) * C_t(b, I);
                        if (std::abs(coeff) < 1e-15) continue;
                        double bar_scale = (I == J) ? 1.0 : 0.5;
                        add_barF(afe, BAR_XPRED, I, J, sc * coeff * bar_scale);
                    }
                }
            }
        }

        // ============================================================
        // S1: Wasserstein LMI for Sigma_w (2*nx × 2*nx)
        // [[Sigma_w_hat, W],
        //  [W^T,         Sigma_w]] >> 0
        // ============================================================

        // Top-left: Sigma_w_hat (constant)
        for (int c = 0; c < nx_; ++c) {
            for (int r_i = c; r_i < nx_; ++r_i) {
                MSKint64t afe = afe_wass_w + svec_idx(r_i, c, wass_w_n);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                add_g(afe, sc * Sigma_w_hat(r_i, c));
            }
        }

        // Bottom-right: Sigma_w
        for (int c = 0; c < nx_; ++c) {
            for (int r_i = c; r_i < nx_; ++r_i) {
                MSKint64t afe = afe_wass_w + svec_idx(nx_ + r_i, nx_ + c, wass_w_n);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_SW, r_i, c, sc * bar_scale);
            }
        }

        // Off-diagonal: W at position (nx+a, i)
        for (int a = 0; a < nx_; ++a) {
            for (int i = 0; i < nx_; ++i) {
                int block_r = nx_ + a;
                int block_c = i;
                MSKint64t afe = afe_wass_w + svec_idx(block_r, block_c, wass_w_n);
                double sc = SQRT2;
                int rr = std::max(i, a);
                int cc = std::min(i, a);
                double bar_scale = (rr == cc) ? 1.0 : 0.5;
                add_barF(afe, BAR_W, rr, cc, sc * bar_scale);
            }
        }

        // ============================================================
        // S2: Wasserstein LMI for Sigma_v (2*ny × 2*ny)
        // ============================================================

        // Top-left: Sigma_v_hat (constant)
        for (int c = 0; c < ny_; ++c) {
            for (int r_i = c; r_i < ny_; ++r_i) {
                MSKint64t afe = afe_wass_v + svec_idx(r_i, c, wass_v_n);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                add_g(afe, sc * Sigma_v_hat(r_i, c));
            }
        }

        // Bottom-right: Sigma_v
        for (int c = 0; c < ny_; ++c) {
            for (int r_i = c; r_i < ny_; ++r_i) {
                MSKint64t afe = afe_wass_v + svec_idx(ny_ + r_i, ny_ + c, wass_v_n);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_SV, r_i, c, sc * bar_scale);
            }
        }

        // Off-diagonal: Z at (ny+a, i)
        for (int a = 0; a < ny_; ++a) {
            for (int i = 0; i < ny_; ++i) {
                int block_r = ny_ + a;
                int block_c = i;
                MSKint64t afe = afe_wass_v + svec_idx(block_r, block_c, wass_v_n);
                double sc = SQRT2;
                int rr = std::max(i, a);
                int cc = std::min(i, a);
                double bar_scale = (rr == cc) ? 1.0 : 0.5;
                add_barF(afe, BAR_Z, rr, cc, sc * bar_scale);
            }
        }

        // ============================================================
        // S3: X >> eps*I
        // ============================================================
        for (int c = 0; c < nx_; ++c) {
            for (int r_i = c; r_i < nx_; ++r_i) {
                MSKint64t afe = afe_X_pd + svec_idx(r_i, c, nx_);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_X, r_i, c, sc * bar_scale);
                if (r_i == c) add_g(afe, -eps_pd);
            }
        }

        // ============================================================
        // S4: X_pred >> 0
        // ============================================================
        for (int c = 0; c < nx_; ++c) {
            for (int r_i = c; r_i < nx_; ++r_i) {
                MSKint64t afe = afe_Xp_pd + svec_idx(r_i, c, nx_);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_XPRED, r_i, c, sc * bar_scale);
            }
        }

        // ============================================================
        // S5: Sigma_v >> lam_v*I
        // ============================================================
        for (int c = 0; c < ny_; ++c) {
            for (int r_i = c; r_i < ny_; ++r_i) {
                MSKint64t afe = afe_Sv_pd + svec_idx(r_i, c, ny_);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_SV, r_i, c, sc * bar_scale);
                if (r_i == c) add_g(afe, -lam_v);
            }
        }

        // ============================================================
        // S6: Sigma_w >> lam_w*I
        // ============================================================
        for (int c = 0; c < nx_; ++c) {
            for (int r_i = c; r_i < nx_; ++r_i) {
                MSKint64t afe = afe_Sw_pd + svec_idx(r_i, c, nx_);
                double sc = (r_i == c) ? 1.0 : SQRT2;
                double bar_scale = (r_i == c) ? 1.0 : 0.5;
                add_barF(afe, BAR_SW, r_i, c, sc * bar_scale);
                if (r_i == c) add_g(afe, -lam_w);
            }
        }

        // Submit barF triplets
        if (!bf_afe.empty()) {
            check_res(MSK_putafebarfblocktriplet(task, (MSKint64t)bf_afe.size(),
                                                  bf_afe.data(), bf_bar.data(),
                                                  bf_k.data(), bf_l.data(), bf_v.data()),
                      "putafebarfblocktriplet");
        }

        // Submit g values
        for (size_t i = 0; i < g_idx.size(); ++i) {
            check_res(MSK_putafeg(task, g_idx[i], g_val[i]), "putafeg");
        }

        // Create PSD cone domains and ACC
        // MSK_appendsvecpsdconedomain takes the svec dimension n*(n+1)/2, not the matrix dimension n
        auto append_psd_acc = [&](MSKint64t afe_start, int mat_dim) {
            MSKint64t domidx;
            int svec_d = svec_dim(mat_dim);
            check_res(MSK_appendsvecpsdconedomain(task, svec_d, &domidx), "appendsvecpsdconedomain");
            std::vector<MSKint64t> afe_list(svec_d);
            for (int i = 0; i < svec_d; ++i) afe_list[i] = afe_start + i;
            check_res(MSK_appendacc(task, domidx, svec_d, afe_list.data(), nullptr), "appendacc");
        };

        append_psd_acc(afe_schur, schur_n);
        append_psd_acc(afe_wass_w, wass_w_n);
        append_psd_acc(afe_wass_v, wass_v_n);
        append_psd_acc(afe_X_pd, nx_);
        append_psd_acc(afe_Xp_pd, nx_);
        append_psd_acc(afe_Sv_pd, ny_);
        append_psd_acc(afe_Sw_pd, nx_);

        // Solve
        MSKrescodee trmcode;
        check_res(MSK_optimizetrm(task, &trmcode), "optimizetrm");

        // Check solution status
        MSKsolstae solsta;
        check_res(MSK_getsolsta(task, MSK_SOL_ITR, &solsta), "getsolsta");

        if (solsta != MSK_SOL_STA_OPTIMAL && solsta != MSK_SOL_STA_PRIM_AND_DUAL_FEAS) {
            MSK_deletetask(&task);
            return {MatXd::Zero(ny_, ny_), MatXd::Zero(nx_, nx_),
                    MatXd::Zero(nx_, nx_), MatXd::Zero(nx_, nx_), false};
        }

        // Extract solution
        int len_nx = svec_dim(nx_);
        int len_ny = svec_dim(ny_);

        std::vector<double> barx_X(len_nx), barx_Xp(len_nx), barx_Sv(len_ny), barx_Sw(len_nx);
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_X, barx_X.data()), "getbarxj X");
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_XPRED, barx_Xp.data()), "getbarxj Xpred");
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_SV, barx_Sv.data()), "getbarxj Sv");
        check_res(MSK_getbarxj(task, MSK_SOL_ITR, BAR_SW, barx_Sw.data()), "getbarxj Sw");

        MatXd wc_Xpost = sym(barx_to_mat(barx_X.data(), nx_));
        MatXd wc_Xprior = sym(barx_to_mat(barx_Xp.data(), nx_));
        MatXd wc_Sigma_v = sym(barx_to_mat(barx_Sv.data(), ny_));
        MatXd wc_Sigma_w = sym(barx_to_mat(barx_Sw.data(), nx_));

        MSK_deletetask(&task);
        return {wc_Sigma_v, wc_Sigma_w, wc_Xprior, wc_Xpost, true};

    } catch (const std::exception& e) {
        if (task) MSK_deletetask(&task);
        std::cerr << "MOSEK TAC solve error: " << e.what() << std::endl;
        return {MatXd::Zero(ny_, ny_), MatXd::Zero(nx_, nx_),
                MatXd::Zero(nx_, nx_), MatXd::Zero(nx_, nx_), false};
    }
}

} // namespace mosek_solver
} // namespace dr_ekf
