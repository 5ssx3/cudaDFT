#pragma once
#include <cstddef>

struct Context;
struct finitePlan;

// 通用错误沿用 cuda_math.hpp 中的 Status；Davidson 专用错误从 100 开始。
enum DavidsonStatus : int {
    kDavidsonSuccess = 0,
    kDavidsonCublasError = 100,
    kDavidsonCusolverError,
    kDavidsonCurandError,
    kDavidsonSolverInvalidParameter,
    kDavidsonSolverNoConvergence,
    kDavidsonNonFinite,
    kDavidsonStagnation,
    kDavidsonMaxIterations,
    kDavidsonNotInitialized,
};

const char* davidson_status_message(int status) noexcept;

struct RayleighRitzSubspace {
    double* d_V = nullptr;
    double* d_HV = nullptr;
    double* d_Hhat = nullptr;
    double* d_correction = nullptr;
    int current_subspace_dim = 0;
    //
    double* d_solver_work = nullptr;
    double* d_tau = nullptr;
    int solver_lwork = 0;
    int* d_solver_info = nullptr;
    double* d_ritzValues = nullptr;
    int status = kDavidsonNotInitialized;
};
struct DavidsonPlan {
    int nstates = 0;          // 要求解的本征态数
    int max_subspace_dim = 0;    // 最大子空间维数，要求 > num_states
    int max_iterations = 100;    // 最大迭代次数
    int nextra_retain = 3;
    double* d_residual = nullptr;  // 残差
    double residual_tol = 1e-8;  // 本征残差范数的收敛阈值
    double tNorm_tol = 1e-8;

    int nx{};
    int ny{};
    int nz{};
    std::size_t npoints{};
    double* d_psi = nullptr;         // 设备端轨道：Ritz 轨道/本征态输出
    double* d_eigenvalues = nullptr; // 设备端本征值输出
    //
    RayleighRitzSubspace RRsubspace{};
    int status = kDavidsonNotInitialized; // 初始化状态；求解结果由函数返回。
};

// 求解 H psi_j = eigenvalue_j * psi_j。
//
// d_veff:   GPU 数组，长度 N，求解期间保持不变。
// davidsonPlan.d_psi: GPU 数组，长度 N*s，输出本征态。
// davidsonPlan.d_eigenvalues: GPU 数组，长度 s，输出对应本征值。
//
// N = finite.nx * finite.ny * finite.nz。
// 波函数按列连续存储：
//     d_psi[point + N * state]
//
// 数组由 DavidsonSolver_init 分配、reset_DavidsonSolver 释放。
// 返回值为通用 Status 或 DavidsonStatus；求解函数仅在全部目标态收敛时返回 0。
// 出错时输出状态说明；库调用错误另外输出原生状态、调用表达式和位置。
// 收敛后向 stderr 输出迭代汇总及各目标态的 Ritz 能级、欧氏残差范数和收敛状态。
// 计算阶段失败后，先 reset 再初始化；停滞或达到迭代上限时计划仍可使用。
int DavidsonSolver_init(Context& context, DavidsonPlan& davidsonPlan,int num_states,
            int nx, int ny, int nz, int nextra_retain,
            int max_subspace_dim,int max_iterations,double residual_tol) noexcept;
int reset_DavidsonSolver(DavidsonPlan& davidsonPlan);
int DavidsonSolver(const Context& context, const finitePlan& finite,
                   DavidsonPlan& davidsonPlan, const double* d_veff) noexcept;
