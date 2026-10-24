#include "cuda_math.hpp"
#include "Davidson.hpp"
#include "LBFGS.hpp"
#include "read_input.hpp"
#include "structure.hpp"
#include "write.hpp"

#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>
struct WS_data
{
    int nx{};
    int ny{};
    int nz{};
    size_t npoints{};
    int nstates{};
    double dV{};
    double* d_rho0{};
    double* d_rho{};
    double* d_veff{};
    double* d_eigenvalues{};
    finitePlan finite{};
    DavidsonPlan davidsonPlan{};
};

using calculate_F = int (*)(
    const Context& context,
    const double* d_x,  // 输入：设备上的试探变量
    double& f,          // 输出：主机上的函数值
    double* d_g,        // 输出：设备上的梯度
    void* user_data     // 外部计算所需的数据
);

void creat_WS_data(WS_data& Ws_data,InputParameters& parameters,DavidsonPlan& davidsonPlan,
        StructGrid& structgrid, finitePlan& finite, double* d_rho0)
{
    int nx = structgrid.rgrids[0];
    int ny = structgrid.rgrids[1];
    int nz = structgrid.rgrids[2];
    Ws_data.nx = nx;
    Ws_data.ny = ny;
    Ws_data.nz = nz;
    Ws_data.npoints = davidsonPlan.npoints;
    Ws_data.nstates = davidsonPlan.nstates;
    Ws_data.dV = structgrid.dV;
    Ws_data.d_rho0 = d_rho0;
    Ws_data.d_eigenvalues = davidsonPlan.d_eigenvalues;
    Ws_data.davidsonPlan = davidsonPlan;
    Ws_data.finite = finite;
    cudaMalloc(reinterpret_cast<void**>(&Ws_data.d_rho),size_t(Ws_data.npoints)*sizeof(double));
}
int begin(InputParameters& parameters,Context& context,
            finitePlan& finite, DavidsonPlan& davidsonPlan,
            StructGrid& structgrid, WS_data& Ws_data, double*& d_rho0,
            const std::string& filename, std::string& error)
{
    // read input
    int ierr = read_parameters(filename, parameters, error);
    if (ierr != kSuccess) return ierr;
    // read density
    int nx = parameters.mesh.rgrids[0];
    int ny = parameters.mesh.rgrids[1];
    int nz = parameters.mesh.rgrids[2];
    ierr = creat_StructGrid(parameters, structgrid, error);
    if (ierr != kSuccess) return ierr;
    int nstates = structgrid.electron_number / 2;
    ierr = read_density_input(parameters.system.density_file, structgrid, error);
    if (ierr != kSuccess) return ierr;
    ierr = CUDA_init(context);
    if (ierr != kSuccess) return ierr;
    ierr = upload(structgrid.target_density.data(),reinterpret_cast<void**>(&d_rho0),structgrid.npoints);
    if (ierr != kSuccess) return ierr;
    // init solver
    int extra_states = parameters.inversion.eigensolver.nextra_retain;
    int max_subspace_dim = parameters.inversion.eigensolver.max_subspace_dim;
    int max_iterations = parameters.inversion.eigensolver.max_iterations;
    double residual_tolerance = parameters.inversion.eigensolver.residual_tolerance;
    ierr = DavidsonSolver_init(context, davidsonPlan,nstates,nx, ny, nz,
         extra_states,max_subspace_dim,max_iterations,residual_tolerance);
    if (ierr != kSuccess) return ierr;
    davidsonPlan.tNorm_tol = parameters.inversion.eigensolver.direction_tolerance;
    // init finite
    int half_width = parameters.mesh.finite_difference_halfwidth;
    double gx = structgrid.gap[0];
    double gy = structgrid.gap[1];
    double gz = structgrid.gap[2];
    ierr = creat_finitePlan(finite,nx,ny,nz,half_width,gx,gy,gz);
    if (ierr != kSuccess) return ierr;
    //
    creat_WS_data(Ws_data, parameters, davidsonPlan, structgrid, finite, d_rho0);
    return Ws_data.d_rho ? kSuccess : kAllocationError;
}
void write_SummaryLogs(const LBFGSState& lbfgs, int result,
                       int outer_iterations, int max_iterations);
// lbfgs 由调用方持有，输出完成后再销毁，保留最终接受的势。
int invks_scf(InputParameters& parameters, Context& context, WS_data& Ws_data, LBFGSState& lbfgs)
{
    const int max_interation = parameters.inversion.optimizer.max_iterations;
    int outer_iterations = 0;
    auto finish = [&](int result) {
        write_SummaryLogs(lbfgs, result, outer_iterations, max_interation);
        return result;
    };
    // 准备 context、n、history_capacity、d_x_guess 和 data ...

    calculate_F iks_Ws = [](
        const Context& context,
        const double* d_x,
        double& f,
        double* d_g,
        void* d_user_data
    ) -> int {
        auto& Ws_data = *static_cast<WS_data*>(d_user_data);
        // 根据本次传入的 d_x 计算：
        // 1. 将目标函数值写入主机变量 f；
        // 2. 将梯度写入设备数组 d_g。
        double* d_rho0 = Ws_data.d_rho0;
        double* d_rho = Ws_data.d_rho;
        const double* d_v = d_x;
        double* d_eigenvalues = Ws_data.d_eigenvalues;
        double dV = Ws_data.dV;
        size_t npoints = Ws_data.npoints;
        int nstates = Ws_data.nstates;
        std::vector<double> h_eigenvalues(nstates);
        const finitePlan& finite = Ws_data.finite;
        DavidsonPlan& davidsonPlan = Ws_data.davidsonPlan;
        //
        int ierr = DavidsonSolver(context, finite, davidsonPlan, d_v);
        if (ierr != kSuccess) return ierr;
        double* d_psi = davidsonPlan.d_psi;
        RealArrayMulRealArray(context,d_psi,d_psi,npoints * nstates);
        double alpha = 2.0/dV;
        double beta  = 0.0;
        double* d_ones = nullptr;
        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&d_ones),
            size_t(nstates) * sizeof(double)
        );
        thrust::fill(
            thrust::cuda::par.on(context.stream),
            d_ones,
            d_ones + nstates,
            1.0
        );
        cublasStatus_t status = cublasDgemv(
            context.cublas,
            CUBLAS_OP_N,
            npoints, nstates,
            &alpha,
            d_psi, npoints,
            d_ones, 1,
            &beta,
            d_rho, 1
        );
        status = cublasDcopy(
            context.cublas, npoints,
            d_rho0, 1, d_g, 1);

        const double minus_one = -1.0;
        status = cublasDaxpy(
            context.cublas, npoints,
            &minus_one, d_rho, 1, d_g, 1);

        status = cublasDscal(
            context.cublas, npoints,
            &dV, d_g, 1);
        // RealArrayMulReal(context,npoints,d_g,dV);
        //
        f=0;
        download(d_eigenvalues,h_eigenvalues.data(),nstates);
        for(int i=0;i<nstates;i++) f -= 2.0 * h_eigenvalues[i];
        double* d_vrho = nullptr;
        double h_vrho = 0.0;
        cudaMalloc(reinterpret_cast<void**>(&d_vrho),sizeof(double));
        RealReal_IntegralFGdV(context,d_v,d_rho0,dV,npoints,d_vrho);
        // download 使用默认流，先等待 context.stream 中的积分完成。
        err = cudaStreamSynchronize(context.stream);
        if (err != cudaSuccess) {
            cudaFree(d_ones);
            cudaFree(d_vrho);
            return kCudaSyncError;
        }
        download(d_vrho, &h_vrho, 1);
        f += h_vrho;
        //
        cudaFree(d_ones);
        cudaFree(d_vrho);
        return 0;  // 实际计算成功后返回 0，失败则返回错误码
    };
    ////////////////////////////////////////////////////////////////
    double* d_x_guess = nullptr;
    const size_t nbytes = Ws_data.npoints * sizeof(double);

    if (cudaMalloc(reinterpret_cast<void**>(&d_x_guess), nbytes) != cudaSuccess)
        return finish(kAllocationError);
    if (cudaMemsetAsync(d_x_guess, 0, nbytes, context.stream) != cudaSuccess) {
        cudaFree(d_x_guess);
        return finish(kCudaMemcpyError);
    }
    int n = Ws_data.npoints;
    int history_capacity=parameters.inversion.optimizer.history;
    double g_tol = parameters.inversion.convergence.gradient_tolerance;
    int result = LBFGSOptimizer_init(
        context, lbfgs, n, history_capacity,
        iks_Ws, &Ws_data, d_x_guess, g_tol
    );
    cudaFree(d_x_guess);
    d_x_guess = nullptr;
    if (result != 0)
        return finish(result);
    for(int i=0;i<max_interation;i++){
        outer_iterations = i + 1;
        result = LBFGSOptimizer(context, lbfgs, iks_Ws, &Ws_data);
        if (result == kLBFGSConverged) break;
        if (result == kLBFGSLineSearchLimitReached) continue;  // 可重试警告
        if (result != kSuccess) break;
    }
    return finish(result);
}
void write_SummaryLogs(const LBFGSState& lbfgs, int result,
                       int outer_iterations, int max_iterations)
{
    const bool outer_limit = outer_iterations >= max_iterations &&
        (result == kSuccess || result == kLBFGSLineSearchLimitReached);
    const char* title = result == kLBFGSConverged ? "IKS | CONVERGED" :
        outer_limit ? "IKS | OUTER LIMIT" :
        result == kLBFGSLineSearchLimitReached ? "IKS | LINE SEARCH LIMIT" : "IKS | FAILED";
    const char* reason = outer_limit ? "Outer iteration limit reached" : lbfgs_status_message(result);
    const char* border = "+------------------------------------------------------------------------------+\n";
    auto row = [](const char* label, const char* value) {
        std::fprintf(stderr, "| %-14s %-61.61s |\n", label, value);
    };
    char value[160];
    std::fprintf(stderr, "\n%s| %-76s |\n", border, title);
    row("Reason", reason);
    std::snprintf(value, sizeof(value), "%d / %d", outer_iterations, max_iterations);
    row("Outer calls", value);
    std::snprintf(value, sizeof(value), "%d (%s)", result, lbfgs_status_message(result));
    row("Last result", value);
    // 初始化失败时优化器可能已重置，此时不把默认零值当成有效计算结果。
    if (lbfgs.evaluations == 0) {
        row("Accepted/evals", "n/a");
        row("f / ||g||_2", "n/a");
    } else {
        std::snprintf(value, sizeof(value), "%zu / %zu", lbfgs.iterations, lbfgs.evaluations);
        row("Accepted/evals", value);
        std::snprintf(value, sizeof(value), "%d / %d", lbfgs.line_search_trials, lbfgs.max_linesearch_step);
        row("Trials / limit", value);
        if (std::isfinite(lbfgs.f)) {
            std::snprintf(value, sizeof(value), "%.12e", lbfgs.f);
            row("f", value);
        } else row("f", "n/a");
        if (std::isfinite(lbfgs.g_norm)) {
            std::snprintf(value, sizeof(value), "%.6e / %.6e", lbfgs.g_norm, lbfgs.g_tol);
            row("||g||_2 / tol", value);
        } else row("||g||_2 / tol", "n/a");
        std::snprintf(value, sizeof(value), "%.6e", lbfgs.step_size);
        row("Step size", value);
    }
    std::fprintf(stderr, "%s", border);
}
int main(int argc, char** argv)
{
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [input.yaml]\n";
        return EXIT_FAILURE;
    }
    const std::string filename = argc == 2 ? argv[1] : "input.yaml";
    InputParameters parameters{};
    Context context{};
    StructGrid structgrid{};
    WS_data Ws_data{};
    LBFGSState lbfgs{};
    double* d_rho0 = nullptr;
    std::string error;
    auto finish = [&](int result) {
        if (context.stream) cudaStreamSynchronize(context.stream);
        destory_LBFGSOptimizer(lbfgs);
        reset_DavidsonSolver(Ws_data.davidsonPlan);
        cudaFree(Ws_data.d_rho);
        cudaFree(d_rho0);
        cudaFree(const_cast<double*>(Ws_data.finite.d_coeff));
        cudaFree(Ws_data.finite.d_halo_work);
        if (context.cusolver) cusolverDnDestroy(context.cusolver);
        if (context.cublas) cublasDestroy(context.cublas);
        if (context.stream) cudaStreamDestroy(context.stream);
        return result;
    };

    // 直接初始化 WS_data 内的计划，求解和释放使用同一份状态。
    int result = begin(parameters, context, Ws_data.finite, Ws_data.davidsonPlan,
                       structgrid, Ws_data, d_rho0, filename, error);
    if (result != kSuccess) {
        std::cerr << "Initialization failed (" << result << "): " << error << '\n';
        return finish(EXIT_FAILURE);
    }
    result = invks_scf(parameters, context, Ws_data, lbfgs);
    if (result != kLBFGSConverged) {
        return finish(EXIT_FAILURE);
    }

    const auto& out = parameters.output;
    if (out.write_density || out.write_density_difference || out.write_wave || out.write_eigenvalues) {
        // 回调曾原地平方 d_psi；在最终接受势上恢复带符号的本征轨道。
        result = DavidsonSolver(context, Ws_data.finite, Ws_data.davidsonPlan, lbfgs.d_x);
        if (result != kSuccess) {
            std::cerr << "Final Davidson solve failed: " << davidson_status_message(result) << '\n';
            return finish(EXIT_FAILURE);
        }
    }
    result = write_results(context, parameters, structgrid, lbfgs, Ws_data.davidsonPlan, error);
    if (result != kSuccess) {
        std::cerr << error << '\n';
        return finish(EXIT_FAILURE);
    }
    return finish(EXIT_SUCCESS);
}
