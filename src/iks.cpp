#include "cuda_math.hpp"
#include "Davidson.hpp"
#include "LBFGS.hpp"
#include "read_input.hpp"
#include "structure.hpp"
#include "write.hpp"
#include "smearing.hpp"

#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

struct WS_data
{
    int nx{};
    int ny{};
    int nz{};
    size_t npoints{};
    int nstates{};
    double dV{};
    double eta{};
    double rho0_nrom{};
    double* d_rho0{};
    double* d_rho{};
    double* d_diff_rho{};
    double* d_veff{};
    double* d_lapl_v{};
    double* d_eigenvalues{};
    Context context{};
    finitePlan finite{};
    DavidsonPlan davidsonPlan{};
    Occupation occ{};
};

using calculate_F = int (*)(
    const Context& context,
    const double* d_x,  // 输入：设备上的试探变量
    double& f,          // 输出：主机上的函数值
    double* d_g,        // 输出：设备上的梯度
    void* user_data     // 外部计算所需的数据
);
void destory_Ws_data(WS_data& Ws_data){
    if(Ws_data.d_rho)         cudaFree(Ws_data.d_rho);
    if(Ws_data.d_lapl_v)      cudaFree(Ws_data.d_lapl_v);
    if(Ws_data.d_diff_rho)    cudaFree(Ws_data.d_diff_rho);
    Ws_data = {};
}
void creat_WS_data(WS_data& Ws_data,Context context,InputParameters& parameters,DavidsonPlan& davidsonPlan,
        StructGrid& structgrid, finitePlan& finite, Occupation& occ,double* d_rho0)
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
    Ws_data.occ = occ;
    Ws_data.context = context;
    Ws_data.eta = parameters.inversion.regularization_eta;
    cudaError_t err;
    size_t bytes = size_t(Ws_data.npoints)*sizeof(double);
    err = cudaMalloc(reinterpret_cast<void**>(&Ws_data.d_lapl_v),bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] d_lapl_v cudaMalloc: %d (%s)\n",
                    static_cast<int>(err), cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&Ws_data.d_rho),bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] d_rho cudaMalloc: %d (%s)\n",
                    static_cast<int>(err), cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&Ws_data.d_diff_rho),bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] d_diff_rho cudaMalloc: %d (%s)\n",
                    static_cast<int>(err), cudaGetErrorString(err));
        return;
    }
    cublasStatus_t status;
    status = cublasDnrm2(
    context.cublas,
    Ws_data.npoints,       // 元素数量
    Ws_data.d_rho0,        // 设备端向量
    1,                     // 元素间距
    &Ws_data.rho0_nrom);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[cuBLAS] cublasDnrm2: %d (%s)\n",
                    static_cast<int>(status),
                    cublasGetStatusName(status));
    }
}
int begin(InputParameters& parameters,Context& context, Occupation& occ,
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
    int nstates = structgrid.electron_number / 2 + parameters.inversion.extra_states;
    ierr = read_density_input(parameters.system.density_file, structgrid, error);
    if (ierr != kSuccess) return ierr;
    ierr = CUDA_init(context);
    if (ierr != kSuccess) return ierr;
    ierr = upload(structgrid.target_density.data(),reinterpret_cast<void**>(&d_rho0),structgrid.npoints);
    if (ierr != kSuccess) return ierr;
    // init solver
    int nextra_retain = parameters.inversion.eigensolver.nextra_retain;
    int max_subspace_dim = parameters.inversion.eigensolver.max_subspace_dim;
    int max_iterations = parameters.inversion.eigensolver.max_iterations;
    double residual_tolerance = parameters.inversion.eigensolver.residual_tolerance;
    double direction_tolerance = parameters.inversion.eigensolver.direction_tolerance;
    ierr = DavidsonSolver_init(context, davidsonPlan,nstates,nx, ny, nz,
         nextra_retain,direction_tolerance,max_subspace_dim,max_iterations,residual_tolerance);
    if (ierr != kSuccess && ierr != kDavidsonMaxIterations) return ierr;
    // init finite
    int half_width = parameters.mesh.finite_difference_halfwidth;
    double gx = structgrid.gap[0];
    double gy = structgrid.gap[1];
    double gz = structgrid.gap[2];
    ierr = creat_finitePlan(finite,nx,ny,nz,half_width,gx,gy,gz);
    if (ierr != kSuccess) return ierr;
    // inti occupation
    std::string occ_mode = parameters.inversion.occupations.method;
    double T = parameters.inversion.occupations.temperature;
    int ne = structgrid.electron_number;
    ierr = Occupation_init(context,occ,occ_mode,nstates,ne,T);
    if (ierr != kSuccess) return ierr;
    //
    creat_WS_data(Ws_data, context, parameters, davidsonPlan, structgrid, finite, occ, d_rho0);
    return Ws_data.d_rho ? kSuccess : kAllocationError;
}
namespace {
constexpr char iks_separator[] =
    "================================================================================================\n";
}
void write_SummaryLogs(const LBFGSState& lbfgs, int result,
                       int outer_iterations, int max_iterations, double elapsed_seconds,
                       double relative_rho_L2norm, double relative_rho_L2tol);
// lbfgs 由调用方持有，输出完成后再销毁，保留最终接受的势。
int invks_scf(InputParameters& parameters, Context& context, WS_data& Ws_data, LBFGSState& lbfgs)
{
    const int max_interation = parameters.inversion.optimizer.max_iterations;
    int outer_iterations = 0;
    double relative_rho_L2norm = std::numeric_limits<double>::quiet_NaN();
    double relative_rho_L2tol = parameters.inversion.relative_rho_L2tol;
    const auto inversion_start = std::chrono::steady_clock::now();
    auto finish = [&](int result) {
        const double elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - inversion_start).count();
        write_SummaryLogs(lbfgs, result, outer_iterations, max_interation, elapsed_seconds,
                          relative_rho_L2norm, relative_rho_L2tol);
        return result;
    };
    std::fprintf(stderr, "\n%sIKS | INITIALIZATION\n", iks_separator);
    std::fprintf(stderr, "  Grid             %d x %d x %d (%zu points)\n",
                 Ws_data.nx, Ws_data.ny, Ws_data.nz, Ws_data.npoints);
    std::fprintf(stderr, "  Computed states  %d\n", Ws_data.nstates);
    std::fprintf(stderr, "  Electrons        %d\n", Ws_data.occ.ne);
    std::fprintf(stderr, "  Occupation mode  %s\n", Ws_data.occ.occ_mode.c_str());
    std::fprintf(stderr, "  Outer limit      %d\n  Gradient tol     %.6e\n\n",
                 max_interation, parameters.inversion.optimizer.gradient_tolerance);
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
        double eta = Ws_data.eta;
        double* d_eigenvalues = Ws_data.d_eigenvalues;
        double dV = Ws_data.dV;
        size_t npoints = Ws_data.npoints;
        int nstates = Ws_data.nstates;
        std::vector<double> h_eigenvalues(nstates);
        std::vector<double> h_occ(nstates);
        finitePlan& finite = Ws_data.finite;
        DavidsonPlan& davidsonPlan = Ws_data.davidsonPlan;
        Occupation& occ = Ws_data.occ;
        //
        int ierr = DavidsonSolver(context, finite, davidsonPlan, d_v);
        for (int retry = 0;
            ierr == kDavidsonMaxIterations && retry < 10;
            ++retry)
        {
            std::fprintf(stderr,
            "[IKS] Davidson iteration limit; continue %d / 10\n",
                        retry + 1);

            ierr = DavidsonSolver(context, finite, davidsonPlan, d_v);
        }
        if (ierr != kSuccess) return ierr;
        double* d_psi = davidsonPlan.d_psi;
        ierr = calculate_rho(Ws_data.context,occ,d_eigenvalues,npoints,dV,d_psi,d_rho);
        if (ierr != kSuccess) return ierr;
        f=0;
        ierr = download(d_eigenvalues,h_eigenvalues.data(),nstates);
        if (ierr != kSuccess) return ierr;
        ierr = download(occ.d_occupation,h_occ.data(),nstates);
        if (ierr != kSuccess) return ierr;
        for(int i=0;i<nstates;i++) f -= h_occ[i]*h_eigenvalues[i];
        //
        double h_vrho = 0.0;
        ierr = RealReal_IntegralFGdV(context,d_v,d_rho0,dV,npoints,h_vrho);
        if (ierr != kSuccess) return ierr;
        f += h_vrho;
        // punishment
        double pun = 0.0;
        ierr=finite_regularization(context,finite,d_v,dV,eta,pun);
        if (ierr != kSuccess) return ierr;
        f += pun;
        // g=rho0
        cublasStatus_t status;
        status = cublasDcopy(
            context.cublas, npoints,
            d_rho0, 1, d_g, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[cuBLAS] cublasDcopy: %d (%s)\n",
                        static_cast<int>(status),
                        cublasGetStatusName(status));
            return kCublasError;
        }
        // g=rho0-rho
        const double minus_one = -1.0;
        status = cublasDaxpy(
            context.cublas, npoints,
            &minus_one, d_rho, 1, d_g, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[cuBLAS] cublasDaxpy: %d (%s)\n",
                        static_cast<int>(status),
                        cublasGetStatusName(status));
            return kCublasError;
        }
        // g=[rho0-rho]*dv
        status = cublasDscal(
            context.cublas, npoints,
            &dV, d_g, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[cuBLAS] cublasDscal: %d (%s)\n",
                        static_cast<int>(status),
                        cublasGetStatusName(status));
            return kCublasError;
        }
        // g=[rho0-rho]*dv-2*eta*Lv*dv
        ierr = finite_regularization_contributeTograd(context,finite,d_v,eta,dV,Ws_data.d_lapl_v);
        if (ierr != kSuccess) return ierr;
        const double one = 1.0;
        status = cublasDaxpy(
            context.cublas, npoints,
            &one, Ws_data.d_lapl_v, 1, d_g, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[cuBLAS] cublasDaxpy: %d (%s)\n",
                        static_cast<int>(status),
                        cublasGetStatusName(status));
            return kCublasError;
        }
        //
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
    double g_tol = parameters.inversion.optimizer.gradient_tolerance;
    double c1 = parameters.inversion.optimizer.c1;
    double c2 = parameters.inversion.optimizer.c2;
    double max_linesearch_step = parameters.inversion.optimizer.max_linesearch_step;
    int result = LBFGSOptimizer_init(
        context, lbfgs, n, c1, c2, max_linesearch_step, history_capacity,
        iks_Ws, &Ws_data, d_x_guess, g_tol
    );
    cudaFree(d_x_guess);
    d_x_guess = nullptr;
    if (result != 0)
        return finish(result);
    //
    cublasStatus_t status;
    const double minus_one = -1.0;
    for(int i=0;i<max_interation;i++){
        outer_iterations = i + 1;
        const auto step_start = std::chrono::steady_clock::now();
        const size_t accepted_before = lbfgs.iterations;
        const size_t evaluations_before = lbfgs.evaluations;
        std::fprintf(stderr, "\n%sIKS | OUTER ITERATION %d / %d\n\n",
                     iks_separator, outer_iterations, max_interation);
        
        relative_rho_L2norm = std::numeric_limits<double>::quiet_NaN();
        result = LBFGSOptimizer(context, lbfgs, iks_Ws, &Ws_data);\
        const double step_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - step_start).count();
        std::fprintf(stderr,
                     "IKS step %d: result %d (%s)\n"
                     "  This step        %zu accepted / %zu evaluations / %.3f s\n",
                     outer_iterations, result, lbfgs_status_message(result),
                     lbfgs.iterations - accepted_before,
                     lbfgs.evaluations - evaluations_before, step_seconds);
        if (result == kLBFGSLineSearchLimitReached)
            std::fprintf(stderr, "  Warning          Line search limit; accepted point retained.\n");
        
        if (result == kLBFGSConverged){
            // diff = rho
            status = cublasDcopy(
                context.cublas, Ws_data.npoints,
                Ws_data.d_rho, 1, Ws_data.d_diff_rho, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "[cuBLAS] cublasDcopy: %d (%s)\n",
                            static_cast<int>(status),
                            cublasGetStatusName(status));
                return finish(kCublasError);
            }
            // diff = rho-rho0
            status = cublasDaxpy(
                        context.cublas, Ws_data.npoints,
                        &minus_one, Ws_data.d_rho0, 1, Ws_data.d_diff_rho, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "[cuBLAS] cublasDaxpy: %d (%s)\n",
                            static_cast<int>(status),
                            cublasGetStatusName(status));
                return finish(kCublasError);
            }
            // relative_rho_L2norm = diff_norm
            double diff_norm = 0.0;
            status = cublasDnrm2(
            context.cublas,
            Ws_data.npoints,       // 元素数量
            Ws_data.d_diff_rho,    // 设备端向量
            1,                     // 元素间距
            &diff_norm);
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "[cuBLAS] cublasDnrm2: %d (%s)\n",
                            static_cast<int>(status),
                            cublasGetStatusName(status));
                return finish(kCublasError);
            }
            // relative_rho_L2norm = diff_norm/rho_norm
            relative_rho_L2norm = diff_norm/Ws_data.rho0_nrom;
            std::fprintf(stderr, "  relative density diff L2 / tol %.6e / %.6e (%s)\n",
                         relative_rho_L2norm, relative_rho_L2tol,
                         relative_rho_L2norm < relative_rho_L2tol ?
                         "converged" : "not converged");
            if(relative_rho_L2norm < relative_rho_L2tol) break;

            lbfgs.g_norm=1+parameters.inversion.optimizer.gradient_tolerance;
        }
        if (result == kLBFGSLineSearchLimitReached) continue;  // 可重试警告
        if (result != kSuccess && result != kLBFGSConverged) break;
    }
    if ((result == kSuccess || result == kLBFGSConverged ||
         result == kLBFGSLineSearchLimitReached) &&
        !(relative_rho_L2norm < relative_rho_L2tol)) {
        std::fprintf(stderr, "Density tolerance not reached.\n");
        result = -401;
    }
    return finish(result);
}
void write_SummaryLogs(const LBFGSState& lbfgs, int result,
                       int outer_iterations, int max_iterations, double elapsed_seconds,
                       double relative_rho_L2norm, double relative_rho_L2tol)
{
    const bool outer_limit = outer_iterations >= max_iterations &&
        (result == kSuccess || result == kLBFGSLineSearchLimitReached);
    const char* title = result == kLBFGSConverged ? "IKS | CONVERGED" :
        outer_limit ? "IKS | OUTER LIMIT" :
        result == kLBFGSLineSearchLimitReached ? "IKS | LINE SEARCH LIMIT" : "IKS | FAILED";
    const char* result_message = result == -401 ?
        "Density error tolerance not reached" : lbfgs_status_message(result);
    const char* reason = outer_limit ? "Outer iteration limit reached" : result_message;
    auto row = [](const char* label, const char* value) {
        std::fprintf(stderr, "  %-16s %s\n", label, value);
    };
    char value[160];
    std::fprintf(stderr, "\n%s%s\n", iks_separator, title);
    row("Reason", reason);
    std::snprintf(value, sizeof(value), "%d / %d", outer_iterations, max_iterations);
    row("Outer calls", value);
    std::snprintf(value, sizeof(value), "%d (%s)", result, result_message);
    row("Last result", value);
    if (std::isfinite(relative_rho_L2norm)) {
        std::snprintf(value, sizeof(value), "%.6e / %.6e (%s)",
                      relative_rho_L2norm, relative_rho_L2tol,
                      relative_rho_L2norm < relative_rho_L2tol ? "converged" : "not converged");
    } else {
        std::snprintf(value, sizeof(value), "n/a / %.6e (not available)", relative_rho_L2tol);
    }
    row("Density L2 / tol", value);
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
    std::snprintf(value, sizeof(value), "%.3f s (initial evaluation and outer steps)", elapsed_seconds);
    row("Inversion time", value);
    std::fprintf(stderr, "%s", iks_separator);
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
    Occupation occ{};
    double* d_rho0 = nullptr;
    std::string error;
    auto finish = [&](int result) {
        if (context.stream) cudaStreamSynchronize(context.stream);

        destory_LBFGSOptimizer(lbfgs);
        reset_DavidsonSolver(Ws_data.davidsonPlan);
        destroy_Occupation(occ);

        cudaFree(d_rho0);
        const int finite_result = destory_finitePlan(Ws_data.finite);
        if (finite_result != kSuccess) {
            std::cerr << "destory_finitePlan failed (" << finite_result
                      << "): " << status_message(finite_result) << '\n';
            result = EXIT_FAILURE;
        }

        destory_Ws_data(Ws_data);

        if (context.cusolver) cusolverDnDestroy(context.cusolver);
        if (context.cublas) cublasDestroy(context.cublas);
        if (context.stream) cudaStreamDestroy(context.stream);
        return result;
    };


    int result = begin(parameters, context, occ, Ws_data.finite, Ws_data.davidsonPlan,
                       structgrid, Ws_data, d_rho0, filename, error);
    if (result != kSuccess) {
        std::cerr << "Initialization failed (" << result << "): " << error << '\n';
        return finish(EXIT_FAILURE);
    }
    int scf_result = invks_scf(parameters, context, Ws_data, lbfgs);
    result = write_results(context, parameters, structgrid, lbfgs, Ws_data.davidsonPlan, Ws_data.occ, error);
    if (result != kSuccess) {
        std::cerr << error << '\n';
        return finish(EXIT_FAILURE);
    }
    if (scf_result != kLBFGSConverged) return finish(EXIT_FAILURE);

    return finish(EXIT_SUCCESS);
}
