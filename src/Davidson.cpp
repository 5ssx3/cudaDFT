#include "Davidson.hpp"
#include "cuda_math.hpp"

#include <memory>  // std::unique_ptr
#include <new>     // std::nothrow
#include <cmath>   // std::isfinite
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdio>
#include <algorithm>
#include <climits>
#include <limits>

const char* davidson_status_message(int status) noexcept
{
    switch (status) {
        case kDavidsonCublasError: return "cuBLAS call failed";
        case kDavidsonCusolverError: return "cuSOLVER call failed";
        case kDavidsonCurandError: return "cuRAND call failed";
        case kDavidsonSolverInvalidParameter: return "cuSOLVER devInfo reports an invalid parameter";
        case kDavidsonSolverNoConvergence: return "Projected eigensolver did not converge";
        case kDavidsonNonFinite: return "Residual or correction norm is not finite";
        case kDavidsonStagnation: return "Unconverged states remain but no direction was added";
        case kDavidsonMaxIterations: return "Davidson iteration limit reached";
        case kDavidsonNotInitialized: return "Davidson plan is not initialized";
        default: return status_message(status);
    }
}

namespace {
int report_davidson_status(int status, const char* stage, int iteration = 0) noexcept
{
    std::fprintf(stderr, "[Davidson] stage=%s, status=%d (%s)",
                 stage, status, davidson_status_message(status));
    if (iteration > 0) std::fprintf(stderr, ", iteration=%d", iteration);
    std::fprintf(stderr, "\n");
    return status;
}
}
//////////////////////////////////////////////////////////////////////////////
inline bool check_cusolver_status(cusolverStatus_t status,
    const char* expression,const char* file,int line)
{
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    std::fprintf(stderr,
        "[cuSOLVER] status=%d, call=%s, at %s:%d\n",
        static_cast<int>(status),expression,file,line);
    return false;
}
inline bool check_cublas_status(cublasStatus_t status,
    const char* expression,const char* file,int line)
{
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr,
        "[cuBLAS] status=%d (%s), call=%s, at %s:%d\n",
        static_cast<int>(status),
        cublasGetStatusName(status),
        expression,file,line);
    return false;
}
inline bool check_cuda_status(cudaError_t status,
    const char* expression, const char* file, int line)
{
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[CUDA] status=%d (%s: %s), call=%s, at %s:%d\n",
        static_cast<int>(status), cudaGetErrorName(status), cudaGetErrorString(status),
        expression, file, line);
    return false;
}
inline bool check_curand_status(curandStatus_t status,
    const char* expression, const char* file, int line)
{
    if (status == CURAND_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "[cuRAND] status=%d, call=%s, at %s:%d\n",
        static_cast<int>(status), expression, file, line);
    return false;
}
#define CHECK_CUBLAS(expression) \
    check_cublas_status((expression), #expression, __FILE__, __LINE__)
#define CHECK_CUSOLVER(expression) \
    check_cusolver_status((expression), #expression, __FILE__, __LINE__)
#define CHECK_CUDA(expression) \
    check_cuda_status((expression), #expression, __FILE__, __LINE__)
#define CHECK_CURAND(expression) \
    check_curand_status((expression), #expression, __FILE__, __LINE__)
//////////////////////////////////////////////////////////////////////////////
namespace {
template<class T>
int release_device(T*& ptr) noexcept
{
    if (ptr == nullptr) return kSuccess;
    if (!CHECK_CUDA(cudaFree(ptr))) return kCudaFreeError;
    ptr = nullptr;
    return kSuccess;
}

int validate_context(const Context& context) noexcept
{
    if (context.cublas == nullptr || context.cusolver == nullptr)
        return kInvalidState;
    cudaStream_t blas_stream = nullptr, solver_stream = nullptr;
    cublasPointerMode_t mode{};
    if (!CHECK_CUBLAS(cublasGetStream(context.cublas, &blas_stream)) ||
        !CHECK_CUBLAS(cublasGetPointerMode(context.cublas, &mode)))
        return kDavidsonCublasError;
    if (!CHECK_CUSOLVER(cusolverDnGetStream(context.cusolver, &solver_stream)))
        return kDavidsonCusolverError;
    if (blas_stream != context.stream || solver_stream != context.stream ||
        mode != CUBLAS_POINTER_MODE_HOST)
        return kInvalidState;
    return kSuccess;
}

int check_solver_info(const Context& context, const RayleighRitzSubspace& rr,
                      const char* operation) noexcept
{
    int info = 0;
    if (!CHECK_CUDA(cudaMemcpyAsync(&info, rr.d_solver_info, sizeof(info),
                                   cudaMemcpyDeviceToHost, context.stream)))
        return kCudaMemcpyError;
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return kCudaSyncError;
    if (info == 0) return kSuccess;
    std::fprintf(stderr, "[cuSOLVER] call=%s, devInfo=%d\n", operation, info);
    return info < 0 ? kDavidsonSolverInvalidParameter : kDavidsonSolverNoConvergence;
}
}
///////////////////////////////////////////////////////////////////////////////
int destory_RayleighRitzSubspace(RayleighRitzSubspace& RRsubspace){
    int status = kSuccess;
    auto release = [&status](auto*& ptr) {
        const int ierr = release_device(ptr);
        if (status == kSuccess) status = ierr;
    };
    release(RRsubspace.d_V);
    release(RRsubspace.d_HV);
    release(RRsubspace.d_Hhat);
    release(RRsubspace.d_correction);
    release(RRsubspace.d_solver_work);
    release(RRsubspace.d_solver_info);
    release(RRsubspace.d_ritzValues);
    release(RRsubspace.d_tau);

    RRsubspace.status = kDavidsonNotInitialized;
    RRsubspace.current_subspace_dim = 0;
    if (status == kSuccess) RRsubspace = RayleighRitzSubspace{};
    return status;
}
int creat_RayleighRitzSubspace(Context& context, RayleighRitzSubspace& RRsubspace,
                               int npoints,int nstates,int max_subspace_dim)
{
    if (npoints <= 0 || nstates <= 0 ||
        nstates >= max_subspace_dim || max_subspace_dim > npoints)
        return kInvalidArgument;
    if (RRsubspace.status == kSuccess ||
        RRsubspace.d_V || RRsubspace.d_HV || RRsubspace.d_Hhat ||
        RRsubspace.d_correction || RRsubspace.d_solver_work || RRsubspace.d_tau ||
        RRsubspace.d_solver_info || RRsubspace.d_ritzValues)
        return kInvalidState;

    size_t V_bytes = 0, Hhat_bytes = 0, correction_bytes = 0;
    if (!checked_product(size_t(npoints) * max_subspace_dim, sizeof(double), V_bytes) ||
        !checked_product(size_t(max_subspace_dim) * max_subspace_dim, sizeof(double), Hhat_bytes) ||
        !checked_product(size_t(npoints) * nstates, sizeof(double), correction_bytes))
        return kSizeOverflow;
    const size_t ritzValues_bytes = size_t(max_subspace_dim) * sizeof(double);
    auto fail = [&](int error) {
        const int cleanup = destory_RayleighRitzSubspace(RRsubspace);
        if (cleanup != kSuccess) report_davidson_status(cleanup, "subspace cleanup");
        return error;
    };
    if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_V),V_bytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_HV),V_bytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_Hhat),Hhat_bytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_correction),correction_bytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_ritzValues),ritzValues_bytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_tau),ritzValues_bytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_solver_info),sizeof(int))))
        return fail(kAllocationError);

    int syevd_lwork = 0, geqrf_lwork = 0, orgqr_lwork = 0;
    if (!CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
            context.cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
            max_subspace_dim, RRsubspace.d_Hhat, max_subspace_dim,
            RRsubspace.d_ritzValues, &syevd_lwork)) ||
        !CHECK_CUSOLVER(cusolverDnDgeqrf_bufferSize(
            context.cusolver, npoints, nstates, RRsubspace.d_V, npoints,
            &geqrf_lwork)) ||
        !CHECK_CUSOLVER(cusolverDnDorgqr_bufferSize(
            context.cusolver, npoints, nstates, nstates, RRsubspace.d_V, npoints,
            RRsubspace.d_tau, &orgqr_lwork)))
        return fail(kDavidsonCusolverError);
    RRsubspace.solver_lwork = std::max(syevd_lwork,std::max(geqrf_lwork, orgqr_lwork));
    if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&RRsubspace.d_solver_work),
                               size_t(RRsubspace.solver_lwork) * sizeof(double))))
        return fail(kAllocationError);

    const size_t count = size_t(npoints) * nstates;
    const size_t generate_count = count + count % 2;
    curandGenerator_t generator = nullptr;
    int ierr = kSuccess;
    if (!CHECK_CURAND(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT)) ||
        !CHECK_CURAND(curandSetStream(generator, context.stream)) ||
        !CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL)) ||
        !CHECK_CURAND(curandGenerateNormalDouble(generator, RRsubspace.d_V, generate_count, 0.0, 1.0)))
        ierr = kDavidsonCurandError;
    if (generator && !CHECK_CURAND(curandDestroyGenerator(generator)))
        ierr = kDavidsonCurandError;
    if (ierr != kSuccess) return fail(ierr);

    if (!CHECK_CUSOLVER(cusolverDnDgeqrf(
            context.cusolver, npoints, nstates, RRsubspace.d_V, npoints,
            RRsubspace.d_tau, RRsubspace.d_solver_work,
            RRsubspace.solver_lwork, RRsubspace.d_solver_info)))
        return fail(kDavidsonCusolverError);
    ierr = check_solver_info(context, RRsubspace, "cusolverDnDgeqrf");
    if (ierr != kSuccess) return fail(ierr);
    if (!CHECK_CUSOLVER(cusolverDnDorgqr(
            context.cusolver, npoints, nstates, nstates, RRsubspace.d_V, npoints,
            RRsubspace.d_tau, RRsubspace.d_solver_work,
            RRsubspace.solver_lwork, RRsubspace.d_solver_info)))
        return fail(kDavidsonCusolverError);
    ierr = check_solver_info(context, RRsubspace, "cusolverDnDorgqr");
    if (ierr != kSuccess) return fail(ierr);

    RRsubspace.current_subspace_dim = nstates;
    RRsubspace.status = kSuccess;
    return kSuccess;
}
int reset_DavidsonSolver(DavidsonPlan& davidsonPlan){
    int status = kSuccess;
    auto release = [&status](auto*& ptr) {
        const int ierr = release_device(ptr);
        if (status == kSuccess) status = ierr;
    };
    release(davidsonPlan.d_psi);
    release(davidsonPlan.d_eigenvalues);
    release(davidsonPlan.d_residual);
    const int ierr = destory_RayleighRitzSubspace(davidsonPlan.RRsubspace);
    if (status == kSuccess) status = ierr;
    davidsonPlan.status = kDavidsonNotInitialized;
    if (status != kSuccess)
        return report_davidson_status(status, "reset_DavidsonSolver");
    davidsonPlan = DavidsonPlan{};
    return kSuccess;
}
int DavidsonSolver_init(Context& context, DavidsonPlan& davidsonPlan,int num_states,
            int nx, int ny, int nz, int nextra_retain,
            int max_subspace_dim,int max_iterations,double residual_tol)noexcept
{
    if (nx <= 0 || ny <= 0 || nz <= 0 || num_states <= 0 ||
        max_subspace_dim <= num_states || nextra_retain < 0 ||
        nextra_retain >= max_subspace_dim - num_states || max_iterations <= 0 ||
        !std::isfinite(residual_tol) || residual_tol <= 0.0)
        return report_davidson_status(kInvalidArgument, "DavidsonSolver_init");
    if (davidsonPlan.status == kSuccess || davidsonPlan.d_psi ||
        davidsonPlan.d_eigenvalues || davidsonPlan.d_residual)
        return report_davidson_status(kInvalidState, "DavidsonSolver_init");
    int ierr = validate_context(context);
    if (ierr != kSuccess) return report_davidson_status(ierr, "DavidsonSolver_init");

    size_t npoints = 0, count = 0;
    if (!checked_product(size_t(nx), size_t(ny), npoints) ||
        !checked_product(npoints, size_t(nz), npoints) || npoints > INT_MAX ||
        !checked_product(npoints, size_t(num_states), count) ||
        count >= std::numeric_limits<size_t>::max() / sizeof(double))
        return report_davidson_status(kSizeOverflow, "DavidsonSolver_init");
    if (size_t(max_subspace_dim) > npoints)
        return report_davidson_status(kInvalidArgument, "DavidsonSolver_init");
    const size_t allocated_count = count + count % 2;
    auto fail = [&](int error) {
        const int cleanup = reset_DavidsonSolver(davidsonPlan);
        if (cleanup != kSuccess) report_davidson_status(cleanup, "initialization cleanup");
        return report_davidson_status(error, "DavidsonSolver_init");
    };
    if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&davidsonPlan.d_psi),
                               allocated_count * sizeof(double))) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&davidsonPlan.d_eigenvalues),
                               size_t(num_states) * sizeof(double))) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&davidsonPlan.d_residual),
                               count * sizeof(double))))
        return fail(kAllocationError);
    ierr = creat_RayleighRitzSubspace(context,davidsonPlan.RRsubspace,
                                     int(npoints),num_states,max_subspace_dim);
    if (ierr != kSuccess) return fail(ierr);
    davidsonPlan.nx = nx;
    davidsonPlan.ny = ny;
    davidsonPlan.nz = nz;
    davidsonPlan.npoints = npoints;
    davidsonPlan.nstates = num_states;
    davidsonPlan.max_subspace_dim = max_subspace_dim;
    davidsonPlan.max_iterations = max_iterations;
    davidsonPlan.residual_tol = residual_tol;
    davidsonPlan.nextra_retain = nextra_retain;
    davidsonPlan.status = kSuccess;
    return kSuccess;
}
int DavidsonPreconditioner(
    const Context& context,
    const finitePlan& finite,
    DavidsonPlan& davidsonPlan,
    const double* d_veff,
    int& nactive,
    double* h_residual_norms) noexcept
{
    nactive = 0;
    RayleighRitzSubspace& RRsubspace = davidsonPlan.RRsubspace;
    int current_subspace_dim = RRsubspace.current_subspace_dim;
    int npoints = davidsonPlan.npoints;
    int nstates = davidsonPlan.nstates;
    double* d_V = RRsubspace.d_V;
    double* d_HV = RRsubspace.d_HV;
    double* d_C = RRsubspace.d_Hhat;
    double* d_psi = davidsonPlan.d_psi;
    double* d_residual = davidsonPlan.d_residual;
    
    const double one = 1.0;
    const double zero = 0.0;
    const double negative_one = -1.0;

    // 1. U = V * C(:, 0:nstates)
    // V: N×m，C: m×p，U: N×p。
    if (!CHECK_CUBLAS(cublasDgemm(
        context.cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        npoints, nstates, current_subspace_dim,
        &one,
        d_V, npoints,
        d_C, current_subspace_dim,
        &zero,
        d_psi, npoints)))
        return kDavidsonCublasError;
    
    // 2. 暂存 R = U * Θ。
    // RIGHT 表示每一列乘对应本征值。
    // d_ritzValues 是设备端数组，取前 nstates 项。
    if (!CHECK_CUBLAS(cublasDdgmm(
        context.cublas,
        CUBLAS_SIDE_RIGHT,
        npoints, nstates,
        d_psi, npoints,
        RRsubspace.d_ritzValues, 1,
        d_residual, npoints)))
        return kDavidsonCublasError;

    // 3. R = (HV) * C - R = (HV) * C - U * Θ。
    if (!CHECK_CUBLAS(cublasDgemm(
        context.cublas,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        npoints, nstates, current_subspace_dim,
        &one,
        d_HV, npoints,
        d_C, current_subspace_dim,
        &negative_one,
        d_residual, npoints)))
        return kDavidsonCublasError;

    std::unique_ptr<int[]> active(new (std::nothrow) int[nstates]);
    if (!active) return kAllocationError;
    int selected = 0;

    // 前提：U 已欧氏归一化，cuBLAS 使用 HOST pointer mode。
    for (int a = 0; a < nstates; ++a) {
        double norm = 0.0;

        if (!CHECK_CUBLAS(cublasDnrm2(
            context.cublas,
            npoints,
            d_residual + size_t(a) * npoints,
            1,
            &norm)))
            return kDavidsonCublasError;

        if (!std::isfinite(norm))
            return kDavidsonNonFinite;

        h_residual_norms[a] = norm;
        if (norm > davidsonPlan.residual_tol){
            active[selected] = a;
            selected = selected + 1;
        }
    }

    // 所有目标态已收敛，无需分配修正块。
    if (selected == 0) return kSuccess;
    // 已在初始化时分配，容量至少为 npoints * nstates。
    double* d_correction = RRsubspace.d_correction;

    for (int b = 0; b < selected; ++b) {
        const int a = active[b];

        int ierr = apply_davidson_preconditioner(
            context,
            finite,
            d_veff,
            RRsubspace.d_ritzValues + a,
            d_residual + size_t(a) * npoints,
            d_correction + size_t(b) * npoints);

        if (ierr != kSuccess)
            return ierr;
    }

    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream)))
        return kCudaSyncError;

    nactive = selected;
    return kSuccess;
}
int RayleighRitzDiagonalize(
    const Context& context,
    const finitePlan& finite,
    const DavidsonPlan& davidsonPlan,
    const double* d_veff,
    const int nadded,
    bool& hv_initialized){
        
    const RayleighRitzSubspace& RRsubspace = davidsonPlan.RRsubspace;
    int current_subspace_dim = RRsubspace.current_subspace_dim;
    int npoints = davidsonPlan.npoints;
    int nstates = davidsonPlan.nstates;
    double* d_V = RRsubspace.d_V;
    double* d_HV = RRsubspace.d_HV;
    double* d_Hhat = RRsubspace.d_Hhat;
    size_t stride = npoints;
    if(!hv_initialized){
        for(int i=0;i<current_subspace_dim;i++){
            int ierr = apply_HamiltonOperator(context,finite,&d_V[i*stride],
                                            d_veff,&d_HV[i*stride]);
            if(ierr != kSuccess) return ierr;
        }
    }else{
        const int first_column = current_subspace_dim - nadded;
        for (int i = first_column; i < current_subspace_dim; ++i) {
            int ierr = apply_HamiltonOperator(
                context, finite,
                d_V + size_t(i) * stride,
                d_veff,
                d_HV + size_t(i) * stride);
            if(ierr != kSuccess) return ierr;
        }
    }
    // cublasDgemm 矩阵乘法
    const double one = 1.0;
    const double zero = 0.0;
    if (!CHECK_CUBLAS(cublasDgemm(
        context.cublas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        current_subspace_dim,     // 输出矩阵行数
        current_subspace_dim,     // 输出矩阵列数
        npoints,                  // 求和维度
        &one,
        d_V,
        npoints,                  // V 的 leading dimension
        d_HV,
        npoints,                  // HV 的 leading dimension
        &zero,
        d_Hhat,
        current_subspace_dim)))    // Hhat 的 leading dimension
        return kDavidsonCublasError;

    // cusolverDnDsygvd 对角化求解器
    // 标准实对称本征问题：Hhat * C = C * Θ。
    // 成功后 d_Hhat 被覆盖为本征向量矩阵 C。
    if (!CHECK_CUSOLVER(cusolverDnDsyevd(
        context.cusolver,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        current_subspace_dim,
        d_Hhat,
        current_subspace_dim,
        RRsubspace.d_ritzValues,
        RRsubspace.d_solver_work,
        RRsubspace.solver_lwork,
        RRsubspace.d_solver_info)))
        return kDavidsonCusolverError;
    int ierr = check_solver_info(context, RRsubspace, "cusolverDnDsyevd");
    if (ierr != kSuccess) return ierr;
    //
    if (!CHECK_CUDA(cudaMemcpyAsync(
        davidsonPlan.d_eigenvalues,
        RRsubspace.d_ritzValues,
        size_t(nstates) * sizeof(double),
        cudaMemcpyDeviceToDevice,
        context.stream))) return kCudaMemcpyError;
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return kCudaSyncError;
    hv_initialized = true;
    return kSuccess;
}
int Gram_Schmidt_project(const Context& context, double* d_W, double* d_t, int npoints, int current_W_dim){
    if (d_W == nullptr || d_t == nullptr || npoints <= 0 ||
        current_W_dim < 0 || current_W_dim > npoints)
        return kInvalidArgument;
    if (current_W_dim == 0) return kSuccess;
    double* d_c = nullptr;
    if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_c),
                               size_t(current_W_dim) * sizeof(double))))
        return kAllocationError;

    const double one = 1.0, zero = 0.0, negative_one = -1.0;
    int result = kSuccess;
    for (int i=0;i<2;i++) {
        // c = W^T t; t = t - Wc.
        if (!CHECK_CUBLAS(cublasDgemm(
                context.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                current_W_dim, 1, npoints, &one,
                d_W, npoints, d_t, npoints, &zero, d_c, current_W_dim)) ||
            !CHECK_CUBLAS(cublasDgemm(
                context.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                npoints, 1, current_W_dim, &negative_one,
                d_W, npoints, d_c, current_W_dim, &one, d_t, npoints))) {
            result = kDavidsonCublasError;
            break;
        }
    }
    const int cleanup = release_device(d_c);
    return result == kSuccess ? cleanup : result;
}
int ExtendSubspace(
    const Context& context, DavidsonPlan& davidsonPlan,
    int nactive, int& nadded) noexcept
{
    nadded = 0;
    if (nactive < 0 || nactive > davidsonPlan.nstates) return kInvalidArgument;
    RayleighRitzSubspace& RRsubspace = davidsonPlan.RRsubspace;
    double* d_V = RRsubspace.d_V;
    double* d_correction = RRsubspace.d_correction;
    int current_W_dim = RRsubspace.current_subspace_dim;
    int old_dim = RRsubspace.current_subspace_dim;
    int npoints = davidsonPlan.npoints;
    int capacity = davidsonPlan.max_subspace_dim;
    //
    int ierr = kSuccess;
    //
    double* d_W = d_V;
    for (int i=0;i<nactive && current_W_dim < capacity ;i++){
        double norm_old = 0;
        double norm_new = 0;
        double* d_t = d_correction+size_t(npoints)*i;
        // nrom_old
        if (!CHECK_CUBLAS(cublasDnrm2(
            context.cublas,
            npoints,       // 元素数量
            d_t,           // 设备端向量
            1,             // 元素间距
            &norm_old)))   // 主机端结果
            return kDavidsonCublasError;
        if (!std::isfinite(norm_old)) return kDavidsonNonFinite;
        if (norm_old==0) continue;

        ierr = Gram_Schmidt_project(context,d_W,d_t,npoints,current_W_dim);

        if (ierr != kSuccess) return ierr;
        // nrom_new
        if (!CHECK_CUBLAS(cublasDnrm2(
            context.cublas,
            npoints,       // 元素数量
            d_t,           // 设备端向量
            1,             // 元素间距
            &norm_new)))   // 主机端结果
            return kDavidsonCublasError;
        if (!std::isfinite(norm_new)) return kDavidsonNonFinite;
        // extend d_W
        if (norm_new > davidsonPlan.tNorm_tol*norm_old){
            const double scale = 1.0 / norm_new;
            if (!CHECK_CUBLAS(cublasDscal(
                context.cublas,
                npoints,
                &scale,
                d_t, 1)))
                return kDavidsonCublasError;
            double* d_dst = d_W + size_t(npoints) * current_W_dim;
            if (!CHECK_CUDA(cudaMemcpyAsync(
                d_dst,
                d_t,
                size_t(npoints) * sizeof(double),
                cudaMemcpyDeviceToDevice,
                context.stream))) return kCudaMemcpyError;
            current_W_dim++;
        }
    }
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream)))
        return kCudaSyncError;
    //
    RRsubspace.current_subspace_dim = current_W_dim;
    nadded = current_W_dim - old_dim;
    return kSuccess;
}
int UpdateSubspace(const Context& context, DavidsonPlan& davidsonPlan) noexcept{
    RayleighRitzSubspace& RRsubspace = davidsonPlan.RRsubspace;
    const int m = RRsubspace.current_subspace_dim;
    const int capacity = davidsonPlan.max_subspace_dim;
    if (m < capacity) return kSuccess;
    const int k = davidsonPlan.nstates + davidsonPlan.nextra_retain;
    if (m != capacity || k < davidsonPlan.nstates || k >= capacity)
        return kInvalidState;
    const int npoints = int(davidsonPlan.npoints);
    const size_t count = size_t(npoints) * k;
    const size_t bytes = count * sizeof(double);
    if (bytes > std::numeric_limits<size_t>::max() / 2) return kSizeOverflow;
    double* d_temp = nullptr;
    if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_temp), 2 * bytes)))
        return kAllocationError;
    double* d_V_new = d_temp;
    double* d_HV_new = d_temp + count;
    const double one = 1.0, zero = 0.0;
    int result = kSuccess;
    // V_new = VC; HV_new = (HV)C.
    if (!CHECK_CUBLAS(cublasDgemm(
            context.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            npoints, k, m, &one, RRsubspace.d_V, npoints,
            RRsubspace.d_Hhat, m, &zero, d_V_new, npoints)) ||
        !CHECK_CUBLAS(cublasDgemm(
            context.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            npoints, k, m, &one, RRsubspace.d_HV, npoints,
            RRsubspace.d_Hhat, m, &zero, d_HV_new, npoints)))
        result = kDavidsonCublasError;
    else if (!CHECK_CUDA(cudaMemcpyAsync(RRsubspace.d_V, d_V_new, bytes,
                                        cudaMemcpyDeviceToDevice, context.stream)) ||
             !CHECK_CUDA(cudaMemcpyAsync(RRsubspace.d_HV, d_HV_new, bytes,
                                        cudaMemcpyDeviceToDevice, context.stream)))
        result = kCudaMemcpyError;
    else if (!CHECK_CUDA(cudaStreamSynchronize(context.stream)))
        result = kCudaSyncError;
    else
        RRsubspace.current_subspace_dim = k;
    const int cleanup = release_device(d_temp);
    return result == kSuccess ? cleanup : result;
}
int write_log(
    const Context& context,
    const DavidsonPlan& davidsonPlan,
    int iteration,
    int nactive,
    const double* h_residual_norms) noexcept
{
    std::unique_ptr<double[]> h_eigenvalues(
        new (std::nothrow) double[davidsonPlan.nstates]);
    if (!h_eigenvalues) return kAllocationError;
    // 下载成功并完成同步后再输出，避免失败时留下不完整的日志框。
    if (!CHECK_CUDA(cudaMemcpyAsync(
            h_eigenvalues.get(), davidsonPlan.d_eigenvalues,
            size_t(davidsonPlan.nstates) * sizeof(double),
            cudaMemcpyDeviceToHost, context.stream)))
        return kCudaMemcpyError;
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream)))
        return kCudaSyncError;
    // 框内仅使用 ASCII，避免查看器中中文字体宽度不同导致右边框错位。
    auto summary_row = [](const char* label, const char* value) {
        std::fprintf(stderr, "│  %-14s %-61s  │\n", label, value);
    };
    char value[128];
    std::snprintf(value, sizeof(value), "Davidson | Iter %d / %d",
                  iteration, davidsonPlan.max_iterations);
    std::fprintf(stderr, "\n╭────────────────────────────────────────────────────────────────────────────────╮\n│  %-76s  │\n", value);
    // if (iteration == 1) {
    //     std::snprintf(value, sizeof(value), "lowest %d eigenstates", davidsonPlan.nstates);
    //     summary_row("Target states", value);
    //     std::snprintf(value, sizeof(value), "||Hu - theta*u||_2 <= %.6e", davidsonPlan.residual_tol);
    //     summary_row("Criterion", value);
    //     std::snprintf(value, sizeof(value), "%.6e (relative norm)", davidsonPlan.tNorm_tol);
    //     summary_row("Direction tol", value);
    // }
    std::snprintf(value, sizeof(value), "%d / %d",
                  davidsonPlan.nstates-nactive, davidsonPlan.nstates);
    summary_row("Converged", value);
    std::snprintf(value, sizeof(value), "%d", nactive);
    summary_row("Pending", value);
    std::snprintf(value, sizeof(value), "%d / %d",
                  davidsonPlan.RRsubspace.current_subspace_dim, davidsonPlan.max_subspace_dim);
    summary_row("Subspace", value);
    const double max_residual = *std::max_element(
        h_residual_norms, h_residual_norms + davidsonPlan.nstates);
    std::snprintf(value, sizeof(value), "%.6e", max_residual);
    summary_row("Max residual", value);
    std::snprintf(value, sizeof(value), "%.6e", davidsonPlan.residual_tol);
    summary_row("Residual tol", value);
    std::snprintf(value, sizeof(value), "%.6e", max_residual / davidsonPlan.residual_tol);
    summary_row("Residual/tol", value);
    std::fprintf(stderr,
        "├────────────────────────────────────────────────────────────────────────────────┤\n"
        "│       State          Ritz eigenvalue  Residual norm   Residual/tol  Status     │\n");
    for (int a=0; a<davidsonPlan.nstates; ++a) {
        std::fprintf(stderr,
            "│  %10d  %+23.15e  %13.6e  %13.6e  %-9s  │\n",
            a+1, h_eigenvalues[a], h_residual_norms[a],
            h_residual_norms[a] / davidsonPlan.residual_tol,
            h_residual_norms[a] <= davidsonPlan.residual_tol ? "converged" : "pending");
    }
    std::snprintf(value, sizeof(value), "END Davidson | Iter %d", iteration);
    std::fprintf(stderr, "│  %-76s  │\n╰────────────────────────────────────────────────────────────────────────────────╯\n", value);
    return kSuccess;
}
// PHPT-TΘ=-R,P=I-UU+
int DavidsonSolver(
    const Context& context,
    const finitePlan& finite,
    DavidsonPlan& davidsonPlan,
    const double* d_veff
) noexcept {
    const RayleighRitzSubspace& rr = davidsonPlan.RRsubspace;
    if (davidsonPlan.status != kSuccess || rr.status != kSuccess)
        return report_davidson_status(kDavidsonNotInitialized, "DavidsonSolver");
    if (d_veff == nullptr || davidsonPlan.npoints == 0 || davidsonPlan.npoints > INT_MAX ||
        davidsonPlan.nstates <= 0 || davidsonPlan.max_iterations <= 0 ||
        !std::isfinite(davidsonPlan.residual_tol) || davidsonPlan.residual_tol <= 0 ||
        !std::isfinite(davidsonPlan.tNorm_tol) || davidsonPlan.tNorm_tol <= 0 ||
        davidsonPlan.tNorm_tol >= 1 || davidsonPlan.nextra_retain < 0 ||
        davidsonPlan.max_subspace_dim <= davidsonPlan.nstates ||
        size_t(davidsonPlan.max_subspace_dim) > davidsonPlan.npoints ||
        davidsonPlan.nextra_retain >= davidsonPlan.max_subspace_dim - davidsonPlan.nstates)
        return report_davidson_status(kInvalidArgument, "DavidsonSolver");
    if (finite.status != kSuccess || finite.d_coeff == nullptr ||
        davidsonPlan.d_psi == nullptr || davidsonPlan.d_eigenvalues == nullptr ||
        davidsonPlan.d_residual == nullptr || rr.d_V == nullptr || rr.d_HV == nullptr ||
        rr.d_Hhat == nullptr || rr.d_correction == nullptr || rr.d_ritzValues == nullptr ||
        rr.d_solver_work == nullptr || rr.d_solver_info == nullptr || rr.solver_lwork <= 0 ||
        rr.current_subspace_dim < davidsonPlan.nstates ||
        rr.current_subspace_dim > davidsonPlan.max_subspace_dim)
        return report_davidson_status(kInvalidState, "DavidsonSolver");
    if (finite.nx != davidsonPlan.nx || finite.ny != davidsonPlan.ny ||
        finite.nz != davidsonPlan.nz || finite.npoints != int(davidsonPlan.npoints))
        return report_davidson_status(kInvalidArgument, "DavidsonSolver");
    int ierr = validate_context(context);
    if (ierr != kSuccess) return report_davidson_status(ierr, "DavidsonSolver");
    auto fail = [&](int error, const char* stage, int iteration) {
        // A failed compute step may have partially overwritten the basis.
        davidsonPlan.status = kInvalidState;
        return report_davidson_status(error, stage, iteration);
    };
    // 每次求解分配一次主机范数缓冲，复用已有残差范数。
    std::unique_ptr<double[]> residual_norms(
        new (std::nothrow) double[davidsonPlan.nstates]);
    if (!residual_norms)
        return report_davidson_status(kAllocationError, "residual norm allocation");
    double* h_residual_norms = residual_norms.get();
    int nactive = 0, nadded = 0;
    bool hv_initialized = false;
    for (int iter=0;iter<davidsonPlan.max_iterations;iter++) {
        ierr = RayleighRitzDiagonalize(context,finite,davidsonPlan,d_veff,nadded,hv_initialized);
        if (ierr != kSuccess) return fail(ierr, "RayleighRitzDiagonalize", iter+1);
        ierr = DavidsonPreconditioner(context, finite, davidsonPlan, d_veff, nactive,
                                     h_residual_norms);
        if (ierr != kSuccess) return fail(ierr, "DavidsonPreconditioner", iter+1);
        if (nactive == 0)
        {
            ierr = write_log(context, davidsonPlan, iter+1, nactive, h_residual_norms);
            if (ierr != kSuccess) return fail(ierr, "write_log", iter+1);
            return report_davidson_status(kSuccess, "DavidsonSolver", iter+1);
        }
        ierr = UpdateSubspace(context,davidsonPlan);
        if (ierr != kSuccess) return fail(ierr, "UpdateSubspace", iter+1);
        ierr = ExtendSubspace(context,davidsonPlan,nactive,nadded);
        if (ierr != kSuccess) return fail(ierr, "ExtendSubspace", iter+1);
        if (nadded == 0)
            return report_davidson_status(kDavidsonStagnation, "DavidsonSolver", iter+1);
    }
    return report_davidson_status(kDavidsonMaxIterations, "DavidsonSolver",
                                   davidsonPlan.max_iterations);
}








// QR 分解
// cusolverStatus_t status = cusolverDnDgeqrf(
//     context.cusolver,           // cuSOLVER 句柄

//     npoints,                   // m：输入矩阵 V 的行数，即网格点数
//     nstates,                   // n：输入矩阵 V 的列数，即轨道数

//     RRsubspace.d_V,            // A：GPU 上的输入/输出矩阵
//                                // 输入：原始 V
//                                // 输出：上三角保存 R
//                                //       下三角保存 Householder 向量的信息
//                                // 此时还没有显式生成 Q

//     npoints,                   // lda：矩阵相邻两列起点之间的元素数
//                                // 无填充的列优先存储下，lda = m

//     RRsubspace.d_tau,          // tau：GPU 数组，由本函数写入
//                                // 保存 min(m,n) 个 Householder 反射系数
//                                // 当前需要 nstates 个 double

//     RRsubspace.d_solver_work,  // work：GPU 上的临时工作区

//     RRsubspace.solver_lwork,   // lwork：工作区长度，以 double 元素个数计
//                                // 必须满足 geqrf_bufferSize 查询的要求

//     RRsubspace.d_solver_info   // devInfo：GPU 上的一个 int
//                                // 0 表示成功
//                                // -i 表示第 i 个参数不合法
// );
// 利用上述结果生成薄正交矩阵 Q
// cusolverStatus_t status = cusolverDnDorgqr(
//     context.cusolver,           // cuSOLVER 句柄

//     npoints,                   // m：输出 Q 的行数
//     nstates,                   // n：输出 Q 的列数
//     nstates,                   // k：使用的 Householder 反射数量
//                                // 要求 0 <= k <= n <= m

//     RRsubspace.d_V,            // A：GPU 上的输入/输出矩阵
//                                // 输入：geqrf 留下的 Householder 向量信息
//                                // 输出：显式 Q，尺寸为 npoints × nstates
//                                // 原先存放的 R 会被覆盖

//     npoints,                   // lda：相邻两列起点之间的元素数

//     RRsubspace.d_tau,          // tau：读取 geqrf 生成的反射系数
//                                // 与 d_V 中的 Householder 信息配套使用

//     RRsubspace.d_solver_work,  // work：GPU 上的临时工作区

//     RRsubspace.solver_lwork,   // lwork：工作区长度，以 double 元素个数计
//                                // 必须满足 orgqr_bufferSize 查询的要求

//     RRsubspace.d_solver_info   // devInfo：GPU 上的一个 int
//                                // 0 表示成功
//                                // -i 表示第 i 个参数不合法
// );
