#include "LBFGS.hpp"
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

const char* lbfgs_status_message(int status) noexcept
{
    switch (status) {
        case kLBFGSConverged: return "Gradient tolerance reached";
        case kLBFGSCublasError: return "cuBLAS call failed";
        case kLBFGSCusolverError: return "cuSOLVER call failed";
        case kLBFGSCurandError: return "cuRAND call failed";
        case kLBFGSNonFinite: return "Non-finite objective, norm or scalar";
        case kLBFGSNotDescent: return "Search direction is not a descent direction";
        case kLBFGSInvalidCurvature: return "Non-positive or unusable history curvature";
        case kLBFGSLineSearchFailed: return "Line search could not find an acceptable step";
        case kLBFGSEvaluationError: return "Objective/gradient callback failed";
        case kLBFGSNotInitialized: return "Optimizer is not initialized or is unusable";
        default: return status_message(status);
    }
}

namespace {
inline bool check_cublas_status(cublasStatus_t status,
    const char* expression, const char* file, int line)
{
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "[cuBLAS] status=%d (%s), call=%s, at %s:%d\n",
        static_cast<int>(status), cublasGetStatusName(status), expression, file, line);
    return false;
}
inline bool check_cusolver_status(cusolverStatus_t status,
    const char* expression, const char* file, int line)
{
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "[cuSOLVER] status=%d, call=%s, at %s:%d\n",
        static_cast<int>(status), expression, file, line);
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

int report_lbfgs_status(int result, const char* stage, const LBFGSState& lbfgs)
{
    if (result != kSuccess) write_log(lbfgs, result, stage);
    return result;
}

template<class T>
int release_device(T*& ptr) noexcept
{
    if (ptr == nullptr) return kSuccess;
    if (!CHECK_CUDA(cudaFree(ptr))) return kCudaFreeError;
    ptr = nullptr;
    return kSuccess;
}

int validate_context(const Context& context)
{
    if (context.cublas == nullptr) return kInvalidState;
    cudaStream_t stream = nullptr;
    cublasPointerMode_t mode{};
    if (!CHECK_CUBLAS(cublasGetStream(context.cublas, &stream)) ||
        !CHECK_CUBLAS(cublasGetPointerMode(context.cublas, &mode)))
        return kLBFGSCublasError;
    return stream == context.stream && mode == CUBLAS_POINTER_MODE_HOST
        ? kSuccess : kInvalidState;
}

bool valid_parameters(const LBFGSState& lbfgs)
{
    return std::isfinite(lbfgs.c1) && std::isfinite(lbfgs.c2) &&
        std::isfinite(lbfgs.g_tol) && lbfgs.c1 > 0 &&
        lbfgs.c1 < lbfgs.c2 && lbfgs.c2 < 1 && lbfgs.g_tol >= 0;
}

int evaluate_point(Context& context, LBFGSState& lbfgs, calculate_F evaluate,
                   void* user_data, const double* d_x, double& f,
                   double* d_g, double& g_norm)
{
    ++lbfgs.evaluations;
    const int ierr = evaluate(context, d_x, f, d_g, user_data);
    if (ierr != 0) {
        std::fprintf(stderr, "[L-BFGS callback] status=%d, evaluation=%zu\n",
            ierr, lbfgs.evaluations);
        return kLBFGSEvaluationError;
    }
    const int context_status = validate_context(context);
    if (context_status != kSuccess) return context_status;
    if (!CHECK_CUDA(cudaGetLastError())) return kCudaLaunchError;
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return kCudaSyncError;
    if (!CHECK_CUBLAS(cublasDnrm2(context.cublas, static_cast<int>(lbfgs.n),
                                 d_g, 1, &g_norm)))
        return kLBFGSCublasError;
    if (!std::isfinite(f) || !std::isfinite(g_norm)) return kLBFGSNonFinite;
    return kSuccess;
}
}
// 时间顺序下标 t -> 数组实际下标 i
inline int t2i(const LBFGSState& lbfgs, int t) noexcept {
    const int i = lbfgs.history_next - lbfgs.history_count + t;
    return i < 0 ? i + lbfgs.history_capacity : i;
}
// 数组实际下标 i -> 时间顺序下标 t
inline int i2t(const LBFGSState& lbfgs, int i) noexcept {
    int oldest = lbfgs.history_next - lbfgs.history_count;
    if (oldest < 0)
        oldest += lbfgs.history_capacity;

    const int t = i - oldest;
    return t < 0 ? t + lbfgs.history_capacity : t;
}
int destory_LBFGSOptimizer(LBFGSState& lbfgs) {
    int result = kSuccess;
    auto release = [&result](auto*& ptr) {
        const int ierr = release_device(ptr);
        if (result == kSuccess) result = ierr;
    };
    release(lbfgs.d_x);
    release(lbfgs.d_g);
    release(lbfgs.d_x_trial);
    release(lbfgs.d_g_trial);
    release(lbfgs.d_p);
    release(lbfgs.d_s);
    release(lbfgs.d_y);
    release(lbfgs.d_rho);
    release(lbfgs.d_alpha);
    lbfgs.status = kLBFGSNotInitialized;
    if (result == kSuccess) lbfgs = LBFGSState{};
    return report_lbfgs_status(result, "destory_LBFGSOptimizer", lbfgs);
}
int LBFGSOptimizer_init(Context& context,LBFGSState& lbfgs,int n, 
    int history_capacity, calculate_F evaluate, void* user_data, double* d_x_guess){

    if (n <= 0 || history_capacity <= 0 || evaluate == nullptr ||
        d_x_guess == nullptr || !valid_parameters(lbfgs))
        return report_lbfgs_status(kInvalidArgument, "LBFGSOptimizer_init", lbfgs);
    if (lbfgs.d_x || lbfgs.d_g || lbfgs.d_x_trial || lbfgs.d_g_trial ||
        lbfgs.d_p || lbfgs.d_rho || lbfgs.d_alpha || lbfgs.d_s || lbfgs.d_y)
        return report_lbfgs_status(kInvalidState, "LBFGSOptimizer_init", lbfgs);
    const int context_status = validate_context(context);
    if (context_status != kSuccess)
        return report_lbfgs_status(context_status, "LBFGSOptimizer_init", lbfgs);
    size_t nbytes = 0, historynbytes = 0, historybytes = 0;
    if (!checked_product(size_t(n), sizeof(double), nbytes) ||
        !checked_product(size_t(history_capacity), nbytes, historynbytes) ||
        !checked_product(size_t(history_capacity), sizeof(double), historybytes))
        return report_lbfgs_status(kSizeOverflow, "LBFGSOptimizer_init", lbfgs);

    lbfgs.n = n;
    lbfgs.history_capacity = history_capacity;
    lbfgs.history_next = 0;
    lbfgs.history_count = 0;
    lbfgs.max_linesearch_step = 50;
    lbfgs.status = kLBFGSNotInitialized;
    lbfgs.iterations = lbfgs.evaluations = 0;
    lbfgs.line_search_trials = 0;
    lbfgs.step_size = 0.0;
    lbfgs.f = lbfgs.g_norm = std::numeric_limits<double>::quiet_NaN();
    auto fail = [&](int error) {
        report_lbfgs_status(error, "LBFGSOptimizer_init", lbfgs);
        destory_LBFGSOptimizer(lbfgs);
        return error;
    };
    if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_x),nbytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_g),nbytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_x_trial),nbytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_g_trial),nbytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_p),nbytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_rho),historybytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_alpha),historybytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_s),historynbytes)) ||
        !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&lbfgs.d_y),historynbytes)))
        return fail(kAllocationError);
    const int ierr = evaluate_point(context, lbfgs, evaluate, user_data,
                                    d_x_guess, lbfgs.f, lbfgs.d_g, lbfgs.g_norm);
    if (ierr != kSuccess) return fail(ierr);
    if (!CHECK_CUDA(cudaMemcpyAsync(lbfgs.d_x, d_x_guess, nbytes,
                                    cudaMemcpyDeviceToDevice, context.stream)))
        return fail(kCudaMemcpyError);
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return fail(kCudaSyncError);
    lbfgs.status = kSuccess;
    return kSuccess;
}
// p=-Hg
int DoubleLoopHg(Context& context,LBFGSState& lbfgs){
    double* d_a = lbfgs.d_alpha;
    double* d_r = lbfgs.d_p;
    // 线搜索尚未开始，复用试探数组，避免失败路径遗留临时分配。
    double* d_q = lbfgs.d_x_trial;
    double* d_bi = lbfgs.d_g_trial;
    double* d_g = lbfgs.d_g;

    int n = lbfgs.n;
    int history_count = lbfgs.history_count;
    size_t nbytes = n*sizeof(double);
    if (history_count == 0) {
        if (!CHECK_CUDA(cudaMemcpyAsync(d_r, d_g, nbytes,
                         cudaMemcpyDeviceToDevice, context.stream)))
            return kCudaMemcpyError;
        const int ierr = RealArrayMulReal(context, n, d_r, -1.0);
        if (ierr != kSuccess) return ierr;
        return CHECK_CUDA(cudaGetLastError()) ? kSuccess : kCudaLaunchError;
    }
    // 所有退出路径恢复 HOST，错误时也不会将设备标量模式留给调用者。
    auto calculate = [&]() -> int {
    // q=g
    if (!CHECK_CUDA(cudaMemcpyAsync(d_q, d_g, nbytes,
                                    cudaMemcpyDeviceToDevice, context.stream)))
        return kCudaMemcpyError;
    if (!CHECK_CUBLAS(cublasSetPointerMode(context.cublas, CUBLAS_POINTER_MODE_DEVICE)))
        return kLBFGSCublasError;
    // at=(st)q; q=q-rhot*at*yt
    for(int t=lbfgs.history_count-1;t>=0;t--){
        int i = t2i(lbfgs,t);
        const size_t offset = size_t(n) * i;
        if (!CHECK_CUBLAS(cublasDdot(
            context.cublas,
            n,
            lbfgs.d_s+offset, 1,
            d_q , 1,
            d_a+i
        ))) return kLBFGSCublasError;
        const int ierr = update_q(context,n,d_q,lbfgs.d_y+offset,lbfgs.d_rho+i,d_a+i);
        if (ierr != kSuccess) return ierr;
        if (!CHECK_CUDA(cudaGetLastError())) return kCudaLaunchError;
    }
    // r=gamma q
    if (!CHECK_CUBLAS(cublasSetPointerMode(context.cublas, CUBLAS_POINTER_MODE_HOST)))
        return kLBFGSCublasError;
    if (!CHECK_CUDA(cudaMemcpyAsync(d_r, d_q, nbytes,
                                    cudaMemcpyDeviceToDevice, context.stream)))
        return kCudaMemcpyError;
    double sy = 0;
    double yy = 0;
    const int latest_i = t2i(lbfgs, history_count - 1);
    const size_t latest_offset = size_t(n) * latest_i;
    if (!CHECK_CUBLAS(cublasDdot(
        context.cublas,
        n,
        lbfgs.d_s+latest_offset, 1,
        lbfgs.d_y+latest_offset, 1,
        &sy
    ))) return kLBFGSCublasError;
    if (!CHECK_CUBLAS(cublasDdot(
        context.cublas,
        n,
        lbfgs.d_y+latest_offset, 1,
        lbfgs.d_y+latest_offset, 1,
        &yy
    ))) return kLBFGSCublasError;
    if (!std::isfinite(sy) || !std::isfinite(yy)) return kLBFGSNonFinite;
    if (sy <= 0 || yy <= 0) return kLBFGSInvalidCurvature;
    lbfgs.gamma=sy/yy;
    if (!std::isfinite(lbfgs.gamma) || lbfgs.gamma <= 0) return kLBFGSInvalidCurvature;
    if (!CHECK_CUBLAS(cublasDscal(
        context.cublas,
        n,
        &lbfgs.gamma,    // 主机上的缩放系数
        d_r, 1           // 被原地缩放的数据
    ))) return kLBFGSCublasError;
    // bt=(yt)r; r=r+rhot*st*(at-bt)
    if (!CHECK_CUBLAS(cublasSetPointerMode(context.cublas, CUBLAS_POINTER_MODE_DEVICE)))
        return kLBFGSCublasError;
    for(int t=0;t<history_count;t++){
        int i = t2i(lbfgs,t);
        const size_t offset = size_t(n) * i;
        if (!CHECK_CUBLAS(cublasDdot(
            context.cublas,
            n,
            lbfgs.d_y+offset, 1,
            d_r, 1,
            d_bi
        ))) return kLBFGSCublasError;
        const int ierr = update_r(context, n, d_r, lbfgs.d_s+offset, d_a+i, d_bi, lbfgs.d_rho+i);
        if (ierr != kSuccess) return ierr;
        if (!CHECK_CUDA(cudaGetLastError())) return kCudaLaunchError;
    }
    const int ierr = RealArrayMulReal(context,n,d_r,-1);
    if (ierr != kSuccess) return ierr;
    return CHECK_CUDA(cudaGetLastError()) ? kSuccess : kCudaLaunchError;
    };
    //
    const int result = calculate();
    const bool restored = CHECK_CUBLAS(cublasSetPointerMode(context.cublas, CUBLAS_POINTER_MODE_HOST));
    if (result != kSuccess) return result;
    if (!restored) return kLBFGSCublasError;
    return kSuccess;
}
int UpdataHistory(Context& context, LBFGSState& lbfgs) {
    const int n = static_cast<int>(lbfgs.n);
    const int i = lbfgs.history_next;
    const size_t offset = lbfgs.n * static_cast<size_t>(i);

    double* d_s = lbfgs.d_s + offset;
    double* d_y = lbfgs.d_y + offset;
    const double minus_one = -1.0;

    // s = x_trial - x
    if (!CHECK_CUBLAS(cublasDcopy(
        context.cublas, n,
        lbfgs.d_x_trial, 1, d_s, 1))) return kLBFGSCublasError;

    if (!CHECK_CUBLAS(cublasDaxpy(
        context.cublas, n, &minus_one,
        lbfgs.d_x, 1, d_s, 1))) return kLBFGSCublasError;

    // y = g_trial - g
    if (!CHECK_CUBLAS(cublasDcopy(
        context.cublas, n,
        lbfgs.d_g_trial, 1, d_y, 1))) return kLBFGSCublasError;

    if (!CHECK_CUBLAS(cublasDaxpy(
        context.cublas, n, &minus_one,
        lbfgs.d_g, 1, d_y, 1))) return kLBFGSCublasError;

    // rho = 1 / dot(s, y)
    double sy = 0.0;
    if (!CHECK_CUBLAS(cublasDdot(
        context.cublas, n,
        d_s, 1, d_y, 1, &sy))) return kLBFGSCublasError;

    if (!std::isfinite(sy)) return kLBFGSNonFinite;
    if (sy <= 0) return kLBFGSInvalidCurvature;
    const double rho = 1.0 / sy;
    if (!std::isfinite(rho)) return kLBFGSInvalidCurvature;
    if (!CHECK_CUDA(cudaMemcpyAsync(
        lbfgs.d_rho + i, &rho, sizeof(double),
        cudaMemcpyHostToDevice, context.stream))) return kCudaMemcpyError;

    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return kCudaSyncError;

    lbfgs.history_next = (i + 1) % lbfgs.history_capacity;
    if (lbfgs.history_count < lbfgs.history_capacity)
        ++lbfgs.history_count;
    return kSuccess;
}
int WolfeLineSearch(Context& context, LBFGSState& lbfgs, calculate_F evaluate, void* user_data){
    int n = lbfgs.n;
    double* d_p=lbfgs.d_p;
    double* d_g=lbfgs.d_g;
    double* d_x_trial=lbfgs.d_x_trial;
    double* d_g_trial=lbfgs.d_g_trial;
    double d0 = 0.0;
    if (!CHECK_CUBLAS(cublasDdot(
        context.cublas,
        n,
        d_g, 1,
        d_p, 1,
        &d0
    ))) return kLBFGSCublasError;
    if (!std::isfinite(d0)) return kLBFGSNonFinite;
    if (d0 >= 0) return kLBFGSNotDescent;
    //
    double eta = 1.0;
    double eta_lo = 0.0;
    double eta_hi = std::numeric_limits<double>::infinity();
    //
    double f_trial = 0.0;
    int max_trials = lbfgs.max_linesearch_step;
    for (int i=0;i<max_trials;i++){

        if (!CHECK_CUDA(cudaMemcpyAsync(d_x_trial, lbfgs.d_x, size_t(n) * sizeof(double),
                         cudaMemcpyDeviceToDevice, context.stream)))
            return kCudaMemcpyError;

        // d_x_trial += eta * d_p；不修改 d_p
        if (!CHECK_CUBLAS(cublasDaxpy(
            context.cublas,  // cublas 句柄
            n,               // 向量长度：一共处理 n 个元素
            &eta,            // α：标量系数（这里是 eta，主机端指针）
            d_p,             // x：设备端向量 d_p
            1,               // x 的步长 incx = 1（连续内存）
            d_x_trial,       // y：设备端向量 d_x_trial
            1                // y 的步长 incy = 1（连续内存）
        ))) return kLBFGSCublasError;
        lbfgs.line_search_trials = i + 1;
        double g_trial_norm = 0.0;
        const int ierr = evaluate_point(context, lbfgs, evaluate, user_data,
            d_x_trial, f_trial, d_g_trial, g_trial_norm);
        if (ierr != kSuccess) return ierr;
        if(f_trial > lbfgs.f+lbfgs.c1*eta*d0){
            eta_hi = eta;
        }
        else
        {
            double dtrial = 0.0;
            if (!CHECK_CUBLAS(cublasDdot(
                context.cublas,
                n,
                d_g_trial, 1,
                d_p, 1,
                &dtrial))) return kLBFGSCublasError;
            if (!std::isfinite(dtrial)) return kLBFGSNonFinite;
            const bool trial_converged = g_trial_norm <= lbfgs.g_tol;
            if (trial_converged || std::abs(dtrial) <= lbfgs.c2 * std::abs(d0)) {
                if (!trial_converged) {
                    const int history_status = UpdataHistory(context, lbfgs);
                    if (history_status != kSuccess) return history_status;
                }
                if (!CHECK_CUDA(cudaMemcpyAsync(lbfgs.d_x, d_x_trial,
                        size_t(n) * sizeof(double), cudaMemcpyDeviceToDevice, context.stream)) ||
                    !CHECK_CUDA(cudaMemcpyAsync(lbfgs.d_g, d_g_trial,
                        size_t(n) * sizeof(double), cudaMemcpyDeviceToDevice, context.stream)))
                    return kCudaMemcpyError;
                if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return kCudaSyncError;
                lbfgs.f = f_trial;
                lbfgs.g_norm = g_trial_norm;
                lbfgs.step_size = eta;
                ++lbfgs.iterations;
                if (trial_converged) return kLBFGSConverged;
                return kSuccess;
            }
            if(dtrial>0) 
                eta_hi=eta;
            else
                eta_lo=eta;
        }
        eta = std::isfinite(eta_hi)
            ? eta_lo + 0.5 * (eta_hi - eta_lo)
            : 2.0 * eta;
        if (!std::isfinite(eta) || eta <= 0) return kLBFGSLineSearchFailed;
    }
    return kLBFGSLineSearchFailed;
}
int LBFGSOptimizer(Context& context, LBFGSState& lbfgs, calculate_F evaluate, void* user_data)
{   
    lbfgs.step_size = 0.0;
    lbfgs.line_search_trials = 0;
    auto report = [&](int result) {
        return report_lbfgs_status(result, "LBFGSOptimizer", lbfgs);
    };
    if (lbfgs.status != kSuccess) return report(kLBFGSNotInitialized);
    if (evaluate == nullptr || !valid_parameters(lbfgs)) return report(kInvalidArgument);
    if (lbfgs.n == 0 || lbfgs.n > INT_MAX || lbfgs.history_capacity <= 0 ||
        lbfgs.history_count < 0 || lbfgs.history_count > lbfgs.history_capacity ||
        lbfgs.history_next < 0 || lbfgs.history_next >= lbfgs.history_capacity ||
        !lbfgs.d_x || !lbfgs.d_g || !lbfgs.d_x_trial || !lbfgs.d_g_trial ||
        !lbfgs.d_p || !lbfgs.d_s || !lbfgs.d_y || !lbfgs.d_rho || !lbfgs.d_alpha)
        return report(kInvalidState);
    const int context_status = validate_context(context);
    if (context_status != kSuccess) return report(context_status);
    auto finish = [&](int result) {
        // 确认异步操作已结束；硬件/数值故障后禁用此状态，防止继续使用部分更新。
        if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) result = kCudaSyncError;
        if (result != kSuccess && result != kLBFGSConverged &&
            result != kLBFGSEvaluationError && result != kLBFGSLineSearchFailed)
            lbfgs.status = result;
        return report(result);
    };
    if (!CHECK_CUBLAS(cublasDnrm2(context.cublas, static_cast<int>(lbfgs.n),
                                  lbfgs.d_g, 1, &lbfgs.g_norm)))
        return finish(kLBFGSCublasError);
    if (!std::isfinite(lbfgs.f) || !std::isfinite(lbfgs.g_norm))
        return finish(kLBFGSNonFinite);
    if (lbfgs.g_norm <= lbfgs.g_tol) {
        return finish(kLBFGSConverged);
    }
    // pk=-Hgk
    const int ierr = DoubleLoopHg(context,lbfgs);
    if (ierr != kSuccess) return finish(ierr);
    //
    return finish(WolfeLineSearch(context, lbfgs, evaluate, user_data));
}
void write_log(const LBFGSState& lbfgs, int result, const char* stage) noexcept {
    if (result == kSuccess) return;
    // 框内只用 ASCII 英文标签，终端中保持列宽一致；不额外下载设备数据。
    const char* border = "+------------------------------------------------------------------------------+\n";
    auto row = [](const char* label, const char* value) {
        std::fprintf(stderr, "| %-14s %-61.61s |\n", label, value);
    };
    char value[160];
    std::fprintf(stderr, "\n%s| %-76s |\n", border,
        result == kLBFGSConverged ? "L-BFGS | CONVERGED" : "L-BFGS | FAILED");
    row("Stage", stage);
    std::snprintf(value, sizeof(value), "%d (%s)", result, lbfgs_status_message(result));
    row("Status", value);
    std::snprintf(value, sizeof(value), "%zu accepted / %zu evaluations / %d last trials",
        lbfgs.iterations, lbfgs.evaluations, lbfgs.line_search_trials);
    row("Work", value);
    if (lbfgs.evaluations == 0) {
        row("f / ||g||_2", "n/a");
    } else {
        std::snprintf(value, sizeof(value), "%.12e", lbfgs.f);
        row("f", value);
        std::snprintf(value, sizeof(value), "%.6e / %.6e", lbfgs.g_norm, lbfgs.g_tol);
        row("||g||_2 / tol", value);
    }
    std::snprintf(value, sizeof(value), "%.6e", lbfgs.step_size);
    row("Step size", value);
    std::fprintf(stderr, "%s", border);
}
