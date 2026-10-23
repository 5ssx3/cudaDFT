#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include <cstddef>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cublas_v2.h>   // cublasHandle_t
#include <cusolverDn.h>  // cusolverDnHandle_t

#include <cstddef>
// Status values shared by the mathematical wrappers below. CUDA/cuFFT native
// error codes are translated at their call sites, never returned as Status.
enum Status : int {
    kSuccess = 0,
    kInvalidArgument = 1,
    kInvalidState = 2,
    kSizeOverflow = 3,
    kAllocationError = 4,
    kCudaMemcpyError = 5,
    kCudaLaunchError = 6,
    kCudaSyncError = 7,
    kCudaFreeError = 8,
    kFftPlanError = 9,
    kFftStreamError = 10,
    kFftExecuteError = 11,
    kFftDestroyError = 12,
};

struct Context;
struct Vector {
    double x;
    double y;
    double z;
};
struct Fft3dPlan {
    cufftHandle Z2Zplan{};
    cufftHandle D2Zplan{};
    cufftHandle Z2Dplan{};
    cudaStream_t stream = nullptr;
    Context* context = nullptr;
    int nx{};
    int ny{};
    int nz{};
    Vector* d_G{};
    Vector* d_G_half{};
    double* d_lapG2_half{};
    int status = kInvalidState;
};
struct Context {
    int device_id = -1;
    cudaStream_t stream = nullptr;
    Fft3dPlan* fft = nullptr;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    int status = -1;
};

struct finitePlan
{
    int nx{};
    int ny{};
    int nz{};
    int npoints{};
    int half_width{};
    Vector gap{};
    const double* d_coeff = nullptr;
    double* d_halo_work = nullptr;
    int status = -1;
};

// Status messages.
const char* status_message(int status) noexcept;

// Host/device transfers.
// upload allocates *d_data; the caller releases it with cudaFree after success.
// Copy failures preserve kCudaMemcpyError. If cleanup also fails, *d_data keeps
// the allocation so the caller can retry releasing it; otherwise it is null.
int upload(double* h_data, void** d_data, int npoints);

int download(double* d_data, double* h_data, int npoints);

// Scalar and small object inputs use values; large host objects use references.
// Numerical buffers use pointers.
// Scalar outputs and newly allocated plan pointers use non-const references.
bool checked_product(std::size_t left, std::size_t right,
                     std::size_t& result) noexcept;

int grid_blocks(int npoints, int& blocks) noexcept;

// FFT plans.
int destroy_fft_plan(Fft3dPlan& fft) noexcept;

int create_fft_plan(Fft3dPlan& fft, int nx, int ny, int nz,
                    Context& context) noexcept;

// FFT execution and normalization are queued on fft.stream. Callers must
// synchronize before reading results on the host or reusing host input buffers.
int complex_fft_forward(const Fft3dPlan& fft, cuDoubleComplex* d_psi,
                        cuDoubleComplex* d_psi_fft) noexcept;

int real_fft_forward(const Fft3dPlan& fft, double* d_psi,
                     cufftDoubleComplex* d_psi_fft) noexcept;

int complex_fft_inverse(const Context& context, const Fft3dPlan& fft,
                        cuDoubleComplex* d_psi_fft, cuDoubleComplex* d_psi) noexcept;

int real_fft_inverse(const Context& context, const Fft3dPlan& fft,
                     cuDoubleComplex* d_psi_fft, double* d_psi) noexcept;

// Product integrals.
int RealReal_IntegralFGdV(const Context& context, const double* d_F,
                         const double* d_G, double dv,
                         int npoints, double* d_result) noexcept;

int ComplexReal_IntegralFGdV(const Context& context, const cuDoubleComplex* d_F,
                            const double* d_G, double dv,
                            int npoints, cuDoubleComplex* d_result) noexcept;

// This is integral(F * G) dV, not the conjugated inner product integral(conj(F) * G) dV.
int ComplexComplex_IntegralFGdV(const Context& context, const cuDoubleComplex* d_F,
                               const cuDoubleComplex* d_G, double dv,
                               int npoints, cuDoubleComplex* d_result) noexcept;

// FFT derivatives.
int complex_gradient(const Context& context, const Fft3dPlan& fft, cuDoubleComplex* d_psi,
                     cuDoubleComplex* d_grad_psi_x, cuDoubleComplex* d_grad_psi_y,
                     cuDoubleComplex* d_grad_psi_z) noexcept;

int real_gradient(const Context& context, const Fft3dPlan& fft, double* d_psi,
                     double* d_grad_psi_x, double* d_grad_psi_y,
                     double* d_grad_psi_z) noexcept;

int complex_laplacian(const Context& context, const Fft3dPlan& fft,
                      cuDoubleComplex* d_psi, cuDoubleComplex* d_lapl_psi) noexcept;

int real_laplacian(const Context& context, const Fft3dPlan& fft,
                      double* d_psi, double* d_lapl_psi) noexcept;

// Overloaded FFT interfaces.
int fft_forward(const Fft3dPlan& fft,
                double* psi, cuDoubleComplex* d_psi_fft) noexcept;

int fft_forward(const Fft3dPlan& fft,
                cuDoubleComplex* psi, cuDoubleComplex* d_psi_fft) noexcept;

int fft_inverse(const Context& context, const Fft3dPlan& fft,
                cuDoubleComplex* d_input_fft, double* psi) noexcept;

int fft_inverse(const Context& context, const Fft3dPlan& fft,
                cuDoubleComplex* d_input_fft, cuDoubleComplex* psi) noexcept;

int fft_laplacian(const Context& context, const Fft3dPlan& fft,
                  double* psi, double* result) noexcept;

int fft_laplacian(const Context& context, const Fft3dPlan& fft,
                  cuDoubleComplex* psi, cuDoubleComplex* result) noexcept;

int fft_gradient(const Context& context, const Fft3dPlan& fft,
                 double* psi, double* result_x,
                 double* result_y, double* result_z) noexcept;

int fft_gradient(const Context& context, const Fft3dPlan& fft,
                 cuDoubleComplex* psi, cuDoubleComplex* result_x,
                 cuDoubleComplex* result_y, cuDoubleComplex* result_z) noexcept;

int fft_init(Context& context, Fft3dPlan*& fft_out,
             int nx, int ny, int nz,
             const double* h_latticeVectors) noexcept;

// Finite differences and Hamiltonian application.
int creat_finitePlan(finitePlan& finite, int nx, int ny, int nz, int half_width,
                     double hx, double hy, double hz);

int apply_laplacian_finite(const Context& context, const finitePlan& finite,
                    const double* d_psi, double* d_lapl_psi) noexcept;

int apply_gradient_finite(const Context& context, const finitePlan& finite,
                    const double* d_psi, double* d_grad_psi_x, double* d_grad_psi_y, double* d_grad_psi_z) noexcept;

// Requires an initialized finitePlan. Execution is queued on context.stream;
// synchronize the stream before reading results and to check execution errors.
// Invalid arrays/grid/spacing/stencil return kInvalidArgument; an unready plan
// returns kInvalidState, excessive grid size kSizeOverflow, and launch failures
// kCudaLaunchError.
int apply_HamiltonOperator(const Context& context, const finitePlan& finite,
                    const double* d_psi, const double* d_veff, double* d_Hpsi) noexcept;

int apply_davidson_preconditioner(
    const Context& context,const finitePlan& finite,
    const double* d_veff,       // 有效势
    const double* d_eigenvalue, // 当前 Ritz 值，设备端单个 double
    const double* d_residual,   // 输入：一列残差，长度 npoints
    double* d_correction       // 输出：一列修正方向，长度 npoints
) noexcept;
// lbfgs
int update_q(Context& context,
    int n,
    double* q,
    const double* y,
    const double* rho,
    const double* dot) noexcept;

int update_r(Context& context,
    int n,
    double* r,
    const double* s,
    const double* a,
    const double* b,
    const double* rho) noexcept;

int RealArrayMulReal(Context& context, int n, double* d_RealArray, double a) noexcept;
int RealArrayPlusRealArray(Context& context, int n, double* d_RealArray1, double* d_RealArray2) noexcept;