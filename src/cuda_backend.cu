#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cusolverDn.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <cufft.h>
#include <cuComplex.h>
#include <cstddef>
#include <iostream>
struct Fft3dPlan;
namespace {

enum Status : int {
    kSuccess = 0,
    kInvalidArgument = 1,
    kNoDevice = 2,
    kCudaRuntimeError = 3,
    kCublasError = 4,
    kCusolverError = 5,
    kAllocationError = 6,
    kInvalidState = 7,
};

enum SubspaceKind : int {
    kRealSubspace = 1,
    kComplexSubspace = 2,
    kNoncollinearSubspace = 3,
};

struct Context {
    int device_id = -1;
    int grid_shape[3] = {0, 0, 0};
    int local_states = 0;
    int fd_half_width = 0;
    int npoints = 0;
    int nshift = 0;
    bool configured = false;
    bool fd_uploaded = false;
    int subspace_count = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    Fft3dPlan* fft = nullptr;

    double* d_lapl = nullptr;
    double* d_grad = nullptr;
    double* d_tbmat = nullptr;
    double* d_kvec = nullptr;
    int* d_cell_mu = nullptr;
    double* d_veff = nullptr;
    double* d_vec_real = nullptr;
    double* d_hvec_real = nullptr;
    cuDoubleComplex* d_vec_complex = nullptr;
    cuDoubleComplex* d_hvec_complex = nullptr;
};

struct ResidentSubspace {
    Context* context = nullptr;
    SubspaceKind kind = kRealSubspace;
    int rows = 0;
    int nstates = 0;
    int nkpoints = 0;
    int potential_components = 0;
    bool potential_ready = false;
    bool registered = false;

    double* d_psi_real = nullptr;
    double* d_aux_real = nullptr;
    cuDoubleComplex* d_psi_complex = nullptr;
    cuDoubleComplex* d_aux_complex = nullptr;

    double* d_veff = nullptr;
    double* d_kvec = nullptr;
    double* d_hhat_real = nullptr;
    double* d_shat_real = nullptr;
    cuDoubleComplex* d_hhat_complex = nullptr;
    cuDoubleComplex* d_shat_complex = nullptr;
    double* d_eigenvalues = nullptr;
    double* d_weights = nullptr;
    double* d_density = nullptr;
    double* d_filter_params = nullptr;
    void* d_solver_work = nullptr;
    std::size_t solver_work_bytes = 0;
    int* d_info = nullptr;
};

int spectral_hamiltonian(Context* context, SubspaceKind kind,
                         const double* veff, const double* kvec,
                         const void* input, void* output,
                         int nstates, int nkpoints) noexcept;
int run_chebyshev_fft(ResidentSubspace* subspace, int degree) noexcept;
int lanczos_fft_real(ResidentSubspace* subspace, const double* vectors,
                     int steps, double* alpha, double* beta) noexcept;
int lanczos_fft_complex(ResidentSubspace* subspace,
                        const cuDoubleComplex* vectors, int steps,
                        double* alpha, double* beta) noexcept;

bool checked_product(std::size_t left, std::size_t right,
                     std::size_t* result) noexcept {
    if (result == nullptr) return false;
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    *result = left * right;
    return true;
}

cudaError_t free_device_buffers(Context* context) noexcept {
    if (context == nullptr) return cudaSuccess;
    cudaError_t first_error = cudaSuccess;
#define INVKS_FREE_DEVICE(pointer)                                             \
    if ((pointer) != nullptr) {                                                \
        const cudaError_t error = cudaFree(pointer);                           \
        if (first_error == cudaSuccess && error != cudaSuccess) first_error = error; \
        (pointer) = nullptr;                                                   \
    }
    INVKS_FREE_DEVICE(context->d_lapl);
    INVKS_FREE_DEVICE(context->d_grad);
    INVKS_FREE_DEVICE(context->d_tbmat);
    INVKS_FREE_DEVICE(context->d_kvec);
    INVKS_FREE_DEVICE(context->d_cell_mu);
    INVKS_FREE_DEVICE(context->d_veff);
    INVKS_FREE_DEVICE(context->d_vec_real);
    INVKS_FREE_DEVICE(context->d_hvec_real);
    INVKS_FREE_DEVICE(context->d_vec_complex);
    INVKS_FREE_DEVICE(context->d_hvec_complex);
#undef INVKS_FREE_DEVICE
    return first_error;
}

void release_context(Context* context) noexcept {
    if (context == nullptr) return;
    free_device_buffers(context);
    if (context->cusolver != nullptr) cusolverDnDestroy(context->cusolver);
    if (context->cublas != nullptr) cublasDestroy(context->cublas);
    if (context->stream != nullptr) cudaStreamDestroy(context->stream);
    delete context;
}

int allocate_device_buffers(Context* context) noexcept {
    if (context == nullptr || context->npoints <= 0 || context->nshift <= 0) {
        return kInvalidArgument;
    }
    const std::size_t n = static_cast<std::size_t>(context->npoints);
    const std::size_t nshift = static_cast<std::size_t>(context->nshift);
    std::size_t count = 0;
    std::size_t bytes = 0;

#define INVKS_ALLOC_DEVICE(pointer, byte_count)                               \
    if (cudaMalloc(reinterpret_cast<void**>(&(pointer)), (byte_count)) != cudaSuccess) { \
        free_device_buffers(context);                                         \
        return kAllocationError;                                              \
    }
    if (!checked_product(nshift, 6, &count) ||
        !checked_product(count, sizeof(double), &bytes)) {
        return kAllocationError;
    }
    INVKS_ALLOC_DEVICE(context->d_lapl, bytes);
    if (!checked_product(nshift, 3, &count) ||
        !checked_product(count, sizeof(double), &bytes)) {
        free_device_buffers(context);
        return kAllocationError;
    }
    INVKS_ALLOC_DEVICE(context->d_grad, bytes);
    INVKS_ALLOC_DEVICE(context->d_tbmat, 9 * sizeof(double));
    INVKS_ALLOC_DEVICE(context->d_kvec, 3 * sizeof(double));
    INVKS_ALLOC_DEVICE(context->d_cell_mu, 9 * sizeof(int));
    if (!checked_product(n, 4, &count) ||
        !checked_product(count, sizeof(double), &bytes)) {
        free_device_buffers(context);
        return kAllocationError;
    }
    INVKS_ALLOC_DEVICE(context->d_veff, bytes);
    if (!checked_product(n, sizeof(double), &bytes)) {
        free_device_buffers(context);
        return kAllocationError;
    }
    INVKS_ALLOC_DEVICE(context->d_vec_real, bytes);
    INVKS_ALLOC_DEVICE(context->d_hvec_real, bytes);
    if (!checked_product(n, 2, &count) ||
        !checked_product(count, sizeof(cuDoubleComplex), &bytes)) {
        free_device_buffers(context);
        return kAllocationError;
    }
    INVKS_ALLOC_DEVICE(context->d_vec_complex, bytes);
    INVKS_ALLOC_DEVICE(context->d_hvec_complex, bytes);
#undef INVKS_ALLOC_DEVICE
    return kSuccess;
}

cudaError_t free_subspace_buffers(ResidentSubspace* subspace) noexcept {
    if (subspace == nullptr) return cudaSuccess;
    cudaError_t first_error = cudaSuccess;
#define INVKS_FREE_SUBSPACE(pointer)                                           \
    if ((pointer) != nullptr) {                                                \
        const cudaError_t error = cudaFree(pointer);                           \
        if (first_error == cudaSuccess && error != cudaSuccess) first_error = error; \
        (pointer) = nullptr;                                                   \
    }
    INVKS_FREE_SUBSPACE(subspace->d_psi_real);
    INVKS_FREE_SUBSPACE(subspace->d_aux_real);
    INVKS_FREE_SUBSPACE(subspace->d_psi_complex);
    INVKS_FREE_SUBSPACE(subspace->d_aux_complex);
    INVKS_FREE_SUBSPACE(subspace->d_veff);
    INVKS_FREE_SUBSPACE(subspace->d_kvec);
    INVKS_FREE_SUBSPACE(subspace->d_hhat_real);
    INVKS_FREE_SUBSPACE(subspace->d_shat_real);
    INVKS_FREE_SUBSPACE(subspace->d_hhat_complex);
    INVKS_FREE_SUBSPACE(subspace->d_shat_complex);
    INVKS_FREE_SUBSPACE(subspace->d_eigenvalues);
    INVKS_FREE_SUBSPACE(subspace->d_weights);
    INVKS_FREE_SUBSPACE(subspace->d_density);
    INVKS_FREE_SUBSPACE(subspace->d_filter_params);
    INVKS_FREE_SUBSPACE(subspace->d_solver_work);
    INVKS_FREE_SUBSPACE(subspace->d_info);
#undef INVKS_FREE_SUBSPACE
    subspace->solver_work_bytes = 0;
    return first_error;
}

void release_subspace(ResidentSubspace* subspace) noexcept {
    if (subspace == nullptr) return;
    Context* context = subspace->context;
    free_subspace_buffers(subspace);
    if (subspace->registered && context != nullptr &&
        context->subspace_count > 0) {
        --context->subspace_count;
    }
    delete subspace;
}

int allocate_subspace_buffers(ResidentSubspace* subspace) noexcept {
    if (subspace == nullptr || subspace->context == nullptr ||
        subspace->rows <= 0 || subspace->nstates <= 0 ||
        subspace->nkpoints <= 0 ||
        subspace->potential_components <= 0) {
        return kInvalidArgument;
    }

    std::size_t wave_elements = 0;
    std::size_t wave_bytes = 0;
    std::size_t projected_elements = 0;
    std::size_t projected_bytes = 0;
    std::size_t potential_elements = 0;
    std::size_t density_elements = 0;
    std::size_t state_elements = 0;
    const std::size_t scalar_bytes =
        subspace->kind == kRealSubspace ? sizeof(double)
                                        : sizeof(cuDoubleComplex);
    if (!checked_product(static_cast<std::size_t>(subspace->rows),
                         static_cast<std::size_t>(subspace->nstates),
                         &wave_elements) ||
        !checked_product(wave_elements,
                         static_cast<std::size_t>(subspace->nkpoints),
                         &wave_elements) ||
        !checked_product(static_cast<std::size_t>(subspace->nstates),
                         static_cast<std::size_t>(subspace->nstates),
                         &projected_elements) ||
        !checked_product(projected_elements,
                         static_cast<std::size_t>(subspace->nkpoints),
                         &projected_elements) ||
        !checked_product(
            static_cast<std::size_t>(subspace->context->npoints),
            static_cast<std::size_t>(subspace->potential_components),
            &potential_elements) ||
        !checked_product(
            static_cast<std::size_t>(subspace->context->npoints),
            static_cast<std::size_t>(subspace->potential_components),
            &density_elements) ||
        !checked_product(density_elements,
                         static_cast<std::size_t>(subspace->nkpoints),
                         &density_elements) ||
        !checked_product(static_cast<std::size_t>(subspace->nstates),
                         static_cast<std::size_t>(subspace->nkpoints),
                         &state_elements)) {
        return kAllocationError;
    }
    if (!checked_product(wave_elements, scalar_bytes, &wave_bytes) ||
        !checked_product(projected_elements, scalar_bytes, &projected_bytes)) {
        return kAllocationError;
    }

#define INVKS_ALLOC_SUBSPACE(pointer, byte_count)                              \
    if (cudaMalloc(reinterpret_cast<void**>(&(pointer)), (byte_count)) != cudaSuccess) { \
        free_subspace_buffers(subspace);                                       \
        return kAllocationError;                                               \
    }
    if (subspace->kind == kRealSubspace) {
        INVKS_ALLOC_SUBSPACE(subspace->d_psi_real, wave_bytes);
        INVKS_ALLOC_SUBSPACE(subspace->d_aux_real, wave_bytes);
        INVKS_ALLOC_SUBSPACE(subspace->d_hhat_real, projected_bytes);
        INVKS_ALLOC_SUBSPACE(subspace->d_shat_real, projected_bytes);
    } else {
        INVKS_ALLOC_SUBSPACE(subspace->d_psi_complex, wave_bytes);
        INVKS_ALLOC_SUBSPACE(subspace->d_aux_complex, wave_bytes);
        INVKS_ALLOC_SUBSPACE(subspace->d_hhat_complex, projected_bytes);
        INVKS_ALLOC_SUBSPACE(subspace->d_shat_complex, projected_bytes);
        INVKS_ALLOC_SUBSPACE(
            subspace->d_kvec,
            3 * static_cast<std::size_t>(subspace->nkpoints) * sizeof(double));
    }
    INVKS_ALLOC_SUBSPACE(
        subspace->d_veff, potential_elements * sizeof(double));
    INVKS_ALLOC_SUBSPACE(
        subspace->d_eigenvalues,
        state_elements * sizeof(double));
    INVKS_ALLOC_SUBSPACE(
        subspace->d_weights,
        state_elements * sizeof(double));
    INVKS_ALLOC_SUBSPACE(
        subspace->d_density, density_elements * sizeof(double));
    INVKS_ALLOC_SUBSPACE(
        subspace->d_filter_params,
        6 * static_cast<std::size_t>(subspace->nkpoints) * sizeof(double));
    INVKS_ALLOC_SUBSPACE(subspace->d_info, sizeof(int));
#undef INVKS_ALLOC_SUBSPACE
    return kSuccess;
}

__device__ inline cuDoubleComplex scale_complex(cuDoubleComplex value,
                                                 double factor) {
    return make_cuDoubleComplex(factor * cuCreal(value), factor * cuCimag(value));
}

__global__ void update_filter_params_kernel(
    double* filter_params, int nkpoints) {
    const int batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= nkpoints) return;
    const double sigma = filter_params[3 * nkpoints + batch];
    const double tau = filter_params[4 * nkpoints + batch];
    const double inverse_radius = filter_params[5 * nkpoints + batch];
    const double sigma_new = 1.0 / (tau - sigma);
    filter_params[nkpoints + batch] =
        2.0 * sigma_new * inverse_radius;
    filter_params[2 * nkpoints + batch] = sigma * sigma_new;
    filter_params[3 * nkpoints + batch] = sigma_new;
}

__global__ void density_real_kernel(
    const double* psi, const double* weights, double* density,
    int npoints, int nstates, int nkpoints, double scale) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= npoints * nkpoints) return;
    const int batch = linear / npoints;
    const int point = linear - batch * npoints;
    const double* batch_psi = psi + batch * npoints * nstates;
    const double* batch_weights = weights + batch * nstates;
    double value = 0.0;
    for (int state = 0; state < nstates; ++state) {
        const double wave = batch_psi[point + state * npoints];
        value += batch_weights[state] * wave * wave;
    }
    density[linear] = scale * value;
}

__global__ void density_complex_kernel(
    const cuDoubleComplex* psi, const double* weights, double* density,
    int npoints, int nstates, int nkpoints, double scale) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= npoints * nkpoints) return;
    const int batch = linear / npoints;
    const int point = linear - batch * npoints;
    const cuDoubleComplex* batch_psi = psi + batch * npoints * nstates;
    const double* batch_weights = weights + batch * nstates;
    double value = 0.0;
    for (int state = 0; state < nstates; ++state) {
        const cuDoubleComplex wave = batch_psi[point + state * npoints];
        value += batch_weights[state] *
                 (cuCreal(wave) * cuCreal(wave) +
                  cuCimag(wave) * cuCimag(wave));
    }
    density[linear] = scale * value;
}

__global__ void density_noncollinear_kernel(
    const cuDoubleComplex* psi, const double* weights, double* density,
    int npoints, int nstates, int nkpoints, double scale) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= npoints * nkpoints) return;
    const int batch = linear / npoints;
    const int point = linear - batch * npoints;
    const cuDoubleComplex* batch_psi =
        psi + batch * (2 * npoints) * nstates;
    const double* batch_weights = weights + batch * nstates;
    double* batch_density = density + batch * 4 * npoints;
    double charge = 0.0;
    double magnetization_x = 0.0;
    double magnetization_y = 0.0;
    double magnetization_z = 0.0;
    for (int state = 0; state < nstates; ++state) {
        const cuDoubleComplex* state_psi =
            batch_psi + state * (2 * npoints);
        const cuDoubleComplex up = state_psi[point];
        const cuDoubleComplex down = state_psi[npoints + point];
        const double up_norm = cuCreal(up) * cuCreal(up) +
                               cuCimag(up) * cuCimag(up);
        const double down_norm = cuCreal(down) * cuCreal(down) +
                                 cuCimag(down) * cuCimag(down);
        const cuDoubleComplex coupling = cuCmul(cuConj(up), down);
        const double weight = batch_weights[state];
        charge += weight * (up_norm + down_norm);
        magnetization_x += weight * 2.0 * cuCreal(coupling);
        magnetization_y += weight * 2.0 * cuCimag(coupling);
        magnetization_z += weight * (up_norm - down_norm);
    }
    batch_density[point] = scale * charge;
    batch_density[npoints + point] = scale * magnetization_x;
    batch_density[2 * npoints + point] = scale * magnetization_y;
    batch_density[3 * npoints + point] = scale * magnetization_z;
}

int synchronize(Context* context) noexcept {
    if (context == nullptr || context->stream == nullptr) return kInvalidState;
    return cudaStreamSynchronize(context->stream) == cudaSuccess
               ? kSuccess
               : kCudaRuntimeError;
}

int compute_blocks_count(int npoints, int* blocks) noexcept {
    if (blocks == nullptr || npoints <= 0) return kInvalidArgument;
    constexpr int threads = 256;
    *blocks = npoints / threads + ((npoints % threads) != 0 ? 1 : 0);
    return *blocks > 0 ? kSuccess : kInvalidArgument;
}

bool subspace_is_ready(const ResidentSubspace* subspace) noexcept {
    return subspace != nullptr && subspace->context != nullptr &&
           subspace->context->configured && subspace->context->fd_uploaded &&
           subspace->rows > 0 && subspace->nstates > 0 &&
           subspace->nkpoints > 0;
}

int launch_subspace_hamiltonian(
    ResidentSubspace* subspace, const void* input, void* output) noexcept {
    if (!subspace_is_ready(subspace) || !subspace->potential_ready ||
        input == nullptr || output == nullptr) {
        return kInvalidState;
    }
    Context* context = subspace->context;
    if (context->fft == nullptr) return kInvalidState;
    return spectral_hamiltonian(context, subspace->kind, subspace->d_veff,
                                subspace->d_kvec, input, output,
                                subspace->nstates, subspace->nkpoints);
}

int ensure_solver_workspace(
    ResidentSubspace* subspace, int* lwork) noexcept {
    if (!subspace_is_ready(subspace) || lwork == nullptr) return kInvalidState;
    Context* context = subspace->context;
    const int n = subspace->nstates;
    cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
    if (subspace->kind == kRealSubspace) {
        solver_status = cusolverDnDsygvd_bufferSize(
            context->cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, n, subspace->d_hhat_real, n,
            subspace->d_shat_real, n, subspace->d_eigenvalues, lwork);
    } else {
        solver_status = cusolverDnZhegvd_bufferSize(
            context->cusolver, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER, n, subspace->d_hhat_complex, n,
            subspace->d_shat_complex, n, subspace->d_eigenvalues, lwork);
    }
    if (solver_status != CUSOLVER_STATUS_SUCCESS || *lwork <= 0) {
        return kCusolverError;
    }

    const std::size_t element_size =
        subspace->kind == kRealSubspace ? sizeof(double)
                                        : sizeof(cuDoubleComplex);
    std::size_t required_bytes = 0;
    if (!checked_product(static_cast<std::size_t>(*lwork), element_size,
                         &required_bytes)) {
        return kAllocationError;
    }
    if (required_bytes <= subspace->solver_work_bytes) return kSuccess;
    if (subspace->d_solver_work != nullptr) {
        if (cudaFree(subspace->d_solver_work) != cudaSuccess) {
            subspace->d_solver_work = nullptr;
            subspace->solver_work_bytes = 0;
            return kCudaRuntimeError;
        }
        subspace->d_solver_work = nullptr;
        subspace->solver_work_bytes = 0;
    }
    if (cudaMalloc(&subspace->d_solver_work, required_bytes) != cudaSuccess) {
        return kAllocationError;
    }
    subspace->solver_work_bytes = required_bytes;
    return kSuccess;
}

int create_resident_subspace(
    Context* context, SubspaceKind kind, int nstates, int nkpoints,
    const void* host_wavefunctions, void** subspace_handle_out) noexcept {
    if (subspace_handle_out == nullptr) return kInvalidArgument;
    *subspace_handle_out = nullptr;
    if (context == nullptr || !context->configured || !context->fd_uploaded) {
        return kInvalidState;
    }
    if (host_wavefunctions == nullptr || nstates <= 0 || nkpoints <= 0 ||
        nstates > context->local_states) {
        return kInvalidArgument;
    }
    if (kind == kNoncollinearSubspace && context->npoints > INT_MAX / 2) {
        return kInvalidArgument;
    }
    const int rows = kind == kNoncollinearSubspace
                         ? 2 * context->npoints
                         : context->npoints;
    std::size_t wave_elements = 0;
    if (!checked_product(static_cast<std::size_t>(rows),
                         static_cast<std::size_t>(nstates),
                         &wave_elements) ||
        !checked_product(wave_elements, static_cast<std::size_t>(nkpoints),
                         &wave_elements) ||
        wave_elements > static_cast<std::size_t>(INT_MAX)) {
        return kInvalidArgument;
    }
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }

    ResidentSubspace* subspace = new (std::nothrow) ResidentSubspace;
    if (subspace == nullptr) return kAllocationError;
    subspace->context = context;
    subspace->kind = kind;
    subspace->rows = rows;
    subspace->nstates = nstates;
    subspace->nkpoints = nkpoints;
    subspace->potential_components =
        kind == kNoncollinearSubspace ? 4 : 1;
    int status = allocate_subspace_buffers(subspace);
    if (status != kSuccess) {
        release_subspace(subspace);
        return status;
    }
    int lwork = 0;
    status = ensure_solver_workspace(subspace, &lwork);
    if (status != kSuccess) {
        release_subspace(subspace);
        return status;
    }

    const std::size_t wave_bytes = wave_elements *
        (kind == kRealSubspace ? sizeof(double) : sizeof(cuDoubleComplex));
    void* device_wavefunctions = kind == kRealSubspace
        ? static_cast<void*>(subspace->d_psi_real)
        : static_cast<void*>(subspace->d_psi_complex);
    if (cudaMemcpyAsync(device_wavefunctions, host_wavefunctions, wave_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        synchronize(context) != kSuccess) {
        release_subspace(subspace);
        return kCudaRuntimeError;
    }
    ++context->subspace_count;
    subspace->registered = true;
    *subspace_handle_out = subspace;
    return kSuccess;
}

int update_subspace_hamiltonian(
    ResidentSubspace* subspace, const double* kvec,
    const double* veff) noexcept {
    if (!subspace_is_ready(subspace)) return kInvalidState;
    if (veff == nullptr ||
        (subspace->kind != kRealSubspace && kvec == nullptr)) {
        return kInvalidArgument;
    }
    Context* context = subspace->context;
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    subspace->potential_ready = false;
    const std::size_t potential_bytes =
        static_cast<std::size_t>(context->npoints) *
        static_cast<std::size_t>(subspace->potential_components) *
        sizeof(double);
    if (cudaMemcpyAsync(subspace->d_veff, veff, potential_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    if (subspace->kind != kRealSubspace &&
        cudaMemcpyAsync(
            subspace->d_kvec, kvec,
            3 * static_cast<std::size_t>(subspace->nkpoints) * sizeof(double),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const int status = synchronize(context);
    if (status == kSuccess) subspace->potential_ready = true;
    return status;
}

int run_chebyshev_batch(
    ResidentSubspace* subspace, int degree, const double* lower_filter_bounds,
    const double* upper_bounds,
    const double* lower_spectrum_bounds) noexcept {
    if (!subspace_is_ready(subspace) || !subspace->potential_ready) {
        return kInvalidState;
    }
    if (degree < 2 || lower_filter_bounds == nullptr ||
        upper_bounds == nullptr || lower_spectrum_bounds == nullptr) {
        return kInvalidArgument;
    }
    Context* context = subspace->context;
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const int nkpoints = subspace->nkpoints;
    std::unique_ptr<double[]> filter_params(
        new (std::nothrow) double[6 * static_cast<std::size_t>(nkpoints)]);
    if (!filter_params) return kAllocationError;
    for (int batch = 0; batch < nkpoints; ++batch) {
        const double lower_filter_bound = lower_filter_bounds[batch];
        const double upper_bound = upper_bounds[batch];
        const double lower_spectrum_bound = lower_spectrum_bounds[batch];
        if (!std::isfinite(lower_filter_bound) ||
            !std::isfinite(upper_bound) ||
            !std::isfinite(lower_spectrum_bound) ||
            upper_bound <= lower_filter_bound) {
            return kInvalidArgument;
        }
        const double radius = 0.5 * (upper_bound - lower_filter_bound);
        const double center = 0.5 * (upper_bound + lower_filter_bound);
        const double denominator = center - lower_spectrum_bound;
        if (std::abs(radius) <= std::numeric_limits<double>::epsilon() ||
            std::abs(denominator) <= std::numeric_limits<double>::epsilon()) {
            return kInvalidArgument;
        }
        const double sigma = radius / denominator;
        if (!std::isfinite(sigma) ||
            std::abs(sigma) <= std::numeric_limits<double>::epsilon()) {
            return kInvalidArgument;
        }
        filter_params[batch] = center;
        filter_params[nkpoints + batch] = sigma / radius;
        filter_params[2 * nkpoints + batch] = 0.0;
        filter_params[3 * nkpoints + batch] = sigma;
        filter_params[4 * nkpoints + batch] = 2.0 / sigma;
        filter_params[5 * nkpoints + batch] = 1.0 / radius;
    }
    const std::size_t filter_bytes =
        6 * static_cast<std::size_t>(nkpoints) * sizeof(double);
    if (cudaMemcpyAsync(subspace->d_filter_params, filter_params.get(),
                        filter_bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess ||
        synchronize(context) != kSuccess) {
        return kCudaRuntimeError;
    }
    if (context->fft == nullptr) return kInvalidState;
    return run_chebyshev_fft(subspace, degree);

}

int project_subspace(ResidentSubspace* subspace) noexcept {
    if (!subspace_is_ready(subspace) || !subspace->potential_ready) {
        return kInvalidState;
    }
    Context* context = subspace->context;
    const long long wave_stride =
        static_cast<long long>(subspace->rows) * subspace->nstates;
    const long long projected_stride =
        static_cast<long long>(subspace->nstates) * subspace->nstates;
    int status = kSuccess;
    if (subspace->kind == kRealSubspace) {
        status = launch_subspace_hamiltonian(
            subspace, subspace->d_psi_real, subspace->d_aux_real);
        if (status != kSuccess) return status;
        const double one = 1.0;
        const double zero = 0.0;
        if (cublasDgemmStridedBatched(
                context->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                subspace->nstates, subspace->nstates, subspace->rows,
                &one, subspace->d_psi_real, subspace->rows, wave_stride,
                subspace->d_aux_real, subspace->rows, wave_stride, &zero,
                subspace->d_hhat_real, subspace->nstates, projected_stride,
                subspace->nkpoints) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasDgemmStridedBatched(
                context->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                subspace->nstates, subspace->nstates, subspace->rows,
                &one, subspace->d_psi_real, subspace->rows, wave_stride,
                subspace->d_psi_real, subspace->rows, wave_stride, &zero,
                subspace->d_shat_real, subspace->nstates, projected_stride,
                subspace->nkpoints) !=
                CUBLAS_STATUS_SUCCESS) {
            return kCublasError;
        }
    } else {
        status = launch_subspace_hamiltonian(
            subspace, subspace->d_psi_complex, subspace->d_aux_complex);
        if (status != kSuccess) return status;
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
        if (cublasZgemmStridedBatched(
                context->cublas, CUBLAS_OP_C, CUBLAS_OP_N,
                subspace->nstates, subspace->nstates, subspace->rows,
                &one, subspace->d_psi_complex, subspace->rows, wave_stride,
                subspace->d_aux_complex, subspace->rows, wave_stride, &zero,
                subspace->d_hhat_complex, subspace->nstates,
                projected_stride, subspace->nkpoints) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasZgemmStridedBatched(
                context->cublas, CUBLAS_OP_C, CUBLAS_OP_N,
                subspace->nstates, subspace->nstates, subspace->rows,
                &one, subspace->d_psi_complex, subspace->rows, wave_stride,
                subspace->d_psi_complex, subspace->rows, wave_stride, &zero,
                subspace->d_shat_complex, subspace->nstates,
                projected_stride, subspace->nkpoints) !=
                CUBLAS_STATUS_SUCCESS) {
            return kCublasError;
        }
    }
    return kSuccess;
}

int solve_and_rotate_subspace(
    ResidentSubspace* subspace, double* eigenvalues) noexcept {
    if (!subspace_is_ready(subspace) || eigenvalues == nullptr) {
        return kInvalidArgument;
    }
    Context* context = subspace->context;
    int lwork = 0;
    int status = ensure_solver_workspace(subspace, &lwork);
    if (status != kSuccess) return status;
    status = project_subspace(subspace);
    if (status != kSuccess) return status;

    const std::size_t projected_stride =
        static_cast<std::size_t>(subspace->nstates) * subspace->nstates;
    for (int batch = 0; batch < subspace->nkpoints; ++batch) {
        cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
        double* eigenvalues_batch =
            subspace->d_eigenvalues + batch * subspace->nstates;
        if (subspace->kind == kRealSubspace) {
            solver_status = cusolverDnDsygvd(
                context->cusolver, CUSOLVER_EIG_TYPE_1,
                CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                subspace->nstates,
                subspace->d_hhat_real + batch * projected_stride,
                subspace->nstates,
                subspace->d_shat_real + batch * projected_stride,
                subspace->nstates, eigenvalues_batch,
                static_cast<double*>(subspace->d_solver_work), lwork,
                subspace->d_info);
        } else {
            solver_status = cusolverDnZhegvd(
                context->cusolver, CUSOLVER_EIG_TYPE_1,
                CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                subspace->nstates,
                subspace->d_hhat_complex + batch * projected_stride,
                subspace->nstates,
                subspace->d_shat_complex + batch * projected_stride,
                subspace->nstates, eigenvalues_batch,
                static_cast<cuDoubleComplex*>(subspace->d_solver_work),
                lwork, subspace->d_info);
        }
        if (solver_status != CUSOLVER_STATUS_SUCCESS) {
            return kCusolverError;
        }

        int info = 0;
        if (cudaMemcpyAsync(&info, subspace->d_info, sizeof(int),
                            cudaMemcpyDeviceToHost, context->stream) !=
                cudaSuccess ||
            synchronize(context) != kSuccess) {
            return kCudaRuntimeError;
        }
        if (info != 0) return kCusolverError;
    }

    const long long wave_stride =
        static_cast<long long>(subspace->rows) * subspace->nstates;
    const long long matrix_stride =
        static_cast<long long>(subspace->nstates) * subspace->nstates;
    if (subspace->kind == kRealSubspace) {
        const double one = 1.0;
        const double zero = 0.0;
        if (cublasDgemmStridedBatched(
                context->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                subspace->rows, subspace->nstates, subspace->nstates,
                &one, subspace->d_psi_real, subspace->rows, wave_stride,
                subspace->d_hhat_real, subspace->nstates, matrix_stride,
                &zero, subspace->d_aux_real, subspace->rows, wave_stride,
                subspace->nkpoints) != CUBLAS_STATUS_SUCCESS) {
            return kCublasError;
        }
        double* swap = subspace->d_psi_real;
        subspace->d_psi_real = subspace->d_aux_real;
        subspace->d_aux_real = swap;
    } else {
        const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
        if (cublasZgemmStridedBatched(
                context->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                subspace->rows, subspace->nstates, subspace->nstates,
                &one, subspace->d_psi_complex, subspace->rows, wave_stride,
                subspace->d_hhat_complex, subspace->nstates, matrix_stride,
                &zero, subspace->d_aux_complex, subspace->rows, wave_stride,
                subspace->nkpoints) !=
            CUBLAS_STATUS_SUCCESS) {
            return kCublasError;
        }
        cuDoubleComplex* swap = subspace->d_psi_complex;
        subspace->d_psi_complex = subspace->d_aux_complex;
        subspace->d_aux_complex = swap;
    }
    if (cudaMemcpyAsync(
            eigenvalues, subspace->d_eigenvalues,
            static_cast<std::size_t>(subspace->nstates) *
                subspace->nkpoints * sizeof(double),
            cudaMemcpyDeviceToHost, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return synchronize(context);
}

}  // namespace

extern "C" int invks_cuda_device_count(int* count) noexcept {
    if (count == nullptr) return kInvalidArgument;
    *count = 0;
    const cudaError_t error = cudaGetDeviceCount(count);
    if (error == cudaErrorNoDevice) return kNoDevice;
    if (error != cudaSuccess) return kCudaRuntimeError;
    if (*count == 0) return kNoDevice;
    return kSuccess;
}

extern "C" int invks_cuda_memory_info(
    void* context_handle, std::size_t* free_bytes, std::size_t* total_bytes) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return kInvalidArgument;
    }
    *free_bytes = 0;
    *total_bytes = 0;
    Context* context = static_cast<Context*>(context_handle);
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess
               ? kSuccess
               : kCudaRuntimeError;
}

extern "C" int invks_cuda_create(int device_id, void** context_handle_out) noexcept {
    if (context_handle_out == nullptr) return kInvalidArgument;
    *context_handle_out = nullptr;
    int count = 0;
    const int count_status = invks_cuda_device_count(&count);
    if (count_status != kSuccess) return count_status;
    if (device_id < 0 || device_id >= count) return kInvalidArgument;
    if (cudaSetDevice(device_id) != cudaSuccess) return kCudaRuntimeError;
    Context* context = new (std::nothrow) Context;
    if (context == nullptr) return kAllocationError;
    context->device_id = device_id;
    if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) {
        release_context(context);
        return kCudaRuntimeError;
    }
    if (cublasCreate(&context->cublas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(context->cublas, context->stream) != CUBLAS_STATUS_SUCCESS) {
        release_context(context);
        return kCublasError;
    }
    if (cusolverDnCreate(&context->cusolver) != CUSOLVER_STATUS_SUCCESS ||
        cusolverDnSetStream(context->cusolver, context->stream) != CUSOLVER_STATUS_SUCCESS) {
        release_context(context);
        return kCusolverError;
    }
    *context_handle_out = context;
    return kSuccess;
}

extern "C" int invks_cuda_setup(void* context_handle, int nx, int ny, int nz,
                                  int local_states, int fd_half_width) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    if (nx <= 0 || ny <= 0 || nz <= 0 || local_states <= 0 || fd_half_width <= 0) {
        return kInvalidArgument;
    }
    if (fd_half_width > (INT_MAX - 1) / 2) return kInvalidArgument;
    const std::size_t npoints_size = static_cast<std::size_t>(nx) *
                                     static_cast<std::size_t>(ny) *
                                     static_cast<std::size_t>(nz);
    if (npoints_size == 0 || npoints_size > static_cast<std::size_t>(INT_MAX)) {
        return kInvalidArgument;
    }
    Context* context = static_cast<Context*>(context_handle);
    if (context->subspace_count != 0) return kInvalidState;
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    context->configured = false;
    context->fd_uploaded = false;
    if (free_device_buffers(context) != cudaSuccess) return kCudaRuntimeError;
    context->grid_shape[0] = nx;
    context->grid_shape[1] = ny;
    context->grid_shape[2] = nz;
    context->local_states = local_states;
    context->fd_half_width = fd_half_width;
    context->npoints = static_cast<int>(npoints_size);
    context->nshift = 2 * fd_half_width + 1;
    const int allocation_status = allocate_device_buffers(context);
    if (allocation_status != kSuccess) {
        context->npoints = 0;
        context->nshift = 0;
        return allocation_status;
    }
    context->configured = true;
    return kSuccess;
}

extern "C" int invks_cuda_upload_fd(
    void* context_handle, const double* lapl, int nshift, const double* grad,
    const int* cell_mu, const double* tbmat) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    Context* context = static_cast<Context*>(context_handle);
    if (!context->configured || context->stream == nullptr) return kInvalidState;
    if (lapl == nullptr || grad == nullptr || cell_mu == nullptr || tbmat == nullptr ||
        nshift != context->nshift) return kInvalidArgument;
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    context->fd_uploaded = false;
    std::size_t count = 0;
    std::size_t bytes = 0;
    if (!checked_product(static_cast<std::size_t>(nshift), 6, &count) ||
        !checked_product(count, sizeof(double), &bytes) ||
        cudaMemcpyAsync(context->d_lapl, lapl, bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess) return kCudaRuntimeError;
    if (!checked_product(static_cast<std::size_t>(nshift), 3, &count) ||
        !checked_product(count, sizeof(double), &bytes) ||
        cudaMemcpyAsync(context->d_grad, grad, bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess) return kCudaRuntimeError;
    if (cudaMemcpyAsync(context->d_cell_mu, cell_mu, 9 * sizeof(int),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->d_tbmat, tbmat, 9 * sizeof(double),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const int status = synchronize(context);
    if (status == kSuccess) context->fd_uploaded = true;
    return status;
}

extern "C" int invks_cuda_subspace_create_real(
    void* context_handle, const double* wavefunctions, int nstates,
    void** subspace) noexcept {
    return create_resident_subspace(
        static_cast<Context*>(context_handle), kRealSubspace, nstates, 1,
        wavefunctions, subspace);
}

extern "C" int invks_cuda_subspace_create_complex(
    void* context_handle, const cuDoubleComplex* wavefunctions, int nstates,
    void** subspace) noexcept {
    return create_resident_subspace(
        static_cast<Context*>(context_handle), kComplexSubspace, nstates, 1,
        wavefunctions, subspace);
}

extern "C" int invks_cuda_subspace_create_noncollinear(
    void* context_handle, const cuDoubleComplex* wavefunctions, int nstates,
    void** subspace) noexcept {
    return create_resident_subspace(
        static_cast<Context*>(context_handle), kNoncollinearSubspace, nstates, 1,
        wavefunctions, subspace);
}

extern "C" int invks_cuda_subspace_create_real_batch(
    void* context_handle, const double* wavefunctions, int nstates, int nkpoints,
    void** subspace) noexcept {
    return create_resident_subspace(
        static_cast<Context*>(context_handle), kRealSubspace, nstates, nkpoints,
        wavefunctions, subspace);
}

extern "C" int invks_cuda_subspace_create_complex_batch(
    void* context_handle, const cuDoubleComplex* wavefunctions, int nstates,
    int nkpoints, void** subspace) noexcept {
    return create_resident_subspace(
        static_cast<Context*>(context_handle), kComplexSubspace, nstates, nkpoints,
        wavefunctions, subspace);
}

extern "C" int invks_cuda_subspace_create_noncollinear_batch(
    void* context_handle, const cuDoubleComplex* wavefunctions, int nstates,
    int nkpoints, void** subspace) noexcept {
    return create_resident_subspace(
        static_cast<Context*>(context_handle), kNoncollinearSubspace, nstates,
        nkpoints, wavefunctions, subspace);
}

extern "C" int invks_cuda_subspace_update_hamiltonian_real(
    void* subspace_handle, const double* veff) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (subspace == nullptr || subspace->kind != kRealSubspace) {
        return kInvalidArgument;
    }
    return update_subspace_hamiltonian(subspace, nullptr, veff);
}

extern "C" int invks_cuda_subspace_update_hamiltonian_complex(
    void* subspace_handle, const double* kvec, const double* veff) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (subspace == nullptr || subspace->kind != kComplexSubspace) {
        return kInvalidArgument;
    }
    return update_subspace_hamiltonian(subspace, kvec, veff);
}

extern "C" int invks_cuda_subspace_update_hamiltonian_noncollinear(
    void* subspace_handle, const double* kvec, const double* veff) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (subspace == nullptr || subspace->kind != kNoncollinearSubspace) {
        return kInvalidArgument;
    }
    return update_subspace_hamiltonian(subspace, kvec, veff);
}

extern "C" int invks_cuda_subspace_update_hamiltonian_complex_batch(
    void* subspace_handle, const double* kvecs, const double* veff) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (subspace == nullptr || subspace->kind != kComplexSubspace) {
        return kInvalidArgument;
    }
    return update_subspace_hamiltonian(subspace, kvecs, veff);
}

extern "C" int invks_cuda_subspace_update_hamiltonian_noncollinear_batch(
    void* subspace_handle, const double* kvecs, const double* veff) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (subspace == nullptr || subspace->kind != kNoncollinearSubspace) {
        return kInvalidArgument;
    }
    return update_subspace_hamiltonian(subspace, kvecs, veff);
}

extern "C" int invks_cuda_subspace_estupb_real_batch(
    void* subspace_handle, const double* host_vectors, int lanczos_steps,
    double* alpha, double* beta) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (!subspace_is_ready(subspace) || !subspace->potential_ready) {
        return kInvalidState;
    }
    if (subspace->kind != kRealSubspace || host_vectors == nullptr ||
        alpha == nullptr || beta == nullptr || lanczos_steps <= 0 ||
        lanczos_steps > 10) {
        return kInvalidArgument;
    }
    if (subspace->context->fft == nullptr) return kInvalidState;
    return lanczos_fft_real(subspace, host_vectors, lanczos_steps, alpha, beta);

}

extern "C" int invks_cuda_subspace_estupb_complex_batch(
    void* subspace_handle, const cuDoubleComplex* host_vectors, int lanczos_steps,
    double* alpha, double* beta) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (!subspace_is_ready(subspace) || !subspace->potential_ready) {
        return kInvalidState;
    }
    if (subspace->kind != kComplexSubspace || host_vectors == nullptr ||
        alpha == nullptr || beta == nullptr || lanczos_steps <= 0 ||
        lanczos_steps > 10) {
        return kInvalidArgument;
    }
    if (subspace->context->fft == nullptr) return kInvalidState;
    return lanczos_fft_complex(subspace, host_vectors, lanczos_steps, alpha, beta);

}

extern "C" int invks_cuda_subspace_chebyshev(
    void* subspace_handle, int degree, double lower_filter_bound,
    double upper_bound, double lower_spectrum_bound) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (subspace == nullptr || subspace->nkpoints != 1) {
        return kInvalidArgument;
    }
    return run_chebyshev_batch(
        subspace, degree, &lower_filter_bound, &upper_bound,
        &lower_spectrum_bound);
}

extern "C" int invks_cuda_subspace_chebyshev_batch(
    void* subspace_handle, int degree, const double* lower_filter_bounds,
    const double* upper_bounds,
    const double* lower_spectrum_bounds) noexcept {
    return run_chebyshev_batch(
        static_cast<ResidentSubspace*>(subspace_handle), degree,
        lower_filter_bounds, upper_bounds, lower_spectrum_bounds);
}

extern "C" int invks_cuda_subspace_rayleigh_ritz(
    void* subspace_handle, double* eigenvalues) noexcept {
    return solve_and_rotate_subspace(
        static_cast<ResidentSubspace*>(subspace_handle), eigenvalues);
}

extern "C" int invks_cuda_subspace_rayleigh_ritz_batch(
    void* subspace_handle, double* eigenvalues) noexcept {
    return solve_and_rotate_subspace(
        static_cast<ResidentSubspace*>(subspace_handle), eigenvalues);
}

extern "C" int invks_cuda_subspace_density(
    void* subspace_handle, const double* weights, double scale,
    double* density) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (!subspace_is_ready(subspace)) return kInvalidState;
    if (weights == nullptr || density == nullptr || !std::isfinite(scale)) {
        return kInvalidArgument;
    }
    Context* context = subspace->context;
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    if (cudaMemcpyAsync(
            subspace->d_weights, weights,
            static_cast<std::size_t>(subspace->nstates) *
                subspace->nkpoints * sizeof(double),
            cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    int blocks = 0;
    if (compute_blocks_count(context->npoints * subspace->nkpoints, &blocks) !=
        kSuccess) {
        return kInvalidArgument;
    }
    if (subspace->kind == kRealSubspace) {
        density_real_kernel<<<blocks, 256, 0, context->stream>>>(
            subspace->d_psi_real, subspace->d_weights,
            subspace->d_density, context->npoints, subspace->nstates,
            subspace->nkpoints, scale);
    } else if (subspace->kind == kComplexSubspace) {
        density_complex_kernel<<<blocks, 256, 0, context->stream>>>(
            subspace->d_psi_complex, subspace->d_weights,
            subspace->d_density, context->npoints, subspace->nstates,
            subspace->nkpoints, scale);
    } else {
        density_noncollinear_kernel<<<blocks, 256, 0, context->stream>>>(
            subspace->d_psi_complex, subspace->d_weights,
            subspace->d_density, context->npoints, subspace->nstates,
            subspace->nkpoints, scale);
    }
    if (cudaGetLastError() != cudaSuccess) return kCudaRuntimeError;
    const std::size_t density_bytes =
        static_cast<std::size_t>(context->npoints) *
        static_cast<std::size_t>(subspace->potential_components) *
        static_cast<std::size_t>(subspace->nkpoints) *
        sizeof(double);
    if (cudaMemcpyAsync(density, subspace->d_density, density_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return synchronize(context);
}

extern "C" int invks_cuda_subspace_download(
    void* subspace_handle, void* wavefunctions) noexcept {
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(subspace_handle);
    if (!subspace_is_ready(subspace)) return kInvalidState;
    if (wavefunctions == nullptr) return kInvalidArgument;
    Context* context = subspace->context;
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const std::size_t wave_elements =
        static_cast<std::size_t>(subspace->rows) *
        static_cast<std::size_t>(subspace->nstates) *
        static_cast<std::size_t>(subspace->nkpoints);
    const std::size_t wave_bytes = wave_elements *
        (subspace->kind == kRealSubspace
             ? sizeof(double)
             : sizeof(cuDoubleComplex));
    const void* device_wavefunctions = subspace->kind == kRealSubspace
        ? static_cast<const void*>(subspace->d_psi_real)
        : static_cast<const void*>(subspace->d_psi_complex);
    if (cudaMemcpyAsync(wavefunctions, device_wavefunctions, wave_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return synchronize(context);
}

extern "C" int invks_cuda_subspace_destroy(void** subspace_handle_io) noexcept {
    if (subspace_handle_io == nullptr || *subspace_handle_io == nullptr) return kSuccess;
    ResidentSubspace* subspace = static_cast<ResidentSubspace*>(*subspace_handle_io);
    Context* context = subspace->context;
    if (context == nullptr) return kInvalidState;
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const int sync_status = synchronize(context);
    const cudaError_t free_status = free_subspace_buffers(subspace);
    if (subspace->registered && context->subspace_count > 0) {
        --context->subspace_count;
    }
    delete subspace;
    *subspace_handle_io = nullptr;
    if (sync_status != kSuccess || free_status != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return kSuccess;
}

extern "C" int invks_cuda_hvec_real(
    void* context_handle, const double* veff, const double* vec, double* hvec,
    int npoints) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    Context* context = static_cast<Context*>(context_handle);
    if (!context->configured || !context->fd_uploaded ||
        context->stream == nullptr) return kInvalidState;
    if (veff == nullptr || vec == nullptr || hvec == nullptr ||
        npoints != context->npoints) return kInvalidArgument;
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    const std::size_t bytes = static_cast<std::size_t>(npoints) * sizeof(double);
    if (cudaMemcpyAsync(context->d_veff, veff, bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->d_vec_real, vec, bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess) return kCudaRuntimeError;
    const int status = spectral_hamiltonian(context, kRealSubspace,
        context->d_veff, nullptr, context->d_vec_real,
        context->d_hvec_real, 1, 1);
    if (status != kSuccess) return status;
    if (cudaMemcpyAsync(hvec, context->d_hvec_real, bytes, cudaMemcpyDeviceToHost,
                        context->stream) != cudaSuccess) return kCudaRuntimeError;
    return synchronize(context);
}

extern "C" int invks_cuda_hvec_complex(
    void* context_handle, const double* kvec, const double* veff,
    const cuDoubleComplex* vec, cuDoubleComplex* hvec, int npoints) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    Context* context = static_cast<Context*>(context_handle);
    if (!context->configured || !context->fd_uploaded ||
        context->stream == nullptr) return kInvalidState;
    if (kvec == nullptr || veff == nullptr || vec == nullptr || hvec == nullptr ||
        npoints != context->npoints) return kInvalidArgument;
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    const std::size_t real_bytes = static_cast<std::size_t>(npoints) * sizeof(double);
    const std::size_t complex_bytes = static_cast<std::size_t>(npoints) *
                                      sizeof(cuDoubleComplex);
    if (cudaMemcpyAsync(context->d_veff, veff, real_bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->d_vec_complex, vec, complex_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->d_kvec, kvec, 3 * sizeof(double),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const int status = spectral_hamiltonian(context, kComplexSubspace,
        context->d_veff, context->d_kvec, context->d_vec_complex,
        context->d_hvec_complex, 1, 1);
    if (status != kSuccess) return status;
    if (cudaMemcpyAsync(hvec, context->d_hvec_complex, complex_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return synchronize(context);
}

extern "C" int invks_cuda_hvec_noncollinear(
    void* context_handle, const double* kvec, const double* veff,
    const cuDoubleComplex* vec, cuDoubleComplex* hvec, int npoints) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    Context* context = static_cast<Context*>(context_handle);
    if (!context->configured || !context->fd_uploaded ||
        context->stream == nullptr) return kInvalidState;
    if (kvec == nullptr || veff == nullptr || vec == nullptr || hvec == nullptr ||
        npoints != context->npoints) return kInvalidArgument;
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    const std::size_t real_bytes = static_cast<std::size_t>(npoints) * 4 *
                                   sizeof(double);
    const std::size_t complex_bytes = static_cast<std::size_t>(npoints) * 2 *
                                      sizeof(cuDoubleComplex);
    if (cudaMemcpyAsync(context->d_veff, veff, real_bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->d_vec_complex, vec, complex_bytes,
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess ||
        cudaMemcpyAsync(context->d_kvec, kvec, 3 * sizeof(double),
                        cudaMemcpyHostToDevice, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    const int status = spectral_hamiltonian(context, kNoncollinearSubspace,
        context->d_veff, context->d_kvec, context->d_vec_complex,
        context->d_hvec_complex, 1, 1);
    if (status != kSuccess) return status;
    if (cudaMemcpyAsync(hvec, context->d_hvec_complex, complex_bytes,
                        cudaMemcpyDeviceToHost, context->stream) != cudaSuccess) {
        return kCudaRuntimeError;
    }
    return synchronize(context);
}

// New Fortran bridge: allocate npoints * size_per_point bytes on this context's GPU.
// The caller owns device_out and must release it with invks_cuda_array_free.
extern "C" int invks_cuda_array_alloc(void* context_handle, std::size_t npoints,
                                       std::size_t size_per_point,
                                       void** device_out) noexcept {
    if (context_handle == nullptr || device_out == nullptr || npoints == 0 ||
        size_per_point == 0) return kInvalidArgument;
    *device_out = nullptr;
    std::size_t bytes = 0;
    if (!checked_product(npoints, size_per_point, &bytes)) return kInvalidArgument;
    Context* context = static_cast<Context*>(context_handle);
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    return cudaMalloc(device_out, bytes) == cudaSuccess ? kSuccess : kAllocationError;
}

// New Fortran bridge: allocate a device array and queue host-to-device copy.
// Subsequent FFT work on context->stream sees the completed copy in stream order.
extern "C" int invks_cuda_array_upload(void* context_handle, const void* host,
                                        std::size_t npoints,
                                        std::size_t size_per_point,
                                        void** device_out) noexcept {
    if (host == nullptr) return kInvalidArgument;
    const int status = invks_cuda_array_alloc(context_handle, npoints, size_per_point,
                                               device_out);
    if (status != kSuccess) return status;
    Context* context = static_cast<Context*>(context_handle);
    const std::size_t bytes = npoints * size_per_point;
    if (cudaMemcpyAsync(*device_out, host, bytes, cudaMemcpyHostToDevice,
                        context->stream) != cudaSuccess) {
        cudaFree(*device_out);
        *device_out = nullptr;
        return kCudaRuntimeError;
    }
    return kSuccess;
}

// New Fortran bridge: queue device-to-host copy and synchronize before returning.
// The host output is ready when this function reports success.
extern "C" int invks_cuda_array_download(void* context_handle, const void* device,
                                          void* host, std::size_t npoints,
                                          std::size_t size_per_point) noexcept {
    if (context_handle == nullptr || device == nullptr || host == nullptr ||
        npoints == 0 || size_per_point == 0) return kInvalidArgument;
    std::size_t bytes = 0;
    if (!checked_product(npoints, size_per_point, &bytes)) return kInvalidArgument;
    Context* context = static_cast<Context*>(context_handle);
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    if (cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost,
                        context->stream) != cudaSuccess) return kCudaRuntimeError;
    return synchronize(context);
}

// New Fortran bridge: release an array allocated by alloc/upload and null its handle.
extern "C" int invks_cuda_array_free(void* context_handle, void** device_io) noexcept {
    if (context_handle == nullptr || device_io == nullptr) return kInvalidArgument;
    if (*device_io == nullptr) return kSuccess;
    Context* context = static_cast<Context*>(context_handle);
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    if (cudaFree(*device_io) != cudaSuccess) return kCudaRuntimeError;
    *device_io = nullptr;
    return kSuccess;
}

extern "C" int invks_cuda_synchronize(void* context_handle) noexcept {
    if (context_handle == nullptr) return kInvalidState;
    Context* context = static_cast<Context*>(context_handle);
    if (cudaSetDevice(context->device_id) != cudaSuccess) return kCudaRuntimeError;
    return synchronize(context);
}

extern "C" int invks_cuda_destroy(void** context_handle_io) noexcept {
    if (context_handle_io == nullptr || *context_handle_io == nullptr) return kSuccess;
    Context* context = static_cast<Context*>(*context_handle_io);
    if (context->subspace_count != 0 || context->fft != nullptr) return kInvalidState;
    int status = kSuccess;
    if (cudaSetDevice(context->device_id) != cudaSuccess) {
        status = kCudaRuntimeError;
    } else {
        if (synchronize(context) != kSuccess) status = kCudaRuntimeError;
        if (free_device_buffers(context) != cudaSuccess) status = kCudaRuntimeError;
    }
    if (context->cusolver != nullptr &&
        cusolverDnDestroy(context->cusolver) != CUSOLVER_STATUS_SUCCESS) {
        status = kCusolverError;
    }
    if (context->cublas != nullptr &&
        cublasDestroy(context->cublas) != CUBLAS_STATUS_SUCCESS) {
        status = kCublasError;
    }
    if (context->stream != nullptr && cudaStreamDestroy(context->stream) != cudaSuccess) {
        status = kCudaRuntimeError;
    }
    delete context;
    *context_handle_io = nullptr;
    return status;
}
/////////////////////////////////////////////////////////////////////////////////////////
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
    cuDoubleComplex* d_work_fft{};
    Vector* d_a1{};
    Vector* d_a2{};
    Vector* d_a3{};
    int status = -1;
};
// Compute the G-vectors for a given set of lattice vectors a1, a2, a3
__global__ void getG_vector(const Vector* d_a1, const Vector* d_a2, const Vector* d_a3,
               int nx, int ny, int nz, Vector* d_G){
    if (d_a1 == nullptr || d_a2 == nullptr || d_a3 == nullptr || d_G == nullptr) {
        printf("Error: Null pointer passed to getG_vector kernel.\n");
        return;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny * nz) return;
    double volume = d_a1->x * (d_a2->y * d_a3->z - d_a2->z * d_a3->y) -
                    d_a1->y * (d_a2->x * d_a3->z - d_a2->z * d_a3->x) +
                    d_a1->z * (d_a2->x * d_a3->y - d_a2->y * d_a3->x);
    double factor = 2.0 * M_PI / volume;
    int ix = i % nx;
    int iy = (i / nx) % ny;
    int iz = i / (nx * ny);
    Vector b1,b2,b3;
    b1.x = (d_a2->y * d_a3->z - d_a2->z * d_a3->y) * factor;
    b1.y = (d_a2->z * d_a3->x - d_a2->x * d_a3->z) * factor;
    b1.z = (d_a2->x * d_a3->y - d_a2->y * d_a3->x) * factor;
    b2.x = (d_a3->y * d_a1->z - d_a3->z * d_a1->y) * factor;
    b2.y = (d_a3->z * d_a1->x - d_a3->x * d_a1->z) * factor;
    b2.z = (d_a3->x * d_a1->y - d_a3->y * d_a1->x) * factor;
    b3.x = (d_a1->y * d_a2->z - d_a1->z * d_a2->y) * factor;
    b3.y = (d_a1->z * d_a2->x - d_a1->x * d_a2->z) * factor;
    b3.z = (d_a1->x * d_a2->y - d_a1->y * d_a2->x) * factor;
    const int mx = ix <= (nx - 1) / 2 ? ix : ix - nx;
    const int my = iy <= (ny - 1) / 2 ? iy : iy - ny;
    const int mz = iz <= (nz - 1) / 2 ? iz : iz - nz;
    d_G[i].x = mx*b1.x + my*b2.x + mz*b3.x;
    d_G[i].y = mx*b1.y + my*b2.y + mz*b3.y;
    d_G[i].z = mx*b1.z + my*b2.z + mz*b3.z;
}
__global__ void getG_half(
    const Vector* d_a1, const Vector* d_a2, const Vector* d_a3,
    int nx, int ny, int nz,
    Vector* d_gradG, double* d_lapG2)
{
    const size_t p =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int nxh = nx / 2 + 1;
    const size_t nfreq = static_cast<size_t>(nxh) * ny * nz;
    if (p >= nfreq) return;

    const int ix = static_cast<int>(p % nxh);
    const int iy = static_cast<int>((p / nxh) % ny);
    const int iz = static_cast<int>(
        p / (static_cast<size_t>(nxh) * ny));

    const double volume =
        d_a1->x * (d_a2->y*d_a3->z - d_a2->z*d_a3->y) -
        d_a1->y * (d_a2->x*d_a3->z - d_a2->z*d_a3->x) +
        d_a1->z * (d_a2->x*d_a3->y - d_a2->y*d_a3->x);
    const double factor = 2.0 * M_PI / volume;
    const Vector b1{
        (d_a2->y*d_a3->z - d_a2->z*d_a3->y)*factor,
        (d_a2->z*d_a3->x - d_a2->x*d_a3->z)*factor,
        (d_a2->x*d_a3->y - d_a2->y*d_a3->x)*factor};
    const Vector b2{
        (d_a3->y*d_a1->z - d_a3->z*d_a1->y)*factor,
        (d_a3->z*d_a1->x - d_a3->x*d_a1->z)*factor,
        (d_a3->x*d_a1->y - d_a3->y*d_a1->x)*factor};
    const Vector b3{
        (d_a1->y*d_a2->z - d_a1->z*d_a2->y)*factor,
        (d_a1->z*d_a2->x - d_a1->x*d_a2->z)*factor,
        (d_a1->x*d_a2->y - d_a1->y*d_a2->x)*factor};

    const double mx = static_cast<double>(ix);
    const double my = static_cast<double>(iy <= (ny - 1) / 2 ? iy : iy - ny);
    const double mz = static_cast<double>(iz <= (nz - 1) / 2 ? iz : iz - nz);
    const bool nyqx = nx % 2 == 0 && ix == nx / 2;
    const bool nyqy = ny % 2 == 0 && iy == ny / 2;
    const bool nyqz = nz % 2 == 0 && iz == nz / 2;

    Vector g{0.0, 0.0, 0.0};
    if (!nyqx) {
        g.x += mx*b1.x; g.y += mx*b1.y; g.z += mx*b1.z;
    }
    if (!nyqy) {
        g.x += my*b2.x; g.y += my*b2.y; g.z += my*b2.z;
    }
    if (!nyqz) {
        g.x += mz*b3.x; g.y += mz*b3.y; g.z += mz*b3.z;
    }
    if (d_gradG != nullptr) d_gradG[p] = g;
    if (d_lapG2 != nullptr) {
        double g2 = g.x*g.x + g.y*g.y + g.z*g.z;
        if (nyqx) g2 += mx*mx*(b1.x*b1.x + b1.y*b1.y + b1.z*b1.z);
        if (nyqy) g2 += my*my*(b2.x*b2.x + b2.y*b2.y + b2.z*b2.z);
        if (nyqz) g2 += mz*mz*(b3.x*b3.x + b3.y*b3.y + b3.z*b3.z);
        d_lapG2[p] = g2;
    }
}
__global__ void getG2_full(const Vector* d_G, int npoints, double* d_lapG2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;
    const Vector g = d_G[i];
    d_lapG2[i] = g.x*g.x + g.y*g.y + g.z*g.z;
}
/////////////////////////////////////////////////////////////////////////////////////////
template <int BlockSize>
__device__ double block_sum(double value) noexcept{
    __shared__ double shared[BlockSize];
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int offset = BlockSize / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    return shared[0];
}
__global__ void block_thread_sum(const double* input, const int npoints, double* sum_block){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double value = i<npoints ? input[i]:0.0;
    double sum_part = block_sum<256>(value);
    if(threadIdx.x == 0){
        sum_block[blockIdx.x] = sum_part;
    }
    __syncthreads();
}
__global__ void all_thread_sum(const double* partial,const int num_blocks, double* result){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    for (int j = threadIdx.x; j < num_blocks; j += blockDim.x) {
        value += partial[j];
    }
    double sum = block_sum<256>(value);
    if(i == 0){
        *result = sum;
    }
    __syncthreads();
}
__device__ cuDoubleComplex ComplexMulReal(const cuDoubleComplex a, const double b) noexcept{
    double Real = b * cuCreal(a);
    double Imag = b * cuCimag(a);
    cuDoubleComplex result = make_cuDoubleComplex(Real,Imag);
    return result;
}
__device__ cuDoubleComplex ComplexDeviseReal(const cuDoubleComplex a, const double b) noexcept{
    double Real = cuCreal(a)/b;
    double Imag = cuCimag(a)/b;
    cuDoubleComplex result = make_cuDoubleComplex(Real,Imag);
    return result;
}
__global__ void ComplexArrayDiviseReal(cuDoubleComplex* ComplexArray, const size_t npoints,const double a){
    const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>= npoints) return;
    ComplexArray[i] =  ComplexDeviseReal(ComplexArray[i],a);
}
__global__ void RealArrayDiviseReal(double* RealArray, const size_t npoints,const double a){
    const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>= npoints) return;
    RealArray[i] = RealArray[i]/a;
}
__global__ void RealReal_FGdv_kernel(const double* F, const double* G, const double dv,
                             const int nponts, double* result){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>= nponts) return;
    result[i] = F[i]*G[i]*dv;
}
__global__ void ComplexComplex_FGdv_kernel(const cuDoubleComplex* F, const cuDoubleComplex* G, const double dv,
                             const int nponts, cuDoubleComplex* result){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>= nponts) return;
    result[i] = ComplexMulReal(cuCmul(F[i],G[i]), dv);
}
__global__ void ComplexReal_FGdv_kernel(const cuDoubleComplex* F, const double* G, const double dv,
                             const int nponts, cuDoubleComplex* result){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>= nponts) return;
    result[i] = ComplexMulReal(ComplexMulReal(F[i],G[i]), dv);
}
__global__ void laplacian_fft_kernel(const double* d_lapG2, const cuDoubleComplex* psi_fft,
                                     const int npoints, cuDoubleComplex* laplacian_psi_fft) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= npoints) return;
    const double factor = -d_lapG2[i];
    const cuDoubleComplex v = psi_fft[i];
    laplacian_psi_fft[i] =make_cuDoubleComplex(factor*v.x, factor*v.y);
}
__global__ void gradient_fft_kernel(const Vector* vector, const cuDoubleComplex* psi_fft, const int npoints,
                                    cuDoubleComplex* gradient_psi_fft_x, cuDoubleComplex* gradient_psi_fft_y,
                                    cuDoubleComplex* gradient_psi_fft_z) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>= npoints) return;
    gradient_psi_fft_x[i] = cuCmul(psi_fft[i],make_cuDoubleComplex(0.0,vector[i].x));
    gradient_psi_fft_y[i] = cuCmul(psi_fft[i],make_cuDoubleComplex(0.0,vector[i].y));
    gradient_psi_fft_z[i] = cuCmul(psi_fft[i],make_cuDoubleComplex(0.0,vector[i].z));
}
/////////////////////////////////////////////////////////////////////////////////////////
cufftResult destroy_fft_plan(Fft3dPlan* fft) noexcept{
    if (fft == nullptr) return CUFFT_INVALID_VALUE;

    cufftResult first_error = CUFFT_SUCCESS;

    auto destroy_one = [&](cufftHandle& handle) noexcept {
        if (handle == 0) return;

        const cufftResult status = cufftDestroy(handle);
        if (status == CUFFT_SUCCESS) {
            handle = 0;
        } else if (first_error == CUFFT_SUCCESS) {
            first_error = status;
        }
    };

    destroy_one(fft->Z2Zplan);
    destroy_one(fft->D2Zplan);
    destroy_one(fft->Z2Dplan);

    auto free_one = [&](auto*& pointer) noexcept {
        if (pointer == nullptr) return;
        if (cudaFree(pointer) == cudaSuccess) {
            pointer = nullptr;
        } else if (first_error == CUFFT_SUCCESS) {
            first_error = CUFFT_INTERNAL_ERROR;
        }
    };
    free_one(fft->d_G);
    free_one(fft->d_G_half);
    free_one(fft->d_lapG2_half);
    free_one(fft->d_work_fft);
    free_one(fft->d_a1);
    free_one(fft->d_a2);
    free_one(fft->d_a3);

    fft->status = -1;
    if (fft->Z2Zplan == 0 &&
        fft->D2Zplan == 0 &&
        fft->Z2Dplan == 0 &&
        fft->d_G == nullptr &&
        fft->d_G_half == nullptr &&
        fft->d_lapG2_half == nullptr &&
        fft->d_work_fft == nullptr &&
        fft->d_a1 == nullptr &&
        fft->d_a2 == nullptr &&
        fft->d_a3 == nullptr) {
        fft->nx = fft->ny = fft->nz = 0;
        fft->stream = nullptr;
        fft->context = nullptr;
    }

    return first_error;
}

int create_fft_plan(
    Fft3dPlan* fft,
    int nx, int ny, int nz,
    Context* context) noexcept{
    if (fft == nullptr || nx <= 0 || ny <= 0 || nz <= 0 || context == nullptr || context->stream == nullptr) {
        return 1;
    }
    if (fft->Z2Zplan || fft->D2Zplan || fft->Z2Dplan) {
        return 5;
    }

    fft->nx = nx;
    fft->ny = ny;
    fft->nz = nz;
    fft->stream = context->stream;
    fft->context = context;
    fft->status = -1;

    auto fail = [&](int code) noexcept {
        const cufftResult cleanup = destroy_fft_plan(fft);
        if (cleanup != CUFFT_SUCCESS) {
            std::cerr << "销毁 FFT plan 失败，cuFFT 错误码："
                      << cleanup << '\n';
        }
        return code;
    };

    if (cufftPlan3d(&fft->Z2Zplan, nz, ny, nx, CUFFT_Z2Z)
        != CUFFT_SUCCESS) return fail(2);
    if (cufftSetStream(fft->Z2Zplan, context->stream)
        != CUFFT_SUCCESS) return fail(21);

    if (cufftPlan3d(&fft->Z2Dplan, nz, ny, nx, CUFFT_Z2D)
        != CUFFT_SUCCESS) return fail(3);
    if (cufftSetStream(fft->Z2Dplan, context->stream)
        != CUFFT_SUCCESS) return fail(31);

    if (cufftPlan3d(&fft->D2Zplan, nz, ny, nx, CUFFT_D2Z)
        != CUFFT_SUCCESS) return fail(4);
    if (cufftSetStream(fft->D2Zplan, context->stream)
        != CUFFT_SUCCESS) return fail(41);

    fft->status = 0;
    return 0;
}
int complex_fft_forward(const Fft3dPlan* fft, cuDoubleComplex* d_psi, cuDoubleComplex* d_psi_fft) noexcept{
    if (fft == nullptr || d_psi == nullptr || d_psi_fft == nullptr || fft->status != 0) {
        std::cerr << "complex_fft_forward 执行失败，错误码：" << 1 << '\n';
        return 1;
    };
    if (cufftExecZ2Z(fft->Z2Zplan,
                     reinterpret_cast<cufftDoubleComplex*>(d_psi),
                     reinterpret_cast<cufftDoubleComplex*>(d_psi_fft),
                    CUFFT_FORWARD) == CUFFT_SUCCESS){return 0;}else{return 2;}
}
int real_fft_forward(const Fft3dPlan* fft, double* d_psi, cufftDoubleComplex* d_psi_fft) noexcept{
    if (fft == nullptr || d_psi == nullptr || d_psi_fft == nullptr || fft->status != 0){
        std::cerr << "real_fft_forward 执行失败，错误码：" << 1 << '\n';
        return 1;
    };
    if (cufftExecD2Z(fft->D2Zplan,
                     reinterpret_cast<double*>(d_psi),
                     reinterpret_cast<cufftDoubleComplex*>(d_psi_fft))
                     == CUFFT_SUCCESS){return 0;}else{return 2;}
}

int complex_fft_inverse(const Context* context, const Fft3dPlan* fft, cuDoubleComplex* d_psi_fft, cuDoubleComplex* d_psi) noexcept{
    if (context == nullptr || fft == nullptr ||
        d_psi_fft == nullptr || d_psi == nullptr ||
        fft->status != 0 || fft->stream != context->stream) {
        return -1;
    }
    if (cufftExecZ2Z(fft->Z2Zplan,
                    reinterpret_cast<cufftDoubleComplex*>(d_psi_fft),
                    reinterpret_cast<cufftDoubleComplex*>(d_psi),
                    CUFFT_INVERSE) != CUFFT_SUCCESS) return 1;
    // norm
    const std::size_t size = static_cast<std::size_t>(fft->nx)*
                             static_cast<std::size_t>(fft->ny)*
                             static_cast<std::size_t>(fft->nz);
    ComplexArrayDiviseReal<<<(size + 255) / 256, 256, 0, context->stream>>>(d_psi, size, size);
    if(cudaGetLastError() != cudaSuccess) return 2;
    return 0;
}
int real_fft_inverse(const Context* context,const Fft3dPlan* fft, cuDoubleComplex* d_psi_fft, double* d_psi) noexcept{
    if (context == nullptr || fft == nullptr ||
        d_psi_fft == nullptr || d_psi == nullptr ||
        fft->status != 0 || fft->stream != context->stream) {
        return -1;
    }
    if (cufftExecZ2D(fft->Z2Dplan,
                    reinterpret_cast<cufftDoubleComplex*>(d_psi_fft),
                    reinterpret_cast<double*>(d_psi))
                    != CUFFT_SUCCESS) return 1;
    // norm
    const std::size_t size = static_cast<std::size_t>(fft->nx)*
                             static_cast<std::size_t>(fft->ny)*
                             static_cast<std::size_t>(fft->nz);
    RealArrayDiviseReal<<<(size + 255) / 256, 256, 0, context->stream>>>(d_psi, size, size);
    if(cudaGetLastError() != cudaSuccess) return 2;
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////
int RealReal_IntegralFGdV(const Context* context, const double* d_F, const double* d_G, const double dv,
                          const int npoints, double* d_result) noexcept{
    void* d_IntLocal = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_IntLocal), sizeof(double) * npoints) != cudaSuccess) {
        return 1;
    }
    int blocks = 0;
    if (compute_blocks_count(npoints, &blocks) != kSuccess) return 2;
    RealReal_FGdv_kernel<<<blocks, 256, 0, context->stream>>>(d_F, d_G, dv, npoints, static_cast<double*>(d_IntLocal));
    if (cudaGetLastError() != cudaSuccess) return 3;

    void* d_partial_sum = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_partial_sum), sizeof(double) * blocks) != cudaSuccess) {
        cudaFree(d_IntLocal);
        return 4;
    }
    block_thread_sum<<<blocks, 256, 0, context->stream>>>(static_cast<double*>(d_IntLocal), npoints, static_cast<double*>(d_partial_sum));
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_IntLocal);
        cudaFree(d_partial_sum);
        return 5;
    }
    all_thread_sum<<<1, 256, 0, context->stream>>>(static_cast<double*>(d_partial_sum), blocks, d_result);
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_IntLocal);
        cudaFree(d_partial_sum);
        return 6;
    }
    cudaFree(d_IntLocal);
    cudaFree(d_partial_sum);
    return 0;
}
int ComplexReal_IntegralFGdV(const Context* context, const cuDoubleComplex* d_F, const double* d_G, const double dv,
                             const int npoints, cuDoubleComplex* d_result) noexcept{
    void* d_IntLocal = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_IntLocal), sizeof(cuDoubleComplex) * npoints) != cudaSuccess) {
        return 1;
    }
    int blocks = 0;
    if (compute_blocks_count(npoints, &blocks) != kSuccess) return 2;
    ComplexReal_FGdv_kernel<<<blocks, 256, 0, context->stream>>>(d_F, d_G, dv, npoints, static_cast<cuDoubleComplex*>(d_IntLocal));
    if (cudaGetLastError() != cudaSuccess) return 3;

    void* d_partial_sum = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_partial_sum), sizeof(cuDoubleComplex) * blocks) != cudaSuccess) {
        cudaFree(d_IntLocal);
        return 4;
    }
    block_thread_sum<<<blocks, 256, 0, context->stream>>>(static_cast<double*>(d_IntLocal), npoints, static_cast<double*>(d_partial_sum));
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_IntLocal);
        cudaFree(d_partial_sum);
        return 5;
    }
    all_thread_sum<<<1, 256, 0, context->stream>>>(static_cast<double*>(d_partial_sum), blocks, reinterpret_cast<double*>(d_result));
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_IntLocal);
        cudaFree(d_partial_sum);
        return 6;
    }
    cudaFree(d_IntLocal);
    cudaFree(d_partial_sum);
    return 0;
}
int ComplexComplex_IntegralFGdV(const Context* context, const cuDoubleComplex* d_F, const cuDoubleComplex* d_G, const double dv,
                             const int npoints, cuDoubleComplex* d_result) noexcept{
    void* d_IntLocal = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_IntLocal), sizeof(cuDoubleComplex) * npoints) != cudaSuccess) {
        return 1;
    }
    int blocks = 0;
    if (compute_blocks_count(npoints, &blocks) != kSuccess) return 2;
    ComplexComplex_FGdv_kernel<<<blocks, 256, 0, context->stream>>>(d_F, d_G, dv, npoints, static_cast<cuDoubleComplex*>(d_IntLocal));
    if (cudaGetLastError() != cudaSuccess) return 3;

    void* d_partial_sum = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_partial_sum), sizeof(cuDoubleComplex) * blocks) != cudaSuccess) {
        cudaFree(d_IntLocal);
        return 4;
    }
    block_thread_sum<<<blocks, 256, 0, context->stream>>>(static_cast<double*>(d_IntLocal), npoints, static_cast<double*>(d_partial_sum));
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_IntLocal);
        cudaFree(d_partial_sum);
        return 5;
    }
    all_thread_sum<<<1, 256, 0, context->stream>>>(static_cast<double*>(d_partial_sum), blocks, reinterpret_cast<double*>(d_result));
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_IntLocal);
        cudaFree(d_partial_sum);
        return 6;
    }
    cudaFree(d_IntLocal);
    cudaFree(d_partial_sum);
    return 0;
}
//
int complex_gradient(const Context* context, const Fft3dPlan* fft, cuDoubleComplex* d_psi,
                     cuDoubleComplex* d_grad_psi_x, cuDoubleComplex* d_grad_psi_y,
                     cuDoubleComplex* d_grad_psi_z) noexcept{
    if(context == nullptr || fft == nullptr || fft->status != 0 ||
       fft->stream != context->stream ||
       d_psi == nullptr || d_grad_psi_x == nullptr || d_grad_psi_y == nullptr ||
       d_grad_psi_z == nullptr || fft->d_G == nullptr) {
        std::cerr << "complex_gradient 执行失败，错误码：" << -1 << '\n';
        return -1;
    }
    const int npoints = fft->nx * fft->ny * fft->nz;
    cuDoubleComplex* d_psi_fft = nullptr;
    cuDoubleComplex* d_grad_psi_x_fft = nullptr;
    cuDoubleComplex* d_grad_psi_y_fft = nullptr;
    cuDoubleComplex* d_grad_psi_z_fft = nullptr;
    auto cleanup = [&]() noexcept {
        cudaFree(d_psi_fft);
        cudaFree(d_grad_psi_x_fft);
        cudaFree(d_grad_psi_y_fft);
        cudaFree(d_grad_psi_z_fft);
    };
    const size_t bytes = sizeof(cuDoubleComplex) * static_cast<size_t>(npoints);
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_grad_psi_x_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_grad_psi_y_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_grad_psi_z_fft), bytes) != cudaSuccess) {
        cleanup();
        return 1;
    }
    if (complex_fft_forward(fft, d_psi, d_psi_fft) != 0) { cleanup(); return 2; }
    int blocks = 0;
    if (compute_blocks_count(npoints, &blocks) != kSuccess) { cleanup(); return 2; }
    gradient_fft_kernel<<<blocks, 256, 0, context->stream>>>(
        fft->d_G, d_psi_fft, npoints, d_grad_psi_x_fft, d_grad_psi_y_fft, d_grad_psi_z_fft);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return 3; }
    const int x_status = complex_fft_inverse(context, fft, d_grad_psi_x_fft, d_grad_psi_x);
    const int y_status = x_status == 0 ? complex_fft_inverse(context, fft, d_grad_psi_y_fft, d_grad_psi_y) : 0;
    const int z_status = x_status == 0 && y_status == 0 ?
        complex_fft_inverse(context, fft, d_grad_psi_z_fft, d_grad_psi_z) : 0;
    cleanup();
    if (x_status != 0) return 4;
    if (y_status != 0) return 5;
    return z_status == 0 ? 0 : 6;
}
int real_gradient(const Context* context, const Fft3dPlan* fft, double* d_psi,
                  double* d_grad_psi_x, double* d_grad_psi_y,
                  double* d_grad_psi_z) noexcept{
    if(context == nullptr || fft == nullptr || fft->status != 0 ||
       fft->stream != context->stream ||
       d_psi == nullptr || d_grad_psi_x == nullptr || d_grad_psi_y == nullptr ||
       d_grad_psi_z == nullptr || fft->d_G_half == nullptr) {
        std::cerr << "real_gradient 执行失败，错误码：" << -1 << '\n';
        return -1;
    }
    const int nfreq = (fft->nx / 2 + 1) * fft->ny * fft->nz;
    cuDoubleComplex* d_psi_fft = nullptr;
    cuDoubleComplex* d_grad_psi_x_fft = nullptr;
    cuDoubleComplex* d_grad_psi_y_fft = nullptr;
    cuDoubleComplex* d_grad_psi_z_fft = nullptr;
    auto cleanup = [&]() noexcept {
        cudaFree(d_psi_fft);
        cudaFree(d_grad_psi_x_fft);
        cudaFree(d_grad_psi_y_fft);
        cudaFree(d_grad_psi_z_fft);
    };
    const size_t bytes = sizeof(cuDoubleComplex) * static_cast<size_t>(nfreq);
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_grad_psi_x_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_grad_psi_y_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_grad_psi_z_fft), bytes) != cudaSuccess) {
        cleanup();
        return 1;
    }
    int blocks = 0;
    if (compute_blocks_count(nfreq, &blocks) != kSuccess) { cleanup(); return 2; }
    if (real_fft_forward(fft, d_psi, d_psi_fft) != 0) { cleanup(); return 2; }
    gradient_fft_kernel<<<blocks, 256, 0, context->stream>>>(
        fft->d_G_half, d_psi_fft, nfreq,
        d_grad_psi_x_fft, d_grad_psi_y_fft, d_grad_psi_z_fft);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return 3; }
    const int x_status = real_fft_inverse(context, fft, d_grad_psi_x_fft, d_grad_psi_x);
    const int y_status = x_status == 0 ? real_fft_inverse(context, fft, d_grad_psi_y_fft, d_grad_psi_y) : 0;
    const int z_status = x_status == 0 && y_status == 0 ?
        real_fft_inverse(context, fft, d_grad_psi_z_fft, d_grad_psi_z) : 0;
    cleanup();
    if (x_status != 0) return 4;
    if (y_status != 0) return 5;
    return z_status == 0 ? 0 : 6;
}
//
int complex_laplacian(const Context* context, const Fft3dPlan* fft,
                      cuDoubleComplex* d_psi, cuDoubleComplex* d_lapl_psi) noexcept{
    if(context == nullptr || fft == nullptr || fft->status != 0 ||
       fft->stream != context->stream || d_psi == nullptr ||
       d_lapl_psi == nullptr || fft->d_G == nullptr) {
        std::cerr << "complex_laplacian 执行失败，错误码：" << -1 << '\n';
        return -1;
    }
    const int npoints = fft->nx * fft->ny * fft->nz;
    cuDoubleComplex* d_psi_fft = nullptr;
    cuDoubleComplex* d_lapl_psi_fft = nullptr;
    double* d_lapG2 = nullptr;
    auto cleanup = [&]() noexcept {
        cudaFree(d_psi_fft);
        cudaFree(d_lapl_psi_fft);
        cudaFree(d_lapG2);
    };
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), sizeof(cuDoubleComplex) * npoints) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_lapl_psi_fft), sizeof(cuDoubleComplex) * npoints) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_lapG2), sizeof(double) * npoints) != cudaSuccess) {
        cleanup();
        return 11;
    }
    int blocks = 0;
    if (compute_blocks_count(npoints, &blocks) != kSuccess) { cleanup(); return 3; }
    getG2_full<<<blocks, 256, 0, context->stream>>>(fft->d_G, npoints, d_lapG2);
    if (cudaGetLastError() != cudaSuccess ||
        complex_fft_forward(fft, d_psi, d_psi_fft) != 0) { cleanup(); return 2; }
    laplacian_fft_kernel<<<blocks, 256, 0, context->stream>>>(
        d_lapG2, d_psi_fft, npoints, d_lapl_psi_fft);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return 4; }
    const int status = complex_fft_inverse(context, fft, d_lapl_psi_fft, d_lapl_psi);
    cleanup();
    return status == 0 ? 0 : 5;
}
int real_laplacian(const Context* context, const Fft3dPlan* fft,
                   double* d_psi, double* d_lapl_psi) noexcept{
    if(context == nullptr || fft == nullptr || fft->status != 0 ||
       fft->stream != context->stream ||
       d_psi == nullptr || d_lapl_psi == nullptr ||
       fft->d_lapG2_half == nullptr) {
        std::cerr << "real_laplacian 执行失败，错误码：" << -1 << '\n';
        return -1;
    }
    const int nfreq = (fft->nx / 2 + 1) * fft->ny * fft->nz;
    cuDoubleComplex* d_psi_fft = nullptr;
    cuDoubleComplex* d_lapl_psi_fft = nullptr;
    auto cleanup = [&]() noexcept {
        cudaFree(d_psi_fft);
        cudaFree(d_lapl_psi_fft);
    };
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), sizeof(cuDoubleComplex) * nfreq) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_lapl_psi_fft), sizeof(cuDoubleComplex) * nfreq) != cudaSuccess) {
        cleanup();
        return 11;
    }
    int blocks = 0;
    if (compute_blocks_count(nfreq, &blocks) != kSuccess) { cleanup(); return 3; }
    if (real_fft_forward(fft, d_psi, d_psi_fft) != 0) { cleanup(); return 2; }
    laplacian_fft_kernel<<<blocks, 256, 0, context->stream>>>(
        fft->d_lapG2_half, d_psi_fft, nfreq, d_lapl_psi_fft);
    if (cudaGetLastError() != cudaSuccess) { cleanup(); return 4; }
    const int status = real_fft_inverse(context, fft, d_lapl_psi_fft, d_lapl_psi);
    cleanup();
    return status == 0 ? 0 : 5;
}

namespace {

// Added for GPU FFT derivatives: apply |G|^2/2 to a real field's half spectrum.
__global__ void spectral_real_kinetic(cuDoubleComplex* d_work_fft,
                                      const double* g2, int nfreq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nfreq) return;
    const double factor = 0.5 * g2[i];
    d_work_fft[i].x *= factor;
    d_work_fft[i].y *= factor;
}

// Added for GPU FFT derivatives: apply |G+k|^2/2 to a complex spectrum.
__global__ void spectral_complex_kinetic(cuDoubleComplex* d_work_fft,
                                         const Vector* g, const double* kvec,
                                         int npoints) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;
    const double gx = g[i].x + kvec[0];
    const double gy = g[i].y + kvec[1];
    const double gz = g[i].z + kvec[2];
    const double factor = 0.5 * (gx*gx + gy*gy + gz*gz);
    d_work_fft[i].x *= factor;
    d_work_fft[i].y *= factor;
}

// Added for GPU FFT derivatives: normalize inverse FFT and add the real potential.
__global__ void spectral_real_finish(const double* input, const double* veff,
                                     double* output, int npoints) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;
    output[i] = output[i] / npoints + veff[i] * input[i];
}

// Added for GPU FFT derivatives: normalize inverse FFT and add the complex potential.
__global__ void spectral_complex_finish(const cuDoubleComplex* input,
                                        const double* veff,
                                        cuDoubleComplex* output, int npoints) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;
    output[i] = cuCadd(scale_complex(output[i], 1.0 / npoints),
                       scale_complex(input[i], veff[i]));
}

// Added for GPU FFT derivatives: finish the two-spinor Hamiltonian with local coupling.
__global__ void spectral_noncollinear_finish(
    const cuDoubleComplex* input, const double* veff,
    cuDoubleComplex* output, int npoints) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npoints) return;
    const cuDoubleComplex up = input[i];
    const cuDoubleComplex down = input[npoints + i];
    const double v0 = veff[i];
    const double bx = veff[npoints + i];
    const double by = veff[2*npoints + i];
    const double bz = veff[3*npoints + i];
    output[i] = cuCadd(scale_complex(output[i], 1.0 / npoints),
        cuCadd(scale_complex(up, v0 + bz),
               cuCmul(make_cuDoubleComplex(bx, -by), down)));
    output[npoints + i] = cuCadd(
        scale_complex(output[npoints + i], 1.0 / npoints),
        cuCadd(cuCmul(make_cuDoubleComplex(bx, by), up),
               scale_complex(down, v0 - bz)));
}

// Added for GPU FFT derivatives: shared cuFFT Hamiltonian for real, complex and spinor batches.
int spectral_hamiltonian(Context* context, SubspaceKind kind,
                         const double* veff, const double* kvec,
                         const void* input, void* output,
                         int nstates, int nkpoints) noexcept {
    if (context == nullptr || context->fft == nullptr || veff == nullptr ||
        input == nullptr || output == nullptr || nstates <= 0 || nkpoints <= 0)
        return kInvalidState;
    Fft3dPlan* fft = context->fft;
    if (fft->status != 0 || fft->stream != context->stream ||
        (kind != kRealSubspace && kvec == nullptr)) return kInvalidState;
    const int npoints = context->npoints;
    const int nfreq = (fft->nx / 2 + 1) * fft->ny * fft->nz;
    int blocks = 0, half_blocks = 0;
    if (compute_blocks_count(npoints, &blocks) != kSuccess ||
        compute_blocks_count(nfreq, &half_blocks) != kSuccess) return kInvalidArgument;
    const int count = nstates * nkpoints;
    if (kind == kRealSubspace) {
        const auto* psi = static_cast<const double*>(input);
        auto* hpsi = static_cast<double*>(output);
        for (int state = 0; state < count; ++state) {
            const double* x = psi + static_cast<size_t>(state) * npoints;
            double* y = hpsi + static_cast<size_t>(state) * npoints;
            if (cufftExecD2Z(fft->D2Zplan, const_cast<double*>(x),
                reinterpret_cast<cufftDoubleComplex*>(fft->d_work_fft)) != CUFFT_SUCCESS)
                return kCudaRuntimeError;
            spectral_real_kinetic<<<half_blocks, 256, 0, context->stream>>>(
                fft->d_work_fft, fft->d_lapG2_half, nfreq);
            if (cudaGetLastError() != cudaSuccess ||
                cufftExecZ2D(fft->Z2Dplan, reinterpret_cast<cufftDoubleComplex*>(fft->d_work_fft), y)
                    != CUFFT_SUCCESS) return kCudaRuntimeError;
            spectral_real_finish<<<blocks, 256, 0, context->stream>>>(x, veff, y, npoints);
            if (cudaGetLastError() != cudaSuccess) return kCudaRuntimeError;
        }
    } else {
        const auto* psi = static_cast<const cuDoubleComplex*>(input);
        auto* hpsi = static_cast<cuDoubleComplex*>(output);
        const int components = kind == kNoncollinearSubspace ? 2 : 1;
        for (int state = 0; state < count; ++state) {
            const double* state_k = kvec + 3 * (state / nstates);
            const cuDoubleComplex* x = psi + static_cast<size_t>(state) * components * npoints;
            cuDoubleComplex* y = hpsi + static_cast<size_t>(state) * components * npoints;
            for (int spin = 0; spin < components; ++spin) {
                const cuDoubleComplex* xs = x + spin*npoints;
                cuDoubleComplex* ys = y + spin*npoints;
                if (cufftExecZ2Z(fft->Z2Zplan,
                    const_cast<cufftDoubleComplex*>(reinterpret_cast<const cufftDoubleComplex*>(xs)),
                    reinterpret_cast<cufftDoubleComplex*>(fft->d_work_fft),
                    CUFFT_FORWARD) != CUFFT_SUCCESS) return kCudaRuntimeError;
                spectral_complex_kinetic<<<blocks, 256, 0, context->stream>>>(
                    fft->d_work_fft, fft->d_G, state_k, npoints);
                if (cudaGetLastError() != cudaSuccess ||
                    cufftExecZ2Z(fft->Z2Zplan,
                        reinterpret_cast<cufftDoubleComplex*>(fft->d_work_fft),
                        reinterpret_cast<cufftDoubleComplex*>(ys), CUFFT_INVERSE)
                        != CUFFT_SUCCESS) return kCudaRuntimeError;
            }
            if (components == 1) {
                spectral_complex_finish<<<blocks, 256, 0, context->stream>>>(
                    x, veff, y, npoints);
            } else {
                spectral_noncollinear_finish<<<blocks, 256, 0, context->stream>>>(
                    x, veff, y, npoints);
            }
            if (cudaGetLastError() != cudaSuccess) return kCudaRuntimeError;
        }
    }
    return kSuccess;
}

// Added for the FFT Hamiltonian path: real Chebyshev recurrence update.
__global__ void chebyshev_fft_update_real(
    const double* hpsi, const double* input, double* output,
    const double* params, int rows, int nstates, int nkpoints,
    int total, bool recurrence) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int batch = i / (rows * nstates);
    const double center = params[batch];
    const double scale = params[nkpoints + batch];
    const double previous = recurrence
        ? params[2*nkpoints + batch] * output[i] : 0.0;
    output[i] = scale * (hpsi[i] - center * input[i]) - previous;
}

// Added for the FFT Hamiltonian path: complex Chebyshev recurrence update.
__global__ void chebyshev_fft_update_complex(
    const cuDoubleComplex* hpsi, const cuDoubleComplex* input,
    cuDoubleComplex* output, const double* params,
    int rows, int nstates, int nkpoints, int total, bool recurrence) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int batch = i / (rows * nstates);
    const double center = params[batch];
    const double scale = params[nkpoints + batch];
    const cuDoubleComplex filtered = scale_complex(
        cuCsub(hpsi[i], scale_complex(input[i], center)), scale);
    output[i] = recurrence
        ? cuCsub(filtered, scale_complex(output[i], params[2*nkpoints + batch]))
        : filtered;
}

// Added for the FFT Hamiltonian path: repeatedly apply H through cuFFT during filtering.
int run_chebyshev_fft(ResidentSubspace* subspace, int degree) noexcept {
    Context* context = subspace->context;
    const int total = subspace->rows * subspace->nstates * subspace->nkpoints;
    int blocks = 0, param_blocks = 0;
    if (compute_blocks_count(total, &blocks) != kSuccess ||
        compute_blocks_count(subspace->nkpoints, &param_blocks) != kSuccess)
        return kInvalidArgument;
    const bool real = subspace->kind == kRealSubspace;
    void* d_hpsi = nullptr;
    const size_t bytes = static_cast<size_t>(total) *
        (real ? sizeof(double) : sizeof(cuDoubleComplex));
    if (cudaMalloc(&d_hpsi, bytes) != cudaSuccess) return kAllocationError;
    auto finish = [&](int status) noexcept {
        const cudaError_t error = cudaFree(d_hpsi);
        return status == kSuccess && error != cudaSuccess ?
            kCudaRuntimeError : status;
    };
    if (real) {
        double* previous = subspace->d_psi_real;
        double* current = subspace->d_aux_real;
        int status = launch_subspace_hamiltonian(subspace, previous, d_hpsi);
        if (status != kSuccess) return finish(status);
        chebyshev_fft_update_real<<<blocks, 256, 0, context->stream>>>(
            static_cast<double*>(d_hpsi), previous, current,
            subspace->d_filter_params, subspace->rows, subspace->nstates,
            subspace->nkpoints, total, false);
        if (cudaGetLastError() != cudaSuccess) return finish(kCudaRuntimeError);
        for (int iteration = 2; iteration <= degree; ++iteration) {
            update_filter_params_kernel<<<param_blocks, 256, 0, context->stream>>>(
                subspace->d_filter_params, subspace->nkpoints);
            if (cudaGetLastError() != cudaSuccess) return finish(kCudaRuntimeError);
            status = launch_subspace_hamiltonian(subspace, current, d_hpsi);
            if (status != kSuccess) return finish(status);
            chebyshev_fft_update_real<<<blocks, 256, 0, context->stream>>>(
                static_cast<double*>(d_hpsi), current, previous,
                subspace->d_filter_params, subspace->rows, subspace->nstates,
                subspace->nkpoints, total, true);
            if (cudaGetLastError() != cudaSuccess) return finish(kCudaRuntimeError);
            double* swap = previous;
            previous = current;
            current = swap;
        }
        subspace->d_psi_real = current;
        subspace->d_aux_real = previous;
    } else {
        cuDoubleComplex* previous = subspace->d_psi_complex;
        cuDoubleComplex* current = subspace->d_aux_complex;
        int status = launch_subspace_hamiltonian(subspace, previous, d_hpsi);
        if (status != kSuccess) return finish(status);
        chebyshev_fft_update_complex<<<blocks, 256, 0, context->stream>>>(
            static_cast<cuDoubleComplex*>(d_hpsi), previous, current,
            subspace->d_filter_params, subspace->rows, subspace->nstates,
            subspace->nkpoints, total, false);
        if (cudaGetLastError() != cudaSuccess) return finish(kCudaRuntimeError);
        for (int iteration = 2; iteration <= degree; ++iteration) {
            update_filter_params_kernel<<<param_blocks, 256, 0, context->stream>>>(
                subspace->d_filter_params, subspace->nkpoints);
            if (cudaGetLastError() != cudaSuccess) return finish(kCudaRuntimeError);
            status = launch_subspace_hamiltonian(subspace, current, d_hpsi);
            if (status != kSuccess) return finish(status);
            chebyshev_fft_update_complex<<<blocks, 256, 0, context->stream>>>(
                static_cast<cuDoubleComplex*>(d_hpsi), current, previous,
                subspace->d_filter_params, subspace->rows, subspace->nstates,
                subspace->nkpoints, total, true);
            if (cudaGetLastError() != cudaSuccess) return finish(kCudaRuntimeError);
            cuDoubleComplex* swap = previous;
            previous = current;
            current = swap;
        }
        subspace->d_psi_complex = current;
        subspace->d_aux_complex = previous;
    }
    return finish(kSuccess);
}

// Added for the FFT Hamiltonian path: real Lanczos spectral-bound iteration.
int lanczos_fft_real(ResidentSubspace* subspace, const double* vectors,
                     int steps, double* alpha, double* beta) noexcept {
    Context* context = subspace->context;
    const int n = context->npoints;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    double *previous = nullptr, *current = nullptr, *residual = nullptr;
    auto finish = [&](int status) noexcept {
        cudaError_t error = cudaSuccess;
        if (previous && cudaFree(previous) != cudaSuccess) error = cudaErrorUnknown;
        if (current && cudaFree(current) != cudaSuccess) error = cudaErrorUnknown;
        if (residual && cudaFree(residual) != cudaSuccess) error = cudaErrorUnknown;
        return status == kSuccess && error != cudaSuccess ? kCudaRuntimeError : status;
    };
    if (cudaMalloc(reinterpret_cast<void**>(&previous), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&current), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&residual), bytes) != cudaSuccess)
        return finish(kAllocationError);
    for (int ik = 0; ik < subspace->nkpoints; ++ik) {
        if (cudaMemcpyAsync(current, vectors + static_cast<size_t>(ik)*n,
                            bytes, cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
            return finish(kCudaRuntimeError);
        double beta_previous = 0.0;
        for (int step = 0; step < steps; ++step) {
            const int status = spectral_hamiltonian(context, kRealSubspace,
                subspace->d_veff, nullptr, current, residual, 1, 1);
            if (status != kSuccess) return finish(status);
            if (step > 0) {
                const double negbeta = -beta_previous;
                if (cublasDaxpy(context->cublas, n, &negbeta, previous, 1,
                                residual, 1) != CUBLAS_STATUS_SUCCESS)
                    return finish(kCublasError);
            }
            const size_t index = static_cast<size_t>(ik)*steps + step;
            if (cublasDdot(context->cublas, n, current, 1, residual, 1,
                           &alpha[index]) != CUBLAS_STATUS_SUCCESS)
                return finish(kCublasError);
            const double negalpha = -alpha[index];
            if (cublasDaxpy(context->cublas, n, &negalpha, current, 1,
                            residual, 1) != CUBLAS_STATUS_SUCCESS ||
                cublasDnrm2(context->cublas, n, residual, 1,
                             &beta[index]) != CUBLAS_STATUS_SUCCESS)
                return finish(kCublasError);
            if (!std::isfinite(alpha[index]) || !std::isfinite(beta[index]) ||
                (step + 1 < steps && beta[index] <= 1.0e-14))
                return finish(kInvalidState);
            if (step + 1 == steps) break;
            const double inverse_beta = 1.0 / beta[index];
            if (cublasDcopy(context->cublas, n, current, 1, previous, 1) != CUBLAS_STATUS_SUCCESS ||
                cublasDcopy(context->cublas, n, residual, 1, current, 1) != CUBLAS_STATUS_SUCCESS ||
                cublasDscal(context->cublas, n, &inverse_beta, current, 1) != CUBLAS_STATUS_SUCCESS)
                return finish(kCublasError);
            beta_previous = beta[index];
        }
    }
    if (synchronize(context) != kSuccess) return finish(kCudaRuntimeError);
    return finish(kSuccess);
}

// Added for the FFT Hamiltonian path: complex Lanczos spectral-bound iteration.
int lanczos_fft_complex(ResidentSubspace* subspace,
                        const cuDoubleComplex* vectors, int steps,
                        double* alpha, double* beta) noexcept {
    Context* context = subspace->context;
    const int n = context->npoints;
    const size_t bytes = static_cast<size_t>(n) * sizeof(cuDoubleComplex);
    cuDoubleComplex *previous = nullptr, *current = nullptr, *residual = nullptr;
    auto finish = [&](int status) noexcept {
        cudaError_t error = cudaSuccess;
        if (previous && cudaFree(previous) != cudaSuccess) error = cudaErrorUnknown;
        if (current && cudaFree(current) != cudaSuccess) error = cudaErrorUnknown;
        if (residual && cudaFree(residual) != cudaSuccess) error = cudaErrorUnknown;
        return status == kSuccess && error != cudaSuccess ? kCudaRuntimeError : status;
    };
    if (cudaMalloc(reinterpret_cast<void**>(&previous), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&current), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&residual), bytes) != cudaSuccess)
        return finish(kAllocationError);
    for (int ik = 0; ik < subspace->nkpoints; ++ik) {
        if (cudaMemcpyAsync(current, vectors + static_cast<size_t>(ik)*n,
                            bytes, cudaMemcpyHostToDevice, context->stream) != cudaSuccess)
            return finish(kCudaRuntimeError);
        double beta_previous = 0.0;
        for (int step = 0; step < steps; ++step) {
            const int status = spectral_hamiltonian(context, kComplexSubspace,
                subspace->d_veff, subspace->d_kvec + 3*ik,
                current, residual, 1, 1);
            if (status != kSuccess) return finish(status);
            if (step > 0) {
                const cuDoubleComplex negbeta = make_cuDoubleComplex(-beta_previous, 0.0);
                if (cublasZaxpy(context->cublas, n, &negbeta, previous, 1,
                                residual, 1) != CUBLAS_STATUS_SUCCESS)
                    return finish(kCublasError);
            }
            const size_t index = static_cast<size_t>(ik)*steps + step;
            cuDoubleComplex dot{};
            if (cublasZdotc(context->cublas, n, current, 1, residual, 1,
                            &dot) != CUBLAS_STATUS_SUCCESS)
                return finish(kCublasError);
            alpha[index] = cuCreal(dot);
            const cuDoubleComplex negalpha = make_cuDoubleComplex(-alpha[index], 0.0);
            if (cublasZaxpy(context->cublas, n, &negalpha, current, 1,
                            residual, 1) != CUBLAS_STATUS_SUCCESS ||
                cublasDznrm2(context->cublas, n, residual, 1,
                              &beta[index]) != CUBLAS_STATUS_SUCCESS)
                return finish(kCublasError);
            if (!std::isfinite(alpha[index]) || !std::isfinite(beta[index]) ||
                (step + 1 < steps && beta[index] <= 1.0e-14))
                return finish(kInvalidState);
            if (step + 1 == steps) break;
            const cuDoubleComplex inverse_beta = make_cuDoubleComplex(1.0 / beta[index], 0.0);
            if (cublasZcopy(context->cublas, n, current, 1, previous, 1) != CUBLAS_STATUS_SUCCESS ||
                cublasZcopy(context->cublas, n, residual, 1, current, 1) != CUBLAS_STATUS_SUCCESS ||
                cublasZscal(context->cublas, n, &inverse_beta, current, 1) != CUBLAS_STATUS_SUCCESS)
                return finish(kCublasError);
            beta_previous = beta[index];
        }
    }
    if (synchronize(context) != kSuccess) return finish(kCudaRuntimeError);
    return finish(kSuccess);
}
}  // namespace
//
int fft_forward(const Fft3dPlan* fft,
                double* psi, cuDoubleComplex* d_psi_fft) noexcept {
    return real_fft_forward(fft, psi, d_psi_fft);
}

int fft_forward(const Fft3dPlan* fft,
                cuDoubleComplex* psi, cuDoubleComplex* d_psi_fft) noexcept {
    return complex_fft_forward(fft, psi, d_psi_fft);
}

int fft_inverse(const Context* context, const Fft3dPlan* fft,
                cuDoubleComplex* d_input_fft, double* psi) noexcept {
    return real_fft_inverse(context, fft, d_input_fft, psi);
}

int fft_inverse(const Context* context, const Fft3dPlan* fft,
                cuDoubleComplex* d_input_fft, cuDoubleComplex* psi) noexcept {
    return complex_fft_inverse(context, fft, d_input_fft, psi);
}

int fft_laplacian(const Context* context, const Fft3dPlan* fft,
                  double* psi, double* result) noexcept {
    return real_laplacian(context, fft, psi, result);
}

int fft_laplacian(const Context* context, const Fft3dPlan* fft,
                  cuDoubleComplex* psi, cuDoubleComplex* result) noexcept {
    return complex_laplacian(context, fft, psi, result);
}

int fft_gradient(const Context* context, const Fft3dPlan* fft,
                 double* psi, double* result_x,
                 double* result_y, double* result_z) noexcept {
    return real_gradient(context, fft, psi, result_x, result_y, result_z);
}

int fft_gradient(const Context* context, const Fft3dPlan* fft,
                 cuDoubleComplex* psi, cuDoubleComplex* result_x,
                 cuDoubleComplex* result_y, cuDoubleComplex* result_z) noexcept {
    return complex_gradient(context, fft, psi, result_x, result_y, result_z);
}

extern "C" int invks_fft_init(Context* context, Fft3dPlan** fft_out,
                              int nx, int ny, int nz,
                              const double* h_latticeVectors) noexcept {
    if (context == nullptr || fft_out == nullptr || h_latticeVectors == nullptr ||
        context->fft != nullptr)
        return -1;
    *fft_out = nullptr;

    auto* fft = new (std::nothrow) Fft3dPlan{};
    if (fft == nullptr) return -2;
    const int status = create_fft_plan(fft, nx, ny, nz, context);
    if (status != 0) {
        delete fft;
        return status;
    }
    auto cleanup = [&](int code) noexcept {
        destroy_fft_plan(fft);
        delete fft;
        return code;
    };

    std::size_t npoints = 0, nfreq = 0;
    std::size_t full_bytes = 0, half_bytes = 0, half_lap_bytes = 0;
    if (!checked_product(static_cast<std::size_t>(nx), ny, &npoints) ||
        !checked_product(npoints, nz, &npoints) ||
        npoints > INT_MAX ||
        !checked_product(static_cast<std::size_t>(nx / 2 + 1), ny, &nfreq) ||
        !checked_product(nfreq, nz, &nfreq) ||
        !checked_product(npoints, sizeof(Vector), &full_bytes) ||
        !checked_product(nfreq, sizeof(Vector), &half_bytes) ||
        !checked_product(nfreq, sizeof(double), &half_lap_bytes))
        return cleanup(-3);

    if (cudaMalloc(reinterpret_cast<void**>(&fft->d_a1), sizeof(Vector)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_a2), sizeof(Vector)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_a3), sizeof(Vector)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_G), full_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_G_half), half_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_lapG2_half), half_lap_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_work_fft),
                   npoints * sizeof(cuDoubleComplex)) != cudaSuccess)
        return cleanup(-3);

    const Vector h_a1{h_latticeVectors[0], h_latticeVectors[1], h_latticeVectors[2]};
    const Vector h_a2{h_latticeVectors[3], h_latticeVectors[4], h_latticeVectors[5]};
    const Vector h_a3{h_latticeVectors[6], h_latticeVectors[7], h_latticeVectors[8]};
    if (cudaMemcpy(fft->d_a1, &h_a1, sizeof(Vector), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(fft->d_a2, &h_a2, sizeof(Vector), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(fft->d_a3, &h_a3, sizeof(Vector), cudaMemcpyHostToDevice) != cudaSuccess)
        return cleanup(-4);

    int full_blocks = 0, half_blocks = 0;
    if (compute_blocks_count(static_cast<int>(npoints), &full_blocks) != kSuccess ||
        compute_blocks_count(static_cast<int>(nfreq), &half_blocks) != kSuccess)
        return cleanup(-5);
    getG_vector<<<full_blocks, 256, 0, context->stream>>>(
        fft->d_a1, fft->d_a2, fft->d_a3, nx, ny, nz, fft->d_G);
    if (cudaGetLastError() != cudaSuccess) {
        cudaStreamSynchronize(context->stream);
        return cleanup(-5);
    }
    getG_half<<<half_blocks, 256, 0, context->stream>>>(
        fft->d_a1, fft->d_a2, fft->d_a3, nx, ny, nz,
        fft->d_G_half, fft->d_lapG2_half);
    if (cudaGetLastError() != cudaSuccess) {
        cudaStreamSynchronize(context->stream);
        return cleanup(-5);
    }
    if (cudaStreamSynchronize(context->stream) != cudaSuccess) return cleanup(-6);

    *fft_out = fft;
    context->fft = fft;
    return 0;
}
extern "C" int invks_create_fft_plan(Fft3dPlan** fft_out, int nx, int ny, int nz, Context* context) noexcept{
    if (fft_out == nullptr) return -1;
    *fft_out = nullptr;
    auto* fft = new (std::nothrow) Fft3dPlan{};
    if (fft == nullptr) return -2;
    const int status = create_fft_plan(fft, nx, ny, nz, context);
    if (status != 0) {
        delete fft;
        return status;
    }
    *fft_out = fft;
    return status;
}
extern "C" int invks_destroy_fft_plan(Fft3dPlan** fft_io) noexcept{
    if (fft_io == nullptr || *fft_io == nullptr) return CUFFT_INVALID_VALUE;
    Context* context = (*fft_io)->context;
    const cufftResult status = destroy_fft_plan(*fft_io);
    if (status == CUFFT_SUCCESS) {
        if (context != nullptr && context->fft == *fft_io) context->fft = nullptr;
        delete *fft_io;
        *fft_io = nullptr;
    }
    return status;
}
extern "C" int invks_fft_laplacian_real(const Context* context, const Fft3dPlan* fft,
                                         double* d_psi, double* d_lapl_psi) noexcept{
    return real_laplacian(context, fft, d_psi, d_lapl_psi);
}
extern "C" int invks_fft_laplacian_complex(const Context* context, const Fft3dPlan* fft,
                                            cuDoubleComplex* d_psi, cuDoubleComplex* d_lapl_psi) noexcept{
    return complex_laplacian(context, fft, d_psi, d_lapl_psi);
}
extern "C" int invks_fft_gradient_real(const Context* context, const Fft3dPlan* fft, double* d_psi,
                                       double* d_grad_psi_x, double* d_grad_psi_y,
                                       double* d_grad_psi_z) noexcept{
    return real_gradient(context, fft, d_psi, d_grad_psi_x, d_grad_psi_y, d_grad_psi_z);
}
extern "C" int invks_fft_gradient_complex(const Context* context, const Fft3dPlan* fft, cuDoubleComplex* d_psi,
                                          cuDoubleComplex* d_grad_psi_x, cuDoubleComplex* d_grad_psi_y,
                                          cuDoubleComplex* d_grad_psi_z) noexcept{
    return complex_gradient(context, fft, d_psi, d_grad_psi_x, d_grad_psi_y, d_grad_psi_z);
}
int invKS_calculate_Ws(const Context* context,
        double* d_veff, double* d_grad_veff, double* d_lapl_veff, double* d_rho,
        const double* d_rho0, const double* d_eigenvalues, double* d_wks,
        const double etpy,const double dv, const Fft3dPlan* fft,
        const int nx, const int ny, const int nz, int npoints, int n4, double* d_nwsp,
        const Vector* d_G, const Vector* d_a1, const Vector* d_a2, const Vector* d_a3) noexcept{
    if (context == nullptr || d_nwsp == nullptr) return kInvalidArgument;
    if (cudaMemsetAsync(d_nwsp, 0, sizeof(double), context->stream) != cudaSuccess)
        return kCudaRuntimeError;
    if(real_laplacian(context, fft, d_veff, d_lapl_veff) != 0){
        return kCudaRuntimeError;
    }
    return kInvalidState; // Remaining Ws terms have not been implemented.
}
