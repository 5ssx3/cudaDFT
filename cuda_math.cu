#include "cuda_math.hpp"

#include <climits>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <new>
#include <iostream>

// 将项目定义的 Status 转换为说明文本；不直接接收 CUDA/cuFFT 等库的原生错误码。
// 返回静态字符串，无需释放；未识别的状态码返回默认说明。
const char* status_message(int status) noexcept {
    switch (status) {
        case kSuccess: return "Success";
        case kInvalidArgument: return "Null pointer, invalid grid or invalid scalar argument";
        case kInvalidState: return "Plan is not ready, already initialized or uses another context/stream";
        case kSizeOverflow: return "Grid or allocation size exceeds the supported range";
        case kAllocationError: return "Host or device memory allocation failed";
        case kCudaMemcpyError: return "CUDA memory copy failed";
        case kCudaLaunchError: return "CUDA mathematical kernel launch failed";
        case kCudaSyncError: return "CUDA stream execution or synchronization failed";
        case kCudaFreeError: return "CUDA device memory release failed";
        case kFftPlanError: return "cuFFT plan creation failed";
        case kFftStreamError: return "cuFFT stream binding failed";
        case kFftExecuteError: return "cuFFT forward or inverse transform failed";
        case kFftDestroyError: return "cuFFT plan destruction failed";
        case kCudaInitError: return "CUDA device query, selection or runtime initialization failed";
        case kCublasError: return "cuBLAS handle creation, configuration or operation failed";
        case kCusolverError: return "cuSOLVER handle creation, configuration or operation failed";
        default: return "Unknown mathematical-tool status";
    }
}
// upload and download
int upload(double* h_data, void** d_data, int npoints){
    if (d_data == nullptr) return kInvalidArgument;
    *d_data = nullptr;
    if (npoints <= 0 || h_data == nullptr) return kInvalidArgument;
    const std::size_t bytes = static_cast<std::size_t>(npoints)*sizeof(double);
    if (cudaMalloc(d_data, bytes) != cudaSuccess) return kAllocationError;
    if (cudaMemcpy(*d_data, h_data, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (cudaFree(*d_data) == cudaSuccess) *d_data = nullptr;
        return kCudaMemcpyError;
    }
    return kSuccess;
}
int download(double* d_data, double* h_data, int npoints){
    if (npoints <= 0 || h_data == nullptr || d_data == nullptr)
        return kInvalidArgument;
    const std::size_t bytes = static_cast<std::size_t>(npoints)*sizeof(double);
    if (cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return kCudaMemcpyError;
    return kSuccess;
}
// Scalar and small object inputs use values; large host objects use references.
// Numerical buffers use pointers.
// Scalar outputs and newly allocated plan pointers use non-const references.
bool checked_product(std::size_t left, std::size_t right,
                     std::size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}
int grid_blocks(int npoints, int& blocks) noexcept {
    if (npoints <= 0) return kInvalidArgument;
    constexpr int threads = 256;
    blocks = npoints / threads + ((npoints % threads) != 0 ? 1 : 0);
    return blocks > 0 ? kSuccess : kInvalidArgument;
}

// Compute the G-vectors for a given set of lattice vectors a1, a2, a3
__global__ void getG_vector(const Vector a1, const Vector a2, const Vector a3,
               int nx, int ny, int nz, Vector* d_G){
    if (d_G == nullptr) {
        printf("Error: Null pointer passed to getG_vector kernel.\n");
        return;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny * nz) return;
    double volume = a1.x * (a2.y * a3.z - a2.z * a3.y) -
                    a1.y * (a2.x * a3.z - a2.z * a3.x) +
                    a1.z * (a2.x * a3.y - a2.y * a3.x);
    double factor = 2.0 * M_PI / volume;
    int ix = i % nx;
    int iy = (i / nx) % ny;
    int iz = i / (nx * ny);
    Vector b1,b2,b3;
    b1.x = (a2.y * a3.z - a2.z * a3.y) * factor;
    b1.y = (a2.z * a3.x - a2.x * a3.z) * factor;
    b1.z = (a2.x * a3.y - a2.y * a3.x) * factor;
    b2.x = (a3.y * a1.z - a3.z * a1.y) * factor;
    b2.y = (a3.z * a1.x - a3.x * a1.z) * factor;
    b2.z = (a3.x * a1.y - a3.y * a1.x) * factor;
    b3.x = (a1.y * a2.z - a1.z * a2.y) * factor;
    b3.y = (a1.z * a2.x - a1.x * a2.z) * factor;
    b3.z = (a1.x * a2.y - a1.y * a2.x) * factor;
    const int mx = ix <= (nx - 1) / 2 ? ix : ix - nx;
    const int my = iy <= (ny - 1) / 2 ? iy : iy - ny;
    const int mz = iz <= (nz - 1) / 2 ? iz : iz - nz;
    d_G[i].x = mx*b1.x + my*b2.x + mz*b3.x;
    d_G[i].y = mx*b1.y + my*b2.y + mz*b3.y;
    d_G[i].z = mx*b1.z + my*b2.z + mz*b3.z;
}
__global__ void getG_half(
    const Vector a1, const Vector a2, const Vector a3,
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
        a1.x * (a2.y*a3.z - a2.z*a3.y) -
        a1.y * (a2.x*a3.z - a2.z*a3.x) +
        a1.z * (a2.x*a3.y - a2.y*a3.x);
    const double factor = 2.0 * M_PI / volume;
    const Vector b1{
        (a2.y*a3.z - a2.z*a3.y)*factor,
        (a2.z*a3.x - a2.x*a3.z)*factor,
        (a2.x*a3.y - a2.y*a3.x)*factor};
    const Vector b2{
        (a3.y*a1.z - a3.z*a1.y)*factor,
        (a3.z*a1.x - a3.x*a1.z)*factor,
        (a3.x*a1.y - a3.y*a1.x)*factor};
    const Vector b3{
        (a1.y*a2.z - a1.z*a2.y)*factor,
        (a1.z*a2.x - a1.x*a2.z)*factor,
        (a1.x*a2.y - a1.y*a2.x)*factor};

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
__device__ double sum_values(double a, double b) noexcept { return a + b; }
__device__ cuDoubleComplex sum_values(cuDoubleComplex a, cuDoubleComplex b) noexcept {
    return cuCadd(a, b);
}
template <int BlockSize>
__device__ cuDoubleComplex block_sum(cuDoubleComplex value) noexcept {
    const double real = block_sum<BlockSize>(value.x);
    // All threads must finish reading the first reduction before shared reuse.
    __syncthreads();
    const double imag = block_sum<BlockSize>(value.y);
    return make_cuDoubleComplex(real, imag);
}
template <typename T>
__global__ void block_thread_sum(const T* input, const int npoints, T* sum_block){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    T value = i<npoints ? input[i]:T{};
    T sum_part = block_sum<256>(value);
    if(threadIdx.x == 0){
        sum_block[blockIdx.x] = sum_part;
    }
    __syncthreads();
}
template <typename T>
__global__ void all_thread_sum(const T* partial,const int num_blocks, T* result){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    T value{};
    for (int j = threadIdx.x; j < num_blocks; j += blockDim.x) {
        value = sum_values(value, partial[j]);
    }
    T sum = block_sum<256>(value);
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
__global__ void RealArrayMulReal_kernel(double* RealArray, const size_t npoints,const double a){
    const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>= npoints) return;
    RealArray[i] = RealArray[i]*a;
}
__global__ void RealArrayPlusRealArray_kernel(const size_t npoints, double* RealArray1, double* RealArray2){
    const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>= npoints) return;
    RealArray1[i] = RealArray1[i]+RealArray2[i];
}
__global__ void RealArrayMulRealArray_kernel(const size_t npoints, double* RealArray1, double* RealArray2){
    const size_t i = static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>= npoints) return;
    RealArray1[i] = RealArray1[i]*RealArray2[i];
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
    result[i] = ComplexMulReal(cuCmul(cuConj(F[i]),G[i]), dv);
}
__global__ void ComplexReal_FGdv_kernel(const cuDoubleComplex* F, const double* G, const double dv,
                             const int nponts, cuDoubleComplex* result){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>= nponts) return;
    result[i] = ComplexMulReal(ComplexMulReal(F[i],G[i]), dv);
}

///////////////////////////////////////////////////////////////////////////////////////////
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
int destroy_fft_plan(Fft3dPlan& fft) noexcept {
    int first_error = kSuccess;
    if (fft.stream != nullptr && cudaStreamSynchronize(fft.stream) != cudaSuccess)
        first_error = kCudaSyncError;

    auto destroy_one = [&](cufftHandle& handle) noexcept {
        if (handle == 0) return;
        if (cufftDestroy(handle) == CUFFT_SUCCESS) {
            handle = 0;
        } else if (first_error == kSuccess) {
            first_error = kFftDestroyError;
        }
    };
    destroy_one(fft.Z2Zplan);
    destroy_one(fft.D2Zplan);
    destroy_one(fft.Z2Dplan);

    auto free_one = [&](auto*& pointer) noexcept {
        if (pointer == nullptr) return;
        if (cudaFree(pointer) == cudaSuccess) {
            pointer = nullptr;
        } else if (first_error == kSuccess) {
            first_error = kCudaFreeError;
        }
    };
    free_one(fft.d_G);
    free_one(fft.d_G_half);
    free_one(fft.d_lapG2_half);

    fft.status = kInvalidState;
    if (fft.Z2Zplan == 0 && fft.D2Zplan == 0 && fft.Z2Dplan == 0 &&
        fft.d_G == nullptr && fft.d_G_half == nullptr &&
        fft.d_lapG2_half == nullptr) {
        if (fft.context != nullptr && fft.context->fft == &fft)
            fft.context->fft = nullptr;
        fft.nx = fft.ny = fft.nz = 0;
        fft.stream = nullptr;
        fft.context = nullptr;
    }
    return first_error;
}

int create_fft_plan(Fft3dPlan& fft, int nx, int ny, int nz,
                    Context& context) noexcept {
    if (context.stream == nullptr ||
        nx <= 0 || ny <= 0 || nz <= 0)
        return kInvalidArgument;
    std::size_t npoints = 0;
    if (!checked_product(static_cast<std::size_t>(nx), ny, npoints) ||
        !checked_product(npoints, nz, npoints) || npoints > INT_MAX)
        return kSizeOverflow;
    if (fft.Z2Zplan || fft.D2Zplan || fft.Z2Dplan)
        return kInvalidState;

    fft.nx = nx;
    fft.ny = ny;
    fft.nz = nz;
    fft.stream = context.stream;
    fft.context = &context;
    fft.status = kInvalidState;
    auto fail = [&](int code) noexcept {
        const int cleanup = destroy_fft_plan(fft);
        if (cleanup != kSuccess)
            std::cerr << "FFT cleanup: " << status_message(cleanup) << '\n';
        return code;
    };
    if (cufftPlan3d(&fft.Z2Zplan, nz, ny, nx, CUFFT_Z2Z) != CUFFT_SUCCESS)
        return fail(kFftPlanError);
    if (cufftSetStream(fft.Z2Zplan, context.stream) != CUFFT_SUCCESS)
        return fail(kFftStreamError);
    if (cufftPlan3d(&fft.Z2Dplan, nz, ny, nx, CUFFT_Z2D) != CUFFT_SUCCESS)
        return fail(kFftPlanError);
    if (cufftSetStream(fft.Z2Dplan, context.stream) != CUFFT_SUCCESS)
        return fail(kFftStreamError);
    if (cufftPlan3d(&fft.D2Zplan, nz, ny, nx, CUFFT_D2Z) != CUFFT_SUCCESS)
        return fail(kFftPlanError);
    if (cufftSetStream(fft.D2Zplan, context.stream) != CUFFT_SUCCESS)
        return fail(kFftStreamError);
    fft.status = kSuccess;
    return kSuccess;
}

// FFT execution and normalization are queued on fft.stream. Callers must
// synchronize before reading results on the host or reusing host input buffers.
int complex_fft_forward(const Fft3dPlan& fft, cuDoubleComplex* d_psi,
                        cuDoubleComplex* d_psi_fft) noexcept {
    if (d_psi == nullptr || d_psi_fft == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.Z2Zplan == 0) return kInvalidState;
    return cufftExecZ2Z(fft.Z2Zplan, d_psi, d_psi_fft, CUFFT_FORWARD) == CUFFT_SUCCESS
        ? kSuccess : kFftExecuteError;
}
int real_fft_forward(const Fft3dPlan& fft, double* d_psi,
                     cufftDoubleComplex* d_psi_fft) noexcept {
    if (d_psi == nullptr || d_psi_fft == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.D2Zplan == 0) return kInvalidState;
    return cufftExecD2Z(fft.D2Zplan, d_psi, d_psi_fft) == CUFFT_SUCCESS
        ? kSuccess : kFftExecuteError;
}
int complex_fft_inverse(const Context& context, const Fft3dPlan& fft,
                        cuDoubleComplex* d_psi_fft, cuDoubleComplex* d_psi) noexcept {
    if (d_psi_fft == nullptr || d_psi == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.Z2Zplan == 0 ||
        fft.context != &context || fft.stream != context.stream)
        return kInvalidState;
    if (cufftExecZ2Z(fft.Z2Zplan, d_psi_fft, d_psi, CUFFT_INVERSE) != CUFFT_SUCCESS)
        return kFftExecuteError;
    const std::size_t size = static_cast<std::size_t>(fft.nx) * fft.ny * fft.nz;
    ComplexArrayDiviseReal<<<(size + 255) / 256, 256, 0, context.stream>>>(d_psi, size, size);
    return cudaGetLastError() == cudaSuccess ? kSuccess : kCudaLaunchError;
}
int real_fft_inverse(const Context& context, const Fft3dPlan& fft,
                     cuDoubleComplex* d_psi_fft, double* d_psi) noexcept {
    if (d_psi_fft == nullptr || d_psi == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.Z2Dplan == 0 ||
        fft.context != &context || fft.stream != context.stream)
        return kInvalidState;
    if (cufftExecZ2D(fft.Z2Dplan, d_psi_fft, d_psi) != CUFFT_SUCCESS)
        return kFftExecuteError;
    const std::size_t size = static_cast<std::size_t>(fft.nx) * fft.ny * fft.nz;
    RealArrayDiviseReal<<<(size + 255) / 256, 256, 0, context.stream>>>(d_psi, size, size);
    return cudaGetLastError() == cudaSuccess ? kSuccess : kCudaLaunchError;
}
/////////////////////////////////////////////////////////////////////////////////////////
// Reduce real or complex products without reinterpreting interleaved complex
// storage as a real array. Both components of a complex integral are retained.
template <typename T>
int sum_integral(const Context& context, const T* d_products,
                 int npoints, T* d_result) noexcept {
    int blocks = 0;
    const int status = grid_blocks(npoints, blocks);
    if (status != kSuccess) return status;
    T* d_partial = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_partial), sizeof(T) * blocks) != cudaSuccess)
        return kAllocationError;
    auto finish = [&](int code) noexcept {
        const cudaError_t error = cudaFree(d_partial);
        return code == kSuccess && error != cudaSuccess ? kCudaFreeError : code;
    };
    block_thread_sum<<<blocks, 256, 0, context.stream>>>(d_products, npoints, d_partial);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    all_thread_sum<<<1, 256, 0, context.stream>>>(d_partial, blocks, d_result);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    if (cudaStreamSynchronize(context.stream) != cudaSuccess) return finish(kCudaSyncError);
    return finish(kSuccess);
}

int RealReal_IntegralFGdV(const Context& context, const double* d_F,
                         const double* d_G, double dv,
                         int npoints, double* d_result) noexcept {
    if (d_F == nullptr || d_G == nullptr ||
        d_result == nullptr || npoints <= 0 || !std::isfinite(dv))
        return kInvalidArgument;
    double* d_products = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_products),
                   sizeof(double) * static_cast<std::size_t>(npoints)) != cudaSuccess)
        return kAllocationError;
    int blocks = 0;
    grid_blocks(npoints, blocks);
    RealReal_FGdv_kernel<<<blocks, 256, 0, context.stream>>>(d_F, d_G, dv, npoints, d_products);
    const int status = cudaGetLastError() == cudaSuccess
        ? sum_integral(context, d_products, npoints, d_result) : kCudaLaunchError;
    const cudaError_t error = cudaFree(d_products);
    return status == kSuccess && error != cudaSuccess ? kCudaFreeError : status;
}
int ComplexReal_IntegralFGdV(const Context& context, const cuDoubleComplex* d_F,
                            const double* d_G, double dv,
                            int npoints, cuDoubleComplex* d_result) noexcept {
    if (d_F == nullptr || d_G == nullptr ||
        d_result == nullptr || npoints <= 0 || !std::isfinite(dv))
        return kInvalidArgument;
    cuDoubleComplex* d_products = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_products),
                   sizeof(cuDoubleComplex) * static_cast<std::size_t>(npoints)) != cudaSuccess)
        return kAllocationError;
    int blocks = 0;
    grid_blocks(npoints, blocks);
    ComplexReal_FGdv_kernel<<<blocks, 256, 0, context.stream>>>(d_F, d_G, dv, npoints, d_products);
    const int status = cudaGetLastError() == cudaSuccess
        ? sum_integral(context, d_products, npoints, d_result) : kCudaLaunchError;
    const cudaError_t error = cudaFree(d_products);
    return status == kSuccess && error != cudaSuccess ? kCudaFreeError : status;
}
// This is integral(conj(F) * G) dV
int ComplexComplex_IntegralFGdV(const Context& context, const cuDoubleComplex* d_F,
                               const cuDoubleComplex* d_G, double dv,
                               int npoints, cuDoubleComplex* d_result) noexcept {
    if (d_F == nullptr || d_G == nullptr ||
        d_result == nullptr || npoints <= 0 || !std::isfinite(dv))
        return kInvalidArgument;
    cuDoubleComplex* d_products = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_products),
                   sizeof(cuDoubleComplex) * static_cast<std::size_t>(npoints)) != cudaSuccess)
        return kAllocationError;
    int blocks = 0;
    grid_blocks(npoints, blocks);
    ComplexComplex_FGdv_kernel<<<blocks, 256, 0, context.stream>>>(d_F, d_G, dv, npoints, d_products);
    const int status = cudaGetLastError() == cudaSuccess
        ? sum_integral(context, d_products, npoints, d_result) : kCudaLaunchError;
    const cudaError_t error = cudaFree(d_products);
    return status == kSuccess && error != cudaSuccess ? kCudaFreeError : status;
}
// int RealReal_IntegralFGdV(const Context& context, const double* d_F,
//                                const double* d_G, double dv,
//                                int npoints, double* d_result) noexcept {
//     if (d_F == nullptr || d_G == nullptr ||
//         d_result == nullptr || npoints <= 0 || !std::isfinite(dv))
//         return kInvalidArgument;
//     double* d_products = nullptr;
//     if (cudaMalloc(reinterpret_cast<void**>(&d_products),
//                    sizeof(double) * static_cast<std::size_t>(npoints)) != cudaSuccess)
//         return kAllocationError;
//     int blocks = 0;
//     grid_blocks(npoints, blocks);
//     RealReal_FGdv_kernel<<<blocks, 256, 0, context.stream>>>(d_F, d_G, dv, npoints, d_products);
//     const int status = cudaGetLastError() == cudaSuccess
//         ? sum_integral(context, d_products, npoints, d_result) : kCudaLaunchError;
//     const cudaError_t error = cudaFree(d_products);
//     return status == kSuccess && error != cudaSuccess ? kCudaFreeError : status;
// }
int RealArrayMulRealArray(const Context& context, double* d_F,
                               double* d_G, int npoints) noexcept {
    int blocks = 0;
    grid_blocks(npoints, blocks);
    RealArrayMulRealArray_kernel<<<blocks, 256, 0, context.stream>>>(npoints,d_F,d_G);
    return 0;
}
//
int complex_gradient(const Context& context, const Fft3dPlan& fft, cuDoubleComplex* d_psi,
                     cuDoubleComplex* d_grad_psi_x, cuDoubleComplex* d_grad_psi_y,
                     cuDoubleComplex* d_grad_psi_z) noexcept {
    if (d_psi == nullptr ||
        d_grad_psi_x == nullptr || d_grad_psi_y == nullptr || d_grad_psi_z == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.context != &context ||
        fft.stream != context.stream || fft.d_G == nullptr)
        return kInvalidState;
    const int count = fft.nx * fft.ny * fft.nz;
    cuDoubleComplex *d_psi_fft = nullptr, *d_x_fft = nullptr,
                    *d_y_fft = nullptr, *d_z_fft = nullptr;
    auto finish = [&](int code) noexcept {
        if (cudaStreamSynchronize(context.stream) != cudaSuccess && code == kSuccess)
            code = kCudaSyncError;
        for (auto* pointer : {d_psi_fft, d_x_fft, d_y_fft, d_z_fft}) {
            if (cudaFree(pointer) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        }
        return code;
    };
    const std::size_t bytes = sizeof(cuDoubleComplex) * static_cast<std::size_t>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_x_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_y_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_z_fft), bytes) != cudaSuccess)
        return finish(kAllocationError);
    int status = complex_fft_forward(fft, d_psi, d_psi_fft);
    if (status != kSuccess) return finish(status);
    int blocks = 0;
    status = grid_blocks(count, blocks);
    if (status != kSuccess) return finish(status);
    gradient_fft_kernel<<<blocks, 256, 0, context.stream>>>(
        fft.d_G, d_psi_fft, count, d_x_fft, d_y_fft, d_z_fft);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    status = complex_fft_inverse(context, fft, d_x_fft, d_grad_psi_x);
    if (status == kSuccess) status = complex_fft_inverse(context, fft, d_y_fft, d_grad_psi_y);
    if (status == kSuccess) status = complex_fft_inverse(context, fft, d_z_fft, d_grad_psi_z);
    return finish(status);
}
int real_gradient(const Context& context, const Fft3dPlan& fft, double* d_psi,
                     double* d_grad_psi_x, double* d_grad_psi_y,
                     double* d_grad_psi_z) noexcept {
    if (d_psi == nullptr ||
        d_grad_psi_x == nullptr || d_grad_psi_y == nullptr || d_grad_psi_z == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.context != &context ||
        fft.stream != context.stream || fft.d_G_half == nullptr)
        return kInvalidState;
    const int count = (fft.nx / 2 + 1) * fft.ny * fft.nz;
    cuDoubleComplex *d_psi_fft = nullptr, *d_x_fft = nullptr,
                    *d_y_fft = nullptr, *d_z_fft = nullptr;
    auto finish = [&](int code) noexcept {
        if (cudaStreamSynchronize(context.stream) != cudaSuccess && code == kSuccess)
            code = kCudaSyncError;
        for (auto* pointer : {d_psi_fft, d_x_fft, d_y_fft, d_z_fft}) {
            if (cudaFree(pointer) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        }
        return code;
    };
    const std::size_t bytes = sizeof(cuDoubleComplex) * static_cast<std::size_t>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_x_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_y_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_z_fft), bytes) != cudaSuccess)
        return finish(kAllocationError);
    int status = real_fft_forward(fft, d_psi, d_psi_fft);
    if (status != kSuccess) return finish(status);
    int blocks = 0;
    status = grid_blocks(count, blocks);
    if (status != kSuccess) return finish(status);
    gradient_fft_kernel<<<blocks, 256, 0, context.stream>>>(
        fft.d_G_half, d_psi_fft, count, d_x_fft, d_y_fft, d_z_fft);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    status = real_fft_inverse(context, fft, d_x_fft, d_grad_psi_x);
    if (status == kSuccess) status = real_fft_inverse(context, fft, d_y_fft, d_grad_psi_y);
    if (status == kSuccess) status = real_fft_inverse(context, fft, d_z_fft, d_grad_psi_z);
    return finish(status);
}
int complex_laplacian(const Context& context, const Fft3dPlan& fft,
                      cuDoubleComplex* d_psi, cuDoubleComplex* d_lapl_psi) noexcept {
    if (d_psi == nullptr || d_lapl_psi == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.context != &context ||
        fft.stream != context.stream || fft.d_G == nullptr)
        return kInvalidState;
    const int count = fft.nx * fft.ny * fft.nz;
    cuDoubleComplex *d_psi_fft = nullptr, *d_lapl_psi_fft = nullptr;
    double* d_lapG2 = nullptr;
    auto finish = [&](int code) noexcept {
        if (cudaStreamSynchronize(context.stream) != cudaSuccess && code == kSuccess)
            code = kCudaSyncError;
        if (cudaFree(d_psi_fft) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        if (cudaFree(d_lapl_psi_fft) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        if (cudaFree(d_lapG2) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        return code;
    };
    const std::size_t bytes = sizeof(cuDoubleComplex) * static_cast<std::size_t>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_lapl_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_lapG2), sizeof(double) * static_cast<std::size_t>(count)) != cudaSuccess)
        return finish(kAllocationError);
    int blocks = 0;
    int status = grid_blocks(count, blocks);
    if (status != kSuccess) return finish(status);
    getG2_full<<<blocks, 256, 0, context.stream>>>(fft.d_G, count, d_lapG2);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    status = complex_fft_forward(fft, d_psi, d_psi_fft);
    if (status != kSuccess) return finish(status);
    laplacian_fft_kernel<<<blocks, 256, 0, context.stream>>>(
        d_lapG2, d_psi_fft, count, d_lapl_psi_fft);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    status = complex_fft_inverse(context, fft, d_lapl_psi_fft, d_lapl_psi);
    return finish(status);
}
int real_laplacian(const Context& context, const Fft3dPlan& fft,
                      double* d_psi, double* d_lapl_psi) noexcept {
    if (d_psi == nullptr || d_lapl_psi == nullptr)
        return kInvalidArgument;
    if (fft.status != kSuccess || fft.context != &context ||
        fft.stream != context.stream || fft.d_lapG2_half == nullptr)
        return kInvalidState;
    const int count = (fft.nx / 2 + 1) * fft.ny * fft.nz;
    cuDoubleComplex *d_psi_fft = nullptr, *d_lapl_psi_fft = nullptr;
    auto finish = [&](int code) noexcept {
        if (cudaStreamSynchronize(context.stream) != cudaSuccess && code == kSuccess)
            code = kCudaSyncError;
        if (cudaFree(d_psi_fft) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        if (cudaFree(d_lapl_psi_fft) != cudaSuccess && code == kSuccess) code = kCudaFreeError;
        return code;
    };
    const std::size_t bytes = sizeof(cuDoubleComplex) * static_cast<std::size_t>(count);
    if (cudaMalloc(reinterpret_cast<void**>(&d_psi_fft), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_lapl_psi_fft), bytes) != cudaSuccess)
        return finish(kAllocationError);
    int blocks = 0;
    int status = grid_blocks(count, blocks);
    if (status != kSuccess) return finish(status);
    status = real_fft_forward(fft, d_psi, d_psi_fft);
    if (status != kSuccess) return finish(status);
    laplacian_fft_kernel<<<blocks, 256, 0, context.stream>>>(
        fft.d_lapG2_half, d_psi_fft, count, d_lapl_psi_fft);
    if (cudaGetLastError() != cudaSuccess) return finish(kCudaLaunchError);
    status = real_fft_inverse(context, fft, d_lapl_psi_fft, d_lapl_psi);
    return finish(status);
}

//
int fft_forward(const Fft3dPlan& fft,
                double* psi, cuDoubleComplex* d_psi_fft) noexcept {
    return real_fft_forward(fft, psi, d_psi_fft);
}

int fft_forward(const Fft3dPlan& fft,
                cuDoubleComplex* psi, cuDoubleComplex* d_psi_fft) noexcept {
    return complex_fft_forward(fft, psi, d_psi_fft);
}

int fft_inverse(const Context& context, const Fft3dPlan& fft,
                cuDoubleComplex* d_input_fft, double* psi) noexcept {
    return real_fft_inverse(context, fft, d_input_fft, psi);
}

int fft_inverse(const Context& context, const Fft3dPlan& fft,
                cuDoubleComplex* d_input_fft, cuDoubleComplex* psi) noexcept {
    return complex_fft_inverse(context, fft, d_input_fft, psi);
}

int fft_laplacian(const Context& context, const Fft3dPlan& fft,
                  double* psi, double* result) noexcept {
    return real_laplacian(context, fft, psi, result);
}

int fft_laplacian(const Context& context, const Fft3dPlan& fft,
                  cuDoubleComplex* psi, cuDoubleComplex* result) noexcept {
    return complex_laplacian(context, fft, psi, result);
}

int fft_gradient(const Context& context, const Fft3dPlan& fft,
                 double* psi, double* result_x,
                 double* result_y, double* result_z) noexcept {
    return real_gradient(context, fft, psi, result_x, result_y, result_z);
}

int fft_gradient(const Context& context, const Fft3dPlan& fft,
                 cuDoubleComplex* psi, cuDoubleComplex* result_x,
                 cuDoubleComplex* result_y, cuDoubleComplex* result_z) noexcept {
    return complex_gradient(context, fft, psi, result_x, result_y, result_z);
}

int creat_fftPlan(Context& context, Fft3dPlan*& fft_out,
             int nx, int ny, int nz,
             const double* h_latticeVectors) noexcept {
    fft_out = nullptr;
    if (h_latticeVectors == nullptr || context.stream == nullptr ||
        nx <= 0 || ny <= 0 || nz <= 0)
        return kInvalidArgument;
    if (context.fft != nullptr) return kInvalidState;
    for (int i = 0; i < 9; ++i)
        if (!std::isfinite(h_latticeVectors[i])) return kInvalidArgument;
    const Vector h_a1{h_latticeVectors[0], h_latticeVectors[1], h_latticeVectors[2]};
    const Vector h_a2{h_latticeVectors[3], h_latticeVectors[4], h_latticeVectors[5]};
    const Vector h_a3{h_latticeVectors[6], h_latticeVectors[7], h_latticeVectors[8]};
    const double volume =
        h_a1.x * (h_a2.y*h_a3.z - h_a2.z*h_a3.y) -
        h_a1.y * (h_a2.x*h_a3.z - h_a2.z*h_a3.x) +
        h_a1.z * (h_a2.x*h_a3.y - h_a2.y*h_a3.x);
    if (!std::isfinite(volume) || volume == 0.0) return kInvalidArgument;

    std::size_t npoints = 0, nfreq = 0;
    std::size_t full_bytes = 0, half_bytes = 0, half_lap_bytes = 0;
    if (!checked_product(static_cast<std::size_t>(nx), ny, npoints) ||
        !checked_product(npoints, nz, npoints) || npoints > INT_MAX ||
        !checked_product(static_cast<std::size_t>(nx / 2 + 1), ny, nfreq) ||
        !checked_product(nfreq, nz, nfreq) ||
        !checked_product(npoints, sizeof(Vector), full_bytes) ||
        !checked_product(nfreq, sizeof(Vector), half_bytes) ||
        !checked_product(nfreq, sizeof(double), half_lap_bytes))
        return kSizeOverflow;

    auto* fft = new (std::nothrow) Fft3dPlan{};
    if (fft == nullptr) return kAllocationError;
    const int status = create_fft_plan(*fft, nx, ny, nz, context);
    if (status != kSuccess) {
        delete fft;
        return status;
    }
    auto cleanup = [&](int code) noexcept {
        const int error = destroy_fft_plan(*fft);
        if (error != kSuccess)
            std::cerr << "FFT cleanup: " << status_message(error) << '\n';
        delete fft;
        return code;
    };
    if (cudaMalloc(reinterpret_cast<void**>(&fft->d_G), full_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_G_half), half_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&fft->d_lapG2_half), half_lap_bytes) != cudaSuccess)
        return cleanup(kAllocationError);

    int full_blocks = 0, half_blocks = 0;
    grid_blocks(static_cast<int>(npoints), full_blocks);
    grid_blocks(static_cast<int>(nfreq), half_blocks);
    getG_vector<<<full_blocks, 256, 0, context.stream>>>(
        h_a1, h_a2, h_a3, nx, ny, nz, fft->d_G);
    if (cudaGetLastError() != cudaSuccess) return cleanup(kCudaLaunchError);
    getG_half<<<half_blocks, 256, 0, context.stream>>>(
        h_a1, h_a2, h_a3, nx, ny, nz,
        fft->d_G_half, fft->d_lapG2_half);
    if (cudaGetLastError() != cudaSuccess) return cleanup(kCudaLaunchError);
    if (cudaStreamSynchronize(context.stream) != cudaSuccess)
        return cleanup(kCudaSyncError);
    fft_out = fft;
    context.fft = fft;
    return kSuccess;
}
/////////////////////////////////////////////////////////////////////////////
// Finite difference
int creat_finitePlan(finitePlan& finite, int nx, int ny, int nz, int half_width,
                     double hx, double hy, double hz){
    if (half_width<1) return 1;
    if (hx<=0 || hy<=0 || hz<=0) return 2;
    if (nx<=0 || ny<=0 || nz<=0) return 3;
    if (half_width>nx || half_width>ny || half_width>nz) return 4;
    if (finite.d_coeff != nullptr) return 5;
    //
    double* h_coeff = new double[half_width+1]{};
    double ratio = 1.0;

    for (int k = 1; k <=half_width ; ++k) {
        ratio *= double(half_width - k + 1) / (double(half_width) + k);
        double value = 2.0 * ratio / (double(k) * double(k));
        if (k % 2 == 0) value = -value;
        h_coeff[k] = value;
        h_coeff[0] -= 2.0 * value;
    }
    double* d_coeff = nullptr;
    size_t bytes = (size_t(half_width)+1)*sizeof(double);
    cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&d_coeff), bytes);
    if (status != cudaSuccess) return 6;
    status = cudaMemcpy(d_coeff,h_coeff,bytes,cudaMemcpyHostToDevice);
    int g = half_width;
    int Nx = nx+2*g;
    int Ny = ny+2*g;
    int Nz = nz+2*g;
    // double* d_halo_work = nullptr;
    // bytes = (size_t(Nx*Ny*Nz))*sizeof(double);
    // cudaError_t status = cudaMalloc(reinterpret_cast<void**>(&d_halo_work), bytes);
    // if (status != cudaSuccess) return 7;
    //
    finite.nx = nx;
    finite.ny = ny;
    finite.nz = nz;
    finite.npoints = nx*ny*nz;
    finite.half_width = half_width;
    finite.gap.x = hx;
    finite.gap.y = hy;
    finite.gap.z = hz;
    finite.d_coeff = d_coeff;
    // finite->d_halo_work = d_halo_work;
    finite.status = 0;
    //
    delete[] h_coeff;
    return 0;
}
int CUDA_init(Context& context) noexcept {
    // 拒绝覆盖已有资源。
    if (context.stream || context.cublas ||
        context.cusolver || context.fft)
        return kInvalidState;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
        return kCudaInitError;
    constexpr int device_id = 0;

    if (cudaSetDevice(device_id) != cudaSuccess)
        return kCudaInitError;

    // 所有步骤成功后，再交给调用者。
    Context next{};
    next.device_id = device_id;
    next.device_count = count;

    auto fail = [&](int error) noexcept {
        if (next.cusolver)
            cusolverDnDestroy(next.cusolver);
        if (next.cublas)
            cublasDestroy(next.cublas);
        if (next.stream)
            cudaStreamDestroy(next.stream);
        return error;
    };

    if (cudaGetDeviceProperties(&next.device_properties, device_id) != cudaSuccess ||
        cudaDriverGetVersion(&next.driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&next.runtime_version) != cudaSuccess)
        return fail(kCudaInitError);

    if (cudaStreamCreateWithFlags(
            &next.stream, cudaStreamNonBlocking) != cudaSuccess)
        return fail(kCudaInitError);

    if (cublasCreate(&next.cublas) != CUBLAS_STATUS_SUCCESS)
        return fail(kCublasError);

    if (cublasSetStream(next.cublas, next.stream)
            != CUBLAS_STATUS_SUCCESS)
        return fail(kCublasError);

    if (cublasSetPointerMode(
            next.cublas, CUBLAS_POINTER_MODE_HOST)
            != CUBLAS_STATUS_SUCCESS)
        return fail(kCublasError);

    if (cusolverDnCreate(&next.cusolver)
            != CUSOLVER_STATUS_SUCCESS)
        return fail(kCusolverError);

    if (cusolverDnSetStream(next.cusolver, next.stream)
            != CUSOLVER_STATUS_SUCCESS)
        return fail(kCusolverError);

    if (cudaMemGetInfo(&next.free_memory_bytes, &next.total_memory_bytes) != cudaSuccess)
        return fail(kCudaInitError);

    next.status = kSuccess;
    context = next;
    return kSuccess;
}
// __global__ void halo_fill_kernel(const finitePlan finite, double* d_input,double* d_output) {
//     if (d_input == nullptr || d_output == nullptr) return;
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int nx = finite.nx;
//     int ny = finite.ny;
//     int nz = finite.nz;
//     int g = finite.half_width;
//     int Nx = nx+2*g;
//     int Ny = ny+2*g;
//     int Nz = nz+2*g;
//     if (i>= Nx*Ny*Nz) return;
//     int Ix = i%Nx;
//     int Iy = (i/Nx)%Ny;
//     int Iz = (i/(Nx*Ny))%Nz;
//     int ix = Ix - g;
//     int iy = Iy - g;
//     int iz = Iz - g;
//     int parity = 1;

//     if (ix ==0 || iy==0 || Iz==0 || ix==nx+1 || iy==ny+1 || iz==nz+1){
//         d_output[i]=0;
//         return;
//     }
//     if (ix < 0) {
//         ix = -ix;
//         parity = -parity;
//     } else if (ix > nx + 1) {
//         ix = 2 * (nx + 1) - ix;
//         parity = -parity;
//     }
//     if (iy < 0) {
//         iy = -iy;
//         parity = -parity;
//     } else if (iy > ny + 1) {
//         iy = 2 * (ny + 1) - iy;
//         parity = -parity;
//     }
//     if (iz < 0) {
//         iz = -iz;
//         parity = -parity;
//     } else if (iz > nz + 1) {
//         iz = 2 * (nz + 1) - iz;
//         parity = -parity;
//     }
//     const size_t source = (size_t(iz - 1) * ny + (iy - 1)) * nx + (ix - 1);

//     d_output[i] = double(parity) * d_input[source];
// }
// __global__ void finite_laplacian_kernel(const finitePlan finite, double* d_halo_input,double* d_output){
//     if (d_halo_input == nullptr || d_output == nullptr) return;
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     int nx = finite.nx;
//     int ny = finite.ny;
//     int nz = finite.nz;
//     int g = finite.half_width;
//     int Nx = nx+2*g;
//     int Ny = ny+2*g;
//     int Nz = nz+2*g;
//     if (i>= nx*ny*nz) return;
//     int ix = i%nx;
//     int iy = (i/nx)%ny;
//     int iz = (i/(nx*ny))%nz;
//     int Ix = ix+g;
//     int Iy = iy+g;
//     int Iz = iz+g;
//     // 当前内部点在扩展数组中的地址。
//     const size_t stride_y = Nx;
//     const size_t stride_z = Nx * Ny;
//     const size_t center =
//         (size_t(iz) + g) * stride_z +
//         (size_t(iy) + g) * stride_y +
//         (size_t(ix) + g);
//     const double inv_hx2 = 1.0 / (finite.gap.x * finite.gap.x);
//     const double inv_hy2 = 1.0 / (finite.gap.y * finite.gap.y);
//     const double inv_hz2 = 1.0 / (finite.gap.z * finite.gap.z);
//     double lap = finite.d_coeff[0] * d_halo_input[center] * (inv_hx2 + inv_hy2 + inv_hz2);
//     for (int r = 1; r <= g; ++r) {
//         const size_t dx = size_t(r);
//         const size_t dy = size_t(r) * stride_y;
//         const size_t dz = size_t(r) * stride_z;
//         lap += finite.d_coeff[r] * (
//             (d_halo_input[center - dx] +
//              d_halo_input[center + dx]) * inv_hx2 +
//             (d_halo_input[center - dy] +
//              d_halo_input[center + dy]) * inv_hy2 +
//             (d_halo_input[center - dz] +
//              d_halo_input[center + dz]) * inv_hz2);
//     }
//     d_output[i] = lap;
// }
//
__device__ __forceinline__
double read_odd_axis(
    const double* input,
    size_t base,
    size_t stride,
    long long pos,
    int count)
{
    if (pos == -1 || pos == count)
        return 0.0;

    if (pos < -1)
        return -input[base + size_t(-2 - pos) * stride];

    if (pos > count)
        return -input[base + size_t(2LL * count - pos) * stride];

    return input[base + size_t(pos) * stride];
}
__global__ void finite_laplacian_kernel(
    int nx, int ny, int nz, int g, const Vector gap,
    const double* d_coeff, const double* d_input,
    double* d_output)
{
    if (d_input == nullptr ||
        d_output == nullptr ||
        d_coeff == nullptr)
        return;
    const size_t i =
        size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(nx) * ny * nz)
        return;
    const int ix = int(i % nx);
    const int iy = int((i / nx) % ny);
    const int iz = int(i / (size_t(nx) * ny));
    const size_t sy = size_t(nx);
    const size_t sz = size_t(nx) * ny;
    const double inv_hx2 =
        1.0 / (gap.x * gap.x);
    const double inv_hy2 =
        1.0 / (gap.y * gap.y);
    const double inv_hz2 =
        1.0 / (gap.z * gap.z);
    double lap = d_coeff[0] * d_input[i] *
                 (inv_hx2 + inv_hy2 + inv_hz2);
    const bool interior =
        ix >= g && ix < nx - g &&
        iy >= g && iy < ny - g &&
        iz >= g && iz < nz - g;
    if (interior) {
        for (int r = 1; r <= g; ++r) {
            const size_t dx = size_t(r);
            const size_t dy = size_t(r) * sy;
            const size_t dz = size_t(r) * sz;
            lap += d_coeff[r] * (
                (d_input[i - dx] + d_input[i + dx]) * inv_hx2 +
                (d_input[i - dy] + d_input[i + dy]) * inv_hy2 +
                (d_input[i - dz] + d_input[i + dz]) * inv_hz2);
        }
    } else {
        const size_t base_x = i - size_t(ix);
        const size_t base_y = i - size_t(iy) * sy;
        const size_t base_z = i - size_t(iz) * sz;

        const long long x = ix;
        const long long y = iy;
        const long long z = iz;

        for (int r = 1; r <= g; ++r) {
            const double x_pair =
                read_odd_axis(d_input, base_x, 1, x - r, nx) +
                read_odd_axis(d_input, base_x, 1, x + r, nx);

            const double y_pair =
                read_odd_axis(d_input, base_y, sy, y - r, ny) +
                read_odd_axis(d_input, base_y, sy, y + r, ny);

            const double z_pair =
                read_odd_axis(d_input, base_z, sz, z - r, nz) +
                read_odd_axis(d_input, base_z, sz, z + r, nz);

            lap += d_coeff[r] * (
                x_pair * inv_hx2 +
                y_pair * inv_hy2 +
                z_pair * inv_hz2);
        }
    }

    d_output[i] = lap;
}
__global__ void finite_gradient_kernel(
    int nx, int ny, int nz, int g, const Vector gap,
    const double* d_coeff, const double* d_input,
    double* d_gx, double* d_gy, double* d_gz)
{
    if (d_input == nullptr ||
        d_gx == nullptr ||
        d_gy == nullptr ||
        d_gz == nullptr ||
        d_coeff == nullptr)
        return;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(nx) * ny * nz) return;
    const int ix = int(i % nx);
    const int iy = int((i / nx) % ny);
    const int iz = int(i / (size_t(nx) * ny));

    const size_t sy = size_t(nx);
    const size_t sz = size_t(nx) * ny;

    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;

    const bool interior =
        ix >= g && ix < nx - g &&
        iy >= g && iy < ny - g &&
        iz >= g && iz < nz - g;

    if (interior) {
        for (int r = 1; r <= g; ++r) {
            const size_t dx = size_t(r);
            const size_t dy = size_t(r) * sy;
            const size_t dz = size_t(r) * sz;

            // 从标准二阶导数系数转换为一阶导数系数。
            const double a = 0.5 * r * d_coeff[r];

            gx += a * (d_input[i + dx] - d_input[i - dx]);
            gy += a * (d_input[i + dy] - d_input[i - dy]);
            gz += a * (d_input[i + dz] - d_input[i - dz]);
        }
    } else {
        const size_t base_x = i - size_t(ix);
        const size_t base_y = i - size_t(iy) * sy;
        const size_t base_z = i - size_t(iz) * sz;

        const long long x = ix;
        const long long y = iy;
        const long long z = iz;

        for (int r = 1; r <= g; ++r) {
            const double a = 0.5 * r * d_coeff[r];

            gx += a * (
                read_odd_axis(d_input, base_x, 1, x + r, nx) -
                read_odd_axis(d_input, base_x, 1, x - r, nx));

            gy += a * (
                read_odd_axis(d_input, base_y, sy, y + r, ny) -
                read_odd_axis(d_input, base_y, sy, y - r, ny));

            gz += a * (
                read_odd_axis(d_input, base_z, sz, z + r, nz) -
                read_odd_axis(d_input, base_z, sz, z - r, nz));
        }
    }
    d_gx[i] = gx / gap.x;
    d_gy[i] = gy / gap.y;
    d_gz[i] = gz / gap.z;
}
int apply_laplacian_finite(const Context& context, const finitePlan& finite,
                    const double* d_psi, double* d_lapl_psi) noexcept {
    if (finite.d_coeff == nullptr || d_psi == nullptr || d_lapl_psi == nullptr) return kInvalidArgument;
    const int npoints = finite.nx * finite.ny * finite.nz;
    int blocks = 0;
    const int status = grid_blocks(npoints, blocks);
    if (status != kSuccess) return status;
    finite_laplacian_kernel<<<blocks,256,0,context.stream>>>(
        finite.nx, finite.ny, finite.nz, finite.half_width, finite.gap,
        finite.d_coeff, d_psi, d_lapl_psi);
    return cudaGetLastError() == cudaSuccess ? kSuccess : kCudaLaunchError;
}
int apply_gradient_finite(const Context& context, const finitePlan& finite,
                    const double* d_psi, double* d_grad_psi_x, double* d_grad_psi_y, double* d_grad_psi_z) noexcept {
    if (finite.d_coeff == nullptr || d_psi == nullptr || d_grad_psi_x == nullptr 
        || d_grad_psi_y == nullptr || d_grad_psi_z == nullptr) return kInvalidArgument;
    const int npoints = finite.nx * finite.ny * finite.nz;
    int blocks = 0;
    const int status = grid_blocks(npoints, blocks);
    if (status != kSuccess) return status;          
    finite_gradient_kernel<<<blocks,256,0,context.stream>>>(
        finite.nx, finite.ny, finite.nz, finite.half_width, finite.gap,
        finite.d_coeff, d_psi, d_grad_psi_x,d_grad_psi_y,d_grad_psi_z);
    return cudaGetLastError() == cudaSuccess ? kSuccess : kCudaLaunchError;
}

//
    // cublasDgemm 矩阵乘法
    // cublasStatus_t cublasDgemm(
    //     cublasHandle_t handle,
    //     cublasOperation_t transa,
    //     cublasOperation_t transb,
    //     int m,
    //     int n,
    //     int k,
    //     const double* alpha,
    //     const double* A,
    //     int lda,
    //     const double* B,
    //     int ldb,
    //     const double* beta,
    //     double* C,
    //     int ldc);
    //cusolverDnDsygvd 查询工作空间
    //     int lwork = 0;
    // CUSOLVER_CHECK(cusolverDnDsygvd_bufferSize(
    //     cuSolver,
    //     CUSOLVER_EIG_TYPE_1,
    //     CUSOLVER_EIG_MODE_VECTOR,
    //     CUBLAS_FILL_MODE_LOWER,
    //     nstates,
    //     d_hhat, nstates,
    //     d_Shat, nstates,
    //     d_eigenvalues,
    //     &lwork
    // ));
    // cusolverDnDsygvd 对角化求解器
    // cusolverStatus_t cusolverDnDsygvd(
    //     cusolverDnHandle_t handle,
    //     cusolverEigType_t itype,
    //     cusolverEigMode_t jobz,
    //     cublasFillMode_t uplo,
    //     int n,
    //     double* A, int lda,
    //     double* B, int ldb,
    //     double* W,
    //     double* work, int lwork,
    //     int* devInfo);
/////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void finite_Hpsi_kernel(
    int nx, int ny, int nz, int g, const Vector gap,
    const double* d_coeff, const double* d_psi, const double* d_veff,
    double* d_Hpsi)
{
    if (d_psi == nullptr ||
        d_veff == nullptr ||
        d_Hpsi == nullptr ||
        d_coeff == nullptr)
        return;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(nx) * ny * nz)
        return;
    const int ix = int(i % nx);
    const int iy = int((i / nx) % ny);
    const int iz = int(i / (size_t(nx) * ny));
    const size_t sy = size_t(nx);
    const size_t sz = size_t(nx) * ny;
    const double inv_hx2 =
        1.0 / (gap.x * gap.x);
    const double inv_hy2 =
        1.0 / (gap.y * gap.y);
    const double inv_hz2 =
        1.0 / (gap.z * gap.z);
    double lap = d_coeff[0] * d_psi[i] *
                 (inv_hx2 + inv_hy2 + inv_hz2);
    const bool interior =
        ix >= g && ix < nx - g &&
        iy >= g && iy < ny - g &&
        iz >= g && iz < nz - g;
    if (interior) {
        for (int r = 1; r <= g; ++r) {
            const size_t dx = size_t(r);
            const size_t dy = size_t(r) * sy;
            const size_t dz = size_t(r) * sz;
            lap += d_coeff[r] * (
                (d_psi[i - dx] + d_psi[i + dx]) * inv_hx2 +
                (d_psi[i - dy] + d_psi[i + dy]) * inv_hy2 +
                (d_psi[i - dz] + d_psi[i + dz]) * inv_hz2);
        }
    } else {
        const size_t base_x = i - size_t(ix);
        const size_t base_y = i - size_t(iy) * sy;
        const size_t base_z = i - size_t(iz) * sz;

        const long long x = ix;
        const long long y = iy;
        const long long z = iz;

        for (int r = 1; r <= g; ++r) {
            const double x_pair =
                read_odd_axis(d_psi, base_x, 1, x - r, nx) +
                read_odd_axis(d_psi, base_x, 1, x + r, nx);

            const double y_pair =
                read_odd_axis(d_psi, base_y, sy, y - r, ny) +
                read_odd_axis(d_psi, base_y, sy, y + r, ny);

            const double z_pair =
                read_odd_axis(d_psi, base_z, sz, z - r, nz) +
                read_odd_axis(d_psi, base_z, sz, z + r, nz);

            lap += d_coeff[r] * (
                x_pair * inv_hx2 +
                y_pair * inv_hy2 +
                z_pair * inv_hz2);
        }
    }
    d_Hpsi[i] = -0.5 * lap + d_veff[i]*d_psi[i];
}
__device__ __forceinline__
double calculate_c_r(int r, int g, const double* d_coeff)
{
    return r > g ? 0.0 : d_coeff[r];
}
__device__ __forceinline__
double calculate_a(int j, int n, int g, const double* d_coeff)
{
    return d_coeff[0] - calculate_c_r(2 * (j + 1), g, d_coeff) - calculate_c_r(2 * (n - j), g, d_coeff);
}
__global__ void davidson_precondition_kernel(
    int nx, int ny, int nz, int npoints,
    int half_width, double gx, double gy, double gz,
    const double* d_coeff,
    const double* d_veff,       // 有效势
    const double* d_eigenvalue, // 当前 Ritz 值，设备端单个 double
    const double* d_residual,   // 输入：一列残差，长度 npoints
    double* d_correction       // 输出：一列修正方向，长度 npoints
){
    if (d_veff == nullptr ||
        d_eigenvalue == nullptr ||
        d_residual == nullptr ||
        d_correction == nullptr)
        return;
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(nx) * ny * nz) return;
    const int ix = int(i % nx);
    const int iy = int((i / nx) % ny);
    const int iz = int(i / (size_t(nx) * ny));

    double Di = d_veff[i]-0.5*(calculate_a(ix,nx,half_width,d_coeff)/gx/gx+
                               calculate_a(iy,ny,half_width,d_coeff)/gy/gy+
                               calculate_a(iz,nz,half_width,d_coeff)/gz/gz);
    const double delta = 1e-10;
    double denominator = Di - *d_eigenvalue;
    if (fabs(denominator) < delta)
        denominator = copysign(delta, denominator);

    d_correction[i] = -d_residual[i] / denominator;
}
//
int apply_HamiltonOperator(const Context& context, const finitePlan& finite,
                    const double* d_psi, const double* d_veff, double* d_Hpsi) noexcept {
    if (d_psi == nullptr || d_veff == nullptr || d_Hpsi == nullptr)
        return kInvalidArgument;
    if (finite.status != kSuccess || finite.d_coeff == nullptr)
        return kInvalidState;
    if (finite.nx <= 0 || finite.ny <= 0 || finite.nz <= 0 ||
        finite.half_width < 1 || finite.half_width > finite.nx ||
        finite.half_width > finite.ny || finite.half_width > finite.nz ||
        !std::isfinite(finite.gap.x) || finite.gap.x <= 0 ||
        !std::isfinite(finite.gap.y) || finite.gap.y <= 0 ||
        !std::isfinite(finite.gap.z) || finite.gap.z <= 0)
        return kInvalidArgument;
    std::size_t npoints = 0;
    if (!checked_product(static_cast<std::size_t>(finite.nx), finite.ny, npoints) ||
        !checked_product(npoints, finite.nz, npoints) || npoints > INT_MAX)
        return kSizeOverflow;
    int blocks = 0;
    const int status = grid_blocks(static_cast<int>(npoints), blocks);
    if (status != kSuccess) return status;
    finite_Hpsi_kernel<<<blocks,256,0,context.stream>>>(
        finite.nx, finite.ny, finite.nz, finite.half_width, finite.gap,
        finite.d_coeff, d_psi, d_veff,d_Hpsi);
    return cudaGetLastError() == cudaSuccess ? kSuccess : kCudaLaunchError;
}
int apply_davidson_preconditioner(
    const Context& context,
    const finitePlan& finite,
    const double* d_veff,       // 有效势
    const double* d_eigenvalue, // 当前 Ritz 值，设备端单个 double
    const double* d_residual,   // 输入：一列残差，长度 npoints
    double* d_correction       // 输出：一列修正方向，长度 npoints
) noexcept{
    if (d_veff == nullptr || d_eigenvalue == nullptr || d_residual == nullptr)
        return kInvalidArgument;
    if (finite.status != kSuccess || finite.d_coeff == nullptr)
        return kInvalidState;
    if (finite.nx <= 0 || finite.ny <= 0 || finite.nz <= 0 ||
        finite.half_width < 1 || finite.half_width > finite.nx ||
        finite.half_width > finite.ny || finite.half_width > finite.nz ||
        !std::isfinite(finite.gap.x) || finite.gap.x <= 0 ||
        !std::isfinite(finite.gap.y) || finite.gap.y <= 0 ||
        !std::isfinite(finite.gap.z) || finite.gap.z <= 0)
        return kInvalidArgument;
    
    int nx = finite.nx;
    int ny = finite.ny;
    int nz = finite.nz;
    double gx = finite.gap.x;
    double gy = finite.gap.y;
    double gz = finite.gap.z;
    int npoints = finite.npoints;
    int blocks = 0;
    const int status = grid_blocks(static_cast<int>(npoints), blocks);
    if (status != kSuccess) return status;
    davidson_precondition_kernel<<<blocks,256,0,context.stream>>>(
                            nx,ny,nz,npoints,finite.half_width,gx,gy,gz,
                            finite.d_coeff,d_veff,d_eigenvalue,d_residual,d_correction);
    return cudaGetLastError() == cudaSuccess ? kSuccess : kCudaLaunchError;
}
// lbfgs
__global__ void update_q_kernel(
    int n,
    double* q,
    const double* y,
    const double* rho,
    const double* dot)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n) {
        double a = (*rho) * (*dot);
        q[j] = q[j] - a * y[j];
    }
}
__global__ void update_r_kernel(
    int n,
    double* r,
    const double* s,
    const double* a,
    const double* b, 
    const double* rho)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < n) {
        r[j] = r[j] + (*rho) * s[j] * ((*a)-(*b));
    }
}
int update_q(Context& context,
    int n,
    double* q,
    const double* y,
    const double* rho,
    const double* dot) noexcept
{
    int blocks = 0;
    grid_blocks(n,blocks);
    update_q_kernel<<<blocks,256,0,context.stream>>>(
                        n, q, y, rho, dot);
    return kSuccess;
}
int update_r(Context& context,
    int n,
    double* r,
    const double* s,
    const double* a,
    const double* b,
    const double* rho) noexcept
{
    int blocks = 0;
    grid_blocks(n,blocks);
    update_r_kernel<<<blocks,256,0,context.stream>>>(
                        n, r, s, a, b, rho);
    return kSuccess;
}
int RealArrayMulReal(Context& context, int n, 
                     double* d_RealArray, double a) noexcept
{
    int blocks = 0;
    grid_blocks(n,blocks);
    RealArrayMulReal_kernel<<<blocks,256,0,context.stream>>>(d_RealArray,n,a);
    return kSuccess;
}
int RealArrayPlusRealArray(Context& context, int n, 
         double* d_RealArray1, double* d_RealArray2) noexcept
{
    int blocks = 0;
    grid_blocks(n,blocks);
    RealArrayPlusRealArray_kernel<<<blocks,256,0,context.stream>>>(n,d_RealArray1,d_RealArray2);
    return kSuccess;
}
