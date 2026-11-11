#include "smearing.hpp"
#include "cuda_math.hpp"

#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <new>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

const char* occupation_status_message(int status) noexcept
{
    switch (status) {
        case kOccupationNotInitialized: return "Occupation is not initialized or is unusable";
        case kOccupationNonFinite: return "Non-finite eigenvalue or Fermi-level scalar";
        case kOccupationNoConvergence: return "Fermi-level iteration limit reached";
        default: return status_message(status);
    }
}

namespace {
bool check_cuda_status(cudaError_t status,
    const char* expression, const char* file, int line) noexcept
{
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "[CUDA] status=%d (%s: %s), call=%s, at %s:%d\n",
        static_cast<int>(status), cudaGetErrorName(status), cudaGetErrorString(status),
        expression, file, line);
    return false;
}

#define CHECK_CUDA(expression) \
    check_cuda_status((expression), #expression, __FILE__, __LINE__)

int report_occupation_status(int status, const char* stage) noexcept
{
    if (status != kSuccess)
        std::fprintf(stderr, "[Occupation] stage=%s, status=%d (%s)\n",
            stage, status, occupation_status_message(status));
    return status;
}

bool valid_parameters(int nstates, int ne, double temperature,
                      const std::string& occ_mode) noexcept
{
    return nstates > 0 && ne >= 0 && static_cast<long long>(ne) <= 2LL * nstates &&
        (occ_mode == "fixed" ||
         (occ_mode == "fd" && std::isfinite(temperature) && temperature > 0.0));
}

int fixed(Occupation& occupation)
{
    int g = 2;
    int nstates = occupation.nstates;
    int ne = occupation.ne;
    Context& context = occupation.context;
    std::vector<double> h_occ(nstates,0.0);
    for(int i=0;i<nstates;i++){
        if(g*i<ne){
            if(g*i+1==ne) h_occ[i]=1;
            else h_occ[i] = g;
        }else{
            h_occ[i]=0;
        }
    }
    if (!CHECK_CUDA(cudaMemcpyAsync(occupation.d_occupation,h_occ.data(),
            size_t(nstates)*sizeof(double),cudaMemcpyHostToDevice,context.stream)))
        return kCudaMemcpyError;
    if (!CHECK_CUDA(cudaStreamSynchronize(context.stream))) return kCudaSyncError;
    return kSuccess;
}
int fermi_level(Occupation& occupation, double* d_eigenvalues, double& result)
{
    double T = occupation.temperature;
    double sigma = 3.166811563e-6 * T;
    int nstates = occupation.nstates;
    int ne = occupation.ne;
    int g = 2;
    std::vector<double> h_eig(nstates);
    std::vector<double> h_occ(nstates);
    if (!CHECK_CUDA(cudaMemcpyAsync(h_eig.data(),d_eigenvalues,size_t(nstates)*sizeof(double),
                    cudaMemcpyDeviceToHost,occupation.context.stream)))
        return kCudaMemcpyError;
    if (!CHECK_CUDA(cudaStreamSynchronize(occupation.context.stream))) return kCudaSyncError;
    for (double eigenvalue : h_eig)
        if (!std::isfinite(eigenvalue)) return kOccupationNonFinite;
    double Elo = h_eig[0]- 40.0 * sigma;
    double Ehi = h_eig[nstates-1] + 40.0 * sigma;
    if (!std::isfinite(Elo) || !std::isfinite(Ehi)) return kOccupationNonFinite;
    for(int j=0;j<1000;j++){
        double Efermi = (Elo+Ehi)/2;
        if (!std::isfinite(Efermi)) return kOccupationNonFinite;
        double n_trial = 0;
        for(int i=0;i<nstates;i++){
            h_occ[i] = g / (1.0 + std::exp((h_eig[i] - Efermi) / sigma));
            n_trial += h_occ[i];
        }
        if (!std::isfinite(n_trial)) return kOccupationNonFinite;
        if(fabs(n_trial-ne)<1e-6){
            if (!CHECK_CUDA(cudaMemcpyAsync(occupation.d_occupation,h_occ.data(),
                size_t(nstates)*sizeof(double),cudaMemcpyHostToDevice,occupation.context.stream)))
                return kCudaMemcpyError;
            if (!CHECK_CUDA(cudaStreamSynchronize(occupation.context.stream))) return kCudaSyncError;
            result = Efermi;
            return kSuccess;
        }
        if(n_trial<ne) Elo = Efermi;
        if(n_trial>ne) Ehi = Efermi;
    }
    return kOccupationNoConvergence;
}
}

int destroy_Occupation(Occupation& occupation) noexcept
{
    occupation.status = kOccupationNotInitialized;
    if (occupation.d_occupation && !CHECK_CUDA(cudaFree(occupation.d_occupation)))
        return report_occupation_status(kCudaFreeError, "destroy_Occupation");
    occupation = {};
    return kSuccess;
}

int Occupation_init(const Context context,Occupation& occupation,
     std::string occ_mode, int nstates, int ne, double temperature) noexcept
{
    if (occupation.d_occupation || occupation.status == kSuccess)
        return report_occupation_status(kInvalidState, "Occupation_init");
    if (!valid_parameters(nstates, ne, temperature, occ_mode))
        return report_occupation_status(kInvalidArgument, "Occupation_init");
    auto fail = [&](int status) {
        destroy_Occupation(occupation);
        return report_occupation_status(status, "Occupation_init");
    };
    try {
        occupation.nstates = nstates;
        occupation.temperature = temperature;
        occupation.ne = ne;
        occupation.occ_mode = occ_mode;
        occupation.context = context;
        if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&occupation.d_occupation),
                                  size_t(nstates)*sizeof(double))))
            return fail(kAllocationError);
        if (occ_mode == "fixed") {
            const int status = fixed(occupation);
            if (status != kSuccess) return fail(status);
        }
    } catch (const std::bad_alloc&) {
        return fail(kAllocationError);
    }
    occupation.status = kSuccess;
    return kSuccess;
}

int calculate_occupation(Occupation& occupation,double* d_eigenvalues) noexcept
{
    if (occupation.status != kSuccess || occupation.d_occupation == nullptr)
        return report_occupation_status(kOccupationNotInitialized, "calculate_occupation");
    if (!valid_parameters(occupation.nstates, occupation.ne,
                          occupation.temperature, occupation.occ_mode) ||
        (occupation.occ_mode != "fixed" && d_eigenvalues == nullptr))
        return report_occupation_status(kInvalidArgument, "calculate_occupation");
    if (occupation.occ_mode == "fixed") return kSuccess;
    int status;
    try {
        status = fermi_level(occupation, d_eigenvalues, occupation.fermilevel);
    } catch (const std::bad_alloc&) {
        status = kAllocationError;
    }
    if (status != kSuccess) occupation.status = kOccupationNotInitialized;
    return report_occupation_status(status, "calculate_occupation");
}
int calculate_rho(Context& context,Occupation& occupation,double* d_eigenvalues,
                    int npoints,double dV,double* d_psi,double* d_rho, double& TS) noexcept
{
    TS = 0.0;
    if (npoints <= 0 || !std::isfinite(dV) || dV <= 0.0 ||
        d_psi == nullptr || d_rho == nullptr)
        return report_occupation_status(kInvalidArgument, "calculate_rho");
    if (context.stream != occupation.context.stream)
        return report_occupation_status(kInvalidState, "calculate_rho");
    int status = calculate_occupation(occupation,d_eigenvalues);
    if (status != kSuccess) return status;
    status = real_calrho(context,occupation.d_occupation,d_psi,npoints,occupation.nstates,dV,d_rho);
    if (status != kSuccess) return report_occupation_status(status, "calculate_rho");
    if (occupation.occ_mode == "fixed") {
        if (!CHECK_CUDA(cudaStreamSynchronize(context.stream)))
            return report_occupation_status(kCudaSyncError, "calculate_rho");
        return kSuccess;
    }
    try {
        std::vector<double> h_occ(occupation.nstates);
        if (!CHECK_CUDA(cudaMemcpyAsync(h_occ.data(), occupation.d_occupation,
                size_t(occupation.nstates) * sizeof(double), cudaMemcpyDeviceToHost, context.stream)))
            return report_occupation_status(kCudaMemcpyError, "calculate_rho");
        if (!CHECK_CUDA(cudaStreamSynchronize(context.stream)))
            return report_occupation_status(kCudaSyncError, "calculate_rho");
        const double sigma = 3.166811563e-6 * occupation.temperature;
        double entropy_term = 0.0;
        for (double n : h_occ) {
            if (!std::isfinite(n))
                return report_occupation_status(kOccupationNonFinite, "calculate_rho");
            if (n < 0.0 || n > 2.0)
                return report_occupation_status(kInvalidArgument, "calculate_rho");
            const double p = n / 2.0;
            if (p > 0.0) entropy_term -= 2.0 * sigma * p * std::log(p);
            if (p < 1.0) entropy_term -= 2.0 * sigma * (1.0 - p) * std::log1p(-p);
        }
        if (!std::isfinite(entropy_term))
            return report_occupation_status(kOccupationNonFinite, "calculate_rho");
        TS = entropy_term;
    } catch (const std::bad_alloc&) {
        return report_occupation_status(kAllocationError, "calculate_rho");
    }
    return kSuccess;
}
