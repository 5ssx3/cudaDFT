#pragma once
#include "cuda_math.hpp"

#include <string>

// 通用错误沿用 Status；占据数模块专用错误从 300 开始。
enum OccupationStatus : int {
    kOccupationSuccess = 0,
    kOccupationNotInitialized = 300,
    kOccupationNonFinite,
    kOccupationNoConvergence,
};

const char* occupation_status_message(int status) noexcept;

struct Occupation
{
    double* d_occupation{};
    int nstates{};
    int ne{};
    double temperature{};
    double fermilevel{};
    std::string occ_mode{};
    Context context{};
    int status = kOccupationNotInitialized;
};

// 返回通用 Status 或 OccupationStatus；失败时向 stderr 输出诊断。
// 释放失败时保留设备指针，可再次调用 destroy_Occupation。
int destroy_Occupation(Occupation& occupation) noexcept;
int Occupation_init(const Context context,Occupation& occupation,
     std::string occ_mode, int nstates, int ne, double temperature) noexcept;
// fixed 模式直接返回成功；fd 模式使用费米分布（温度 K，本征值 Ha）。
// 计算成功后更新 fermilevel；计算阶段失败后须销毁并重新初始化。
int calculate_occupation(Occupation& occupation,
                          double* d_eigenvalues) noexcept;
// TS 返回 Hartree 单位的熵项，fixed 模式及失败时为零。
int calculate_rho(Context& context,Occupation& occupation,double* d_eigenvalues,
                    int npoints,double dV,double* d_psi,double* d_rho,double& TS)noexcept;
