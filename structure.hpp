#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

struct InputParameters;

struct Atom {
    int nuclear_charge{};
    std::array<double, 3> position{}; // bohr，来自 PySCF 内部 _env
};

struct GaussianShell {
    int atom_index{};
    int angular_momentum{};
    int ncontractions{};
    std::size_t ao_offset{};
    std::vector<double> exponents;
    // PySCF _env 中已归一化的径向系数：[contraction][primitive]。
    std::vector<double> coefficients;
};

struct StructGrid {
    std::vector<Atom> atoms;
    std::vector<GaussianShell> shells;
    int charge{};
    int electron_number{};
    std::size_t nao{};

    std::array<int, 3> rgrids{}; // 盒内未知点数，不包含物理零边界或 halo
    std::array<double, 3> lower{};
    std::array<double, 3> upper{};
    std::array<double, 3> gap{}; // (upper-lower)/(rgrids+1)
    std::array<std::vector<double>, 3> coordinates;
    int finite_difference_halfwidth{};
    std::size_t npoints{};
    double dV{};

    // 主机密度，bohr^-3；索引 (z*ny+y)*nx+x，x 连续，无边界/halo。
    std::vector<double> target_density;
    double density_electron_number{}; // dV * sum(target_density)，归一化至 electron_number
};

// 读取 PySCF molecule.json 并建立内部节点网格，不读取 YAML 或密度矩阵。
// 当前支持无 ECP、spin=0 的球谐 s/p/d/f 基组，保留原文件的 AO 顺序。
// 返回 0 表示成功并清空 error；失败返回 1，grid 保持不变。
int creat_StructGrid(const InputParameters& parameters,
                     StructGrid& grid, std::string& error);

// 读取 NPY v1.0、<f8、C 顺序的 AO 总密度矩阵，计算 phi^T P phi。
// 不重复乘自旋因子；网格密度按 electron_number/(dV*sum(rho)) 缩放。
// 仅读取成功后替换 grid 的密度与积分，不修改源 NPY 文件。
// grid 必须先由 creat_StructGrid 成功建立；不分配 GPU 内存。
int read_density_input(const std::filesystem::path& density_file,
                       StructGrid& grid, std::string& error);
