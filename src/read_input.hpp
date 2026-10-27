#pragma once

#include <array>
#include <filesystem>
#include <string>

struct InputParameters {
    struct System {
        std::filesystem::path molecule_file;
        std::filesystem::path density_file;
        std::string spin;
    } system;

    struct Domain {
        std::array<double, 3> padding{};
        std::string orbital_boundary;
    } domain;

    struct Mesh {
        std::string type;
        int finite_difference_halfwidth{}; // YAML: finite_half_width，原始半宽值
        std::array<int, 3> rgrids{};
    } mesh;

    struct Inversion {
        std::string initial_potential;
        double regularization_eta{};
        double relative_rho_L2tol{}; // YAML: relativ_rho_L2tol
        int extra_states{};

        struct Eigensolver {
            std::string method;
            double residual_tolerance=1e-9;
            int max_subspace_dim{};
            int max_iterations=10000;
            int nextra_retain{};
            double direction_tolerance=1e-8;
        } eigensolver;

        struct Occupations {
            std::string method;
            double temperature{};
        } occupations;

        struct Optimizer {
            std::string method;
            int history{};
            int max_iterations = 1000;
            int max_linesearch_step = 50;
            double gradient_tolerance = 5e-6;
            double c1 = 1e-4;
            double c2 = 0.9;
        } optimizer;
    } inversion;

    struct Output {
        std::filesystem::path directory;
        bool write_veff{};
        bool write_density{};
        bool write_density_difference{};
        bool write_wave{};
        bool write_eigenvalues{};
    } output;
};

// C++17 + yaml-cpp。仅读取 YAML；不读取分子/密度文件或创建目录、网格。
// 相对路径以 YAML 所在目录为基准，保存为规范化的绝对路径。
// 返回 0 表示成功并清空 error；失败返回 1，填写 error，parameters 保持不变。
// 除 write_eigenvalues（默认 false）外，其余字段必填。
// 实际态数相关的 Davidson 约束仍需在读取分子电子数后检查。
int read_parameters(const std::string& filename,
                    InputParameters& parameters, std::string& error);
