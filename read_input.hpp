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
        // 未提供 Hartree 边界时保持空字符串，不启用隐含默认值。
        std::string hartree_boundary;
        int multipole_order{};
    } domain;

    struct Mesh {
        std::string type;
        int finite_difference_halfwidth{}; // YAML: stencil_half_width，原始半宽值
        std::array<int, 3> rgrids{};
    } mesh;

    struct Inversion {
        std::string hartree_density;
        std::string initial_potential;

        struct Eigensolver {
            std::string method;
            double residual_tolerance{};
            int extra_states{};
            int max_subspace_dim{};
            int max_iterations{};
            int nextra_retain{};
            double direction_tolerance{};
        } eigensolver;

        struct Occupations {
            std::string method;
        } occupations;

        struct Optimizer {
            std::string method;
            int history{};
            int max_iterations{};
            int max_linesearch_step{};
            double c1{};
            double c2{};
        } optimizer;

        struct Objective {
            std::string density_metric;
            struct Regularization {
                std::string type;
                double strength{};
            } regularization;
        } objective;

        struct Convergence {
            double density_relative_l2{};
            double gradient_tolerance{};
        } convergence;
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
// 除可选的 Hartree 边界设置及 write_eigenvalues（默认 false）外，其余字段必填。
// 实际态数相关的 Davidson 约束仍需在读取分子电子数后检查。
int read_parameters(const std::string& filename,
                    InputParameters& parameters, std::string& error);
