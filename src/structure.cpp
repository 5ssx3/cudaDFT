#include "structure.hpp"
#include "read_input.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const std::string& field, const std::string& reason)
{
    if (!condition) throw std::runtime_error(field + ": " + reason);
}

std::vector<int> integer_row(const YAML::Node& row, std::size_t columns,
                             const std::string& field)
{
    require(row.IsSequence() && row.size() == columns, field, "unexpected row shape");
    try {
        return row.as<std::vector<int>>();
    } catch (const YAML::Exception&) {
        throw std::runtime_error(field + ": expected integer entries");
    }
}

void check_span(int offset, std::size_t count, std::size_t size,
                const std::string& field)
{
    require(offset >= 0 && std::size_t(offset) <= size && count <= size - std::size_t(offset),
            field, "offset outside _env");
}

void read_molecule(const std::filesystem::path& file, StructGrid& grid)
{
    const auto documents = YAML::LoadAllFromFile(file.string());
    require(documents.size() == 1 && documents.front().IsMap(), "molecule", "expected one JSON object");
    const YAML::Node molecule = documents.front();
    require(molecule["cart"].IsDefined() && !molecule["cart"].as<bool>(), "cart", "only spherical AOs are supported");
    require(molecule["spin"].as<int>() == 0, "spin", "unpolarized input requires spin=0");
    for (const char* key : {"_ecpbas", "_ecp", "_pseudo"}) {
        require(!molecule[key].IsDefined() || molecule[key].size() == 0, "ECP", "only all-electron molecules are supported");
    }
    grid.charge = molecule["charge"].as<int>();
    const auto env_node = molecule["_env"];
    require(env_node.IsDefined() && env_node.IsSequence(), "_env", "expected a numeric array");
    const auto env = env_node.as<std::vector<double>>();
    for (double number : env) require(std::isfinite(number), "_env", "entries must be finite");

    const auto atm = molecule["_atm"];
    require(atm.IsDefined() && atm.IsSequence() && atm.size() > 0, "_atm", "expected at least one atom");
    std::int64_t electrons = -std::int64_t(grid.charge);
    for (const auto& entry : atm) {
        const auto row = integer_row(entry, 6, "_atm");
        require(row[0] > 0 && row[2] == 1, "_atm", "expected positive point nuclear charges");
        check_span(row[1], 3, env.size(), "_atm");
        Atom atom;
        atom.nuclear_charge = row[0];
        for (int axis = 0; axis < 3; ++axis) atom.position[axis] = env[std::size_t(row[1]) + axis];
        grid.atoms.push_back(atom);
        electrons += row[0];
    }
    require(electrons > 0 && electrons <= std::numeric_limits<int>::max() && electrons % 2 == 0,
            "electron_number", "expected a positive even closed-shell electron count");
    grid.electron_number = int(electrons);

    const auto bas = molecule["_bas"];
    require(bas.IsDefined() && bas.IsSequence() && bas.size() > 0, "_bas", "expected Gaussian shells");
    for (const auto& entry : bas) {
        const auto row = integer_row(entry, 8, "_bas");
        require(row[0] >= 0 && std::size_t(row[0]) < grid.atoms.size(), "_bas", "invalid atom index");
        require(row[1] >= 0 && row[1] <= 3, "_bas.angular_momentum", "only s/p/d/f shells are supported");
        require(row[2] > 0 && row[3] > 0 && row[4] == 0, "_bas", "expected scalar Gaussian primitives and contractions");
        const std::size_t nprim = row[2];
        const std::size_t nctr = row[3];
        require(nprim <= env.size() && nctr <= env.size() / nprim, "_bas", "coefficient count exceeds _env size");
        check_span(row[5], nprim, env.size(), "_bas");
        check_span(row[6], nprim * nctr, env.size(), "_bas");
        GaussianShell shell;
        shell.atom_index = row[0];
        shell.angular_momentum = row[1];
        shell.ncontractions = row[3];
        shell.ao_offset = grid.nao;
        shell.exponents.assign(env.begin() + row[5], env.begin() + row[5] + nprim);
        shell.coefficients.assign(env.begin() + row[6], env.begin() + row[6] + nprim * nctr);
        for (double exponent : shell.exponents) require(exponent > 0, "_bas.exponent", "must be positive");
        const std::size_t shell_size = (2 * shell.angular_momentum + 1) * nctr;
        require(shell_size <= std::size_t(std::numeric_limits<int>::max()) - grid.nao,
                "_bas", "AO count exceeds int range");
        grid.nao += shell_size;
        grid.shells.push_back(std::move(shell));
    }
}

// PySCF/libcint 的实球谐实心多项式。p 轨道顺序为 x,y,z，d/f 为 m=-l,...,+l。
// _env 已包含径向归一化；这里只乘角向因子，不重复归一化收缩系数。
std::array<double, 7> solid_harmonics(int l, double x, double y, double z)
{
    constexpr double pi = 3.14159265358979323846;
    const double x2 = x * x, y2 = y * y, z2 = z * z;
    switch (l) {
    case 0:
        return {std::sqrt(1 / (4 * pi)), 0, 0, 0, 0, 0, 0};
    case 1: {
        const double c = std::sqrt(3 / (4 * pi));
        return {c * x, c * y, c * z, 0, 0, 0, 0};
    }
    case 2: {
        const double c = std::sqrt(15 / (4 * pi));
        return {c * x * y, c * y * z, std::sqrt(5 / (16 * pi)) * (2 * z2 - x2 - y2),
                c * x * z, c * (x2 - y2) / 2, 0, 0};
    }
    default:
        return {std::sqrt(35 / (32 * pi)) * y * (3 * x2 - y2),
                std::sqrt(105 / (4 * pi)) * x * y * z,
                std::sqrt(21 / (32 * pi)) * y * (4 * z2 - x2 - y2),
                std::sqrt(7 / (16 * pi)) * z * (2 * z2 - 3 * x2 - 3 * y2),
                std::sqrt(21 / (32 * pi)) * x * (4 * z2 - x2 - y2),
                std::sqrt(105 / (16 * pi)) * z * (x2 - y2),
                std::sqrt(35 / (32 * pi)) * x * (x2 - 3 * y2)};
    }
}

void evaluate_ao(const StructGrid& grid, const std::array<double, 3>& point,
                 std::vector<double>& ao, std::vector<double>& primitives)
{
    for (const auto& shell : grid.shells) {
        const auto& atom = grid.atoms[shell.atom_index];
        const double x = point[0] - atom.position[0];
        const double y = point[1] - atom.position[1];
        const double z = point[2] - atom.position[2];
        const double r2 = x * x + y * y + z * z;
        const auto angular = solid_harmonics(shell.angular_momentum, x, y, z);
        const std::size_t nprim = shell.exponents.size();
        const int nang = 2 * shell.angular_momentum + 1;
        for (std::size_t p = 0; p < nprim; ++p) primitives[p] = std::exp(-shell.exponents[p] * r2);
        for (int c = 0; c < shell.ncontractions; ++c) {
            double radial = 0;
            for (std::size_t p = 0; p < nprim; ++p) radial += shell.coefficients[c * nprim + p] * primitives[p];
            for (int m = 0; m < nang; ++m) ao[shell.ao_offset + c * nang + m] = radial * angular[m];
        }
    }
}

std::vector<double> read_matrix(const std::filesystem::path& file, std::size_t nao)
{
    std::ifstream input(file, std::ios::binary);
    require(bool(input), "density_file", "cannot open '" + file.string() + "'");
    unsigned char prefix[10]{};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    require(bool(input) && std::memcmp(prefix, "\x93NUMPY", 6) == 0, "NPY", "invalid magic or truncated prefix");
    require(prefix[6] == 1 && prefix[7] == 0, "NPY", "only version 1.0 is supported");
    const std::size_t header_size = prefix[8] + (std::size_t(prefix[9]) << 8);
    std::string header(header_size, '\0');
    input.read(header.data(), header.size());
    require(bool(input), "NPY", "truncated header");
    // NPY 头是 Python 字面字典；将 shape 元组转为序列后复用 YAML 解析器，不执行文本。
    std::replace(header.begin(), header.end(), '(', '[');
    std::replace(header.begin(), header.end(), ')', ']');
    const YAML::Node metadata = YAML::Load(header);
    require(metadata.IsMap() && metadata.size() == 3, "NPY header", "expected descr, fortran_order and shape");
    const auto descr = metadata["descr"];
    require(descr.IsDefined() && descr.IsScalar() && descr.as<std::string>() == "<f8",
            "NPY descr", "expected little-endian float64 (<f8)");
    const auto fortran_order = metadata["fortran_order"];
    require(fortran_order.IsDefined() && fortran_order.IsScalar() && !fortran_order.as<bool>(),
            "NPY fortran_order", "expected C order");
    const auto shape = metadata["shape"];
    require(shape.IsDefined() && shape.IsSequence() && shape.size() == 2, "NPY shape", "expected a two-dimensional matrix");
    require(shape[0].as<std::size_t>() == nao && shape[1].as<std::size_t>() == nao,
            "NPY shape", "matrix dimensions must match the molecule AO count");
    require(nao > 0 && nao <= std::numeric_limits<std::size_t>::max() / sizeof(double) / nao,
            "NPY shape", "matrix byte count overflows");
    const std::size_t bytes = nao * nao * sizeof(double);
    require(std::filesystem::file_size(file) == sizeof(prefix) + header_size + bytes,
            "NPY size", "file size does not match the matrix shape");
    const std::uint16_t native = 1;
    require(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 &&
            *reinterpret_cast<const unsigned char*>(&native) == 1,
            "NPY descr", "this reader requires a little-endian IEEE float64 host");
    std::vector<double> matrix(nao * nao);
    input.read(reinterpret_cast<char*>(matrix.data()), bytes);
    require(bool(input), "NPY size", "incomplete matrix data");
    double scale = 1;
    for (double element : matrix) {
        require(std::isfinite(element), "density matrix", "entries must be finite");
        scale = std::max(scale, std::abs(element));
    }
    for (std::size_t i = 0; i < nao; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            require(std::abs(matrix[i * nao + j] - matrix[j * nao + i]) <= 1e-12 * scale,
                    "density matrix", "must be symmetric");
        }
    }
    return matrix;
}

} // namespace

int creat_StructGrid(const InputParameters& parameters,
                     StructGrid& grid, std::string& error)
{
    try {
        require(parameters.mesh.type == "uniform", "mesh.type", "expected uniform");
        require(parameters.system.spin == "unpolarized", "system.spin", "expected unpolarized");
        require(parameters.domain.orbital_boundary == "dirichlet_zero", "domain.orbital_boundary", "expected dirichlet_zero");
        StructGrid parsed;
        read_molecule(parameters.system.molecule_file, parsed);
        parsed.rgrids = parameters.mesh.rgrids;
        parsed.finite_difference_halfwidth = parameters.mesh.finite_difference_halfwidth;
        parsed.npoints = 1;
        for (int count : parsed.rgrids) {
            require(count > 0 && parsed.npoints <= std::size_t(std::numeric_limits<int>::max()) / count,
                    "mesh.rgrids", "expected positive dimensions with total point count in int range");
            parsed.npoints *= count;
        }
        require(parsed.finite_difference_halfwidth > 0 &&
                parsed.finite_difference_halfwidth <= *std::min_element(parsed.rgrids.begin(), parsed.rgrids.end()),
                "mesh.finite_difference_halfwidth", "must be positive and no larger than any grid dimension");
        parsed.dV = 1;
        for (int axis = 0; axis < 3; ++axis) {
            const double padding = parameters.domain.padding[axis];
            require(std::isfinite(padding) && padding >= 0, "domain.padding", "must be finite and nonnegative");
            double lower = parsed.atoms.front().position[axis];
            double upper = lower;
            for (const auto& atom : parsed.atoms) {
                lower = std::min(lower, atom.position[axis]);
                upper = std::max(upper, atom.position[axis]);
            }
            parsed.lower[axis] = lower - padding;
            parsed.upper[axis] = upper + padding;
            const double length = parsed.upper[axis] - parsed.lower[axis];
            require(std::isfinite(length) && length > 0, "domain.length", "each box length must be finite and positive");
            parsed.gap[axis] = length / (double(parsed.rgrids[axis]) + 1);
            parsed.dV *= parsed.gap[axis];
            auto& points = parsed.coordinates[axis];
            points.resize(parsed.rgrids[axis]);
            for (int i = 0; i < parsed.rgrids[axis]; ++i) points[i] = parsed.lower[axis] + (double(i) + 1) * parsed.gap[axis];
        }
        require(std::isfinite(parsed.dV) && parsed.dV > 0, "grid.dV", "volume element must be finite and positive");
        grid = std::move(parsed);
        error.clear();
        return 0;
    } catch (const std::exception& exception) {
        error = "structure '" + parameters.system.molecule_file.string() + "': " + exception.what();
        return 1;
    }
}

int read_density_input(const std::filesystem::path& density_file,
                       StructGrid& grid, std::string& error)
{
    try {
        require(grid.npoints > 0 && grid.nao > 0 && !grid.shells.empty() && grid.dV > 0,
                "grid", "call creat_StructGrid before reading density");
        const auto matrix = read_matrix(density_file, grid.nao);
        std::vector<double> ao(grid.nao);
        std::size_t max_primitives = 0;
        for (const auto& shell : grid.shells) max_primitives = std::max(max_primitives, shell.exponents.size());
        std::vector<double> primitives(max_primitives);
        std::vector<double> density(grid.npoints);
        const std::size_t nx = grid.rgrids[0], ny = grid.rgrids[1];
        double integral = 0;
        for (std::size_t p = 0; p < grid.npoints; ++p) {
            const std::array<double, 3> point{grid.coordinates[0][p % nx],
                grid.coordinates[1][(p / nx) % ny], grid.coordinates[2][p / (nx * ny)]};
            evaluate_ao(grid, point, ao, primitives);
            double rho = 0;
            for (std::size_t i = 0; i < grid.nao; ++i) {
                rho += matrix[i * grid.nao + i] * ao[i] * ao[i];
                for (std::size_t j = 0; j < i; ++j) {
                    rho += (matrix[i * grid.nao + j] + matrix[j * grid.nao + i]) * ao[i] * ao[j];
                }
            }
            require(std::isfinite(rho), "density", "nonfinite value on grid");
            density[p] = rho;
            integral += rho;
        }
        integral *= grid.dV;
        require(std::isfinite(integral) && integral > 0, "density", "expected a positive finite integrated electron count");
        require(grid.electron_number > 0, "electron_number", "expected a positive electron count");
        // 将网格目标密度的离散积分缩放至分子电子数。
        const double scale = double(grid.electron_number) / integral;
        integral = 0;
        for (double& rho : density) {
            rho *= scale;
            require(std::isfinite(rho), "density", "nonfinite normalized value on grid");
            integral += rho;
        }
        grid.target_density = std::move(density);
        grid.density_electron_number = integral * grid.dV;
        error.clear();
        return 0;
    } catch (const std::exception& exception) {
        error = "density '" + density_file.string() + "': " + exception.what();
        return 1;
    }
}
