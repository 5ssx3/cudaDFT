#include "read_input.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

void require(bool condition, const std::string& path, const std::string& reason)
{
    if (!condition) throw std::runtime_error(path + ": " + reason);
}

void check_mapping(const YAML::Node& node, const std::string& path,
                   std::initializer_list<const char*> allowed)
{
    require(node.IsDefined() && node.IsMap(), path, "expected a mapping");
    std::set<std::string> keys;
    for (const auto& entry : node) {
        require(entry.first.IsScalar(), path, "expected a scalar key");
        const std::string key = entry.first.as<std::string>();
        const std::string field = path == "root" ? key : path + "." + key;
        require(keys.insert(key).second, field, "duplicate key");
        require(std::find(allowed.begin(), allowed.end(), key) != allowed.end(),
                field, "unknown parameter");
    }
}

template <typename T>
T value(const YAML::Node& node, const char* key, const std::string& path)
{
    const YAML::Node field = node[key];
    require(field.IsDefined(), path + "." + key, "required parameter is missing");
    require(field.IsScalar(), path + "." + key, "expected a scalar value");
    try {
        return field.as<T>();
    } catch (const YAML::Exception&) {
        throw std::runtime_error(path + "." + key + ": invalid value or type");
    }
}

std::string choice(const YAML::Node& node, const char* key,
                   const std::string& path, const char* supported)
{
    const auto result = value<std::string>(node, key, path);
    require(result == supported, path + "." + key,
            std::string("supported value is '") + supported + "'");
    return result;
}

int integer(const YAML::Node& node, const char* key, const std::string& path,
            int minimum)
{
    const int result = value<int>(node, key, path);
    require(result >= minimum, path + "." + key,
            "must be >= " + std::to_string(minimum));
    return result;
}

double number(const YAML::Node& node, const char* key, const std::string& path,
              bool positive)
{
    const double result = value<double>(node, key, path);
    require(std::isfinite(result) && (positive ? result > 0 : result >= 0),
            path + "." + key, positive ? "must be finite and > 0" : "must be finite and >= 0");
    return result;
}

std::filesystem::path file_path(const YAML::Node& node, const char* key,
                                const std::string& path,
                                const std::filesystem::path& base)
{
    const auto text = value<std::string>(node, key, path);
    require(!text.empty(), path + "." + key, "path must not be empty");
    return (base / text).lexically_normal();
}

} // namespace

int read_parameters(const std::string& filename,
                    InputParameters& parameters, std::string& error)
{
    try {
        require(!filename.empty(), "input", "YAML filename must not be empty");
        const auto input = std::filesystem::absolute(filename).lexically_normal();
        const auto documents = YAML::LoadAllFromFile(input.string());
        require(documents.size() == 1, "document", "expected exactly one YAML document");
        const YAML::Node root = documents.front();
        check_mapping(root, "root", {"system", "domain", "mesh", "inversion", "output"});
        InputParameters parsed;
        const auto base = input.parent_path();

        const auto system = root["system"];
        check_mapping(system, "system", {"molecule_file", "density_file", "spin"});
        parsed.system.molecule_file = file_path(system, "molecule_file", "system", base);
        parsed.system.density_file = file_path(system, "density_file", "system", base);
        parsed.system.spin = choice(system, "spin", "system", "unpolarized");

        const auto domain = root["domain"];
        check_mapping(domain, "domain", {"padding", "orbital_boundary"});
        const auto padding = domain["padding"];
        require(padding.IsDefined() && padding.IsSequence() && padding.size() == 3, "domain.padding",
                "expected exactly three nonnegative numbers");
        for (int axis = 0; axis < 3; ++axis) {
            double length;
            try {
                length = padding[axis].as<double>();
            } catch (const YAML::Exception&) {
                throw std::runtime_error("domain.padding: invalid number at axis " + std::to_string(axis));
            }
            require(std::isfinite(length) && length >= 0, "domain.padding",
                    "each padding must be finite and >= 0");
            parsed.domain.padding[axis] = length;
        }
        parsed.domain.orbital_boundary = choice(domain, "orbital_boundary", "domain", "dirichlet_zero");
        const auto mesh = root["mesh"];
        check_mapping(mesh, "mesh", {"type", "finite_half_width", "rgrids"});
        parsed.mesh.type = choice(mesh, "type", "mesh", "uniform");
        parsed.mesh.finite_difference_halfwidth = integer(mesh, "finite_half_width", "mesh", 1);
        std::istringstream grid_text(value<std::string>(mesh, "rgrids", "mesh"));
        int npoints = 1;
        for (int& dimension : parsed.mesh.rgrids) {
            require(bool(grid_text >> dimension) && dimension > 0, "mesh.rgrids",
                    "expected exactly three positive integers");
            require(npoints <= std::numeric_limits<int>::max() / dimension, "mesh.rgrids",
                    "grid point count exceeds the solver int range");
            npoints *= dimension;
        }
        grid_text >> std::ws;
        require(grid_text.eof(), "mesh.rgrids", "expected exactly three positive integers");
        require(parsed.mesh.finite_difference_halfwidth <= *std::min_element(parsed.mesh.rgrids.begin(), parsed.mesh.rgrids.end()),
                "mesh.finite_half_width", "half width exceeds grid dimensions");

        const auto inversion = root["inversion"];
        check_mapping(inversion, "inversion", {"initial_potential", "regularization_eta", "relativ_rho_L2tol",
                      "extra_states", "eigensolver", "occupations", "optimizer"});
        parsed.inversion.initial_potential = choice(inversion, "initial_potential", "inversion", "zero");
        parsed.inversion.regularization_eta = number(inversion, "regularization_eta", "inversion", false);
        parsed.inversion.relative_rho_L2tol = number(inversion, "relativ_rho_L2tol", "inversion", true);
        parsed.inversion.extra_states = integer(inversion, "extra_states", "inversion", 0);

        const auto eigensolver = inversion["eigensolver"];
        const std::string eig_path = "inversion.eigensolver";
        check_mapping(eigensolver, eig_path, {"method", "residual_tolerance", "max_subspace_dim",
                      "max_iterations", "nextra_retain", "direction_tolerance"});
        auto& eig = parsed.inversion.eigensolver;
        eig.method = choice(eigensolver, "method", eig_path, "davidson");
        eig.residual_tolerance = number(eigensolver, "residual_tolerance", eig_path, true);
        eig.max_subspace_dim = integer(eigensolver, "max_subspace_dim", eig_path, 1);
        eig.max_iterations = integer(eigensolver, "max_iterations", eig_path, 1);
        eig.nextra_retain = integer(eigensolver, "nextra_retain", eig_path, 0);
        eig.direction_tolerance = number(eigensolver, "direction_tolerance", eig_path, true);
        require(eig.direction_tolerance < 1, eig_path + ".direction_tolerance", "must be < 1");
        // 至少有一个占据态；实际 nstates = Ne/2 + extra_states 需在读取分子后确定。
        require(eig.max_subspace_dim <= npoints && parsed.inversion.extra_states < eig.max_subspace_dim - 1,
                eig_path + ".max_subspace_dim", "must exceed extra_states + 1 and be <= grid point count");
        require(eig.nextra_retain < eig.max_subspace_dim - parsed.inversion.extra_states - 1,
                eig_path + ".nextra_retain", "must leave room to expand beyond the retained states");

        const auto occupations = inversion["occupations"];
        check_mapping(occupations, "inversion.occupations", {"method", "temperature"});
        parsed.inversion.occupations.method = choice(occupations, "method", "inversion.occupations", "fixed");
        parsed.inversion.occupations.temperature = number(occupations, "temperature", "inversion.occupations", false);

        const auto optimizer = inversion["optimizer"];
        const std::string opt_path = "inversion.optimizer";
        check_mapping(optimizer, opt_path, {"method", "history", "max_iterations", "max_linesearch_step",
                      "gradient_tolerance", "c1", "c2"});
        auto& opt = parsed.inversion.optimizer;
        opt.method = choice(optimizer, "method", opt_path, "lbfgs");
        opt.history = integer(optimizer, "history", opt_path, 1);
        opt.max_iterations = integer(optimizer, "max_iterations", opt_path, 1);
        opt.max_linesearch_step = integer(optimizer, "max_linesearch_step", opt_path, 1);
        opt.gradient_tolerance = number(optimizer, "gradient_tolerance", opt_path, false);
        opt.c1 = number(optimizer, "c1", opt_path, true);
        opt.c2 = number(optimizer, "c2", opt_path, true);
        require(opt.c1 < 1, opt_path + ".c1", "must be < 1");
        require(opt.c1 < opt.c2 && opt.c2 < 1, opt_path + ".c2", "must satisfy c1 < c2 < 1");

        const auto output = root["output"];
        check_mapping(output, "output", {"directory", "write_veff", "write_density", "write_density_difference",
                      "write_wave", "write_eigenvalues"});
        auto& out = parsed.output;
        out.directory = file_path(output, "directory", "output", base);
        out.write_veff = value<bool>(output, "write_veff", "output");
        out.write_density = value<bool>(output, "write_density", "output");
        out.write_density_difference = value<bool>(output, "write_density_difference", "output");
        out.write_wave = value<bool>(output, "write_wave", "output");
        out.write_eigenvalues = output["write_eigenvalues"].IsDefined()
            ? value<bool>(output, "write_eigenvalues", "output") : false;

        parameters = std::move(parsed);
        error.clear();
        return 0;
    } catch (const YAML::Exception& exception) {
        error = "YAML input '" + filename + "': " + exception.what();
    } catch (const std::exception& exception) {
        error = "input '" + filename + "': " + exception.what();
    }
    return 1;
}
