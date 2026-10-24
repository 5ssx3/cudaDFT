#include "write.hpp"
#include "cuda_math.hpp"
#include "read_input.hpp"
#include "structure.hpp"
#include "LBFGS.hpp"
#include "Davidson.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr double bohr_to_angstrom = 0.529177210544;
constexpr double hartree_to_ev = 27.211386245981;
constexpr const char* symbols[] = {
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

std::vector<double> download_field(const Context& context, const double* device,
                                   std::size_t count, const char* name)
{
    require(device != nullptr, std::string(name) + ": null device pointer");
    require(count <= std::numeric_limits<std::size_t>::max() / sizeof(double),
            std::string(name) + ": byte count overflow");
    std::vector<double> host(count);
    const auto copy = cudaMemcpyAsync(host.data(), device, count * sizeof(double),
                                      cudaMemcpyDeviceToHost, context.stream);
    // 即使提交失败也先结束流上的操作，主机缓冲区才能安全释放。
    const auto sync = cudaStreamSynchronize(context.stream);
    require(copy == cudaSuccess,
            std::string(name) + ": CUDA copy: " + cudaGetErrorString(copy));
    require(sync == cudaSuccess,
            std::string(name) + ": CUDA sync: " + cudaGetErrorString(sync));
    for (double v : host) require(std::isfinite(v), std::string(name) + ": non-finite value");
    return host;
}

void write_locpot(const std::filesystem::path& path, const StructGrid& grid,
                  const char* field, const char* unit, const double* values)
{
    std::ofstream file(path);
    require(bool(file), "cannot open '" + path.string() + "'");
    file.imbue(std::locale::classic());
    file << std::scientific << std::setprecision(17);
    file << "cudaDFT " << field << " [" << unit << "]; origin_bohr=";
    for (int a = 0; a < 3; ++a) file << grid.lower[a] + grid.gap[a] << ',';
    file << " lower_bohr=";
    for (double v : grid.lower) file << v << ',';
    file << " upper_bohr=";
    for (double v : grid.upper) file << v << ',';
    file << "\n1.0\n";
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b)
            file << (a == b ? grid.rgrids[a] * grid.gap[a] * bohr_to_angstrom : 0.0) << ' ';
        file << '\n';
    }
    std::vector<int> species;
    for (const auto& atom : grid.atoms)
        if (std::find(species.begin(), species.end(), atom.nuclear_charge) == species.end())
            species.push_back(atom.nuclear_charge);
    for (int z : species) file << symbols[z] << ' ';
    file << '\n';
    for (int z : species)
        file << std::count_if(grid.atoms.begin(), grid.atoms.end(),
                             [z](const Atom& atom) { return atom.nuclear_charge == z; }) << ' ';
    file << "\nCartesian\n";
    for (int z : species) {
        for (const auto& atom : grid.atoms) {
            if (atom.nuclear_charge != z) continue;
            for (int a = 0; a < 3; ++a)
                file << (atom.position[a] - grid.lower[a] - grid.gap[a]) * bohr_to_angstrom << ' ';
            file << '\n';
        }
    }
    file << '\n' << grid.rgrids[0] << ' ' << grid.rgrids[1] << ' ' << grid.rgrids[2] << '\n';
    for (std::size_t p = 0; p < grid.npoints; ++p) {
        file << values[p];
        file << ((p % 5 == 4 || p + 1 == grid.npoints) ? '\n' : ' ');
    }
    file.close();
    require(bool(file), "cannot finish writing '" + path.string() + "'");
}

} // namespace

int write_results(const Context& context, const InputParameters& parameters,
                  const StructGrid& grid, const LBFGSState& lbfgs,
                  const DavidsonPlan& davidson, std::string& error)
{
    try {
        const auto& out = parameters.output;
        if (!out.write_veff && !out.write_density && !out.write_density_difference &&
            !out.write_wave && !out.write_eigenvalues) {
            error.clear();
            return 0;
        }
        require(!out.directory.empty(), "output.directory is empty");
        require(context.status == kSuccess, "CUDA context is not initialized");
        std::size_t count = 1;
        for (int a = 0; a < 3; ++a) {
            require(grid.rgrids[a] > 0 && std::isfinite(grid.gap[a]) && grid.gap[a] > 0 &&
                    std::isfinite(grid.lower[a]) && std::isfinite(grid.upper[a]) &&
                    grid.upper[a] > grid.lower[a], "invalid output grid geometry");
            require(count <= std::numeric_limits<std::size_t>::max() / std::size_t(grid.rgrids[a]),
                    "output grid size overflow");
            count *= grid.rgrids[a];
        }
        require(count == grid.npoints, "output grid point count mismatch");
        require(!grid.atoms.empty(), "output grid has no atoms");
        for (const auto& atom : grid.atoms) {
            require(atom.nuclear_charge > 0 && atom.nuclear_charge < int(sizeof(symbols) / sizeof(*symbols)),
                    "unsupported atomic number in output");
            for (double v : atom.position) require(std::isfinite(v), "non-finite atom position");
        }

        std::vector<double> veff, psi, density, difference, eigenvalues;
        if (out.write_eigenvalues) {
            require(davidson.nstates > 0, "no Davidson eigenvalues to output");
            eigenvalues = download_field(context, davidson.d_eigenvalues,
                                        std::size_t(davidson.nstates), "eigenvalues");
        }
        if (out.write_veff) {
            require(lbfgs.n == count, "LBFGS potential size does not match grid");
            veff = download_field(context, lbfgs.d_x, count, "veff");
            for (double& v : veff) {
                v *= hartree_to_ev;
                require(std::isfinite(v), "veff overflows eV units");
            }
        }
        const bool need_density = out.write_density || out.write_density_difference;
        if (need_density || out.write_wave) {
            require(davidson.npoints == count && davidson.nx == grid.rgrids[0] &&
                    davidson.ny == grid.rgrids[1] && davidson.nz == grid.rgrids[2],
                    "Davidson grid does not match output grid");
            require(std::isfinite(grid.dV) && grid.dV > 0, "invalid grid volume element");
            require(grid.electron_number > 0 && grid.electron_number % 2 == 0,
                    "unpolarized double occupation requires a positive even electron count");
            const int occupied = grid.electron_number / 2;
            require(davidson.nstates >= occupied, "insufficient occupied Davidson orbitals");
            const int states = out.write_wave ? davidson.nstates : occupied;
            require(std::size_t(states) <= std::numeric_limits<std::size_t>::max() / count,
                    "orbital output size overflow");
            psi = download_field(context, davidson.d_psi, count * std::size_t(states), "orbitals");
            if (need_density) {
                density.assign(count, 0.0);
                for (int s = 0; s < occupied; ++s)
                    for (std::size_t p = 0; p < count; ++p) {
                        const double v = psi[std::size_t(s) * count + p];
                        density[p] += 2.0 * v * v / grid.dV;
                    }
                for (double v : density) require(std::isfinite(v), "non-finite reconstructed density");
            }
            if (out.write_wave) {
                const double normalization = std::sqrt(grid.dV);
                for (double& v : psi) {
                    v /= normalization;
                    require(std::isfinite(v), "non-finite normalized orbital");
                }
            }
        }
        if (out.write_density_difference) {
            require(grid.target_density.size() == count, "target density size does not match grid");
            difference.resize(count);
            for (std::size_t p = 0; p < count; ++p) {
                difference[p] = density[p] - grid.target_density[p];
                require(std::isfinite(difference[p]), "non-finite density difference");
            }
        }

        std::filesystem::create_directories(out.directory);
        if (out.write_eigenvalues) {
            const auto path = out.directory / "EIGENVALUES";
            std::ofstream file(path);
            require(bool(file), "cannot open '" + path.string() + "'");
            file.imbue(std::locale::classic());
            file << "# state eigenvalue_Ha\n" << std::scientific << std::setprecision(17);
            for (std::size_t s = 0; s < eigenvalues.size(); ++s)
                file << s + 1 << ' ' << eigenvalues[s] << '\n';
            file.close();
            require(bool(file), "cannot finish writing '" + path.string() + "'");
        }
        if (out.write_veff)
            write_locpot(out.directory / "LOCPOT", grid, "veff", "eV", veff.data());
        if (out.write_density)
            write_locpot(out.directory / "DENSITY", grid, "density", "bohr^-3", density.data());
        if (out.write_density_difference)
            write_locpot(out.directory / "DENSITY_DIFFERENCE", grid, "density-target", "bohr^-3", difference.data());
        if (out.write_wave) {
            for (int s = 0; s < davidson.nstates; ++s) {
                std::ostringstream name;
                name << "ORBITAL_" << std::setfill('0') << std::setw(4) << s + 1;
                write_locpot(out.directory / name.str(), grid, "orbital", "bohr^-3/2",
                              psi.data() + std::size_t(s) * count);
            }
        }
        error.clear();
        return 0;
    } catch (const std::exception& exception) {
        error = "output: " + std::string(exception.what());
        return 1;
    }
}
