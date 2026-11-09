#pragma once

// Restarted, right-preconditioned complex GMRES for a fully assembled
// time-independent NZ Liouvillian. The physical generator is NOT modified.
// The preconditioner is the inverse action of the reaction-only Sylvester
// operator, Q*X + X*Q^H, in MolSpin's ROW-major supervector convention.
// Armadillo syl(Y,A,B,C) solves A*Y + Y*B + C = 0.

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace RunSection { namespace NZKrylovStationary {

struct Options {
    unsigned restart = 32;
    unsigned max_iterations = 320;
    double relative_tolerance = 1.0e-10;
};

struct Report {
    bool converged = false;
    bool preconditioner_failed = false;
    unsigned iterations = 0;
    unsigned restarts = 0;
    double relative_residual = std::numeric_limits<double>::infinity();
};

// Input/output are row-major by definition: SpinSpace::OperatorToSuperspace
// uses vectorise(rho.st()). Armadillo's reshape uses column-major storage.
inline bool ApplySylvesterInverse(const arma::cx_mat &q,
                                  const arma::cx_vec &source,
                                  arma::cx_vec &destination)
{
    const arma::uword d = q.n_rows;
    if (d == 0 || q.n_cols != d || source.n_elem != d * d)
        return false;
    const arma::cx_mat rhs = arma::reshape(source, d, d).st();
    arma::cx_mat integrated;
    // syl() solves q*integrated + integrated*q^H = rhs
    try {
        if (!arma::syl(integrated, q, q.t(), -rhs)) return false;
    } catch (...) {
        return false;
    }
    if (integrated.n_rows != d || integrated.n_cols != d ||
        !integrated.is_finite())
        return false;
    destination = arma::vectorise(integrated.st());
    return destination.is_finite();
}

inline bool Solve(const arma::cx_mat &liouvillian,
                  const arma::cx_vec &rhs,
                  const arma::cx_mat &reaction_generator,
                  arma::cx_vec &solution,
                  Report &report,
                  const Options &options = Options())
{
    report = Report{};
    const arma::uword n = rhs.n_elem;
    const arma::uword d = reaction_generator.n_rows;
    if (n == 0 || liouvillian.n_rows != n || liouvillian.n_cols != n ||
        d == 0 || reaction_generator.n_cols != d || d * d != n ||
        options.restart < 2 || options.restart > 256 ||
        options.max_iterations < 1 || options.max_iterations > 10000 ||
        !std::isfinite(options.relative_tolerance) ||
        options.relative_tolerance < 1.0e-14 ||
        options.relative_tolerance > 1.0e-3 ||
        !rhs.is_finite() || !liouvillian.is_finite() ||
        !reaction_generator.is_finite())
        return false;

    const double rhs_norm = arma::norm(rhs, 2);
    solution.zeros(n);
    if (rhs_norm == 0.0) {
        report.relative_residual = 0.0;
        report.converged = true;
        return true;
    }
    if (!std::isfinite(rhs_norm)) return false;

    const unsigned restart = std::min(options.restart, options.max_iterations);
    std::vector<arma::cx_vec> v(restart + 1);
    std::vector<arma::cx_vec> z(restart);
    arma::cx_mat h(restart + 1, restart, arma::fill::zeros);

    while (report.iterations < options.max_iterations) {
        const arma::cx_vec residual = rhs - liouvillian * solution;
        const double beta = arma::norm(residual, 2);
        report.relative_residual = beta / rhs_norm;
        if (!std::isfinite(beta)) return false;
        if (report.relative_residual <= options.relative_tolerance) {
            report.converged = true;
            return true;
        }
        h.zeros();
        v[0] = residual / beta;
        const unsigned this_cycle =
            std::min(restart, options.max_iterations - report.iterations);
        bool updated = false;

        for (unsigned j = 0; j < this_cycle; ++j) {
            if (!ApplySylvesterInverse(reaction_generator, v[j], z[j])) {
                report.preconditioner_failed = true;
                return false;
            }
            arma::cx_vec w = liouvillian * z[j];
            if (!w.is_finite()) return false;
            // Two modified Gram-Schmidt passes control loss of orthogonality
            // for strongly non-normal NZ generators.
            for (unsigned pass = 0; pass != 2; ++pass) {
                for (unsigned k = 0; k <= j; ++k) {
                    const arma::cx_double c = arma::cdot(v[k], w);
                    h(k, j) += c;
                    w -= c * v[k];
                }
            }
            const double next_norm = arma::norm(w, 2);
            if (!std::isfinite(next_norm)) return false;
            h(j + 1, j) = next_norm;
            const bool breakdown = next_norm < 1.0e-14;
            if (!breakdown) v[j + 1] = w / next_norm;

            arma::cx_mat hj = h.submat(0, 0, j + 1, j);
            arma::cx_vec g(j + 2, arma::fill::zeros);
            g(0) = beta;
            arma::cx_vec coefficients;
            // Rectangular solve gives the least-squares minimizer.
            try {
                if (!arma::solve(coefficients, hj, g)) return false;
            } catch (...) {
                return false;
            }
            if (coefficients.n_elem != j + 1 || !coefficients.is_finite())
                return false;
            const double projected = arma::norm(g - hj * coefficients, 2);
            if (!std::isfinite(projected)) return false;
            ++report.iterations;
            if (projected / rhs_norm <= options.relative_tolerance ||
                breakdown || j + 1 == this_cycle) {
                for (unsigned k = 0; k <= j; ++k)
                    solution += coefficients(k) * z[k];
                ++report.restarts;
                updated = true;
                break;
            }
        }
        if (!updated) return false;
    }
    report.relative_residual = arma::norm(rhs - liouvillian * solution, 2) / rhs_norm;
    report.converged = std::isfinite(report.relative_residual) &&
                       report.relative_residual <= options.relative_tolerance;
    return report.converged;
}

}} // namespace RunSection::NZKrylovStationary
