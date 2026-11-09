// Standalone test: g++ -std=c++14 -O2 test_nz_krylov_stationary.cpp -o test_nz_krylov_stationary -larmadillo
#include "NZKrylovStationary.h"
#include <armadillo>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
    using namespace RunSection::NZKrylovStationary;
    for (const arma::uword dim : {4u, 8u, 16u, 48u}) {
        arma::arma_rng::set_seed(1217 + dim);
        const arma::cx_mat random = arma::randn<arma::cx_mat>(dim, dim);
        const arma::cx_mat H = (random + random.t()) / 2.0;
        const arma::cx_mat K = 0.3 * arma::eye<arma::cx_mat>(dim, dim);
        const arma::cx_mat Q = -arma::cx_double(0.0, 1.0) * H - K;
        const arma::cx_mat eye = arma::eye<arma::cx_mat>(dim, dim);
        const arma::cx_mat L0 = arma::kron(Q, eye) + arma::kron(eye, arma::conj(Q));
        arma::cx_mat rho = arma::randn<arma::cx_mat>(dim, dim);
        rho = rho * rho.t();
        rho /= arma::trace(rho);
        const arma::cx_vec rhs = arma::vectorise(rho.st());  // MolSpin row-major
        Options options;
        options.restart = 35;
        options.max_iterations = 160;
        options.relative_tolerance = 1e-10;
        Report report;
        arma::cx_vec result;
        if (!Solve(L0, rhs, Q, result, report, options) || report.iterations != 1)
            throw std::runtime_error("reaction-only Sylvester preconditioner is not exact");
        if (dim <= 16) {
            const arma::cx_vec reference = arma::solve(L0, rhs);
            if (arma::norm(result - reference, 2) / arma::norm(reference, 2) > 1e-8)
                throw std::runtime_error("reaction-only direct parity failed");
        }
        // A diagonal (in the row-major eigenoperator basis) decoherence term.
        // In particular this is a full nonzero superoperator, not a term
        // implicitly absorbed into Q; the GMRES step must include it.
        arma::cx_mat L = L0;
        for (arma::uword i = 0; i < dim; ++i)
            for (arma::uword j = 0; j < dim; ++j)
                L(i * dim + j, i * dim + j) -= 0.01 * (1.0 + std::abs(double(i) - double(j)) / double(dim));
        if (!Solve(L, rhs, Q, result, report, options) || !report.converged)
            throw std::runtime_error("full dissipator GMRES failed");
        const double residual = arma::norm(L * result - rhs, 2) / arma::norm(rhs, 2);
        if (!(residual < options.relative_tolerance))
            throw std::runtime_error("full Liouvillian true residual failed");
        if (dim <= 16) {
            const arma::cx_vec reference = arma::solve(L, rhs);
            if (arma::norm(result - reference, 2) / arma::norm(reference, 2) > 1e-8)
                throw std::runtime_error("full dissipator direct parity failed");
        }
        std::cout << "PASS d=" << dim << " GMRES_iterations=" << report.iterations
                  << " true_relative_residual=" << residual << '\n';
    }
    return 0;
}
