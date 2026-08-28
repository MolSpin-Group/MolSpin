/////////////////////////////////////////////////////////////////////////
// ResonanceFieldJacobian implementation (RunSection::General::Resonance)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Field-to-frequency Jacobian support for resonance spectra.
//
// What is done here:
//   - Evaluates how transition frequencies change with magnetic field near a resonance.
//   - Provides the field-domain Jacobian used when converting/intensifying frequency-domain transition information.
//
// Connections to the General framework / SpinAPI:
//   - Consumes eigeninformation from the resonance Hamiltonian layer.
//   - Does not own powder sampling, lineshapes or SpinAPI interaction construction.
//
// Why this ownership is used:
//   - Keeping the Jacobian explicit avoids hiding field/frequency conversion inside empirical linewidth code.
/////////////////////////////////////////////////////////////////////////

#include "ResonanceFieldJacobian.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RunSection::General::Resonance
{
	bool ResonanceFieldJacobian::ResolveDegenerateSubspaces(const arma::vec &energies,
		arma::cx_mat &eigenvectors, const arma::sp_cx_mat &dHdB,
		arma::vec &dEdB, std::string &error)
	{
		error.clear();
		if (energies.n_elem == 0 || !energies.is_finite() ||
			eigenvectors.n_rows != energies.n_elem ||
			eigenvectors.n_cols != energies.n_elem)
		{
			error = "energy and eigenvector dimensions are inconsistent";
			return false;
		}
		if (dHdB.n_rows != energies.n_elem || dHdB.n_cols != energies.n_elem)
		{
			error = "dH/dB dimension does not match the eigensystem";
			return false;
		}

		const arma::cx_mat derivative(dHdB);
		const double derivativeScale = std::max(1.0, arma::norm(derivative, "fro"));
		if (!derivative.is_finite() ||
			arma::norm(derivative - derivative.t(), "fro") > 1.0e-11 * derivativeScale)
		{
			error = "dH/dB must be a finite Hermitian operator";
			return false;
		}

		// Hellmann-Feynman slopes <n|dH/dB|n> are basis dependent when H
		// is degenerate. Degenerate perturbation theory selects the physical
		// field-following states by diagonalizing dH/dB inside each degenerate
		// block. This also makes transition moments independent of eig_sym's
		// arbitrary basis choice at a level crossing.
		const double energyScale = std::max(1.0, arma::abs(energies).max());
		const double degeneracyTolerance = 256.0 *
			std::numeric_limits<double>::epsilon() * energyScale;
		dEdB.set_size(energies.n_elem);
		arma::uword first = 0;
		while (first < energies.n_elem)
		{
			arma::uword end = first + 1;
			while (end < energies.n_elem &&
				std::abs(energies(end) - energies(first)) <= degeneracyTolerance)
				++end;

			if (end == first + 1)
			{
				const arma::cx_vec vector = eigenvectors.col(first);
				dEdB(first) = std::real(arma::cdot(vector, derivative * vector));
			}
			else
			{
				const arma::uvec columns = arma::regspace<arma::uvec>(first, end - 1);
				const arma::cx_mat block = eigenvectors.cols(columns);
				arma::cx_mat projected = block.t() * derivative * block;
				projected = 0.5 * (projected + projected.t());
				arma::vec slopes;
				arma::cx_mat mixing;
				if (!arma::eig_sym(slopes, mixing, projected))
				{
					error = "failed to resolve dH/dB inside a degenerate energy subspace";
					dEdB.reset();
					return false;
				}
				eigenvectors.cols(columns) = block * mixing;
				dEdB.subvec(first, end - 1) = slopes;
			}
			first = end;
		}

		if (!eigenvectors.is_finite() || !dEdB.is_finite())
		{
			error = "degenerate field-derivative resolution produced non-finite values";
			dEdB.reset();
			return false;
		}
		return true;
	}

	bool ResonanceFieldJacobian::DiagonalEnergyDerivatives(const arma::cx_mat &eigenvectors,
        const arma::sp_cx_mat &dHdB, arma::vec &dEdB, std::string &error)
    {
        error.clear();
        if (eigenvectors.n_rows == 0 || eigenvectors.n_rows != eigenvectors.n_cols)
        {
            error = "eigenvector matrix must be non-empty and square";
            return false;
        }
        if (dHdB.n_rows != eigenvectors.n_rows || dHdB.n_cols != eigenvectors.n_rows)
        {
            error = "dH/dB dimension does not match the eigensystem";
            return false;
        }

        const arma::cx_mat derivativeVectors = dHdB * eigenvectors;
        dEdB.set_size(eigenvectors.n_cols);
        for (arma::uword n = 0; n < eigenvectors.n_cols; ++n)
        {
            const double value = std::real(arma::cdot(eigenvectors.col(n), derivativeVectors.col(n)));
            if (!std::isfinite(value))
            {
                error = "non-finite eigenstate field derivative";
                dEdB.reset();
                return false;
            }
            dEdB(n) = value;
        }
        return true;
    }
}
