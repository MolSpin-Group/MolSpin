/////////////////////////////////////////////////////////////////////////
// QuantumMap implementation (SpinAPI Module)
// ----------------------------------------------------------------------
// ROLE
//   QuantumMap contains instantaneous completely-positive operations only.
//   General/MultiSS's EventController decides *when* to apply them; this class
//   has no scheduler and does not use mutable PulseSequence Active flags.
//
// PHYSICS
//   For a transfer fraction f on a projector-valued source support G,
//
//       K0 = I - (1-sqrt(1-f)) G,
//       rho_s' = K0 rho_s K0^dagger,
//       rho_t' = rho_t + f sum_mu C_mu rho_s C_mu^dagger.
//
//   The sqrt(1-f) factor is essential: coherences between transferred and
//   untransferred source subspaces transform as amplitudes, not probabilities.
//   This is the appropriate delta-pulse limit of a finite incoherent transfer.
//
//   Mims et al. model pump/push redistribution as an idealized instantaneous,
//   spin-selective operation in radical-pair dynamics:
//       DOI: 10.1126/science.abl4254
//
// NUMERICAL SAFETY
//   The implementation verifies Hermiticity dimensions, checks that G is a
//   projector (the closed-form K0 formula assumes this), validates f in [0,1],
//   and checks total source+target trace conservation after the map.
/////////////////////////////////////////////////////////////////////////
#include "QuantumMap.h"

#include <cmath>

namespace SpinAPI
{
    bool QuantumMap::ApplyTransferEvent(const TransferChannel &_channel,
        double _fraction, arma::cx_mat &_sourceDensity,
        arma::cx_mat &_targetDensity, std::string &_error, double _tolerance)
    {
        if (!_channel.HasTarget() || _channel.KrausOperators().empty())
        { _error = "instantaneous transfer map requires a target and Kraus operator"; return false; }
        return ApplyTransferEvent(_channel.SourceEffect(), _channel.KrausOperators(),
            _fraction, _sourceDensity, _targetDensity, _error, _tolerance);
    }

    bool QuantumMap::ApplyTransferEvent(const arma::sp_cx_mat &_sourceEffect,
        const std::vector<arma::sp_cx_mat> &_kraus, double _fraction,
        arma::cx_mat &_sourceDensity, arma::cx_mat &_targetDensity,
        std::string &_error, double _tolerance)
    {
        _error.clear();
        if (_kraus.empty())
        { _error = "instantaneous transfer map requires at least one Kraus operator"; return false; }
        if (!std::isfinite(_fraction) || _fraction < 0.0 || _fraction > 1.0)
        { _error = "instantaneous transfer fraction must satisfy 0 <= f <= 1"; return false; }

        const arma::cx_mat G(_sourceEffect);
        if (_sourceDensity.n_rows != G.n_rows || _sourceDensity.n_cols != G.n_cols)
        { _error = "source density dimension does not match instantaneous transfer channel"; return false; }
        const arma::cx_mat C0(_kraus.front());
        if (_targetDensity.n_rows != C0.n_rows || _targetDensity.n_cols != C0.n_rows || C0.n_cols != G.n_rows)
        { _error = "target density dimension does not match instantaneous transfer channel"; return false; }

        // Event semantics are currently defined for a partial isometry, hence
        // G=sum C_mu^dagger C_mu must be a projector. General non-projective
        // POVM events can be added later without changing continuous channels.
        const double gScale = std::max(1.0, arma::norm(G, "fro"));
        if (arma::norm(G * G - G, "fro") > 50.0 * _tolerance * gScale)
        { _error = "instantaneous transfer SourceEffect is not a projector"; return false; }

        const arma::cx_mat I = arma::eye<arma::cx_mat>(G.n_rows, G.n_cols);
        const arma::cx_mat K0 = I - (1.0 - std::sqrt(1.0 - _fraction)) * G;
        const arma::cx_mat beforeSource = _sourceDensity;
        const arma::cx_mat beforeTarget = _targetDensity;
        arma::cx_mat gain(_targetDensity.n_rows, _targetDensity.n_cols, arma::fill::zeros);
        for (const auto &sparseC : _kraus)
        {
            const arma::cx_mat Cm(sparseC);
            if (Cm.n_rows != _targetDensity.n_rows || Cm.n_cols != _sourceDensity.n_rows)
            { _error = "inconsistent Kraus dimensions in instantaneous transfer map"; return false; }
            gain += Cm * beforeSource * Cm.t();
        }

        _sourceDensity = K0 * beforeSource * K0.t();
        _targetDensity = beforeTarget + _fraction * gain;
        _sourceDensity = 0.5 * (_sourceDensity + _sourceDensity.t());
        _targetDensity = 0.5 * (_targetDensity + _targetDensity.t());

        const arma::cx_double trBefore = arma::trace(beforeSource) + arma::trace(beforeTarget);
        const arma::cx_double trAfter = arma::trace(_sourceDensity) + arma::trace(_targetDensity);
        if (std::abs(trAfter - trBefore) > 100.0 * _tolerance * std::max(1.0, std::abs(trBefore)))
        { _error = "instantaneous transfer map failed trace conservation"; return false; }
        return true;
    }
}
