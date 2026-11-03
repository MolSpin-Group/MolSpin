// Shared mathematical operations for action-spectrum histogram tasks.
// Frequency gaps are angular frequencies in rad/ns; the supplied scale
// converts them to the frequency units requested by the calling task.
#ifndef MOD_RunSection_ActionSpectrumHistogramUtils
#define MOD_RunSection_ActionSpectrumHistogramUtils

#include <armadillo>
#include <array>
#include <cmath>
#include <stdexcept>

namespace RunSection
{
namespace ActionSpectrumHistogramUtils
{
    inline arma::vec BinCenters(double upper_limit, double bin_width)
    {
        if (!std::isfinite(upper_limit) || !std::isfinite(bin_width) ||
            bin_width <= 0.0 || upper_limit < bin_width / 2.0)
            throw std::invalid_argument("Action histogram requires finite upper_limit >= bin_width/2 and bin_width > 0.");
        return arma::regspace(bin_width / 2.0, bin_width, upper_limit);
    }

    inline arma::vec NormalizedHeights(const arma::vec &energy_gaps,
                                       const arma::vec &resonance_effects,
                                       double bin_width, arma::uword num_bins,
                                       double frequency_scale)
    {
        if (energy_gaps.n_elem != resonance_effects.n_elem ||
            !std::isfinite(bin_width) || bin_width <= 0.0 ||
            !std::isfinite(frequency_scale) || frequency_scale <= 0.0)
            throw std::invalid_argument("Invalid action-histogram inputs.");

        arma::vec heights(num_bins, arma::fill::zeros);
        const double upper_edge = static_cast<double>(num_bins) * bin_width;
        for (arma::uword k = 0; k < energy_gaps.n_elem; ++k)
        {
            const double gap = std::abs(energy_gaps(k)) * frequency_scale;
            const double intensity = std::abs(resonance_effects(k));
            if (!std::isfinite(gap) || !std::isfinite(intensity))
                throw std::invalid_argument("Non-finite action-histogram frequency or intensity.");
            // The n-th bin covers [n*width, (n+1)*width). Out-of-range
            // resonances are excluded; never access a nonexistent bin.
            if (gap >= upper_edge || intensity == 0.0)
                continue;
            const arma::uword bin = static_cast<arma::uword>(std::floor(gap / bin_width));
            if (bin < num_bins)
                heights(bin) += intensity;
        }
        const double total = arma::accu(heights);
        // An identically zero spectrum is a valid outcome (e.g. no allowed
        // transitions), not a reason to produce NaNs.
        if (total > 0.0 && std::isfinite(total))
            heights /= total;
        else if (!std::isfinite(total))
            throw std::overflow_error("Action-histogram intensity sum is not finite.");
        return heights;
    }

    // For two uncoupled radical subspaces, P_S = 1/4 - S1 dot S2.
    // Spin operators must be embedded using SpinSpace::CreateOperator, so
    // the result does not assume that the electron is first in either basis.
    inline arma::cx_vec SingletPopulations(
        const arma::cx_mat &V1, const arma::cx_mat &V2,
        const std::array<arma::cx_mat, 3> &S1,
        const std::array<arma::cx_mat, 3> &S2)
    {
        if (V1.n_rows == 0 || V2.n_rows == 0 ||
            V1.n_rows != V1.n_cols || V2.n_rows != V2.n_cols)
            throw std::invalid_argument("Invalid radical eigenvector dimensions.");

        arma::cx_vec populations(V1.n_cols * V2.n_cols, arma::fill::ones);
        populations *= 0.25;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (S1[axis].n_rows != V1.n_rows || S1[axis].n_cols != V1.n_cols ||
                S2[axis].n_rows != V2.n_rows || S2[axis].n_cols != V2.n_cols)
                throw std::invalid_argument("Radical spin operator dimension mismatch.");
            const arma::cx_mat rotated1 = V1.t() * S1[axis] * V1;
            const arma::cx_mat rotated2 = V2.t() * S2[axis] * V2;
            const arma::cx_vec d1 = rotated1.diag();
            const arma::cx_vec d2 = rotated2.diag();
            populations -= arma::kron(d1, d2);
        }
        return populations;
    }
}
}
#endif
