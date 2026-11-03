// Deterministic unit checks for the action-spectrum histogram math.
#include "ActionSpectrumHistogramUtils.h"
#include <array>
#include <cmath>
#include <vector>

bool ActionHistogramBinningRegression()
{
    using namespace RunSection::ActionSpectrumHistogramUtils;
    const arma::vec centers = BinCenters(2.0, 1.0);
    if (centers.n_elem != 2 || std::abs(centers(0) - 0.5) > 1e-14 ||
        std::abs(centers(1) - 1.5) > 1e-14)
        return false;
    const arma::vec gaps = {0.0, 0.99, 1.0, 1.999, 2.0, 10.0};
    const arma::vec effects = {1.0, 2.0, 3.0, 4.0, 100.0, 5.0};
    const arma::vec h = NormalizedHeights(gaps, effects, 1.0, centers.n_elem, 1.0);
    if (h.n_elem != 2 || !h.is_finite() ||
        std::abs(h(0) - 0.3) > 1e-13 || std::abs(h(1) - 0.7) > 1e-13)
        return false;
    const arma::vec excluded = NormalizedHeights(arma::vec({2.0, 30.0}),
        arma::vec({2.0, 50.0}), 1.0, centers.n_elem, 1.0);
    if (!excluded.is_finite() || arma::accu(excluded) != 0.0)
        return false;
    const arma::vec zero = NormalizedHeights(arma::vec({0.2, 1.5}),
        arma::vec({0.0, 0.0}), 1.0, centers.n_elem, 1.0);
    if (!zero.is_finite() || arma::accu(zero) != 0.0)
        return false;
    try { (void)BinCenters(10.0, 0.0); return false; }
    catch (const std::invalid_argument &) {}
    return true;
}

bool ActionHistogramSingletRegression()
{
    using namespace RunSection::ActionSpectrumHistogramUtils;
    arma::cx_mat sx(2, 2, arma::fill::zeros);
    arma::cx_mat sy(2, 2, arma::fill::zeros);
    arma::cx_mat sz(2, 2, arma::fill::zeros);
    sx(0, 1) = sx(1, 0) = 0.5;
    sy(0, 1) = std::complex<double>(0.0, -0.5);
    sy(1, 0) = std::complex<double>(0.0, 0.5);
    sz(0, 0) = 0.5; sz(1, 1) = -0.5;
    const std::array<arma::cx_mat, 3> spin = {sx, sy, sz};
    arma::cx_mat id2(2, 2, arma::fill::eye);
    const arma::cx_vec p = SingletPopulations(id2, id2, spin, spin);
    if (p.n_elem != 4 || std::abs(p(0)) > 1e-13 ||
        std::abs(p(1) - std::complex<double>(0.5)) > 1e-13 ||
        std::abs(p(2) - std::complex<double>(0.5)) > 1e-13 ||
        std::abs(p(3)) > 1e-13)
        return false;

    // Electron-last subspace: |nuclear> tensor |electron>, not the
    // electron-first assumption made by the original RP-only task.
    arma::cx_mat id4(4, 4, arma::fill::eye);
    const std::array<arma::cx_mat, 3> nuclear_first = {
        arma::kron(id2, sx), arma::kron(id2, sy), arma::kron(id2, sz)};
    const arma::cx_vec p16 = SingletPopulations(id4, id4, nuclear_first, nuclear_first);
    if (p16.n_elem != 16 || std::abs(p16(1) - std::complex<double>(0.5)) > 1e-13)
        return false;
    const arma::vec sorted = arma::sort(arma::real(p16));
    for (arma::uword i = 0; i < 16; ++i)
        if (std::abs(sorted(i) - (i < 8 ? 0.0 : 0.5)) > 1e-13)
            return false;
    return true;
}

void AddActionSpectrumHistogramTests(std::vector<test_case> &cases)
{
    cases.emplace_back("Action histogram binning / zero intensity", ActionHistogramBinningRegression);
    cases.emplace_back("Action histogram uncoupled singlet populations", ActionHistogramSingletRegression);
}
