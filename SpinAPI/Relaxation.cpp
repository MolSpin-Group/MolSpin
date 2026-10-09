/////////////////////////////////////////////////////////////////////////
// Relaxation correlation expansions (SpinAPI Module)
// ------------------
// Implements validated ordered correlation channels shared by Redfield and
// Nakajima-Zwanzig relaxation builders.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "Relaxation.h"

#include <cmath>
#include <complex>

namespace SpinAPI::Relaxation
{
    namespace
    {
        bool Fail(std::string *error, const std::string &message)
        {
            if (error) *error = message;
            return false;
        }

        bool AppendTerms(const std::vector<double> &amplitudes,
            const std::vector<double> &tauC,
            std::vector<ExponentialTerm> &terms, std::string *error)
        {
            if (amplitudes.empty() || amplitudes.size() != tauC.size())
                return Fail(error, "correlation amplitudes and tau_c must be equally sized and non-empty");

            terms.clear();
            terms.reserve(amplitudes.size());
            for (std::size_t i = 0; i < amplitudes.size(); ++i)
            {
                if (!std::isfinite(amplitudes[i]) || !std::isfinite(tauC[i]))
                    return Fail(error, "correlation amplitudes and tau_c must be finite");
                // Zero-amplitude entries are padding in several published fit
                // matrices.  They carry no physics and may safely be omitted.
                if (amplitudes[i] == 0.0) continue;
                if (!(tauC[i] > 0.0))
                    return Fail(error, "every nonzero exponential requires tau_c > 0");
                terms.push_back({amplitudes[i], tauC[i]});
            }
            return true;
        }

        std::size_t RequiredChannels(std::size_t operatorCount, bool diagonalOnly)
        {
            return diagonalOnly ? operatorCount : operatorCount * operatorCount;
        }
    }

    bool CorrelationExpansion::Shared(std::size_t count, bool diagonal,
        const std::vector<double> &amplitudes, const std::vector<double> &tauC,
        CorrelationExpansion &result, std::string *error)
    {
        if (error) error->clear();
        if (count == 0) return Fail(error, "a correlation expansion requires at least one operator");
        CorrelationExpansion candidate;
        candidate.operatorCount = count;
        candidate.diagonalOnly = diagonal;
        candidate.shared = true;
        candidate.channels.resize(1);
        if (!AppendTerms(amplitudes, tauC, candidate.channels[0], error)) return false;
        result = std::move(candidate);
        return true;
    }

    bool CorrelationExpansion::FromOperatorFactors(std::size_t count,
        bool diagonal, const std::vector<double> &factors, double tauC,
        CorrelationExpansion &result, std::string *error)
    {
        if (error) error->clear();
        if (count == 0 || factors.size() != count)
            return Fail(error, "def_g=1 requires one g factor per relaxation operator");
        if (!std::isfinite(tauC) || !(tauC > 0.0))
            return Fail(error, "def_g=1 requires one finite positive tau_c");
        for (double factor : factors)
            if (!std::isfinite(factor)) return Fail(error, "def_g=1 factors must be finite");

        CorrelationExpansion candidate;
        candidate.operatorCount = count;
        candidate.diagonalOnly = diagonal;
        candidate.channels.resize(RequiredChannels(count, diagonal));
        for (std::size_t first = 0; first < count; ++first)
        {
            const std::size_t secondBegin = diagonal ? first : 0;
            const std::size_t secondEnd = diagonal ? first + 1 : count;
            for (std::size_t second = secondBegin; second < secondEnd; ++second)
            {
                const double amplitude = factors[first] * factors[second];
                if (amplitude == 0.0) continue;
                const std::size_t channel = diagonal ? first : first * count + second;
                candidate.channels[channel].push_back({amplitude, tauC});
            }
        }
        result = std::move(candidate);
        return true;
    }

    bool CorrelationExpansion::PerChannel(std::size_t count, bool diagonal,
        const arma::mat &amplitudes, const arma::mat &tauC,
        CorrelationExpansion &result, std::string *error)
    {
        if (error) error->clear();
        if (count == 0) return Fail(error, "a correlation expansion requires at least one operator");
        if (amplitudes.n_rows != tauC.n_rows || amplitudes.n_cols != tauC.n_cols ||
            amplitudes.n_cols == 0)
            return Fail(error, "matrix-valued g and tau_c must have identical non-empty dimensions");
        const std::size_t expected = RequiredChannels(count, diagonal);
        if (amplitudes.n_rows != expected)
            return Fail(error, "matrix-valued g/tau_c row count does not match the selected operator correlations");

        CorrelationExpansion candidate;
        candidate.operatorCount = count;
        candidate.diagonalOnly = diagonal;
        candidate.channels.resize(expected);
        for (std::size_t row = 0; row < expected; ++row)
        {
            std::vector<double> rowAmplitudes(amplitudes.n_cols);
            std::vector<double> rowTau(tauC.n_cols);
            for (std::size_t column = 0; column < amplitudes.n_cols; ++column)
            {
                rowAmplitudes[column] = amplitudes(row, column);
                rowTau[column] = tauC(row, column);
            }
            if (!AppendTerms(rowAmplitudes, rowTau, candidate.channels[row], error)) return false;
        }
        result = std::move(candidate);
        return true;
    }

    std::size_t CorrelationExpansion::ChannelCount() const
    {
        return RequiredChannels(operatorCount, diagonalOnly);
    }

    bool CorrelationExpansion::Terms(std::size_t first, std::size_t second,
        const std::vector<ExponentialTerm> *&terms, std::string *error) const
    {
        if (error) error->clear();
        terms = nullptr;
        if (first >= operatorCount || second >= operatorCount)
            return Fail(error, "correlation operator index is out of range");
        if (diagonalOnly && first != second)
            return Fail(error, "cross-correlation requested from an autocorrelation-only expansion");
        const std::size_t channel = shared ? 0 :
            (diagonalOnly ? first : first * operatorCount + second);
        if (channel >= channels.size()) return Fail(error, "correlation channel is unavailable");
        terms = &channels[channel];
        return true;
    }

    bool EvaluateSpectralDensity(const std::vector<ExponentialTerm> &terms,
        const arma::cx_mat &frequencies, SpectralDensityFunction function,
        bool diagonalOutput, arma::cx_mat &spectralDensity, std::string *error)
    {
        if (error) error->clear();
        if (frequencies.n_rows == 0 || frequencies.n_rows != frequencies.n_cols ||
            !frequencies.is_finite())
            return Fail(error, "spectral-density frequencies must form a finite square matrix");

        spectralDensity.zeros(frequencies.n_rows, frequencies.n_cols);
        const arma::uword rowEnd = frequencies.n_rows;
        const arma::uword columnEnd = diagonalOutput ? 1 : frequencies.n_cols;
        for (arma::uword row = 0; row < rowEnd; ++row)
        {
            for (arma::uword offset = 0; offset < columnEnd; ++offset)
            {
                const arma::uword column = diagonalOutput ? row : offset;
                const arma::cx_double omega = frequencies(row, column);
                arma::cx_double value = 0.0;
                for (const auto &term : terms)
                {
                    if (!std::isfinite(term.amplitude) || !std::isfinite(term.tauC) ||
                        !(term.tauC > 0.0))
                        return Fail(error, "invalid exponential in correlation expansion");
                    if (function == SpectralDensityFunction::RealLorentzian)
                        value += term.amplitude * term.tauC /
                            (arma::cx_double(1.0, 0.0) + omega * omega * term.tauC * term.tauC);
                    else
                        value += term.amplitude /
                            (arma::cx_double(1.0 / term.tauC, 0.0) - arma::cx_double(0.0, 1.0) * omega);
                }
                spectralDensity(row, column) = value;
            }
        }
        return spectralDensity.is_finite();
    }
}
