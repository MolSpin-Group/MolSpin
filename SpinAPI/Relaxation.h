/////////////////////////////////////////////////////////////////////////
// Relaxation correlation expansions (SpinAPI Module)
// ------------------
// A stationary fluctuation channel is represented as a sum of exponentials
//
//     C_ab(t) = sum_r g_ab,r exp(-t/tau_ab,r),   t >= 0.
//
// The channel index is either one autocorrelation per operator (`terms=1`)
// or one ordered pair (a,b) (`terms=0`).  This value-object replaces the raw
// matrix-pointer layout duplicated by the original Redfield and NZ tasks.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Relaxation
#define MOD_SpinAPI_Relaxation

#include <armadillo>
#include <cstddef>
#include <string>
#include <vector>

namespace SpinAPI::Relaxation
{
    enum class SpectralDensityFunction
    {
        ComplexOneSided,
        RealLorentzian
    };

    struct ExponentialTerm
    {
        double amplitude = 0.0;
        double tauC = 0.0;
    };

    class CorrelationExpansion
    {
    public:
        static bool Shared(std::size_t _operatorCount, bool _diagonalOnly,
            const std::vector<double> &_amplitudes,
            const std::vector<double> &_tauC,
            CorrelationExpansion &_result, std::string *_error = nullptr);

        static bool FromOperatorFactors(std::size_t _operatorCount,
            bool _diagonalOnly, const std::vector<double> &_factors,
            double _tauC, CorrelationExpansion &_result,
            std::string *_error = nullptr);

        static bool PerChannel(std::size_t _operatorCount, bool _diagonalOnly,
            const arma::mat &_amplitudes, const arma::mat &_tauC,
            CorrelationExpansion &_result, std::string *_error = nullptr);

        std::size_t OperatorCount() const { return operatorCount; }
        std::size_t ChannelCount() const;
        bool DiagonalOnly() const { return diagonalOnly; }

        bool Terms(std::size_t _first, std::size_t _second,
            const std::vector<ExponentialTerm> *&_terms,
            std::string *_error = nullptr) const;

    private:
        std::size_t operatorCount = 0;
        bool diagonalOnly = true;
        bool shared = false;
        std::vector<std::vector<ExponentialTerm>> channels;
    };

    // Evaluate J(omega) elementwise.  With diagonalOutput=true, only the
    // diagonal frequencies are represented, as required by this NZ kernel.
    bool EvaluateSpectralDensity(const std::vector<ExponentialTerm> &_terms,
        const arma::cx_mat &_frequencies, SpectralDensityFunction _function,
        bool _diagonalOutput, arma::cx_mat &_spectralDensity,
        std::string *_error = nullptr);
}

#endif
