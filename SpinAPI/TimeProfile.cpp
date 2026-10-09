/////////////////////////////////////////////////////////////////////////
// TimeProfile implementation (SpinAPI Module)
// ----------------------------------------------------------------------
// ROLE IN THE NEW HIERARCHY
//   SpinAPI::TimeProfile is the lowest-level, immutable representation of a
//   scalar coefficient k(t).  It is consumed by SpinAPI::TransferChannel;
//   General/MultiSS evaluates the resulting channel while assembling a global
//   direct-sum Liouvillian; TaskMultiSSGeneral only orchestrates the calculation.
//   This file must therefore contain no RunSection task logic and no knowledge
//   of existing TaskMultiStaticSS implementations.
//
// PHYSICAL CONTRACT FOR OPTICAL DRIVING
//   A Gaussian TimeProfile represents an *incoherent population-transfer rate*,
//
//       k(t) = k_peak exp[-4 ln(2) (t-t0)^2 / FWHM^2],
//
//   not an optical electric-field amplitude and not a coherent Rabi drive.  In
//   rate-equation optical pumping one may identify k_opt(t)=sigma I(t)/(hbar w)
//   after optical coherence has been eliminated.  An NV-centre rate-equation
//   treatment explicitly uses a Gaussian temporal optical-transition rate:
//       DOI: 10.1038/ncomms14000
//
//   If coherent ground/excited-state optical superpositions are required, the
//   appropriate model is an enlarged Hilbert space with a time-dependent
//   Hamiltonian/Rabi term; that physics is deliberately outside TimeProfile.
//
// RATE-AREA NORMALIZATION
//   For one-way whole-support transfer dp/dt=-k(t)p,
//
//       p(+inf)/p(-inf) = exp[-A],  A = integral k(t) dt,
//       f_transferred    = 1-exp[-A].
//
//   The Gaussian full area is k_peak*FWHM*sqrt(pi/(4 ln2)).  The helper
//   PeakForTransferredFraction implements the inverse relation exactly; this
//   avoids treating a user-specified pulse fraction as if it were itself a rate.
//
// NUMERICAL CONTRACT
//   Objects are immutable after construction.  Integral() is analytical for
//   constant/Gaussian/rectangular profiles and exact piecewise-trapezoidal for
//   the piecewise-linear trajectory definition. Unlike mutable
//   Transition trajectories, evaluating a TimeProfile has no side effects.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "TimeProfile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace SpinAPI
{
    namespace
    {
        constexpr double kLn2 = 0.693147180559945309417232121458176568;
        double RequireFiniteNonNegative(double value, const char *name)
        {
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument(std::string(name) + " must be finite and non-negative");
            return value;
        }
    }

    ConstantTimeProfile::ConstantTimeProfile(double _value)
        : value(RequireFiniteNonNegative(_value, "constant profile value")) {}

    double ConstantTimeProfile::Value(double) const { return value; }

    double ConstantTimeProfile::Integral(double _start, double _end) const
    {
        if (!std::isfinite(_start) || !std::isfinite(_end))
            throw std::invalid_argument("constant profile integration bounds must be finite");
        return value * (_end - _start);
    }

    std::string ConstantTimeProfile::Description() const
    {
        std::ostringstream out; out << "constant(" << value << ")"; return out.str();
    }

    GaussianTimeProfile::GaussianTimeProfile(double _center, double _fwhm, double _peak)
        : center(_center), fwhm(_fwhm), peak(_peak)
    {
        if (!std::isfinite(center))
            throw std::invalid_argument("Gaussian center must be finite");
        if (!std::isfinite(fwhm) || !(fwhm > 0.0))
            throw std::invalid_argument("Gaussian FWHM must be finite and positive");
        RequireFiniteNonNegative(peak, "Gaussian peak rate");
    }

    double GaussianTimeProfile::Value(double _time) const
    {
        if (!std::isfinite(_time))
            throw std::invalid_argument("Gaussian profile time must be finite");
        const double x = (_time - center) / fwhm;
        return peak * std::exp(-4.0 * kLn2 * x * x);
    }

    double GaussianTimeProfile::Integral(double _start, double _end) const
    {
        if (!std::isfinite(_start) || !std::isfinite(_end))
            throw std::invalid_argument("Gaussian integration bounds must be finite");
        if (_start == _end) return 0.0;
        const double sign = _end >= _start ? 1.0 : -1.0;
        const double lo = sign > 0.0 ? _start : _end;
        const double hi = sign > 0.0 ? _end : _start;
        const double a = 2.0 * std::sqrt(kLn2) / fwhm;
        const double prefactor = peak * fwhm * std::sqrt(M_PI) / (4.0 * std::sqrt(kLn2));
        return sign * prefactor * (std::erf(a * (hi - center)) - std::erf(a * (lo - center)));
    }

    double GaussianTimeProfile::FullArea() const
    {
        return peak * fwhm * std::sqrt(M_PI / (4.0 * kLn2));
    }

    double GaussianTimeProfile::PeakForTransferredFraction(double _fraction, double _fwhm)
    {
        if (!std::isfinite(_fraction) || _fraction < 0.0 || _fraction >= 1.0)
            throw std::invalid_argument("Gaussian transferred fraction must satisfy 0 <= f < 1");
        if (!std::isfinite(_fwhm) || !(_fwhm > 0.0))
            throw std::invalid_argument("Gaussian FWHM must be finite and positive");
        if (_fraction == 0.0) return 0.0;

        // For dp/dt=-k(t)p: p(+inf)/p(-inf)=exp[-A_k].  Therefore the
        // transferred fraction is f=1-exp(-A_k), A_k=-ln(1-f).  For the
        // Gaussian intensity/rate envelope used here,
        // A_k=k_peak*FWHM*sqrt(pi/(4 ln 2)).
        return -std::log1p(-_fraction) /
            (_fwhm * std::sqrt(M_PI / (4.0 * kLn2)));
    }

    std::string GaussianTimeProfile::Description() const
    {
        std::ostringstream out;
        out << "gaussian(center=" << center << ",fwhm=" << fwhm << ",peak=" << peak << ")";
        return out.str();
    }

    RectangularTimeProfile::RectangularTimeProfile(double _start, double _end, double _value)
        : start(_start), end(_end), value(_value)
    {
        if (!std::isfinite(start) || !std::isfinite(end) || !(end >= start))
            throw std::invalid_argument("rectangular profile requires finite end >= start");
        RequireFiniteNonNegative(value, "rectangular profile value");
    }

    double RectangularTimeProfile::Value(double _time) const
    {
        if (!std::isfinite(_time))
            throw std::invalid_argument("rectangular profile time must be finite");
        return (_time >= start && _time <= end) ? value : 0.0;
    }

    double RectangularTimeProfile::Integral(double _start, double _end) const
    {
        if (!std::isfinite(_start) || !std::isfinite(_end))
            throw std::invalid_argument("rectangular integration bounds must be finite");
        if (_start == _end) return 0.0;
        const double sign = _end >= _start ? 1.0 : -1.0;
        const double lo = sign > 0.0 ? _start : _end;
        const double hi = sign > 0.0 ? _end : _start;
        const double overlap = std::max(0.0, std::min(hi, end) - std::max(lo, start));
        return sign * value * overlap;
    }

    std::string RectangularTimeProfile::Description() const
    {
        std::ostringstream out;
        out << "rectangular(start=" << start << ",end=" << end << ",value=" << value << ")";
        return out.str();
    }

    TrajectoryTimeProfile::TrajectoryTimeProfile(std::vector<double> _times, std::vector<double> _values)
        : times(std::move(_times)), values(std::move(_values))
    {
        if (times.empty() || times.size() != values.size())
            throw std::invalid_argument("trajectory profile requires equally sized non-empty time/value lists");
        for (size_t i = 0; i < times.size(); ++i)
        {
            if (!std::isfinite(times[i]) || !std::isfinite(values[i]) || values[i] < 0.0)
                throw std::invalid_argument("trajectory times/rates must be finite and rates non-negative");
            if (i > 0 && !(times[i] > times[i - 1]))
                throw std::invalid_argument("trajectory times must be strictly increasing");
        }
    }

    double TrajectoryTimeProfile::Value(double _time) const
    {
        if (!std::isfinite(_time))
            throw std::invalid_argument("trajectory profile time must be finite");
        if (_time <= times.front()) return values.front();
        if (_time >= times.back()) return values.back();
        const auto upper = std::upper_bound(times.begin(), times.end(), _time);
        const size_t hi = static_cast<size_t>(upper - times.begin());
        const size_t lo = hi - 1;
        const double f = (_time - times[lo]) / (times[hi] - times[lo]);
        return values[lo] + f * (values[hi] - values[lo]);
    }

    double TrajectoryTimeProfile::Integral(double _start, double _end) const
    {
        if (!std::isfinite(_start) || !std::isfinite(_end))
            throw std::invalid_argument("trajectory integration bounds must be finite");
        if (_start == _end) return 0.0;
        const double sign = _end >= _start ? 1.0 : -1.0;
        const double lo = sign > 0.0 ? _start : _end;
        const double hi = sign > 0.0 ? _end : _start;

        std::vector<double> knots{lo};
        for (double t : times) if (t > lo && t < hi) knots.push_back(t);
        knots.push_back(hi);
        double area = 0.0;
        for (size_t i = 1; i < knots.size(); ++i)
            area += 0.5 * (Value(knots[i - 1]) + Value(knots[i])) * (knots[i] - knots[i - 1]);
        return sign * area;
    }

    std::string TrajectoryTimeProfile::Description() const
    {
        std::ostringstream out; out << "trajectory(points=" << times.size() << ")"; return out.str();
    }
}
