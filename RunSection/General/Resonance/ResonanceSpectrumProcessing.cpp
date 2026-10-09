/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceSpectrumProcessing.h"
#include "ResonanceLineshape.h"
#include <algorithm>
#include <cmath>
#include <ostream>

namespace RunSection::General::Resonance
{
    void ResonanceSpectrumProcessing::Accumulate(SpectrumPoint &total, const SpectrumPoint &point,
                                                 double weight)
    {
        if (total.channels.empty())
            total.channels.resize(point.channels.size());
        total.totalX += weight * point.totalX;
        total.totalY += weight * point.totalY;
        total.totalPerpendicular += weight * point.totalPerpendicular;
        total.crossX += weight * point.crossX;
        total.crossY += weight * point.crossY;
        for (size_t i = 0; i < point.channels.size(); ++i)
        {
            auto &a = total.channels[i];
            const auto &b = point.channels[i];
            a.x += weight * b.x;
            a.y += weight * b.y;
            a.perpendicular += weight * b.perpendicular;
            a.plus += weight * b.plus;
            a.minus += weight * b.minus;
        }
    }

    ResonanceSample ResonanceSpectrumProcessing::Sample(const ResonanceSpectrum &s, size_t i)
    {
        ResonanceSample r;
        r.field_mT = s.field_mT.at(i);
        r.spin_names = s.spin_names;
        r.point.totalX = s.total_x.at(i);
        r.point.totalY = s.total_y.at(i);
        r.point.totalPerpendicular = s.total_perp.at(i);
        r.point.crossX = s.cross_x.at(i);
        r.point.crossY = s.cross_y.at(i);
        r.point.channels.resize(s.spin_names.size());
        for (size_t k = 0; k < s.spin_names.size(); ++k)
        {
            auto &c = r.point.channels[k];
            c.x = s.spin_x.at(k).at(i);
            c.y = s.spin_y.at(k).at(i);
            c.perpendicular = s.spin_perp.at(k).at(i);
            c.plus = s.spin_p.at(k).at(i);
            c.minus = s.spin_m.at(k).at(i);
        }
        return r;
    }

    void ResonanceSpectrumProcessing::WriteHeader(const std::string &system,
                                                  const std::vector<std::string> &spins, std::ostream &out)
    {
        for (const auto &label : {"Field_mT", "Total_x", "Total_y", "Total_perp", "Cross_x", "Cross_y"})
            out << system << "." << label << " ";
        for (const auto &spin : spins)
            for (const auto &label : {"_x", "_y", "_perp", "_p", "_m"})
                out << system << "." << spin << label << " ";
    }

    void ResonanceSpectrumProcessing::WriteSample(const ResonanceSample &r, std::ostream &out)
    {
        const auto &p = r.point;
        out << r.field_mT << " " << p.totalX << " " << p.totalY << " " << p.totalPerpendicular << " "
            << p.crossX << " " << p.crossY << " ";
        for (const auto &c : p.channels)
            out << c.x << " " << c.y << " " << c.perpendicular << " " << c.plus << " " << c.minus << " ";
    }
    double ResonanceSpectrumProcessing::LineshapeValue(const ResonanceExecutionPlan &plan, double _delta,
                                                       double _fwhm)
    {
        return ResonanceLineshape::Evaluate(
            plan.lineshape == "lorentzian" ? Lineshape::Lorentzian : Lineshape::Gaussian, _delta, _fwhm);
    }
    std::vector<double> ResonanceSpectrumProcessing::ApplyFieldHarmonic(const ResonanceExecutionPlan &plan,
                                                                        const std::vector<double> &_field_mT,
                                                                        const std::vector<double> &_channel)
    {
        if (plan.detectionHarmonic <= 0 || _field_mT.size() != _channel.size() || _field_mT.size() < 3)
            return _channel;

        double meanStep = 0.0;
        size_t stepCount = 0;
        for (size_t i = 1; i < _field_mT.size(); ++i)
        {
            const double dx = std::abs(_field_mT[i] - _field_mT[i - 1]);
            if (std::isfinite(dx) && dx > 0.0)
            {
                meanStep += dx;
                ++stepCount;
            }
        }
        if (stepCount == 0)
            return _channel;
        meanStep /= static_cast<double>(stepCount);

        const double span = (plan.modulationAmplitude_mT > 0.0) ? plan.modulationAmplitude_mT : meanStep;
        const double h = (plan.modulationAmplitude_mT > 0.0) ? (0.5 * span) : span;
        if (!std::isfinite(h) || h <= 0.0)
            return _channel;

        const bool ascending = (_field_mT.back() >= _field_mT.front());
        auto interp = [&](double x) -> double
        {
            if (ascending)
            {
                if (x <= _field_mT.front())
                    return _channel.front();
                if (x >= _field_mT.back())
                    return _channel.back();
                auto it = std::lower_bound(_field_mT.begin(), _field_mT.end(), x);
                const size_t hi = static_cast<size_t>(std::distance(_field_mT.begin(), it));
                const size_t lo = hi - 1;
                const double denom = _field_mT[hi] - _field_mT[lo];
                if (std::abs(denom) <= 0.0)
                    return _channel[lo];
                const double t = (x - _field_mT[lo]) / denom;
                return (1.0 - t) * _channel[lo] + t * _channel[hi];
            }

            if (x >= _field_mT.front())
                return _channel.front();
            if (x <= _field_mT.back())
                return _channel.back();
            auto it = std::lower_bound(_field_mT.begin(), _field_mT.end(), x, std::greater<double>());
            const size_t hi = static_cast<size_t>(std::distance(_field_mT.begin(), it));
            const size_t lo = hi - 1;
            const double denom = _field_mT[hi] - _field_mT[lo];
            if (std::abs(denom) <= 0.0)
                return _channel[lo];
            const double t = (x - _field_mT[lo]) / denom;
            return (1.0 - t) * _channel[lo] + t * _channel[hi];
        };

        std::vector<double> out(_channel.size(), 0.0);
        for (size_t i = 0; i < _channel.size(); ++i)
        {
            const double x = _field_mT[i];
            const double ym = interp(x - h);
            const double y0 = _channel[i];
            const double yp = interp(x + h);

            if (plan.detectionHarmonic == 1)
                out[i] = (yp - ym) / (2.0 * h);
            else
                out[i] = (yp - 2.0 * y0 + ym) / (h * h);
        }

        return out;
    }
    void ResonanceSpectrumProcessing::ApplyDetectionHarmonic(const ResonanceExecutionPlan &plan,
                                                             ResonanceSpectrum &_cache)
    {
        if (plan.detectionHarmonic <= 0)
            return;

        auto apply = [&](std::vector<double> &channel)
        { channel = ApplyFieldHarmonic(plan, _cache.field_mT, channel); };

        apply(_cache.total_x);
        apply(_cache.total_y);
        apply(_cache.total_perp);
        apply(_cache.cross_x);
        apply(_cache.cross_y);
        for (size_t i = 0; i < _cache.spin_names.size(); ++i)
        {
            apply(_cache.spin_x[i]);
            apply(_cache.spin_y[i]);
            apply(_cache.spin_perp[i]);
            apply(_cache.spin_p[i]);
            apply(_cache.spin_m[i]);
        }
    }
} // namespace RunSection::General::Resonance
