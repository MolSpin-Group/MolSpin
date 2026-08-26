/////////////////////////////////////////////////////////////////////////
// GeneralOrientationSampler implementation (RunSection::General)
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "GeneralOrientationSampler.h"

#include <cmath>
#include <numeric>

namespace RunSection::General
{
    namespace
    {
        double WrapAngle(double value)
        {
            const double period = 2.0 * arma::datum::pi;
            value = std::fmod(value, period);
            if (value < 0.0) value += period;
            return value;
        }

        bool Append(double alpha, double beta, double gamma, double weight,
            std::vector<GeneralOrientation> &out)
        {
            if (!std::isfinite(alpha) || !std::isfinite(beta) ||
                !std::isfinite(gamma) || !std::isfinite(weight))
                return false;
            GeneralOrientation item;
            item.alpha = alpha;
            item.beta = beta;
            item.gamma = gamma;
            item.weight = weight;
            if (!SpinAPI::CreateZYZRotationMatrix(alpha,beta,gamma,item.frameToLab))
                return false;
            out.push_back(std::move(item));
            return true;
        }

        bool BuildBaseGrid(const GeneralOrientationRequest &request,
            SpinAPI::PowderGrid &base, std::ostream &log, std::string &error)
        {
            bool ok = false;
            switch (request.gridType)
            {
            case SpinAPI::PowderGridType::Uniform:
                ok = SpinAPI::CreateUniformPowderGrid(request.powderPoints,
                    request.domain, base);
                if (ok)
                    log << request.owner << " uses SpinAPI uniform powder grid with "
                        << base.size() << " theta/phi orientations over the "
                        << (request.domain == SpinAPI::PowderGridDomain::FullSphere
                            ? "full sphere." : "upper hemisphere.") << std::endl;
                break;
            case SpinAPI::PowderGridType::Sophe:
                ok = SpinAPI::CreateSophePowderGrid(request.powderGridSize,
                    request.powderSymmetry, base);
                if (ok)
                    log << request.owner << " uses SpinAPI SOPHE powder grid with grid size "
                        << request.powderGridSize << ", symmetry " << request.powderSymmetry
                        << ", and " << base.size() << " theta/phi orientations." << std::endl;
                break;
            case SpinAPI::PowderGridType::Octant:
                ok = SpinAPI::CreateOctantPowderGrid(request.powderPoints, base);
                if (ok)
                    log << request.owner << " uses SpinAPI historical octant powder grid with "
                        << base.size() << " theta/phi orientations." << std::endl;
                break;
            }
            if (!ok || base.empty())
            {
                error = "failed to construct the " + request.owner + " powder grid";
                return false;
            }
            return true;
        }

        bool NormalizeGeneratedWeights(std::vector<GeneralOrientation> &out,
            const std::string &owner, std::ostream &log, std::string &error)
        {
            double sum = 0.0;
            for (const auto &o : out)
            {
                if (!std::isfinite(o.weight) || o.weight < 0.0)
                {
                    error = owner + " generated a non-finite or negative powder weight";
                    return false;
                }
                sum += o.weight;
            }
            if (!(sum > 0.0) || !std::isfinite(sum))
            {
                error = owner + " powder weights do not have a finite positive sum";
                return false;
            }
            for (auto &o : out) o.weight /= sum;
            log << owner << " normalized generated powder weights from raw sum "
                << sum << " to unit ensemble weight." << std::endl;
            return true;
        }
    }

    bool GeneralOrientationSampler::Build(const GeneralOrientationRequest &request,
        std::vector<GeneralOrientation> &out, std::ostream &log, std::string &error)
    {
        out.clear();
        error.clear();

        if (request.mode == GeneralOrientationMode::Identity)
        {
            if (!Append(0.0,0.0,0.0,1.0,out))
            {
                error = "failed to construct identity orientation";
                return false;
            }
            log << request.owner << " orientation sampling = identity." << std::endl;
            return true;
        }

        if (request.mode == GeneralOrientationMode::Explicit)
        {
            if (!Append(request.explicitAlpha, request.explicitBeta,
                request.explicitGamma, request.explicitWeight, out))
            {
                error = "failed to construct explicit " + request.owner + " orientation";
                return false;
            }
            log << request.owner << " orientation sampling = explicit ZYZ orientation; "
                << "supplied weight is preserved." << std::endl;
            return true;
        }

        SpinAPI::PowderGrid base;
        if (!BuildBaseGrid(request,base,log,error)) return false;

        if (request.mode == GeneralOrientationMode::Powder2D)
        {
            for (const auto &p : base)
                if (!Append(0.0,p.theta,p.phi,p.weight,out))
                {
                    error = "failed to construct a theta/phi powder rotation";
                    return false;
                }
        }
        else
        {
            if (request.extraAnglePoints < 1)
            {
                error = request.owner + " SO(3) sampling requires at least one extra-angle point";
                return false;
            }
            log << request.owner << " samples the third Euler angle with "
                << request.extraAnglePoints << " points." << std::endl;
            for (const auto &p : base)
                for (int k=0;k<request.extraAnglePoints;++k)
                {
                    const double extra = WrapAngle(request.extraAngleOffset +
                        2.0*arma::datum::pi*(static_cast<double>(k)+0.5)/
                        static_cast<double>(request.extraAnglePoints));
                    // Historical MolSpin convention: R(extra, theta, phi).
                    if (!Append(extra,p.theta,p.phi,
                        p.weight/static_cast<double>(request.extraAnglePoints),out))
                    {
                        error = "failed to construct a theta/phi/third-angle powder rotation";
                        return false;
                    }
                }
            log << request.owner << " uses " << out.size()
                << " total SO(3) orientations." << std::endl;
        }

        return NormalizeGeneratedWeights(out,request.owner,log,error);
    }
}
