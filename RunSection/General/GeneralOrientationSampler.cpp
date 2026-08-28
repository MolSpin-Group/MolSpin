/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// Shared orientation and powder-ensemble adapter.
//
// What is done here:
//   - Builds canonical molecular-to-laboratory ZYZ rotations and weights.
//   - Delegates uniform/Fibonacci, SOPHE and octant theta/phi grids to SpinAPI.
//   - Adds the optional third Euler angle and normalizes generated weights once.
//
// Connections to the General framework / SpinAPI:
//   - Called by HSOrientationSampler, SSOrientationSampler and MultiSSOrientationSampler.
//   - The returned frameToLab rotation must be reused by Hamiltonian, state, transition, relaxation and observable builders for one crystallite.
//
// Why this ownership is used:
//   - Powder averaging is an outer ensemble integral, not a propagation method.
//   - Keeping one sampler avoids incompatible Euler-angle conventions between General tasks.
//
// Mathematical / physical references:
//   - Swinbank & Purser, Q. J. R. Meteorol. Soc. 132, 1769-1793 (2006), DOI: 10.1256/qj.05.227.
//
// TODO:
//   - Keep future strain / Schulten-Wolynes Hamiltonian realizations as a second ensemble index; combine them with powder points only in a shared outer ensemble driver.
/////////////////////////////////////////////////////////////////////////
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
                !std::isfinite(gamma) || !std::isfinite(weight) || !(weight > 0.0))
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
                    log << request.owner << " uses the SpinAPI octant powder grid with "
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
                error = "explicit " + request.owner +
                    " orientation requires finite Euler angles and a positive weight";
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
                    // Established MolSpin convention: R(extra, theta, phi).
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
