/////////////////////////////////////////////////////////////////////////
// HSOrientationSampler implementation (RunSection::General::HS)
// ------------------
// Thin HS adapter over the shared GeneralOrientationSampler.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSOrientationSampler.h"
#include "../GeneralOrientationSampler.h"

namespace RunSection::General::HS
{
    namespace
    {
        ::RunSection::General::GeneralOrientationMode ConvertMode(OrientationMode mode)
        {
            using G = ::RunSection::General::GeneralOrientationMode;
            switch (mode)
            {
            case OrientationMode::Identity: return G::Identity;
            case OrientationMode::Explicit: return G::Explicit;
            case OrientationMode::Powder2D: return G::Powder2D;
            case OrientationMode::PowderSO3: return G::PowderSO3;
            }
            return G::Identity;
        }
    }

    bool HSOrientationSampler::Build(const HSExecutionPlan &plan,
        std::vector<HSOrientation> &orientations, std::ostream &log, std::string &error)
    {
        ::RunSection::General::GeneralOrientationRequest request;
        request.mode = ConvertMode(plan.orientation);
        request.gridType = plan.powderGridType;
        request.domain = plan.powderDomain;
        request.powderPoints = plan.powderPoints;
        request.powderGridSize = plan.powderGridSize;
        request.powderSymmetry = plan.powderSymmetry;
        request.extraAnglePoints = plan.powderGammaPoints;
        request.extraAngleOffset = plan.powderGammaOffset;
        // Preserve the established HS explicit-orientation mapping
        // R(powdergamma, theta, phi).
        request.explicitAlpha = plan.powderGammaOffset;
        request.explicitBeta = plan.explicitTheta;
        request.explicitGamma = plan.explicitPhi;
        request.explicitWeight = plan.explicitWeight;
        request.owner = "HSGeneral";

        if (plan.powderDomainAutoExpanded)
            log << "Full-sphere theta/phi sampling was selected automatically because gamma sampling requests a complete SO(3) grid." << std::endl;

        std::vector<::RunSection::General::GeneralOrientation> common;
        if (!::RunSection::General::GeneralOrientationSampler::Build(request,common,log,error))
            return false;

        orientations.clear();
        orientations.reserve(common.size());
        for (const auto &source : common)
        {
            HSOrientation target;
            target.alpha = source.alpha;
            target.beta = source.beta;
            target.gamma = source.gamma;
            target.weight = source.weight;
            target.frameToLab = source.frameToLab;
            orientations.push_back(std::move(target));
        }
        return true;
    }
}
