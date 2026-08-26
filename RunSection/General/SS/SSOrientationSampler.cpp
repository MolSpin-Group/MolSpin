/////////////////////////////////////////////////////////////////////////
// SSOrientationSampler implementation (RunSection::General::SS)
// ------------------
// Thin SS adapter over the shared GeneralOrientationSampler.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "SSOrientationSampler.h"
#include "../GeneralOrientationSampler.h"

namespace RunSection::General::SS
{
    namespace
    {
        ::RunSection::General::GeneralOrientationMode ConvertMode(SSOrientationMode mode)
        {
            using G = ::RunSection::General::GeneralOrientationMode;
            switch (mode)
            {
            case SSOrientationMode::Identity: return G::Identity;
            case SSOrientationMode::Explicit: return G::Explicit;
            case SSOrientationMode::Powder2D: return G::Powder2D;
            case SSOrientationMode::PowderSO3: return G::PowderSO3;
            }
            return G::Identity;
        }
    }

    bool SSOrientationSampler::Build(const SSExecutionPlan &plan,
        std::vector<SSOrientation> &out, std::ostream &log, std::string &error)
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
        request.explicitAlpha = plan.explicitAlpha;
        request.explicitBeta = plan.explicitBeta;
        request.explicitGamma = plan.explicitGamma;
        request.explicitWeight = plan.explicitWeight;
        request.owner = "SSGeneral";

        std::vector<::RunSection::General::GeneralOrientation> common;
        if (!::RunSection::General::GeneralOrientationSampler::Build(request,common,log,error))
            return false;

        out.clear();
        out.reserve(common.size());
        for (const auto &source : common)
        {
            SSOrientation target;
            target.alpha=source.alpha; target.beta=source.beta; target.gamma=source.gamma;
            target.weight=source.weight; target.frameToLab=source.frameToLab;
            out.push_back(std::move(target));
        }
        return true;
    }
}
