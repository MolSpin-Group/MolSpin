/////////////////////////////////////////////////////////////////////////
// MultiSSOrientationSampler implementation (RunSection::General::MultiSS)
// ------------------
// Thin MultiSS adapter over the shared GeneralOrientationSampler. Every
// manifold receives the same physical crystallite orientation.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "MultiSSOrientationSampler.h"
#include "../GeneralOrientationSampler.h"

namespace RunSection::General::MultiSS
{
    namespace
    {
        ::RunSection::General::GeneralOrientationMode ConvertMode(MultiSSOrientationMode mode)
        {
            using G = ::RunSection::General::GeneralOrientationMode;
            switch (mode)
            {
            case MultiSSOrientationMode::Identity: return G::Identity;
            case MultiSSOrientationMode::Explicit: return G::Explicit;
            case MultiSSOrientationMode::Powder2D: return G::Powder2D;
            case MultiSSOrientationMode::PowderSO3: return G::PowderSO3;
            }
            return G::Identity;
        }
    }

    bool MultiSSOrientationSampler::Build(const MultiSSExecutionPlan &plan,
        std::vector<MultiSSOrientation> &out,std::ostream &log,std::string &error)
    {
        ::RunSection::General::GeneralOrientationRequest request;
        request.mode=ConvertMode(plan.orientation);
        request.gridType=plan.powderGridType;
        request.domain=plan.powderDomain;
        request.powderPoints=plan.powderPoints;
        request.powderGridSize=plan.powderGridSize;
        request.powderSymmetry=plan.powderSymmetry;
        request.extraAnglePoints=plan.powderGammaPoints;
        request.extraAngleOffset=plan.powderGammaOffset;
        request.explicitAlpha=plan.explicitAlpha;
        request.explicitBeta=plan.explicitBeta;
        request.explicitGamma=plan.explicitGamma;
        request.explicitWeight=plan.explicitWeight;
        request.owner="MultiSSGeneral";

        std::vector<::RunSection::General::GeneralOrientation> common;
        if(!::RunSection::General::GeneralOrientationSampler::Build(request,common,log,error))
            return false;

        out.clear();
        out.reserve(common.size());
        for(const auto &source:common)
        {
            MultiSSOrientation target;
            target.alpha=source.alpha; target.beta=source.beta; target.gamma=source.gamma;
            target.weight=source.weight; target.frameToLab=source.frameToLab;
            out.push_back(std::move(target));
        }
        return true;
    }
}
