/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Validated resonance policy; no task ownership or numerical execution.
#ifndef MOLSPIN_RESONANCEEXECUTIONPLAN_H
#define MOLSPIN_RESONANCEEXECUTIONPLAN_H
#include <string>
#include <vector>
#include <iosfwd>
#include "MSDParserfwd.h"
#include "ResonanceDiagnostics.h"

namespace RunSection::General::Resonance
{

    enum class ResonanceSweepMode
    {
        Exact,
        Approximate,
        Projection,
        RefinedRoots,
        PowderMesh
    };
    std::string ResonanceLowercase(std::string value);
    struct ResonanceExecutionPlan
    {
        double mwFrequencyGHz = 0.0;
        double linewidth_mT = 0.0;
        std::string lineshape = "gaussian";

        int detectionHarmonic = 0;
        double modulationAmplitude_mT = 0.0;

        std::string powderGridType = "sophe";
        std::string powderGridSymmetry = "auto";
        int powderGridSize = 0;
        int powdersamplingpoints = 0;
        int powderGammaPoints = 1;
        bool powderFullSphere = true;
        bool fullTensorRotation = true;
        bool useMzBlocks = true;
        bool useSweepCache = true;
        int meshCosPoints = 33;
        int meshPhiPoints = 64;
        int meshFieldPoints = 129;
        double meshFieldScale = 0.25;
        int sweepCacheResfieldPoints = 0;
        std::vector<std::string> detectSpinNames;
        std::string fieldInteractionName = "";
        bool enforceZeemanSync = false;
        std::string initialStateName = "";
        std::vector<std::string> hamiltonianH0list;

        // General Resonance backend policy. Exact remains the historical default.
        // The first hybrid task route is deliberately explicit: the caller names
        // perturbative nuclei and unsupported state/cache semantics fail closed.
        std::string resonanceSolverMode = "exact";
        std::vector<std::string> hybridPerturbativeNucleusNames;
        double hybridFieldStepT = 1.0e-4;
        double hybridMinimumCoreStateOverlap = 0.90;
        double hybridMinimumNuclearStateOverlap = 0.90;
        double hybridJacobianRelativeTolerance = 1.0e-4;
        double hybridJacobianAbsoluteTolerance = 1.0e-5;
        double hybridOverlapThreshold = 1.0e-14;
        double hybridMinimumCumulativeOverlapWeight = 0.0;
        std::size_t hybridMaximumComponentsPerCoreTransition = 0;

        ResonanceSweepMode sweepMode = ResonanceSweepMode::Exact;
        bool meshClusterAxes = false;
        std::string meshRawFile;
        OrientationDiagnostics diagnostics;
        bool RequiresSweep() const
        {
            return sweepMode == ResonanceSweepMode::RefinedRoots ||
                   sweepMode == ResonanceSweepMode::PowderMesh;
        }
    };
    bool ResolveResonanceExecutionPlan(const MSDParser::ObjectParser &properties,
                                       ResonanceExecutionPlan &plan, std::ostream &log, std::string &error);

} // namespace RunSection::General::Resonance
#endif
