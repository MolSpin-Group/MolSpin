/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceExecutionPlan.h"
#include "ObjectParser.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <ostream>

namespace RunSection::General::Resonance
{
    std::string ResonanceLowercase(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }
    bool ResolveResonanceExecutionPlan(const MSDParser::ObjectParser &properties,
                                       ResonanceExecutionPlan &plan, std::ostream &log, std::string &error)
    {
        plan = ResonanceExecutionPlan();
        error.clear();
        bool sweepCacheExact = true, sweepCacheResfields = false, sweepCacheRefinedRoots = false,
             sweepCachePowderMesh = false;

        bool hasFrequency = properties.Get("mwfrequency", plan.mwFrequencyGHz);
        if (!hasFrequency)
            hasFrequency = properties.Get("frequency", plan.mwFrequencyGHz);
        if (!hasFrequency)
        {
            log << "Failed to obtain mwfrequency/frequency. Using frequency = 0 by default." << std::endl;
        }

        if (!properties.Get("linewidth", plan.linewidth_mT))
        {
            log << "Failed to obtain linewidth. Using linewidth = 0 by default." << std::endl;
            plan.linewidth_mT = 0.0;
        }

        if (properties.Get("lineshape", plan.lineshape))
        {
            plan.lineshape = ResonanceLowercase(plan.lineshape);
        }
        else
        {
            plan.lineshape = "gaussian";
        }

        // Harmonic post-processing models field-modulated detection after the
        // absorption spectrum has been assembled on the sweep cache.
        if (!properties.Get("harmonic", plan.detectionHarmonic) &&
            !properties.Get("detectionharmonic", plan.detectionHarmonic) &&
            !properties.Get("detection_harmonic", plan.detectionHarmonic))
        {
            plan.detectionHarmonic = 0;
        }
        if (plan.detectionHarmonic < 0)
        {
            log << "Negative harmonic values are not supported here. Using harmonic = 0." << std::endl;
            plan.detectionHarmonic = 0;
        }
        if (plan.detectionHarmonic > 2)
        {
            log << "Only harmonic = 0, 1, or 2 is supported. Using harmonic = 2." << std::endl;
            plan.detectionHarmonic = 2;
        }

        if (!properties.Get("modamp", plan.modulationAmplitude_mT) &&
            !properties.Get("modulationamplitude", plan.modulationAmplitude_mT) &&
            !properties.Get("modulation_amplitude", plan.modulationAmplitude_mT) &&
            !properties.Get("fieldmodulation", plan.modulationAmplitude_mT))
        {
            plan.modulationAmplitude_mT = 0.0;
        }
        if (plan.modulationAmplitude_mT < 0.0)
        {
            log << "Negative modulation amplitudes are not supported. Using modulation amplitude = 0 mT."
                << std::endl;
            plan.modulationAmplitude_mT = 0.0;
        }

        if (!properties.Get("powdersamplingpoints", plan.powdersamplingpoints))
        {
            plan.powdersamplingpoints = 0;
        }

        if (!properties.Get("sweepcache", plan.useSweepCache) &&
            !properties.Get("cache_sweep", plan.useSweepCache) &&
            !properties.Get("sweep_cache", plan.useSweepCache))
        {
            plan.useSweepCache = true;
        }

        if (plan.detectionHarmonic > 0 && !plan.useSweepCache)
        {
            log << "Detection harmonic post-processing requires the sweep cache. Enabling sweepcache=true."
                << std::endl;
            plan.useSweepCache = true;
        }

        std::string sweepCacheMode;
        if (properties.Get("sweepcachemode", sweepCacheMode) ||
            properties.Get("sweep_cache_mode", sweepCacheMode) ||
            properties.Get("cache_sweep_mode", sweepCacheMode))
        {
            sweepCacheMode = ResonanceLowercase(sweepCacheMode);
            if (sweepCacheMode == "powdermesh")
            {
                sweepCacheExact = false;
                sweepCacheResfields = false;
                sweepCacheRefinedRoots = false;
                sweepCachePowderMesh = true;
            }
            else if (sweepCacheMode == "refinedroots")
            {
                sweepCacheExact = false;
                sweepCacheResfields = false;
                sweepCacheRefinedRoots = true;
            }
            else if (sweepCacheMode == "exact" || sweepCacheMode == "direct" || sweepCacheMode == "matrix")
            {
                sweepCacheExact = true;
                sweepCacheResfields = false;
            }
            else if (sweepCacheMode == "resonanceprojection" || sweepCacheMode == "projection" ||
                     sweepCacheMode == "projectedresfields" || sweepCacheMode == "resfields" ||
                     sweepCacheMode == "resfield")
            {
                sweepCacheExact = false;
                sweepCacheResfields = true;
            }
            else if (sweepCacheMode == "approx" || sweepCacheMode == "approximate" ||
                     sweepCacheMode == "crossing" || sweepCacheMode == "resonance")
            {
                sweepCacheExact = false;
                sweepCacheResfields = false;
            }
            else
            {
                error = "unknown sweepcachemode: " + sweepCacheMode;
                return false;
            }
        }

        if (sweepCacheRefinedRoots &&
            (!plan.useSweepCache || plan.lineshape != "gaussian" || !(plan.linewidth_mT > 0.0)))
        {
            log << "sweepcachemode=refinedroots requires sweepcache=true and a positive Gaussian linewidth."
                << std::endl;
            return false;
        }
        properties.Get("meshcospoints", plan.meshCosPoints);
        properties.Get("meshphipoints", plan.meshPhiPoints);
        properties.Get("meshfieldpoints", plan.meshFieldPoints);
        properties.Get("meshfieldscale", plan.meshFieldScale);
        if (sweepCachePowderMesh &&
            (!plan.useSweepCache || plan.lineshape != "gaussian" || !(plan.linewidth_mT > 0) ||
             plan.meshCosPoints < 3 || plan.meshPhiPoints < 4 || plan.meshFieldPoints < 3 ||
             !(plan.meshFieldScale >= 0) || !std::isfinite(plan.meshFieldScale)))
        {
            log << "powdermesh requires a cached positive Gaussian width and valid mesh dimensions/scale."
                << std::endl;
            return false;
        }

        int resfieldPoints = 0;
        if (properties.Get("resfieldspoints", resfieldPoints) ||
            properties.Get("resfields_points", resfieldPoints) ||
            properties.Get("sweepcachepoints", resfieldPoints) ||
            properties.Get("sweep_cache_points", resfieldPoints))
        {
            if (resfieldPoints >= 2)
                plan.sweepCacheResfieldPoints = resfieldPoints;
            else
                plan.sweepCacheResfieldPoints = 0;
        }

        if (properties.Get("powdergridtype", plan.powderGridType))
        {
            plan.powderGridType = ResonanceLowercase(plan.powderGridType);
        }
        else
        {
            plan.powderGridType = "sophe";
        }
        properties.Get("powdergridsymmetry", plan.powderGridSymmetry);
        if (!properties.Get("powdergridsize", plan.powderGridSize))
        {
            plan.powderGridSize = 0;
        }

        if (!properties.Get("powdergammapoints", plan.powderGammaPoints))
        {
            properties.Get("powdergammastps", plan.powderGammaPoints);
        }
        if (plan.powderGammaPoints < 1)
        {
            plan.powderGammaPoints = 1;
        }

        properties.Get("powderfullsphere", plan.powderFullSphere);
        properties.Get("fulltensorrotation", plan.fullTensorRotation);
        properties.Get("mzblocks", plan.useMzBlocks);

        plan.resonanceSolverMode = "exact";
        std::string resonanceSolver;
        if (properties.Get("solver", resonanceSolver) || properties.Get("resonancesolver", resonanceSolver) ||
            properties.Get("resonance_solver", resonanceSolver))
        {
            resonanceSolver = ResonanceLowercase(resonanceSolver);
            if (resonanceSolver == "exact")
                plan.resonanceSolverMode = "exact";
            else if (resonanceSolver == "hybrid")
                plan.resonanceSolverMode = "hybrid";
            else if (resonanceSolver == "auto")
            {
                log << "solver=auto is not yet qualified for ResonanceGeneral; select exact or explicit "
                       "hybrid."
                    << std::endl;
                return false;
            }
            else
            {
                log << "Unknown resonance solver \"" << resonanceSolver << "\"." << std::endl;
                return false;
            }
        }

        plan.hybridPerturbativeNucleusNames.clear();
        if (!properties.GetList("perturbativenuclei", plan.hybridPerturbativeNucleusNames, ',') &&
            !properties.GetList("hybridperturbativenuclei", plan.hybridPerturbativeNucleusNames, ','))
        {
            properties.GetList("hybrid_perturbative_nuclei", plan.hybridPerturbativeNucleusNames, ',');
        }

        if (!properties.Get("hybridfieldstep", plan.hybridFieldStepT))
            properties.Get("hybrid_field_step", plan.hybridFieldStepT);
        if (!properties.Get("hybridminimumcorestateoverlap", plan.hybridMinimumCoreStateOverlap))
            properties.Get("hybrid_minimum_core_state_overlap", plan.hybridMinimumCoreStateOverlap);
        if (!properties.Get("hybridminimumnuclearstateoverlap", plan.hybridMinimumNuclearStateOverlap))
            properties.Get("hybrid_minimum_nuclear_state_overlap", plan.hybridMinimumNuclearStateOverlap);
        if (!properties.Get("hybridjacobianreltol", plan.hybridJacobianRelativeTolerance))
            properties.Get("hybrid_jacobian_relative_tolerance", plan.hybridJacobianRelativeTolerance);
        if (!properties.Get("hybridjacobianabstol", plan.hybridJacobianAbsoluteTolerance))
            properties.Get("hybrid_jacobian_absolute_tolerance", plan.hybridJacobianAbsoluteTolerance);
        if (!properties.Get("hybridoverlapthreshold", plan.hybridOverlapThreshold))
            properties.Get("hybrid_overlap_threshold", plan.hybridOverlapThreshold);
        if (!properties.Get("hybridminimumcumulativeoverlapweight",
                            plan.hybridMinimumCumulativeOverlapWeight))
            properties.Get("hybrid_minimum_cumulative_overlap_weight",
                           plan.hybridMinimumCumulativeOverlapWeight);

        int hybridMaximumComponents = 0;
        if (properties.Get("hybridmaximumcomponentspercoretransition", hybridMaximumComponents) ||
            properties.Get("hybrid_maximum_components_per_core_transition", hybridMaximumComponents))
        {
            if (hybridMaximumComponents < 0)
            {
                log << "Hybrid maximum component count must be non-negative." << std::endl;
                return false;
            }
            plan.hybridMaximumComponentsPerCoreTransition = static_cast<std::size_t>(hybridMaximumComponents);
        }

        if (plan.resonanceSolverMode == "hybrid")
        {
            if (plan.hybridPerturbativeNucleusNames.empty())
            {
                log << "solver=hybrid requires an explicit perturbativenuclei list." << std::endl;
                return false;
            }
            if (plan.useSweepCache)
            {
                log << "solver=hybrid R2K-B requires sweepcache=false; hybrid cache semantics are not yet "
                       "qualified."
                    << std::endl;
                return false;
            }
            if (plan.detectionHarmonic != 0)
            {
                log << "solver=hybrid R2K-B supports harmonic=0 only." << std::endl;
                return false;
            }
            if (!std::isfinite(plan.hybridFieldStepT) || plan.hybridFieldStepT <= 0.0 ||
                !std::isfinite(plan.hybridMinimumCoreStateOverlap) ||
                plan.hybridMinimumCoreStateOverlap <= 0.0 || plan.hybridMinimumCoreStateOverlap > 1.0 ||
                !std::isfinite(plan.hybridMinimumNuclearStateOverlap) ||
                plan.hybridMinimumNuclearStateOverlap <= 0.0 || plan.hybridMinimumNuclearStateOverlap > 1.0 ||
                !std::isfinite(plan.hybridJacobianRelativeTolerance) ||
                plan.hybridJacobianRelativeTolerance < 0.0 ||
                !std::isfinite(plan.hybridJacobianAbsoluteTolerance) ||
                plan.hybridJacobianAbsoluteTolerance < 0.0 || !std::isfinite(plan.hybridOverlapThreshold) ||
                plan.hybridOverlapThreshold < 0.0 || plan.hybridOverlapThreshold > 1.0 ||
                !std::isfinite(plan.hybridMinimumCumulativeOverlapWeight) ||
                plan.hybridMinimumCumulativeOverlapWeight < 0.0 ||
                plan.hybridMinimumCumulativeOverlapWeight > 1.0)
            {
                log << "Invalid explicit hybrid resonance numerical controls." << std::endl;
                return false;
            }
            log << "General Resonance solver = explicit hybrid nuclear treatment." << std::endl;
        }

        log << "Full-Hamiltonian resonance detection model: mwfrequency = " << plan.mwFrequencyGHz
            << " GHz, linewidth = " << plan.linewidth_mT << " mT, lineshape = " << plan.lineshape
            << ", harmonic = " << plan.detectionHarmonic
            << ", modulation amplitude = " << plan.modulationAmplitude_mT << " mT." << std::endl;
        log << "Full-Hamiltonian resonance powder grid request: type = " << plan.powderGridType
            << ", symmetry = " << plan.powderGridSymmetry
            << ", sampling points = " << plan.powdersamplingpoints
            << ", gamma points = " << plan.powderGammaPoints
            << ", full sphere = " << (plan.powderFullSphere ? "true" : "false") << "." << std::endl;

        plan.detectSpinNames.clear();
        properties.GetList("detectspins", plan.detectSpinNames, ',');

        properties.Get("fieldinteraction", plan.fieldInteractionName);
        properties.Get("enforce_zeeman_sync", plan.enforceZeemanSync);
        properties.Get("enforcezeemansync", plan.enforceZeemanSync);
        properties.Get("initialstate", plan.initialStateName);

        if (properties.GetList("hamiltonianh0list", plan.hamiltonianH0list, ','))
        {
            log << "HamiltonianH0list = [";
            for (size_t j = 0; j < plan.hamiltonianH0list.size(); j++)
            {
                log << plan.hamiltonianH0list[j];
                if (j < plan.hamiltonianH0list.size() - 1)
                    log << ", ";
            }
            log << "]" << std::endl;
        }

        plan.sweepMode = sweepCachePowderMesh     ? ResonanceSweepMode::PowderMesh
                         : sweepCacheRefinedRoots ? ResonanceSweepMode::RefinedRoots
                         : sweepCacheExact        ? ResonanceSweepMode::Exact
                         : sweepCacheResfields    ? ResonanceSweepMode::Projection
                                                  : ResonanceSweepMode::Approximate;
        properties.Get("meshclusteraxes", plan.meshClusterAxes);
        properties.Get("meshrawfile", plan.meshRawFile);
        plan.diagnostics = OrientationDiagnostics::FromProperties(properties);
        if (!std::isfinite(plan.mwFrequencyGHz) || plan.mwFrequencyGHz <= 0 ||
            !std::isfinite(plan.linewidth_mT) || plan.linewidth_mT < 0 ||
            !std::isfinite(plan.modulationAmplitude_mT) ||
            (plan.lineshape != "gaussian" && plan.lineshape != "lorentzian"))
        {
            error =
                "resonance requires a positive finite microwave frequency, a non-negative finite width and a "
                "supported lineshape";
            return false;
        }
        if (plan.sweepMode == ResonanceSweepMode::PowderMesh && plan.meshClusterAxes &&
            ((plan.meshCosPoints - 1) % 2 || plan.meshPhiPoints % 4))
        {
            error = "axis-clustered mesh requires odd meshcospoints and meshphipoints divisible by four";
            return false;
        }
        log << "--- ResonanceGeneral resolved calculation ---\n"
            << "solver=" << plan.resonanceSolverMode << "; cached=" << plan.useSweepCache << std::endl;
        return true;
    }
} // namespace RunSection::General::Resonance
