/////////////////////////////////////////////////////////////////////////
// General resonance task: parse policy, dispatch services, write one sweep row.
// No eigensolvers, tensor/state rotations, line physics or legacy task calls.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "TaskResonanceGeneral.h"
#include "ResonanceCalculations.h"
#include "ResonanceSystemPreparation.h"
#include "ResonanceFieldSweep.h"
#include "Settings.h"
#include "SpinSystem.h"
#include "Interaction.h"
#include <ostream>

namespace RunSection::General::Resonance
{
    TaskResonanceGeneral::TaskResonanceGeneral(const MSDParser::ObjectParser &properties,
                                               const RunSection &runsection)
        : BasicTask(properties, runsection)
    {
    }

    bool TaskResonanceGeneral::Validate()
    {
        std::string error;
        if (!ResolveResonanceExecutionPlan(*Properties(), plan, Log(), error))
        {
            Log() << "ERROR: Invalid ResonanceGeneral plan. " << error << std::endl;
            return false;
        }
        if (SpinSystems().empty())
        {
            Log() << "ERROR: ResonanceGeneral needs a SpinSystem." << std::endl;
            return false;
        }
        for (const auto &system : SpinSystems())
        {
            std::vector<std::string> names;
            std::vector<SpinAPI::spin_ptr> spins;
            SpinAPI::interaction_ptr field;
            if (!SystemPreparation::ResolveHamiltonian(plan, system, names, error) ||
                !SystemPreparation::ResolveFieldInteraction(plan, system, field) ||
                !SystemPreparation::ResolveDetectionSpins(plan, system, field, spins, names))
            {
                Log() << "ERROR: Invalid resonance system/field/detection selection: " << error << std::endl;
                return false;
            }
        }
        planResolved = true;
        return true;
    }

    bool TaskResonanceGeneral::WriteHeader()
    {
        Data() << "Step Time ";
        WriteStandardOutputHeader(Data());
        for (const auto &system : SpinSystems())
        {
            SpinAPI::interaction_ptr field;
            std::vector<SpinAPI::spin_ptr> spins;
            std::vector<std::string> names;
            if (!SystemPreparation::ResolveFieldInteraction(plan, system, field) ||
                !SystemPreparation::ResolveDetectionSpins(plan, system, field, spins, names))
                return false;
            ResonanceSpectrumProcessing::WriteHeader(system->Name(), names, Data());
        }
        Data() << std::endl;
        return true;
    }

    bool TaskResonanceGeneral::RunLocal()
    {
        if (!planResolved)
            return false;
        const auto settings = RunSettings();
        const auto step = settings->CurrentStep();
        if (step == 1)
        {
            // A new run on the same task must rebuild from the current system.
            spectra.clear();
            if (!WriteHeader())
                return false;
        }
        std::vector<ResonanceSample> row;
        for (const auto &system : SpinSystems())
        {
            ResonanceSample sample;
            std::string error;
            bool ok = false;
            if (plan.resonanceSolverMode == "hybrid")
                ok = ResonanceHybridCalculation::Point(plan, system, sample, Log(), error, step == 1);
            else
            {
                auto found = spectra.find(system->Name());
                if (step == 1 && plan.useSweepCache)
                {
                    SpinAPI::interaction_ptr field;
                    if (!SystemPreparation::ResolveFieldInteraction(plan, system, field))
                        return false;
                    ResonanceFieldSweep sweep;
                    if (FieldSweep::Prepare(plan, system, field, Actions(), settings->Steps(), sweep, error))
                    {
                        std::vector<std::string> hnames;
                        if (!SystemPreparation::ResolveHamiltonian(plan, system, hnames, error))
                            return false;
                        FieldSyncGuard sync;
                        if (plan.enforceZeemanSync)
                            sync.Apply(SystemPreparation::CollectZeemanInteractions(system, hnames),
                                       field->Field());
                        ResonanceSpectrum spectrum;
                        const bool built = plan.sweepMode == ResonanceSweepMode::PowderMesh
                                               ? ResonanceMeshCalculation::Sweep(plan, system, field, sweep,
                                                                                 spectrum, Log(), error)
                                               : ResonanceExactCalculation::Sweep(plan, system, field, sweep,
                                                                                  spectrum, Log(), error);
                        if (!built)
                        {
                            Log() << "ERROR: Resonance sweep failed: " << error << std::endl;
                            return false;
                        }
                        found = spectra.emplace(system->Name(), std::move(spectrum)).first;
                    }
                    else if (plan.RequiresSweep() || plan.detectionHarmonic > 0)
                    {
                        Log() << "ERROR: Requested resonance mode needs a valid sweep: " << error
                              << std::endl;
                        return false;
                    }
                    else
                        Log() << "Resonance cache unavailable; evaluating each current field: " << error
                              << std::endl;
                }
                if (found != spectra.end())
                {
                    if (step < 1 || step > found->second.field_mT.size())
                        return false;
                    sample = ResonanceSpectrumProcessing::Sample(found->second, step - 1);
                    ok = true;
                }
                else if (plan.RequiresSweep() || plan.detectionHarmonic > 0)
                {
                    error = "no prepared sweep is available; start this resonance run at step 1";
                }
                else
                    ok = ResonanceExactCalculation::Point(plan, system, sample, Log(), error);
            }
            if (!ok)
            {
                Log() << "ERROR: ResonanceGeneral failed for " << system->Name() << ": " << error
                      << std::endl;
                return false;
            }
            row.push_back(std::move(sample));
        }
        // Independent SpinSystems occupy the same row as their header columns.
        Data() << step << " " << settings->Time() << " ";
        WriteStandardOutput(Data());
        for (const auto &sample : row)
            ResonanceSpectrumProcessing::WriteSample(sample, Data());
        Data() << std::endl;
        return true;
    }
} // namespace RunSection::General::Resonance
