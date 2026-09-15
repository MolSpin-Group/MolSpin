/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Exact, hybrid and mesh spectrum services; no task lifecycle or output streams.
#ifndef MOLSPIN_RESONANCECALCULATIONS_H
#define MOLSPIN_RESONANCECALCULATIONS_H
#include "ResonanceExecutionPlan.h"
#include "ResonanceFieldSweep.h"
#include "ResonanceSpectrumProcessing.h"

namespace RunSection::General::Resonance
{

    class ResonanceExactCalculation
    {
      public:
        static bool Point(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                          ResonanceSample &sample, std::ostream &log, std::string &error);
        static bool Sweep(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                          const SpinAPI::interaction_ptr &field, const ResonanceFieldSweep &sweep,
                          ResonanceSpectrum &spectrum, std::ostream &log, std::string &error);
    };
    class ResonanceHybridCalculation
    {
      public:
        static bool Point(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                          ResonanceSample &sample, std::ostream &log, std::string &error,
                          bool firstStep = false);
    };
    class ResonanceMeshCalculation
    {
      public:
        static bool Sweep(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                          const SpinAPI::interaction_ptr &field, const ResonanceFieldSweep &sweep,
                          ResonanceSpectrum &spectrum, std::ostream &log, std::string &error);
    };

} // namespace RunSection::General::Resonance
#endif
