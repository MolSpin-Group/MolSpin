/////////////////////////////////////////////////////////////////////////
// TaskResonanceGeneral: field-swept spectroscopy lifecycle and output.
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOLSPIN_TASK_RESONANCE_GENERAL_H
#define MOLSPIN_TASK_RESONANCE_GENERAL_H
#include "BasicTask.h"
#include "ResonanceExecutionPlan.h"
#include "ResonanceSpectrumProcessing.h"
#include <map>

namespace RunSection::General::Resonance
{
    class TaskResonanceGeneral : public BasicTask
    {
      public:
        TaskResonanceGeneral(const MSDParser::ObjectParser &, const RunSection &);
        ~TaskResonanceGeneral() override = default;

      protected:
        bool Validate() override;
        bool RunLocal() override;

      private:
        ResonanceExecutionPlan plan;
        bool planResolved = false;
        std::map<std::string, ResonanceSpectrum> spectra;
        bool WriteHeader();
    };
} // namespace RunSection::General::Resonance
#endif
