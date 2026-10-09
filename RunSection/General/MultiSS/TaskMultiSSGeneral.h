/////////////////////////////////////////////////////////////////////////
// TaskMultiSSGeneral (RunSection::General::MultiSS)
// ------------------
// Thin task shell for the modular multi-manifold superspace framework.
//
// THIS CLASS OWNS ONLY
//   - MolSpin task lifecycle / validation entry points,
//   - logging,
//   - powder-average orchestration,
//   - tabular output.
//
// PHYSICS IS DELEGATED TO
//   SpinAPI::TimeProfile / TransferChannel / QuantumMap
//   ::RunSection::General::SS::SSLiouvillianBuilder
//   RunSection::General::MultiSS::{NetworkBuilder,Propagator,...}
//
// Existing RunSection/Tasks classes are intentionally not called or selected
// from here.  They remain independent reference/user tasks so old and new
// implementations can be compared during migration.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_TaskMultiSSGeneral
#define MOD_RunSection_General_MultiSS_TaskMultiSSGeneral

#include "BasicTask.h"
#include "MultiSSExecutionPlan.h"

namespace RunSection::General::MultiSS
{
    class TaskMultiSSGeneral final : public ::RunSection::BasicTask
    {
    public:
        TaskMultiSSGeneral(const MSDParser::ObjectParser &_parser,
            const ::RunSection::RunSection &_runsection);
        ~TaskMultiSSGeneral() override = default;
    protected:
        bool RunLocal() override;
        bool Validate() override;
    private:
        MultiSSExecutionPlan plan;
        bool WriteHeader(const std::vector<std::string> &_labels,bool _timeEvolution);
    };
}
#endif
