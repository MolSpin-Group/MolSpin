/////////////////////////////////////////////////////////////////////////
// TaskHSGeneral (RunSection::General::HS)
// ------------------
// Unified production task for single-SpinSystem Hilbert-space propagation.
// The task owns RunSection lifecycle/output only and delegates state, orientation,
// Hamiltonian, reaction/relaxation, propagation, and observable operations to the
// modular General/HS components. Legacy tasks remain independent references.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_TaskHSGeneral
#define MOD_RunSection_General_HS_TaskHSGeneral

#include "BasicTask.h"
#include "HSExecutionPlan.h"

namespace RunSection::General::HS
{
	class TaskHSGeneral : public BasicTask
	{
	public:
		TaskHSGeneral(const MSDParser::ObjectParser &, const RunSection &);
		~TaskHSGeneral() override = default;

	protected:
		bool RunLocal() override;
		bool Validate() override;

	private:
		bool WriteHeader(std::ostream &);

		HSExecutionPlan plan;
		bool planResolved = false;
	};
}

#endif
