/////////////////////////////////////////////////////////////////////////
// HSHamiltonianBuilder (RunSection::General::HS)
// ------------------
// Constructs orientation- and time-specific Hilbert Hamiltonians by composing
// SpinSpace primitives. With an explicit H0/H1 split, only H0 is secularized;
// H1 always remains full and uses the same molecular-to-lab powder rotation.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSHamiltonianBuilder
#define MOD_RunSection_General_HS_HSHamiltonianBuilder

#include <armadillo>
#include <string>
#include "SpinSpace.h"
#include "HSExecutionPlan.h"
#include "HSOrientationSampler.h"

namespace RunSection::General::HS
{
	class HSHamiltonianBuilder
	{
	public:
		HSHamiltonianBuilder(const HSExecutionPlan &_plan, SpinAPI::SpinSpace &_space)
			: plan(_plan), space(_space) {}

		bool BuildStatic(const HSOrientation &_orientation, arma::sp_cx_mat &_hamiltonian,
			arma::sp_cx_mat *_h0, std::string &_error) const;
		bool BuildAtTime(const HSOrientation &_orientation, double _time,
			const arma::sp_cx_mat &_staticHamiltonian, arma::sp_cx_mat &_hamiltonian,
			std::string &_error);

	private:
		const HSExecutionPlan &plan;
		SpinAPI::SpinSpace &space;
	};
}
#endif
