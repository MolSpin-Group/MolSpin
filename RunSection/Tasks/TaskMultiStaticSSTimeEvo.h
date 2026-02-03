/////////////////////////////////////////////////////////////////////////
// TaskMultiStaticSSTimeEvo (RunSection module)
// ------------------
//
// Simple time-evolution calculation in Liouville space.
//
// -- Multi-system version: Allows transitions between SpinSystems --
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskMultiStaticSSTimeEvo
#define MOD_RunSection_TaskMultiStaticSSTimeEvo

#include "BasicTask.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "SpinAPIDefines.h"

namespace RunSection
{
	class TaskMultiStaticSSTimeEvo : public BasicTask
	{
	private:
		double timestep;
		double totaltime;
		bool ProductYieldsOnly;
		SpinAPI::ReactionOperatorType reactionOperators;

		void WriteHeader(std::ostream &); // Write header for the output file

		// Private method that gathers and outputs the results from a given time-integrated density operator
		void GatherResults(const arma::cx_mat &, const SpinAPI::SpinSystem &, const SpinAPI::SpinSpace &);
		bool SeperateSpinSystems(const arma::cx_vec& rho0, const std::vector<std::pair<SpinAPI::system_ptr, std::shared_ptr<SpinAPI::SpinSpace>>>& spaces, bool, double);

		static arma::cx_vec ComputeRhoDot(double t, arma::sp_cx_mat& L, arma::cx_vec& K, arma::cx_vec RhoNaught);

		void UpdateTimeDependences(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>>& _spaces, arma::sp_cx_mat& dL, double ctime);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskMultiStaticSSTimeEvo(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskMultiStaticSSTimeEvo();												   // Destructor
	};
}

#endif
