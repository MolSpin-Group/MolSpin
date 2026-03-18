/////////////////////////////////////////////////////////////////////////
// TaskMultiDynamicHSTimeEvo (RunSection module)
// ------------------
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskMultiDynamicHSTimeEvo
#define MOD_RunSection_TaskMultiDynamicHSTimeEvo

#include "SpinSpace.h"
#include "BasicTask.h"
#include "Utility.h"

namespace RunSection
{
	class TaskMultiDynamicHSTimeEvo : public BasicTask
	{
	private:
		double timestep;
		double totaltime;
		unsigned int outputstride;

	private:
		// Method to obtain creation operators
		void GetCreationOperators(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &,
								  std::vector<arma::sp_cx_mat> &, const std::vector<arma::cx_mat> &);
		void GetCreationOperators(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &,
								  arma::sp_cx_mat &, const arma::cx_mat &, int i);

		// Timestep function
		void AdvanceStep_AsyncLeapfrog(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &,
									   const std::vector<arma::cx_mat> &, const std::vector<arma::sp_cx_mat> &,
									   const std::vector<arma::cx_mat> &, const std::vector<arma::sp_cx_mat> &,
									   const std::vector<arma::sp_cx_mat> &, std::vector<arma::cx_mat> &);

		void AdvanceStep_RK4(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &_spaces,
							 const std::vector<arma::cx_mat> &_H, const std::vector<arma::sp_cx_mat> &_dH,
							 const std::vector<arma::cx_mat> &_K, const std::vector<arma::sp_cx_mat> &_dK,
							 const std::vector<arma::sp_cx_mat> &_C, std::vector<arma::cx_mat> &_rho);

		arma::cx_mat ComputeRhoDot(const arma::cx_mat &H, const arma::sp_cx_mat &dH,
								   const arma::cx_mat &K, const arma::sp_cx_mat &dK,
								   const arma::sp_cx_mat &C, const arma::cx_mat &rho);

		// Method to obtain the results from the current state
		void OutputResults(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &, const std::vector<arma::cx_mat> &, const double);

		// Method to update time-dependent interactions or reactions
		void UpdateTimeDependences(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &,
								   std::vector<arma::sp_cx_mat> &, std::vector<arma::sp_cx_mat> &, const double);

		// Write header for the output file
		void WriteHeader(std::ostream &);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskMultiDynamicHSTimeEvo(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskMultiDynamicHSTimeEvo();													// Destructor
	};
}

#endif
