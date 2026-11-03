/////////////////////////////////////////////////////////////////////////
// TaskActionSpectrumHistogram (RunSection module)
// ------------------
// Computes the action-spectrum histogram of an uncoupled radical pair, see Hiscock et al. (2017) and Leberecht et al. (2022).
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// Task implemented by Siu Ying Wong and Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskActionSpectrumHistogramRPOnlyDec
#define MOD_RunSection_TaskActionSpectrumHistogramRPOnlyDec

#include "SpinSpace.h"
#include <array>
#include "BasicTask.h"

namespace RunSection
{
	class TaskActionSpectrumHistogramRPOnlyDec : public BasicTask
	{
	private:
		// Data members

		// Private methods
		long long GetNumDifferencesRPOnlyDec(long long dimension);
		arma::vec GetDifferencesRPOnlyDec(arma::vec values);
		arma::cx_vec GetPopulationVector(const arma::cx_mat &V1, const arma::cx_mat &V2,
			const std::array<arma::cx_mat, 3> &S1, const std::array<arma::cx_mat, 3> &S2);
		arma::vec GetPerpendicular3DVectorRPOnlyDec(arma::vec vector, bool normalise);
		arma::vec GetResonanceEffectsRPOnlyDec(arma::cx_vec initialstatepopulation, arma::sp_cx_mat transitionhamiltonian);
		double GetHistogramPrefactorRPOnlyDec(bool useMHzunits);
		arma::vec GetNormalisedHistogramHeightsRPOnlyDec(arma::vec energygaps, arma::vec resonanceeffects, double binwidth, int numbins, bool useMHzunits);
		arma::vec GetHistogramBinCenters();
		void WriteHeader(std::ostream &); // Write header for the output file

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskActionSpectrumHistogramRPOnlyDec(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskActionSpectrumHistogramRPOnlyDec();												   // Destructor
	};
}

#endif
