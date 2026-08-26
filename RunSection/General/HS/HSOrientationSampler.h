/////////////////////////////////////////////////////////////////////////
// HSOrientationSampler (RunSection::General::HS)
// ------------------
// Builds molecular-to-lab orientation samples from the shared SpinAPI PowderGrid
// constructors through the shared GeneralOrientationSampler. Rotations use the
// established active ZYZ convention. Generated powder weights are normalized
// once to unit ensemble weight; explicit user-supplied weights are preserved.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSOrientationSampler
#define MOD_RunSection_General_HS_HSOrientationSampler

#include <armadillo>
#include <iosfwd>
#include <string>
#include <vector>
#include "HSExecutionPlan.h"

namespace RunSection::General::HS
{
	struct HSOrientation
	{
		double alpha = 0.0;
		double beta = 0.0;
		double gamma = 0.0;
		double weight = 1.0;
		arma::mat frameToLab = arma::eye<arma::mat>(3, 3);
	};

	class HSOrientationSampler
	{
	public:
		static bool Build(const HSExecutionPlan &_plan, std::vector<HSOrientation> &_orientations,
			std::ostream &_log, std::string &_error);
	};
}
#endif
