/////////////////////////////////////////////////////////////////////////
// HSOrientationSampler implementation (RunSection::General::HS)
// ------------------
// Identity, explicit, theta/phi, and full theta/phi/gamma SO(3) sampling using
// the shared SpinAPI PowderGrid and ZYZ rotation convention.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSOrientationSampler.h"
#include "PowderGrid.h"

#include <cmath>

namespace RunSection::General::HS
{
	namespace
	{
		double WrapAngle(double value)
		{
			const double period = 2.0 * arma::datum::pi;
			value = std::fmod(value, period);
			if (value < 0.0) value += period;
			return value;
		}

		bool AppendOrientation(double alpha, double beta, double gamma, double weight,
			std::vector<HSOrientation> &out)
		{
			HSOrientation item;
			item.alpha = alpha;
			item.beta = beta;
			item.gamma = gamma;
			item.weight = weight;
			if (!SpinAPI::CreateZYZRotationMatrix(alpha, beta, gamma, item.frameToLab)) return false;
			out.push_back(std::move(item));
			return true;
		}
	}

	bool HSOrientationSampler::Build(const HSExecutionPlan &plan,
		std::vector<HSOrientation> &orientations, std::ostream &log, std::string &error)
	{
		orientations.clear(); error.clear();
		if (plan.orientation == OrientationMode::Identity)
		{
			if (!AppendOrientation(0.0, 0.0, 0.0, 1.0, orientations))
			{ error = "failed to construct the identity orientation"; return false; }
			log << "HS orientation sampling = identity." << std::endl;
			return true;
		}

		if (plan.orientation == OrientationMode::Explicit)
		{
			if (!AppendOrientation(plan.powderGammaOffset, plan.explicitTheta, plan.explicitPhi,
				plan.explicitWeight, orientations))
			{ error = "failed to construct the explicit powder orientation"; return false; }
			log << "HS orientation sampling = explicit theta/phi/gamma orientation with supplied weight." << std::endl;
			return true;
		}

		SpinAPI::PowderGrid base;
		switch (plan.powderGridType)
		{
		case SpinAPI::PowderGridType::Uniform:
			if (!SpinAPI::CreateUniformPowderGrid(plan.powderPoints, plan.powderDomain, base))
			{
				error = "failed to construct the uniform theta/phi powder grid";
				return false;
			}
			log << "Using SpinAPI uniform (golden-angle) powder grid with " << base.size()
				<< " theta/phi orientations over the "
				<< (plan.powderDomain == SpinAPI::PowderGridDomain::FullSphere
					? "full sphere." : "upper hemisphere.") << std::endl;
			break;

		case SpinAPI::PowderGridType::Sophe:
			if (!SpinAPI::CreateSophePowderGrid(plan.powderGridSize, plan.powderSymmetry, base))
			{
				error = "failed to construct the SOPHE powder grid";
				return false;
			}
			log << "Using SpinAPI SOPHE powder grid with grid size " << plan.powderGridSize
				<< ", symmetry " << plan.powderSymmetry << ", and " << base.size()
				<< " theta/phi orientations." << std::endl;
			break;

		case SpinAPI::PowderGridType::Octant:
			if (!SpinAPI::CreateOctantPowderGrid(plan.powderPoints, base))
			{
				error = "failed to construct the octant powder grid";
				return false;
			}
			log << "Using SpinAPI historical octant powder grid with " << base.size()
				<< " theta/phi orientations." << std::endl;
			break;
		}

		if (plan.orientation == OrientationMode::Powder2D)
		{
			for (const auto &p : base)
				if (!AppendOrientation(0.0, p.theta, p.phi, p.weight, orientations))
				{ error = "failed to construct a theta/phi powder rotation"; return false; }
			return true;
		}

		const int ngamma = plan.powderGammaPoints;
		log << "Sampling powder gamma with " << ngamma << " points." << std::endl;
		for (const auto &p : base)
		{
			for (int g = 0; g < ngamma; ++g)
			{
				const double gamma = WrapAngle(plan.powderGammaOffset + 2.0 * arma::datum::pi *
					(static_cast<double>(g) + 0.5) / static_cast<double>(ngamma));
				if (!AppendOrientation(gamma, p.theta, p.phi,
					p.weight / static_cast<double>(ngamma), orientations))
				{ error = "failed to construct a theta/phi/gamma powder rotation"; return false; }
			}
		}
		log << "Using " << orientations.size() << " total SO(3) orientations." << std::endl;
		return true;
	}
}
