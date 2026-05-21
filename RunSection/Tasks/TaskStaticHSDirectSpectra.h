/////////////////////////////////////////////////////////////////////////
// TaskStaticHSDirectSpectra (RunSection module) by Luca Gerhards
// ------------------
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskStaticHSDirectSpectra
#define MOD_RunSection_TaskStaticHSDirectSpectra

#include <armadillo>
#include <tuple>
#include <string>
#include <vector>
#include "BasicTask.h"
#include "SpinAPIDefines.h"

namespace RunSection
{
	class TaskStaticHSDirectSpectra : public BasicTask
	{
	private:

		double timestep;
		double totaltime;
		double mwFrequencyGHz;
		double secularTolerance;
		int powderGammaPoints;
		bool powderFullSphere;
		bool fullTensorRotation;
		bool rwaEnabled;
		bool secularizeInteractions;
		std::string fieldInteractionName;
		std::vector<std::string> rwaSpinNames;

		SpinAPI::ReactionOperatorType reactionOperators;

		void WriteHeader(std::ostream &); // Write header for the output file
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const;
		bool CreateExplicitPowderGrid(std::vector<std::tuple<double, double, double>> &_grid);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskStaticHSDirectSpectra(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskStaticHSDirectSpectra();												   // Destructor
	};
}

#endif
