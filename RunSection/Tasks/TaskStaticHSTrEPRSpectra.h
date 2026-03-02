#ifndef MOD_RunSection_TaskStaticHSTrEPRSpectra
#define MOD_RunSection_TaskStaticHSTrEPRSpectra

#include "BasicTask.h"
#include "SpinSpace.h"
#include <map>
#include <tuple>

namespace RunSection
{
	class TaskStaticHSTrEPRSpectra : public BasicTask
	{
	private:
		struct SpectrumCache
		{
			unsigned int steps = 0;
			std::vector<double> field_mT;
			std::vector<double> total_x;
			std::vector<double> total_y;
			std::vector<double> total_perp;
			std::vector<double> cross_x;
			std::vector<double> cross_y;
			std::vector<std::string> spin_names;
			std::vector<std::vector<double>> spin_x;
			std::vector<std::vector<double>> spin_y;
			std::vector<std::vector<double>> spin_perp;
			std::vector<std::vector<double>> spin_p;
			std::vector<std::vector<double>> spin_m;
		};

		double mwFrequencyGHz;
		double linewidth_mT;
		double linewidthFad_mT;
		double linewidthDonor_mT;
		std::string lineshape;
		std::string powderGridType;
		std::string powderGridSymmetry;
		int powderGridSize;
		int powdersamplingpoints;
		int powderGammaPoints;
		bool powderFullSphere;
		bool fullTensorRotation;
		bool useSweepCache;
		bool sweepCacheExact;
		bool sweepCacheResfields;
		int sweepCacheResfieldPoints;
		std::vector<std::string> detectSpinNames;
		std::string electron1Name;
		std::string electron2Name;
		std::string fieldInteractionName;
		bool enforceZeemanSync;
		std::string initialStateName;
		std::vector<std::string> hamiltonianH0list;
		std::map<std::string, SpectrumCache> spectrumCache;

		double LineshapeValue(double _delta, double _fwhm) const;
		double LinewidthToOmega(double _fwhm_mT, double _giso) const;
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateSopheGrid(int _gridSize, const std::string &_symmetry, std::vector<std::tuple<double, double, double>> &_grid) const;
		bool SopheGridParams(const std::string &_symmetry, double &_maxPhi, bool &_closedPhi, int &_nOctants) const;
		bool CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const;
		bool ResolveFieldInteraction(const SpinAPI::system_ptr &_system, SpinAPI::interaction_ptr &_fieldInteraction) const;
		bool ResolveDetectionSpins(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, std::vector<SpinAPI::spin_ptr> &_spins, std::vector<std::string> &_spinNames) const;
		void WriteHeader(std::ostream &_stream);
		bool GetLinearFieldSweep(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0, arma::vec &_fieldStep) const;
		bool BuildCachedSpectrum(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, const arma::vec &_field0, const arma::vec &_fieldStep, SpectrumCache &_cache);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		TaskStaticHSTrEPRSpectra(const MSDParser::ObjectParser &, const RunSection &);
		~TaskStaticHSTrEPRSpectra();
	};
}

#endif
