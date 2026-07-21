#ifndef MOD_RunSection_TaskStaticHSTrEPRSpectra
#define MOD_RunSection_TaskStaticHSTrEPRSpectra

#include "BasicTask.h"
#include "PowderGrid.h"
#include "SpinSpace.h"
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <utility>

namespace RunSection
{
	class TaskStaticHSTrEPRSpectra : public BasicTask
	{
	private:
		// Sweep-cache output channels. The task can assemble these either by
		// direct diagonalization at every sweep field or by resonance projection.
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

		// Optional developer diagnostics for inspecting individual powder
		// orientations without mixing debug I/O into the spectral workflow.
		struct OrientationDiagnostics
		{
			bool enabled = false;
			double fieldMinT = -std::numeric_limits<double>::infinity();
			double fieldMaxT = std::numeric_limits<double>::infinity();
			int maxOrientations = 0;
			std::string file;

			static OrientationDiagnostics FromProperties(const MSDParser::ObjectParser &_props);
			void InitialiseFile(const std::string &_systemName, const std::vector<std::string> &_spinNames) const;
			bool ShouldRecord(size_t _gridIndex, double _resonanceFieldT) const;
		};

		// Algebraic acceleration for Hamiltonians that conserve total Mz. This
		// changes only the diagonalization route, not the Hamiltonian itself.
		struct MzBlocks
		{
			std::vector<int> mz2;		  // total Mz in units of 1/2
			std::vector<arma::uvec> blocks; // basis indices grouped by Mz
		};

		// Temporarily synchronizes all Zeeman fields to one sweep field and
		// restores them when the local calculation leaves scope.
		struct FieldSyncGuard
		{
			std::vector<std::pair<SpinAPI::interaction_ptr, arma::vec>> saved;

			void Apply(const std::vector<SpinAPI::interaction_ptr> &_interactions, const arma::vec &_field);
			~FieldSyncGuard();
		};

		// Tensor-symmetry summary used to choose a SOPHE grid symmetry when the
		// user requests "auto".
		struct SymmetryFlags
		{
			bool allIsotropic = true;
			bool allDiag = true;
			bool allAxialZ = true;
			bool anyTensor = false;
		};

		double mwFrequencyGHz;
		double linewidth_mT;
		std::string lineshape;

		int detectionHarmonic;
		double modulationAmplitude_mT;

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
		std::string fieldInteractionName;
		bool enforceZeemanSync;
		std::string initialStateName;
		std::vector<std::string> hamiltonianH0list;
		std::map<std::string, SpectrumCache> spectrumCache;

		// Small numerical and grid helpers used by the task workflow below.
		static std::string ToLower(std::string _value);
		static MzBlocks BuildMzBlocks(const std::vector<SpinAPI::spin_ptr> &_spins);
		static arma::mat PassiveZYZRotation(const arma::vec &_fr);
		static bool IsZeemanInteraction(const SpinAPI::interaction_ptr &_interaction);
		static std::vector<SpinAPI::interaction_ptr> CollectZeemanInteractions(const SpinAPI::system_ptr &_system, const std::vector<std::string> &_h0list);
		static SpinAPI::interaction_ptr FindZeemanForSpin(const SpinAPI::spin_ptr &_spin, const std::vector<SpinAPI::interaction_ptr> &_zeemanList);
		static bool IsParallel(const arma::vec &_a, const arma::vec &_b, double _tol);
		static bool CollectAddVectorSteps(const std::vector<std::shared_ptr<Action>> &_actions, unsigned int _steps, std::map<std::string, arma::vec> &_stepsOut, std::string &_error);
		static void UpdateSymmetryFlags(const arma::mat &_matrix, SymmetryFlags &_flags, bool _fullTensorRotation, double _relTol);
		static std::string AutoDetectSopheSymmetry(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, const std::vector<std::string> &_h0list, bool _fullTensorRotation);
		static bool IsBlockDiagonalMz(const arma::sp_cx_mat &_H, const std::vector<int> &_mz2, double _relTol);
		static bool EigSymBlockMz(const arma::sp_cx_mat &_H, const std::vector<arma::uvec> &_blocks, arma::vec &_eigval, arma::cx_mat &_eigvec);

		// Task workflow helpers. These map validated input to the concrete
		// Hamiltonians, grids, detection operators, and cache channels.
		double LineshapeValue(double _delta, double _fwhm) const;
		bool CreatePassiveZYZRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, SpinAPI::PowderGrid &_uniformGrid) const;
		bool ResolveFieldInteraction(const SpinAPI::system_ptr &_system, SpinAPI::interaction_ptr &_fieldInteraction) const;
		bool ResolveDetectionSpins(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, std::vector<SpinAPI::spin_ptr> &_spins, std::vector<std::string> &_spinNames) const;
		void WriteHeader(std::ostream &_stream);
		bool GetLinearFieldSweep(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0, arma::vec &_fieldStep) const;
		bool BuildCachedSpectrum(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, const arma::vec &_field0, const arma::vec &_fieldStep, SpectrumCache &_cache);

		std::vector<double> ApplyFieldHarmonic(const std::vector<double> &_field_mT, const std::vector<double> &_channel) const;
		void ApplyDetectionHarmonic(SpectrumCache &_cache) const;


	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		TaskStaticHSTrEPRSpectra(const MSDParser::ObjectParser &, const RunSection &);
		~TaskStaticHSTrEPRSpectra();
	};
}

#endif
