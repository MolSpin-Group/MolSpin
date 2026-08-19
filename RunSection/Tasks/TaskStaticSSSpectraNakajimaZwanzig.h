/////////////////////////////////////////////////////////////////////////
// TaskStaticSSSpectraNakajimaZwanzig (RunSection module)  developed by Irina Anisimova and Luca Gerhards.
// ------------------
//
// Simple quantum yield calculation in Liouville space, derived from the
// properties of the Laplace transformation.
//
// Molecular Spin Dynamics Software - developed by Irina Anisimova and Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskStaticSSSpectraNakajimaZwanzig
#define MOD_RunSection_TaskStaticSSSpectraNakajimaZwanzig

#include "BasicTask.h"
#include "SpinAPIDefines.h"
#include "SpinSpace.h"

namespace RunSection
{
	//Because the declaration of i is long define it as a new variable that is easier to use
	using SystemIterator = std::vector<SpinAPI::system_ptr>::const_iterator;

	class TaskStaticSSSpectraNakajimaZwanzig : public BasicTask
	{
	private:
		struct ProjectionCache
		{
			bool has_spinlist = false;
			bool ready = false;
			std::vector<arma::cx_mat> spin_Ix;
			std::vector<arma::cx_mat> spin_Iy;
			std::vector<arma::cx_mat> spin_Iz;
			std::vector<arma::cx_mat> spin_Ip;
			std::vector<arma::cx_mat> spin_Im;
			std::vector<arma::cx_mat> transition_proj;
			std::vector<double> transition_rates;
		};

		double timestep;
		double totaltime;
		SpinAPI::ReactionOperatorType reactionOperators;

		void WriteHeader(std::ostream &); // Write header for the output file

		bool NakajimaZwanzigtensorSpectra(const arma::cx_mat &_op1, const arma::cx_mat &_op2, const arma::cx_mat &_specdens, arma::cx_mat &_NakajimaZwanzigtensor); // Contruction of NakajimaZwanzigtensor with operator basis
		bool ConstructSpecDensGeneralSpectra(const std::vector<double> &_ampl_list, const std::vector<double> &_tau_c_list, const arma::cx_mat &_omega, arma::cx_mat &_specdens);
		bool ConstructSpecDensSpecificSpectra(const std::complex<double> &_ampl, const std::complex<double> &_tau_c, const arma::cx_mat &_omega, arma::cx_mat &_specdens);
		bool BuildProjectionCache(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space, const arma::cx_mat &_rotationmtx, bool _cidsp, ProjectionCache &_cache, std::ostream &_log_stream);

		bool ProjectAndPrintOutputLine(SystemIterator &_i, SpinAPI::SpinSpace &_space, const ProjectionCache &_cache, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_data_stream, std::ostream &_log_stream);
		bool ProjectAndPrintOutputLineInf(SystemIterator &_i, SpinAPI::SpinSpace &_space, const ProjectionCache &_cache, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);

	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskStaticSSSpectraNakajimaZwanzig(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskStaticSSSpectraNakajimaZwanzig();													// Destructor
	};

}

#endif
