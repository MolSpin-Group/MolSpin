/////////////////////////////////////////////////////////////////////////
// TaskStaticSSPowderSpectraNakajimaZwanzig (RunSection module)  developed by Irina Anisimova.
// ------------------
//
// Simple quantum yield calculation in Liouville space, derived from the
// properties of the Laplace transformation.
//
// Molecular Spin Dynamics Software.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskStaticSSPowderSpectraNakajimaZwanzig
#define MOD_RunSection_TaskStaticSSPowderSpectraNakajimaZwanzig

#include "BasicTask.h"
#include "SpinAPIDefines.h"
#include "SpinSpace.h"
#include "Utility.h"

namespace RunSection
{
	//Because the declaration of i is long define it as a new variable that is easier to use
	using SystemIterator = std::vector<SpinAPI::system_ptr>::const_iterator;

	class TaskStaticSSPowderSpectraNakajimaZwanzig : public BasicTask
	{
	private:
		double timestep;
		double totaltime;
		SpinAPI::ReactionOperatorType reactionOperators;

		void WriteHeader(std::ostream &); // Write header for the output file
		static arma::cx_vec ComputeRhoDot(double t, arma::sp_cx_mat& L, arma::cx_vec& K, arma::cx_vec RhoNaught);
		bool BuildNakajimaZwanzigLiouvillian(SystemIterator &_i, SpinAPI::SpinSpace &_space, const arma::cx_mat &_H, arma::cx_mat &_A, arma::cx_mat &_eigenvec);
		bool ConvertSuperspaceToLab(SpinAPI::SpinSpace &_space, const arma::cx_vec &_rho_vec_eig, const arma::cx_mat &_eigenvec, arma::cx_vec &_rho_vec_lab);
		bool NakajimaZwanzigtensorSpectra(const arma::cx_mat &_op1, const arma::cx_mat &_op2, const arma::cx_mat &_specdens, arma::cx_mat &_NakajimaZwanzigtensor);
		bool ConstructSpecDensGeneralSpectra(const std::vector<double> &_ampl_list, const std::vector<double> &_tau_c_list, const arma::cx_mat &_omega, arma::cx_mat &_specdens);
		bool ConstructSpecDensSpecificSpectra(const std::complex<double> &_ampl, const std::complex<double> &_tau_c, const arma::cx_mat &_omega, arma::cx_mat &_specdens);
		bool ProjectAndPrintOutputLine(SystemIterator &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);
		bool ProjectAndPrintOutputLineInf(SystemIterator &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);


	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskStaticSSPowderSpectraNakajimaZwanzig(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskStaticSSPowderSpectraNakajimaZwanzig();	                                                  // Destructor
		
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const;
		bool CreateCustomGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_Grid) const;

	};

}

#endif
