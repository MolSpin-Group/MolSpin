/////////////////////////////////////////////////////////////////////////
// TaskStaticSSPowderSpectra (RunSection module)  developed by Irina Anisimova.
// ------------------
//
// Simple quantum yield calculation in Liouville space, derived from the
// properties of the Laplace transformation.
//
// Molecular Spin Dynamics Software.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_TaskStaticSSPowderSpectra
#define MOD_RunSection_TaskStaticSSPowderSpectra

#include "BasicTask.h"
#include "SpinAPIDefines.h"
#include "SpinSpace.h"
#include "Utility.h"

namespace RunSection
{
	//Because the declaration of i is long define it as a new variable that is easier to use
	using SystemIterator = std::vector<SpinAPI::system_ptr>::const_iterator;

	class TaskStaticSSPowderSpectra : public BasicTask
	{
	private:
		double timestep;
		double totaltime;
		SpinAPI::ReactionOperatorType reactionOperators;

		void WriteHeader(std::ostream &); // Write header for the output file
		static arma::cx_vec ComputeRhoDot(double t, arma::sp_cx_mat& L, arma::cx_vec& K, arma::cx_vec RhoNaught);
		bool ProjectAndPrintOutputLine(SystemIterator &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);
		bool ProjectAndPrintOutputLine(SystemIterator &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, arma::sp_cx_mat &_eigen_vec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);
		bool ProjectAndPrintOutputLineInf(SystemIterator &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);
		bool ProjectAndPrintOutputLineInf(SystemIterator &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, arma::sp_cx_mat &_eigen_vec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);

		bool GetEigenvectors_H0(SpinAPI::SpinSpace &_space, arma::vec &_eigen_val, arma::sp_cx_mat &_eigen_vec_sp) const;
		bool GetEigenvectors_H0_Thermal(SpinAPI::SpinSpace &_space, std::vector<std::string> &_thermalhamiltonian_list, arma::vec &_eigen_val, arma::sp_cx_mat &_eigen_vec_sp) const;
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const;
		bool CreateCustomGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_Grid) const;

		bool Create_A_for_current_orientation(SystemIterator &_i, SpinAPI::SpinSpace &_space, double &_theta, double &_phi, arma::sp_cx_mat &_A, std::ostream &_logstream) const;



	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskStaticSSPowderSpectra(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskStaticSSPowderSpectra();	                                                  // Destructor

	};

}

#endif
