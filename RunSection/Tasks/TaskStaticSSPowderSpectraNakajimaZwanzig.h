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
		void WriteTransitionYieldHeader(const SpinAPI::system_ptr &_system, std::ostream &_stream) const;
		bool ProjectTransitionYields(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space, const arma::cx_mat &_rho, std::ostream &_datastream, std::ostream &_logstream) const;
		bool BuildOrientedInitialDensity(SpinAPI::SpinSpace &space, const arma::cx_mat &referenceDensity, const arma::mat &orientationRotation, SpinAPI::StateFrame stateFrame, bool discardHamiltonianCoherences, const std::vector<std::string> &dephasingHamiltonian, arma::cx_mat &orientedDensity, std::ostream &log_stream) const;
		std::vector<arma::cx_mat> RotateRank1OperatorBasis(const std::vector<arma::cx_mat> &cartesian_operators, const arma::mat &frame_to_lab) const;
		std::vector<arma::cx_mat> RotateRank2OperatorBasis(const std::vector<arma::cx_mat> &cartesian_operators, const arma::mat &frame_to_lab) const;
		std::vector<arma::cx_mat> SingleSpinSphericalTensors(const std::vector<arma::cx_mat> &spin_operators, const arma::cx_vec &field) const;
		std::vector<arma::cx_mat> DoubleSpinSphericalTensors(const std::vector<arma::cx_mat> &spin1_operators, const std::vector<arma::cx_mat> &spin2_operators) const;
		bool TransformSuperoperatorBetweenEigenbases(SpinAPI::SpinSpace &space, const arma::cx_mat &sourceEigenvectors, const arma::cx_mat &targetEigenvectors, const arma::cx_mat &sourceSuperoperator, arma::cx_mat &targetSuperoperator) const;
		bool BuildNakajimaZwanzigLiouvillian(auto &_i, SpinAPI::SpinSpace &_space, const arma::cx_mat &_H, const arma::cx_mat &_relaxationBasisHamiltonian, const arma::mat &_rotationmatrix, arma::cx_mat &_A, arma::cx_mat &_eigenvec);
		bool ConvertSuperspaceToLab(auto &_space, const arma::cx_vec &_rho_vec_eig, const arma::cx_mat &_eigenvec, arma::cx_vec &_rho_vec_lab);
		bool NakajimaZwanzigtensorSpectra(const arma::cx_mat &_op1, const arma::cx_mat &_op2, const arma::cx_mat &_specdens, arma::cx_mat &_NakajimaZwanzigtensor);
		bool ConstructSpecDensGeneralSpectra(const std::vector<double> &_ampl_list, const std::vector<double> &_tau_c_list, const arma::cx_mat &_omega, arma::cx_mat &_specdens);
		bool ConstructSpecDensSpecificSpectra(const std::complex<double> &_ampl, const std::complex<double> &_tau_c, const arma::cx_mat &_omega, arma::cx_mat &_specdens);
		bool ProjectAndPrintOutputLine(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);
		bool ProjectAndPrintOutputLineInf(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream);


	protected:
		bool RunLocal() override;
		bool Validate() override;

	public:
		// Constructors / Destructors
		TaskStaticSSPowderSpectraNakajimaZwanzig(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		~TaskStaticSSPowderSpectraNakajimaZwanzig();	                                                  // Destructor
		
		bool CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const;
		bool CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const;
		bool CreateExplicitPowderGrid(std::vector<std::tuple<double, double, double>> &_grid);
		bool CreateCustomGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_Grid) const;

	};

}

#endif
