/////////////////////////////////////////////////////////////////////////
// TaskStaticSSPowderSpectra implementation (RunSection module)  developed by Irina Anisimova.
//
// Notes: no pulses implemented at the current state
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include "TaskStaticSSPowderSpectra.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Operator.h"
#include "Pulse.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSSPowderSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSPowderSpectra::TaskStaticSSPowderSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticSSPowderSpectra::~TaskStaticSSPowderSpectra()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSSPowderSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticSSPowderSpectra::RunLocal()
	{
		this->Log() << "Running method StaticSS-PowderSpectra." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Decline density matrix and density vector variables
		arma::cx_mat rho0;
		arma::cx_vec rho0vec;

		// Obtain spin systems
		auto systems = this->SpinSystems();

		// Loop through all SpinSystems
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Make sure we have an initial state
			auto initial_states = (*i)->InitialState();
			if (initial_states.size() < 1)
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
				continue;
			}

			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(true);
			space.SetReactionOperatorType(this->reactionOperators);

			std::vector<double> weights;
			weights = (*i)->Weights();

			// Normalize the weights
			double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
			if (sum_weights > 0)
			{
				for (double &weight : weights)
				{
					weight /= sum_weights;
				}
			}

			// Get the initial state
			if (weights.size() > 1)
			{
				this->Log() << "Using weighted density matrix for initial state. Be sure that the sum of weights equals to 1." << std::endl;
				// Get the initial state
				int counter = 0;
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;
					if (!space.GetState(*j, tmp_rho0))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					if (j == initial_states.cbegin())
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[0] << "\"." << std::endl;
						rho0 = weights[0] * tmp_rho0;
						counter += 1;
					}
					else
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[counter] << "\"." << std::endl;
						rho0 += weights[counter] * tmp_rho0;
						counter += 1;
					}
				}
			}
			else
			// Get the initial state without weights
			{
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;

					// Get the initial state in thermal equilibrium
					if ((*j) == nullptr) // "Thermal initial state"
					{
						this->Log() << "Initial state = thermal " << std::endl;

						// Get the thermalhamiltonianlist
						std::vector<std::string> thermalhamiltonian_list = (*i)->ThermalHamiltonianList();

						this->Log() << "ThermalHamiltonianList = [";
						for (size_t j = 0; j < thermalhamiltonian_list.size(); j++)
						{
							this->Log() << thermalhamiltonian_list[j];
							if (j < thermalhamiltonian_list.size() - 1)
								this->Log() << ", "; // Add a comma between elements
						}
						this->Log() << "]" << std::endl;

						// Get temperature
						double temperature = (*i)->Temperature();
						this->Log() << "Temperature = " << temperature << "K" << std::endl;

						// Get the initial state with thermal equilibrium
						if (!space.GetThermalState(space, temperature, thermalhamiltonian_list, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto thermal state, initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}
					else // Get the initial state without thermal equilibrium
					{
						if (!space.GetState(*j, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}

					// Obtain the initial density matrix
					if (j == initial_states.cbegin())
					{
						rho0 = tmp_rho0;
					}
					else
					{
						rho0 += tmp_rho0;
					}
				}
			}

			rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

			// Convert initial state to superoperator space
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

			// Read in input parameters

			// Read the method from the input file
			std::string Method;
			if (!this->Properties()->Get("method", Method))
			{
				this->Log() << "Failed to obtain an input for a Method. Please specify method = timeinf or method = timeevo." << std::endl;
			}

			// Read if the result should be integrated or not if method.
			bool integration = false;
			if (!this->Properties()->Get("integration", integration))
			{
				this->Log() << "Failed to obtain an input for an integtation. Plese use integration = true/false. Using integration = false by default. " << std::endl;
			}
			this->Log() << "Integration of the yield in time on a grid  = " << integration << std::endl;

			// Read CIDSP from the input file
			bool CIDSP = false;
			if (!this->Properties()->Get("cidsp", CIDSP))
			{
				this->Log() << "Failed to obtain an input for a CIDSP. Plese use cidsp = true/false. Using cidsp = false by default. " << std::endl;
			}

			// Read in the number of points on the sampling grid
			int numPoints = 1000;
			if (!this->Properties()->Get("powdersamplingpoints", numPoints))
			{
				this->Log() << "Failed to obtain an input for a number of sampling points. Plese use powdersamplingpoints = N. Using powdersamplingpoints = 1000 by default. " << std::endl;
			}

			double Printedtime = 0;

			// Construct grid
			std::vector<std::tuple<double, double, double>> grid;
			if (!this->CreateUniformGrid(numPoints, grid))
			{
				this->Log() << "Failed to obtain an Uniform grid." << std::endl;
			}

			// Method Propagation to infinity
			if (Method.compare("timeinf") == 0)
			{
				// Perform the calculation
				this->Log() << "Ready to perform calculation." << std::endl;

				this->Log() << "Method = " << Method << std::endl;

				std::vector<arma::cx_vec> rho_tmp(numPoints);
				for (auto &v : rho_tmp)
					v.zeros(size(rho0vec));

				arma::cx_vec integral;
				integral.zeros(size(rho0vec));

				// Initialize a first step
				arma::cx_vec rhovec = rho0vec;

				for (int grid_num = 0; grid_num < numPoints; ++grid_num)
				{
					auto [theta, phi, weight] = grid[grid_num];

					// Construct the rotation matrix
					arma::mat Rot_mat;
					double gamma = 0;
					if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
					{
						this->Log() << "Failed to obtain an Lebedev grid." << std::endl;
					}

					// Construct the hamiltonian H0
					std::vector<std::string> HamiltonianH0list;
					if (!this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ','))
					{
						this->Log() << "Failed to obtain an input for a HamiltonianH0." << std::endl;
					}

					space.UseSuperoperatorSpace(false);
					// Get the Hamiltonian
					arma::sp_cx_mat H0;
					if (!space.BaseHamiltonianRotated(HamiltonianH0list, Rot_mat, H0))
					{
						this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
						continue;
					}

					// Transforming into superspace
					arma::sp_cx_mat lhs;
					arma::sp_cx_mat rhs;
					arma::sp_cx_mat H_SS;
					space.SuperoperatorFromLeftOperator(H0, lhs);
					space.SuperoperatorFromRightOperator(H0, rhs);

					H_SS = lhs - rhs;

					// Get a matrix to collect all the terms (the total Liouvillian)
					arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * H_SS;

					std::vector<std::string> HamiltonianH1list;
					if (!this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ','))
					{
						this->Log() << "Failed to obtain an input for a HamiltonianH1." << std::endl;
					}

					// Get the Hamiltonian H1
					arma::sp_cx_mat H1;
					if (!space.ThermalHamiltonian(HamiltonianH1list, H1))
					{
						this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
						continue;
					}

					arma::sp_cx_mat H1lhs;
					arma::sp_cx_mat H1rhs;
					arma::sp_cx_mat H1_SS;

					space.SuperoperatorFromLeftOperator(H1, H1lhs);
					space.SuperoperatorFromRightOperator(H1, H1rhs);

					H1_SS = H1lhs - H1rhs;
					A += arma::cx_double(0.0, -1.0) * H1_SS;

					space.UseSuperoperatorSpace(true);

					////////////////////
					// Add reaction
					arma::sp_cx_mat K;
					if (!space.TotalReactionOperator(K))
					{
						this->Log() << "Warning: Failed to obtain matrix representation of the reaction operators!" << std::endl;
					}
					A -= K;

					// Get the relaxation terms, assuming that they can just be added to the Liouvillian superoperator
					arma::sp_cx_mat R;
					for (auto j = (*i)->operators_cbegin(); j != (*i)->operators_cend(); j++)
					{
						if (space.RelaxationOperator((*j), R))
						{
							A += R;
							this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to the Liouvillian.\n";
						}
					}
					/////////////////////////////

					arma::cx_vec result = -solve(arma::conv_to<arma::cx_mat>::from(A), rhovec);

					// Integrate over all grid points
					integral += weight * result;
				}

				rhovec = integral;

				if (!this->ProjectAndPrintOutputLineInf(i, space, rhovec, Printedtime, this->timestep, CIDSP, this->Data(), this->Log()))
					this->Log() << "Could not project the state vector and print the result into a file" << std::endl;

				this->Log() << "Done with calculation." << std::endl;
			}
			// Method TIME EVOLUTION
			else if (Method.compare("timeevo") == 0)
			{

				if (!this->totaltime == 0)
				{
					// Perform the calculation
					this->Log() << "Ready to perform calculation." << std::endl;

					this->Log() << "Method = " << Method << std::endl;

					// Create a holder vector for an averaged density
					int firststep = 0;
					unsigned int time_steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
					std::vector<arma::cx_vec> rho_avg(time_steps + 1);
					for (auto &v : rho_avg)
						v.zeros(size(rho0vec));

					for (int grid_num = 0; grid_num < numPoints; ++grid_num)
					{
						auto [theta, phi, weight] = grid[grid_num];

						// Initialize a first step
						arma::cx_vec rhovec = rho0vec;

						// Create rotation matrix
						arma::mat Rot_mat;
						double gamma = 0;
						if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
						{
							this->Log() << "Failed to obtain an Lebedev grid." << std::endl;
						}

						// Create Hamiltonian H0
						std::vector<std::string> HamiltonianH0list;
						if (!this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ','))
						{
							this->Log() << "Failed to obtain an input for a HamiltonianH0." << std::endl;
						}

						space.UseSuperoperatorSpace(false);
						// Get the Hamiltonian
						arma::sp_cx_mat H0;
						if (!space.BaseHamiltonianRotated(HamiltonianH0list, Rot_mat, H0))
						{
							this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
							continue;
						}

						// Transforming into superspace
						arma::sp_cx_mat lhs;
						arma::sp_cx_mat rhs;
						arma::sp_cx_mat H_SS;
						space.SuperoperatorFromLeftOperator(H0, lhs);
						space.SuperoperatorFromRightOperator(H0, rhs);

						H_SS = lhs - rhs;

						// Get a matrix to collect all the terms (the total Liouvillian)
						arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * H_SS;

						// Create Hamiltonian H1
						std::vector<std::string> HamiltonianH1list;
						if (!this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ','))
						{
							this->Log() << "Failed to obtain an input for a HamiltonianH1." << std::endl;
						}

						// Get the Hamiltonian
						arma::sp_cx_mat H1;
						if (!space.ThermalHamiltonian(HamiltonianH1list, H1))
						{
							this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
							continue;
						}

						arma::sp_cx_mat H1lhs;
						arma::sp_cx_mat H1rhs;
						arma::sp_cx_mat H1_SS;

						space.SuperoperatorFromLeftOperator(H1, H1lhs);
						space.SuperoperatorFromRightOperator(H1, H1rhs);

						H1_SS = H1lhs - H1rhs;
						A += arma::cx_double(0.0, -1.0) * H1_SS;

						space.UseSuperoperatorSpace(true);

						////////////////////

						// Get transition operator
						arma::sp_cx_mat K;
						if (!space.TotalReactionOperator(K))
						{
							this->Log() << "Warning: Failed to obtain matrix representation of the reaction operators!" << std::endl;
						}
						A -= K;

						// Get the relaxation terms, assuming that they can just be added to the Liouvillian superoperator
						arma::sp_cx_mat R;
						for (auto j = (*i)->operators_cbegin(); j != (*i)->operators_cend(); j++)
						{
							if (space.RelaxationOperator((*j), R))
							{
								A += R;
								this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to the Liouvillian.\n";
							}
						}
						////

						arma::cx_vec rhoavg_n;
						rhoavg_n.zeros(size(rho0vec));

						// Create array containing a propagator and the current state of each system
						std::pair<arma::sp_cx_mat, arma::cx_vec> G;
						arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from(A * this->timestep)));
						// Get the propagator and put it into the array together with the initial state
						G = std::pair<arma::sp_cx_mat, arma::cx_vec>(A_sp, rhovec);

						// unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
						for (unsigned int n = firststep; n <= time_steps; n++)
						{
							if (n == 0)
							{
								rho_avg[n] += weight * G.second;
							}
							else
							{
								// Take a step, "first" is propagator and "second" is current state
								rhovec = G.first * G.second;

								// Integrate the density vector over the current time interval
								if (integration)
								{
									rhoavg_n += this->timestep * (G.second + rhovec) / 2;
								}

								// Get the new current state density vector
								G.second = rhovec;

								// Save the result if there were some changes
								if (!rhoavg_n.is_zero(0))
								{
									rhovec = rhoavg_n;
								}

								rho_avg[n] += weight * rhovec;
							}
						}
					}

					for (unsigned int n = firststep; n <= time_steps; n++)
					{
						if (!this->ProjectAndPrintOutputLine(i, space, rho_avg[n], Printedtime, this->timestep, n, CIDSP, this->Data(), this->Log()))
							this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
					}

					// double InitialTimestep = this->timestep;
					// double Currenttime = 0;
					// double time;
					// unsigned int n = 0;
					// while (Currenttime <= this->totaltime)
					// {
					// 	if (!n == 0)
					// 	{
					// 		{
					// 			Currenttime += this->timestep;
					// 			time = RungeKutta45Armadillo(A, rhovec, rhovec, this->timestep, ComputeRhoDot, {1e-7, 1e-6}, InitialTimestep * 1e-3, InitialTimestep * 1e4);
					// 		}
					// 	}

					// 	// n++; //delete if not RK
				}

				this->Log() << "Done with calculation." << std::endl;
			}
			else
			{
				this->Log() << "Undefined spectroscopy method. Please choose between timeinf or timeevo methods." << std::endl;
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		// Terminate the line in the data file after iteration through all spin systems
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticSSPowderSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		std::vector<std::string> spinList;
		bool CIDSP = false;
		int m;

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{

			if (this->Properties()->GetList("spinlist", spinList, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					std::string spintype;

					(*l)->Properties()->Get("type", spintype);

					for (m = 0; m < (int)spinList.size(); m++)
					{

						if ((*l)->Name() == spinList[m])
						{
							// Yields are written per transition
							// bool CIDSP = false;
							if (this->Properties()->Get("cidsp", CIDSP) && CIDSP == true)
							{
								// Write each transition name
								auto transitions = (*i)->Transitions();
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Ix ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Iy ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Iz ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Ip ";
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
											<< ".Im ";
								}
							}
							else
							{
								// Write each state name
								auto states = (*i)->States();
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ix ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iy ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iz ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ip ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Im ";
							}
						}
					}
				}
			}
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticSSPowderSpectra::Validate()
	{

		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;

		// Get timestep
		if (this->Properties()->Get("timestep", inputTimestep))
		{
			if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
			{
				this->timestep = inputTimestep;
			}
			else
			{
				this->Log() << "# WARNING: undefined timestep, using by default 0.1 ns!" << std::endl;
				this->timestep = 0.1;
			}
		}

		// Get totaltime
		if (this->Properties()->Get("totaltime", inputTotaltime))
		{
			if (std::isfinite(inputTotaltime) && inputTotaltime >= 0.0)
			{
				this->totaltime = inputTotaltime;
			}
			else
			{
				this->Log() << "# ERROR: invalid total time!" << std::endl;
				return false;
			}
		}

		// Get the reacton operator type
		std::string str;
		if (this->Properties()->Get("reactionoperators", str))
		{
			if (str.compare("haberkorn") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Haberkorn;
				this->Log() << "Setting reaction operator type to Haberkorn." << std::endl;
			}
			else if (str.compare("lindblad") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Lindblad;
				this->Log() << "Setting reaction operator type to Lindblad." << std::endl;
			}
			else
			{
				this->Log() << "Warning: Unknown reaction operator type specified. Using default reaction operators." << std::endl;
			}
		}

		return true;
	}

	bool TaskStaticSSPowderSpectra::GetEigenvectors_H0(SpinAPI::SpinSpace &_space, arma::vec &_eigen_val, arma::sp_cx_mat &_eigen_vec_sp) const
	{
		_space.UseSuperoperatorSpace(false);

		arma::cx_mat H;

		if (!_space.Hamiltonian(H))
		{
			// this->Log() << "Failed to obtain Static Hamiltonian in Hilbert Space." << std::endl;
		}

		arma::cx_mat _eigen_vec;

		// this->Log() << "Starting diagonalization..." << std::endl;
		arma::eig_sym(_eigen_val, _eigen_vec, (H));
		// this->Log() << "Diagonalization done! Eigenvalues: " << _eigen_val.n_elem << ", eigenvectors: " << _eigen_vec.n_cols << std::endl;

		_eigen_vec_sp = arma::conv_to<arma::sp_cx_mat>::from(_eigen_vec);

		_space.UseSuperoperatorSpace(true);

		return true;
	}

	bool TaskStaticSSPowderSpectra::GetEigenvectors_H0_Thermal(SpinAPI::SpinSpace &_space, std::vector<std::string> &_thermalhamiltonian_list, arma::vec &_eigen_val, arma::sp_cx_mat &_eigen_vec_sp) const
	{
		_space.UseSuperoperatorSpace(false);

		arma::cx_mat H;

		if (!_space.ThermalHamiltonian(_thermalhamiltonian_list, H))
		{
			// this->Log() << "Failed to obtain Static Hamiltonian in Hilbert Space." << std::endl;
		}

		arma::cx_mat _eigen_vec;

		// this->Log() << "Starting diagonalization..." << std::endl;
		arma::eig_sym(_eigen_val, _eigen_vec, (H));
		// this->Log() << "Diagonalization done! Eigenvalues: " << _eigen_val.n_elem << ", eigenvectors: " << _eigen_vec.n_cols << std::endl;

		_eigen_vec_sp = arma::conv_to<arma::sp_cx_mat>::from(_eigen_vec);

		_space.UseSuperoperatorSpace(true);

		return true;
	}

	arma::cx_vec TaskStaticSSPowderSpectra::ComputeRhoDot(double t, arma::sp_cx_mat &L, arma::cx_vec &K, arma::cx_vec RhoNaught)
	{
		arma::cx_vec ReturnVec(L.n_rows);
		RhoNaught = RhoNaught + K;
		ReturnVec = L * RhoNaught;
		return ReturnVec;
	}

	bool TaskStaticSSPowderSpectra::ProjectAndPrintOutputLine(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)) && (_n == 0))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		// Get nuclei of interest for CIDNP spectrum
		arma::cx_mat Iprojx;
		arma::cx_mat Iprojy;
		arma::cx_mat Iprojz;
		arma::cx_mat Iprojp;
		arma::cx_mat Iprojm;

		std::vector<std::string> spinList;

		if (_n == 0)
			_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
		_datastream << std::setprecision(12) << _printedtime + (_n * _timestep) << " ";
		this->WriteStandardOutput(_datastream);

		if (this->Properties()->GetList("spinlist", spinList, ','))
		{

			for (auto l = (*_i)->spins_cbegin(); l != (*_i)->spins_cend(); l++)
			{
				for (int m = 0; m < (int)spinList.size(); m++)
				{
					if ((*l)->Name() == spinList[m])
					{
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
						{
							return false;
						}

						arma::cx_mat P;

						// There are two result modes - either write results per transition  if CIDSP is true or for each defined state if CIDSP is false

						if (_cidsp == true)
						{
							// Loop through all defind transitions
							auto transitions = (*_i)->Transitions();
							for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
							{
								// Make sure that there is a state object
								if ((*j)->SourceState() == nullptr)
									continue;

								if ((!_space.GetState((*j)->SourceState(), P)) && (_n == 0))
								{
									_logstream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*_i)->Name() << "\"." << std::endl;
									continue;
								}

								// Return the yield for this transition
								_datastream << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojp * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojm * (*j)->Rate() * P * rho0)) << " ";
							}
						}
						else if (_cidsp == false)
						{
							// Return the yield for this state - note that no reaction rates are included here.
							_datastream << std::real(arma::trace(Iprojx * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojy * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojz * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojp * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojm * rho0)) << " ";
						}
					}
				}
			}
		}
		else
		{
			if (_n == 0)
				_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		_datastream << std::endl;

		return true;
	}

	bool TaskStaticSSPowderSpectra::ProjectAndPrintOutputLine(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, arma::sp_cx_mat &_eigen_vec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)) && (_n == 0))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		// rho0 = (_eigen_vec * rho0 * _eigen_vec.t());

		// Get nuclei of interest for CIDNP spectrum
		arma::cx_mat Iprojx;
		arma::cx_mat Iprojy;
		arma::cx_mat Iprojz;
		arma::cx_mat Iprojp;
		arma::cx_mat Iprojm;

		std::vector<std::string> spinList;

		if (_n == 0)
			_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
		_datastream << std::setprecision(12) << _printedtime + (_n * _timestep) << " ";
		this->WriteStandardOutput(_datastream);

		if (this->Properties()->GetList("spinlist", spinList, ','))
		{

			for (auto l = (*_i)->spins_cbegin(); l != (*_i)->spins_cend(); l++)
			{
				for (int m = 0; m < (int)spinList.size(); m++)
				{
					if ((*l)->Name() == spinList[m])
					{
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
						{
							return false;
						}

						Iprojx = (_eigen_vec.t() * Iprojx * _eigen_vec);
						Iprojy = (_eigen_vec.t() * Iprojy * _eigen_vec);
						Iprojz = (_eigen_vec.t() * Iprojz * _eigen_vec);
						Iprojp = (_eigen_vec.t() * Iprojp * _eigen_vec);
						Iprojm = (_eigen_vec.t() * Iprojm * _eigen_vec);

						arma::cx_mat P;

						// There are two result modes - either write results per transition  if CIDSP is true or for each defined state if CIDSP is false

						if (_cidsp == true)
						{
							// Loop through all defind transitions
							auto transitions = (*_i)->Transitions();
							for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
							{
								// Make sure that there is a state object
								if ((*j)->SourceState() == nullptr)
									continue;

								if ((!_space.GetState((*j)->SourceState(), P)) && (_n == 0))
								{
									_logstream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*_i)->Name() << "\"." << std::endl;
									continue;
								}

								// Return the yield for this transition
								_datastream << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojp * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojm * (*j)->Rate() * P * rho0)) << " ";
							}
						}
						else if (_cidsp == false)
						{
							// Return the yield for this state - note that no reaction rates are included here.
							_datastream << std::real(arma::trace(_eigen_vec * Iprojx * rho0 * _eigen_vec.t())) << " ";
							_datastream << std::real(arma::trace(_eigen_vec * Iprojy * rho0 * _eigen_vec.t())) << " ";
							_datastream << std::real(arma::trace(_eigen_vec * Iprojz * rho0 * _eigen_vec.t())) << " ";
							_datastream << std::real(arma::trace(_eigen_vec * Iprojp * rho0 * _eigen_vec.t())) << " ";
							_datastream << std::real(arma::trace(_eigen_vec * Iprojm * rho0 * _eigen_vec.t())) << " ";
						}
					}
				}
			}
		}
		else
		{
			if (_n == 0)
				_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		_datastream << std::endl;

		return true;
	}

	bool TaskStaticSSPowderSpectra::ProjectAndPrintOutputLineInf(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		// Get nuclei of interest for CIDNP spectrum
		arma::cx_mat Iprojx;
		arma::cx_mat Iprojy;
		arma::cx_mat Iprojz;
		arma::cx_mat Iprojp;
		arma::cx_mat Iprojm;

		std::vector<std::string> spinList;

		_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
		_datastream << "inf" << " ";
		this->WriteStandardOutput(_datastream);

		if (this->Properties()->GetList("spinlist", spinList, ','))
		{

			for (auto l = (*_i)->spins_cbegin(); l != (*_i)->spins_cend(); l++)
			{
				for (int m = 0; m < (int)spinList.size(); m++)
				{
					if ((*l)->Name() == spinList[m])
					{
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
						{
							return false;
						}

						arma::cx_mat P;

						// There are two result modes - either write results per transition  if CIDSP is true or for each defined state if CIDSP is false

						if (_cidsp == true)
						{
							// Loop through all defind transitions
							auto transitions = (*_i)->Transitions();
							for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
							{
								// Make sure that there is a state object
								if ((*j)->SourceState() == nullptr)
									continue;

								if ((!_space.GetState((*j)->SourceState(), P)))
								{
									_logstream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*_i)->Name() << "\"." << std::endl;
									continue;
								}

								// Return the yield for this transition
								_datastream << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojp * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojm * (*j)->Rate() * P * rho0)) << " ";
							}
						}
						else if (_cidsp == false)
						{
							// Return the yield for this state - note that no reaction rates are included here.
							_datastream << std::real(arma::trace(Iprojx * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojy * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojz * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojp * rho0)) << " ";
							_datastream << std::real(arma::trace(Iprojm * rho0)) << " ";
						}
					}
				}
			}
		}
		else
		{
			_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		return true;
	}

	bool TaskStaticSSPowderSpectra::ProjectAndPrintOutputLineInf(auto &_i, SpinAPI::SpinSpace &_space, arma::cx_vec &_rhovec, arma::sp_cx_mat &_eigen_vec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		// rho0 = (_eigen_vec * rho0 * _eigen_vec.t());

		// Get nuclei of interest for CIDNP spectrum
		arma::cx_mat Iprojx;
		arma::cx_mat Iprojy;
		arma::cx_mat Iprojz;
		arma::cx_mat Iprojp;
		arma::cx_mat Iprojm;

		std::vector<std::string> spinList;

		_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
		_datastream << "inf" << " ";
		this->WriteStandardOutput(_datastream);

		if (this->Properties()->GetList("spinlist", spinList, ','))
		{

			for (auto l = (*_i)->spins_cbegin(); l != (*_i)->spins_cend(); l++)
			{
				for (int m = 0; m < (int)spinList.size(); m++)
				{
					if ((*l)->Name() == spinList[m])
					{
						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
						{
							return false;
						}

						if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
						{
							return false;
						}

						Iprojx = (_eigen_vec * Iprojx * _eigen_vec.t());
						Iprojy = (_eigen_vec * Iprojy * _eigen_vec.t());
						Iprojz = (_eigen_vec * Iprojz * _eigen_vec.t());
						Iprojp = (_eigen_vec * Iprojp * _eigen_vec.t());
						Iprojm = (_eigen_vec * Iprojm * _eigen_vec.t());

						arma::cx_mat P;

						// There are two result modes - either write results per transition  if CIDSP is true or for each defined state if CIDSP is false

						if (_cidsp == true)
						{
							// Loop through all defind transitions
							auto transitions = (*_i)->Transitions();
							for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
							{
								// Make sure that there is a state object
								if ((*j)->SourceState() == nullptr)
									continue;

								if ((!_space.GetState((*j)->SourceState(), P)))
								{
									_logstream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*_i)->Name() << "\"." << std::endl;
									continue;
								}

								// Return the yield for this transition
								_datastream << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojp * (*j)->Rate() * P * rho0)) << " ";
								_datastream << std::real(arma::trace(Iprojm * (*j)->Rate() * P * rho0)) << " ";
							}
						}
						else if (_cidsp == false)
						{
							// Return the yield for this state - note that no reaction rates are included here.
							_datastream << std::setprecision(12) << std::real(arma::trace(Iprojx * rho0)) << " ";
							_datastream << std::setprecision(12) << std::real(arma::trace(Iprojy * rho0)) << " ";
							_datastream << std::setprecision(12) << std::real(arma::trace(Iprojz * rho0)) << " ";
							_datastream << std::setprecision(12) << std::real(arma::trace(Iprojp * rho0)) << " ";
							_datastream << std::setprecision(12) << std::real(arma::trace(Iprojm * rho0)) << " ";
						}
					}
				}
			}
		}
		else
		{
			_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		return true;
	}

	bool TaskStaticSSPowderSpectra::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		arma::mat R1 = {
			{std::cos(_alpha), -std::sin(_alpha), 0.0},
			{std::sin(_alpha), std::cos(_alpha), 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{std::cos(_beta), 0.0, std::sin(_beta)},
			{0.0, 1.0, 0.0},
			{-std::sin(_beta), 0.0, std::cos(_beta)}};

		arma::mat R3 = {
			{std::cos(_gamma), -std::sin(_gamma), 0.0},
			{std::sin(_gamma), std::cos(_gamma), 0.0},
			{0.0, 0.0, 1.0}};

		_R = R1 * R2 * R3;

		return true;
	}

	bool TaskStaticSSPowderSpectra::CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const
	{
		std::vector<double> theta(_Npoints);
		std::vector<double> phi(_Npoints);
		std::vector<double> weight(_Npoints);

		_uniformGrid.resize(_Npoints);

		const double golden = M_PI * (1.0 + std::sqrt(5.0)); // not standart golden angle

		for (int i = 0; i < _Npoints; ++i)
		{
			double index = static_cast<double>(i) + 0.5;

			theta[i] = std::acos(1.0 - index / _Npoints);		  // hemisphere
			phi[i] = golden * index;							  // hemisphere
			weight[i] = std::sin(theta[i]) * 2 * M_PI / _Npoints; // 2 * pi for hemisphere
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

	bool TaskStaticSSPowderSpectra::CreateCustomGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_Grid) const
	{
		std::vector<double> theta(_Npoints * _Npoints);
		std::vector<double> phi(_Npoints * _Npoints);
		std::vector<double> weight(_Npoints * _Npoints);

		_Grid.resize(_Npoints * _Npoints);

		int idx = 0;
		for (int k = 0; k < _Npoints; ++k)
		{
			double u = (k + 0.5) / _Npoints; // cosine-spaced
			double th = acos(u);			 // θ

			for (int j = 0; j < _Npoints; ++j)
			{
				double ph = (j + 0.5) * (M_PI / 2.0) / _Npoints; // uniform φ

				theta[idx] = th;
				phi[idx] = ph;

				weight[idx] = (M_PI / 2.0 / _Npoints) * (1.0 / _Npoints); // Δφ * Δ(cosθ)
				_Grid[idx] = {theta[idx], phi[idx], weight[idx]};
				idx++;
			}
		}

		return true;
	}

	// -----------------------------------------------------
}
