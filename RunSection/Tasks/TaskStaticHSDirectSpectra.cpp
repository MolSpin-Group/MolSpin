/////////////////////////////////////////////////////////////////////////
// TaskStaticHSDirectSpectra implementation (RunSection module) by Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "TaskStaticHSDirectSpectra.h"
#include "Transition.h"
#include "Operator.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "Interaction.h"
#include "Pulse.h"
#include <cmath>
#include <iomanip> // std::setprecision
#include <numeric>
#include <sstream>
#include <omp.h>

namespace RunSection
{
	namespace
	{
		double TraceSparseDense(const arma::sp_cx_mat &A, const arma::cx_mat &B)
		{
			arma::cx_double sum = arma::cx_double(0.0, 0.0);
			for (auto it = A.begin(); it != A.end(); ++it)
			{
				sum += (*it) * B(it.col(), it.row());
			}
			return std::real(sum);
		}

		std::string TrimCopy(const std::string &value)
		{
			const auto begin = value.find_first_not_of(" \t\n\r");
			if (begin == std::string::npos)
				return std::string();

			const auto end = value.find_last_not_of(" \t\n\r");
			return value.substr(begin, end - begin + 1);
		}

		bool BuildInitialDensityMatrix(const SpinAPI::system_ptr &system,
									   SpinAPI::SpinSpace &space,
									   const std::shared_ptr<MSDParser::ObjectParser> &task_properties,
									   arma::cx_mat &rho0,
									   std::ostream &log_stream)
		{
			auto initial_states = system->InitialState();
			if (initial_states.empty() && task_properties != nullptr)
			{
				std::string legacy_initial_state;
				if (task_properties->Get("initialstate", legacy_initial_state))
				{
					log_stream << "Using legacy task-level initialstate input \"" << legacy_initial_state
							   << "\". Prefer spin-system initialstate for new inputs." << std::endl;

					std::istringstream stream(legacy_initial_state);
					for (std::string token; std::getline(stream, token, ',');)
					{
						token = TrimCopy(token);
						if (token.empty())
							continue;

						if (token == "Thermal" || token == "thermal")
						{
							initial_states.push_back(nullptr);
							continue;
						}

						bool found = false;
						auto states = system->States();
						for (auto state = states.cbegin(); state != states.cend(); ++state)
						{
							if ((*state)->Name() == token)
							{
								initial_states.push_back(*state);
								found = true;
								break;
							}
						}

						if (!found)
						{
							log_stream << "Failed to resolve legacy initial state \"" << token
									   << "\" on SpinSystem \"" << system->Name() << "\"." << std::endl;
						}
					}
				}
			}

			if (initial_states.empty())
				return false;

			std::vector<double> weights = system->Weights();
			bool use_weights = (initial_states.size() > 1 && weights.size() == initial_states.size());
			if (weights.size() > 1 && !use_weights)
			{
				log_stream << "Ignoring initial-state weights for SpinSystem \"" << system->Name()
						   << "\" because the number of weights does not match the number of initial states." << std::endl;
			}

			if (use_weights)
			{
				const double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
				if (sum_weights > 0.0)
				{
					for (double &weight : weights)
						weight /= sum_weights;
				}
				else
				{
					log_stream << "Ignoring non-positive initial-state weights for SpinSystem \"" << system->Name()
							   << "\"." << std::endl;
					use_weights = false;
				}
			}

			bool assigned = false;
			for (size_t idx = 0; idx < initial_states.size(); ++idx)
			{
				arma::cx_mat tmp_rho0;
				if (initial_states[idx] == nullptr)
				{
					auto thermal_hamiltonian_list = system->ThermalHamiltonianList();
					double temperature = system->Temperature();
					log_stream << "Initial state = thermal, T = " << temperature << " K." << std::endl;
					if (!space.GetThermalState(space, temperature, thermal_hamiltonian_list, tmp_rho0))
					{
						log_stream << "Failed to obtain thermal initial state for SpinSystem \"" << system->Name() << "\"." << std::endl;
						continue;
					}
				}
				else
				{
					if (!space.GetState(initial_states[idx], tmp_rho0))
					{
						log_stream << "Failed to obtain projection matrix onto state \"" << initial_states[idx]->Name()
								   << "\", initial state of SpinSystem \"" << system->Name() << "\"." << std::endl;
						continue;
					}
				}

				const double weight = use_weights ? weights[idx] : 1.0;
				if (!assigned)
				{
					rho0 = weight * tmp_rho0;
					assigned = true;
				}
				else
				{
					rho0 += weight * tmp_rho0;
				}
			}

			if (!assigned)
				return false;

			const arma::cx_double rho_trace = arma::trace(rho0);
			if (std::abs(rho_trace) == 0.0)
				return false;

			rho0 /= rho_trace;
			return true;
		}

		arma::cx_mat FactorizeDensityMatrix(const arma::cx_mat &rho0, std::ostream &log_stream)
		{
			const arma::cx_mat hermitian_rho0 = 0.5 * (rho0 + rho0.st());
			arma::vec eigenvalues;
			arma::cx_mat eigenvectors;
			if (!arma::eig_sym(eigenvalues, eigenvectors, hermitian_rho0))
			{
				log_stream << "Failed to diagonalize the initial density matrix in Hilbert space." << std::endl;
				return arma::cx_mat();
			}

			const double max_eigenvalue = eigenvalues.is_empty() ? 0.0 : std::abs(eigenvalues.max());
			const double tolerance = std::max(1.0e-12, 1.0e-10 * max_eigenvalue);
			if (eigenvalues.min() < -tolerance)
			{
				log_stream << "Initial density matrix has significantly negative eigenvalues (" << eigenvalues.min()
						   << "). Cannot construct Hilbert-space factorization." << std::endl;
				return arma::cx_mat();
			}

			arma::uvec keep = arma::find(eigenvalues > tolerance);
			if (keep.is_empty())
			{
				log_stream << "Initial density matrix is numerically rank-zero after factorization." << std::endl;
				return arma::cx_mat();
			}

			arma::cx_mat B(rho0.n_rows, keep.n_elem, arma::fill::zeros);
			for (arma::uword col = 0; col < keep.n_elem; ++col)
			{
				const arma::uword idx = keep(col);
				B.col(col) = std::sqrt(eigenvalues(idx)) * eigenvectors.col(idx);
			}

			return B;
		}
	}

	// -----------------------------------------------------
	// TaskStaticHSDirectSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSDirectSpectra::TaskStaticHSDirectSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticHSDirectSpectra::~TaskStaticHSDirectSpectra()
	{
	}
	// -----------------------------------------------------
	// TaskStaticHSDirectSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticHSDirectSpectra::RunLocal()
	{
		this->Log() << "Running task StaticHS-Direct-Spectra." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++) // iteration through all spin systems, in this case (or usually), this is one
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);
			space.SetReactionOperatorType(this->reactionOperators);

			arma::cx_mat rho0;
			if (!BuildInitialDensityMatrix(*i, space, this->Properties(), rho0, this->Log()))
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no valid initial state could be constructed." << std::endl;
				continue;
			}

			arma::cx_mat Binitial = FactorizeDensityMatrix(rho0, this->Log());
			if (Binitial.is_empty())
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" because the initial density matrix could not be factorized." << std::endl;
				continue;
			}

			const int dim = static_cast<int>(rho0.n_rows);
			this->Log() << "Hilbert Space Size " << dim << " x " << dim << std::endl;
			this->Log() << "Initial-state rank " << Binitial.n_cols << std::endl;

			// Get Information about the polarization of choice
			bool CIDSP = false;
			if (!this->Properties()->Get("cidsp", CIDSP))
			{
				this->Log() << "Failed to obtain input for CIDSP. Using default false." << std::endl;
			}

			// Get projectors of interest of the spectrum
			arma::sp_cx_mat Iprojx;
			arma::sp_cx_mat Iprojy;
			arma::sp_cx_mat Iprojz;

			std::vector<std::string> spinList;
			int m;

			// Check transitions, rates and projection operators
			auto transitions = (*i)->Transitions();
			arma::sp_cx_mat P;
			int num_transitions = 0;

			int projection_counter = 0;
			std::map<int, arma::sp_cx_mat> Operators;
			std::vector<arma::sp_cx_mat> OperatorsSparse;
			std::vector<arma::cx_mat> OperatorsDense;
			arma::vec rates(1, 1);

			// Getting the projection operators
			if (this->Properties()->GetList("spinlist", spinList, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					for (m = 0; m < (int)spinList.size(); m++)
					{
						if ((*l)->Name() == spinList[m])
						{
							if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sx()), (*l), Iprojx))
							{
								return false;
							}
								if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sy()), (*l), Iprojy))
							{
								return false;
							}
								if (!space.CreateOperator(arma::conv_to<arma::sp_cx_mat>::from((*l)->Sz()), (*l), Iprojz))
							{
								return false;
							}

							if (CIDSP == true)
							{
								// Gather rates and operators
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									if ((*j)->SourceState() == nullptr)
										continue;
									if (!space.GetState((*j)->SourceState(), P))
									{
										this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
										return 1;
									}
									if (num_transitions != 0)
									{
										rates.insert_rows(num_transitions, 1);
									}

									Operators[projection_counter] = (*j)->Rate() * Iprojx * P;
									projection_counter += 1;
									Operators[projection_counter] = (*j)->Rate() * Iprojy * P;
									projection_counter += 1;
									Operators[projection_counter] = (*j)->Rate() * Iprojz * P;
									projection_counter += 1;

									rates(num_transitions) = (*j)->Rate();
									num_transitions++;
								}
							}
							else
							{
								// Gather rates and operators
								Operators[projection_counter] = Iprojx;
								projection_counter += 1;
								Operators[projection_counter] = Iprojy;
								projection_counter += 1;
								Operators[projection_counter] = Iprojz;
								projection_counter += 1;
							
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									if ((*j)->SourceState() == nullptr)
										continue;

									if (num_transitions != 0)
									{
										rates.insert_rows(num_transitions, 1);
									}

									rates(num_transitions) = (*j)->Rate();
									num_transitions++;
								}
							}
						}	
					}
				}
			}

			OperatorsSparse.resize(projection_counter);
			double total_nnz = 0.0;
			double total_size = 0.0;
			for (const auto &entry : Operators)
			{
				OperatorsSparse[entry.first] = entry.second;
				total_nnz += static_cast<double>(entry.second.n_nonzero);
				total_size += static_cast<double>(entry.second.n_rows) * entry.second.n_cols;
			}
			bool use_sparse_ops = (total_size > 0.0) && ((total_nnz / total_size) < 0.1);
			if (!use_sparse_ops)
			{
				OperatorsDense.resize(projection_counter);
				for (const auto &entry : Operators)
				{
					OperatorsDense[entry.first] = arma::cx_mat(entry.second);
				}
			}
			else if (projection_counter > 0)
			{
				this->Log() << "Using sparse operators for expectation values." << std::endl;
			}

			// Get the Hamiltonian
			arma::sp_cx_mat K;
			K.zeros(dim, dim);

			for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
			{
				if ((*j)->SourceState() == nullptr)
					continue;
				space.GetState((*j)->SourceState(), P);
				K += (*j)->Rate() / 2 * P;
			}
			SpinAPI::HilbertRelaxationCache relaxation_cache;
			bool use_density_matrix = false;
			for (auto j = (*i)->operators_cbegin(); j != (*i)->operators_cend(); j++)
			{
				if (space.RelaxationOperator((*j), relaxation_cache))
				{
					use_density_matrix = true;
					this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to Hilbert-space propagation.\n";
				}
			}
			if (use_density_matrix)
			{
				this->Log() << "Relaxation operators detected. Using density-matrix propagation in Hilbert space." << std::endl;
			}

			// Setting or calculating total time.
			double totaltime = this->totaltime;
			double inputTotaltime = 0.0;
			if (this->Properties()->Get("totaltime", inputTotaltime))
			{
				if (std::isfinite(inputTotaltime) && inputTotaltime >= 0.0)
				{
					totaltime = inputTotaltime;
				}
				else
				{
					this->Log() << "# ERROR: invalid total time!" << std::endl;
					return false;
				}
			}

			// Setting timestep
			double dt = this->timestep;
			double inputTimestep = 0.0;
			if (this->Properties()->Get("timestep", inputTimestep))
			{
				if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
				{
					dt = inputTimestep;
				}
				else
				{
					this->Log() << "WARNING: Undefined timestep. Using default 0.1 ns." << std::endl;
					dt = 0.1;
				}
			}

			this->Log() << "Time step is chosen as " << dt << " ns." << std::endl;

			// Number of time propagation steps
			int num_steps = std::ceil(totaltime / dt);
			this->Log() << "Number of time propagation steps: " << num_steps << "." << std::endl;
			if (num_steps == 0)
			{
				num_steps = 1;
				this->Log() << "Changing number of propagation steps to: " << num_steps << " in order to propagate one step." << std::endl;
			}

			// Choose Propagation Method and other parameters
			std::string propmethod;
			this->Properties()->Get("propagationmethod", propmethod);

			std::string precision;
			this->Properties()->Get("precision", precision);

			int krylovsize;
			this->Properties()->Get("krylovsize", krylovsize);

			double krylovtol;
			this->Properties()->Get("krylovtol", krylovtol);

			if (propmethod == "autoexpm")
			{
				this->Log() << "Autoexpm is chosen as the propagation method." << std::endl;
				if (precision == "double")
				{
					this->Log() << "Double precision is chosen for the autoexpm method." << std::endl;
				}
				else if (precision == "single")
				{
					this->Log() << "Single precision is chosen for the autoexpm method." << std::endl;
				}
				else if (precision == "half")
				{
					this->Log() << "Half precision is chosen for the autoexpm method." << std::endl;
				}
				else
				{
					this->Log() << "Undefined precision for autoexpm method. Using single precision." << std::endl;
					precision = "single";
				}
			}
			else if (propmethod == "krylov")
			{
				if (krylovsize > 0)
				{
					this->Log() << "Krylov basis size is chosen as " << krylovsize << "." << std::endl;
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
				else
				{
					this->Log() << "Undefined size of the krylov subspace. Using the default size of 16." << std::endl;
					krylovsize = 16;
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
			}
			else if (propmethod == "rk4" || propmethod == "explicit")
			{
				this->Log() << "Explicit RK4 is chosen as the propagation method." << std::endl;
			}
			else
			{
				this->Log() << "WARNING: Undefined propagation method. Using normal exponential method." << std::endl;
				propmethod = "normal";
			}

			bool relax_use_split_expm = false;
			arma::cx_mat K_dense;
			if (use_density_matrix)
			{
				relax_use_split_expm = (propmethod != "rk4" && propmethod != "explicit");
				if (relax_use_split_expm)
				{
					K_dense = arma::cx_mat(K);
					this->Log() << "Relaxation operators active; using split-exponential propagation (Hamiltonian expm + RK4 relaxation)." << std::endl;
					if (propmethod != "normal")
					{
						this->Log() << "Note: propagationmethod is ignored for relaxation; use propagationmethod = rk4 to force explicit RK4." << std::endl;
					}
				}
				else
				{
					this->Log() << "Relaxation operators active; using explicit RK4 density-matrix propagation." << std::endl;
				}
			}

			// Powder averaging options (shared keywords with superspace powder task)
			std::string Method = "timeevo";
			if (!this->Properties()->Get("method", Method))
			{
				this->Log() << "Failed to obtain an input for a Method. Please specify method = timeinf or method = timeevo. Using timeevo by default." << std::endl;
				Method = "timeevo";
			}
			bool method_timeevo = Method.compare("timeevo") == 0;
			bool method_timeinf = Method.compare("timeinf") == 0;
			if (!method_timeevo && !method_timeinf)
			{
				this->Log() << "Method \"" << Method << "\" is not supported for Hilbert space spectra. Using timeevo." << std::endl;
				Method = "timeevo";
				method_timeevo = true;
				method_timeinf = false;
			}

			// Read if the result should be integrated or not.
			bool integration = false;
			if (!this->Properties()->Get("integration", integration))
			{
				this->Log() << "Failed to obtain input for integration. Please use integration = true/false. Using integration = false by default." << std::endl;
			}

			// Read integrationwindow from the input file
			std::string Integrationwindow;
			if (!this->Properties()->Get("integrationtimeframe", Integrationwindow))
			{
				this->Log() << "Failed to obtain input for integrationtimeframe. Please choose integrationtimeframe = pulse / freeevo / full. Using freeevo by default." << std::endl;
				Integrationwindow = "freeevo";
			}
			this->Log() << "Timewindow for the propagation integration: " << Integrationwindow << std::endl;

			// Read printtimeframe from the input file
			std::string Timewindow;
			if (!this->Properties()->Get("printtimeframe", Timewindow))
			{
				this->Log() << "Failed to obtain input for printtimeframe. Please choose printtimeframe = pulse / freeevo / full. Using full by default." << std::endl;
				Timewindow = "full";
			}
			this->Log() << "Timewindow for the propagation printing: " << Timewindow << std::endl;

			bool print_pulses = (Timewindow.compare("freeevo") != 0);
			bool print_freeevo = (Timewindow.compare("pulse") != 0);
			bool integrate_pulses = integration && (Integrationwindow.compare("freeevo") != 0);
			bool integrate_freeevo = integration && (Integrationwindow.compare("pulse") != 0);

			int numPoints = 0;
			bool hasPowderPoints = this->Properties()->Get("powdersamplingpoints", numPoints);
			if (!hasPowderPoints)
			{
				this->Log() << "No powdersamplingpoints provided. Powder averaging is disabled by default." << std::endl;
				numPoints = 0;
			}
			if (numPoints < 1)
			{
				numPoints = 0;
				if (hasPowderPoints)
				{
					this->Log() << "Powder averaging disabled (powdersamplingpoints <= 0)." << std::endl;
				}
			}

			std::vector<std::tuple<double, double, double>> grid;
			if (numPoints > 0)
			{
				if (!this->CreateUniformGrid(numPoints, grid))
				{
					this->Log() << "Failed to obtain a powder grid." << std::endl;
				}
				else if (numPoints > 1)
				{
					this->Log() << "Using powder averaging with " << numPoints << " orientations." << std::endl;
				}
			}
			else
			{
				grid.push_back({0.0, 0.0, 1.0});
			}

			std::vector<std::string> HamiltonianH0list;
			std::vector<std::string> HamiltonianH1list;
			bool hasH0list = this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ',');
			bool hasH1list = this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ',');

			// Read a pulse sequence from the input
			std::vector<std::tuple<std::string, double>> Pulsesequence;
			bool hasPulseSequence = this->Properties()->GetPulseSequence("pulsesequence", Pulsesequence);
			if (hasPulseSequence)
			{
				this->Log() << "Pulse sequence:" << std::endl;
			}

			std::vector<double> pulse_times;
			std::vector<double> pulse_dts;
			bool has_pulse_output = false;
			bool pulse_has_initial_step = false;
			double pulse_total_time = 0.0;
			if (print_pulses && hasPulseSequence)
			{
				double current_time = 0.0;
				bool include_initial_step = true;

				for (const auto &seq : Pulsesequence)
				{
					std::string pulse_name = std::get<0>(seq);
					double timerelaxation = std::get<1>(seq);

					SpinAPI::pulse_ptr pulse_ptr = nullptr;
					for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
					{
						if ((*pulse)->Name().compare(pulse_name) == 0)
						{
							pulse_ptr = *pulse;
							break;
						}
					}

					if (pulse_ptr == nullptr)
					{
						this->Log() << "Pulse \"" << pulse_name << "\" was not found in SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					double pulse_dt = pulse_ptr->Timestep();
					if (!std::isfinite(pulse_dt) || pulse_dt <= 0.0)
					{
						this->Log() << "Invalid timestep for pulse \"" << pulse_name << "\". Skipping pulse timeline generation." << std::endl;
						continue;
					}

					if (pulse_ptr->Type() == SpinAPI::PulseType::LongPulseStaticField || pulse_ptr->Type() == SpinAPI::PulseType::LongPulse)
					{
						unsigned int steps = static_cast<unsigned int>(std::abs(pulse_ptr->Pulsetime() / pulse_dt));

						if (include_initial_step)
						{
							pulse_times.push_back(current_time);
							pulse_dts.push_back(0.0);
							include_initial_step = false;
						}

						for (unsigned int n = 1; n <= steps; ++n)
						{
							current_time += pulse_dt;
							pulse_times.push_back(current_time);
							pulse_dts.push_back(pulse_dt);
						}
					}

					if (timerelaxation != 0.0)
					{
						unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / pulse_dt));
						for (unsigned int n = 1; n <= steps; ++n)
						{
							current_time += pulse_dt;
							pulse_times.push_back(current_time);
							pulse_dts.push_back(pulse_dt);
						}
					}
				}

				pulse_total_time = current_time;
				has_pulse_output = !pulse_times.empty();
				pulse_has_initial_step = has_pulse_output && (pulse_dts.front() == 0.0);
			}

			arma::vec time;
			if (method_timeevo)
			{
				time.set_size(num_steps);
				for (int k = 0; k < num_steps; ++k)
				{
					time(k) = k * dt;
				}
			}

			arma::mat ExptValues; // reused for timeevo
			if (method_timeevo)
			{
				ExptValues.zeros(num_steps, projection_counter);
			}
			arma::cx_mat rho_integrated;
			if (method_timeinf)
			{
				rho_integrated.zeros(dim, dim);
			}

			size_t grid_size = grid.size();
			int nthreads = 1;
#ifdef _OPENMP
			nthreads = omp_get_max_threads();
#endif

			std::vector<arma::mat> ExptValuesPartial;
			if (method_timeevo)
			{
				ExptValuesPartial.resize(nthreads);
				for (auto &m : ExptValuesPartial)
				{
					m.zeros(num_steps, projection_counter);
				}
			}

			std::vector<arma::mat> ExptValuesPulsePartial;
			if (has_pulse_output)
			{
				ExptValuesPulsePartial.resize(nthreads);
				for (auto &m : ExptValuesPulsePartial)
				{
					m.zeros(pulse_times.size(), projection_counter);
				}
			}

			std::vector<arma::cx_mat> rho_integrated_partial;
			if (method_timeinf)
			{
				rho_integrated_partial.resize(nthreads);
				for (auto &m : rho_integrated_partial)
				{
					m.zeros(dim, dim);
				}
			}

			arma::cx_mat Iden_dense;
			if (method_timeinf)
			{
				Iden_dense = arma::eye<arma::cx_mat>(dim, dim);
			}

			arma::cx_mat relaxation_super;
			if (method_timeinf && use_density_matrix)
			{
				space.RelaxationSuperoperatorHilbert(relaxation_cache, relaxation_super);
			}

			SpinAPI::SpinSpace base_space(space);
			base_space.SetReactionOperatorType(this->reactionOperators);
			base_space.UseSuperoperatorSpace(false);

			std::vector<SpinAPI::SpinSpace> spaces;
			spaces.resize(nthreads);
			for (int t = 0; t < nthreads; ++t)
			{
				spaces[t] = base_space;
			}

			if (grid_size > static_cast<size_t>(std::numeric_limits<int>::max()))
				throw std::runtime_error("Too many grid points for OpenMP loop indexing.");

			const int omp_grid_size = static_cast<int>(grid_size);
#pragma omp parallel for schedule(static) if (grid_size > 1)
			for (int grid_num = 0; grid_num < omp_grid_size; ++grid_num)
			{
				int tid = 0;
#ifdef _OPENMP
				tid = omp_get_thread_num();
#endif
				SpinAPI::SpinSpace &space_thread = spaces[tid];

				const auto &grid_point = grid[grid_num];
				double theta, phi, weight;
				std::tie(theta, phi, weight) = grid_point;

				arma::mat Rot_mat = arma::eye<arma::mat>(3, 3);
				double gamma = 0.0;
				if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
				{
					this->Log() << "Failed to obtain rotation matrix for powder orientation." << std::endl;
				}

				arma::sp_cx_mat H;
				if (hasH0list)
				{
					arma::sp_cx_mat H0;
					if (!space_thread.BaseHamiltonianRotated_SA(HamiltonianH0list, Rot_mat, H0))
					{
						this->Log() << "Failed to obtain rotated Hamiltonian for SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}
					H = H0;

					if (hasH1list)
					{
						arma::sp_cx_mat H1;
						if (!space_thread.ThermalHamiltonian(HamiltonianH1list, H1))
						{
							this->Log() << "Failed to obtain Hamiltonian H1 in Hilbert Space." << std::endl;
							continue;
						}

						H += H1;
					}
				}
				else
				{
					if (!space_thread.Hamiltonian(H))
					{
						this->Log() << "Failed to obtain the Hamiltonian in Hilbert Space." << std::endl;
						continue;
					}
				}

				if (use_density_matrix)
				{
					arma::cx_mat rho = rho0;
					int dim = static_cast<int>(rho.n_rows);

					arma::cx_mat work_left(dim, dim, arma::fill::zeros);
					arma::cx_mat work_right(dim, dim, arma::fill::zeros);
					arma::cx_mat relax(dim, dim, arma::fill::zeros);
					arma::cx_mat k1(dim, dim, arma::fill::zeros);
					arma::cx_mat k2(dim, dim, arma::fill::zeros);
					arma::cx_mat k3(dim, dim, arma::fill::zeros);
					arma::cx_mat k4(dim, dim, arma::fill::zeros);
					arma::cx_mat tmp_state(dim, dim, arma::fill::zeros);
					arma::cx_mat rk_accum(dim, dim, arma::fill::zeros);

					bool use_dense_H = false;
					arma::cx_mat H_dense;
					double H_density = 0.0;
					if (relax_use_split_expm)
					{
						H_dense = arma::cx_mat(H);
						use_dense_H = true;
					}
					else if (H.n_rows > 0 && H.n_cols > 0)
					{
						H_density = static_cast<double>(H.n_nonzero) / (static_cast<double>(H.n_rows) * static_cast<double>(H.n_cols));
						if (H_density > 0.15)
						{
							H_dense = arma::cx_mat(H);
							use_dense_H = true;
						}
					}

					const arma::cx_double imag_unit(0.0, 1.0);

					auto record_expectation_rho = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state) {
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							double val = use_sparse_ops ? TraceSparseDense(OperatorsSparse[idx], state)
														: std::real(arma::trace(OperatorsDense[idx] * state));
							target(row_index, idx) = val;
						}
					};

					auto drho = [&](const arma::cx_mat &state, const arma::sp_cx_mat &H_total, const arma::cx_mat *H_dense_ptr, arma::cx_mat &out) {
						if (H_dense_ptr != nullptr)
						{
							work_left = (*H_dense_ptr) * state;
							work_right = state * (*H_dense_ptr);
						}
						else
						{
							work_left = H_total * state;
							work_right = state * H_total;
						}
						out = -imag_unit * (work_left - work_right);
						work_left = K * state;
						work_right = state * K;
						out -= (work_left + work_right);
						space_thread.ApplyRelaxationHilbert(relaxation_cache, state, relax);
						out += relax;
					};

					auto rk4_step = [&](arma::cx_mat &state, const arma::sp_cx_mat &H_total, const arma::cx_mat *H_dense_ptr, double step_dt) {
						drho(state, H_total, H_dense_ptr, k1);
						tmp_state = state;
						tmp_state += (0.5 * step_dt) * k1;
						drho(tmp_state, H_total, H_dense_ptr, k2);
						tmp_state = state;
						tmp_state += (0.5 * step_dt) * k2;
						drho(tmp_state, H_total, H_dense_ptr, k3);
						tmp_state = state;
						tmp_state += step_dt * k3;
						drho(tmp_state, H_total, H_dense_ptr, k4);
						rk_accum = k1;
						rk_accum += 2.0 * k2;
						rk_accum += 2.0 * k3;
						rk_accum += k4;
						state += (step_dt / 6.0) * rk_accum;
					};

					auto relax_rhs = [&](const arma::cx_mat &state, arma::cx_mat &out) {
						space_thread.ApplyRelaxationHilbert(relaxation_cache, state, out);
					};

					auto rk4_relax_step = [&](arma::cx_mat &state, double step_dt) {
						relax_rhs(state, k1);
						tmp_state = state;
						tmp_state += (0.5 * step_dt) * k1;
						relax_rhs(tmp_state, k2);
						tmp_state = state;
						tmp_state += (0.5 * step_dt) * k2;
						relax_rhs(tmp_state, k3);
						tmp_state = state;
						tmp_state += step_dt * k3;
						relax_rhs(tmp_state, k4);
						rk_accum = k1;
						rk_accum += 2.0 * k2;
						rk_accum += 2.0 * k3;
						rk_accum += k4;
						state += (step_dt / 6.0) * rk_accum;
					};

					auto build_unitary_half = [&](const arma::cx_mat &H_total_dense, double step_dt, arma::cx_mat &U_half, arma::cx_mat &U_half_st) {
						arma::cx_mat A_dense = -imag_unit * H_total_dense - K_dense;
						U_half = arma::expmat(A_dense * (0.5 * step_dt));
						U_half_st = U_half.st();
					};

					auto apply_unitary_half = [&](arma::cx_mat &state, const arma::cx_mat &U_half, const arma::cx_mat &U_half_st) {
						work_left = U_half * state;
						state = work_left * U_half_st;
					};

					auto split_step = [&](arma::cx_mat &state, const arma::cx_mat &U_half, const arma::cx_mat &U_half_st, double step_dt) {
						apply_unitary_half(state, U_half, U_half_st);
						rk4_relax_step(state, step_dt);
						apply_unitary_half(state, U_half, U_half_st);
					};

					size_t pulse_step_index = 0;
					arma::mat ExptValuesPulseOrientation;
					if (has_pulse_output)
					{
						ExptValuesPulseOrientation.zeros(pulse_times.size(), projection_counter);
						if (pulse_has_initial_step)
						{
							record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
							pulse_step_index = 1;
						}
					}

					if (hasPulseSequence)
					{
						for (const auto &seq : Pulsesequence)
						{
							if (grid_num == 0)
							{
								this->Log() << std::get<0>(seq) << ", " << std::get<1>(seq) << std::endl;
							}

							std::string pulse_name = std::get<0>(seq);
							double timerelaxation = std::get<1>(seq);

							for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
							{
								if ((*pulse)->Name().compare(pulse_name) == 0)
								{
									double pulse_dt = (*pulse)->Timestep();
									if (!std::isfinite(pulse_dt) || pulse_dt <= 0.0)
									{
										this->Log() << "Invalid timestep for pulse \"" << (*pulse)->Name() << "\". Skipping pulse propagation." << std::endl;
										continue;
									}

									if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
									{
										arma::sp_cx_mat pulse_operator;
										if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
										{
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
											continue;
										}
										arma::cx_mat U = arma::cx_mat(pulse_operator);
										rho = U * rho * U.st();
									}
									else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulseStaticField)
									{
										arma::sp_cx_mat pulse_operator;
										if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
										{
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
											continue;
										}

										unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / pulse_dt));
										if (relax_use_split_expm)
										{
											arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator);
											arma::cx_mat U_half;
											arma::cx_mat U_half_st;
											build_unitary_half(H_pulse_dense, pulse_dt, U_half, U_half_st);
											for (unsigned int n = 1; n <= steps; ++n)
											{
												split_step(rho, U_half, U_half_st, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
										else if (use_dense_H)
										{
											arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator);
											for (unsigned int n = 1; n <= steps; ++n)
											{
												rk4_step(rho, H, &H_pulse_dense, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
										else
										{
											arma::sp_cx_mat H_pulse = H + pulse_operator;
											for (unsigned int n = 1; n <= steps; ++n)
											{
												rk4_step(rho, H_pulse, nullptr, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
									}
									else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
									{
										arma::sp_cx_mat pulse_operator;
										if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
										{
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
											continue;
										}

										unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / pulse_dt));
										if (relax_use_split_expm)
										{
											for (unsigned int n = 1; n <= steps; ++n)
											{
												double t = n * pulse_dt;
												double pulse_factor = std::cos((*pulse)->Frequency() * t);
												arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator) * pulse_factor;
												arma::cx_mat U_half;
												arma::cx_mat U_half_st;
												build_unitary_half(H_pulse_dense, pulse_dt, U_half, U_half_st);
												split_step(rho, U_half, U_half_st, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
										else if (use_dense_H)
										{
											for (unsigned int n = 1; n <= steps; ++n)
											{
												double t = n * pulse_dt;
												double pulse_factor = std::cos((*pulse)->Frequency() * t);
												arma::cx_mat H_pulse_dense = H_dense + arma::cx_mat(pulse_operator) * pulse_factor;
												rk4_step(rho, H, &H_pulse_dense, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
										else
										{
											for (unsigned int n = 1; n <= steps; ++n)
											{
												double t = n * pulse_dt;
												double t_mid = t + 0.5 * pulse_dt;
												double pulse_factor = std::cos((*pulse)->Frequency() * t_mid);
												arma::sp_cx_mat H_pulse = H + pulse_operator * pulse_factor;
												rk4_step(rho, H_pulse, nullptr, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
									}
									else
									{
										this->Log() << "Not implemented yet, sorry." << std::endl;
									}

									unsigned int relax_steps = static_cast<unsigned int>(std::abs(timerelaxation / pulse_dt));
									if (relax_steps > 0)
									{
										if (relax_use_split_expm)
										{
											arma::cx_mat U_half;
											arma::cx_mat U_half_st;
											build_unitary_half(H_dense, pulse_dt, U_half, U_half_st);
											for (unsigned int n = 1; n <= relax_steps; ++n)
											{
												split_step(rho, U_half, U_half_st, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
										else
										{
											for (unsigned int n = 1; n <= relax_steps; ++n)
											{
												rk4_step(rho, H, use_dense_H ? &H_dense : nullptr, pulse_dt);

												if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
												{
													record_expectation_rho(ExptValuesPulseOrientation, pulse_step_index, rho);
													++pulse_step_index;
												}
											}
										}
									}
								}
							}
						}
					}

					if (has_pulse_output && pulse_step_index != ExptValuesPulseOrientation.n_rows)
					{
						if (grid_num == 0)
						{
							this->Log() << "Warning: Pulse output step count mismatch. Expected " << ExptValuesPulseOrientation.n_rows
										<< ", recorded " << pulse_step_index << "." << std::endl;
						}
					}

					arma::mat ExptValuesOrientation;
					if (method_timeevo)
					{
						ExptValuesOrientation.zeros(num_steps, projection_counter);
						if (relax_use_split_expm)
						{
							arma::cx_mat U_half;
							arma::cx_mat U_half_st;
							build_unitary_half(H_dense, dt, U_half, U_half_st);
							for (int k = 0; k < num_steps; ++k)
							{
								record_expectation_rho(ExptValuesOrientation, k, rho);
								split_step(rho, U_half, U_half_st, dt);
							}
						}
						else
						{
							for (int k = 0; k < num_steps; ++k)
							{
								record_expectation_rho(ExptValuesOrientation, k, rho);
								rk4_step(rho, H, use_dense_H ? &H_dense : nullptr, dt);
							}
						}
					}

					if (method_timeinf)
					{
						arma::cx_mat rho0mat = rho;
						arma::cx_mat A_dense = -arma::cx_double(0.0, 1.0) * arma::cx_mat(H) - arma::cx_mat(K);
						arma::cx_mat A_star = arma::conj(A_dense);

						arma::cx_mat L = arma::kron(A_star, Iden_dense) + arma::kron(Iden_dense, A_dense);
						if (!relaxation_super.is_empty())
						{
							L += relaxation_super;
						}
						arma::cx_vec rhs = arma::vectorise(-rho0mat);
						arma::cx_vec sol = arma::solve(L, rhs);
						if (sol.is_empty())
						{
							this->Log() << "Failed to solve timeinf Lyapunov equation in Hilbert space." << std::endl;
							continue;
						}
						arma::cx_mat X = arma::reshape(sol, rho0mat.n_rows, rho0mat.n_cols);

						rho_integrated_partial[tid] += weight * X;
					}

					if (has_pulse_output)
						ExptValuesPulsePartial[tid] += weight * ExptValuesPulseOrientation;

					if (method_timeevo)
						ExptValuesPartial[tid] += weight * ExptValuesOrientation;

					continue;
				}

				arma::cx_mat B = Binitial;

				auto record_expectation = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state) {
					arma::cx_mat state_conj = arma::conj(state);
					for (int idx = 0; idx < projection_counter; ++idx)
					{
						arma::cx_mat OB;
						if (use_sparse_ops)
						{
							OB = OperatorsSparse[idx] * state;
						}
						else
						{
							OB = OperatorsDense[idx] * state;
						}
						double abs_trace = std::real(arma::accu(state_conj % OB));
						target(row_index, idx) = abs_trace;
					}
				};

				size_t pulse_step_index = 0;
				arma::mat ExptValuesPulseOrientation;
				if (has_pulse_output)
				{
					ExptValuesPulseOrientation.zeros(pulse_times.size(), projection_counter);
					if (pulse_has_initial_step)
					{
						record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
						pulse_step_index = 1;
					}
				}

				// Get pulses and pulse the system for this orientation
				arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * H - K;

				if (hasPulseSequence)
				{
					// Loop through all pulse sequences
					for (const auto &seq : Pulsesequence)
					{
						// Write which pulse in pulsesequence is calculating now
						if (grid_num == 0)
						{
							this->Log() << std::get<0>(seq) << ", " << std::get<1>(seq) << std::endl;
						}

						// Save the parameters from the input as variables
						std::string pulse_name = std::get<0>(seq);
						double timerelaxation = std::get<1>(seq);

						for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
						{
							if ((*pulse)->Name().compare(pulse_name) == 0)
							{

								// Apply a pulse to our density vector
								if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
								{
									// Create a Pulse operator in HS; only one side of exponentials as we only propagate wavevectors
									arma::sp_cx_mat pulse_operator;
									if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
									{
										this->Log() << "Failed to create a pulse operator in HS." << std::endl;
										continue;
									}

									// Take a step, "first" is propagator and "second" is current state
									B = pulse_operator * B;
								}
								else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulseStaticField)
								{

									// Create a Pulse operator in HS; only one side of exponentials as we only propagate wavevectors
									arma::sp_cx_mat pulse_operator;
									if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
									{
										this->Log() << "Failed to create a pulse operator in HS." << std::endl;
										continue;
									}

									// Create array containing a propagator and the current state of each system
									std::pair<arma::sp_cx_mat, arma::cx_mat> G;

									// Get the propagator and put it into the array together with the initial state
									arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator)) * (*pulse)->Timestep())));
									G = std::pair<arma::sp_cx_mat, arma::cx_mat>(A_sp, B);

									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = 1; n <= steps; n++)
									{
										// Take a step, "first" is propagator and "second" is current state
										B = G.first * G.second;

										// Get the new current state vector matrix
										G.second = B;

										if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
										{
											record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
											++pulse_step_index;
										}
									}
								}
								else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
								{
									// Create a Pulse operator in SS
									arma::sp_cx_mat pulse_operator;
									if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
									{
										this->Log() << "Failed to create a pulse operator in HS." << std::endl;
										continue;
									}

									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = 1; n <= steps; n++)
									{
										double t = n * (*pulse)->Timestep();
										double pulse_factor = std::cos((*pulse)->Frequency() * t);
										arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(
											arma::expmat(
												arma::conv_to<arma::cx_mat>::from(
													(A + (arma::cx_double(0.0, -1.0) * pulse_operator * pulse_factor)) * (*pulse)->Timestep())));

										B = A_sp * B;

										if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
										{
											record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
											++pulse_step_index;
										}
									}
								}
								else
								{
									this->Log() << "Not implemented yet, sorry." << std::endl;
								}

								// Get the system relax during the time

								// Create array containing a propagator and the current state of each system
								std::pair<arma::sp_cx_mat, arma::cx_mat> G;
								arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from(A * (*pulse)->Timestep())));
								// Get the propagator and put it into the array together with the initial state
								G = std::pair<arma::sp_cx_mat, arma::cx_mat>(A_sp, B);

								unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / (*pulse)->Timestep()));
								for (unsigned int n = 1; n <= steps; n++)
								{
									// Take a step, "first" is propagator and "second" is current state
									B = G.first * G.second;

									// Get the new current state density vector
									G.second = B;

									if (has_pulse_output && pulse_step_index < ExptValuesPulseOrientation.n_rows)
									{
										record_expectation(ExptValuesPulseOrientation, pulse_step_index, B);
										++pulse_step_index;
									}
								}
							}
						}
					}
				}

				if (has_pulse_output && pulse_step_index != ExptValuesPulseOrientation.n_rows)
				{
					if (grid_num == 0)
					{
						this->Log() << "Warning: Pulse output step count mismatch. Expected " << ExptValuesPulseOrientation.n_rows
									<< ", recorded " << pulse_step_index << "." << std::endl;
					}
				}

				arma::mat ExptValuesOrientation;
				if (method_timeevo)
					ExptValuesOrientation.zeros(num_steps, projection_counter);

				// Propagate the system in time using the specified method
				if (method_timeevo && propmethod == "autoexpm")
				{
					arma::mat M; // used for variable estimation
					arma::sp_cx_mat H_prop = H - arma::cx_double(0.0, 1.0) * K;

					for (int k = 0; k < num_steps; k++)
					{
						arma::cx_mat Bconj = arma::conj(B);
						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < projection_counter; idx++)
						{
							arma::cx_mat OB;
							if (use_sparse_ops)
							{
								OB = OperatorsSparse[idx] * B;
							}
							else
							{
								OB = OperatorsDense[idx] * B;
							}
								double abs_trace = std::real(arma::accu(Bconj % OB));
								double expected_value = abs_trace;
								ExptValuesOrientation(k, idx) = expected_value;
							}

						// Update B using the Higham propagator
						B = space_thread.HighamProp(H_prop, B, -arma::cx_double(0.0, 1.0) * dt, precision, M);
					}
				}
				else if (method_timeevo && propmethod == "krylov")
				{
					arma::sp_cx_mat H_prop = H - arma::cx_double(0.0, 1.0) * K;

					for (arma::uword itr = 0; itr < B.n_cols; itr++)
					{
						arma::cx_vec prop_state = B.col(itr);

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < projection_counter; idx++)
						{
							double result = std::real(arma::cdot(prop_state, Operators[idx] * prop_state));
							ExptValuesOrientation(0, idx) += result;
						}
						prop_state = space_thread.KrylovExpmGeneral(H_prop, prop_state, dt, krylovsize, dim);

						int k = 1;

						while (k < num_steps)
						{
							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < projection_counter; idx++)
							{
								double result = std::real(arma::cdot(prop_state, Operators[idx] * prop_state));
								ExptValuesOrientation(k, idx) += result;
							}
							// Update the state using the shared Krylov propagator.
							prop_state = space_thread.KrylovExpmGeneral(H_prop, prop_state, dt, krylovsize, dim);
							k++;
						}
					}
				}
				else if (method_timeevo)
				{
					if (grid_num == 0)
					{
						this->Log() << "Using robust matrix exponential propagator for time-independent Hamiltonian." << std::endl;
					}

					// Include the recombination operator K
					arma::sp_cx_mat H_total = arma::cx_double(0.0, -1.0) * H - K;

					// Precompute the matrix exponential for the entire time step
					arma::cx_mat exp_H = arma::expmat(arma::cx_mat(H_total) * dt);

					// Propagate B
					for (int k = 0; k < num_steps; ++k)
					{
						arma::cx_mat Bconj = arma::conj(B);
						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							arma::cx_mat OB;
							if (use_sparse_ops)
							{
								OB = OperatorsSparse[idx] * B;
							}
							else
							{
								OB = OperatorsDense[idx] * B;
							}
							double abs_trace = std::real(arma::accu(Bconj % OB));
							double expected_value = abs_trace;
							ExptValuesOrientation(k, idx) = expected_value;
						}

						B = exp_H * B;
					}
				}

				if (has_pulse_output)
					ExptValuesPulsePartial[tid] += weight * ExptValuesPulseOrientation;

				if (method_timeevo)
					ExptValuesPartial[tid] += weight * ExptValuesOrientation;
				if (method_timeinf)
				{
					// Compute integrated density matrix in Hilbert space via Sylvester/Lyapunov:
					// A_state X + X A_state^† = -rho0, with A_state = -i H - K
					arma::cx_mat rho0mat = B * B.st();
					arma::cx_mat A_dense = -arma::cx_double(0.0, 1.0) * arma::cx_mat(H) - arma::cx_mat(K);
					arma::cx_mat A_star = arma::conj(A_dense);

					// Solve (A_state^* ⊗ I + I ⊗ A_state) vec(X) = -vec(rho0)
					// This corresponds to A_state X + X A_state^† = -rho0.
					arma::cx_mat L = arma::kron(A_star, Iden_dense) + arma::kron(Iden_dense, A_dense);
					arma::cx_vec rhs = arma::vectorise(-rho0mat);
					arma::cx_vec sol = arma::solve(L, rhs);
					if (sol.is_empty())
					{
						this->Log() << "Failed to solve timeinf Lyapunov equation in Hilbert space." << std::endl;
						continue;
					}
					arma::cx_mat X = arma::reshape(sol, rho0mat.n_rows, rho0mat.n_cols);

					rho_integrated_partial[tid] += weight * X;
				}
			}

			if (method_timeevo)
			{
				for (auto &m : ExptValuesPartial)
				{
					ExptValues += m;
				}
			}
			arma::mat ExptValuesPulse;
			if (has_pulse_output)
			{
				ExptValuesPulse.zeros(pulse_times.size(), projection_counter);
				for (auto &m : ExptValuesPulsePartial)
				{
					ExptValuesPulse += m;
				}
			}
			if (method_timeinf)
			{
				for (auto &m : rho_integrated_partial)
				{
					rho_integrated += m;
				}
			}

			double time_offset = print_pulses ? pulse_total_time : 0.0;

			if (has_pulse_output && print_pulses)
			{
				if (integrate_pulses)
				{
					this->Log() << "Writing integrated polarisation during pulse sequence." << std::endl;
				}

				arma::mat integrated_pulse;
				if (integrate_pulses)
				{
					integrated_pulse.zeros(pulse_times.size(), projection_counter);
					for (size_t k = 1; k < pulse_times.size(); ++k)
					{
						double dt_pulse = pulse_dts[k];
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							integrated_pulse(k, idx) = integrated_pulse(k - 1, idx) + dt_pulse * (ExptValuesPulse(k - 1, idx) + ExptValuesPulse(k, idx)) / 2.0;
						}
					}
					if (!pulse_times.empty())
					{
						integrated_pulse.row(0) = ExptValuesPulse.row(0);
					}
				}

				for (size_t k = 0; k < pulse_times.size(); ++k)
				{
					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << std::setprecision(12) << pulse_times[k] << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < projection_counter; ++idx)
					{
						if (integrate_pulses)
						{
							this->Data() << " " << integrated_pulse(k, idx);
						}
						else
						{
							this->Data() << " " << ExptValuesPulse(k, idx);
						}
					}
					this->Data() << std::endl;
				}
			}

			if (method_timeinf)
			{
				if (print_freeevo)
				{
					this->Log() << "Writing time-integrated (time -> inf) polarisation." << std::endl;

					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << "inf"
								 << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < projection_counter; idx++)
					{
						double val = use_sparse_ops ? TraceSparseDense(OperatorsSparse[idx], rho_integrated)
													: std::real(arma::trace(OperatorsDense[idx] * rho_integrated));
						this->Data() << std::setprecision(12) << val << " ";
					}
					this->Data() << std::endl;
				}
			}
			else if (method_timeevo && print_freeevo)
			{
				if (integrate_freeevo)
				{
					this->Log() << "Writing integrated polarisation over time." << std::endl;

					arma::mat integrated;
					integrated.zeros(num_steps, projection_counter);

					for (int k = 1; k < num_steps; ++k)
					{
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							integrated(k, idx) = integrated(k - 1, idx) + dt * (ExptValues(k - 1, idx) + ExptValues(k, idx)) / 2.0;
						}
					}

					if (num_steps > 0)
					{
						integrated.row(0) = ExptValues.row(0);
					}

					for (int k = 0; k < num_steps; k++)
					{
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << std::setprecision(12) << time_offset + time(k) << " ";
						this->WriteStandardOutput(this->Data());

						for (int idx = 0; idx < projection_counter; idx++)
						{
							this->Data() << " " << integrated(k, idx);
						}
						this->Data() << std::endl;
					}
				}
				else
				{
					for (int k = 0; k < num_steps; k++)
					{
						// Write results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << std::setprecision(12) << time_offset + time(k) << " ";
						this->WriteStandardOutput(this->Data());

						for (int idx = 0; idx < projection_counter; idx++)
						{
							this->Data() << " " << ExptValues(k, idx);
						}
						this->Data() << std::endl;
					}
				}
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}
		this->Data() << std::endl;
		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticHSDirectSpectra::WriteHeader(std::ostream &_stream)
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
								}
							}
							else
							{
								// Write each state name
								auto states = (*i)->States();
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Ix ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iy ";
								_stream << (*i)->Name() << "." << (*l)->Name() << ".Iz ";
							}
						}
					}
				}
			}
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticHSDirectSpectra::Validate()
	{
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

	bool TaskStaticHSDirectSpectra::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
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

	bool TaskStaticHSDirectSpectra::CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const
	{
		std::vector<double> theta(_Npoints);
		std::vector<double> phi(_Npoints);
		std::vector<double> weight(_Npoints);

		_uniformGrid.resize(_Npoints);

		const double golden = arma::datum::pi * (1.0 + std::sqrt(5.0)); // not standard golden angle

		for (int i = 0; i < _Npoints; ++i)
		{
			double index = static_cast<double>(i) + 0.5;

			theta[i] = std::acos(1.0 - index / _Npoints);							// hemisphere
			phi[i] = golden * index;												// hemisphere
			weight[i] = std::sin(theta[i]) * 2 * arma::datum::pi / _Npoints; // 2 * pi for hemisphere
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

}
