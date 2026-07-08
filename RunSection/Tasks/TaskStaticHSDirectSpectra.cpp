/////////////////////////////////////////////////////////////////////////
// TaskStaticHSDirectSpectra implementation (RunSection module) by Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
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
#ifdef _OPENMP
#include <omp.h>
#endif

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

		double TraceDenseDense(const arma::cx_mat &A, const arma::cx_mat &B)
		{
			// Avoid constructing A * B just to read its trace. This contraction
			// is called for every output operator, time point, and orientation.
			arma::cx_double sum = arma::cx_double(0.0, 0.0);
			for (arma::uword row = 0; row < A.n_rows; ++row)
			{
				for (arma::uword col = 0; col < A.n_cols; ++col)
				{
					sum += A(row, col) * B(col, row);
				}
			}
			return std::real(sum);
		}

		void WriteTransitionYieldHeader(const SpinAPI::system_ptr &_system, std::ostream &_stream)
		{
			auto transitions = _system->Transitions();
			for (auto transition = transitions.cbegin(); transition != transitions.cend(); ++transition)
			{
				if ((*transition)->SourceState() == nullptr)
					continue;
				_stream << _system->Name() << "." << (*transition)->Name() << ".yield ";
			}
		}

		bool BuildInitialDensityMatrix(const SpinAPI::system_ptr &system,
									   SpinAPI::SpinSpace &space,
									   arma::cx_mat &rho0,
									   std::ostream &log_stream)
		{
			auto initial_states = system->InitialState();
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
			const arma::cx_mat hermitian_rho0 = 0.5 * (rho0 + rho0.t());
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

		bool AddPhenomenologicalTerm(const SpinAPI::operator_ptr &relaxationOperator,
									 std::vector<SpinAPI::HilbertRelaxationPhenomenologicalTerm> &terms)
		{
			if (relaxationOperator == nullptr ||
				relaxationOperator->Type() != SpinAPI::OperatorType::RelaxationPhenomenological ||
				relaxationOperator->Rate1() < 0.0 ||
				relaxationOperator->Rate2() < 0.0)
			{
				return false;
			}

			if (relaxationOperator->Rate1() == 0.0 && relaxationOperator->Rate2() == 0.0)
				return false;

			SpinAPI::HilbertRelaxationPhenomenologicalTerm term;
			term.populationRate = relaxationOperator->Rate1();
			term.coherenceRate = relaxationOperator->Rate2();
			terms.push_back(term);
			return true;
		}

		bool DiagonalizeRelaxationBasis(const arma::sp_cx_mat &basisHamiltonian,
										arma::cx_mat &basisEigenvectors,
										std::ostream &log_stream)
		{
			arma::vec eigenvalues;
			if (!arma::eig_sym(eigenvalues, basisEigenvectors, arma::cx_mat(basisHamiltonian)))
			{
				log_stream << "Failed to diagonalize the Hamiltonian used for phenomenological relaxation." << std::endl;
				return false;
			}

			return true;
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
			if (!BuildInitialDensityMatrix(*i, space, rho0, this->Log()))
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no valid initial state could be constructed." << std::endl;
				continue;
			}

			const int dim = static_cast<int>(rho0.n_rows);
			this->Log() << "Hilbert Space Size " << dim << " x " << dim << std::endl;

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
			bool transitionYields = false;
			this->Properties()->Get("transitionyields", transitionYields);

			// Getting the projection operators
			if (transitionYields)
			{
				for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
				{
					if ((*j)->SourceState() == nullptr)
						continue;
					if (!space.GetState((*j)->SourceState(), P))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						return false;
					}
					if (num_transitions != 0)
					{
						rates.insert_rows(num_transitions, 1);
					}

					Operators[projection_counter] = (*j)->Rate() * P;
					projection_counter += 1;
					rates(num_transitions) = (*j)->Rate();
					num_transitions++;
				}
			}
			else if (this->Properties()->GetList("spinlist", spinList, ','))
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
			// Compact density-map propagation keeps rho vectorized. Precompute
			// matching trace contractions so observables do not require a
			// matrix conversion at every output point.
			std::vector<arma::cx_vec> OperatorsVector(projection_counter);
			for (int idx = 0; idx < projection_counter; ++idx)
			{
				OperatorsVector[idx].zeros(dim * dim);
				for (auto entry = OperatorsSparse[idx].begin(); entry != OperatorsSparse[idx].end(); ++entry)
				{
					// OperatorToSuperspace stores rho(row,col) at row * dim + col.
					// trace(O rho) therefore uses O(row,col) at col * dim + row.
					OperatorsVector[idx](entry.col() * dim + entry.row()) = *entry;
				}
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
			// Phenomenological relaxation is basis-local and has an exact
			// finite-step map. Explicit spin-operator relaxation must keep
			// its operator cache so anisotropic axes can be rebuilt for
			// every powder orientation.
			std::vector<SpinAPI::HilbertRelaxationPhenomenologicalTerm> phenomenological_relaxation_terms;
			std::vector<SpinAPI::operator_ptr> explicit_relaxation_operators;
			bool use_density_matrix = false;
			for (auto j = (*i)->operators_cbegin(); j != (*i)->operators_cend(); j++)
			{
				if ((*j)->Type() == SpinAPI::OperatorType::RelaxationPhenomenological)
				{
					if (AddPhenomenologicalTerm((*j), phenomenological_relaxation_terms))
					{
						use_density_matrix = true;
						this->Log() << "Added eigenbasis relaxation operator \"" << (*j)->Name() << "\" to Hilbert-space propagation.\n";
					}
					continue;
				}

				SpinAPI::HilbertRelaxationCache validation_cache;
				if (space.RelaxationOperator((*j), validation_cache))
				{
					explicit_relaxation_operators.push_back(*j);
					use_density_matrix = true;
					this->Log() << "Added powder-aware relaxation operator \"" << (*j)->Name() << "\" to Hilbert-space propagation.\n";
				}
			}
			const bool use_phenomenological_relaxation = !phenomenological_relaxation_terms.empty();
			const bool use_only_phenomenological_relaxation = use_phenomenological_relaxation && explicit_relaxation_operators.empty();
			if (use_density_matrix)
			{
				this->Log() << "Relaxation operators detected. Using density-matrix propagation in Hilbert space." << std::endl;
			}
			if (use_phenomenological_relaxation)
			{
				this->Log() << "Phenomenological relaxation is evaluated in the orientation-specific H0 eigenbasis." << std::endl;
			}
			if (!explicit_relaxation_operators.empty())
			{
				this->Log() << "Explicit relaxation operators are rebuilt for every powder orientation so their axes follow the molecular rotation." << std::endl;
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
				if (use_density_matrix)
				{
					this->Log() << "Autoexpm was requested. The relaxation-aware propagation strategy is selected below." << std::endl;
				}
				else
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
			}
			else if (propmethod == "krylov")
			{
				if (use_density_matrix)
				{
					this->Log() << "Krylov propagation was requested. The relaxation-aware propagation strategy is selected below." << std::endl;
				}
				else
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
			}
			else if (propmethod == "rk4" || propmethod == "explicit")
			{
				this->Log() << "Explicit RK4 was requested as the propagation method." << std::endl;
			}
			else if (propmethod == "normal")
			{
				this->Log() << "Normal exponential Hamiltonian propagation is chosen." << std::endl;
			}
			else if (propmethod.empty())
			{
				this->Log() << "No propagation method specified. Using normal exponential Hamiltonian propagation." << std::endl;
				propmethod = "normal";
			}
			else
			{
				this->Log() << "WARNING: Undefined propagation method. Using normal exponential method." << std::endl;
				propmethod = "normal";
			}

			bool relax_use_split_expm = false;
			bool relax_use_exact_phenomenological_split = false;
			arma::cx_mat K_dense;
			if (use_density_matrix)
			{
				if (use_only_phenomenological_relaxation)
				{
					relax_use_split_expm = true;
					relax_use_exact_phenomenological_split = true;
					K_dense = arma::cx_mat(K);
					this->Log() << "Propagation strategy: optimized phenomenological eigenbasis density propagation. The H0 eigenbasis is prepared once per powder orientation, and adjacent Hamiltonian half-steps are merged during free evolution." << std::endl;
					if (propmethod == "rk4" || propmethod == "explicit")
					{
						this->Log() << "Note: explicit RK4 was requested but is not used for phenomenological-only relaxation because the analytical finite-step map is exact." << std::endl;
					}
					else if (propmethod != "normal")
					{
						this->Log() << "Note: propagationmethod = " << propmethod << " is replaced by dense Hamiltonian exponential half-steps for phenomenological-only relaxation." << std::endl;
					}
				}
				else
				{
					relax_use_split_expm = (propmethod != "rk4" && propmethod != "explicit");
					if (relax_use_split_expm)
					{
						K_dense = arma::cx_mat(K);
						this->Log() << "Propagation strategy: explicit relaxation density propagation with Hamiltonian exponential half-steps and an RK4 relaxation fallback." << std::endl;
						this->Log() << "Performance note: explicit spin-operator relaxation is substantially slower than Hilbert-factor or phenomenological-only propagation." << std::endl;
						if (use_phenomenological_relaxation)
						{
							this->Log() << "Phenomenological terms are included in the RK4 fallback because explicit relaxation operators are also active." << std::endl;
						}
						if (propmethod != "normal")
						{
							this->Log() << "Note: propagationmethod is ignored for relaxation splitting; use propagationmethod = rk4 to force full explicit RK4." << std::endl;
						}
					}
					else
					{
						this->Log() << "Propagation strategy: explicit relaxation density propagation with full RK4 as requested." << std::endl;
						this->Log() << "Performance note: full RK4 density propagation is the slowest Hilbert spectra branch." << std::endl;
					}
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
			this->Log() << "Spectrum evaluation method = " << Method << "." << std::endl;
			if (method_timeinf && use_density_matrix)
			{
				this->Log() << "Steady-state relaxation is assembled as an orientation-specific Liouvillian before solving the time-integrated density matrix." << std::endl;
			}
			if (method_timeevo && relax_use_exact_phenomenological_split)
			{
				const int density_map_dimension = dim * dim;
				if (density_map_dimension <= 64)
				{
					this->Log() << "Small Hilbert density matrix detected (" << density_map_dimension << " components). A compact finite-step map will be prepared per orientation." << std::endl;
				}
				else
				{
					this->Log() << "Hilbert density matrix has " << density_map_dimension << " components. Using direct matrix propagation to avoid compact-map memory growth." << std::endl;
				}
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

			std::vector<std::tuple<double, double, double>> grid;
			int numPoints = 0;
			bool explicitPowderGrid = this->CreateExplicitPowderGrid(grid);
			if (explicitPowderGrid)
			{
				numPoints = static_cast<int>(grid.size());
				this->Log() << "Using explicit single-orientation powder input. This is the external-distribution route; the supplied orientation and weight are preserved." << std::endl;
			}
			else
			{
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
					else
					{
						this->Log() << "Using one internally generated identity orientation with unit weight (non-powdered limit)." << std::endl;
					}
				}
			}
			if (grid.empty())
			{
				grid.push_back({0.0, 0.0, 1.0});
				numPoints = 0;
				this->Log() << "Using one identity orientation with unit weight (non-powdered limit)." << std::endl;
			}

			std::vector<std::string> HamiltonianH0list;
			std::vector<std::string> HamiltonianH1list;
			bool hasH0list = this->Properties()->GetList("hamiltonianh0list", HamiltonianH0list, ',');
			bool hasH1list = this->Properties()->GetList("hamiltonianh1list", HamiltonianH1list, ',');

			const SpinAPI::StateFrame initialStateFrame = (*i)->InitialStateFrame();
			SpinAPI::HilbertStateRotationCache initialStateRotationCache;
			const SpinAPI::HilbertStateRotationCache *initialStateRotationCachePtr = nullptr;
			if (initialStateFrame == SpinAPI::StateFrame::Molecular)
			{
				if (!space.CreateStateRotationCache(rho0, initialStateRotationCache))
				{
					this->Log() << "Failed to prepare molecular-frame initial-state rotations." << std::endl;
					return false;
				}
				initialStateRotationCachePtr = &initialStateRotationCache;
				if (initialStateRotationCache.rotationInvariant)
				{
					this->Log() << "Initial state frame = molecular, but the density matrix is rotationally invariant. Powder-state rotations are skipped." << std::endl;
				}
				else
				{
					this->Log() << "Initial state frame = molecular. Rotating the density matrix once per powder orientation." << std::endl;
				}
			}
			else if (initialStateFrame == SpinAPI::StateFrame::Fixed)
			{
				this->Log() << "Initial state frame = fixed. Reusing the supplied lab-frame density matrix for every orientation." << std::endl;
			}
			else
			{
				this->Log() << "Initial state frame = eigen. The prepared density matrix is already defined in its Hamiltonian frame." << std::endl;
			}

			const SpinAPI::InitialStateCoherenceMode initialCoherences = (*i)->InitialStateCoherences();
			const bool dephaseInitialState = (initialCoherences == SpinAPI::InitialStateCoherenceMode::DephaseEigenbasis);
			std::vector<std::string> initialStateHamiltonianList;
			if (dephaseInitialState)
			{
				// The dephasing basis has to be generated with the same
				// orientation as the propagation Hamiltonian. By default we use
				// H0, but an input file may name a separate Hamiltonian list.
				if (!this->Properties()->GetList("initialstatehamiltonian", initialStateHamiltonianList, ',') &&
					!this->Properties()->GetList("hamiltonianh0list", initialStateHamiltonianList, ','))
				{
					this->Log() << "Initial-state eigenbasis dephasing requires initialstatehamiltonian or hamiltonianh0list." << std::endl;
					return false;
				}
				this->Log() << "Initial-state coherences = eigenbasis populations. Off-diagonal elements are discarded per orientation." << std::endl;
			}
			else
			{
				this->Log() << "Initial-state coherences = keep. Coherences are retained after any molecular-frame rotation." << std::endl;
			}

			// Without relaxation, propagation uses a low-rank Hilbert factor B
			// with rho = B B^dagger. Reuse that factor whenever preparation does
			// not change the density matrix between powder orientations.
			const bool initialDensityOrientationInvariant =
				initialStateFrame != SpinAPI::StateFrame::Molecular ||
				initialStateRotationCache.rotationInvariant;
			const bool reuseInitialFactor = !use_density_matrix &&
											initialDensityOrientationInvariant &&
											!dephaseInitialState;
			arma::cx_mat orientationInvariantInitialFactor;
			if (reuseInitialFactor)
			{
				orientationInvariantInitialFactor = FactorizeDensityMatrix(rho0, this->Log());
				if (orientationInvariantInitialFactor.is_empty())
				{
					this->Log() << "Skipping SpinSystem \"" << (*i)->Name()
								<< "\" because its initial density matrix could not be factorized." << std::endl;
					continue;
				}
				this->Log() << "Propagation strategy: reusable Hilbert factor with rank "
							<< orientationInvariantInitialFactor.n_cols << " for every powder orientation." << std::endl;
			}

			// Read a pulse sequence from the input
			std::vector<std::tuple<std::string, double>> pulseSequence;
			bool hasPulseSequence = this->Properties()->GetPulseSequence("pulsesequence", pulseSequence);
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

				for (const auto &seq : pulseSequence)
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

			SpinAPI::SpinSpace base_space(space);
			base_space.SetReactionOperatorType(this->reactionOperators);
			base_space.UseSuperoperatorSpace(false);

			std::vector<SpinAPI::SpinSpace> spaces;
			spaces.resize(nthreads);
			for (int t = 0; t < nthreads; ++t)
			{
				spaces[t] = base_space;
			}

#pragma omp parallel for schedule(static) if (grid_size > 1)
			for (size_t grid_num = 0; grid_num < grid_size; ++grid_num)
			{
				int tid = 0;
#ifdef _OPENMP
				tid = omp_get_thread_num();
#endif
				SpinAPI::SpinSpace &space_thread = spaces[tid];

				const auto &grid_point = grid[grid_num];
				double theta, phi, weight;
				std::tie(theta, phi, weight) = grid_point;
				if (grid_size <= 1 && !explicitPowderGrid)
				{
					// A single point means "no powder average" for this task.
					// Keep the identity orientation and unit weight so that HS
					// and SS spectra have the same normalization.
					theta = 0.0;
					phi = 0.0;
					weight = 1.0;
				}

				arma::mat Rot_mat = arma::eye<arma::mat>(3, 3);
				double gamma = 0.0;
				if (!this->CreateRotationMatrix(gamma, theta, phi, Rot_mat))
				{
					this->Log() << "Failed to obtain rotation matrix for powder orientation." << std::endl;
				}

				arma::sp_cx_mat H;
				arma::sp_cx_mat relaxation_basis_hamiltonian;
				if (hasH0list)
				{
					arma::sp_cx_mat H0;
					if (!space_thread.BaseHamiltonianRotated_SA(HamiltonianH0list, Rot_mat, H0))
					{
						this->Log() << "Failed to obtain rotated Hamiltonian for SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}
					H = H0;
					relaxation_basis_hamiltonian = H0;

					if (hasH1list)
					{
						arma::sp_cx_mat H1;
						// CHANGED 2026-07-07: H1 must be rotated with the same powder crystallite as H0;
						// otherwise the microwave operator is evaluated in a different molecular frame.
						if (!space_thread.BaseHamiltonianRotatedZYZ(HamiltonianH1list, Rot_mat, H1))
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
					relaxation_basis_hamiltonian = H;
				}

				arma::cx_mat rho_initial;
				if (!reuseInitialFactor &&
					!space_thread.PrepareInitialDensityForPowder(rho0, Rot_mat, initialStateFrame, dephaseInitialState, initialStateHamiltonianList, initialStateRotationCachePtr, rho_initial))
				{
					this->Log() << "Failed to prepare initial density matrix for powder orientation." << std::endl;
					continue;
				}

				arma::cx_mat phenomenological_basis_eigenvectors;
				if (use_phenomenological_relaxation &&
					!DiagonalizeRelaxationBasis(relaxation_basis_hamiltonian, phenomenological_basis_eigenvectors, this->Log()))
				{
					continue;
				}

				// Explicit spin-operator relaxation depends on the spatial
				// orientation. Rebuild this cache beside the rotated H0
				// instead of reusing the validation cache created above.
				SpinAPI::HilbertRelaxationCache relaxation_cache;
				bool relaxation_cache_valid = true;
				for (const auto &relaxationOperator : explicit_relaxation_operators)
				{
					if (!space_thread.PowderRelaxationOperatorHilbert(relaxationOperator, Rot_mat, relaxation_cache))
					{
						this->Log() << "Failed to construct powder-aware Hilbert-space relaxation operator \"" << relaxationOperator->Name() << "\"." << std::endl;
						relaxation_cache_valid = false;
						break;
					}
				}
				if (!relaxation_cache_valid)
				{
					continue;
				}

				arma::cx_mat relaxation_super;
				if (method_timeinf && !explicit_relaxation_operators.empty() &&
					!space_thread.RelaxationSuperoperatorHilbert(relaxation_cache, relaxation_super))
				{
					this->Log() << "Failed to construct powder-aware Hilbert-space relaxation superoperator." << std::endl;
					continue;
				}

				if (use_density_matrix)
				{
					arma::cx_mat rho = rho_initial;
					int dim = static_cast<int>(rho.n_rows);

					arma::cx_mat work_left(dim, dim, arma::fill::zeros);
					arma::cx_mat work_right(dim, dim, arma::fill::zeros);
					arma::cx_mat relax(dim, dim, arma::fill::zeros);
					arma::cx_mat phenomenological_relax(dim, dim, arma::fill::zeros);
					arma::cx_mat k1(dim, dim, arma::fill::zeros);
					arma::cx_mat k2(dim, dim, arma::fill::zeros);
					arma::cx_mat k3(dim, dim, arma::fill::zeros);
					arma::cx_mat k4(dim, dim, arma::fill::zeros);
					arma::cx_mat tmp_state(dim, dim, arma::fill::zeros);
					arma::cx_mat rk_accum(dim, dim, arma::fill::zeros);
					SpinAPI::HilbertPhenomenologicalRelaxationMap timeevo_relaxation_map;
					const SpinAPI::HilbertPhenomenologicalRelaxationMap *timeevo_relaxation_map_ptr = nullptr;
					if (relax_use_exact_phenomenological_split && method_timeevo)
					{
						if (!space_thread.CreatePhenomenologicalRelaxationMapHilbert(phenomenological_relaxation_terms, phenomenological_basis_eigenvectors, dt, timeevo_relaxation_map))
						{
							this->Log() << "Failed to prepare the analytical phenomenological relaxation map for this powder orientation." << std::endl;
							continue;
						}
						timeevo_relaxation_map_ptr = &timeevo_relaxation_map;
					}

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

					const bool propagateInPhenomenologicalBasis =
						relax_use_exact_phenomenological_split && method_timeevo;
					arma::cx_mat K_propagation_basis = K_dense;
					std::vector<arma::cx_mat> operatorsPhenomenologicalBasis;
					std::vector<arma::cx_vec> operatorsPhenomenologicalBasisVector;
					if (propagateInPhenomenologicalBasis)
					{
						// Phenomenological relaxation is diagonal in this H0
						// eigenbasis. Keep every recurring operation in that
						// basis for the full orientation instead of performing
						// two dense basis round trips during every time step.
						rho = phenomenological_basis_eigenvectors.t() * rho * phenomenological_basis_eigenvectors;
						H_dense = phenomenological_basis_eigenvectors.t() * H_dense * phenomenological_basis_eigenvectors;
						K_propagation_basis = phenomenological_basis_eigenvectors.t() * K_dense * phenomenological_basis_eigenvectors;

						operatorsPhenomenologicalBasis.resize(projection_counter);
						operatorsPhenomenologicalBasisVector.resize(projection_counter);
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							const arma::cx_mat operatorLab = use_sparse_ops
																 ? arma::cx_mat(OperatorsSparse[idx])
																 : OperatorsDense[idx];
							operatorsPhenomenologicalBasis[idx] =
								phenomenological_basis_eigenvectors.t() * operatorLab * phenomenological_basis_eigenvectors;
							operatorsPhenomenologicalBasisVector[idx].zeros(dim * dim);
							for (int row = 0; row < dim; ++row)
							{
								for (int col = 0; col < dim; ++col)
								{
									operatorsPhenomenologicalBasisVector[idx](col * dim + row) =
										operatorsPhenomenologicalBasis[idx](row, col);
								}
							}
						}
					}

					const arma::cx_double imag_unit(0.0, 1.0);

					auto record_expectation_rho = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state)
					{
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							double val = propagateInPhenomenologicalBasis
											 ? TraceDenseDense(operatorsPhenomenologicalBasis[idx], state)
											 : (use_sparse_ops ? TraceSparseDense(OperatorsSparse[idx], state)
															   : TraceDenseDense(OperatorsDense[idx], state));
							target(row_index, idx) = val;
						}
					};

					auto record_expectation_vector = [&](arma::mat &target, size_t row_index, const arma::cx_vec &state)
					{
						for (int idx = 0; idx < projection_counter; ++idx)
						{
							const arma::cx_vec &operatorVector = propagateInPhenomenologicalBasis
																	 ? operatorsPhenomenologicalBasisVector[idx]
																	 : OperatorsVector[idx];
							target(row_index, idx) = std::real(arma::accu(operatorVector % state));
						}
					};

					auto drho = [&](const arma::cx_mat &state, const arma::sp_cx_mat &H_total, const arma::cx_mat *H_dense_ptr, arma::cx_mat &out)
					{
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
						if (use_phenomenological_relaxation)
						{
							space_thread.ApplyPhenomenologicalRelaxationHilbert(phenomenological_relaxation_terms, phenomenological_basis_eigenvectors, state, phenomenological_relax);
							relax += phenomenological_relax;
						}
						out += relax;
					};

					auto rk4_step = [&](arma::cx_mat &state, const arma::sp_cx_mat &H_total, const arma::cx_mat *H_dense_ptr, double step_dt)
					{
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

					auto relax_rhs = [&](const arma::cx_mat &state, arma::cx_mat &out)
					{
						space_thread.ApplyRelaxationHilbert(relaxation_cache, state, out);
						if (use_phenomenological_relaxation)
						{
							space_thread.ApplyPhenomenologicalRelaxationHilbert(phenomenological_relaxation_terms, phenomenological_basis_eigenvectors, state, phenomenological_relax);
							out += phenomenological_relax;
						}
					};

					auto rk4_relax_step = [&](arma::cx_mat &state, double step_dt)
					{
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

					auto build_unitary_half = [&](const arma::cx_mat &H_total_dense, double step_dt, arma::cx_mat &U_half, arma::cx_mat &U_half_st)
					{
						arma::cx_mat A_dense = -imag_unit * H_total_dense - K_propagation_basis;
						U_half = arma::expmat(A_dense * (0.5 * step_dt));
						U_half_st = U_half.t();
					};

					auto apply_unitary_half = [&](arma::cx_mat &state, const arma::cx_mat &U_half, const arma::cx_mat &U_half_st)
					{
						work_left = U_half * state;
						state = work_left * U_half_st;
					};

					auto split_step = [&](arma::cx_mat &state, const arma::cx_mat &U_half, const arma::cx_mat &U_half_st, const SpinAPI::HilbertPhenomenologicalRelaxationMap *phenomenological_map, double step_dt)
					{
						apply_unitary_half(state, U_half, U_half_st);
						if (phenomenological_map != nullptr)
						{
							if (propagateInPhenomenologicalBasis)
								space_thread.ApplyPhenomenologicalRelaxationMapInBasisHilbert(*phenomenological_map, state);
							else
								space_thread.ApplyPhenomenologicalRelaxationMapHilbert(*phenomenological_map, state, work_right);
						}
						else
						{
							rk4_relax_step(state, step_dt);
						}
						apply_unitary_half(state, U_half, U_half_st);
					};

					auto build_compact_split_propagator = [&](const arma::cx_mat &U_half, const arma::cx_mat &U_half_st, const SpinAPI::HilbertPhenomenologicalRelaxationMap *phenomenological_map, double step_dt, arma::sp_cx_mat &propagator)
					{
						// For small Hilbert spaces, materializing the composed
						// density map is faster than repeatedly dispatching
						// several tiny matrix products in split_step().
						const arma::uword density_dim = static_cast<arma::uword>(dim * dim);
						arma::cx_mat dense_propagator(density_dim, density_dim, arma::fill::zeros);
						arma::cx_mat basis_state(dim, dim, arma::fill::zeros);
						arma::cx_vec propagated_state;
						for (arma::uword col = 0; col < density_dim; ++col)
						{
							basis_state.zeros();
							basis_state(col / dim, col % dim) = 1.0;
							split_step(basis_state, U_half, U_half_st, phenomenological_map, step_dt);
							space_thread.OperatorToSuperspace(basis_state, propagated_state);
							dense_propagator.col(col) = propagated_state;
						}
						propagator = arma::sp_cx_mat(dense_propagator);
					};

					auto operator_for_propagation_basis = [&](const arma::sp_cx_mat &operatorLab)
					{
						arma::cx_mat result(operatorLab);
						if (propagateInPhenomenologicalBasis)
						{
							result = phenomenological_basis_eigenvectors.t() * result * phenomenological_basis_eigenvectors;
						}
						return result;
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
						for (const auto &seq : pulseSequence)
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
									SpinAPI::HilbertPhenomenologicalRelaxationMap pulse_relaxation_map;
									const SpinAPI::HilbertPhenomenologicalRelaxationMap *pulse_relaxation_map_ptr = nullptr;
									if (relax_use_exact_phenomenological_split)
									{
										if (!space_thread.CreatePhenomenologicalRelaxationMapHilbert(phenomenological_relaxation_terms, phenomenological_basis_eigenvectors, pulse_dt, pulse_relaxation_map))
										{
											this->Log() << "Failed to prepare the analytical phenomenological relaxation map for pulse \"" << (*pulse)->Name() << "\"." << std::endl;
											continue;
										}
										pulse_relaxation_map_ptr = &pulse_relaxation_map;
									}

									if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
									{
										arma::sp_cx_mat pulse_operator;
										if (!space_thread.PulseOperatorOnStatevector((*pulse), pulse_operator))
										{
											this->Log() << "Failed to create a pulse operator in HS." << std::endl;
											continue;
										}
										arma::cx_mat U = operator_for_propagation_basis(pulse_operator);
										rho = U * rho * U.t();
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
											arma::cx_mat H_pulse_dense = H_dense + operator_for_propagation_basis(pulse_operator);
											arma::cx_mat U_half;
											arma::cx_mat U_half_st;
											build_unitary_half(H_pulse_dense, pulse_dt, U_half, U_half_st);
											for (unsigned int n = 1; n <= steps; ++n)
											{
												split_step(rho, U_half, U_half_st, pulse_relaxation_map_ptr, pulse_dt);

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
											const arma::cx_mat pulse_operator_propagation = operator_for_propagation_basis(pulse_operator);
											for (unsigned int n = 1; n <= steps; ++n)
											{
												double t = n * pulse_dt;
												double pulse_factor = std::cos((*pulse)->Frequency() * t);
												arma::cx_mat H_pulse_dense = H_dense + pulse_operator_propagation * pulse_factor;
												arma::cx_mat U_half;
												arma::cx_mat U_half_st;
												build_unitary_half(H_pulse_dense, pulse_dt, U_half, U_half_st);
												split_step(rho, U_half, U_half_st, pulse_relaxation_map_ptr, pulse_dt);

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
												split_step(rho, U_half, U_half_st, pulse_relaxation_map_ptr, pulse_dt);

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
							const bool use_compact_split_propagator = relax_use_exact_phenomenological_split && dim * dim <= 64;
							if (use_compact_split_propagator)
							{
								arma::sp_cx_mat compact_split_propagator;
								build_compact_split_propagator(U_half, U_half_st, timeevo_relaxation_map_ptr, dt, compact_split_propagator);
								arma::cx_vec rho_vector;
								space_thread.OperatorToSuperspace(rho, rho_vector);
								for (int k = 0; k < num_steps; ++k)
								{
									record_expectation_vector(ExptValuesOrientation, k, rho_vector);
									rho_vector = compact_split_propagator * rho_vector;
								}
							}
							else if (propagateInPhenomenologicalBasis)
							{
								// Consecutive Strang steps share their
								// neighboring Hamiltonian half-steps:
								// U(1/2) R U(1/2) U(1/2) R U(1/2).
								// Keep a half-step-shifted density matrix and
								// transform each observable once. The exact
								// same split evolution then needs two dense
								// products per time point instead of four.
								const arma::cx_mat U_full = U_half * U_half;
								const arma::cx_mat U_full_st = U_full.t();
								const arma::cx_mat U_half_inverse = arma::inv(U_half);
								std::vector<arma::cx_mat> operatorsCentered(projection_counter);
								for (int idx = 0; idx < projection_counter; ++idx)
								{
									operatorsCentered[idx] =
										U_half_inverse.t() * operatorsPhenomenologicalBasis[idx] * U_half_inverse;
								}

								apply_unitary_half(rho, U_half, U_half_st);
								for (int k = 0; k < num_steps; ++k)
								{
									for (int idx = 0; idx < projection_counter; ++idx)
									{
										ExptValuesOrientation(k, idx) = TraceDenseDense(operatorsCentered[idx], rho);
									}
									space_thread.ApplyPhenomenologicalRelaxationMapInBasisHilbert(*timeevo_relaxation_map_ptr, rho);
									work_left = U_full * rho;
									rho = work_left * U_full_st;
								}
							}
							else
							{
								for (int k = 0; k < num_steps; ++k)
								{
									record_expectation_rho(ExptValuesOrientation, k, rho);
									split_step(rho, U_half, U_half_st, timeevo_relaxation_map_ptr, dt);
								}
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

						arma::cx_mat L = arma::kron(A_dense, Iden_dense) + arma::kron(Iden_dense, arma::conj(A_dense));
						arma::cx_mat relaxation_super_orientation = relaxation_super;
						if (use_phenomenological_relaxation)
						{
							arma::cx_mat phenomenological_super;
							if (!space_thread.PhenomenologicalRelaxationSuperoperatorHilbert(phenomenological_relaxation_terms, phenomenological_basis_eigenvectors, phenomenological_super))
							{
								this->Log() << "Failed to construct phenomenological relaxation superoperator for this powder orientation." << std::endl;
								continue;
							}

							if (relaxation_super_orientation.is_empty())
								relaxation_super_orientation = phenomenological_super;
							else
								relaxation_super_orientation += phenomenological_super;
						}
						if (!relaxation_super_orientation.is_empty())
						{
							L += relaxation_super_orientation;
						}
						arma::cx_vec rhs;
						if (!space_thread.OperatorToSuperspace(-rho0mat, rhs))
						{
							this->Log() << "Failed to convert Hilbert timeinf right-hand side to superspace convention." << std::endl;
							continue;
						}
						arma::cx_vec sol = arma::solve(L, rhs);
						if (sol.is_empty())
						{
							this->Log() << "Failed to solve timeinf Lyapunov equation in Hilbert space." << std::endl;
							continue;
						}
						arma::cx_mat X;
						if (!space_thread.OperatorFromSuperspace(sol, X))
						{
							this->Log() << "Failed to convert Hilbert timeinf solution from superspace convention." << std::endl;
							continue;
						}

						rho_integrated_partial[tid] += weight * X;
					}

					if (has_pulse_output)
						ExptValuesPulsePartial[tid] += weight * ExptValuesPulseOrientation;

					if (method_timeevo)
						ExptValuesPartial[tid] += weight * ExptValuesOrientation;

					continue;
				}

				arma::cx_mat B = reuseInitialFactor
									 ? orientationInvariantInitialFactor
									 : FactorizeDensityMatrix(rho_initial, this->Log());
				if (B.is_empty())
				{
					this->Log() << "Skipping powder orientation because the prepared initial density matrix could not be factorized." << std::endl;
					continue;
				}

				auto record_expectation = [&](arma::mat &target, size_t row_index, const arma::cx_mat &state)
				{
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
					for (const auto &seq : pulseSequence)
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
					arma::cx_mat rho0mat = B * B.t();
					arma::cx_mat A_dense = -arma::cx_double(0.0, 1.0) * arma::cx_mat(H) - arma::cx_mat(K);

					// Solve using MolSpin's row-major superspace convention, vec(X^T).
					// This corresponds to A_state X + X A_state^† = -rho0.
					arma::cx_mat L = arma::kron(A_dense, Iden_dense) + arma::kron(Iden_dense, arma::conj(A_dense));
					arma::cx_vec rhs;
					if (!space_thread.OperatorToSuperspace(-rho0mat, rhs))
					{
						this->Log() << "Failed to convert Hilbert timeinf right-hand side to superspace convention." << std::endl;
						continue;
					}
					arma::cx_vec sol = arma::solve(L, rhs);
					if (sol.is_empty())
					{
						this->Log() << "Failed to solve timeinf Lyapunov equation in Hilbert space." << std::endl;
						continue;
					}
					arma::cx_mat X;
					if (!space_thread.OperatorFromSuperspace(sol, X))
					{
						this->Log() << "Failed to convert Hilbert timeinf solution from superspace convention." << std::endl;
						continue;
					}

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
					this->Data() << "inf" << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < projection_counter; idx++)
					{
						double val = use_sparse_ops ? TraceSparseDense(OperatorsSparse[idx], rho_integrated)
													: TraceDenseDense(OperatorsDense[idx], rho_integrated);
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
		// this->Data() << std::endl;
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
			bool transitionYields = false;
			if (this->Properties()->Get("transitionyields", transitionYields) && transitionYields)
			{
				WriteTransitionYieldHeader((*i), _stream);
				continue;
			}

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

			theta[i] = std::acos(1.0 - index / _Npoints); // hemisphere
			phi[i] = golden * index;					  // hemisphere
			weight[i] = 2 * arma::datum::pi / _Npoints;	  // constant in cos(theta) over the hemisphere
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

	bool TaskStaticHSDirectSpectra::CreateExplicitPowderGrid(std::vector<std::tuple<double, double, double>> &_grid)
	{
		// External drivers such as VIKING can request exactly one MolSpin
		// powder orientation per input file. The tuple is theta, phi, weight
		// in radians and MolSpin's hemisphere integration convention. This
		// keeps Hamiltonian rotation and molecular-frame initial-density
		// rotation inside MolSpin instead of duplicating that logic outside.
		arma::vec orientation;
		if (this->Properties()->Get("powderorientation", orientation) ||
			this->Properties()->Get("powder_orientation", orientation))
		{
			if (orientation.n_elem < 2)
			{
				this->Log() << "Explicit powder orientation requires at least theta and phi." << std::endl;
				return false;
			}
			double weight = (orientation.n_elem > 2) ? orientation(2) : 1.0;
			_grid.clear();
			_grid.push_back({orientation(0), orientation(1), weight});
			this->Log() << "Using explicit powder orientation theta=" << orientation(0)
						<< ", phi=" << orientation(1)
						<< ", weight=" << weight << "." << std::endl;
			return true;
		}

		double theta = 0.0;
		double phi = 0.0;
		double weight = 1.0;
		bool hasTheta = this->Properties()->Get("powdertheta", theta) ||
						this->Properties()->Get("powder_theta", theta);
		bool hasPhi = this->Properties()->Get("powderphi", phi) ||
					  this->Properties()->Get("powder_phi", phi);
		bool hasWeight = this->Properties()->Get("powderweight", weight) ||
						 this->Properties()->Get("powder_weight", weight);
		if (!(hasTheta || hasPhi || hasWeight))
		{
			return false;
		}
		if (!(hasTheta && hasPhi))
		{
			this->Log() << "Explicit powder orientation requires powdertheta and powderphi." << std::endl;
			return false;
		}

		_grid.clear();
		_grid.push_back({theta, phi, weight});
		this->Log() << "Using explicit powder orientation theta=" << theta
					<< ", phi=" << phi
					<< ", weight=" << weight << "." << std::endl;
		return true;
	}

}
