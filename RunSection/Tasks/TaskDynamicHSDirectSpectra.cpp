/////////////////////////////////////////////////////////////////////////
// TaskDynamicHSDirectSpectra implementation (RunSection module) by Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskDynamicHSDirectSpectra.h"
#include "Transition.h"
#include "Operator.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include <iomanip> // std::setprecision
#include <chrono>

namespace RunSection
{
	// -----------------------------------------------------
	// TaskDynamicHSDirectSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskDynamicHSDirectSpectra::TaskDynamicHSDirectSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(0.1), totaltime(1000), timedependentInteractions(false), timedependentTransitions(false), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn), productYieldsOnly(true)
	{
	}

	TaskDynamicHSDirectSpectra::~TaskDynamicHSDirectSpectra()
	{
	}
	// -----------------------------------------------------
	// TaskDynamicHSDirectSpectra protected methods
	// -----------------------------------------------------
	bool TaskDynamicHSDirectSpectra::RunLocal()
	{
		this->Log() << "Running method DynamicHS-Direct-Spectra." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++) // iteration through all spin systems, in this case (or usually), this is one
		{
			// Gyromagnetic constant
			double gamma_e = 176.0859644; // gyromagnetic ratio of free electron spin in rad mT^-1 mus^-1

			// Count the number of nuclear spins
			int nucspins = 0;
			for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
			{
				std::string spintype;
				(*l)->Properties()->Get("type", spintype);
				if (spintype != "electron")
				{
					nucspins += 1;
				}
				if (spintype == "electron")
				{
					// Throws an error if the spins are not spin 1/2
					if ((*l)->Multiplicity() != 2)
					{
						std::cout << (*l)->Multiplicity() << std::endl;
						this->Log() << "SkippingSpin System \"" << (*i)->Name() << "\" as electron spins have the wrong multiplicity." << std::endl;
						std::cout << "# ERROR: electron spins have to be spin 1/2! Skipping the SpinSystem." << std::endl;
						return 1;
					}
				}
			}

			// Check if there are any nuclear spins
			if (nucspins == 0)
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no nuclear spins were specified." << std::endl;
				std::cout << "# ERROR: no nuclear spins were specified, skipping the system" << std::endl;
				return 1;
			}

			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);
			space.SetReactionOperatorType(this->reactionOperators);

			// Size of the nuclear spin subspace
			int Z = space.SpaceDimensions() / 4;
			std::cout << "# Hilbert Space Size " << 4 * Z << " x " << 4 * Z << std::endl;
			this->Log() << "Hilbert Space Size " << 4 * Z << " x " << 4 * Z << std::endl;

			// Check whether Hamiltonian terms and/or transitions are time-dependent
			bool time_dependent_hamiltonian;
			this->timedependentInteractions = space.HasTimedependentInteractions();
			if (!space.HasTimedependentInteractions())
			{
				time_dependent_hamiltonian = false;
				this->Log() << "Hamiltonian is time-independent." << std::endl;
			}
			else
			{
				time_dependent_hamiltonian = true;
				this->Log() << "Hamiltonian is time-dependent." << std::endl;
			}

			bool time_dependent_transitions;
			this->timedependentTransitions = space.HasTimedependentTransitions();
			if (!space.HasTimedependentTransitions())
			{
				time_dependent_transitions = false;
				this->Log() << "Transitions are time-independent." << std::endl;
			}
			else
			{
				time_dependent_transitions = true;
				this->Log() << "Transitions are time-dependent." << std::endl;
			}

			if (!time_dependent_transitions && !time_dependent_hamiltonian)
			{
				std::cout << "# ERROR: no time-dependent interactions. Use another method!" << std::endl;
				this->Log() << "No time-dependent interactions. Use another method!" << std::endl;
				return 1;
			}

			// Set up states for time-propagation
			arma::cx_mat InitialStateVector(4, 1);
			std::string InitialState;
			this->Properties()->Get("initialstate", InitialState);
			std::string InitialStateLower;

			// Convert the string to lowercase for case-insensitive comparison
			InitialStateLower.resize(InitialState.size());
			std::transform(InitialState.begin(), InitialState.end(), InitialStateLower.begin(), ::tolower);

			if (InitialStateLower == "singlet")
			{
				arma::cx_mat SingletState(4, 1);
				SingletState(0) = 0.0;
				SingletState(1) = 1.0 / sqrt(2);
				SingletState(2) = -1.0 / sqrt(2);
				SingletState(3) = 0.0;
				InitialStateVector = SingletState;
				this->Log() << "Singlet initial state." << std::endl;
			}
			else if (InitialStateLower == "tripletminus")
			{
				arma::cx_mat TripletMinusState(4, 1);
				TripletMinusState(0) = 0.0;
				TripletMinusState(1) = 0.0;
				TripletMinusState(2) = 0.0;
				TripletMinusState(3) = 1.0;
				InitialStateVector = TripletMinusState;
				this->Log() << "Triplet minus initial state." << std::endl;
			}
			else if (InitialStateLower == "tripletzero")
			{
				arma::cx_mat TripletZeroState(4, 1);
				TripletZeroState(0) = 0.0;
				TripletZeroState(1) = 1.0 / sqrt(2);
				TripletZeroState(2) = 1.0 / sqrt(2);
				TripletZeroState(3) = 0.0;
				InitialStateVector = TripletZeroState;
				this->Log() << "Triplet zero initial state." << std::endl;
			}
			else if (InitialStateLower == "tripletplus")
			{
				arma::cx_mat TripletPlusState(4, 1);
				TripletPlusState(0) = 1.0;
				TripletPlusState(1) = 0.0;
				TripletPlusState(2) = 0.0;
				TripletPlusState(3) = 0.0;
				InitialStateVector = TripletPlusState;
				this->Log() << "Triplet plus initial state." << std::endl;
			}
			else
			{
				std::cout << "# ERROR: Invalid initial state value! It is set to a Singlet state." << std::endl;
				this->Log() << "Initial state is undefined. Setting it to a Singlet state" << std::endl;
				arma::cx_mat SingletState(4, 1);
				SingletState(0) = 0.0;
				SingletState(1) = 1.0 / sqrt(2);
				SingletState(2) = -1.0 / sqrt(2);
				SingletState(3) = 0.0;
				InitialStateVector = SingletState;
			}

			// Check transitions, rates and projection operators
			auto transitions = (*i)->Transitions();
			arma::sp_cx_mat P;
			arma::sp_cx_mat Sum(4 * Z, 4 * Z);
			int num_transitions = 0;

			arma::vec rates(1, 1);
			std::map<int, arma::sp_cx_mat> Operators;
			double kmin;
			kmin = 0.0;
			double kmax;
			kmax = 0.0;

			if (!time_dependent_transitions)
			{
				// Gather rates and operators
				for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
				{
					if ((*j)->SourceState() == nullptr)
						continue;
					if (!space.GetState((*j)->SourceState(), P))
					{
						std::cout << "# ERROR: Could not obtain projection matrix!" << std::endl;
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						return 1;
					}
					if (num_transitions != 0)
					{
						rates.insert_rows(num_transitions, 1);
					}
					Operators[num_transitions] = P;
					rates(num_transitions) = (*j)->Rate();
					num_transitions++;
				}
				kmin = rates.min();
				kmax = rates.max();
			}
			else
			{
				// Gather operators
				for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
				{
					if ((*j)->SourceState() == nullptr)
						continue;
					if (!space.GetState((*j)->SourceState(), P))
					{
						std::cout << "# ERROR: Could not obtain projection matrix!" << std::endl;
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						return 1;
					}
					if (num_transitions != 0)
					{
						rates.insert_rows(num_transitions, 1);
					}
					Operators[num_transitions] = P;
					num_transitions++;
				}
			}

			bool symmetric = false;
			arma::sp_cx_mat K;
			K.zeros(4 * Z, 4 * Z);
			// Do this check only when the transition rates are time-independent
			if (!time_dependent_transitions)
			{
				// Check if the recombination rates are symmetric or not
				if (std::abs(arma::accu(rates - rates.max())) > 0)
				{
					for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
					{
						if ((*j)->SourceState() == nullptr)
							continue;
						space.GetState((*j)->SourceState(), P);
						K += (*j)->Rate() / 2 * P;
					}
				}
				else
				{
					symmetric = true; // Symmetric Recombination
					// Check if one of the transition operators is not an identity operator
					for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
					{
						if ((*j)->SourceState() == nullptr)
							continue;
						space.GetState((*j)->SourceState(), P);
						if (this->is_identity_matrix(P))
						{
							symmetric = false;
						}
						K += (*j)->Rate() / 2 * P;
					}

					if (symmetric)
					{
						K = arma::sp_cx_mat();
						this->Log() << "Recombination rates are equal, hence Symmetric Recombination condition is satisfied and calculations will be simplified." << std::endl;
					}
				}
			}
			else
			{
				space.SetTime(0.0);
				if (!space.StaticTotalReactionOperator(K))
				{
					this->Log() << "Warning: Failed to update matrix representation of transitions!" << std::endl;
				}
			}

			// Obtain the sampling method and set up states for time-propagation
			arma::cx_mat B;
			B.zeros(Z * 4, Z);

			for (int it = 0; it < Z; it++)
			{
				arma::colvec temp(Z);
				temp(it) = 1;
				B.col(it) = arma::kron(InitialStateVector, temp);
			}
			// Setting or calculating total time.
			double ttotal;
			double totaltime;
			this->Properties()->Get("totaltime", totaltime);
			ttotal = totaltime;
			if (ttotal > 2e-53)
			{
				this->Log() << "Total time is chosen as " << ttotal << " ns." << std::endl;
				this->Log() << "Total time is chosen as " << ttotal * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
			}
			else
			{
				this->Log() << "Both total time and epsilon are not given or not appropriately defined. Using the default." << std::endl;
				std::cout << "# ERROR: total time and/or epsilon are not defined/given! Using the default of 10000 ns." << std::endl;
				ttotal = 10000;
				this->Log() << "Total time is chosen as " << ttotal << " ns." << std::endl;
				this->Log() << "Total time is chosen as " << ttotal * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
			}

			// Setting timestep
			double dt;
			double timestep;
			this->Properties()->Get("timestep", timestep);
			dt = timestep;
			if (dt > std::pow(2, -53))
			{
				this->Log() << "Time step is chosen as " << dt << " ns." << std::endl;
				this->Log() << "Time step is chosen as " << dt * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
			}
			else
			{
				this->Log() << "Time step is undefined. Using the default." << std::endl;
				std::cout << "# ERROR: undefined time step! Using the default of 1 ns." << std::endl;
				dt = 1;
				this->Log() << "Time step is chosen as " << dt << " ns." << std::endl;
				this->Log() << "Time step is chosen as " << dt * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
			}

			// Number of time propagation steps
			int num_steps = std::ceil(ttotal / dt);
			this->Log() << "Number of time propagation steps: " << num_steps << "." << std::endl;

			// Quantum yield corrections
			bool correction;

			this->Properties()->Get("yieldcorrections", correction);
			if (correction)
			{
				this->Log() << "Quantum yield corrections are turned on." << std::endl;
			}
			else
			{
				this->Log() << "Quantum yield corrections are turned off." << std::endl;
			}

			// Choose Propagation Method and other parameters
			std::string propmethod;
			this->Properties()->Get("propagationmethod", propmethod);

			std::string precision;
			this->Properties()->Get("precision", precision);

			int krylovsize;
			this->Properties()->Get("krylovsize", krylovsize);

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
					std::cout << "# ERROR: undefined precision. Using single digit precision!" << std::endl;
					this->Log() << "No precision for autoexpm method was defined. Using single digit precision." << std::endl;
					precision = "single";
				}
			}
			else if (propmethod == "krylov")
			{
				if (krylovsize > 0)
				{
					this->Log() << "Krylov basis size is chosen as " << krylovsize << "." << std::endl;
				}
				else
				{
					std::cout << "# ERROR: undefined size of the krylov subspace! Using the default size of 16." << std::endl;
					this->Log() << "Undefined size of the krylov subspace. Using the default size of 16." << std::endl;
					krylovsize = 16;
				}
			}
			else
			{
				std::cout << "# ERROR: undefined propagation method, using autoexpm with single accuracy!" << std::endl;
				this->Log() << "Undefined propagation method, using autoexpm with single accuracy." << std::endl;
				propmethod = "autoexpm";
				precision = "single";
			}

			// Get time-independent Hamiltonian
			space.SetTime(0.0);
			arma::sp_cx_mat H;
			if (!space.StaticHamiltonian(H))
			{
				this->Log() << "Failed to obtain the Hamiltonian in Hilbert Space." << std::endl;
				std::cout << "# ERROR: Failed to obtain the Hamiltonian!" << std::endl;
				return 1;
			}

			arma::mat ExptValues;
			ExptValues.zeros(num_steps, num_transitions);
			arma::vec Identity(num_steps);
			arma::vec time(num_steps);

			if (time_dependent_transitions && !time_dependent_hamiltonian)
			{
				// Case 1: time_dependent_transitions is true but time_dependent_hamiltonian is false
				arma::sp_cx_mat dK(4 * Z, 4 * Z);
				if (propmethod == "autoexpm")
				{
					// Propagation using autoexpm for matrix exponential
					arma::mat M; // used for variable estimation

					// General Case

					// Include the recombination operator K
					H = H - arma::cx_double(0.0, 1.0) * K;
					for (int k = 0; k < num_steps; k++)
					{
						// Set the current time
						double current_time = k * dt;
						time(k) = current_time;

						// Set the currentime for the Dynamic Hamiltonian
						space.SetTime(current_time);

						auto transitions = (*i)->Transitions();

						int idx = 0;
						for (auto o = transitions.begin(); o != transitions.end(); o++)
						{
							double rate = (*o)->Rate();
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = abs_trace / Z;
							ExptValues(k, idx) = rate * expected_value;
							idx++;
						}
						if (!space.DynamicTotalReactionOperator(dK))
						{
							this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
						}

						dK = (-arma::cx_double(0.0, 1.0)) * dK;

						// Update B using the Higham propagator
						H = H + dK;
						B = space.HighamProp(H, B, -arma::cx_double(0.0, 1.0) * dt, precision, M);
						H = H - dK;
					}
				}
				else if (propmethod == "krylov")
				{
					// Propagation using krylov subspace methods
					// Include the recombination operator K
					H = H - arma::cx_double(0.0, 1.0) * K;
					// #pragma omp parallel for
					for (int itr = 0; itr < Z; itr++)
					{
						arma::cx_vec prop_state = B.col(itr);
						for (int k = 0; k < num_steps; k++)
						{
							// Set the current time
							double current_time = k * dt;
							time(k) = current_time;

							// Set the currentime for the Dynamic Hamiltonian
							space.SetTime(current_time);

							auto transitions = (*i)->Transitions();

							int idx = 0;
							for (auto o = transitions.begin(); o != transitions.end(); o++)
							{
								double rate = (*o)->Rate();
								double expected_value = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
								ExptValues(k, idx) += rate * expected_value;
								idx++;
							}

							if (!space.DynamicTotalReactionOperator(dK))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}

							dK = (-arma::cx_double(0.0, 1.0)) * dK;

							// Update B using Krylov Subspace propagator
							H = H + dK;
							prop_state = space.KrylovExpmGeneral(H, prop_state, -arma::cx_double(0.0, 1.0) * dt, krylovsize, 4 * Z);
							H = H - dK;
						}
						B.col(itr) = prop_state;
					}
					ExptValues /= Z;
				}
			}
			else if (time_dependent_hamiltonian && !time_dependent_transitions)
			{
				// Case 2: time_dependent_hamiltonian is true but time_dependent_transitions is false
				arma::sp_cx_mat dH = arma::sp_cx_mat(4 * Z, 4 * Z);
				if (propmethod == "autoexpm")
				{
					// Propagation using autoexpm for matrix exponential
					arma::mat M; // used for variable estimation
					if (symmetric)
					{
						// Recombination reaction rates are equal
						for (int k = 0; k < num_steps; k++)
						{
							// Set the current time
							double current_time = k * dt;
							time(k) = current_time;

							// Set the currentime for the Dynamic Hamiltonian
							space.SetTime(current_time);

							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < num_transitions; idx++)
							{
								double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
								double expected_value = std::exp(-kmin * current_time) * abs_trace / Z;
								ExptValues(k, idx) = expected_value;
							}

							if (!space.DynamicHamiltonian(dH))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}

							// Update B using the Higham propagator
							H = H + dH;
							B = space.HighamProp(H, B, -dt * arma::cx_double(0.0, 1.0), precision, M);
							H = H - dH;
						}
					}
					else
					{
						// General Case
						// Include the recombination operator K
						H = H - arma::cx_double(0.0, 1.0) * K;
						for (int k = 0; k < num_steps; k++)
						{
							// Set the current time
							double current_time = k * dt;
							time(k) = current_time;

							// Set the currentime for the Dynamic Hamiltonian
							space.SetTime(current_time);

							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < num_transitions; idx++)
							{
								double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
								double expected_value = abs_trace / Z;
								ExptValues(k, idx) = expected_value;
							}

							if (!space.DynamicHamiltonian(dH))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}

							// Update B using the Higham propagator
							H = H + dH;
							B = space.HighamProp(H, B, -arma::cx_double(0.0, 1.0) * dt, precision, M);
							H = H - dH;
						}
					}
				}
				else if (propmethod == "krylov")
				{
					// Propagation using krylov subspace method for matrix exponential
					if (symmetric)
					{
						// recombination rates are equal

						// #pragma omp parallel for
						for (int itr = 0; itr < Z; itr++)
						{
							arma::cx_vec prop_state = B.col(itr);
							for (int k = 0; k < num_steps; k++)
							{
								// Set the current time
								double current_time = k * dt;
								time(k) = current_time;

								// Set the currentime for the Dynamic Hamiltonian
								space.SetTime(current_time);

								// Calculate the expected values for each transition operator
								for (int idx = 0; idx < num_transitions; idx++)
								{
									double result = std::exp(-kmin * current_time) * std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
									ExptValues(k, idx) += result;
								}

								if (!space.DynamicHamiltonian(dH))
								{
									this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
								}

								// Update B using Krylov Subspace propagator
								H = H + dH;
								prop_state = space.KrylovExpmSymm(H, prop_state, -arma::cx_double(0.0, 1.0) * dt, krylovsize, 4 * Z);
								H = H - dH;
							}
							B.col(itr) = prop_state;
						}
						ExptValues /= Z;
					}
					else
					{
						// Include the recombination operator K
						H = H - arma::cx_double(0.0, 1.0) * K;
						// #pragma omp parallel for
						for (int itr = 0; itr < Z; itr++)
						{
							arma::cx_vec prop_state = B.col(itr);
							for (int k = 0; k < num_steps; k++)
							{
								// Set the current time
								double current_time = k * dt;
								time(k) = current_time;

								// Set the currentime for the Dynamic Hamiltonian
								space.SetTime(current_time);

								// Calculate the expected values for each transition operator
								for (int idx = 0; idx < num_transitions; idx++)
								{
									double result = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
									ExptValues(k, idx) += result;
								}

								if (!space.DynamicHamiltonian(dH))
								{
									this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
								}

								// Update B using Krylov Subspace propagator
								H = H + dH;
								prop_state = space.KrylovExpmGeneral(H, prop_state, -arma::cx_double(0.0, 1.0) * dt, krylovsize, 4 * Z);
								H = H - dH;
							}
							B.col(itr) = prop_state;
						}
						ExptValues /= Z;
					}
				}
			}
			else if (time_dependent_hamiltonian && time_dependent_transitions)
			{
				// Case 3: both time_dependent_hamiltonian and time_dependent_transitions are true
				arma::sp_cx_mat dK(4 * Z, 4 * Z);
				arma::sp_cx_mat dH(4 * Z, 4 * Z);
				if (propmethod == "autoexpm")
				{
					// Propagation using autoexpm for matrix exponential
					arma::mat M; // used for variable estimation

					// General Case

					// Include the recombination operator K
					H = H - arma::cx_double(0.0, 1.0) * K;
					for (int k = 0; k < num_steps; k++)
					{
						// Set the current time
						double current_time = k * dt;
						time(k) = current_time;

						// Set the currentime for the Dynamic Hamiltonian
						space.SetTime(current_time);

						auto transitions = (*i)->Transitions();

						int idx = 0;
						for (auto o = transitions.begin(); o != transitions.end(); o++)
						{
							double rate = (*o)->Rate();
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = abs_trace / Z;
							ExptValues(k, idx) = rate * expected_value;
							idx++;
						}
						if (!space.DynamicTotalReactionOperator(dK))
						{
							this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
						}

						dK = (-arma::cx_double(0.0, 1.0)) * dK;

						if (!space.DynamicHamiltonian(dH))
						{
							this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
						}

						// Update B using the Higham propagator
						H = H + dK + dH;
						B = space.HighamProp(H, B, -arma::cx_double(0.0, 1.0) * dt, precision, M);
						H = H - dK - dH;
					}
				}
				else if (propmethod == "krylov")
				{
					// Propagation using krylov subspace methods
					// Include the recombination operator K
					H = H - arma::cx_double(0.0, 1.0) * K;
					// #pragma omp parallel for
					for (int itr = 0; itr < Z; itr++)
					{
						arma::cx_vec prop_state = B.col(itr);
						for (int k = 0; k < num_steps; k++)
						{
							// Set the current time
							double current_time = k * dt;
							time(k) = current_time;

							// Set the currentime for the Dynamic Hamiltonian
							space.SetTime(current_time);

							auto transitions = (*i)->Transitions();

							int idx = 0;
							for (auto o = transitions.begin(); o != transitions.end(); o++)
							{
								double rate = (*o)->Rate();
								double expected_value = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
								ExptValues(k, idx) += rate * expected_value;
								idx++;
							}

							if (!space.DynamicTotalReactionOperator(dK))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}

							dK = (-arma::cx_double(0.0, 1.0)) * dK;

							if (!space.DynamicHamiltonian(dH))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}

							// Update B using Krylov Subspace propagator
							H = H + dK + dH;
							prop_state = space.KrylovExpmGeneral(H, prop_state, -arma::cx_double(0.0, 1.0) * dt, krylovsize, 4 * Z);
							H = H - dK - dH;
						}
						B.col(itr) = prop_state;
					}
					ExptValues /= Z;
				}
			}

			// Obtain results
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->WriteStandardOutput(this->Data());

			arma::mat ans = arma::trapz(time, ExptValues);

			for (int it = 0; it < num_transitions; it++)
			{
				if (correction & !time_dependent_transitions)
				{
					// Quantum yields with correction factor
					ans(0, it) = ans(0, it) * rates(it) / (1 - std::exp(-ttotal * kmax));
				}
				else if (!time_dependent_transitions)
				{
					// Quantym yields without correction factor
					ans(0, it) = ans(0, it) * rates(it);
				}
				this->Data() << std::setprecision(6) << ans(0, it) << " ";
			}

			this->Data() << std::endl;

			// This commented section bellow was only for testing purposes

			// std::cout << std::endl;
			// std::cout << "Quantum Yields:" << std::endl;
			// std::cout << std::setprecision(6) <<  ans << std::endl;
			// std::cout << std::endl;

			// std::cout << "Accumulated sum of the quantum yields:" << std::endl;
			// std::cout << std::setprecision(6) << arma::accu(ans) << std::endl;
			// std::cout << std::endl;
		}
		return true;
	}
	bool TaskDynamicHSDirectSpectra::is_identity_matrix(arma::sp_cx_mat &matrix)
	{
		// Check if the matrix is square.
		if (matrix.n_rows != matrix.n_cols)
		{
			return false;
		}

		double EPSILON = 1e-14;
		// Check if all the diagonal elements of the matrix are equal to 1.0.
		for (int i = 0; i < int(matrix.n_rows); i++)
		{
			if (std::abs(matrix(i, i).real() - 1.0) > EPSILON || std::abs(matrix(i, i).imag()) > EPSILON)
			{
				return false;
			}
		}

		// Check if all the non-diagonal elements of the matrix are equal to 0.0.
		for (int i = 0; i < int(matrix.n_rows); i++)
		{
			for (int j = 0; j < int(matrix.n_cols); j++)
			{
				if (i != j && (std::abs(matrix(i, j).real()) > EPSILON || std::abs(matrix(i, j).imag()) > EPSILON))
				{
					return false;
				}
			}
		}

		// If we reach here, then the matrix is an identity matrix.
		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskDynamicHSDirectSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Should yields be written per transition or per defined state?
			if (this->productYieldsOnly)
			{
				// Write each transition name
				auto transitions = (*i)->Transitions();
				for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
					_stream << (*i)->Name() << "." << (*j)->Name() << ".yield ";
			}
			else
			{
				// Write each state name
				auto states = (*i)->States();
				for (auto j = states.cbegin(); j != states.cend(); j++)
					_stream << (*i)->Name() << "." << (*j)->Name() << " ";
			}
		}
		_stream << std::endl;
	}

	bool TaskDynamicHSDirectSpectra::Validate()
	{
		this->Properties()->Get("transitionyields", this->productYieldsOnly);

		// Get the reaction operator type
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
}
