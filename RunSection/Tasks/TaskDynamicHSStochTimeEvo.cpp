/////////////////////////////////////////////////////////////////////////////////////////
// TaskDynamicHSStochTimeEvo implementation (RunSection module) by Gediminas Pazera and Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskDynamicHSStochTimeEvo.h"
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
	// TaskDynamicHSStochTimeEvo Constructors and Destructor
	// -----------------------------------------------------
	TaskDynamicHSStochTimeEvo::TaskDynamicHSStochTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(0.1), totaltime(1000), timedependentInteractions(false), timedependentTransitions(false), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn), productYieldsOnly(false)
	{
	}

	TaskDynamicHSStochTimeEvo::~TaskDynamicHSStochTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskDynamicHSStochTimeEvo protected methods
	// -----------------------------------------------------
	bool TaskDynamicHSStochTimeEvo::RunLocal()
	{
		this->Log() << "Running method DynamicHS-Stoch-TimeEvolution." << std::endl;

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
			std::vector<int> SpinNumbers;
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

			std::string InitialState;
			arma::cx_mat InitialStateVector;
			if (this->Properties()->Get("initialstate", InitialState))
			{
				// Set up states for time-propagation
				arma::cx_mat TaskInitialStateVector(4, 1);
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
					TaskInitialStateVector = SingletState;
					this->Log() << "Singlet initial state." << std::endl;
				}
				else if (InitialStateLower == "tripletminus")
				{
					arma::cx_mat TripletMinusState(4, 1);
					TripletMinusState(0) = 0.0;
					TripletMinusState(1) = 0.0;
					TripletMinusState(2) = 0.0;
					TripletMinusState(3) = 1.0;
					TaskInitialStateVector = TripletMinusState;
					this->Log() << "Triplet minus initial state." << std::endl;
				}
				else if (InitialStateLower == "tripletzero")
				{
					arma::cx_mat TripletZeroState(4, 1);
					TripletZeroState(0) = 0.0;
					TripletZeroState(1) = 1.0 / sqrt(2);
					TripletZeroState(2) = 1.0 / sqrt(2);
					TripletZeroState(3) = 0.0;
					TaskInitialStateVector = TripletZeroState;
					this->Log() << "Triplet zero initial state." << std::endl;
				}
				else if (InitialStateLower == "tripletplus")
				{
					arma::cx_mat TripletPlusState(4, 1);
					TripletPlusState(0) = 1.0;
					TripletPlusState(1) = 0.0;
					TripletPlusState(2) = 0.0;
					TripletPlusState(3) = 0.0;
					TaskInitialStateVector = TripletPlusState;
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
					TaskInitialStateVector = SingletState;
				}
				InitialStateVector = TaskInitialStateVector;
			}
			else
			{
				// Make sure we have an initial state
				auto initial_states = (*i)->InitialState();
				if (initial_states.size() < 1)
				{
					this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
					continue;
				}

				arma::cx_vec tmp_InitialStateVector;

				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					if (!space.GetStateSubSpace(*j, tmp_InitialStateVector))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}
				}

				InitialStateVector = arma::reshape(tmp_InitialStateVector, tmp_InitialStateVector.n_elem, 1);
			}

			int Z = space.SpaceDimensions() / InitialStateVector.n_rows; // Size of the nuclear spin subspace
			std::cout << "# Hilbert Space Size " << InitialStateVector.n_rows * Z << " x " << InitialStateVector.n_rows * Z << std::endl;
			this->Log() << "Hilbert Space Size " << InitialStateVector.n_rows * Z << " x " << InitialStateVector.n_rows * Z << std::endl;
			this->Log() << "Size of Spin Subspace " << Z << std::endl;

			// Random Number Generator Preparation
			std::random_device rand_dev;		// random number generator
			std::mt19937 generator(rand_dev()); // random number generator
			bool autoseed;
			this->Properties()->Get("autoseed", autoseed);

			if (!autoseed)
			{
				this->Log() << "Autoseed is off." << std::endl;
				double seednumber;
				this->Properties()->Get("seed", seednumber);
				if (seednumber != 0)
				{
					generator.seed(seednumber);
					this->Log() << "Seed number is " << seednumber << "." << std::endl;
				}
				else
				{
					this->Log() << "Undefined seed number! Setting to default of 1." << std::endl;
					std::cout << "# ERROR: undefined seed number! Setting to default of 1." << std::endl;
					seednumber = 1;
				}
			}
			else
			{
				this->Log() << "Autoseed is on." << std::endl;
			}

			// Defining the number of Monte Carlo samples
			int mc_samples;
			this->Properties()->Get("montecarlosamples", mc_samples);

			if (mc_samples > 0)
			{
				this->Log() << "Number of Monte Carlo samples is " << mc_samples << "." << std::endl;
			}
			else
			{
				this->Log() << "Undefined number of Monte Carlo samples. Setting it to default of 1000." << std::endl;
				std::cout << "# ERROR: undefined number of Monte Carlo samples! Setting the number to default of 1000." << std::endl;
				mc_samples = 1000;
			}

			// Obtain the sampling method and set up states for time-propagation
			arma::cx_mat B(InitialStateVector.n_rows * Z, mc_samples);
			std::string samplingmethod;

			this->Properties()->Get("samplingmethod", samplingmethod);
			if (samplingmethod == "")
			{
				for (int it = 0; it < mc_samples; it++)
				{
					B.col(it) = arma::kron(InitialStateVector, space.SUZstate(Z, generator));
				}
				this->Log() << "No sampling method was defined. Using SU(Z) spin states for Monte Carlo sampling." << std::endl;
			}
			else if (samplingmethod == "SUZ")
			{
				for (int it = 0; it < mc_samples; it++)
				{
					B.col(it) = arma::kron(InitialStateVector, space.SUZstate(Z, generator));
				}
				this->Log() << "Using SU(Z) spin states for Monte Carlo sampling." << std::endl;
			}
			else if (samplingmethod == "Coherent")
			{
				for (int it = 0; it < mc_samples; it++)
				{
					B.col(it) = arma::kron(InitialStateVector, space.CoherentState(i, generator));
				}
				this->Log() << "Using Coherent spin states for Monte Carlo sampling." << std::endl;
			}
			else
			{
				this->Log() << "Undefined sampling method! Skipping SpinSystem!" << std::endl;
				std::cout << "# ERROR: Undefined sampling method!" << std::endl;
				return 1;
			}

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

			// Check transitions, rates and projection operators
			auto transitions = (*i)->Transitions();
			arma::sp_cx_mat P;
			arma::sp_cx_mat Sum(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
			int num_transitions = 0;

			arma::vec rates(1, 1);
			std::map<int, arma::sp_cx_mat> Operators;
			double kmin;
			kmin = 0.0;

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
			K.zeros(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
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
				arma::sp_cx_mat dK(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
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

						// Obtain results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << current_time << " ";
						this->WriteStandardOutput(this->Data());

						int idx = 0;
						for (auto o = transitions.begin(); o != transitions.end(); o++)
						{
							double rate = (*o)->Rate();
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = abs_trace / mc_samples;
							ExptValues(k, idx) = rate * expected_value;
							this->Data() << " " << expected_value;
							idx++;
						}

						this->Data() << std::endl;

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
			}
			else if (time_dependent_hamiltonian && !time_dependent_transitions)
			{
				// Case 2: time_dependent_hamiltonian is true but time_dependent_transitions is false
				arma::sp_cx_mat dH = arma::sp_cx_mat(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
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

							// Obtain results
							this->Data() << this->RunSettings()->CurrentStep() << " ";
							this->Data() << current_time << " ";
							this->WriteStandardOutput(this->Data());

							// Set the currentime for the Dynamic Hamiltonian
							space.SetTime(current_time);

							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < num_transitions; idx++)
							{
								double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
								double expected_value = std::exp(-kmin * current_time) * abs_trace / mc_samples;
								ExptValues(k, idx) = expected_value;
								this->Data() << " " << expected_value;
							}

							this->Data() << std::endl;

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

							// Obtain results
							this->Data() << this->RunSettings()->CurrentStep() << " ";
							this->Data() << current_time << " ";
							this->WriteStandardOutput(this->Data());

							// Set the currentime for the Dynamic Hamiltonian
							space.SetTime(current_time);

							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < num_transitions; idx++)
							{
								double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
								double expected_value = abs_trace / mc_samples;
								ExptValues(k, idx) = expected_value;
								this->Data() << " " << expected_value;
							}

							this->Data() << std::endl;

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
			}
			else if (time_dependent_hamiltonian && time_dependent_transitions)
			{
				// Case 3: both time_dependent_hamiltonian and time_dependent_transitions are true
				arma::sp_cx_mat dK(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
				arma::sp_cx_mat dH(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
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

						// obtain results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << current_time << " ";
						this->WriteStandardOutput(this->Data());

						// Set the currentime for the Dynamic Hamiltonian
						space.SetTime(current_time);

						auto transitions = (*i)->Transitions();

						int idx = 0;
						for (auto o = transitions.begin(); o != transitions.end(); o++)
						{
							double rate = (*o)->Rate();
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = abs_trace / mc_samples;
							ExptValues(k, idx) = rate * expected_value;
							this->Data() << " " << expected_value;
							idx++;
						}

						this->Data() << std::endl;

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
			}
			else if(propmethod == "krylov")
			{
				arma::sp_cx_mat dK(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
				arma::sp_cx_mat dH(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
				double InitialTimeStep = this->timestep;
				double CurrentTime = 0;
				SpinAPI::SpinSpace::PropParam prop_param = this->GetTimeAdaptiveProperties(InitialTimeStep);
				std::vector<double> TimePoints;
				std::vector<double> TimeSteps;
				std::vector<std::vector<double>> ExptValues;
				prop_param.TimePrefactor = -arma::cx_double(0.0, 1.0);
				prop_param.UsePrefactor = true;
				
				//should stop it going too wild 
				prop_param.f2 = 1.5;
				prop_param.safety = 0.5;

				// Include the recombination operator K
				H = H - arma::cx_double(0.0, 1.0) * K;
				for(int itr = 0; itr < mc_samples; itr++)
				{
					arma::cx_vec prop_state = B.col(itr);
					this->timestep = InitialTimeStep;
					CurrentTime = 0;
					this->Log() << "Starting time evolution with timestep: " << this->timestep << ", total time: " << this->totaltime << ", minimum timestep: " << prop_param.min << ", maximum timestep: " << prop_param.max << std::endl;
					while (CurrentTime <= this->totaltime)
					{
						//set the current time
						if(ExptValues.size() < TimePoints.size() + 1)
						{
							ExptValues.push_back(std::vector<double>(num_transitions, 0.0));
						}
						if(itr == 0)
						{
							TimePoints.push_back(CurrentTime);
							TimeSteps.push_back(this->timestep);
						}
						space.SetTime(CurrentTime);
						SpinAPI::SpinSpace::TimePropReturnInfo r;

						int idx = 0;
						for(auto o = transitions.cbegin(); o != transitions.cend(); o++)
						{
							double rate = (*o)->Rate();
							double expected_value = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
							ExptValues.back()[idx] += (rate * expected_value) / mc_samples;
							idx++;
						}

						//get dh and dk
						if(time_dependent_transitions)
						{
							if (!space.DynamicTotalReactionOperator(dK))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}
							dK = (-arma::cx_double(0.0, 1.0)) * dK;
							H = H + dK;
						}
						if(time_dependent_hamiltonian)
						{
							if (!space.DynamicHamiltonian(dH))
							{
								this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
							}
							H = H + dH;
						}

						if(symmetric)
						{
							r = space.TimeAdaptiveKrylovSymm(H, prop_state, this->timestep, krylovsize, InitialStateVector.n_rows * Z, prop_param);
						}
						else
						{
							r = space.TimeAdaptiveKrylovGeneral(H, prop_state, this->timestep, krylovsize, InitialStateVector.n_rows * Z, prop_param);
						}

						if(time_dependent_transitions)
						{
							H = H - dK;
						}
						if(time_dependent_hamiltonian)
						{
							H = H - dH;
						}

						double t = r.timestep;
						bool a = r.step_accepted;

						CurrentTime += (a == true) ? this->timestep : t;
						this->timestep = t;
						prop_state = r.result;
					}
					prop_param.SetTimePoints = TimeSteps;
					prop_param.UseSetTimePoints = true;
					prop_param.ResetTrajectory();
					B.col(itr) = prop_state;
				}
				this->timestep = InitialTimeStep;
				for (size_t k = 0; k < TimePoints.size(); k++)
				{
					// Obtain results
					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << TimePoints[k] << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < num_transitions; idx++)
					{
						this->Data() << " " << ExptValues[k][idx];
					}
					this->Data() << std::endl;
				}
			}
		}
		return true;
	}

	bool TaskDynamicHSStochTimeEvo::is_identity_matrix(arma::sp_cx_mat &matrix)
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
	void TaskDynamicHSStochTimeEvo::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time(ns) ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Write each state name
			auto states = (*i)->States();
			for (auto j = states.cbegin(); j != states.cend(); j++)
				_stream << (*i)->Name() << "." << (*j)->Name() << " ";
		}
		_stream << std::endl;
	}

	bool TaskDynamicHSStochTimeEvo::Validate()
	{
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
