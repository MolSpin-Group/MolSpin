/////////////////////////////////////////////////////////////////////////
// TaskStaticHSStochTimeEvo implementation (RunSection module) by Gediminas Pazera and Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskStaticHSStochTimeEvo.h"
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

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticHSStochTimeEvo Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSStochTimeEvo::TaskStaticHSStochTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(0.1), totaltime(1000), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn), productYieldsOnly(false)
	{
	}

	TaskStaticHSStochTimeEvo::~TaskStaticHSStochTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskStaticHSStochTimeEvo protected methods
	// -----------------------------------------------------
	bool TaskStaticHSStochTimeEvo::RunLocal()
	{
		this->Log() << "Running method StaticHS_Stochastic_TimeEvo." << std::endl;

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

			// Get the Hamiltonian
			arma::sp_cx_mat H;
			if (!space.Hamiltonian(H))
			{
				this->Log() << "Failed to obtain the Hamiltonian in Hilbert Space." << std::endl;
				std::cout << "# ERROR: Failed to obtain the Hamiltonian!" << std::endl;
				return 1;
			}

			// Check transitions, rates and projection operators
			auto transitions = (*i)->Transitions();
			arma::sp_cx_mat P;
			int num_transitions = 0;

			arma::vec rates(1, 1);
			std::map<int, arma::sp_cx_mat> Operators;
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

			double kmin = rates.min(); // The slowest process

			bool symmetric = false;
			arma::sp_cx_mat K;
			K.zeros(InitialStateVector.n_rows * Z, InitialStateVector.n_rows * Z);
			// Check if symmetric recombination or not
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

			// Setting or calculating total time.
			double ttotal;
			double totaltime;
			this->Properties()->Get("totaltime", totaltime);
			ttotal = totaltime;
			double epsilon;
			this->Properties()->Get("epsilon", epsilon);
			if (ttotal > std::pow(2, -53))
			{
				this->Log() << "Total time is chosen as " << ttotal << " ns." << std::endl;
				this->Log() << "Total time is chosen as " << ttotal * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
			}
			else
			{
				this->Log() << "No total time is given or it is incorrect: checking if epsilon is defined." << std::endl;
				if (epsilon > 0)
				{
					this->Log() << "Epsilon is " << epsilon << "." << std::endl;
					ttotal = std::log(1 / epsilon) / kmin;
					this->Log() << "Estimated total time is " << ttotal << " ns." << std::endl;
					this->Log() << "Estimated total time is " << ttotal * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
				}
				else
				{
					this->Log() << "Both total time and epsilon are not given or not appropriately defined. Using the default." << std::endl;
					std::cout << "# ERROR: total time and/or epsilon are not defined/given! Using the default of 10000 ns." << std::endl;
					ttotal = 10000;
					this->Log() << "Total time is chosen as " << ttotal << " ns." << std::endl;
					this->Log() << "Total time is chosen as " << ttotal * 1e-3 * gamma_e << " rad mT^-1." << std::endl;
				}
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

			int krylovsize = 0;
			this->Properties()->Get("krylovsize", krylovsize);

			double krylovtol = 0;
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
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						std::cout << "# ERROR: undefined tolerance for krylov subspace propagation! Using the default of 1e-16." << std::endl;
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
				else
				{
					std::cout << "# ERROR: undefined size of the krylov subspace! Using the default size of 16." << std::endl;
					this->Log() << "Undefined size of the krylov subspace. Using the default size of 16." << std::endl;
					krylovsize = 16;
					if (krylovtol > 0)
					{
						this->Log() << "Tolerance for krylov propagation is chosen as " << krylovtol << "." << std::endl;
					}
					else
					{
						std::cout << "# ERROR: undefined tolerance for krylov subspace propagation! Using the default of 1e-16." << std::endl;
						this->Log() << "Undefined tolerance for the krylov subspace. Using the default of 1e-16." << std::endl;
						krylovtol = 1e-16;
					}
				}
			}
			else
			{
				std::cout << "# WARNING: Undefined propagation method, using normal exponential method." << std::endl; // autoexpm with single accuracy!" << std::endl;
				this->Log() << "WARNING: Undefined propagation method, using normal exponential method." << std::endl; // autoexpm with single accuracy." << std::endl;
				propmethod = "normal";
			}

			//// Initialize time propagation placeholders
			arma::mat ExptValues;
			ExptValues.zeros(num_steps, num_transitions);
			arma::vec time(num_steps);

			// Propagate the system in time using the specified method

			// Propagation using autoexpm for matrix exponential
			if (propmethod == "autoexpm")
			{
				arma::mat M; // used for variable estimation
				// Symmetric matrix in the exponential
				if (symmetric)
				{
					for (int k = 0; k < num_steps; k++)
					{
						// Set the current time
						double current_time = k * dt;

						// Obtain results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << current_time << " ";
						this->WriteStandardOutput(this->Data());

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < num_transitions; idx++)
						{
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = std::exp(-kmin * current_time) * abs_trace / mc_samples;
							this->Data() << " " << expected_value;
						}

						// Update B using the Higham propagator
						B = space.HighamProp(H, B, -dt * arma::cx_double(0.0, 1.0), precision, M);
						this->Data() << std::endl;
					}
				}
				// Non-symmetric matrix in the exponential
				else
				{
					// Include the recombination operator K
					H = -H * arma::cx_double(0.0, 1.0) - K;
					for (int k = 0; k < num_steps; k++)
					{
						// Set the current time
						double current_time = k * dt;

						// Obtain results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << current_time << " ";
						this->WriteStandardOutput(this->Data());

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < num_transitions; idx++)
						{
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = abs_trace / mc_samples;
							this->Data() << " " << expected_value;
						}

						// Update B using the Higham propagator
						B = space.HighamProp(H, B, dt, precision, M);
						this->Data() << std::endl;
					}
				}

				// Propagation using krylov subspace method for matrix exponential
			}
			else if (propmethod == "krylov")
			{
				// Initialize time propagation placeholders
				arma::mat ExptValues;
				ExptValues.zeros(num_steps, num_transitions);
				arma::vec time(num_steps);

				// Symmetric matrix in the exponential
				if (symmetric)
				{
					// #pragma omp parallel for
					for (int itr = 0; itr < mc_samples; itr++)
					{
						arma::cx_vec prop_state = B.col(itr);

						// Set the current time
						double current_time = 0;
						time(0) = current_time;

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < num_transitions; idx++)
						{
							double result = std::exp(-kmin * current_time) * std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
							ExptValues(0, idx) += result;
						}

						arma::cx_mat Hessen; // Upper Hessenberg matrix
						Hessen.zeros(krylovsize, krylovsize);

						arma::cx_mat KryBasis(InitialStateVector.n_rows * Z, krylovsize, arma::fill::zeros); // Orthogonal krylov subspace

						KryBasis.col(0) = prop_state / norm(prop_state);

						double h_mplusone_m;
						space.LanczosProcess(H, prop_state, KryBasis, Hessen, krylovsize, h_mplusone_m);

						arma::cx_colvec e1;
						e1.zeros(krylovsize);
						e1(0) = 1;
						arma::cx_colvec ek;
						ek.zeros(krylovsize);
						ek(krylovsize - 1) = 1;

						arma::cx_vec cx = arma::expmat(-arma::cx_double(0.0, 1.0) * Hessen * dt) * e1;

						prop_state = norm(prop_state) * KryBasis * cx;

						int k = 1;
						int j = 0;

						while (k < num_steps)
						{
							// Set the current time
							current_time = k * dt;
							time(k) = current_time;

							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < num_transitions; idx++)
							{
								double result = std::exp(-kmin * current_time) * std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
								ExptValues(k, idx) += result;
							}

							// Update Krylov Subspace if the tolerance is reached
							if (h_mplusone_m * std::abs(arma::cdot(ek, cx)) > krylovtol)
							{
								// std::cout << "Restarted after: " << j << " iterations." <<  std::endl;
								j = 0;

								Hessen.zeros(krylovsize, krylovsize);
								KryBasis.zeros(InitialStateVector.n_rows * Z, krylovsize);

								KryBasis.col(0) = prop_state / norm(prop_state);
								space.LanczosProcess(H, prop_state, KryBasis, Hessen, krylovsize, h_mplusone_m);
								cx = arma::expmat(-arma::cx_double(0.0, 1.0) * Hessen * dt) * e1;

								// Update the state using Krylov Subspace propagator
								prop_state = norm(prop_state) * KryBasis * cx;
							}
							else
							{
								j = j + 1;
								cx = arma::expmat(-arma::cx_double(0.0, 1.0) * Hessen * dt) * cx;

								// Update B using Krylov Subspace propagator
								prop_state = norm(prop_state) * KryBasis * cx;
							}
							k++;
						}
					}
					ExptValues /= mc_samples;

					for (int k = 0; k < num_steps; k++)
					{
						// Obtain results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << time(k) << " ";
						this->WriteStandardOutput(this->Data());

						for (int idx = 0; idx < num_transitions; idx++)
						{
							this->Data() << " " << ExptValues(k, idx);
						}
						this->Data() << std::endl;
					}
				}
				// Non-symmetric matrix in the exponential
				else
				{
					// Include the recombination operator K
					H = -(H * arma::cx_double(0.0, 1.0) + K);

					// #pragma omp parallel for
					for (int itr = 0; itr < mc_samples; itr++)
					{
						arma::cx_vec prop_state = B.col(itr);

						// Set the current time
						double current_time = 0;

						time(0) = current_time;

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < num_transitions; idx++)
						{
							double result = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
							ExptValues(0, idx) += result;
						}

						arma::cx_mat Hessen; // Upper Hessenberg matrix
						Hessen.zeros(krylovsize, krylovsize);

						arma::cx_mat KryBasis(InitialStateVector.n_rows * Z, krylovsize, arma::fill::zeros); // Orthogonal krylov subspace

						KryBasis.col(0) = prop_state / norm(prop_state);

						double h_mplusone_m;
						space.ArnoldiProcess(H, prop_state, KryBasis, Hessen, krylovsize, h_mplusone_m);

						arma::cx_colvec e1;
						e1.zeros(krylovsize);
						e1(0) = 1;
						arma::cx_colvec ek;
						ek.zeros(krylovsize);
						ek(krylovsize - 1) = 1;

						arma::cx_vec cx = arma::expmat(Hessen * dt) * e1;

						prop_state = norm(prop_state) * KryBasis * cx;

						int k = 1;

						while (k < num_steps)
						{
							// Set the current time
							current_time = k * dt;
							time(k) = current_time;

							// Calculate the expected values for each transition operator
							for (int idx = 0; idx < num_transitions; idx++)
							{
								double result = std::abs(arma::cdot(prop_state, Operators[idx] * prop_state));
								ExptValues(k, idx) += result;
							}

							Hessen.zeros(krylovsize, krylovsize);
							KryBasis.zeros(InitialStateVector.n_rows * Z, krylovsize);

							KryBasis.col(0) = prop_state / norm(prop_state);

							space.ArnoldiProcess(H, prop_state, KryBasis, Hessen, krylovsize, h_mplusone_m);
							cx = arma::expmat(Hessen * dt) * e1;

							// Update the state using Krylov Subspace propagator
							prop_state = norm(prop_state) * KryBasis * cx;

							k++;
						}
					}
					ExptValues /= mc_samples;

					for (int k = 0; k < num_steps; k++)
					{
						// Obtain results
						this->Data() << this->RunSettings()->CurrentStep() << " ";
						this->Data() << time(k) << " ";
						this->WriteStandardOutput(this->Data());

						for (int idx = 0; idx < num_transitions; idx++)
						{
							this->Data() << " " << ExptValues(k, idx);
						}
						this->Data() << std::endl;
					}
				}
			}
			else if (propmethod == "normal")
			{
				this->Log() << "Using robust matrix exponential propagator for time-independent Hamiltonian." << std::endl;
				// Initialize time propagation placeholders

				if (symmetric)
				{

					arma::sp_cx_mat H_total = arma::cx_double(0.0, -1.0) * H;

					// Precompute the matrix exponential for the entire time step
					arma::cx_mat exp_H = arma::expmat(arma::cx_mat(H_total) * dt);

					// Propagate B
					for (int k = 0; k < num_steps; ++k)
					{
						// Set the current time
						double current_time = k * dt;
						time(k) = current_time;

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < num_transitions; idx++)
						{
							double abs_trace = std::abs(arma::trace(B.t() * Operators[idx] * B));
							double expected_value = std::exp(-kmin * current_time) * abs_trace / mc_samples;
							ExptValues(k, idx) = expected_value;
						}

						for (int i = 0; i < int(B.n_cols); ++i)
						{
							B.col(i) = exp_H * B.col(i);
						}
					}
				}
				else
				{
					// Include the recombination operator K
					arma::sp_cx_mat H_total = arma::cx_double(0.0, -1.0) * H - K;

					// Precompute the matrix exponential for the entire time step
					arma::cx_mat exp_H = arma::expmat(arma::cx_mat(H_total) * dt);

					// Propagate B
					for (int k = 0; k < num_steps; ++k)
					{
						// Set the current time
						double current_time = k * dt;
						time(k) = current_time;

						// Calculate the expected values for each transition operator
						for (int idx = 0; idx < num_transitions; ++idx)
						{
							double abs_trace = std::abs(arma::trace(B.t() * arma::cx_mat(Operators[idx]) * B));
							double expected_value = abs_trace / mc_samples;
							ExptValues(k, idx) = expected_value;
						}

						for (int i = 0; i < int(B.n_cols); ++i)
						{
							B.col(i) = exp_H * B.col(i);
						}
					}
				}

				for (int k = 0; k < num_steps; k++)
				{
					// Obtain results
					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << time(k) << " ";
					this->WriteStandardOutput(this->Data());

					for (int idx = 0; idx < num_transitions; idx++)
					{
						this->Data() << " " << ExptValues(k, idx);
					}
					this->Data() << std::endl;
				}
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}
		return true;
	}

	bool TaskStaticHSStochTimeEvo::is_identity_matrix(arma::sp_cx_mat &matrix)
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
	void TaskStaticHSStochTimeEvo::WriteHeader(std::ostream &_stream)
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

	// Validation
	bool TaskStaticHSStochTimeEvo::Validate()
	{
		this->Properties()->Get("transitionyields", this->productYieldsOnly);

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

}
