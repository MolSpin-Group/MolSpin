/////////////////////////////////////////////////////////////////////////
// TaskPeriodicSSTimeEvo implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskPeriodicSSTimeEvo.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Interaction.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSS Constructors and Destructor
	// -----------------------------------------------------
	TaskPeriodicSSTimeEvo::TaskPeriodicSSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4),
																														  reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn), stepsPerPeriod(50)
	{
	}

	TaskPeriodicSSTimeEvo::~TaskPeriodicSSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskPeriodicSSTimeEvo::RunLocal()
	{
		this->Log() << "Running method PeriodicSSTimeEvolution." << std::endl;

		// If this is the first step, write header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Temporary results
		arma::cx_mat rho0;
		arma::cx_vec rho0vec;

		// Obtain spin systems
		auto systems = this->SpinSystems();
		std::vector<std::pair<std::vector<arma::cx_mat>, arma::cx_vec>> P(systems.size()); // Create array containing a propagator and the current state of each system
		std::vector<SpinAPI::SpinSpace> spaces(systems.size());							   // Keep a SpinSpace object for each spin system
		std::vector<float> propagator_stepsize(systems.size());							   // Keep track of the timestep per propagator (this is not the same as the integration timestep)

		// Loop through all SpinSystems
		int ic = 0; // System counter
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Make sure we have an initial state
			auto initial_states = (*i)->InitialState();
			if (initial_states.size() < 1)
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
				continue;
			}

			// -----------------------------------------------------
			// Find the period to use
			// -----------------------------------------------------
			float period = 0.0;
			for (auto j = (*i)->interactions_cbegin(); j != (*i)->interactions_cend(); j++)
			{
				if ((*j)->FieldType() != SpinAPI::InteractionFieldType::Static)
				{
					if ((*j)->FieldType() == SpinAPI::InteractionFieldType::Trajectory)
					{
						this->Log() << "Found interaction with a trajectory: \"" << (*j)->Name() << "\". Cannot deduce period of trajectory!\n";
					}
					else if ((*j)->FieldType() == SpinAPI::InteractionFieldType::LinearPolarization || (*j)->FieldType() == SpinAPI::InteractionFieldType::CircularPolarization)
					{
						if ((*j)->GetTDFrequency() > 1e-4)
						{
							double p = 2.0 * arma::datum::pi / (*j)->GetTDFrequency();
							this->Log() << "Found interaction with period " << p << " (angular frequency " << (*j)->GetTDFrequency() << " rad/ns, frequency " << (1000.0 / p) << " MHz).\n";
							if (p > period)
							{
								period = p;
							}
						}
						else
						{
							this->Log() << "Ignoring interaction \"" << (*j)->Name() << "\". Period too large!\n";
						}
					}
				}
			}
			if (period < 1e-10)
			{
				this->Log() << "WARNING: No period found, using totaltime as the period instead!" << std::endl;
				period = this->totaltime;
			}
			propagator_stepsize[ic] = period / static_cast<double>(this->stepsPerPeriod);
			if (propagator_stepsize[ic] < this->timestep)
			{
				this->Log() << "WARNING: Propagator stepsize is smaller than integration timestep!" << std::endl;
			}
			this->Log() << "Using period " << period << " ns and a propagator timestep of " << propagator_stepsize[ic] << " ns." << std::endl;
			// -----------------------------------------------------

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(true);
			space.SetReactionOperatorType(this->reactionOperators);
			spaces[ic] = space;

			// Get the initial state
			for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
			{
				arma::cx_mat tmp_rho0;
				if (!space.GetState(*j, tmp_rho0))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
					continue;
				}
				if (j == initial_states.cbegin())
					rho0 = tmp_rho0;
				else
					rho0 += tmp_rho0;
			}
			rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

			// Convert initial state to superoperator space
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

			// Get the static Hamiltonian
			arma::cx_mat sH;
			if (!space.StaticHamiltonian(sH))
			{
				this->Log() << "Failed to obtain static Hamiltonian in superspace." << std::endl;
				continue;
			}

			// Get the reaction operators, and add them to "A"
			arma::cx_mat K;
			if (!space.TotalReactionOperator(K))
			{
				this->Log() << "Warning: Failed to obtain matrix representation of the reaction operators!" << std::endl;
			}

			// Get the propagator and put it into the array together with the initial state
			std::vector<arma::cx_mat> PropagatorList;
			P[ic] = std::pair<std::vector<arma::cx_mat>, arma::cx_vec>(PropagatorList, rho0vec);

			// Fill in the propagators
			for (unsigned int n = 0; n < this->stepsPerPeriod; n++)
			{
				// Set the time
				space.SetTime(static_cast<double>(n) * propagator_stepsize[ic]);

				// Get the dynamic Hamiltonian
				arma::cx_mat dH;
				if (!space.DynamicHamiltonian(dH))
				{
					this->Log() << "Failed to obtain dynamic Hamiltonian in superspace." << std::endl;
					return false;
				}

				// Create the propagator
				P[ic].first.push_back(arma::expmat((arma::cx_double(0.0, -1.0) * (sH + dH) - K) * this->timestep));
			}

			// Move on to next system
			++ic;
		}

		// Output results at the initial step (before calculations)
		this->Data() << this->RunSettings()->CurrentStep() << " 0 ";
		this->WriteStandardOutput(this->Data());
		ic = 0;
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			arma::cx_mat PState;
			auto states = (*i)->States();
			for (auto j = states.cbegin(); j != states.cend(); j++)
			{
				if (!spaces[ic].GetState((*j), PState))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
					continue;
				}

				this->Data() << std::abs(arma::trace(PState * rho0)) << " ";
			}

			++ic;
		}
		this->Data() << std::endl;

		// Perform the calculation
		this->Log() << "Ready to perform calculation." << std::endl;
		unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
		for (unsigned int n = 1; n <= steps; n++)
		{
			// Set the time (in order to write correct standard output)
			for (unsigned int i = 0; i < systems.size(); i++)
				spaces[i].SetTime(static_cast<double>(n) * this->timestep);

			// Write first part of the data output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << (static_cast<double>(n) * this->timestep) << " ";
			this->WriteStandardOutput(this->Data());

			// Loop through the systems again and progress a step
			ic = 0;
			for (auto i = systems.cbegin(); i != systems.cend(); i++)
			{
				// Get the propagator index to use
				unsigned int PIndex = static_cast<unsigned int>(static_cast<double>(n) * this->timestep / propagator_stepsize[ic]) % P[ic].first.size();

				// Take a step "first" is propagator and "second" is current state
				rho0vec = P[ic].first[PIndex] * P[ic].second;
				P[ic].second = rho0vec;

				// Convert the resulting density operator back to its Hilbert space representation
				if (!spaces[ic].OperatorFromSuperspace(rho0vec, rho0))
				{
					this->Log() << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
					continue;
				}

				// Obtain the results
				arma::cx_mat PState;
				auto states = (*i)->States();
				for (auto j = states.cbegin(); j != states.cend(); j++)
				{
					if (!spaces[ic].GetState((*j), PState))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					this->Data() << std::abs(arma::trace(PState * rho0)) << " ";
				}

				++ic;
			}

			// Terminate the line in the data file after iteration through all spin systems
			this->Data() << std::endl;
		}

		this->Log() << "\nDone with calculations!" << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskPeriodicSSTimeEvo::WriteHeader(std::ostream &_stream)
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

	// Validation of the required input
	bool TaskPeriodicSSTimeEvo::Validate()
	{
		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;
		unsigned int inputStepsPerPeriod = 0;

		// Get timestep
		if (this->Properties()->Get("timestep", inputTimestep))
		{
			if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
			{
				this->timestep = inputTimestep;
			}
			else
			{
				// We can run the calculation if an invalid timestep was specified
				return false;
			}
		}

		// Get totaltime
		if (this->Properties()->Get("totaltime", inputTotaltime))
		{
			if (std::isfinite(inputTotaltime) && inputTotaltime > 0.0)
			{
				this->totaltime = inputTotaltime;
			}
			else
			{
				// We can run the calculation if an invalid total time was specified
				return false;
			}
		}

		// Get steps per period
		if (this->Properties()->Get("steps", inputStepsPerPeriod) || this->Properties()->Get("stepsperperiod", inputStepsPerPeriod))
		{
			if (inputStepsPerPeriod > 0)
			{
				this->stepsPerPeriod = inputStepsPerPeriod;
			}
			else
			{
				// We can run the calculation if an invalid number of steps per period was specified
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
	// -----------------------------------------------------
}
