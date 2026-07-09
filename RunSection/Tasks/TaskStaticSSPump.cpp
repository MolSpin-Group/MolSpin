/////////////////////////////////////////////////////////////////////////
// TaskStaticSSPump implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskStaticSSPump.h"
#include "Transition.h"
#include "Operator.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSS Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSPump::TaskStaticSSPump(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticSSPump::~TaskStaticSSPump()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskStaticSSPump::RunLocal()
	{
		this->Log() << "Running method StaticSSPumplution." << std::endl;

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
		std::vector<std::pair<arma::cx_mat, arma::cx_vec>> P(systems.size()); // Create array containing a propagator and the current state of each system
		std::vector<SpinAPI::SpinSpace> spaces(systems.size());				  // Keep a SpinSpace object for each spin system

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

			// Get the Hamiltonian
			arma::cx_mat H;
			if (!space.Hamiltonian(H))
			{
				this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
				continue;
			}

			// Get a matrix to collect all the terms (the total Liouvillian)
			arma::cx_mat A = arma::cx_double(0.0, -1.0) * H;

			// Get the reaction operators, and add them to "A"
			arma::cx_mat K;
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
					this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to the Liouvillian for spin system \"" << (*i)->Name() << "\".\n";
				}
			}

			// Get the propagator and put it into the array together with the initial state
			P[ic] = std::pair<arma::cx_mat, arma::cx_vec>(arma::expmat(A * this->timestep), rho0vec);
			++ic;
		}

		// Output results at the initial step (before calculations)
		this->Data() << this->RunSettings()->CurrentStep() << " 0 "; // "0" refers to the time
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
			// Write first part of the data output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << (static_cast<double>(n) * this->timestep) << " ";
			this->WriteStandardOutput(this->Data());

			// Loop through the systems again and progress a step
			ic = 0;
			for (auto i = systems.cbegin(); i != systems.cend(); i++)
			{
				// Take a step "first" is propagator and "second" is current state
				rho0vec = P[ic].first * P[ic].second;
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
	void TaskStaticSSPump::WriteHeader(std::ostream &_stream)
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
	bool TaskStaticSSPump::Validate()
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
	// -----------------------------------------------------
}
