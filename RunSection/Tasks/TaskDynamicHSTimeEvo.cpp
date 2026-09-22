/////////////////////////////////////////////////////////////////////////
// TaskDynamicHSTimeEvo implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "../General/HS/HSRelaxationValidation.h"
#include <iostream>
#include "TaskDynamicHSTimeEvo.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSystem.h"
#include "Interaction.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSS Constructors and Destructor
	// -----------------------------------------------------
	TaskDynamicHSTimeEvo::TaskDynamicHSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(0.01), totaltime(1.0e+4), outputstride(1), modeQuantumYield(false), productYieldsOnly(false), timedependentInteractions(false), timedependentTransitions(false)
	{
	}

	TaskDynamicHSTimeEvo::~TaskDynamicHSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskDynamicHSTimeEvo::RunLocal()
	{
		std::string relaxationError;
		if (!General::HS::ValidateNoRelaxation(this->SpinSystems(),
			"TaskDynamicHSTimeEvo propagation representation", relaxationError))
		{ this->Log() << "ERROR: " << relaxationError << std::endl; return false; }

		this->Log() << "Running method DynamicHSTimeEvolution." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
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
			space.UseSuperoperatorSpace(false);
			space.SetTime(0.0);

			// Check whether any interactions or transitions are time-dependent
			this->timedependentInteractions = space.HasTimedependentInteractions();
			this->timedependentTransitions = space.HasTimedependentTransitions();

			// Get the Hamiltonian
			arma::cx_mat H;																			// Static part of the Hamiltonian
			arma::sp_cx_mat dH = arma::sp_cx_mat(space.SpaceDimensions(), space.SpaceDimensions()); // Dynamic part of the Hamiltonian
			if (!space.StaticHamiltonian(H) || (this->timedependentInteractions && !space.DynamicHamiltonian(dH)))
			{
				this->Log() << "Failed to obtain the Hamiltonian! Skipping system." << std::endl;
				continue;
			}

			// Get the total reaction operator
			arma::cx_mat K;																			// Static part of total reaction operator
			arma::sp_cx_mat dK = arma::sp_cx_mat(space.SpaceDimensions(), space.SpaceDimensions()); // Dynamic part of total reaction operator
			if (!space.StaticTotalReactionOperator(K) || (this->timedependentTransitions && !space.DynamicTotalReactionOperator(dK)))
			{
				this->Log() << "Warning: Failed to obtain matrix representation of Transitions!" << std::endl;
			}

			// Get the initial state
			arma::cx_mat rho0;
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

			// Get the number of steps
			unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));

			// Create vectors to collect the quantum states at all time steps
			auto states = (*i)->States();
			auto transitions = (*i)->Transitions();
			std::map<std::string, arma::vec> yields;
			if (this->modeQuantumYield)
			{
				// Are quantum yields collected per state or per transition?
				if (this->productYieldsOnly)
				{
					for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
						if ((*j)->SourceState() != nullptr)
							yields[(*j)->Name()] = arma::zeros<arma::vec>(steps);
				}
				else
				{
					for (auto j = states.cbegin(); j != states.cend(); j++)
						yields[(*j)->Name()] = arma::zeros<arma::vec>(steps);
				}
			}

			// Perform the calculation
			this->Log() << "Ready to perform calculation." << std::endl;

			// Output results for time 0
			if (!this->modeQuantumYield)
				OutputTimeEvolution(space, rho0, 0, states);

			// Run the time integration
			for (unsigned int n = 1; n <= steps; n++)
			{
				// Advance a timestep
				AdvanceStep_AsyncLeapfrog(H + dH, K + dK, rho0);

				// Refresh the Hamiltonian only if "n < steps" AND if there actually is a time-dependence
				space.SetTime(static_cast<double>(n) * this->timestep);
				if (n < steps && this->timedependentInteractions && !space.DynamicHamiltonian(dH))
				{
					this->Log() << "Warning: Failed to update the Hamiltonian matrix representation!" << std::endl;
				}

				// Refresh the reaction operators only if "n < steps" AND if there actually is a time-dependence
				if (n < steps && this->timedependentTransitions && !space.DynamicTotalReactionOperator(dK))
				{
					this->Log() << "Warning: Failed to update matrix representation of transitions!" << std::endl;
				}

				// Should we calculate the quantum yield rather than just the time evolution?
				if (this->modeQuantumYield)
				{
					GetQuantumYields(space, rho0, n, states, transitions, yields);
				}
				else
				{
					// If not, just output the time evolution
					if (n % this->outputstride == 0)
						OutputTimeEvolution(space, rho0, n, states);
				}
			}

			this->Log() << "\nFinished numerical integration with density operator trace of " << std::abs(arma::trace(rho0)) << ".\n";

			if (this->modeQuantumYield)
			{
				if (std::abs(arma::trace(rho0)) > 0.1)
					this->Log() << "More than 10% of the spin system has not decayed, you should increase totaltime to improve the accuracy.";
				else if (std::abs(arma::trace(rho0)) < 0.01)
					this->Log() << "Less than 1% the spin system has not decayed. Increasing totaltime would not improve the results.";
				else
					this->Log() << "Between 1% and 10% of the spin system has not decayed. Increasing totaltime could improve the results.";

				OutputQuantumYields(space, yields, steps, states, transitions);
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		return true;
	}

	// Timestep method
	void TaskDynamicHSTimeEvo::AdvanceStep_AsyncLeapfrog(const arma::cx_mat &_H, const arma::cx_mat &_K, arma::cx_mat &_rho)
	{
		arma::cx_mat dRho = arma::cx_double(0.0, -1.0) * (_H * _rho - _rho * _H) - (_K * _rho + _rho * _K);
		_rho += dRho * this->timestep / 2.0;
		dRho = (arma::cx_double(0.0, -1.0) * (_H * _rho - _rho * _H) - (_K * _rho + _rho * _K));
		_rho += dRho * this->timestep / 2.0;
	}

	// Writes the output for a timestep
	void TaskDynamicHSTimeEvo::OutputTimeEvolution(const SpinAPI::SpinSpace &_space, const arma::cx_mat &_rho, const unsigned int _step, const std::vector<SpinAPI::state_ptr> &_states)
	{
		// Obtain the results
		arma::cx_mat PState;
		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->Data() << (static_cast<double>(_step) * this->timestep) << " ";
		this->WriteStandardOutput(this->Data());
		for (auto j = _states.cbegin(); j != _states.cend(); j++)
		{
			if (!_space.GetState((*j), PState))
			{
				this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\"." << std::endl;
				continue;
			}

			this->Data() << std::abs(arma::trace(PState * _rho)) << " ";
		}

		// Terminate the line in the data file after iteration through all steps
		this->Data() << std::endl;
	}

	// Get the state projections
	void TaskDynamicHSTimeEvo::GetQuantumYields(const SpinAPI::SpinSpace &_space, const arma::cx_mat &_rho, const unsigned int _step, const std::vector<SpinAPI::state_ptr> &_states, const std::vector<SpinAPI::transition_ptr> &_transitions, std::map<std::string, arma::vec> &_yields)
	{
		// Obtain the state projections
		arma::cx_mat PState;

		if (this->productYieldsOnly)
		{
			for (auto j = _transitions.cbegin(); j != _transitions.cend(); j++)
			{
				// Make sure that there is a state object
				if ((*j)->SourceState() == nullptr)
					continue;

				if (!_space.GetState((*j)->SourceState(), PState))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\"." << std::endl;
					continue;
				}

				_yields[(*j)->Name()](_step - 1) = std::abs(arma::trace(PState * _rho));
			}
		}
		else
		{
			// Get yields using state projections
			for (auto j = _states.cbegin(); j != _states.cend(); j++)
			{
				if (!_space.GetState((*j), PState))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\"." << std::endl;
					continue;
				}

				_yields[(*j)->Name()](_step - 1) = std::abs(arma::trace(PState * _rho));
			}
		}
	}

	// Output the quantum yield
	void TaskDynamicHSTimeEvo::OutputQuantumYields(const SpinAPI::SpinSpace &_space, std::map<std::string, arma::vec> &_yields, const unsigned int _steps, const std::vector<SpinAPI::state_ptr> &_states, const std::vector<SpinAPI::transition_ptr> &_transitions)
	{
		arma::vec X = arma::linspace<arma::vec>(0.0, this->totaltime, _steps);

		// Write standard outputs and prepare a matrix to hold state projections
		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->WriteStandardOutput(this->Data());

		// There are two result modes - either write results per transition or for each defined state
		if (this->productYieldsOnly)
		{
			// Loop through all defind transitions
			for (auto j = _transitions.cbegin(); j != _transitions.cend(); j++)
			{
				arma::vec yields = _yields[(*j)->Name()];
				this->Data() << (*j)->Rate() * (arma::trapz(X, yields).max()) << " ";
			}
		}
		else
		{
			// Loop through all states
			for (auto j = _states.cbegin(); j != _states.cend(); j++)
			{
				arma::vec yields = _yields[(*j)->Name()];
				this->Data() << (arma::trapz(X, yields).max()) << " ";
			}
		}

		// Terminate the line in the data file
		this->Data() << std::endl;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskDynamicHSTimeEvo::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		if (!this->modeQuantumYield)
			_stream << "Time(ns) ";
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

	// Task validation method
	bool TaskDynamicHSTimeEvo::Validate()
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

		this->Properties()->Get("calculateyields", this->modeQuantumYield);

		if (this->Properties()->Get("transitionyields", this->productYieldsOnly) && !this->modeQuantumYield)
		{
			this->Log() << "Quantum yields based on transition objects are not supported when not in quantum yields mode.\n";
			this->Log() << "To enable quantum yields mode for task \"" << this->Name() << "\" set CalculateYields to true." << std::endl;
			this->productYieldsOnly = false;
		}

		// Get the output stride (i.e. write output for any n'th step, where n is the stride) - Only available for TimeEvolution calculations!
		if (this->Properties()->Get("outputstride", this->outputstride))
		{
			if (this->modeQuantumYield)
			{
				this->Log() << "Cannot use outputstride in quantum yield mode!" << std::endl;
				this->outputstride = 1;
			}
			else if (this->outputstride == 0)
			{
				this->Log() << "Cannot use outputstride 0, must be a non-zero positive integer! Using default value of 1." << std::endl;
				this->outputstride = 1;
			}
		}

		return true;
	}
	// -----------------------------------------------------
}
