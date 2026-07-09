/////////////////////////////////////////////////////////////////////////
// TaskPeriodicHSTimeEvo implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskPeriodicHSTimeEvo.h"
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
	TaskPeriodicHSTimeEvo::TaskPeriodicHSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(0.01), totaltime(1.0e+4), stepsPerPeriod(50), outputstride(1), modeQuantumYield(false), productYieldsOnly(false), timedependentInteractions(false)
	{
	}

	TaskPeriodicHSTimeEvo::~TaskPeriodicHSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskPeriodicHSTimeEvo::RunLocal()
	{
		this->Log() << "Running method PeriodicHSTimeEvolution." << std::endl;

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

			// -----------------------------------------------------
			// Find the period to use
			// -----------------------------------------------------
			double period = this->GetPeriod(*i);
			double propagator_stepsize = period / static_cast<double>(this->stepsPerPeriod);
			if (propagator_stepsize < this->timestep)
			{
				this->Log() << "WARNING: Propagator stepsize is smaller than integration timestep!" << std::endl;
			}
			this->Log() << "Using period " << period << " ns and a propagator timestep of " << propagator_stepsize << " ns." << std::endl;
			// -----------------------------------------------------

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);
			space.SetTime(0.0);

			// Check whether any interactions are time-dependent
			this->timedependentInteractions = space.HasTimedependentInteractions();

			// Get the static part of the Hamiltonian
			arma::cx_mat H; // Static part of the Hamiltonian
			if (!space.StaticHamiltonian(H))
			{
				this->Log() << "Failed to obtain the Hamiltonian! Skipping system." << std::endl;
				continue;
			}

			// Get the total reaction operator
			arma::cx_mat K;
			if (!space.TotalReactionOperator(K))
			{
				this->Log() << "Warning: Failed to obtain matrix representation of Transitions!" << std::endl;
			}

			// Get the dynamic part of the Hamiltonian (NOTE: This changes the time of the spinspace)
			std::vector<arma::sp_cx_mat> dH;
			for (unsigned int n = 0; n < this->stepsPerPeriod; n++)
			{
				// If there are no time-dependent interactions, we still need to fill up the list
				if (!this->timedependentInteractions)
				{
					dH.push_back(arma::sp_cx_mat(space.SpaceDimensions(), space.SpaceDimensions()));
					continue;
				}

				// Get the dynamic part
				space.SetTime(static_cast<double>(n) * propagator_stepsize);
				arma::sp_cx_mat tmpH;
				if (!space.DynamicHamiltonian(tmpH))
				{
					this->Log() << "Failed to obtain the Hamiltonian! Skipping system." << std::endl;
					return false;
				}
				dH.push_back(tmpH);
			}
			this->Log() << "Obtained " << dH.size() << " dynamic Hamiltonians for spin system " << (*i)->Name() << "." << std::endl;

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
				// Get the dynamic Hamiltonian index to use
				unsigned int dHIndex = static_cast<unsigned int>(static_cast<double>(n) * this->timestep / propagator_stepsize) % dH.size();

				// Advance a timestep
				AdvanceStep_AsyncLeapfrog(H + dH[dHIndex], K, rho0);

				// Should we calculate the quantum yield rather than just the time evolution?
				if (this->modeQuantumYield)
				{
					GetQuantumYields(space, rho0, n, states, transitions, yields);
				}
				else
				{
					// If not, just output the time evolution
					if (n % this->outputstride == 0)
					{
						// Set the time (in order to write correct standard output)
						space.SetTime(static_cast<double>(n) * this->timestep);

						OutputTimeEvolution(space, rho0, n, states);
					}
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

				// Set the time to 0 (will be used for the standard output)
				space.SetTime(0.0);

				OutputQuantumYields(space, yields, steps, states, transitions);
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		return true;
	}

	// -----------------------------------------------------
	// Timestep method
	// -----------------------------------------------------
	void TaskPeriodicHSTimeEvo::AdvanceStep_AsyncLeapfrog(const arma::cx_mat &_H, const arma::cx_mat &_K, arma::cx_mat &_rho)
	{
		arma::cx_mat dRho = arma::cx_double(0.0, -1.0) * (_H * _rho - _rho * _H) - (_K * _rho + _rho * _K);
		_rho += dRho * this->timestep / 2.0;
		dRho = (arma::cx_double(0.0, -1.0) * (_H * _rho - _rho * _H) - (_K * _rho + _rho * _K));
		_rho += dRho * this->timestep / 2.0;
	}

	// -----------------------------------------------------
	// Writes the output for a timestep
	// -----------------------------------------------------
	void TaskPeriodicHSTimeEvo::OutputTimeEvolution(const SpinAPI::SpinSpace &_space, const arma::cx_mat &_rho, const unsigned int _step, const std::vector<SpinAPI::state_ptr> &_states)
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

	// -----------------------------------------------------
	// Get the state projections
	// -----------------------------------------------------
	void TaskPeriodicHSTimeEvo::GetQuantumYields(const SpinAPI::SpinSpace &_space, const arma::cx_mat &_rho, const unsigned int _step, const std::vector<SpinAPI::state_ptr> &_states, const std::vector<SpinAPI::transition_ptr> &_transitions, std::map<std::string, arma::vec> &_yields)
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

	// -----------------------------------------------------
	// Output the quantum yield
	// -----------------------------------------------------
	void TaskPeriodicHSTimeEvo::OutputQuantumYields(const SpinAPI::SpinSpace &_space, std::map<std::string, arma::vec> &_yields, const unsigned int _steps, const std::vector<SpinAPI::state_ptr> &_states, const std::vector<SpinAPI::transition_ptr> &_transitions)
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
				// Make sure that there is a state object
				if ((*j)->SourceState() == nullptr)
					continue;

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

	// -----------------------------------------------------
	// Writes the header of the data file (but can also be passed to other streams)
	// -----------------------------------------------------
	void TaskPeriodicHSTimeEvo::WriteHeader(std::ostream &_stream)
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

	// -----------------------------------------------------
	// Gets the period of periodic interactions
	// -----------------------------------------------------
	double TaskPeriodicHSTimeEvo::GetPeriod(const SpinAPI::system_ptr &_sys)
	{
		double period = 0.0;

		// Let the period be specified rather than inferred if needed
		if (this->Properties()->Get("forceperiod", period) || this->Properties()->Get("period", period))
			return period;

		for (auto j = _sys->interactions_cbegin(); j != _sys->interactions_cend(); j++)
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

		return period;
	}

	// -----------------------------------------------------
	// Task validation method
	// -----------------------------------------------------
	bool TaskPeriodicHSTimeEvo::Validate()
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

		return true;
	}
	// -----------------------------------------------------
}
