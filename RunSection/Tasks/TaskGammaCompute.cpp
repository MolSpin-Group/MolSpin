/////////////////////////////////////////////////////////////////////////
// TaskGammaCompute implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskGammaCompute.h"
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
	TaskGammaCompute::TaskGammaCompute(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), steps(100), totaltime(1.0e+4)
	{
	}

	TaskGammaCompute::~TaskGammaCompute()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskGammaCompute::RunLocal()
	{
		this->Log() << "Running Gamma-COMPUTE implementation." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Check if a rate constant was given
		double k = 1e-3;
		bool hasRate = false;
		if (this->Properties()->Get("decayrate", k) || this->Properties()->Get("reactionrate", k) || this->Properties()->Get("rateconstant", k))
		{
			this->Log() << "A general rate constant was specified: " << k << "\nThis rate constant will be used for all SpinSystems." << std::endl;
			hasRate = true;
		}

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// -----------------------------------------------------
			// If no rate was specified, take the rate of the first
			// transition in the SpinSystem, if any
			// -----------------------------------------------------
			if (!hasRate)
			{
				k = 1e-3;
				auto transitions = (*i)->Transitions();
				if (transitions.size() > 0)
				{
					// Use the value of the first transition object found
					k = transitions[0]->Rate();
					this->Log() << "Using rate constant from first transition object (\"" << transitions[0]->Name() << "\") in the SpinSystem \"" << (*i)->Name() << "\"!\nFound rate constant value: " << k << std::endl;
				}
				else
				{
					// Use the default value if no value could be found
					this->Log() << "No rate constant was found for SpinSystem \"" << (*i)->Name() << "\"!\nUsing default rate constant value: " << k << std::endl;
				}
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
				this->Log() << "Unable to perform calculation: No period found." << std::endl;
				continue;
			}
			this->Log() << "Using period " << period << " ns." << std::endl;
			this->totaltime = period;
			// -----------------------------------------------------

			// Make sure we have an initial state
			auto initial_states = (*i)->InitialState();
			if (initial_states.size() < 1)
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
				continue;
			}

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);

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

			// Get list of states to calculate the quantum yield for
			auto states = (*i)->States();
			std::map<SpinAPI::state_ptr, arma::cx_mat> stateProjections;
			{
				arma::cx_mat PTracked;
				for (auto j = states.cbegin(); j != states.cend(); j++)
				{
					if (!space.GetState(*j, PTracked))
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\"." << std::endl;
					else
						stateProjections[*j] = PTracked;
				}
				this->Log() << "Found " << stateProjections.size() << " quantum states to calculate quantum yields for." << std::endl;
			}

			// -----------------------------------------------------
			// Get the propagators for upto a period
			// -----------------------------------------------------
			// Get the number of steps
			double timestep = this->totaltime / static_cast<double>(steps);
			this->Log() << "Using a timestep of " << timestep << " ns for " << steps << " steps during the period." << std::endl;

			// Prepare matrices
			arma::cx_mat H;																					 // Hamiltonian
			std::shared_ptr<arma::cx_mat> A(new arma::cx_mat[steps], std::default_delete<arma::cx_mat[]>()); // Collection of propagators

			// First propagator is the identity
			A.get()[0] = arma::eye<arma::cx_mat>(size(rho0));

			// Loop over a period
			for (unsigned int j = 1; j < steps; j++)
			{
				// Set the time to halfway into the next discretization step
				space.SetTime(timestep * static_cast<double>(j) - timestep / 2.0);

				// Get the Hamiltonian
				if (!space.Hamiltonian(H))
				{
					this->Log() << "ERROR: Failed to obtain the Hamiltonian! Stopping." << std::endl;
					return false;
				}

				// Extend the propagator
				A.get()[j] = arma::expmat(arma::cx_double(0.0, -1.0) * H * timestep) * A.get()[j - 1];
			}

			// -----------------------------------------------------
			// Diagonalize the final propagator
			// -----------------------------------------------------
			// Diagonalize Hamiltonian
			arma::cx_vec lambda; // To hold eigenvalues
			arma::cx_mat X;		 // To hold eigenvectors
			if (!arma::eig_gen(lambda, X, A.get()[steps - 1]))
			{
				this->Log() << "Failed to diagonalize Hamiltonian." << std::endl;
				continue;
			}

			// Get eigenvalues of the average Hamiltonian (lambda is for exp(-i H dT) where H is the avg Hamiltonian)
			// Note that a phase factor will be missing here!
			arma::cx_vec preOmega = arma::cx_double(0.0, 1.0 / period) * lambda.for_each([](arma::cx_vec::elem_type &val)
																						 { val = std::log(val); });
			if (arma::norm(arma::imag(preOmega)) > 1e-10)
			{
				this->Log() << "Warning: Average Hamiltonian eigenvalue vector has an imaginary part with norm " << arma::norm(arma::imag(preOmega)) << "! Hamiltonian eigenvalues should be real." << std::endl;
			}

			// Check condition on number of steps
			{
				unsigned int minsteps = static_cast<unsigned int>(arma::max(arma::abs(preOmega)) * period / arma::datum::pi);
				if (steps <= minsteps)
				{
					this->Log() << "Warning: Number of steps (" << steps << ") may be too small! Use at least " << minsteps << "." << std::endl;
				}
			}

			// Get matrix with differences between eigenvalues
			arma::mat O = arma::ones<arma::mat>(X.n_rows, X.n_cols);
			arma::mat D = arma::diagmat(arma::real(preOmega));
			arma::mat L = O * D - D * O;

			// -----------------------------------------------------
			// Next steps:
			// - Form the quantity "g_rs(x)"
			// - Get "G_rs(x)" using FFT
			// - Calculate the quantum yield
			// -----------------------------------------------------
			// Prepare collection of g-matrices
			std::shared_ptr<arma::cx_mat> g(new arma::cx_mat[steps], std::default_delete<arma::cx_mat[]>());
			std::shared_ptr<arma::cx_mat> G(new arma::cx_mat[steps], std::default_delete<arma::cx_mat[]>());

			// Write standard output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->WriteStandardOutput(this->Data());

			// For the last few steps, loop over all wanted output states
			for (auto s = states.cbegin(); s != states.cend(); s++)
			{

				// Use element-wise exponential
				for (unsigned int j = 0; j < steps; j++)
				{
					g.get()[j] = X.t() * A.get()[j].t() * stateProjections[*s] * A.get()[j] * X;
					g.get()[j] %= arma::exp(arma::cx_double(0.0, -1.0) * L * timestep * static_cast<double>(j));
				}

				// Compute FFT for each j-series of 'g'
				for (unsigned int x = 0; x < L.n_rows; x++)
				{
					for (unsigned int y = 0; y < L.n_cols; y++)
					{
						// Prepare a vector
						arma::cx_vec o(steps);

						// Fill it with a j-series
						for (unsigned int j = 0; j < steps; j++)
							o(j) = g.get()[j](x, y);

						// Calculate the FFT
						arma::cx_vec offt = arma::fft(o);

						// Put it back into 'g'
						for (unsigned int j = 0; j < steps; j++)
							g.get()[j](x, y) = offt(j);
					}
				}

				// Obtain |G|^2 from the Fourier-transformed 'g'
				for (unsigned int j = 0; j < steps; j++)
				{
					// Reset G
					G.get()[j] = arma::zeros<arma::cx_mat>(size(L));

					// Get the correlation function
					for (unsigned int p = 0; p < steps; p++)
					{
						// Use "x mod n" with x = j+p
						unsigned int q = (j + p) % steps;
						G.get()[j] += g.get()[p].t() * g.get()[q];
					}
				}

				// Get the quantum yield
				arma::mat T = arma::zeros<arma::mat>(size(L));
				arma::mat F;
				double w;
				for (unsigned int j = 0; j < steps; j++)
				{
					w = static_cast<double>(j) * 2.0 * arma::datum::pi / period;

					F = L + arma::ones<arma::mat>(size(L)) * w;
					F.transform([k](double val)
								{ return 1.0 / (1.0 + val * val / (k * k)); });
					T += arma::abs(G.get()[j]) % F;
				}

				double quantumYield = std::sqrt(2.0 * arma::datum::pi) / std::abs(arma::trace(stateProjections[*s]) * static_cast<double>(steps * steps)) * arma::accu(T);

				this->Log() << "Calculated quantum yield of " << quantumYield << " for state " << (*s)->Name() << "." << std::endl;
				this->Data() << quantumYield << " ";
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		// Terminate the line
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskGammaCompute::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
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

	// Task validation method
	bool TaskGammaCompute::Validate()
	{
		unsigned int inputSteps = 0;
		double inputTotaltime = 0.0;

		// Get number of steps
		if (this->Properties()->Get("steps", inputSteps))
		{
			if (inputSteps > 0)
			{
				this->steps = inputSteps;
			}
			else
			{
				return false;
			}
		}

		// Get totaltime/period
		if (this->Properties()->Get("totaltime", inputTotaltime) || this->Properties()->Get("period", inputTotaltime))
		{
			if (std::isfinite(inputTotaltime) && inputTotaltime > 0.0)
			{
				this->totaltime = inputTotaltime;
			}
			else
			{
				// We cannot run the calculation if an invalid total time was specified
				return false;
			}
		}

		return true;
	}
	// -----------------------------------------------------
}
