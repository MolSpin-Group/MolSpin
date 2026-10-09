/////////////////////////////////////////////////////////////////////////
// TaskStaticHSSymmetricDecay implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "../General/HS/HSRelaxationValidation.h"
#include <iostream>
#include "TaskStaticHSSymmetricDecay.h"
#include "Transition.h"
#include "State.h"
#include "SpinSpace.h"
#include "ObjectParser.h"
#include "Settings.h"
#include "SpinSystem.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticHSSymmetricDecay Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSSymmetricDecay::TaskStaticHSSymmetricDecay(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection)
	{
	}

	TaskStaticHSSymmetricDecay::~TaskStaticHSSymmetricDecay()
	{
	}
	// -----------------------------------------------------
	// TaskStaticHSSymmetricDecay protected methods
	// -----------------------------------------------------
	bool TaskStaticHSSymmetricDecay::RunLocal()
	{
		std::string relaxationError;
		if (!General::HS::ValidateNoRelaxation(this->SpinSystems(),
			"TaskStaticHSSymmetricDecay propagation representation", relaxationError))
		{ this->Log() << "ERROR: " << relaxationError << std::endl; return false; }

		this->Log() << "Running method StaticHS-SymmetricDecay." << std::endl;

		// Check if a rate constant was given
		double k = 1e-3;
		bool hasRate = false;
		if (this->Properties()->Get("decayrate", k) || this->Properties()->Get("reactionrate", k) || this->Properties()->Get("rateconstant", k))
		{
			this->Log() << "A general rate constant was specified: " << k << "\nThis rate constant will be used for all SpinSystems." << std::endl;
			hasRate = true;
		}

		// If this is the first step, write header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Write general output
		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->WriteStandardOutput(this->Data());

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

			// If no rate was specified, take the rate of the first transition in the SpinSystem, if any
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

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			arma::cx_mat rho0;
			space.UseSuperoperatorSpace(false);

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

			// Get the Hamiltonian
			arma::cx_mat H;
			if (!space.Hamiltonian(H))
			{
				this->Log() << "Failed to obtain Hamiltonian." << std::endl;
				continue;
			}

			// Prepare for calculation
			double ksq = k * k;
			arma::cx_mat V;	  // To hold eigenvectors
			arma::vec lambda; // To hold eigenvalues
			this->Log() << "Starting diagonalization..." << std::endl;
			arma::eig_sym(lambda, V, H);
			this->Log() << "Diagonalization done! Eigenvalues: " << lambda.n_elem << ", eigenvectors: " << V.n_cols << std::endl;
			arma::mat O = arma::ones<arma::mat>(V.n_rows, V.n_cols);
			arma::mat D = arma::diagmat(lambda);
			arma::mat L = O * D - D * O;
			L.transform([ksq](double val)
						{ return 1.0 / (1.0 + val * val / ksq); });
			this->Log() << "Done preparing f(w) = k^2 / (k^2 + w^2)." << std::endl;
			arma::cx_mat QS;
			arma::cx_mat QStmp = rho0 * V;

			// Loop through all the State objects defined on the SpinSystem
			arma::cx_mat P;
			double result;
			auto states = (*i)->States();
			for (auto j = states.cbegin(); j != states.cend(); j++)
			{
				// Obtain a projection onto the state
				if (!space.GetState((*j), P))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
					continue;
				}

				// Calculate the yield for the state
				result = 0.0;
				for (unsigned int h = 0; h < V.n_rows; h++)
				{
					QS = P * V.col(h) * QStmp.col(h).t();
					for (unsigned int g = 0; g < V.n_cols; g++)
						result += std::real(cdot(V.col(g), QS * V.col(g)) * L(h, g));
				}

				// Output the result
				this->Data() << result << " ";
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		// Terminate the line in the data file after iteration through all spin systems
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticHSSymmetricDecay::WriteHeader(std::ostream &_stream)
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

	// Validation
	bool TaskStaticHSSymmetricDecay::Validate()
	{
		return true;
	}
	// -----------------------------------------------------
}
