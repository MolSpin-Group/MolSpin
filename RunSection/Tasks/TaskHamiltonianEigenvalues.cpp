/////////////////////////////////////////////////////////////////////////
// TaskHamiltonianEigenvalues implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskHamiltonianEigenvalues.h"
#include "State.h"
#include "ObjectParser.h"
#include "Settings.h"
#include "SpinSystem.h"
#include "Spin.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskHamiltonianEigenvalues Constructors and Destructor
	// -----------------------------------------------------
	TaskHamiltonianEigenvalues::TaskHamiltonianEigenvalues(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), printEigenvectors(false),
																																	printHamiltonian(false), useSuperspace(false), separateRealImag(false),
																																	initialTime(0.0), totalTime(0.0), timestep(1),
																																	resonanceFrequencies(false), referenceStates(), transitionSpins()
	{
	}

	TaskHamiltonianEigenvalues::~TaskHamiltonianEigenvalues()
	{
	}
	// -----------------------------------------------------
	// TaskStaticHSSymmetricDecay protected methods
	// -----------------------------------------------------
	bool TaskHamiltonianEigenvalues::RunLocal()
	{
		this->Log() << "Running method Hamiltonian-Eigenvalues." << std::endl;

		// If this is the first step, write header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// The calculations can be repeated at different times, but will be done at least once
		bool isFirstTime = true;
		for (double time = this->initialTime; (time <= this->totalTime || isFirstTime); time += this->timestep)
		{
			isFirstTime = false;

			this->Log() << "---------- Setting time to " << time << " / " << this->totalTime << " ----------" << std::endl;

			// -----------------------------------------------------
			// General output
			// -----------------------------------------------------
			this->Data() << this->RunSettings()->CurrentStep() << " " << time << " ";
			this->WriteStandardOutput(this->Data());

			// -----------------------------------------------------
			// Loop through all SpinSystems
			// -----------------------------------------------------
			auto systems = this->SpinSystems();
			for (auto i = systems.cbegin(); i != systems.cend(); i++)
			{
				this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

				// -----------------------------------------------------
				// Get SpinSpace, obtain Hamiltonian, and diagonalize it
				// -----------------------------------------------------
				// Obtain a SpinSpace to describe the system
				SpinAPI::SpinSpace space(*(*i));
				space.UseSuperoperatorSpace(this->useSuperspace);
				space.SetTime(time);
				arma::cx_mat rho0;

				// Get the Hamiltonian
				arma::cx_mat H;
				if (!space.Hamiltonian(H))
				{
					this->Log() << "Failed to obtain Hamiltonian." << std::endl;
					continue;
				}

				// Diagonalization
				arma::cx_mat V;	  // To hold eigenvectors
				arma::vec lambda; // To hold eigenvalues
				this->Log() << "Starting diagonalization..." << std::endl;
				arma::eig_sym(lambda, V, H);
				this->Log() << "Diagonalization done! Eigenvalues: " << lambda.n_elem << ", eigenvectors: " << V.n_cols << std::endl;

				// -----------------------------------------------------
				// Write the main results to the data file
				// -----------------------------------------------------
				// Write the eigenvalues to the data file
				for (auto j = lambda.cbegin(); j != lambda.cend(); j++)
					this->Data() << (*j) << " ";

				// Get the reference state
				for (auto j = this->referenceStates.cbegin(); j != this->referenceStates.cend(); j++)
				{
					// Check whether the current state is relevant for the current spin system
					if (!(*i)->Contains(*j))
					{
						this->Log() << "Skipping state " << (*j)->Name() << " as it does not belong to current spin system." << std::endl;
						continue;
					}

					// Get a projection operator onto the state
					arma::cx_mat R;
					if (!space.GetState((*j), R))
					{
						this->Log() << "Failed to obtain projection matrix onto the reference state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					// Make sure the dimensions fit
					if (R.n_rows != V.n_rows || R.n_cols != V.n_cols)
					{
						this->Log() << "Warning: Problem with the reference state " << (*j)->Name() << ". Reference state ignored." << std::endl;
						continue;
					}

					// Get vector to hold the projections
					arma::vec VRVProj = arma::real(arma::diagvec(V.t() * R * V));

					// Print the results
					for (auto refproj = VRVProj.cbegin(); refproj != VRVProj.cend(); refproj++)
						this->Data() << (*refproj) << " ";
				}
				// -----------------------------------------------------
				// END of main results
				// -----------------------------------------------------

				// -----------------------------------------------------
				// Print Hamiltonian and eigenvectors to log stream
				// -----------------------------------------------------
				// Print the eigenvectors if requested
				if (this->printEigenvectors)
				{
					this->Log() << "\n------ Printing Eigenvectors (one per column) ------\n";
					if (this->separateRealImag)
					{
						this->Log() << "\n------ Real part\n";
						arma::real(V).print(this->Log());
						this->Log() << "\n------ Imaginary part\n";
						arma::imag(V).print(this->Log());
					}
					else
					{
						V.print(this->Log());
					}
					this->Log() << "\n------ End of Eigenvectors ------\n";
				}

				// Print the Hamiltonian if requested
				if (this->printHamiltonian)
				{
					this->Log() << "\n------ Printing Hamiltonian operator ------\n";
					if (this->separateRealImag)
					{
						this->Log() << "\n------ Real part\n";
						arma::real(H).print(this->Log());
						this->Log() << "\n------ Imaginary part\n";
						arma::imag(H).print(this->Log());
					}
					else
					{
						H.print(this->Log());
					}
					this->Log() << "\n------ End of Hamiltonian operator ------\n";
				}
				// -----------------------------------------------------
				// END of print Hamiltonian and eigenvectors to log stream
				// -----------------------------------------------------

				// -----------------------------------------------------
				// Other analyses to perform
				// -----------------------------------------------------
				if (this->resonanceFrequencies)
				{
					this->GetResonanceFrequencies(lambda, V, (*i), space);
				}
				// -----------------------------------------------------
				// END of other analyses to perform
				// -----------------------------------------------------

				this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
			}

			// Terminate the line in the data file after iteration through all spin systems
			this->Data() << std::endl;
		}
		// -----------------------------------------------------
		// END loop over different times
		// -----------------------------------------------------

		return true;
	}

	// -----------------------------------------------------
	// Resonance frequency analysis
	// -----------------------------------------------------
	// Analysis the eigenvalues to obtain the resonance frequencies of the spin system
	void TaskHamiltonianEigenvalues::GetResonanceFrequencies(const arma::vec &_e, const arma::cx_mat &_V, const std::shared_ptr<SpinAPI::SpinSystem> &_system, const SpinAPI::SpinSpace &_space)
	{
		// Write header
		this->Log() << "\n------ Resonance frequency analysis ------" << std::endl;

		// Get a list of spins for calculating the transition frequencies
		std::vector<std::pair<SpinAPI::spin_ptr, std::vector<arma::cx_mat>>> spins;
		for (const std::string &s : this->transitionSpins)
		{
			auto spin = _system->spins_find(s);
			if (spin != nullptr)
			{
				// Helper variables
				std::vector<arma::cx_mat> tmp;
				arma::cx_mat O;

				// Obtain the spin operator matrix representation in the full Hilbert space,
				// and transform by the eigenstates to obtain the transition matrix
				_space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, O);
				tmp.push_back(_V.t() * O * _V);

				_space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, O);
				tmp.push_back(_V.t() * O * _V);

				_space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, O);
				tmp.push_back(_V.t() * O * _V);

				spins.push_back(std::pair<SpinAPI::spin_ptr, std::vector<arma::cx_mat>>(spin, tmp));
			}
		}

		if (spins.size() > 0)
		{
			this->Log() << "\nTransition matrix elements will be calculated for " << spins.size() << " spins found in the SpinSystem " << _system->Name();
			this->Log() << ", and will be denoted Sx(spinname), Sy(spinname) and Sz(spinname).\n";
		}

		this->Log() << "\nThe table below describes transitions from eigenstate 'i' to eigenstate 'j'.";
		this->Log() << "\nThe angular frequency (\"omega\") is denoted 'w' and given in rad/ns,";
		this->Log() << "\nwhile the actual resonance frequency, 'f', is given in MHz.";
		this->Log() << "\n\ni j w f";

		for (const std::pair<SpinAPI::spin_ptr, std::vector<arma::cx_mat>> &s : spins)
		{
			if (this->separateRealImag)
				this->Log() << " Sx(" << s.first->Name() << ").real Sx(" << s.first->Name() << ").imaginary Sy(" << s.first->Name() << ").real Sy(" << s.first->Name() << ").imaginary Sz(" << s.first->Name() << ").real Sz(" << s.first->Name() << ").imaginary";
			else
				this->Log() << " Sx(" << s.first->Name() << ") Sy(" << s.first->Name() << ") Sz(" << s.first->Name() << ")";
		}

		this->Log() << "\n";

		for (unsigned int i = 0; i < _e.n_elem; i++)
		{
			for (unsigned int j = i + 1; j < _e.n_elem; j++)
			{
				this->Log() << i << " ";
				this->Log() << j << " ";
				this->Log() << std::abs(_e[j] - _e[i]) << " ";
				this->Log() << (std::abs(_e[j] - _e[i]) / (2.0 * arma::datum::pi) * 1000.0) << " ";

				// Transition matrix elements
				for (const std::pair<SpinAPI::spin_ptr, std::vector<arma::cx_mat>> &s : spins)
				{
					if (this->separateRealImag)
						this->Log() << std::real(s.second[0](i, j)) << " " << std::imag(s.second[0](i, j)) << " " << std::real(s.second[1](i, j)) << " " << std::imag(s.second[1](i, j)) << " " << std::real(s.second[2](i, j)) << " " << std::imag(s.second[2](i, j)) << " ";
					else
						this->Log() << s.second[0](i, j) << " " << s.second[1](i, j) << " " << s.second[2](i, j) << " ";
				}

				this->Log() << "\n";
			}
		}

		this->Log() << "\n------ END of resonance frequency analysis ------" << std::endl;
	}

	// -----------------------------------------------------
	// Writes the header of the data file (but can also be passed to other streams)
	// -----------------------------------------------------
	void TaskHamiltonianEigenvalues::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step time(ns) ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(this->useSuperspace);

			// Write the headers for the eigenvalues
			for (unsigned int j = 0; j < space.SpaceDimensions(); j++)
				_stream << (*i)->Name() << ".H.lambda" << j << " ";

			// Write headers for the reference states
			for (auto refstate = this->referenceStates.cbegin(); refstate != this->referenceStates.cend(); refstate++)
				for (unsigned int j = 0; j < space.SpaceDimensions(); j++)
					_stream << (*i)->Name() << ".H.ref" << j << "(" << (*refstate)->Name() << ")" << " ";
		}
		_stream << std::endl;
	}

	// -----------------------------------------------------
	// Validation method - gathers input parameters
	// -----------------------------------------------------
	bool TaskHamiltonianEigenvalues::Validate()
	{
		// Should we use superspace Hamiltonian instead of normal Hamiltonian?
		if (this->Properties()->Get("superspace", this->useSuperspace) || this->Properties()->Get("usesuperspace", this->useSuperspace))
		{
			this->Log(MessageType_Details) << "Task " << this->Name() << ": Using superspace Hamiltonian." << std::endl;
		}

		// Extra printing options
		if (this->Properties()->Get("eigenvectors", this->printEigenvectors) || this->Properties()->Get("printeigenvectors", this->printEigenvectors))
		{
			this->Log(MessageType_Details) << "Task " << this->Name() << ": Will print eigenvectors." << std::endl;
		}
		if (this->Properties()->Get("hamiltonian", this->printHamiltonian) || this->Properties()->Get("printhamiltonian", this->printHamiltonian))
		{
			this->Log(MessageType_Details) << "Task " << this->Name() << ": Will print Hamiltonian." << std::endl;
		}
		if (this->Properties()->Get("separatereal", this->separateRealImag) || this->Properties()->Get("separateimaginary", this->separateRealImag) || this->Properties()->Get("separatecomplex", this->separateRealImag))
		{
			this->Log(MessageType_Details) << "Task " << this->Name() << ": Separating real and imaginary parts of output." << std::endl;
		}

		// Enabling extra analyses
		if (this->Properties()->Get("resonancefrequencies", this->resonanceFrequencies) || this->Properties()->Get("frequencies", this->resonanceFrequencies) || this->Properties()->Get("resonances", this->resonanceFrequencies))
		{
			this->Log(MessageType_Details) << "Task " << this->Name() << ": Will perform resonance frequency analysis." << std::endl;
		}

		// Get list of spins to use for transition matrix element calculations
		if (this->resonanceFrequencies)
		{
			if (this->Properties()->GetList("spinlist", this->transitionSpins))
				this->Log() << "Found list of " << this->transitionSpins.size() << " spins to use for transition matrix element calculations." << std::endl;
			else
				this->Log() << "No spin list specified - transition matrix elements will not be calculated." << std::endl;
		}

		// Initial time for time evolution
		if (this->Properties()->Get("initialtime", this->initialTime) || this->Properties()->Get("starttime", this->initialTime) || this->Properties()->Get("begin", this->initialTime))
		{
			// Make sure the the initial time is valid
			if (this->initialTime < 0 || !std::isfinite(this->initialTime))
			{
				this->initialTime = 0.0;
				this->Log(MessageType_Critical | MessageType_Error) << "Task " << this->Name() << ": ERROR: Invalid initial time specified! Setting initial time to " << this->initialTime << "." << std::endl;
			}

			this->Log(MessageType_Details) << "Task " << this->Name() << ": Initial time set to " << this->initialTime << "." << std::endl;
		}

		// Total time for time evolution
		if (this->Properties()->Get("totaltime", this->totalTime) || this->Properties()->Get("stoptime", this->totalTime) || this->Properties()->Get("end", this->totalTime))
		{
			// Make sure the the total time is valid
			if (this->totalTime < this->initialTime || !std::isfinite(this->totalTime))
			{
				this->totalTime = this->initialTime;
				this->Log(MessageType_Critical | MessageType_Error) << "Task " << this->Name() << ": ERROR: Invalid total time specified! Setting total time to " << this->totalTime << "." << std::endl;
			}

			this->Log(MessageType_Details) << "Task " << this->Name() << ": Total time set to " << this->totalTime << "." << std::endl;
		}
		else
		{
			// Notify the user if no total time was specified while initial time was specified
			if (this->totalTime < this->initialTime)
			{
				this->Log(MessageType_Important | MessageType_Warning) << "Task " << this->Name() << ": WARNING: No total time was specified, but an initial time has been specified! If this is not intentional, note you can specify a total time using 'totaltime = <value>;'." << std::endl;
			}
		}

		// Total time for time evolution
		if (this->Properties()->Get("timestep", this->timestep))
		{
			// Make sure the the initial time is valid
			if (this->timestep <= 0 || !std::isfinite(this->timestep))
			{
				this->timestep = (this->totalTime - this->initialTime) / 10.0;
				this->Log(MessageType_Critical | MessageType_Error) << "Task " << this->Name() << ": ERROR: Invalid timestep specified! Setting timestep to " << this->timestep << "." << std::endl;
			}

			this->Log(MessageType_Details) << "Task " << this->Name() << ": Timestep set to " << this->timestep << "." << std::endl;
		}

		// Get a reference state (to get projections onto eigenstates), if one is specified
		std::vector<std::string> strs;
		if (this->Properties()->GetList("refstates", strs) || this->Properties()->GetList("referencestates", strs))
		{
			// Loop through the list of states that was specified
			auto systems = this->SpinSystems();
			for (const std::string &s : strs)
			{
				SpinAPI::state_ptr state = nullptr;

				// Loop through the spin systems until a state is found
				for (auto i = systems.cbegin(); i != systems.cend(); i++)
				{
					// Attemp to find the state in the spin system
					state = (*i)->states_find(s);
					if (state != nullptr)
					{
						// A state was found, no need to look in the other spin systems
						this->referenceStates.push_back(state);
						this->Log(MessageType_Important | MessageType_Warning) << "Task " << this->Name() << ": Found state " << s << " in spin system " << (*i)->Name() << "!" << std::endl;
						break;
					}
				}

				// Notify the user when a state was not found in any spin system
				if (state == nullptr)
				{
					this->Log(MessageType_Details) << "Task " << this->Name() << ": WARNING: Did not find state " << s << " in any spin system!" << std::endl;
				}
			}
		}

		return true;
	}
	// -----------------------------------------------------
}
