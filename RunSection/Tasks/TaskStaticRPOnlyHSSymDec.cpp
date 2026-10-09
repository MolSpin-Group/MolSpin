/////////////////////////////////////////////////////////////////////////
// TaskStaticRPOnlyHSSymDec implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "../General/HS/HSRelaxationValidation.h"
#include <iostream>
#include "TaskStaticRPOnlyHSSymDec.h"
#include "Transition.h"
#include "SpinSpace.h"
#include "ObjectParser.h"
#include "Settings.h"
#include "Spin.h"
#include "SpinSystem.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticRPOnlyHSSymDec Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticRPOnlyHSSymDec::TaskStaticRPOnlyHSSymDec(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection)
	{
	}

	TaskStaticRPOnlyHSSymDec::~TaskStaticRPOnlyHSSymDec()
	{
	}
	// -----------------------------------------------------
	// TaskStaticRPOnlyHSSymDec protected methods
	// -----------------------------------------------------
	bool TaskStaticRPOnlyHSSymDec::RunLocal()
	{
		std::string relaxationError;
		if (!General::HS::ValidateNoRelaxation(this->SpinSystems(),
			"TaskStaticRPOnlyHSSymDec propagation representation", relaxationError))
		{ this->Log() << "ERROR: " << relaxationError << std::endl; return false; }

		this->Log() << "Running method StaticRP-Symmetric/uncoupled.\n"
					<< std::endl;

		// Tolerance is used to skip some term that are zero
		double tolerance = 1e-10;
		if (this->Properties()->Get("tolerance", tolerance))
		{
			this->Log() << "A tolerance was specified: " << tolerance << std::endl;
		}
		else
		{
			this->Log() << "No tolerance was specified, using default value instead: " << tolerance << std::endl;
		}

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

			// We only use k^2 later
			k = k * k;

			// Get a list of subspaces, make sure that we have a pair of uncoupled radicals
			auto subspaces = SpinAPI::CompleteSubspaces(*(*i));
			if (subspaces.size() < 2)
			{
				this->Log() << "Failed to obtain radicals. The spin system does not have two uncoupled subspaces! Skipping SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				continue;
			}

			// Objects to hold the two unpaired electrons and their subspaces
			SpinAPI::spin_ptr radical[2] = {nullptr, nullptr};
			std::vector<SpinAPI::spin_ptr> subspace1;
			std::vector<SpinAPI::spin_ptr> subspace2;

			// Find two subspaces with an electron
			for (auto j = subspaces.cbegin(); j != subspaces.cend(); j++)
			{
				for (auto k = j->cbegin(); k != j->cend(); k++)
				{
					if ((*k)->Type() == SpinAPI::SpinType::Electron)
					{
						if (radical[0] == nullptr)
						{
							radical[0] = (*k);
							subspace1 = (*j);
						}
						else if (radical[1] == nullptr)
						{
							radical[1] = (*k);
							subspace2 = (*j);
						}

						break;
					}
				}
			}

			// Check whether we found a radical pair
			if (radical[0] == nullptr || radical[1] == nullptr)
			{
				this->Log() << "Failed to obtain radicals. Did not find two uncoupled electronic spins! Skipping SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				continue;
			}

			// Obtain SpinSpaces to describe the two subsystems
			SpinAPI::SpinSpace spaces[2] = {SpinAPI::SpinSpace(subspace1), SpinAPI::SpinSpace(subspace2)};
			spaces[0].UseSuperoperatorSpace(false);
			spaces[1].UseSuperoperatorSpace(false);
			spaces[0].Add((*i)->Interactions());
			spaces[1].Add((*i)->Interactions());

			// Provide information about the radicals
			this->Log() << "---------------------------------------" << std::endl;
			this->Log() << "Found radical 1 with " << subspace1.size() << " spins:" << std::endl;
			this->Log() << " - Unpaired Electron: " << radical[0]->Name() << std::endl;
			this->Log() << " - Subspace dimensions: " << spaces[0].SpaceDimensions() << std::endl;

			if (subspace1.size() > 1)
			{
				this->Log() << " - Other spins:" << std::endl;
				for (auto j = subspace1.cbegin(); j != subspace1.cend(); j++)
					if ((*j) != radical[0])
						this->Log() << "   - " << (*j)->Name() << std::endl;
			}
			else
			{
				this->Log() << " - There are no other spins." << std::endl;
			}

			this->Log() << "\nFound radical 2 with " << subspace2.size() << " spins:" << std::endl;
			this->Log() << " - Unpaired Electron: " << radical[1]->Name() << std::endl;
			this->Log() << " - Subspace dimensions: " << spaces[1].SpaceDimensions() << std::endl;

			if (subspace2.size() > 1)
			{
				this->Log() << " - Other spins:" << std::endl;
				for (auto j = subspace2.cbegin(); j != subspace2.cend(); j++)
					if ((*j) != radical[1])
						this->Log() << "   - " << (*j)->Name() << std::endl;
			}
			else
			{
				this->Log() << " - There are no other spins." << std::endl;
			}

			this->Log() << "\nSpins not included in radicals: " << ((*i)->spins_size() - subspace1.size() - subspace2.size()) << " / " << (*i)->spins_size() << std::endl;
			this->Log() << "---------------------------------------" << std::endl;
			this->Log() << "Spins that are not included in the radicals are not considered in the calculations." << std::endl;

			// Matrices to hold spin operators, eigenvectors and eigenvalue-differences
			arma::cx_mat S[6];
			arma::mat L[2];
			double Z = 1;

			// Fill the matrices (get Hamiltonian, diagonalize it, get spin operators...)
			for (unsigned int r = 0; r < 2; r++)
			{
				// Get the Hamiltonian of the first radical subspace
				arma::cx_mat H;
				if (!spaces[r].Hamiltonian(H))
				{
					this->Log() << "Failed to obtain Hamiltonian for radical " << r << "." << std::endl;
					continue;
				}

				// Diagonalize Hamiltonian
				arma::vec lambda; // To hold eigenvalues
				arma::cx_mat V;	  // To hold eigenvectors
				if (!arma::eig_sym(lambda, V, H))
				{
					this->Log() << "Failed to diagonalize Hamiltonian for radical " << r << "." << std::endl;
					continue;
				}

				// Get electronic spin operators
				arma::sp_cx_mat J;
				if (!spaces[r].CreateOperator(radical[r]->Sx(), radical[r], J))
				{
					this->Log() << "Failed to obtain spin operators for radical " << r << "." << std::endl;
					continue;
				}
				S[r * 3 + 0] = V.t() * J * V;
				if (!spaces[r].CreateOperator(radical[r]->Sy(), radical[r], J))
				{
					this->Log() << "Failed to obtain spin operators for radical " << r << "." << std::endl;
					continue;
				}
				S[r * 3 + 1] = V.t() * J * V;
				if (!spaces[r].CreateOperator(radical[r]->Sz(), radical[r], J))
				{
					this->Log() << "Failed to obtain spin operators for radical " << r << "." << std::endl;
					continue;
				}
				S[r * 3 + 2] = V.t() * J * V;

				// Get eigenvalue difference matrix for radical "r"
				arma::mat O = arma::ones<arma::mat>(V.n_rows, V.n_cols);
				arma::mat D = arma::diagmat(lambda);
				L[r] = O * D - D * O;

				// Normalization factor contribution from radical "r"
				Z *= static_cast<double>(spaces[r].SpaceDimensions() / radical[r]->Multiplicity());
			}

			double spincorr = 0.0;

			// Do the actual calculation
			// NOTE: Cannot readily get a speedup from OpenMP due to missing locality of used data
			for (unsigned int m = 0; m < S[0].n_rows; m++)
			{
				for (unsigned int n = 0; n < S[0].n_rows; n++)
				{
					for (unsigned int l = 0; l < S[3].n_rows; l++)
					{
						for (unsigned int h = 0; h < S[3].n_rows; h++)
						{
							double Ldiff = (L[0])(m, n) + (L[1])(l, h);
							double f = k / (k + Ldiff * Ldiff);

							// If the contribution is very small, don't bother with the next calculations
							if (std::abs(f) < tolerance)
							{
								continue;
							}

							arma::cx_double tmp = (((S[0])(m, n)) * ((S[3 + 0])(l, h)) + ((S[1])(m, n)) * ((S[3 + 1])(l, h)) + ((S[2])(m, n)) * ((S[3 + 2])(l, h)));
							spincorr += std::real(tmp * std::conj(tmp)) * f;
						}
					}
				}
			}

			// Define the singlet and triplet yields
			double ps = spincorr / Z + 0.25;
			double pt = 0.75 - spincorr / Z;

			this->Log() << "\n"
						<< std::endl;
			this->Log() << "---------------------------------------" << std::endl;
			this->Log() << "Singlet yield: " << ps << ", Triplet yield: " << pt << std::endl;

			// Write data output
			this->Data() << ps << " " << pt << " ";

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		// Terminate the line in the data file after iteration through all spin systems
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticRPOnlyHSSymDec::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			_stream << (*i)->Name() << ".singlet " << (*i)->Name() << ".triplet ";
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticRPOnlyHSSymDec::Validate()
	{
		return true;
	}
	// -----------------------------------------------------
}
