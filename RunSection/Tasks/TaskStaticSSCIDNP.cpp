/////////////////////////////////////////////////////////////////////////
// TaskStaticSSCIDNP implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <numeric>
#include "TaskStaticSSCIDNP.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Operator.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSSCIDNP Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSCIDNP::TaskStaticSSCIDNP(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn),
																												  productYieldsOnly(false)
	{
	}

	TaskStaticSSCIDNP::~TaskStaticSSCIDNP()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSSCIDNP protected methods
	// -----------------------------------------------------
	bool TaskStaticSSCIDNP::RunLocal()
	{
		this->Log() << "Running method StaticSSCIDNP." << std::endl;

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
			arma::cx_mat rho0;
			space.UseSuperoperatorSpace(true);
			space.SetReactionOperatorType(this->reactionOperators);

			std::vector<double> weights;
			weights = (*i)->Weights();

			// Normalize the weights
			double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
			if (sum_weights > 0)
			{
				for (double &weight : weights)
				{
					weight /= sum_weights;
				}
			}

			if (weights.size() > 1)
			{
				this->Log() << "Using weighted density matrix for initial state. Be sure that the sum of weights equals to 1." << std::endl;
				// Get the initial state
				int counter = 0;
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;
					if (!space.GetState(*j, tmp_rho0))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					if (j == initial_states.cbegin())
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[0] << "\"." << std::endl;
						rho0 = weights[0] * tmp_rho0;
						counter += 1;
					}
					else
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[counter] << "\"." << std::endl;
						rho0 += weights[counter] * tmp_rho0;
						counter += 1;
					}
				}
			}
			else
			{
				// Get the initial state
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					(*j)->Name();
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
			}

			rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

			// Convert initial state to superoperator space
			arma::cx_vec rho0vec;
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

			// Get the Hamiltonian
			arma::sp_cx_mat H;
			if (!space.Hamiltonian(H))
			{
				this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
				continue;
			}

			// Get a matrix to collect all the terms (the total Liouvillian)
			arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * H;

			// Get the reaction operators, and add them to "A"
			arma::sp_cx_mat K;
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
					this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to the Liouvillian.\n";
				}
			}

			// Perform the calculation
			// Here it could be a problem of the right sign
			this->Log() << "Ready to perform calculation." << std::endl;
			arma::cx_vec result = -solve(arma::conv_to<arma::cx_mat>::from(A), rho0vec);
			this->Log() << "Done with calculation." << std::endl;

			// Convert the resulting density operator back to its Hilbert space representation
			if (!space.OperatorFromSuperspace(result, rho0))
			{
				this->Log() << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
				continue;
			}

			// Get nuclei of interest for CIDNP spectrum

			arma::cx_mat Iprojx;
			arma::cx_mat Iprojy;
			arma::cx_mat Iprojz;

			std::vector<std::string> nuclei_list;
			bool Dnp = false;
			int m;

			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->WriteStandardOutput(this->Data());

			if (this->Properties()->GetList("nuclei_list", nuclei_list, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					for (m = 0; m < (int)nuclei_list.size(); m++)
					{
						std::string spintype;
						(*l)->Properties()->Get("type", spintype);

						if (spintype != "electron")
						{
							std::cout << spintype << std::endl;
						}

						if ((*l)->Name() == nuclei_list[m])
						{
							std::cout << (*l)->Name() << std::endl;
							if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
							{
								return false;
							}

							if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
							{
								return false;
							}

							if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
							{
								return false;
							}

							arma::cx_mat P;

							// There are two result modes - either write results per transition or for each defined state
							if (this->productYieldsOnly && Dnp == false)
							{
								std::cout << "Perfoming CIDSP calculation." << std::endl;
								// Loop through all defind transitions
								auto transitions = (*i)->Transitions();
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									// Make sure that there is a state object
									if ((*j)->SourceState() == nullptr)
										continue;

									if (!space.GetState((*j)->SourceState(), P))
									{
										this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
										continue;
									}

									std::cout << "Ix:" << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << std::endl;
									std::cout << "Iy:" << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << std::endl;
									std::cout << "Iz:" << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << std::endl;

									// Return the yield for this transition
									this->Data() << std::real(arma::trace(Iprojx * (*j)->Rate() * P * rho0)) << " ";
									this->Data() << std::real(arma::trace(Iprojy * (*j)->Rate() * P * rho0)) << " ";
									this->Data() << std::real(arma::trace(Iprojz * (*j)->Rate() * P * rho0)) << " ";
								}
							}
							else if (this->Properties()->Get("dnp", Dnp) && Dnp == true)
							{
								std::cout << "Perfoming DNP calculation." << std::endl;
								this->Log() << "Just using the projection operator of " << (*l)->Name() << " and not doing CIDNP." << std::endl;

								std::cout << "rho: " << rho0 << std::endl;
								std::cout << "Ix:" << std::real(arma::trace(Iprojx * rho0)) << std::endl;
								std::cout << "Ix:" << Iprojx << std::endl;
								std::cout << "KE" << Iprojy * rho0 << std::endl;
								std::cout << "Iy:" << std::real(arma::trace(Iprojy * rho0)) << std::endl;
								std::cout << "Iy:" << Iprojy << std::endl;
								std::cout << "Iz:" << std::real(arma::trace(Iprojz * rho0)) << std::endl;
								std::cout << "Iz:" << Iprojz << std::endl;
								// Return the yield for this state - note that no reaction rates are included here.
								this->Data() << std::real(arma::trace(Iprojx * rho0)) << " ";
								this->Data() << std::real(arma::trace(Iprojy * rho0)) << " ";
								this->Data() << std::real(arma::trace(Iprojz * rho0)) << " ";
							}
							else
							{
								std::cout << "Perfoming Double-Projection calculation." << std::endl;
								// Loop through all states
								auto states = (*i)->States();
								for (auto j = states.cbegin(); j != states.cend(); j++)
								{
									if (!space.GetState((*j), P))
									{
										this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
										continue;
									}

									std::cout << "Ix:" << std::real(arma::trace(Iprojx * P * rho0)) << std::endl;
									std::cout << "Iy:" << std::real(arma::trace(Iprojy * P * rho0)) << std::endl;
									std::cout << "Iz:" << std::real(arma::trace(Iprojz * P * rho0)) << std::endl;

									// Return the yield for this state - note that no reaction rates are included here.
									this->Data() << std::real(arma::trace(Iprojx * P * rho0)) << " ";
									this->Data() << std::real(arma::trace(Iprojy * P * rho0)) << " ";
									this->Data() << std::real(arma::trace(Iprojz * P * rho0)) << " ";
								}
							}
						}
					}
				}

				this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
			}
			else
			{
				this->Log() << "No nucleus was specified for projection" << std::endl;
				continue;
			}
		}

		// Terminate the line in the data file after iteration through all spin systems
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticSSCIDNP::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
			{
				std::string spintype;

				(*l)->Properties()->Get("type", spintype);

				if (spintype != "electron")
				{
					// Should yields be written per transition or per defined state?
					if (this->productYieldsOnly)
					{
						// Write each transition name
						auto transitions = (*i)->Transitions();
						for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
						{
							_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
									<< ".Ix ";
							_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
									<< ".Iy ";
							_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield"
									<< ".Iz ";
						}
					}
					else
					{
						// Write each state name
						auto states = (*i)->States();
						for (auto j = states.cbegin(); j != states.cend(); j++)
						{
							_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".Ix ";
							_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".Iy ";
							_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".Iz ";
						}
					}
				}
			}
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticSSCIDNP::Validate()
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
	// -----------------------------------------------------
}
