/////////////////////////////////////////////////////////////////////////
// TaskStaticSSTimeEvo implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "TaskStaticSSTimeEvo.h"
#include "Transition.h"
#include "Operator.h"
#include "Settings.h"
#include "Interaction.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticSS Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSTimeEvo::TaskStaticSSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4),
																													  reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticSSTimeEvo::~TaskStaticSSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskStaticSSTimeEvo::RunLocal()
	{
		this->Log() << "Running method StaticSSTimeEvolution." << std::endl;

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
		std::vector<std::pair<arma::cx_mat, arma::cx_vec>> P(systems.size()); // Create array containing a propagator and the current state of each system //LINE MODIFIED FOR SW
		std::vector<SpinAPI::SpinSpace> spaces(systems.size());				  // Keep a SpinSpace object for each spin system
		std::vector<SCData> SWdata(systems.size());
		std::vector<bool> SW(systems.size(), false);

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

			// Get the initial state
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
			// Get the initial state without weights
			{
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;

					// Get the initial state in thermal equilibrium
					if ((*j) == nullptr) // "Thermal initial state"
					{
						this->Log() << "Initial state = thermal " << std::endl;

						// Get the thermalhamiltonianlist
						std::vector<std::string> thermalhamiltonian_list = (*i)->ThermalHamiltonianList();
						
						this->Log() << "ThermalHamiltonianList = [";
						for (size_t j = 0; j < thermalhamiltonian_list.size(); j++)
						{
							this->Log() << thermalhamiltonian_list[j];
							if (j < thermalhamiltonian_list.size() - 1)
								this->Log() << ", ";  // Add a comma between elements
						}
						this->Log() << "]" << std::endl;
						
						// Get temperature
						double temperature = (*i)->Temperature();
						this->Log() << "Temperature = " << temperature << "K" << std::endl;

						// Get the initial state with thermal equilibrium
						if (!space.GetThermalState(space, temperature, thermalhamiltonian_list, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto thermal state, initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}
					else // Get the initial state without thermal equilibrium
					{
						if (!space.GetState(*j, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}

					// Obtain the initial density matrix
					if (j == initial_states.cbegin())
					{
						rho0 = tmp_rho0;
					}
					else
					{
						rho0 += tmp_rho0;
					}
				}
			}
			rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

			// Convert initial state to superoperator space
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

			// Get the Hamiltonian
			arma::sp_cx_mat H;
			if (!space.Hamiltonian(H, static_cast<int>(this->name)))
			{
				this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
				continue;
			}

			// Get a matrix to collect all the terms (the total Liouvillian)
			SCData DataStruct = GetHamiltonian(H,space.SpaceDimensions());
			arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * DataStruct.H;

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
					this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to the Liouvillian for spin system \"" << (*i)->Name() << "\".\n";
				}
			}

			// Get the propagator and put it into the array together with the initial state
			bool SC = false;
			if(DataStruct.SamplesMatrix.n_nonzero != 0)
			{
				SC = true;
			}


			if(!SC)
			{
				P[ic] = std::pair<arma::cx_mat, arma::cx_vec>(arma::expmat(arma::conv_to<arma::cx_mat>::from(A) * this->timestep), rho0vec);
			}
			else
			{
				P[ic] = std::pair<arma::cx_mat, arma::cx_vec>(arma::conv_to<arma::cx_mat>::from(A),rho0vec);
			}
			SWdata[ic] = DataStruct;
			SW[ic] = SC;
			++ic;
		}

		// Output results at the initial step (before calculations)
		this->Data() << this->RunSettings()->CurrentStep() << " 0 "; // "0" refers to the time
		this->WriteStandardOutput(this->Data());
		ic = 0;

		std::vector<std::vector<std::pair<int,arma::cx_vec>>> SCresults;
		std::vector<std::vector<std::pair<int,double>>> SCweights;
		std::vector<std::vector<arma::sp_cx_mat>> As;
		std::vector<std::vector<std::vector<std::vector<double>>>> AllWeights;
		std::vector<std::pair<std::vector<double>,std::vector<int>>> BLandSamples;
		std::vector<std::vector<std::vector<double>>> SampleSpacing;
		std::vector<std::vector<std::vector<double>>> SampleWeights;
		std::vector<std::vector<std::pair<int,arma::cx_vec>>> rho0s;

		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			if(SW[ic])
			{
				AllWeights.push_back({{}});
				BLandSamples.push_back({});
				SampleSpacing.push_back({});
				As.push_back({});
				SCresults.push_back({});
				SCweights.push_back({});
				SampleWeights.push_back({});
				rho0s.push_back({});

				for(auto e = (*i)->interactions_cbegin(); e != (*i)->interactions_cend(); e++)
				{
					if((*e)->Type() == SpinAPI::InteractionType::SemiClassicalField)
					{
						AllWeights[ic][0].push_back((*e)->GetOriWeights());
						std::vector<double> BL = (*e)->VL();
						double BMax = std::accumulate(BL.begin(), BL.end(), 0.0);
						BLandSamples[ic].first.push_back(BMax);
						BLandSamples[ic].second.push_back((*e)->Orientations());
						SampleSpacing[ic].push_back((*e)->GetSpacing());
					}
				}

				std::vector<SCData> SysData= {SWdata[ic]};
				arma::sp_cx_mat A = arma::conv_to<arma::sp_cx_mat>::from(P[ic].first);
				GetSamples(As[ic],A, SysData, SampleWeights[ic], AllWeights[ic]);
			}

			arma::cx_mat PState;
			auto states = (*i)->States();
			//Convert the resulting density operator back to its Hilbert space representation
			if (!spaces[ic].OperatorFromSuperspace(P[ic].second, rho0))
			{
				this->Log() << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
				continue;
			}

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
		//only testing exp propogator at the moment
		unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
		std::vector<std::vector<std::pair<int, arma::cx_mat>>> Propagators;
		ic = 0;
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			std::vector<std::pair<int,arma::cx_mat>> exp_prop;
			if(SW[ic])
			{
				this->Log() << "Using exponential propogator" << std::endl;
				this->Log() << "Calculating the propagator..." << std::endl;
				arma::cx_mat temp_mat;
				arma::cx_vec temp_vec;
				for (unsigned int e = 0; e < As[ic].size(); e++)
				{
					exp_prop.push_back({0,temp_mat});
					SCweights[ic].push_back({0,0.0});
					rho0s[ic].push_back({e,temp_vec});
				}
					unsigned int threads = GetNumThreads();
					//threads = (unsigned int)std::floor((double)threads / 2.0);
					if (As[ic].size() > static_cast<size_t>(std::numeric_limits<int>::max()))
						throw std::runtime_error("Too many semi-classical samples for OpenMP loop indexing.");

					const int sample_count = static_cast<int>(As[ic].size());
					#pragma omp parallel for num_threads(threads)
					for (int e = 0; e < sample_count; ++e)
					{
						arma::cx_mat expP = arma::expmat(arma::conv_to<arma::cx_mat>::from(As[ic][e]) * this->timestep);
						std::vector<double> weights = SampleWeights[ic][e];
						double weight_product = 1.0;
						for(unsigned int j = 0; j < weights.size(); j++)
						{
							weight_product *= weights[j];
						}
						#pragma omp critical
						{
							exp_prop[e] = {e,expP};
							SCweights[ic][e] = {e,weight_product};
							rho0s[ic][e] = {e,P[ic].second};
						}
					}
				Propagators.push_back(exp_prop);
			}
			else
			{
				exp_prop.push_back({0,P[ic].first});
				Propagators.push_back(exp_prop);
			}
		}

		for (unsigned int n = 1; n <= steps; n++)
		{
			// Write first part of the data output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << (static_cast<double>(n) * this->timestep) << " ";
			this->WriteStandardOutput(this->Data());

			// Loop through the systems again and progress a step
			ic = 0;
			for(auto i = systems.cbegin(); i != systems.cend(); i++)
			{
				arma::cx_vec result = arma::zeros<arma::cx_vec>(rho0vec.n_rows);
					if(SW[ic])
					{
						SCresults[ic].clear();
						if (Propagators[ic].size() > static_cast<size_t>(std::numeric_limits<int>::max()))
							throw std::runtime_error("Too many propagators for OpenMP loop indexing.");

						const int propagator_count = static_cast<int>(Propagators[ic].size());
						#pragma omp parallel for
						for (int e = 0; e < propagator_count; ++e)
						{
							arma::cx_vec tmp = Propagators[ic][e].second * rho0s[ic][e].second;
							#pragma omp critical
							{
								SCresults[ic].push_back({Propagators[ic][e].first, tmp * SCweights[ic][e].second});
								//SCresults[ic].push_back({e, tmp * SCweights[ic][e].second});
								//std::cout << arma::trace(tmp) << std::endl;
							}
							rho0s[ic][e] = {Propagators[ic][e].first,tmp};
						}
					std::pair<arma::cx_vec, double> results = IntegrateSC(SCresults[ic], SCweights[ic], SCIntegrationProperties{BLandSamples[ic].first, BLandSamples[ic].second, SampleSpacing[ic]});
					result = results.first;
					P[ic].second = result;
				}
				else
				{
					result = Propagators[ic][0].second * P[ic].second;
					P[ic].second = result;
				}
				
				rho0vec = P[ic].second;
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
	void TaskStaticSSTimeEvo::WriteHeader(std::ostream &_stream)
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
	bool TaskStaticSSTimeEvo::Validate()
	{
		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;
		this->name = TaskName::STATICSS_TIMEVO;

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

		//Get Propogator
		std::string propagator_str;
		if (!this->Properties()->Get("Propagator", propagator_str) && !this->Properties()->Get("propagator", propagator_str))
		{
			this->Log() << "No propagator defined, using the default propogator (RK45)" << std::endl;
		}
		else
		{
			this->SelectPropagator(propagator_str);
		}
		
		if(this->prop == Propagator::Default)
			this->prop = Propagator::exp;


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
