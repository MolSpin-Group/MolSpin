/////////////////////////////////////////////////////////////////////////
// TaskMultiStaticSSTimeEvo implementation (RunSection module)

// -- Multi-system version: Allows transitions between SpinSystems --
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskMultiStaticSSTimeEvo.h"
#include "Transition.h"
#include "Interaction.h"
#include "Settings.h"
#include "State.h"
#include "ObjectParser.h"
#include "Utility.h"
#include "Operator.h"
#include "Pulse.h"

#include <ctime>

namespace RunSection
{
	// -----------------------------------------------------
	// TaskMultiStaticSSTimeEvo Constructors and Destructor
	// -----------------------------------------------------
	TaskMultiStaticSSTimeEvo::TaskMultiStaticSSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4),
																																reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskMultiStaticSSTimeEvo::~TaskMultiStaticSSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskMultiStaticSSTimeEvo protected methods
	// -----------------------------------------------------
	bool TaskMultiStaticSSTimeEvo::RunLocal()
	{
		this->Log() << "Running method StaticSS-MultiSystem." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems to obtain SpinSpace objects
		auto systems = this->SpinSystems();
		std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> spaces;
		unsigned int dimensions = 0;
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			auto space = std::make_shared<SpinAPI::SpinSpace>(*(*i));

			// We are using superoperator space, and need the total dimensions
			space->UseSuperoperatorSpace(true);
			space->SetReactionOperatorType(this->reactionOperators);
			space->SetTime(0.0);
			dimensions += space->SpaceDimensions();

			// Make sure to save the newly created spin space
			spaces.push_back(std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>(*i, space));
		}

		// Now, create a matrix to hold the Liouvillian superoperator and the initial state
		arma::sp_cx_mat L(dimensions, dimensions);
		arma::sp_cx_mat dL(dimensions,dimensions);
		arma::cx_vec rho0(dimensions);
		unsigned int nextDimension = 0; // Keeps track of the dimension where the next spin space starts

		// Loop through the systems again to fill this matrix and vector
		for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
		{
			// If SpinSpace is made up of multiple SubSystems, this get's handled seperately rejoining when evaluating the creation operators for linking SpinSpaces.
			//  Make sure we have an initial state
			auto initial_states = i->first->InitialState();
			i->second->SetTime(0.0);
			arma::cx_mat rho0HS;
			if (initial_states.size() < 1)
			{
				this->Log() << "Note: No initial state specified for spin system \"" << i->first->Name() << "\", setting the initial state to zero." << std::endl;
				rho0HS = arma::zeros<arma::cx_mat>(i->second->HilbertSpaceDimensions(), i->second->HilbertSpaceDimensions());
			}
			else
			{
				// Get the initial state for the current system
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;
					if (!i->second->GetState(*j, tmp_rho0))
					{
						this->Log() << "ERROR: Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << i->first->Name() << "\"." << std::endl;
						return false;
					}
					if (j == initial_states.cbegin())
						rho0HS = tmp_rho0;
					else
						rho0HS += tmp_rho0;
				}
				rho0HS /= arma::trace(rho0HS); // The density operator should have a trace of 1
			}

			// Now put the initial state into the superspace vector
			arma::cx_vec rho0vec;
			if (!i->second->OperatorToSuperspace(rho0HS, rho0vec))
			{
				this->Log() << "ERROR: Failed convert initial state to superspace for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			rho0.rows(nextDimension, nextDimension + i->second->SpaceDimensions() - 1) = rho0vec;

			// Next, get the Hamiltonian
			arma::sp_cx_mat H;
			if(!i->second->StaticHamiltonian(H))
			{
				this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			SCData DataStruct = GetHamiltonian(H,i->second->SpaceDimensions());
			L.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) = arma::cx_double(0.0, -1.0) * DataStruct.H;
			if(i->second->HasTimedependentInteractions())
			{
				arma::sp_cx_mat dH;
				if (!i->second->DynamicHamiltonian(dH))
				{
					this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}
				dL.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) = arma::cx_double(0.0, -1.0) * dH;
			}


			// Then get the reaction operators
			arma::sp_cx_mat K;
			if(!i->second->StaticTotalReactionOperator(K))
			{
				this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operators for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			L.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) -= K;
			if(i->second->HasTimedependentTransitions())
			{
				arma::sp_cx_mat dK;
				if(!i->second->DynamicTotalReactionOperator(dK))
				{
					this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}
				dL.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) -= dK ;
			}

			// Get the relaxation terms, assuming that they can just be added to the Liouvillian superoperator
			arma::sp_cx_mat R;
			for (auto t = i->first->operators_cbegin(); t != i->first->operators_cend(); t++)
			{
				if(i->second->RelaxationOperator((*t), R))
				{
					L.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) += R;
					this->Log() << "Added relaxation operator \"" << (*t)->Name() << "\" to the Liouvillian.\n";
				}
			}

			// Move on to next spin space
			nextDimension += i->second->SpaceDimensions();
		}

		auto UpdateTimeDependentL = [&](double time) {
			unsigned int row = 0;
			arma::sp_cx_mat Ltd(dimensions,dimensions);
			for(auto i = spaces.begin(); i != spaces.end(); i++)
			{
				i->second->SetTime(time);
				if(i->second->HasTimedependentInteractions())
				{
					arma::sp_cx_mat dH;
					if(!i->second->DynamicHamiltonian(dH))
					{
						this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
						//return false;
					}
					Ltd.submat(row, row, row + i->second->SpaceDimensions() - 1, row + i->second->SpaceDimensions() - 1) = arma::cx_double(0.0, -1.0) * dH;
				}

				if(i->second->HasTimedependentTransitions())
				{
					arma::sp_cx_mat dK;
					if(!i->second->DynamicTotalReactionOperator(dK))
					{
						this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
						//return false;
					}
					Ltd.submat(row, row, row + i->second->SpaceDimensions() - 1, row + i->second->SpaceDimensions() - 1) -= dK ;
				}
				row += i->second->SpaceDimensions();
			}
			return Ltd;
		};
		bool header = false;
		auto GetPulses = [&](double CurrentTime, std::vector<std::pair<SpinAPI::PulseSequence_ptr, std::shared_ptr<SpinAPI::SpinSpace>>>& sequence_pair, arma::cx_vec& rho) {
			auto pulse = SpinAPI::GetPulseOperator(sequence_pair,rho,CurrentTime);
			arma::sp_cx_mat PulseMat(L.n_rows, L.n_cols);
			std::vector<std::shared_ptr<SpinAPI::SpinSpace>> space_vec;
			nextDimension = 0;
			for(auto& seq_space : sequence_pair)
			{
				auto[seq,space] = seq_space;
				auto p = seq->GetActivePulseAtTime(CurrentTime);
				space_vec.push_back(space);
				if(p.first.IsNullptr())
					continue;
				if(!header)
				{
					this->Log() << "Active pulses | active time\n";
					header = true;
				}
				std::string name = "";
				auto[pu,inte,tr] = p.first.get();
				name = (pu) ? pu->Name() : (inte) ? inte->Name() : tr->Name();
				this->Log() << seq->Name() << "." << name << " | " << p.second << std::endl;
			}

			for(auto i = space_vec.cbegin(); i != space_vec.cend(); i++)
			{
				nextDimension = 0;
				for(auto j = spaces.begin(); j != spaces.end(); j++)
				{
					if((*i) == (*j).second)
						break;
					nextDimension += (*j).second->SpaceDimensions();
				}
				unsigned int idx = i - space_vec.begin(); 
				PulseMat.submat(nextDimension, nextDimension, nextDimension + (*i)->SpaceDimensions() - 1, nextDimension + (*i)->SpaceDimensions() - 1) += pulse[idx];
			}
			return PulseMat;
		};

		// Obtain the creation operators - note that we need to loop through the other SpinSystems again to find transitions leading into the current SpinSystem
		auto GetCreationOperators = [&]() {
			arma::sp_cx_mat C_mat(L.n_rows, L.n_cols);
			nextDimension = 0;
			for(auto i = spaces.cbegin(); i != spaces.cend(); i++) 
			{
				unsigned int nextCDimension = 0; // Similar to nextDimension, but to keep track of first dimension for this other SpinSystem
				for (auto j = spaces.cbegin(); j != spaces.cend(); j++)
				{
					// Creation operators are off-diagonal elements
					if (j != i)
					{
						// Check all transitions whether they should produce a creation operator
						for (auto t = j->first->transitions_cbegin(); t != j->first->transitions_cend(); t++)
						{
							// Does the Transition lead into the current spin space?
							if ((*t)->Target() == i->first)
							{
								// Prepare a creation operator
								arma::sp_cx_mat C;
								if (!SpinAPI::CreationOperator((*t), *(j->second), *(i->second), C, true))
								{
									this->Log() << "ERROR: Failed to obtain matrix representation of the creation operator for transition \"" << (*t)->Name() << "\"!" << std::endl;
									//return false;
								}
								// Put it into the total Liouvillian:
								//  - The row should be that of the current spin space (the target space)
								//  - The column should be that of the source spin space (the spin system containing the Transition object)
								C_mat.submat(nextDimension, nextCDimension, nextDimension + i->second->SpaceDimensions() - 1, nextCDimension + j->second->SpaceDimensions() - 1) += C * (*t)->Rate();
							}
						}
					}
					// Move on to check next spin system for transitions into the current spin space
					nextCDimension += j->second->SpaceDimensions();
				}
				// Move on to next spin space
				nextDimension += i->second->SpaceDimensions();
			}
			nextDimension = 0;
			return C_mat;
		};
		arma::sp_cx_mat L_base = L;
		L = L + GetCreationOperators() + dL;

		// Write results for initial state as well (i.e. at time 0)
		this->Data() << this->RunSettings()->CurrentStep() << " 0 ";
		this->WriteStandardOutput(this->Data());
		SeperateSpinSystems(rho0,spaces,this->ProductYieldsOnly, this->timestep);
		this->Data() << std::endl;

		arma::cx_vec result = arma::cx_vec(rho0.n_rows);
		double CurrentTime = 0.0;
		bool NoFail = false;
		
		CheckPropagator(L, this->timestep);
		if(this->prop == Propagator::Default)
		{
			DetermineBestPropagator(L);
		}

		// Very much a quick solution atm
		if (this->prop == Propagator::exp)
		{
			this->Log() << "Using exponential propogator" << std::endl;
			this->Log() << "Calculating the propagator..." << std::endl;
			arma::cx_mat P = arma::expmat(arma::conv_to<arma::cx_mat>::from(L) * this->timestep);
			this->Log() << "Ready to perform calculation." << std::endl;
			unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
			for (unsigned int n = 1; n <= steps; n++)
			{
				this->Data() << this->RunSettings()->CurrentStep() << " ";
				CurrentTime += this->timestep;
				this->Data() << CurrentTime << " ";
				this->WriteStandardOutput(this->Data());

				// Propagate (use special scope to be able to dispose of the temporary vector asap)
				{
					arma::cx_vec tmp = P * rho0;
					rho0 = tmp;
				}
				NoFail = SeperateSpinSystems(rho0,spaces,this->ProductYieldsOnly, this->timestep);
				if (!NoFail)
					return false;
				this->Data() << std::endl;
			}
			this->Log() << "Done with calculation." << std::endl;
			return true;

		}

		// Perform the calculation
		this->Log() << "Ready to perform calculation." << std::endl;
		double InitialTimeStep = this->timestep;
		double MinTimeStep, MaxTimeStep = 0.0;
		double MinTolerance, MaxTolerance = 0.0;
		if (!this->Properties()->Get("minimumtimestep", MinTimeStep) and !this->Properties()->Get("minimum timestep", MinTimeStep))
		{
			MinTimeStep = InitialTimeStep * 1e-3;
		}
		if (!this->Properties()->Get("maximumtimestep", MaxTimeStep) and !this->Properties()->Get("maximum timestep", MaxTimeStep))
		{
			MaxTimeStep = InitialTimeStep * 1e4;
		}

		if (!this->Properties()->Get("absolutetolerance", MinTolerance) and !this->Properties()->Get("absolute tolerance", MinTolerance) and !this->Properties()->Get("atol", MinTolerance))
		{
			MinTolerance = 1e-8;
		}
		if (!this->Properties()->Get("relativetolerance", MaxTolerance) and !this->Properties()->Get("relative tolerance", MaxTolerance) and !this->Properties()->Get("rtol", MinTolerance))
		{
			MaxTolerance = 1e-10;
		}
		
		SpinAPI::SpinSpace::PropParam params;
		params.atol = MinTolerance;
		params.rtol = MaxTolerance;
		params.min = MinTimeStep;
		params.max = MaxTimeStep;
		params.safety = 0.7;
		params.f1 = 0.1;
		params.f2 = 2.0;

		//build timeevo block_structure
		std::vector<SpinAPI::PulseSequence_ptr> sequences;
		std::vector<std::pair<SpinAPI::PulseSequence_ptr, std::shared_ptr<SpinAPI::SpinSpace>>> sequence_space_pair;
		for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
		{
			auto seq_vec = i->first->PulseSequences();
			sequences.insert(sequences.end(), seq_vec.begin(), seq_vec.end());
			for(auto sq = seq_vec.begin(); sq != seq_vec.end(); sq++)
			{
				sequence_space_pair.push_back(std::make_pair((*sq),(*i).second));
			}
		}
		std::vector<block> time_blocks = GenerateTimeEvoBlocking(sequences, {MinTimeStep,MaxTimeStep},this->totaltime);
		this->Log() << PrintOutBlockStructure(time_blocks);

		this->Log() << "Starting time evolution with timestep: " << this->timestep << ", total time: " << this->totaltime << ", minimum timestep: " << MinTimeStep << ", maximum timestep: " << MaxTimeStep << std::endl;
		auto now = std::chrono::high_resolution_clock::now();
		int step = 0;
		size_t current_block = 0;
		while (CurrentTime < this->totaltime)
		{
			// Propagate
			//evaluate pulse sequence 
			ClampTimeEvolution(CurrentTime, this->totaltime, time_blocks, current_block, this->timestep, params);
			SpinAPI::SpinSpace::TimePropReturnInfo r;
			L = L_base + GetCreationOperators();
			dL = UpdateTimeDependentL(CurrentTime);
			L = L + dL + GetPulses(CurrentTime,sequence_space_pair,rho0);

			if(this->timestep + CurrentTime > this->totaltime)
			{
				this->timestep = this->totaltime - CurrentTime;
			}


			if(this->prop == Propagator::RK45)
			{
				r = RungeKutta45Armadillo(L, rho0, rho0, this->timestep, ComputeRhoDot, CurrentTime, params);
			}
			else
			{
				r = spaces[0].second->TimeAdaptiveKrylovGeneral(L, rho0, std::complex<double>(this->timestep,0.0), 30, L.n_rows, params);
			}

			double t = r.timestep;
			bool a = r.step_accepted;

			this->Data() << this->RunSettings()->CurrentStep() << " ";
			CurrentTime += (a == true) ? this->timestep : r.timestep_used;
			this->timestep = t;
			this->Data() << CurrentTime << " ";
			this->WriteStandardOutput(this->Data());
			rho0 = r.result;

			NoFail = SeperateSpinSystems(rho0, spaces, this->ProductYieldsOnly, (a == true) ? this->timestep : t);
			if (!NoFail)
				return false;
			// Terminate the line in the data file after iteration through all spin systems
			this->Data() << std::endl;

			step = floor(CurrentTime);
			if(step % 1000 == 0 && step != 0)
			{
				auto now2 = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now2-now);
				now = now2;
				this->Log() << "Time to step " << step << " : " << duration.count() << "(ms)" << std::endl;
			}
		}
		this->Log() << "Done with calculation." << std::endl;
		this->timestep = InitialTimeStep;

		return true;
	}

	// Gathers and outputs the results from a given time-integrated density operator
	void TaskMultiStaticSSTimeEvo::GatherResults(const arma::cx_mat &_rho, const SpinAPI::SpinSystem &_system, const SpinAPI::SpinSpace &_space)
	{
		// Loop through all states
		arma::cx_mat P;
		auto states = _system.States();
		for (auto j = states.cbegin(); j != states.cend(); j++)
		{
			if (!_space.GetState((*j), P))
			{
				this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << _system.Name() << "\"." << std::endl;
				continue;
			}

			// Return the yield for this state - note that no reaction rates are included here.
			this->Data() << std::abs(arma::trace(P * _rho)) << " ";
		}
	}

    bool TaskMultiStaticSSTimeEvo::SeperateSpinSystems(const arma::cx_vec &rho0, const std::vector<std::pair<SpinAPI::system_ptr, std::shared_ptr<SpinAPI::SpinSpace>>> &spaces, bool transitionyields, double t)
    {
		// Retrieve the resulting density matrix for each spin system and output the results
		int nextDimension = 0;
		double SumYield = 0;
		if (transitionyields)
		{
			int NextDimension = 0;
			for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
			{
				arma::cx_mat rho_result;
				int TempDimension = NextDimension + i->second->SpaceDimensions();
				arma::cx_vec rho_result_vec = rho0.rows(NextDimension,TempDimension);
				NextDimension = TempDimension;
				if (!i->second->OperatorFromSuperspace(rho_result_vec, rho_result))
				{
					this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				// Loop through all defind transitions
				auto transitions = (*i).first->Transitions();
				arma::cx_mat P;
				for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
				{
					// Make sure that there is a state object
					if ((*j)->SourceState() == nullptr)
						continue;	
					if (!i->second->GetState((*j)->SourceState(), P))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i).first->Name() << "\"." << std::endl;
						continue;
					}
					double Yield = std::exp(-1 * (*j)->Rate() * t) * std::abs(arma::trace(P * rho_result));
					SumYield += Yield;
					// Return the yield for this transition
					this->Data() << Yield << " ";
				}
			}
			this->Data() << SumYield << " ";
			SumYield = 0;
			return true;
		}
		for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
		{
			// Get the superspace result vector and convert it back to the native Hilbert space
			arma::cx_mat rho_result;
			arma::cx_vec rho_result_vec;
			rho_result_vec = rho0.rows(nextDimension, nextDimension + i->second->SpaceDimensions() - 1);
			if (!i->second->OperatorFromSuperspace(rho_result_vec, rho_result))
			{
				this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			// Get the results
			this->GatherResults(rho_result, *(i->first), *(i->second));
			// Move on to next spin space
			nextDimension += i->second->SpaceDimensions();
		}
		return true;
	}

	// Right hand side of the master equation for use with Runge Kutta
	arma::cx_vec TaskMultiStaticSSTimeEvo::ComputeRhoDot(double t, arma::sp_cx_mat &L, arma::cx_vec &K, arma::cx_vec RhoNaught)
	{
		arma::cx_vec ReturnVec(L.n_rows);
		RhoNaught = RhoNaught + K;
		ReturnVec = L * RhoNaught;
		return ReturnVec;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskMultiStaticSSTimeEvo::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time(ns) ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Should yields be written per transition or per defined state?
			if (this->ProductYieldsOnly)
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

	// Validation of the required input
	bool TaskMultiStaticSSTimeEvo::Validate()
	{
		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;

		this->Properties()->Get("transitionyields", this->ProductYieldsOnly);

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
		// Get Propogator
		std::string propagator_str;
		if (!this->Properties()->Get("Propagator", propagator_str) && !this->Properties()->Get("propagator", propagator_str))
		{
			this->Log() << "No propagator defined, using the default propogator (RK45)" << std::endl;
			this->propogator_cached = false;
			this->prop = Propagator::Default;
		}
		else
		{
			this->SelectPropagator(propagator_str);
			//this->propogator_cached = true;
		}

		//if (this->prop == Propagator::Default)
		//	this->prop = Propagator::RK45;

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
