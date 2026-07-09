/////////////////////////////////////////////////////////////////////////
// TaskMultiRadicalPairSSTimeEvo implementation (RunSection module)

// -- Multi-system version: Allows transitions between SpinSystems --
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <omp.h>
#include <iomanip>

#include "TaskMultiRadicalPairSSTimeEvo.h"
#include "Transition.h"
#include "Interaction.h"
#include "Settings.h"
#include "State.h"
#include "ObjectParser.h"
#include "SubSystem.h"

//#include <unsupported/Eigen/MatrixFunctions>
//#include <Eigen/Sparse>


namespace RunSection
{
	// -----------------------------------------------------
	// TaskMultiRadicalPairSSTimeEvo Constructors and Destructor
	// -----------------------------------------------------
	TaskMultiRadicalPairSSTimeEvo::TaskMultiRadicalPairSSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4),
																																reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskMultiRadicalPairSSTimeEvo::~TaskMultiRadicalPairSSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskMultiRadicalPairSSTimeEvo protected methods
	// -----------------------------------------------------
	bool TaskMultiRadicalPairSSTimeEvo::RunLocal()
	{
		this->Log() << "Running method StaticSS-MultiRadicalPairSystem." << std::endl;
		std::cout << "This task has been depreciated by the MultiStaticSS Task" << std::endl;
		this->Log() << "This task has been depreciated by the MultiStaticSS Task" << std::endl;
		this->timestep = this->OriginalTimestep;

		// If this is the first step, write first part of header to the data file
		bool YieldOnly = false;
		if (this->Properties()->Get("yieldonly", YieldOnly) || this->Properties()->Get("yield only", YieldOnly))
			YieldOnly = YieldOnly;
		bool SilentMode = false;
		if (this->Properties()->Get("silentmode", SilentMode) || this->Properties()->Get("silent mode", SilentMode))
			SilentMode = SilentMode;
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems to obtain SpinSpace objects
		auto systems = this->SpinSystems();
		if(systems.size() > 1)
		{
			this->Log() << "Error: Only supports one spin system" << std::endl;
			return false;
		}

		//get the number of sub systems
		auto ss = systems[0]->SubSystems();
		int SubSystems = ss.size();

		//change to this->Properties()->Get()
		//if(!this->Properties().get()->Get("subsystems", SubSystems))
		//{
		//	this->Log() << "Error: Number of sub systems not specified. Correct syntax is e.g. subsystems: 2" << std::endl;
		//	return false;
		//}
		
		//std::string SubSystemNames; 
		//if(!this->Properties().get()->Get("subsystemnames", SubSystemNames))
		//{
		//	this->Log() << "Note: subsystemnames not specified. Asssuming names are system1, system2, system3...." << std::endl;
		//}
		
		std::vector<std::pair<std::string,std::vector<std::string>>> SubSystemSpins;
		std::vector<std::pair<std::string,std::vector<SpinAPI::interaction_ptr>>> SubSystemsInteractions;
		std::vector<SubSystemTransition> SubSystemsTransitions;
		//std::vector<std::pair<SpinAPI::transition_ptr, std::pair<std::string, std::string>>>SubSystemsInterTransitions;
		//{
		//	std::stringstream ss(SubSystemNames);
		//	while(ss.good())
		//	{
		//		std::string Name;
		//		std::getline(ss,Name, ',');
		//		std::transform(Name.begin(), Name.end(), Name.begin(), [](unsigned char c){return std::tolower(c);});
		//		SubSystemSpins.push_back({Name, {}});
		//	}
		//}

		for(auto s : ss)
		{
			SubSystemSpins.push_back({s->Name(), s->GetSpinNames()});

			for(auto tr : s->GetTransitions())
			{
				//remove duplicates
				bool duplicate = false;
				for(unsigned int i = 0; i < SubSystemsTransitions.size(); i++)
				{
					if(*tr.TransitionObject == SubSystemsTransitions[i].transition)
					{
						std::string d = "";
						SubSystemsTransitions[i].transition->Properties()->Get("duplicate", d);
						if(d[0] != 't') //only need to check first letter
						{
							duplicate = true;
							break;
						}
					}

				}

				if(duplicate) {continue;}

				SubSystemTransition TransitionData;
				TransitionData.transition = *tr.TransitionObject;
				TransitionData.type = tr.type;
				TransitionData.source = tr.source;

				switch(tr.type)
				{
				case 0:
					SubSystemsTransitions.push_back(TransitionData);
					break;
				case 1:
					TransitionData.target = tr.target;
					SubSystemsTransitions.push_back(TransitionData);
					break;
				default:
					break;
				}
			}

			SubSystemsInteractions.push_back({s->Name(), s->GetInteractions()});

		}

		std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> spaces;
		unsigned int dimensions = 0;
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			auto space = std::make_shared<SpinAPI::SpinSpace>(*(*i));

			// We are using superoperator space, and need the total dimensions
			space->UseSuperoperatorSpace(true);
			space->SetReactionOperatorType(this->reactionOperators);
			dimensions += space->SpaceDimensions();

			// Make sure to save the newly created spin space
			spaces.push_back(std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>(*i, space));
		}
		int super_dimensions = dimensions * SubSystems; //every SubSystem has the same dimension so the total operator dimension has to be bigger depending on the number of subsystems
		// Now, create a matrix to hold the Liouvillian superoperator and the initial state
		arma::sp_cx_mat L(super_dimensions, super_dimensions);
		arma::cx_vec rho0(super_dimensions);
		unsigned int nextDimension = 0; // Keeps track of the dimension where the next spin space starts

		// Loop through the systems again to fill this matrix and vector
		int spinsystem = 0; 
		auto SpinSpace = spaces.cbegin();

		std::vector<SpinAPI::state_ptr> AllStates;
		std::vector<std::string> InitialStates;
		AllStates = SpinSpace->first->GetAllStates();
		std::string InitialStatesString;
		if(!SpinSpace->first->GetProperties()->Get("initialstate", InitialStatesString))
		{
			this->Log() << "No initial states provided, density matrix will be set to 0" << std::endl;
			for(int i = 0; i < SubSystems; i++)
			{
				InitialStates.push_back({""});
			}
		}
		else
		{
			std::stringstream ss(InitialStatesString);
			while(ss.good())
			{
				std::string state;
				std::getline(ss,state, ',');
				InitialStates.push_back(state);
			}
		}
		
		for(auto i = SubSystemSpins.cbegin(); i != SubSystemSpins.end(); i++)
		{
			std::cout << "Radical Pair: " << spinsystem << std::endl;
			//Get intitial state
			auto initial_state = InitialStates[spinsystem]; 
			auto FindState = [&initial_state](auto state)
			{
				return state->Name() == initial_state;
			};
			bool found = true;
			auto FoundState = std::find_if(AllStates.begin(), AllStates.end(), FindState);
			if(*FoundState == nullptr)
			{
				found = false;
			}
			arma::cx_mat rho0HS;
			if(initial_state == "zero" || initial_state == "")
			{
				rho0HS = arma::zeros<arma::cx_mat>(SpinSpace->second->HilbertSpaceDimensions(), SpinSpace->second->HilbertSpaceDimensions());
			}
			else if(found == true)
			{
				if(!SpinSpace->second->GetState((*FoundState), rho0HS))
				{
					this->Log() << "ERROR: Failed to obtain projection matrix onto state \"" << (*FoundState)->Name() << "\", initial state of SpinSubSystem \"" << i->first << "\"." << std::endl;
					return false;
				}
				rho0HS /= arma::trace(rho0HS);
			}
			else
			{
				rho0HS = arma::zeros<arma::cx_mat>(SpinSpace->second->HilbertSpaceDimensions(), SpinSpace->second->HilbertSpaceDimensions());
			}

			// Now put the initial state into the superspace vector
			arma::cx_vec rho0vec;
			if (!SpinSpace->second->OperatorToSuperspace(rho0HS, rho0vec))
			{
				this->Log() << "ERROR: Failed convert initial state to superspace for spin subsystem \"" << SubSystemSpins[spinsystem].first << "\"!" << std::endl;
				return false;
			}
			rho0.rows(nextDimension, nextDimension + SpinSpace->second->SpaceDimensions() - 1) = rho0vec;

			// Next, get the Hamiltonian
			arma::sp_cx_mat H;
			if (!GenerateHamiltonian(SubSystemsInteractions[spinsystem].second, H, dimensions, SpinSpace->second))
			{
				this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin spin subsystem \"" << SubSystemSpins[spinsystem].first << "\"!" << std::endl;
				return false;
			}
			L.submat(nextDimension, nextDimension, nextDimension + SpinSpace->second->SpaceDimensions() - 1, nextDimension + SpinSpace->second->SpaceDimensions() - 1) = arma::cx_double(0.0, -1.0) * H;
			
			// Then get the reaction operators
			arma::sp_cx_mat K;
			if (!GenerateReactionOperator(SubSystemsTransitions, K, dimensions, SpinSpace->second, i->first))
			{
				this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operators for spin spin subsystem \"" << SubSystemSpins[spinsystem].first << "\"!" << std::endl;
				return false;
			}
			L.submat(nextDimension, nextDimension, nextDimension + SpinSpace->second->SpaceDimensions() - 1, nextDimension + SpinSpace->second->SpaceDimensions() - 1) -= K;

			//Loop through all the interactions that involve transitions to and from other subsystems
			unsigned int nextCDimension = 0; // Similar to nextDimension, but to keep track of first dimension for this other SpinSystem
			for (auto j = SubSystemsTransitions.begin(); j != SubSystemsTransitions.end(); j++)
			{
				//only process the transition if the transition is leading into the current spin state
				if(j->type == 0)
				{
					continue;
				}
				if(j->target != SubSystemSpins[spinsystem].first)
				{
					continue;
				}

				//identify what column the source subsystem is
				std::string SubSystemName = j->source;
				auto FindSubSystem = [&SubSystemName](std::pair<std::string,std::vector<std::string>> t)
				{
					return SubSystemName == t.first;		
				};
				auto temp = std::find_if(SubSystemSpins.begin(), SubSystemSpins.end(), FindSubSystem);
				int col = temp - SubSystemSpins.begin();
				nextCDimension = col * dimensions;

				//transition is leading into subsystem
				arma::sp_cx_mat T;
				if(!SpinSpace->second->ReactionOperator(j->transition, T))
				{
					this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operator for transition \"" << j->transition->Name() << "\"!" << std::endl;
					return false;
				}
				//std::cout << T << std::endl;
				L.submat(nextDimension, nextCDimension, nextDimension + SpinSpace->second->SpaceDimensions() - 1, nextCDimension + SpinSpace->second->SpaceDimensions() - 1) += T;
				//T.print();
				//modify L for the transition leading out of the subsytem;
				//L.submat(nextCDimension, nextCDimension, nextCDimension + SpinSpace->second->SpaceDimensions() - 1, nextCDimension + SpinSpace->second->SpaceDimensions() - 1) += -1 * T;
				//std::cout << L << std::endl;
			}
			spinsystem++;
			nextDimension += dimensions;
		}
		//this->WriteHeader(this->Data(), YieldOnly);
		nextDimension = 0;

		//modify this so it's the same that I've got in my code, using the vectors and superspace rather than matracies;

		std::vector<std::pair<double, std::vector<TrajectoryData>>> trajectory;
		if(!YieldOnly)
		{
			// Write results for initial state as well (i.e. at time 0)
			this->Data() << this->RunSettings()->CurrentStep() << " 0 ";
			this->WriteStandardOutput(this->Data());
			trajectory.push_back({0.0, {}});
			for (auto i = SubSystemSpins.cbegin(); i != SubSystemSpins.cend(); i++)
			{
				trajectory[0].second.push_back(TrajectoryData(i->first));
				// Get the superspace result vector and convert it back to the native Hilbert space
				arma::cx_mat rho_result;
				arma::cx_vec rho_result_vec;
				rho_result_vec = rho0.rows(nextDimension, nextDimension + SpinSpace->second->SpaceDimensions() - 1);
				//std::cout << rho_result_vec << std::endl;
				if (!SpinSpace->second->OperatorFromSuperspace(rho_result_vec, rho_result))
				{
					//this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				// Get the results
				this->GatherResults(rho_result, *(SpinSpace->first), *(SpinSpace->second), trajectory[0].second[i - SubSystemSpins.begin()].StateTrace);

				// Move on to next spin space
				nextDimension += SpinSpace->second->SpaceDimensions();
			}
			this->Data() << std::endl;
		}

		//arma::umat locations = { {0,0,1,1},{0,1,0,1}};
		//arma::cx_vec values = {1,2,3,4};
//
		//arma::sp_cx_mat m(locations, values);
		//std::cout << m << std::endl;
//
		//Matrix m2 = ConvertArmadilloToEigen(m);
		//std::cout << Eigen::MatrixXcd(m2) << std::endl;
		//std::cout << ConvertEigenToArmadillo(m2) << std::endl;

		//auto LEigen = ConvertArmadilloToEigen(L);

		// Perform the calculation
		this->Log() << "Ready to perform calculation." << std::endl;
		double Currenttime = 0;
		//unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
		double InitialTimestep = this->timestep;
		unsigned int n = 1;


		if(YieldOnly)
		{
			arma::cx_vec rho1(super_dimensions);
			CalcYieldOnly(L, rho0, rho1);
			std::vector<double> yields;
			for (auto state : SpinSpace->first->States())
			{
				yields.push_back(0);
			}
			for(int i = 0; i < SubSystems; i++)
			{
				for(auto e = SubSystemsTransitions.begin(); e != SubSystemsTransitions.end(); e++)
				{	
					if(e->type != 0)
					{
						continue;
					}
					if(e->source != SubSystemSpins[i].first)
					{
						continue;
					}

					int index = 0;
					SpinAPI::state_ptr TransitionState;
					for(auto state : SpinSpace->first->States())
					{
						if(e->transition->SourceState() != state)
						{
							index++;
							continue;
						}
						TransitionState = state;
						break;
					}
					double rate = e->transition->Rate();
					arma::cx_mat P; 
					//*(SpinSpace->second))->GetState(TransitionState,P);
					SpinSpace->second->GetState(TransitionState,P);
					StateYield(rate,yields[index],i,SubSystems,P,rho1);
				}
			}

			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->WriteStandardOutput(this->Data());
			for(unsigned int i = 0; i < yields.size(); i++)
			{
				this->Data() << std::setprecision(8) << yields[i] << " ";
			}
			this->Data() << std::endl;
			return true;
		}

		//auto LInitial = L;

		double MinTimeStep = 0;
		double MaxTimeStep = 0;
		int flush = 0;
		if(!this->Properties()->Get("minimumtimestep", MinTimeStep) and !this->Properties()->Get("minimum timestep", MinTimeStep))
		{
			MinTimeStep = InitialTimestep * 1e-3;
		}
		if(!this->Properties()->Get("maximumtimestep", MaxTimeStep) and !this->Properties()->Get("maximum timestep", MaxTimeStep))
		{
			MaxTimeStep = InitialTimestep * 1e4;
		}
		if(!this->Properties()->Get("Flush", flush) && !this->Properties()->Get("flush", flush))
		{
			flush = 0;
		}
		//if(!this->Properties())
		std::vector<double> RollingYield;
		for (unsigned int i = 0; i < SpinSpace->first->States().size(); i++)
		{
			RollingYield.push_back(0.0);
		}

		this->Log() << "Starting time evolution with timestep: " << this->timestep << ", total time: " << this->totaltime << ", minimum timestep: " << MinTimeStep << ", maximum timestep: " << MaxTimeStep << std::endl;
		while(Currenttime <= this->totaltime)
		{
		//for (unsigned int n = 1; n <= steps; n++)
		//{
			// Write first part of the data output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			Currenttime += this->timestep;
			if(!SilentMode)
			{
				//this->Data() << this->RunSettings()->CurrentStep() << " ";
				this->Data() << Currenttime << " ";
				this->WriteStandardOutput(this->Data());
			}
			//L = LInitial//* this->timestep;
			trajectory.push_back({Currenttime, {}});
			// Propagate (use special scope to be able to dispose of the temporary vector asap)
			{
				//RungeKutta4(L, rho0, rho0, this->timestep);
				//auto rho0Eigen = ConvertAramdilloToEigen(rho0);
				//this->timestep = RungeKutta4AdaptiveTimeStep(LEigen, rho0Eigen, rho0Eigen, this->timestep, TaskMultiRadicalPairSSTimeEvo::ComputeRhoDot, {1e-4,1e-3}, InitialTimestep * 1e-2);
				//rho0 = ConvertEigenToArmadillo(rho0Eigen).col(0);
				this->timestep = RungeKutta45Armadillo(L,rho0,rho0,this->timestep,ComputeRhoDot,{1e-7,1e-6},MinTimeStep, MaxTimeStep);
			}


			// Retrieve the resulting density matrix for each spin system and output the results
			nextDimension = 0;
			for (auto i = SubSystemSpins.cbegin(); i != SubSystemSpins.cend(); i++)
			{
				// Get the superspace result vector and convert it back to the native Hilbert space
				trajectory[n].second.push_back(TrajectoryData(i->first));
				arma::cx_mat rho_result;
				arma::cx_vec rho_result_vec;
				rho_result_vec = rho0.rows(nextDimension, nextDimension + SpinSpace->second->SpaceDimensions() - 1);
				if (!SpinSpace->second->OperatorFromSuperspace(rho_result_vec, rho_result))
				{
					//this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				// Get the results
				this->GatherResults(rho_result, *(SpinSpace->first), *(SpinSpace->second), trajectory[n].second[i - SubSystemSpins.begin()].StateTrace);

				// Move on to next spin space
				nextDimension += SpinSpace->second->SpaceDimensions();
			}

			if(flush != 0)
			{
				if(n % flush == 0) //clears the trajectory vector and evaluates the yields
				{
					auto evaluatedYields = EvaluateYield(trajectory, SpinSpace->first, SubSystemSpins, SubSystemsTransitions);
					//grab last trajectory entry
					auto last = trajectory.back();
					trajectory.clear();
					trajectory.push_back(last);
					this->Log() << "Flushing trajectory vector, current time: " << Currenttime << ", yields: ";
					for(unsigned int i = 0; i < evaluatedYields.size(); i++)
					{
						RollingYield[i] += evaluatedYields[i];
						this->Log() << RollingYield[i] << " ";
					}
					this->Log() << std::endl;
					n=0;
				}
			}

			// Terminate the line in the data file after iteration through all spin systems
			this->Data() << std::endl;
			n++;
		}

		this->Log() << "Done with calculation." << std::endl;
		this->Log() << "Calculating Total Yield." << std::endl;
		//this->Data() << "\n" << "Yeilds ";
		std::vector<double> time;
		std::vector<double> yields;

		for(unsigned int i = 0; i < SpinSpace->first->States().size(); i++)
		{
			yields.push_back(0.0);
		}

		for(unsigned int i = 0; i < SubSystemSpins.size(); i++)
		{
			std::vector<std::vector<std::complex<double>>> data;
			for(auto e = trajectory.begin(); e != trajectory.end(); e++)
			{
				if(i == 0)
				{
					time.push_back(e->first);
				}
				for(auto a = e->second.begin(); a != e->second.end(); a++)
				{
					if(a->subsystem != SubSystemSpins[i].first)
					{
						continue;
					}

					data.push_back(a->StateTrace);
					break;
				}
			}

			for(auto e = SubSystemsTransitions.begin(); e != SubSystemsTransitions.end(); e++)
			{
				if(e->type != 0)
				{
					continue;
				}
				if(e->source != SubSystemSpins[i].first)
				{
					continue;
				}
				int index = 0;
				for(auto s : SpinSpace->first->States())
				{
					if(e->transition->SourceState() != s)
					{
						index++;
						continue;
					}
					break;
				}
				double rate = e->transition->Rate();
				double stateYield = 0;
				std::vector<std::complex<double>> state_data;
				for(auto a : data)
				{
					state_data.push_back(a[index]);
				}
				StateYield(rate, stateYield, state_data, time);
				yields[index] += stateYield;
			}
		}
		for (unsigned int i = 0; i < yields.size(); i++)
		{
			RollingYield[i] += yields[i];
		}
		this->Log() << "Total Yield: ";
		for(auto y : RollingYield)
		{
			this->Log() << std::setprecision(8) << y << " ";
		}
		this->Log() << "\n" << std::endl;

		return true;
	}

	std::vector<double> TaskMultiRadicalPairSSTimeEvo::EvaluateYield(const std::vector<std::pair<double,std::vector<TrajectoryData>>>& trajectory, const SpinAPI::system_ptr&SpinSpace, const std::vector<std::pair<std::string, std::vector<std::string>>>& SubSystemSpins, const std::vector<SubSystemTransition>& SubSystemsTransitions)
	{
		std::vector<double> yields;
		std::vector<double> time;
		for(unsigned int i = 0; i < SpinSpace->States().size(); i++)
		{
			yields.push_back(0.0);
		}

		for(unsigned int i = 0; i < SubSystemSpins.size(); i++)
		{
			std::vector<std::vector<std::complex<double>>> data;
			for(auto e = trajectory.begin(); e != trajectory.end(); e++)
			{
				if(i == 0)
				{
					time.push_back(e->first);
				}
				for(auto a = e->second.begin(); a != e->second.end(); a++)
				{
					if(a->subsystem != SubSystemSpins[i].first)
					{
						continue;
					}

					data.push_back(a->StateTrace);
					break;
				}
			}

			for(auto e = SubSystemsTransitions.begin(); e != SubSystemsTransitions.end(); e++)
			{
				if(e->type != 0)
				{
					continue;
				}
				if(e->source != SubSystemSpins[i].first)
				{
					continue;
				}
				int index = 0;
				for(auto s : SpinSpace->States())
				{
					if(e->transition->SourceState() != s)
					{
						index++;
						continue;
					}
					break;
				}
				double rate = e->transition->Rate();
				double stateYield = 0;
				std::vector<std::complex<double>> state_data;
				for(auto a : data)
				{
					state_data.push_back(a[index]);
				}
				StateYield(rate, stateYield, state_data, time);
				yields[index] += stateYield;
			}
		}

		return yields;
	}

	// Gathers and outputs the results from a given time-integrated density operator
	void TaskMultiRadicalPairSSTimeEvo::GatherResults(const arma::cx_mat &_rho, const SpinAPI::SpinSystem &_system, const SpinAPI::SpinSpace &_space, std::vector<std::complex<double>>& traj)
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
			//std::cout << P << std::endl;
			double tr = std::abs(arma::trace(P * _rho));
			this->Data() << tr << " ";
			//this->Data() << tr.real() << " ";
			traj.push_back(tr);
		}
	}

	void TaskMultiRadicalPairSSTimeEvo::StateYield(double _rate, double& _yeild, const std::vector<std::complex<double>>& _traj, std::vector<double>& _time)
	{
		auto f = [](double frac, double t, double kr) {return frac * std::exp(-kr * t); };
		std::vector<double> ylist;
		for(unsigned int i = 0; i < _traj.size(); i++)
		{
			ylist.push_back(f(_traj[i].real(), _time[i], _rate));
			//ylist.push_back(_rate * _traj[i].real());
			//ylist.push_back(_traj[i].real());

		}
		_yeild = _rate * simpson_integration(_time, ylist);
		//_yeild = simpson_integration(_time, ylist);
	}

    void TaskMultiRadicalPairSSTimeEvo::StateYield(double rate, double& yield, int spinsystem, int spinsystems, arma::cx_mat& projection_operator, arma::cx_vec& StateDensityVec) 
    {
		arma::cx_vec p = projection_operator.as_row();
		arma::cx_vec vec;
		vec.zeros(StateDensityVec.size());
		int TotalSize = vec.size();
		int SpinSystemSize = TotalSize /spinsystems;
		int index = 0;
		for(auto i = 0; i < spinsystems; i++)
		{
			if(i != spinsystem)
			{
				index = index + SpinSystemSize;
				continue;
			}
			
			for(int e = 0; e < SpinSystemSize; e++)
			{
				vec[e + index] = p[e];
			}

		}
		
		std::complex<double> TempYield = 0;
		for(int i = 0; i < TotalSize; i++)
		{
			TempYield += (rate * vec[i] * StateDensityVec[i]);
		}
		yield += -1 * TempYield.real();
    }

    double TaskMultiRadicalPairSSTimeEvo::simpson_integration(std::vector<double> x_list, std::vector<double> y_list)
	{
		double area = 0;
		for (unsigned int i = 0; i < x_list.size()-1; i++)
		{
			double diff = x_list[i + 1] - x_list[i];
			double ab = y_list[i] + y_list[i + 1];

			area = area + ((ab * 0.5) * diff);
		}
		return area;
	}

	bool TaskMultiRadicalPairSSTimeEvo::GenerateHamiltonian(const std::vector<SpinAPI::interaction_ptr> interactions, arma::sp_cx_mat& H, int dimension, std::shared_ptr<SpinAPI::SpinSpace> SpinSystem)
	{
		if(interactions.size() < 1)
		{
			H = arma::sp_cx_mat(dimension, dimension);
			return true;
		}

		// Get the first interaction contribution
		auto i = interactions.cbegin();
		arma::sp_cx_mat tmp;
		arma::sp_cx_mat result; 
		if (!SpinSystem->InteractionOperator((*i), result))
			return false;

		// We have already used the first interaction
		i++;
		// Loop through the rest
		for (; i != interactions.cend(); i++)
		{
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!SpinSystem->InteractionOperator((*i), tmp))
				return false;
			result += tmp;
			//tmp.print();
		}

		H = result;
		//std::cout << H << std::endl;
		return true;
	}

	bool TaskMultiRadicalPairSSTimeEvo::GenerateReactionOperator(const std::vector<SubSystemTransition> transitions, arma::sp_cx_mat& K, int dimension, std::shared_ptr<SpinAPI::SpinSpace> SpinSystem, std::string source)
	{
		if(transitions.size() < 1)
		{
			K = arma::sp_cx_mat(dimension,dimension);
			return true;
		}

		// Get the first transition contribution
		auto i = transitions.cbegin();
		arma::sp_cx_mat tmp;
		arma::sp_cx_mat result;
		if(i->source == source)
		{
			if (!SpinSystem->ReactionOperator(i->transition, result, SpinAPI::ReactionOperatorType::Unspecified))
				return false;
		}

		//result.print();

		// We have already used the first transition
		i++;

		// Loop through the rest
		for (; i != transitions.cend(); i++)
		{
			if(i->source != source)
			{
				continue;
			}
			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!SpinSystem->ReactionOperator(i->transition, tmp, SpinAPI::ReactionOperatorType::Unspecified))
				return false;
			result += tmp;
			//tmp.print();
		}

		K = result;
		//K.print();
		return true;

	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskMultiRadicalPairSSTimeEvo::WriteHeader(std::ostream &_stream, bool yield)
	{
		_stream << "Step ";
		if(!yield)
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
	bool TaskMultiRadicalPairSSTimeEvo::Validate()
	{
		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;

		// Get timestep
		if (this->Properties()->Get("timestep", inputTimestep))
		{
			if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
			{
				this->timestep = inputTimestep;
				this->OriginalTimestep = inputTimestep;
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

		ValidateSubSystems();

		return true;
	}

	bool TaskMultiRadicalPairSSTimeEvo::ValidateSubSystems()
	{
		int SubSystemsNum = 0;
		if(!this->Properties()->Get("subsystems", SubSystemsNum))
		{
			this->Log() << "Error: Number of sub systems not specified. Correct syntax is e.g. subsystems: 2" << std::endl;
			return false;
		}

		std::string SubSystemNames; 
		if(!this->Properties()->Get("subsystemnames", SubSystemNames))
		{
			this->Log() << "Note: subsystemnames not specified. Asssuming names are system1, system2, system3...." << std::endl;
		}

		std::vector<std::string> SubSystems;
		{
			std::stringstream ss(SubSystemNames);
			while(ss.good())
			{
				std::string Name;
				std::getline(ss,Name, ',');
				std::transform(Name.begin(), Name.end(), Name.begin(), [](unsigned char c){return std::tolower(c);});
				SubSystems.push_back(Name);
			}
		}

		//Verify that the subsytems are valid by checking if all the objects have been loaded
		int num = 0;
		for(auto system : SubSystems)
		{
			std::vector<std::string> SubSystemTemp;
			std::string SubSytemDefinition = "";
			if(!this->Properties().get()->Get(system, SubSytemDefinition))
			{
				this->Log() << "Error: Failed to Load SubSystem " << system << std::endl;
				std::cout << "Error: Failed to Load SubSystem " << system << std::endl;
				num++;
				continue;
			}
		}
		return true;
	}
    
    arma::cx_vec TaskMultiRadicalPairSSTimeEvo::ComputeRhoDot(double t, arma::sp_cx_mat &L, arma::cx_vec &K, arma::cx_vec RhoNaught)
    {
        arma::cx_vec ReturnVec(L.n_rows);
		RhoNaught = RhoNaught + K;
		ReturnVec = L * RhoNaught;
		return ReturnVec;
    }

    bool TaskMultiRadicalPairSSTimeEvo::CalcYieldOnly(arma::sp_cx_mat& L, arma::cx_vec& RhoNaught, arma::cx_vec& ReturnVec)
    {
		//RhoNaught = BiCGSTAB(L, RhoNaught, PreconditionerType::None);
		//return true;

		arma::cx_mat DenseL = arma::conv_to<arma::cx_mat>::from(L);
		bool solution = arma::solve(ReturnVec, DenseL, RhoNaught);
		if(solution)
			return true;

		this->Log() << "Error: Failed to find solution" << std::endl;
		std::cout << "Error: Failed to find solution" << std::endl;
		return false;
	}
}
