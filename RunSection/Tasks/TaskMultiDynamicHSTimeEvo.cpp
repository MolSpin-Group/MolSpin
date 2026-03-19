/////////////////////////////////////////////////////////////////////////
// TaskMultiDynamicHSTimeEvo implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskMultiDynamicHSTimeEvo.h"
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
	TaskMultiDynamicHSTimeEvo::TaskMultiDynamicHSTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection),
																																  timestep(0.01), totaltime(1.0e+4), outputstride(1)
	{
	}

	TaskMultiDynamicHSTimeEvo::~TaskMultiDynamicHSTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSS protected methods
	// -----------------------------------------------------
	bool TaskMultiDynamicHSTimeEvo::RunLocal()
	{
		this->Log() << "Running method DynamicHS-MultiSystem." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems to obtain SpinSpace objects
		auto systems = this->SpinSystems();
		std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> spaces;
		//unsigned int dimensions = 0;
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Create a SpinSpace for the spin system
			auto space = std::make_shared<SpinAPI::SpinSpace>(*(*i));
			space->UseSuperoperatorSpace(false);
			space->SetTime(0.0);
			//dimensions += space->SpaceDimensions();
			spaces.push_back(std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>(*i, space));
		}

		// We need a Hamiltonian, total reaction operator, creation operators, and a current state for each spin system
		std::vector<arma::cx_mat> H;
		std::vector<arma::cx_mat> K;
		std::vector<arma::cx_mat> rho;
		std::vector<arma::sp_cx_mat> C; // Reaction operators in target space (produced by creation operators acting on density operators, or similar means). Note: Sparse matrices!

		// Dynamic parts
		std::vector<arma::sp_cx_mat> dH;
		std::vector<arma::sp_cx_mat> dK;

		bool timedependentInteractions = false;
		bool timedependentTransitions = false;

		// Loop through the systems again to fill these matrices
		for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
		{
			// Make sure we have an initial state
			auto initial_states = i->first->InitialState();
			arma::cx_mat rho0;
			if (initial_states.size() < 1)
			{
				this->Log() << "Note: No initial state specified for spin system \"" << i->first->Name() << "\", setting the initial state to zero." << std::endl;
				rho0 = arma::zeros<arma::cx_mat>(i->second->SpaceDimensions(), i->second->SpaceDimensions());
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
						rho0 = tmp_rho0;
					else
						rho0 += tmp_rho0;
				}
				rho0 /= arma::trace(rho0); // The density operator should have a trace of 1
			}

			// Put the created initial state density operator into the list
			rho.push_back(rho0);

			// Next, get the Hamiltonian
			arma::cx_mat tmp_H;
			arma::sp_cx_mat tmp_dH = arma::sp_cx_mat(i->second->SpaceDimensions(), i->second->SpaceDimensions());
			if (!i->second->StaticHamiltonian(tmp_H) || (i->second->HasTimedependentInteractions() && !i->second->DynamicHamiltonian(tmp_dH)))
			{
				this->Log() << "ERROR: Failed to obtain the Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			//H.push_back(arma::cx_double(0.0, -1.0) * tmp_H);
			//dH.push_back(arma::cx_double(0.0, -1.0) * tmp_dH);
			H.push_back(tmp_H);
			dH.push_back(tmp_dH);

			// Then get the reaction operators (decay part)
			arma::cx_mat tmp_K;
			arma::sp_cx_mat tmp_dK = arma::sp_cx_mat(i->second->SpaceDimensions(), i->second->SpaceDimensions());
			if (!i->second->StaticTotalReactionOperator(tmp_K) || (i->second->HasTimedependentTransitions() && !i->second->DynamicTotalReactionOperator(tmp_dK)))
			{
				this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operators for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			K.push_back(tmp_K);
			dK.push_back(tmp_dK);

			// Allocate matrices for the creation operators
			C.push_back(arma::sp_cx_mat(i->second->SpaceDimensions(), i->second->SpaceDimensions()));

			// Check whether we have any time-dependent interactions or transitions
			timedependentInteractions |= i->second->HasTimedependentInteractions();
			timedependentTransitions |= i->second->HasTimedependentTransitions();
		}

		//Determine best propagator based on the stiffness of the system
		if(this->prop == Propagator::Default)
		{
			std::vector<Propagator> suitableProps;
			for (unsigned int i = 0; i < spaces.size(); i++)
			{
				arma::cx_mat HPK = std::complex(0.0, -1.0) * (H[i] + std::complex(0.0, 0.5) * K[i]);
				this->DetermineBestPropagator(HPK);
				suitableProps.push_back(this->prop);
			}
			//choose most suitable propagator among all systems (if they differ, choose the one that is best for the stiffest system)
			std::sort(suitableProps.begin(), suitableProps.end(), [](Propagator a, Propagator b) { 
				int a_int = static_cast<int>(a);
				int b_int = static_cast<int>(b);
				if(a_int > b_int)
					return false;
				else
					return true;
			 });
			std::vector<int> gaps = {};
			int gap = 0;
			int currentProp = static_cast<int>(suitableProps[0]);
			for(auto p = suitableProps.cbegin(); p != suitableProps.cend(); p++)
			{
				if(static_cast<int>(*p) != currentProp)
				{
					gaps.push_back(gap);
					gap = 0;
					currentProp = static_cast<int>(*p);
				}
				gap++;
			}
			gaps.push_back(gap);
			int maxGap = 0;
			int maxGapIndex = 0;
			int totalGap = 0;
			for(unsigned int i = 0; i < gaps.size(); i++)
			{
				if(gaps[i] > maxGap)
				{
					maxGap = gaps[i];
					maxGapIndex = i;
				}
			}
			for(int i = 0; i < maxGapIndex; i++)
			{
				totalGap += gaps[i];
			}
			this->prop = suitableProps[totalGap];
			this->Log() << "Selected propagator: " << PropogatorToString(this->prop) << std::endl;
			this->propogator_cached = true;
		}

		// ---------------------------------------------------------------------------------------------------
		// Perform the calculation
		// ---------------------------------------------------------------------------------------------------
		this->Log() << "Ready to perform calculation." << std::endl;
		this->OutputResults(spaces, rho, 0); // Write initial state results
		//unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
		SpinAPI::SpinSpace::PropParam propParam = this->GetTimeAdaptiveProperties(this->timestep);
		double currentTime = 0.0;
		double TotalTime = this->totaltime;

		double dw = 0.9;

		propParam.f2 = 1.5;
		propParam.f1 = 0.5;
		propParam.safety = 0.5;
		propParam.UsePrefactor = true;

		std::vector<double> LocalTime;
		std::vector<SpinAPI::SpinSpace::PropParam> LocalPropParams;
		std::vector<std::vector<double>> LocalTimeSteps;
		std::vector<Buffer<arma::cx_mat>> HistoryBuffer;// = Buffer<arma::cx_mat>(5,ValType::matrix);

		//run quick propogation just to check the initial timestep is suitable, if not reduce it until it is
		if(this->prop == Propagator::Krylov)
		{
			double minumum = std::numeric_limits<double>::max();
			for(unsigned int i = 0; i < spaces.size(); i++)
			{
				LocalTime.push_back(0.0);
				LocalPropParams.push_back(propParam);
				HistoryBuffer.push_back(Buffer<arma::cx_mat>(5,ValType::matrix));
				HistoryBuffer[i].push(0.0, rho[i]);
			//	arma::cx_mat H = H[i] + K[i] + dH[i] + dK[i];
			//	auto r = spaces[i].second->TimeAdaptiveKrylovGeneral(H, rho[i], std::complex<double>(this->timestep), 30, H.n_rows, propParam, true);
			//	//forgotten that rho is a matrix not a vector
			//	if(r.step_accepted == false)
			//	{
			//		this->Log() << "Initial timestep of " << this->timestep << " is too large for the Krylov propagator";
			//		this->Log() << ", accepted timestep was: " << r.timestep_used << std::endl;
			//	}
			//	if(r.timestep_used < minumum)
			//	{
			//		minumum = r.timestep_used;
			//	}
			}
			//this->timestep = minumum;
			this->Log() << "Using initial timestep of " << this->timestep << " for the Krylov propagator." << std::endl;
			LocalTimeSteps.push_back({});
			for(unsigned int i = 0; i < spaces.size(); i++)
			{
				LocalTimeSteps[0].push_back(this->timestep);
			}
		}

		auto minTimeStep = [](std::vector<double> timesteps) {
			std::sort(timesteps.begin(), timesteps.end());
			return timesteps[0];
		};

		int step = 0;
		while(currentTime < TotalTime)
		//for (unsigned int n = 0; n < steps; n++)
		{
			//Obtain creation operators
			this->GetCreationOperators(spaces, C, rho); //this is a issue (as we have to work out how to propogate this without killing the system)
//
			//// Propagate
			//// this->AdvanceStep_AsyncLeapfrog(spaces, H, dH, K, dK, C, rho);
			//this->AdvanceStep_RK4(spaces, H, dH, K, dK, C, rho);
//
			//// Print results
			//if (n % this->outputstride == 0)
			//	this->OutputResults(spaces, rho, n + 1);
//
			//// Update time-dependent interactions or reaction operators
			//if ((timedependentInteractions || timedependentTransitions) && n < steps - 1)
			//	this->UpdateTimeDependences(spaces, dH, dK, n + 1);

			//Plan - propogate each system independently, each with there own adaptive timestep and then interpolate after each step 
			//The global max timestep should be capped so that each system is guaranteed to all be in a given global window at the end of each step, 
			//the size of the window is determined by the smallest timestep, e.g. if the current time is 1.0 and the smallest timstep is 0.5 then the middle of the window is 1.5 and the max is 1.95 (90% of the timestep), likewise the minimum is 1.05
			//this ensures that all systems are within one timestep of each at the end of the step 

			//we have to assume the first timestep used is good 
			this->timestep = minTimeStep(LocalTimeSteps[step]);
			LocalTimeSteps.push_back({});
			double midpoint = currentTime + this->timestep;
			double maxTime = midpoint + dw/2.0 * this->timestep;
			double minTime = midpoint - dw/2.0 * this->timestep;
			std::array<double,3> timeWindow = {minTime, midpoint, maxTime};
			for(unsigned int i = 0; i < spaces.size(); i++)
			{
				double ctime = currentTime;
				double max = timeWindow[2] - ctime;
				double min = timeWindow[0] - ctime;
				double maxGrowth = max / LocalTimeSteps[step][i];
				double minGrowth = min / LocalTimeSteps[step][i];
				LocalPropParams[i].f2 = std::min(propParam.f2, maxGrowth);
				LocalPropParams[i].f1 = std::max(propParam.f1, minGrowth);

				
				arma::cx_mat H_eff, H_eff_conj;
				{
					arma::cx_mat H_base = H[i];
					if(timedependentInteractions)
					{
						H_base = H_base + dH[i];
					}
					arma::cx_mat K_base = K[i];
					if(timedependentTransitions)
					{
						K_base = K_base + dK[i];
					}

					H_eff = H_base + std::complex(0.0, 0.5) * K_base;
					//H_eff = std::complex(0.0, -1.0) * H_eff;
					H_eff_conj = H_base + std::complex(0.0,-0.5) * K_base;
				}

				auto GeneratorFunc = [&](const arma::sp_cx_mat& H, const arma::cx_mat& V) {
					arma::cx_mat Z;
					Z = std::complex<double>(0.0,-1.0) * (H * V - V * H_eff_conj);
					return Z;
				};
				arma::cx_mat prop_state = rho[i];

				//ETD2RK - a second order RK method with kyrlov
				{
					auto r = spaces[i].second->TimeAdaptiveKrylovGeneral(H_eff, prop_state, std::complex<double>(LocalTimeSteps[step][i]), 30, H_eff.n_rows, LocalPropParams[i], true);//, GeneratorFunc);
					{
						arma::cx_mat test = r.result;// * r.result.t();
						arma::cx_mat exp1 = arma::expmat(std::complex<double>(0.0,-1.0) * H_eff * std::complex<double>(LocalTimeSteps[step][i]));
						///arma::cx_mat exp2 = arma::expmat(std::complex<double>(0.0,1.0) * H_eff_conj * std::complex<double>(LocalTimeSteps[step][i]));
						arma::cx_mat test2 = exp1 * prop_state * exp1.t();
						std::cout << test << std::endl;
						std::cout << test2 << std::endl;
						double err = arma::norm(test-test2,"fro");
						double norm_test = arma::norm(test,"fro");
						std::cout << err << " : " << (err/norm_test) << std::endl;
					} 
					//midpoint predictor
					this->GetCreationOperators(spaces, C[i], prop_state, i);
					int p = r.result.n_cols;
					int krydim = r.krybasis.n_cols / p;
					arma::cx_mat p1;
					{
						arma::sp_cx_mat Csp = C[i];
						arma::cx_mat C = arma::conv_to<arma::cx_mat>::from(Csp);
						arma::cx_mat b1 = spaces[i].second->project_block(C,r.krybasis,krydim,p);
						arma::cx_mat y1 = r.phi1 * b1;
						p1 = spaces[i].second->reconstruct_block(y1,r.krybasis,krydim,p);
					}
					std::cout << r.result << std::endl;
					arma::cx_mat midpoint_mat = r.result + LocalTimeSteps[step][i] * p1;
					std::cout << midpoint_mat << std::endl;
					arma::sp_cx_mat C_copy = C[i];
					this->GetCreationOperators(spaces, C[i], midpoint_mat*midpoint_mat.t(), i);
					{
						arma::sp_cx_mat Csp = C[i]-C_copy;
						arma::cx_mat C = arma::conv_to<arma::cx_mat>::from(Csp);
						arma::cx_mat b1 = spaces[i].second->project_block(C,r.krybasis,krydim,p);
						arma::cx_mat y1 = r.phi2 * b1;
						p1 = spaces[i].second->reconstruct_block(y1,r.krybasis,krydim,p);
					}
					prop_state = midpoint_mat + LocalTimeSteps[step][i] * p1;
					std::cout << prop_state << std::endl;
					arma::cx_mat err_mat = prop_state - midpoint_mat;
					std::cout << err_mat << std::endl;
					double err = arma::norm(err_mat, "fro");
					//err = err + 1.0;

					prop_state = prop_state * prop_state.t();

					LocalTimeSteps[step+1].push_back(r.timestep);
					LocalTime[i] += r.timestep_used;
					HistoryBuffer[i].push(LocalTime[i],prop_state);
				}


				//leapfrog procedure 
				{
					//full timestep
					//arma::cx_mat prop_state_full = prop_state;
					//prop_state_full = prop_state + (LocalTimeSteps[step][i] / 2.0) * C[i];
					//auto r = spaces[i].second->TimeAdaptiveKrylovGeneral(H_eff, prop_state_full, std::complex<double>(LocalTimeSteps[step][i]), 30, H_eff.n_rows, LocalPropParams[i], true, GeneratorFunc);
					//prop_state_full += r.result;
					//this->GetCreationOperators(spaces, C[i], prop_state_full, i);
					//prop_state_full += (LocalTimeSteps[step][i] / 2.0) * C[i];

					////half timestep
					//arma::cx_mat prop_state_half = prop_state;
					//prop_state_half = prop_state + (LocalTimeSteps[step][i] / 4.0) * C[i];
					//auto r2 = spaces[i].second->TimeAdaptiveKrylovGeneral(H_eff, prop_state_half, std::complex<double>(LocalTimeSteps[step][i]), 30, H_eff.n_rows, LocalPropParams[i], true, GeneratorFunc);
					//prop_state_half += r2.result;
					//this->GetCreationOperators(spaces, C[i], prop_state_half, i);
					//prop_state_half += (LocalTimeSteps[step][i] / 4.0) * C[i];
					//this->GetCreationOperators(spaces, C[i], prop_state_half, i);
					//prop_state_half += (LocalTimeSteps[step][i] / 4.0) * C[i];
					//r2 = spaces[i].second->TimeAdaptiveKrylovGeneral(H_eff, prop_state_half, std::complex<double>(LocalTimeSteps[step][i]), 30, H_eff.n_rows, LocalPropParams[i], true, GeneratorFunc);

				}
				//auto r = spaces[i].second->TimeAdaptiveKrylovGeneral(H_eff, prop_state, std::complex<double>(LocalTimeSteps[step][i]), 30, H_eff.n_rows, LocalPropParams[i], true, GeneratorFunc);
				//LocalTimeSteps[step+1].push_back(r.timestep);
				//LocalTime[i] += r.timestep_used;
				//HistoryBuffer[i].push(LocalTime[i],r.result);
			}
			//get minimum LocalTime and set that to current time, then interpolate all states back to that time
			double minLocalTime = timeWindow[2] + 1.0;
			for (unsigned int i = 0; i < LocalTime.size(); i++)
			{
				minLocalTime = std::min(minLocalTime, LocalTime[i]);
			}
			currentTime = minLocalTime;
			for(int i = 0; i < spaces.size(); i++)
			{
				if(LocalTime[i] == currentTime)
					rho[i] = HistoryBuffer[i].rhoPoints.back();
				else
					rho[i] = CubicSplineInterpolation(HistoryBuffer[i],currentTime);
				HistoryBuffer[i].replace(currentTime,rho[i]);
			}

			// Print results
			if (step % this->outputstride == 0)
				this->OutputResults(spaces, rho, currentTime);

			if ((timedependentInteractions || timedependentTransitions))
				this->UpdateTimeDependences(spaces, dH, dK, currentTime);

			step += 1;

		}
		// ---------------------------------------------------------------------------------------------------

		this->Log() << "Done with calculation." << std::endl;

		return true;
	}

	// -----------------------------------------------------
	// The method that prepares all the creation operators
	void TaskMultiDynamicHSTimeEvo::GetCreationOperators(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &_spaces,
														 std::vector<arma::sp_cx_mat> &_C, const std::vector<arma::cx_mat> &_rho)
	{
		// Reset all creation operators
		for (unsigned int i = 0; i < _spaces.size(); i++)
			_C[i] = arma::sp_cx_mat(_spaces[i].second->SpaceDimensions(), _spaces[i].second->SpaceDimensions());

		// Obtain the new creation operators
		for (unsigned int i = 0; i < _spaces.size(); i++)
		{
			// Loop through all transitions in each spin system
			for (auto j = _spaces[i].first->transitions_cbegin(); j != _spaces[i].first->transitions_cend(); j++)
			{
				// If the transition has a target spin system
				if ((*j)->Target() != nullptr && (*j)->TargetState() != nullptr)
				{
					// Find the index of the target system
					auto target = _spaces.size();
					for (unsigned int n = 0; n < _spaces.size(); n++)
					{
						if (_spaces[n].first == (*j)->Target())
						{
							target = n;
							break;
						}
					}

					// Check that the target system was found
					if (target >= _spaces.size())
					{
						this->Log() << "ERROR: Failed to properly generate creation operators." << std::endl;
						return;
					}

					// Get the creation rate
					arma::cx_mat P;
					if (!_spaces[i].second->GetState((*j)->SourceState(), P))
					{
						this->Log() << "Failed to obtain projection matrix onto source state of transition \"" << (*j)->Name() << "\"." << std::endl;
						continue;
					}
					double cRate = std::abs(arma::trace(P * _rho[i]));

					// Generate the creation operator
					arma::sp_cx_mat sP;
					if (!_spaces[target].second->ReactionTargetOperator((*j), cRate, sP))
					{
						this->Log() << "Failed to obtain reaction operator for target state of transition \"" << (*j)->Name() << "\"." << std::endl;
						continue;
					}
					_C[target] += sP;
				}
			}
		}
	}

    void TaskMultiDynamicHSTimeEvo::GetCreationOperators(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &spaces, arma::sp_cx_mat &C, const arma::cx_mat &rho, int i)
    {
		C = arma::sp_cx_mat(spaces[i].second->SpaceDimensions(), spaces[i].second->SpaceDimensions());

		// Loop through all transitions in each spin system
		for (auto j = spaces[i].first->transitions_cbegin(); j != spaces[i].first->transitions_cend(); j++)
		{
			// If the transition has a target spin system
			if ((*j)->Target() != nullptr && (*j)->TargetState() != nullptr)
			{
				// Find the index of the target system
				auto target = spaces.size();
				for (unsigned int n = 0; n < spaces.size(); n++)
				{
					if (spaces[n].first == (*j)->Target())
					{
						target = n;
						break;
					}
				}
				// Check that the target system was found
				if (target >= spaces.size())
				{
					this->Log() << "ERROR: Failed to properly generate creation operators." << std::endl;
					return;
				}
				// Get the creation rate
				arma::cx_mat P;
				if (!spaces[i].second->GetState((*j)->SourceState(), P))
				{
					this->Log() << "Failed to obtain projection matrix onto source state of transition \"" << (*j)->Name() << "\"." << std::endl;
					continue;
				}
				double cRate = std::abs(arma::trace(P * rho));
				// Generate the creation operator
				arma::sp_cx_mat sP;
				if (!spaces[target].second->ReactionTargetOperator((*j), cRate, sP))
				{
					this->Log() << "Failed to obtain reaction operator for target state of transition \"" << (*j)->Name() << "\"." << std::endl;
					continue;
				}
				C += sP;
			}
		}
    }

    // -----------------------------------------------------
	// The timestep function
	void TaskMultiDynamicHSTimeEvo::AdvanceStep_AsyncLeapfrog(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &_spaces,
															  const std::vector<arma::cx_mat> &_H, const std::vector<arma::sp_cx_mat> &_dH,
															  const std::vector<arma::cx_mat> &_K, const std::vector<arma::sp_cx_mat> &_dK,
															  const std::vector<arma::sp_cx_mat> &_C, std::vector<arma::cx_mat> &_rho)
	{
		// Loop through all spaces and propagate each individually
		for (unsigned int i = 0; i < _spaces.size(); i++)
		{
			arma::cx_mat dRho = arma::cx_double(0.0, -1.0) * ((_H[i] + _dH[i]) * _rho[i] - _rho[i] * (_H[i] + _dH[i])) - ((_K[i] + _dK[i]) * _rho[i] + _rho[i] * (_K[i] + _dK[i])) + _C[i];
			_rho[i] += dRho * this->timestep / 2.0;
			dRho = (arma::cx_double(0.0, -1.0) * ((_H[i] + _dH[i]) * _rho[i] - _rho[i] * (_H[i] + _dH[i])) - ((_K[i] + _dK[i]) * _rho[i] + _rho[i] * (_K[i] + _dK[i])) + _C[i]);
			_rho[i] += dRho * this->timestep / 2.0;
		}
	}

	void TaskMultiDynamicHSTimeEvo::AdvanceStep_RK4(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &_spaces,
													const std::vector<arma::cx_mat> &_H, const std::vector<arma::sp_cx_mat> &_dH,
													const std::vector<arma::cx_mat> &_K, const std::vector<arma::sp_cx_mat> &_dK,
													const std::vector<arma::sp_cx_mat> &_C, std::vector<arma::cx_mat> &_rho)
	{
		for (unsigned int i = 0; i < _spaces.size(); i++)
		{
			arma::cx_mat k1 = ComputeRhoDot(_H[i], _dH[i], _K[i], _dK[i], _C[i], _rho[i]);
			arma::cx_mat k2 = ComputeRhoDot(_H[i], _dH[i], _K[i], _dK[i], _C[i], _rho[i] + 0.5 * this->timestep * k1);
			arma::cx_mat k3 = ComputeRhoDot(_H[i], _dH[i], _K[i], _dK[i], _C[i], _rho[i] + 0.5 * this->timestep * k2);
			arma::cx_mat k4 = ComputeRhoDot(_H[i], _dH[i], _K[i], _dK[i], _C[i], _rho[i] + this->timestep * k3);

			_rho[i] += (this->timestep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
		}
	}

	arma::cx_mat TaskMultiDynamicHSTimeEvo::ComputeRhoDot(const arma::cx_mat &H, const arma::sp_cx_mat &dH,
														  const arma::cx_mat &K, const arma::sp_cx_mat &dK,
														  const arma::sp_cx_mat &C, const arma::cx_mat &rho)
	{
		return arma::cx_double(0.0, -1.0) * ((H + dH) * rho - rho * (H + dH)) - ((K + dK) * rho + rho * (K + dK)) + C;
	}

	// -----------------------------------------------------
	// Writes the output for a timestep
	void TaskMultiDynamicHSTimeEvo::OutputResults(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &_spaces,
												  const std::vector<arma::cx_mat> &_rho, const double ctime)
	{
		// Obtain the results
		arma::cx_mat PState;
		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->Data() << (ctime) << " ";
		this->WriteStandardOutput(this->Data());
		for (unsigned int i = 0; i < _spaces.size(); i++)
		{
			auto states = _spaces[i].first->States();
			for (auto j = states.cbegin(); j != states.cend(); j++)
			{
				if (!_spaces[i].second->GetState((*j), PState))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\"." << std::endl;
					continue;
				}

				this->Data() << std::abs(arma::trace(PState * _rho[i])) << " ";
			}
		}

		// Terminate the line in the data file after iteration through all steps
		this->Data() << std::endl;
	}

	// -----------------------------------------------------
	// Method to update time-dependent interactions and reactions
	void TaskMultiDynamicHSTimeEvo::UpdateTimeDependences(const std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> &_spaces,
														  std::vector<arma::sp_cx_mat> &_H, std::vector<arma::sp_cx_mat> &_K, const double ctime)
	{
		// Handle each spin space
		for (unsigned int i = 0; i < _spaces.size(); i++)
		{
			// First update the time
			_spaces[i].second->SetTime(ctime);

			// Do we need to update the interactions?
			if (_spaces[i].second->HasTimedependentInteractions())
			{
				if (!_spaces[i].second->DynamicHamiltonian(_H[i]))
				{
					this->Log() << "ERROR: Failed to update the Hamiltonian for spin system \"" << _spaces[i].first->Name() << "\"!" << std::endl;
					return;
				}
				_H[i] *= arma::cx_double(0.0, -1.0);
			}

			// Do we need to update the transitions?
			if (_spaces[i].second->HasTimedependentTransitions())
			{
				if (!_spaces[i].second->DynamicTotalReactionOperator(_K[i]))
				{
					this->Log() << "ERROR: Failed to update matrix representation of the reaction operators for spin system \"" << _spaces[i].first->Name() << "\"!" << std::endl;
					return;
				}
			}
		}
	}

	// -----------------------------------------------------
	// Writes the header of the data file (but can also be passed to other streams)
	void TaskMultiDynamicHSTimeEvo::WriteHeader(std::ostream &_stream)
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

	// -----------------------------------------------------
	// Task validation method
	bool TaskMultiDynamicHSTimeEvo::Validate()
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

		this->prop = Propagator::Default;
		this->propogator_cached = false;

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

		// Get the output stride (i.e. write output for any n'th step, where n is the stride) - Only available for TimeEvolution calculations!
		if (this->Properties()->Get("outputstride", this->outputstride))
		{
			if (this->outputstride == 0)
			{
				this->Log() << "Cannot use outputstride 0, must be a non-zero positive integer! Using default value of 1." << std::endl;
				this->outputstride = 1;
			}
		}

		return true;
	}
	// -----------------------------------------------------
}
