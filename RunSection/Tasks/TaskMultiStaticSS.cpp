/////////////////////////////////////////////////////////////////////////
// TaskMultiStaticSSimplementation (RunSection module)

// -- Multi-system version: Allows transitions between SpinSystems --
// -- When applicable uses a Tridagonal Block Solver instead of the Armadillo solver --
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskMultiStaticSS.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "ObjectParser.h"
#include "Utility.h"

#include <chrono>

namespace RunSection
{
	// -----------------------------------------------------
	// TaskMultiStaticSS Constructors and Destructor
	// -----------------------------------------------------
	TaskMultiStaticSS::TaskMultiStaticSS(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4),
																																reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskMultiStaticSS::~TaskMultiStaticSS()
	{
	}
	// -----------------------------------------------------
	// TaskMultiStaticSS protected methods
	// -----------------------------------------------------
	bool TaskMultiStaticSS::RunLocal()
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
		int DimensionSize = 0;
		bool SameDimension = true;
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			auto space = std::make_shared<SpinAPI::SpinSpace>(*(*i));

			// We are using superoperator space, and need the total dimensions
			space->UseSuperoperatorSpace(true);
			space->SetReactionOperatorType(this->reactionOperators);
			int Dimension = space->SpaceDimensions();
			dimensions += Dimension;
			if (Dimension != DimensionSize && DimensionSize != 0)
			{
				SameDimension = false;
			}
			else
			{
				DimensionSize = Dimension;
			}

			// Make sure to save the newly created spin space
			spaces.push_back(std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>(*i, space));
		}

		// Now, create a matrix to hold the Liouvillian superoperator and the initial state
		arma::sp_cx_mat L(dimensions, dimensions);
		arma::cx_vec rho0(dimensions);
		int nextDimension = 0; // Keeps track of the dimension where the next spin space starts

		// Loop through the systems again to fill this matrix and vector
		std::vector<arma::sp_cx_mat> Blocks;
		bool BlockCache = false;
		if(systems.size() > 2 && SameDimension) //Testing has showed that for less than 3 spin systems the BlockTridiagonal method is slower than the standard armadillo solver
		{
			int TriDiagonalBlocks = 3 * systems.size();
			arma::sp_cx_mat ZeroMatrix(DimensionSize,DimensionSize);
			for (int i = 0; i < TriDiagonalBlocks; i++)
			{
				Blocks.push_back(ZeroMatrix);
			}
			BlockCache = true;

		}
		//int space = 0;
		for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
		{

			// Make sure we have an initial state
			auto initial_states = i->first->InitialState();
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
			if (!i->second->Hamiltonian(H))
			{
				this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}
			L.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) = arma::cx_double(0.0, -1.0) * H;

			// Then get the reaction operators
			arma::sp_cx_mat K;
			if (!i->second->TotalReactionOperator(K))
			{
				this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operators for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}

			L.submat(nextDimension, nextDimension, nextDimension + i->second->SpaceDimensions() - 1, nextDimension + i->second->SpaceDimensions() - 1) -= K;

			if(BlockCache)
			{
				arma::sp_cx_mat TempMat = arma::cx_double(0.0, -1.0) * (H + K);
				Blocks[3*(i-spaces.begin()) + 1] = TempMat;
			}

			// Obtain the creation operators - note that we need to loop through the other SpinSystems again to find transitions leading into the current SpinSystem
			int nextCDimension = 0; // Similar to nextDimension, but to keep track of first dimension for this other SpinSystem
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
								return false;
							}

							// Put it into the total Liouvillian:
							//  - The row should be that of the current spin space (the target space)
							//  - The column should be that of the source spin space (the spin system containing the Transition object)
							arma::sp_cx_mat Cmod = C * (*t)->Rate();
							int dim1 = nextDimension + i->second->SpaceDimensions();
							int dim2 = nextCDimension + j->second->SpaceDimensions();
							L.submat(nextDimension, nextCDimension, dim1 - 1, dim2 - 1) += Cmod;
							
							if(!SameDimension)
								continue;
							if(!BlockCache)
								continue;
							if(std::abs(nextCDimension / DimensionSize - nextDimension / DimensionSize) > 1)
							{
								BlockCache = false;
								continue;
							}

							int col = j - spaces.begin();
							int row = i - spaces.begin();
							int BlockIndex = (3*row) + 1 + (col - row);
							Blocks[BlockIndex] = Cmod;							
						}
					}
				}

				// Move on to check next spin system for transitions into the current spin space
				nextCDimension += j->second->SpaceDimensions();
			}

			// Move on to next spin space
			nextDimension += i->second->SpaceDimensions();
		}

		arma::cx_vec result = arma::cx_vec(rho0.n_rows);
		std::cout << rho0 << std::endl;
		this->Log() << "Ready to perform calculation." << std::endl;
		if(BlockCache)
		{
			int BlockSize = rho0.n_rows / spaces.size();
			this->Log() << "Using Tridiagonal solver" << std::endl;
			result = ThomasBlockSolver(L,rho0,BlockSize,Blocks);
		}
		else
		{
			this->Log() << "Using Armadillo solver" << std::endl;
			arma::cx_mat L_Dense = arma::conv_to<arma::cx_mat>::from(L);
			//result = arma::solve(L_Dense, rho0, arma::solve_opts::force_approx);
			std::vector<int> dim = {};
			for(auto i = spaces.cbegin(); i != spaces.cend(); i++)
			{
				dim.push_back(i->second->SpaceDimensions());
			}
			BlockSolver(L, rho0, dim, result);
		}
		rho0 = result;
		this->Log() << "Done with calculation." << std::endl;

		//Obtain results
		arma::cx_mat P;
		double SumYield = 0.0;

		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->WriteStandardOutput(this->Data());
		
		// There are two result modes - either write results per transition or for each defined state
		if (this->productYieldsOnly)
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
					double Yield = (*j)->Rate() * std::abs(arma::trace(P * rho_result));
					SumYield += Yield;
					// Return the yield for this transition
					this->Data() << Yield << " ";
				}
			}
			this->Data() << SumYield << " ";
			SumYield = 0;
		}
		else
		{
			int NextDimension = 0;
			std::cout << rho0 << std::endl;
			for (auto i = spaces.cbegin(); i != spaces.cend(); i++)
			{
				arma::cx_mat rho_result;
				int TempDimension = NextDimension + i->second->SpaceDimensions();
				arma::cx_vec rho_result_vec = rho0.rows(NextDimension,TempDimension-1);
				NextDimension = TempDimension;
				std::cout << rho_result_vec << std::endl;
				if (!i->second->OperatorFromSuperspace(rho_result_vec, rho_result))
				{
					this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				auto states = (*i).first->States();
				for(auto j = states.cbegin(); j != states.cend(); j++)
				{
					if (!i->second->GetState((*j), P))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << (*i).first->Name() << "\"." << std::endl;
						continue;
					}

					// Return the yield for this state - note that no reaction rates are included here.
					std::cout << P * rho_result << std::endl;
					this->Data() << std::abs(arma::trace(P * rho_result)) << " ";
				}
			}
		}
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskMultiStaticSS::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			if(this->productYieldsOnly)
			{
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
	bool TaskMultiStaticSS::Validate()
	{
		this->productYieldsOnly = false;
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
