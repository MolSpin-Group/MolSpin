/////////////////////////////////////////////////////////////////////////
// BasicTask implementation (RunSection module)
//
// NOTE: This is the implementation for the base class used for task
// classes. Be careful when changing this class implementation as it will
// affect all task classes!
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "ObjectParser.h"
#include "RunSection.h"
#include "BasicTask.h"
#include "Settings.h"
#include "Spin.h"
#include "Interaction.h"
#include "Transition.h"
#include "SpinSystem.h"

namespace RunSection
{

	arma::cx_vec operator+ (const std::pair<int, arma::cx_vec> &a, const std::pair<int, arma::cx_vec> &b) {
		return a.second + b.second;
	};

	double operator+ (const std::pair<int, double> &a, const std::pair<int, double> &b) {
		return a.second + b.second;
	};

    // -----------------------------------------------------
	// BasicTask Constructors and Destructor
	// -----------------------------------------------------
	BasicTask::BasicTask(const MSDParser::ObjectParser &_properties, const RunSection &_runsection) : properties(std::make_shared<MSDParser::ObjectParser>(_properties)), runsection(_runsection), output(),
																									  isValid(false), isValidated(false), scalars(nullptr), vectors(nullptr), usedScalars(), usedVectors(), name(TaskName::DEFULAT)
	{
		if (!_runsection.noCalculations)
		{
			bool append;
			std::string str;

			// Check whether a logfile is specified for the calculation method
			if (this->properties->Get("logfile", str) || this->properties->Get("log", str) || this->properties->Get("output", str))
			{
				// Check whether we should append to or overwrite previous files
				if (!this->properties->Get("append", append) && !this->properties->Get("appendlog", append))
					append = false;

				// The commandline "--append" option
				if (_runsection.overruleAppend)
				{
					append = true;
				}

				// Attempt to create the stream
				if (!this->output.SetLogFile(str, append))
				{
					// If it was not possible to create the stream, terminate
					std::cout << "ERROR: Failed to set logfile of task \"" << this->Name() << "\"!" << std::endl;
					std::cout << "----> Requested file \"" << str << "\"." << std::endl;
					std::cout << "----> Make sure that the folder exists and you have permission to write to the file." << std::endl;
					std::exit(1);
				}
			}

			// Check whether a data file is specified for the calculation method
			if (this->properties->Get("datafile", str) || this->properties->Get("data", str))
			{
				// Check whether we should append to or overwrite previous files
				if (!this->properties->Get("append", append) && !this->properties->Get("appenddata", append))
					append = false;

				// The commandline "--append" option
				if (_runsection.overruleAppend)
				{
					append = true;
				}

				// Attempt to create the stream
				if (!this->output.SetDataFile(str, append))
				{
					// If it was not possible to create the stream, terminate
					std::cout << "ERROR: Failed to set datafile of task \"" << this->Name() << "\"!" << std::endl;
					std::cout << "----> Requested file \"" << str << "\"." << std::endl;
					std::cout << "----> Make sure that the folder exists and you have permission to write to the file." << std::endl;
					std::exit(1);
				}
			}
		}

		auto settings = _runsection.GetSettings();
		if (settings != nullptr)
		{
			this->output.SetDataPrecision(settings->DataPrecision());
			this->output.SetLogPrecision(settings->LogPrecision());
		}
	}

	BasicTask::~BasicTask()
	{
	}

	// -----------------------------------------------------
	// Does some extra initialization and then calls Validate
	// -----------------------------------------------------
	bool BasicTask::DoValidation()
	{
		std::string str;

		// Get the general notification level
		MessageType msgtype = this->runsection.GetSettings()->NotificationLevel();

		// If a notification level was specified for this specific task, attempt to parse it
		if (this->properties->Get("notifications", str) || this->properties->Get("notificationlevel", str))
		{
			if (str.compare("details") == 0 || str.compare("showdetails") == 0 || str.compare("all") == 0 || str.compare("verbose") == 0)
			{
				msgtype &= ~NormalMessage;		// Reset normal message type bits
				msgtype |= MessageType_Details; // Set the bit for the requested type
			}
			else if (str.compare("normal") == 0 || str.compare("standard") == 0)
			{
				msgtype &= ~NormalMessage;	   // Reset normal message type bits
				msgtype |= MessageType_Normal; // Set the bit for the requested type
			}
			else if (str.compare("sparse") == 0 || str.compare("important") == 0 || str.compare("importantonly") == 0)
			{
				msgtype &= ~NormalMessage;		  // Reset normal message type bits
				msgtype |= MessageType_Important; // Set the bit for the requested type
			}
			else if (str.compare("quiet") == 0 || str.compare("silent") == 0 || str.compare("critical") == 0 || str.compare("criticalonly") == 0)
			{
				msgtype &= ~NormalMessage;		 // Reset normal message type bits
				msgtype |= MessageType_Critical; // Set the bit for the requested type
			}
			else
			{
				this->Log(MessageType_Warning) << "Failed to set the notification level! Unrecognized value \"" << str << "\"!" << std::endl;
			}
		}

		// The possibility to ignore warnings (not recommended, but possible)
		bool flag;
		if (this->properties->Get("ignorewarnings", flag) || this->properties->Get("nowarn", flag))
		{
			if (flag)
				msgtype &= ~MessageType_Warning; // Reset the warning flag
			else
				msgtype |= MessageType_Warning; // Set the warning flag
		}

		// The possibility to ignore errors (not recommended, but possible)
		if (this->properties->Get("ignoreerrors", flag) || this->properties->Get("noerr", flag))
		{
			if (flag)
				msgtype &= ~MessageType_Error; // Reset the error flag
			else
				msgtype |= MessageType_Error; // Set the error flag
		}

		// Set the notification level
		this->output.SetNotificationLevel(msgtype);

		// Now call the pure virtual Validate method
		return this->Validate();
	}
	// -----------------------------------------------------
	// BasicTask other private methods
	// -----------------------------------------------------
	// Reset time and trajectory step before running the task
	void BasicTask::ResetTimeAndTrajectoryStep()
	{
		bool trajectoryStepFirst = this->runsection.settings->SetTrajectoryStepBeforeTime();
		for (auto i = this->runsection.systems.cbegin(); i != this->runsection.systems.cend(); i++)
		{
			// Prepare Interaction objects
			auto interactions = (*i)->Interactions();
			for (auto j = interactions.begin(); j != interactions.end(); j++)
			{
				if (trajectoryStepFirst)
				{
					(*j)->SetTrajectoryStep(this->runsection.settings->TrajectoryStep());
					(*j)->SetTime(this->runsection.settings->Time());
				}
				else
				{
					(*j)->SetTime(this->runsection.settings->Time());
					(*j)->SetTrajectoryStep(this->runsection.settings->TrajectoryStep());
				}
			}

			// Prepare Transition objects
			auto transitions = (*i)->Transitions();
			for (auto j = transitions.begin(); j != transitions.end(); j++)
			{
				if (trajectoryStepFirst)
				{
					(*j)->SetTrajectoryStep(this->runsection.settings->TrajectoryStep());
					(*j)->SetTime(this->runsection.settings->Time());
				}
				else
				{
					(*j)->SetTime(this->runsection.settings->Time());
					(*j)->SetTrajectoryStep(this->runsection.settings->TrajectoryStep());
				}
			}

			// Prepare Spin objects
			auto spins = (*i)->Spins();
			for (auto j = spins.begin(); j != spins.end(); j++)
			{
				if (trajectoryStepFirst)
				{
					(*j)->SetTrajectoryStep(this->runsection.settings->TrajectoryStep());
					(*j)->SetTime(this->runsection.settings->Time());
				}
				else
				{
					(*j)->SetTime(this->runsection.settings->Time());
					(*j)->SetTrajectoryStep(this->runsection.settings->TrajectoryStep());
				}
			}
		}
	}
	// -----------------------------------------------------
	// BasicTask protected methods
	// -----------------------------------------------------
	// This method can be overriden in order to provide a calculation method that runs on supercomputer clusters
	bool BasicTask::RunMPI()
	{
		std::cout << "ERROR: The task \"" << this->Properties()->Name() << "\" does not have an MPI version!" << std::endl;
		return false;
	}

    void BasicTask::SelectPropagator(std::string str)
    {
		if(str == "RK4")
		{
			this->prop = Propagator::RK4; //not currently supported
			return;
		}
		if(str == "exp")
		{
			this->prop = Propagator::exp;
			return;
		}
		if(str == "RK45")
		{
			this->prop = Propagator::RK45;
			return;
		}
		this->prop = Propagator::Default;
    }

	void BasicTask::GetSamples(std::vector<arma::sp_cx_mat>& H, arma::sp_cx_mat& A, std::vector<SCData>& ori, std::vector<std::vector<double>>& SampleWeights, std::vector<std::vector<std::vector<double>>>& AllWeights)
    {
		std::vector<SampleCombination> Combinations; 
		std::vector<std::vector<int>> samples;
		std::vector<arma::sp_cx_mat> sampleHSC;
		for(auto i : ori)
		{
			samples.push_back(i.samples);
			sampleHSC.push_back(i.SamplesMatrix);
		}
		Combinations = GenerateCombinationsNI(samples);
		for(unsigned int i = 0; i < Combinations.size(); i++)
		{
			H.push_back(GetHamiltonian(A,ori.size(),sampleHSC,Combinations[i]));
			std::vector<double> weights;
			for(unsigned int j = 0; j < Combinations[i].size(); j++)
			{
				for(unsigned int k = 0; k < Combinations[i][j].size(); k++)
				{
					weights.push_back(AllWeights[j][k][Combinations[i][j][k]]);
				}
			}
			SampleWeights.push_back(weights);
		}
    }

    void BasicTask::SCDirectEvaluation(SpinAPI::system_ptr& i, arma::sp_cx_mat& A, SCData& Data, arma::cx_vec& rho0vec, arma::cx_vec &ReturnVec)
    {

		std::vector<std::pair<int,arma::cx_vec>> SCresults;
		std::vector<std::pair<int,double>> SCweights;
		std::vector<arma::sp_cx_mat> As;
		std::vector<std::vector<std::vector<double>>> AllWeights = {{}};
		std::pair<std::vector<double>,std::vector<int>> BLandSamples;
		std::vector<std::vector<double>> SampleSpacing;
		std::vector<std::vector<double>> SampleWeights;

		{
			for(auto e = i->interactions_cbegin(); e != i->interactions_cend(); e++)
			{
				if((*e)->Type() == SpinAPI::InteractionType::SemiClassicalField)
				{
					AllWeights[0].push_back((*e)->GetOriWeights());
					std::vector<double> BL = (*e)->VL();
					double BMax = std::accumulate(BL.begin(), BL.end(), 0.0);
					BLandSamples.first.push_back(BMax);
					BLandSamples.second.push_back((*e)->Orientations());
					SampleSpacing.push_back((*e)->GetSpacing());
				}
			}
		}
		std::vector<SCData> SysData= {Data};
		GetSamples(As,A, SysData, SampleWeights, AllWeights);
		this->Log() << "Ready to perform calculation." << std::endl;
		
		//#pragma omp parallel for
		if (As.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
			throw std::runtime_error("Too many samples for OpenMP loop indexing.");

		const int sample_count = static_cast<int>(As.size());
		#pragma omp parallel for
		for (int i = 0; i < sample_count; ++i)
		{
			arma::cx_vec result = -1 * solve(arma::conv_to<arma::cx_mat>::from(As[i]), rho0vec);
			std::cout << "Sample " << i << " trace: " << arma::trace(arma::reshape(result, std::sqrt(result.n_rows), std::sqrt(result.n_rows))) << std::endl;
			std::vector<double> weights = SampleWeights[i];
			double weight_product = 1.0;
			for(unsigned int j = 0; j < weights.size(); j++)
			{
				result *= weights[j];
				weight_product *= weights[j];
			}
			#pragma omp critical
			{
				SCresults.push_back({i,result});
				SCweights.push_back({i,weight_product});
			}
		}
		std::pair<arma::cx_vec, double> results = IntegrateSC(SCresults, SCweights, SCIntegrationProperties{BLandSamples.first, BLandSamples.second, SampleSpacing});
		arma::cx_vec result = arma::zeros<arma::cx_vec>(rho0vec.n_rows);
		result = results.first;
		ReturnVec = result;		
    }

    std::pair<arma::cx_vec,double> BasicTask::IntegrateSC(std::vector<std::pair<int,arma::cx_vec>>& rhoSC, std::vector<std::pair<int,double>>& rhoweights, SCIntegrationProperties props)
    {
		auto SCresultsSort = [](const std::pair<int,arma::cx_vec> &a, const std::pair<int,arma::cx_vec> &b) {
				return a.first < b.first;
			};
		auto SCweightsSort = [](const std::pair<int,double> &a, const std::pair<int,double> &b) {
				return a.first < b.first;
			};

		std::sort(rhoSC.begin(), rhoSC.end(), SCresultsSort);
		std::sort(rhoweights.begin(), rhoweights.end(), SCweightsSort);

		arma::cx_vec result(rhoSC[0].second.n_rows,arma::fill::zeros);
		int dimensions = props.maxBondLenght.size();
		std::vector<std::vector<double>> stepsizes; 
		for(int i = 0; i < dimensions; i++)
		{
			stepsizes.push_back(props.spacing[i]);
		}
		std::vector<double> slice;
		std::vector<arma::cx_vec> sliceV;

		struct Trapezium
		{
			double L1;
			arma::cx_vec V1;
			double L2;
			arma::cx_vec V2;
			double gap;
			double value;
			arma::cx_vec VectorValue;
			int dimension;
			int sample;

			double Eval()
			{
				VectorValue = 0.5 *(V1 + V2) * gap;
				return 0.5 * (L1 + L2) * gap;
			}
		};

		std::vector<Trapezium> trapezia;
		for(unsigned int i = 0; i < rhoweights.size(); i++)
		{
			slice.push_back(rhoweights[i].second);
			sliceV.push_back(rhoSC[i].second);
		}
		//first create all the 2d trapezia
		int sample = 0;
		int index = 0; 
		for(unsigned int i = 0; i <= slice.size()-1; i++)
		{
			if ((index+1) % props.numSamples[dimensions-1] == 0 && i != 0)
			{
				sample++;
				index = 0;
				continue;
			}
			Trapezium t;
			t.L1 = slice[i];
			t.L2 = slice[i+1];
			t.V1 = sliceV[i];
			t.V2 = sliceV[i+1];
			t.gap = stepsizes[dimensions-1][index+1] - stepsizes[dimensions-1][index];
			t.dimension = 1;
			t.sample = sample;
			t.value = t.Eval();
			index++;
			if(t.value < 0)
			{
				std::cin.get();
			}
			trapezia.push_back(t);
		}

		//evaluates the 2d trapezia
		int samplesize = props.numSamples[props.numSamples.size() - 1] - 1;
		auto Eval = [](std::vector<Trapezium>& trapezia, int dim, int samplesize) {
			double area = trapezia[0].value;
			arma::cx_vec V = trapezia[0].VectorValue;
			for(int e = 1; e < samplesize; e++)
			{
				area += trapezia[e].value;
				V += trapezia[e].VectorValue;
			}
			Trapezium t;
			t.value = area;
			t.dimension = dim;
			t.VectorValue = V;
			trapezia.clear();
			return t;
		};

		std::vector<Trapezium> NewTrapezia;
		std::vector<Trapezium> TempTrapezia;
		for(unsigned int i = 0; i < trapezia.size(); i++)
		{
			if(i % samplesize == 0 && i != 0)
			{
				NewTrapezia.push_back(Eval(TempTrapezia,1, samplesize));
			}
			TempTrapezia.push_back(trapezia[i]);
		}
		NewTrapezia.push_back(Eval(TempTrapezia,1,samplesize));
		trapezia = NewTrapezia;
		NewTrapezia.clear();

		//recursive function that evaluates the integral for n dimensions (starts at the 2nd dimension)
		//each loop creates new sets of "trapezia" so each dimension can be treated as having 1d integration.
		//recursivly goes down until theres one dimension to work and the the "trapezia" it calculates is then propogated back up the stack
		//New trapezia should have a length of one by the end of the last loop
		std::function<std::vector<Trapezium>(std::vector<Trapezium>, std::vector<int>)> Eval2;
		int depth = 0;
		Eval2 = [&depth, &Eval2, &stepsizes, &result](std::vector<Trapezium> trapezia, std::vector<int> samples) {
			int dimensions = samples.size();
			if(dimensions != 1)
			{
				std::vector<int> samples2(samples.begin() + 1, samples.begin() + dimensions);
				depth += 1;
				trapezia = Eval2(trapezia, samples2);
				depth -= 1;
			}
			std::vector<Trapezium> NewTrapezia;
			std::vector<Trapezium> TempTrapezia;
			int ss= samples[0]-1;
			for(unsigned int i = 0; i <= trapezia.size(); i++)
			{
				if(i % (ss+1) == 0 && i != 0)
				{	
					double vol = 0;
					arma::cx_vec V = result;
					for(unsigned int e = 0; e < TempTrapezia.size()-1; e++)
					{
						double gap = stepsizes[depth][e+1] - stepsizes[depth][e];
						Trapezium t2;
						t2.L1 = TempTrapezia[e].value;
						t2.L2 = TempTrapezia[e+1].value;
						t2.V1 = TempTrapezia[e].VectorValue;
						t2.V2 = TempTrapezia[e+1].VectorValue;
						t2.gap = gap;
						t2.value = t2.Eval();
						vol += t2.value;
						V += t2.VectorValue;
					}
					TempTrapezia.clear();
					Trapezium t;
					t.value = vol;
					t.VectorValue = V;
					t.dimension = depth + 1;
					NewTrapezia.push_back(t);
					if(i == trapezia.size())
					{
						break;
					}
				}
				TempTrapezia.push_back(trapezia[i]);
			}
			return NewTrapezia;
		};

		std::vector<int> NumOfTrapezia = {};
		for(int i = 1; i < dimensions; i++)
		{
			NumOfTrapezia.push_back(props.numSamples[dimensions-i-1]);
		}
		if(NumOfTrapezia.size() != 0)
			trapezia = Eval2(trapezia, NumOfTrapezia);
		

		double totalIntegral = 0.0;
		arma::cx_vec totalVec = result;
		index = 0;
		for(auto t : trapezia)
		{
			totalIntegral += t.value;
			totalVec += t.VectorValue;
			index += 1;
		}
		return {totalVec,totalIntegral};

    }

    // Method that provides access to the calculation settings
	std::shared_ptr<const Settings> BasicTask::RunSettings() const
	{
		return this->runsection.settings;
	}

	// Method that provides access to spin systems
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> BasicTask::SpinSystems() const // TODO: Make the shared_ptrs point to const SpinSystems
	{
		return this->runsection.systems;
	}

	// Provides access to the properties object for the task
	const std::shared_ptr<MSDParser::ObjectParser> &BasicTask::Properties() const
	{
		return this->properties;
	}

	// Returns the log stream
	std::ostream &BasicTask::Log(const MessageType &_msgtype)
	{
		return this->output.Log(_msgtype);
	}

	// Returns the data/results stream
	std::ostream &BasicTask::Data()
	{
		return this->output.Data();
	}

	// Call the StandardOutput header method
	bool BasicTask::WriteStandardOutputHeader(std::ostream &_stream)
	{
		return this->runsection.WriteOutputHeader(_stream);
	}

	// Call the StandardOutput data method
	bool BasicTask::WriteStandardOutput(std::ostream &_stream)
	{
		return this->runsection.WriteOutput(_stream);
	}

	const std::vector<std::shared_ptr<Action>> &BasicTask::Actions() const
	{
		return this->runsection.actions;
	}

	// Sets a reference to the requested ActionScalar
	bool BasicTask::Scalar(std::string _name, ActionScalar **_scalar)
	{
		auto i = this->scalars->find(_name);
		if (i != this->scalars->cend())
		{
			// Set the pointer if any
			if (_scalar != nullptr)
			{
				(*_scalar) = const_cast<ActionScalar *>(&(i->second));

				// Save the value of the ActionScalar so it can be restored later
				if (this->usedScalars.find(_name) == this->usedScalars.cend())
				{
					this->usedScalars[_name] = std::pair<ActionScalar *, double>(*_scalar, (*_scalar)->Get());
				}
			}

			return true;
		}

		return false;
	}

	// Sets a reference to the requested ActionVector
	bool BasicTask::Vector(std::string _name, ActionVector **_vector)
	{
		auto i = this->vectors->find(_name);
		if (i != this->vectors->cend())
		{
			// Set the pointer if any
			if (_vector != nullptr)
			{
				(*_vector) = const_cast<ActionVector *>(&(i->second));

				// Save the value of the ActionVector so it can be restored later
				if (this->usedVectors.find(_name) == this->usedVectors.cend())
				{
					this->usedVectors[_name] = std::pair<ActionVector *, arma::vec>(*_vector, (*_vector)->Get());
				}
			}

			return true;
		}

		return false;
	}
	// -----------------------------------------------------
	// BasicTask public methods
	// -----------------------------------------------------
	// Runs the task
	bool BasicTask::Run()
	{
		// Make sure the task is valid
		if (!this->IsValid())
		{
			this->Log() << "Warning: Skipping invalid task \"" << this->Name() << "\"." << std::endl;
			return false;
		}

		// -----------------------------------------------------
		// Prepare for the task
		// -----------------------------------------------------
		// Set the time and trajectory step on all relevant objects
		this->ResetTimeAndTrajectoryStep();

		// Store current value of ActionScalars used by the task
		for (auto i = this->usedScalars.begin(); i != this->usedScalars.end(); i++)
			i->second.second = i->second.first->Get();

		// Store current value of ActionVectors used by the task
		for (auto i = this->usedVectors.begin(); i != this->usedVectors.end(); i++)
			i->second.second = i->second.first->Get();

		// -----------------------------------------------------
		// Run the task
		// -----------------------------------------------------
		this->Log(MessageType_Important) << "--- Running task \"" << this->Name() << "\" ---" << std::endl;
		bool run_status = this->RunLocal();

		// -----------------------------------------------------
		// Clean-up after running the task
		// -----------------------------------------------------
		// Reset ActionScalars to the value stored before the task was run
		for (auto i = this->usedScalars.cbegin(); i != this->usedScalars.cend(); i++)
		{
			if (i->second.first->IsReadonly())
				i->second.first->Reset();
			else
				i->second.first->Set(i->second.second);
		}

		// Reset ActionVectors to the value stored before the task was run
		for (auto i = this->usedVectors.cbegin(); i != this->usedVectors.cend(); i++)
		{
			if (i->second.first->IsReadonly())
				i->second.first->Reset();
			else
				i->second.first->Set(i->second.second);
		}

		return run_status;
	}

	// Checks whether the task is valid
	bool BasicTask::IsValid()
	{
		// Make sure the validation method has been called
		if (!this->isValidated)
		{
			this->isValid = this->DoValidation();
			this->isValidated = true;

			// Notify the user if validation fails
			if (!this->isValid)
				this->Log() << "ERROR: Failed to validate task \"" << this->Name() << "\"!" << std::endl;
		}

		return this->isValid;
	}

	// Returns the name of the task
	std::string BasicTask::Name()
	{
		if (this->properties == nullptr)
			return "undefined";

		return this->properties->Name();
	}

	// Method to provide BasicTask with access to ActionTargets
	void BasicTask::SetActionTargets(const std::map<std::string, ActionScalar> &_scalars, const std::map<std::string, ActionVector> &_vectors)
	{
		this->scalars = &_scalars;
		this->vectors = &_vectors;
	}

	// -----------------------------------------------------
	// Public OutputHandler stream manipulation methods
	// -----------------------------------------------------
	// Change the log stream
	bool BasicTask::SetLogStream(std::ostream &_stream)
	{
		return this->output.SetLogStream(_stream);
	}

	// Change the data stream
	bool BasicTask::SetDataStream(std::ostream &_stream)
	{
		return this->output.SetDataStream(_stream);
	}
	// -----------------------------------------------------
}
