/////////////////////////////////////////////////////////////////////////
// RunSection implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ObjectParser.h"
#include "Settings.h"
#include "BasicTask.h"
#include "Action.h"
#include "StandardOutput.h"
#include "SpinSystem.h"
#include "RunSection.h"

// Include source file with method to create task classes
#include "RunSection_CreateTask.cpp"

// Action classes
#include "ActionRotateVector.h"
#include "ActionScaleVector.h"
#include "ActionAddVector.h"
#include "ActionAddScalar.h"
#include "ActionMultiplyScalar.h"
#include "ActionFibonacciSphere.h"
#include "ActionLogSpace.h"

// object classes
#include "State.h"

namespace RunSection
{
	// -----------------------------------------------------
	// RunSection Constructors and Destructor
	// -----------------------------------------------------
	RunSection::RunSection() : tasks(), actions(), outputs(), settings(std::make_shared<Settings>()), systems(), actionScalars(), actionVectors(), overruleAppend(false), noCalculations(false)
	{
		this->settings->GetActionTargets(this->actionScalars, this->actionVectors);
	}

	RunSection::~RunSection()
	{
	}
	// -----------------------------------------------------
	// RunSection public methods
	// -----------------------------------------------------
	// Runs all tasks sequentially
	bool RunSection::Run(unsigned int _stepNumber)
	{
		// Update the settings object with the current calculation step
		this->settings->SetCurrentStep(_stepNumber);

		// Run all tasks
		for (auto i = this->tasks.cbegin(); i != this->tasks.cend(); i++)
			(*i)->Run();

		return true;
	}

	// Runs all tasks sequentially starting from the task with the given name
	bool RunSection::Run(std::string _checkpoint, unsigned int _stepNumber)
	{
		// Update the settings object with the current calculation step
		this->settings->SetCurrentStep(_stepNumber);

		// Obtain an iterator to the first task
		auto i = this->tasks.cbegin();

		// Skip until checkpoint is found
		while (i != this->tasks.cend() && (*i)->Name().compare(_checkpoint) != 0)
		{
			i++;
		}

		// Run the rest of the tasks from that point on (if any)
		for (; i != this->tasks.cend(); i++)
			(*i)->Run();

		return true;
	}

	// Take a number of steps, calling all the action once per step
	bool RunSection::Step(unsigned int _currentStep)
	{
		for (auto j = this->actions.cbegin(); j != this->actions.cend(); j++)
			(*j)->Step(_currentStep);

		for (auto j = this->actions.cbegin(); j != this->actions.cend(); j++)
		{
			std::string object;
			(*j)->GetProperties()->Get("scalar", object);
			if (object == "")
			{
				continue;
			}

			// get the middle bit
			std::string state;
			std::string system;
			bool start = false;
			for (auto c = object.cbegin(); c != object.cend(); c++)
			{
				if ((*c) == '.')
				{
					start = !start;
					if (!start)
					{
						break;
					}
				}
				else if (start)
				{
					state = state + (*c);
				}
				else
				{
					system = system + (*c);
				}
			}

			SpinAPI::state_ptr ActionState;
			std::string system2 = "";
			int SpinSystem = -1;
			bool IsState = true;
			while (system2 != system && IsState)
			{
				SpinSystem++;
				if ((unsigned int)SpinSystem == this->systems.size())
				{
					IsState = false;
				}
				system2 = this->systems[SpinSystem]->Name();
			}
			if (!IsState)
			{
				continue;
			}
			ActionState = this->systems[SpinSystem]->states_find(state);
			if (ActionState == nullptr)
			{
				continue;
			}
			ActionState->Update();
		}

		return true;
	}

	// Add a Task or Action to the collections
	// Objects derived from BasicTask or Action are created here
	bool RunSection::Add(MSDParser::ObjectType _type, const MSDParser::ObjectParser &_obj)
	{
		// First determine the type of the object to create
		if (_type == MSDParser::ObjectType::Task)
		{
			// ------------------------------------------------------------------------------
			// Create a BasicTask of the correct Task class
			// ------------------------------------------------------------------------------
			// Get the type of task
			std::string type;
			if (!_obj.Get("type", type))
			{
				std::cout << "ERROR: No type specified for task \"" << _obj.Name() << "\"! Ignoring task!" << std::endl;
				return false;
			}

			// Create the task instance from the string
			// NOTE: This method is located in the seperate file "RunSection_CreateTask.cpp"
			std::shared_ptr<BasicTask> task = this->CreateTask(type, _obj);

			// Create a task of the proper type
			if (task == nullptr)
			{
				std::cout << "Warning: The type of the Task \"" << _obj.Name() << "\" was not recognized!" << std::endl;
				return false;
			}

			// Allow access to ActionTargets - note that changes made to an ActionTarget within a Task are reverted once the task finishes
			task->SetActionTargets(this->actionScalars, this->actionVectors);

			// Add the task to the collection
			this->tasks.push_back(task);
		}
		else if (_type == MSDParser::ObjectType::Action)
		{
			// ------------------------------------------------------------------------------
			// Create an Action of the correct Action class
			// ------------------------------------------------------------------------------
			std::shared_ptr<Action> action = nullptr;

			// Get the type of action
			std::string type;
			if (!_obj.Get("type", type))
				return false;

			// Create an action of the proper type
			if (type.compare("rotatevector") == 0)
			{
				action = std::make_shared<ActionRotateVector>(_obj, actionScalars, actionVectors);
			}
			else if (type.compare("fibonaccisphere") == 0)
			{
				action = std::make_shared<ActionFibonacciSphere>(_obj, actionScalars, actionVectors);
			}
			else if (type.compare("scalevector") == 0)
			{
				action = std::make_shared<ActionScaleVector>(_obj, actionScalars, actionVectors);
			}
			else if (type.compare("addvector") == 0)
			{
				action = std::make_shared<ActionAddVector>(_obj, actionScalars, actionVectors);
			}
			else if (type.compare("addscalar") == 0)
			{
				action = std::make_shared<ActionAddScalar>(_obj, actionScalars, actionVectors);
			}
			else if (type.compare("multiplyscalar") == 0 || type.compare("scalescalar") == 0)
			{
				action = std::make_shared<ActionMultiplyScalar>(_obj, actionScalars, actionVectors);
			}
			else if (type.compare("logspace") == 0)
			{
				action = std::make_shared<ActionLogSpace>(_obj, actionScalars, actionVectors);
			}
			else
			{
				std::cout << "Warning: The type of Action \"" << _obj.Name() << "\" was not recognized!" << std::endl;
				return false;
			}

			// Only use the action if it is valid
			if (action->Validate())
			{
				this->actions.push_back(action);
				return true;
			}
			else
			{
				std::cout << "Warning: Failed to validate Action \"" << _obj.Name() << "\"!" << std::endl;
				return false;
			}
		}
		else if (_type == MSDParser::ObjectType::Settings)
		{
			this->settings->Update(_obj);
			SpinAPI::StandardOutput::SetDefaultPrecision(this->settings->DataPrecision());
		}

		return false;
	}

	// Add an ActionScalar to the RunSection object
	bool RunSection::Add(std::string _name, const ActionScalar &_at)
	{
		auto result = this->actionScalars.insert(NamedActionScalar(_name, _at));
		return result.second;
	}

	// Add an ActionVector to the RunSection object
	bool RunSection::Add(std::string _name, const ActionVector &_at)
	{
		auto result = this->actionVectors.insert(NamedActionVector(_name, _at));
		return result.second;
	}

	// Add a SpinSystem object
	bool RunSection::Add(std::shared_ptr<SpinAPI::SpinSystem> _system)
	{
		if (_system != nullptr)
		{
			this->systems.push_back(_system);
			_system->GetActionTargets(this->actionScalars, this->actionVectors);
			return true;
		}

		return false;
	}

	// Add an output object
	bool RunSection::Add(const SpinAPI::output_ptr &_output)
	{
		if (_output->Validate(actionScalars, actionVectors))
		{
			_output->SetDelimiter(this->settings->DataDelimiter());
			this->outputs.push_back(_output);
			return true;
		}

		std::cout << "Warning: Failed to validate output object \"" << _output->Name() << "\"!" << std::endl;
		return false;
	}

	// Output all the objects contained in the RunSection
	void RunSection::PrintSystems(bool _showFullObjectInformation) const
	{
		// Write output for all SpinSystems
		for (auto i = this->systems.cbegin(); i != systems.cend(); i++)
		{
			(*i)->Print(_showFullObjectInformation);
		}

		std::cout << "\n# Tasks (" << this->tasks.size() << "):" << std::endl;
		for (auto i = this->tasks.cbegin(); i != this->tasks.cend(); i++)
			std::cout << " - " << (*i)->Name() << std::endl;

		std::cout << "\n# Actions (" << this->actions.size() << "):" << std::endl;
		for (auto i = this->actions.cbegin(); i != this->actions.cend(); i++)
			std::cout << " - " << (*i)->Name() << std::endl;

		std::cout << "\n# Output objects (" << this->outputs.size() << "):" << std::endl;
		for (auto i = this->outputs.cbegin(); i != this->outputs.cend(); i++)
			std::cout << " - " << (*i)->Name() << std::endl;

		std::cout << std::endl;
	}
	// -----------------------------------------------------
	// Public get-object-by-name methods (needed for tests)
	// -----------------------------------------------------
	std::shared_ptr<BasicTask> RunSection::GetTask(const std::string &_name)
	{
		for (auto i = this->tasks.cbegin(); i != this->tasks.cend(); i++)
			if ((*i)->Name().compare(_name) == 0)
				return (*i);

		return nullptr;
	}
	// -----------------------------------------------------
	// Public methods to get ActionTargets
	// -----------------------------------------------------
	// Returns a copy of all ActionScalars
	std::map<std::string, ActionScalar> RunSection::GetActionScalars()
	{
		return this->actionScalars;
	}

	// Returns a copy of all ActionVectors
	std::map<std::string, ActionVector> RunSection::GetActionVectors()
	{
		return this->actionVectors;
	}
	// -----------------------------------------------------
	// Public output methods
	// -----------------------------------------------------
	// Writes headers for the StandardOutput ojects
	bool RunSection::WriteOutputHeader(std::ostream &_stream) const
	{
		// Call the write methods on the output objects
		for (auto i = this->outputs.cbegin(); i != this->outputs.cend(); i++)
		{
			(*i)->WriteOutputHeader(_stream);
			_stream << this->settings->DataDelimiter();
		}

		return true;
	}

	// Writes data for the StandardOutput objects
	bool RunSection::WriteOutput(std::ostream &_stream) const
	{
		// Call the write methods on the output objects
		for (auto i = this->outputs.cbegin(); i != this->outputs.cend(); i++)
		{
			(*i)->WriteOutput(_stream);
			_stream << this->settings->DataDelimiter();
		}

		return true;
	}
	// -----------------------------------------------------
}
