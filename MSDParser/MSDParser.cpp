/////////////////////////////////////////////////////////////////////////
// MSD Parser implementations, used to load MSD input-files for the
// application.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "RunSection.h"
#include "MSDParser.h"
#include "FileReader.h"
#include <exception>

// SpinAPI files
#include "Spin.h"
#include "Interaction.h"
#include "Transition.h"
#include "Operator.h"
#include "State.h"
#include "Pulse.h"
#include "StandardOutput.h"
#include "SpinSystem.h"
#include "SubSystem.h"
#include "PulseSequence.h"

namespace MSDParser
{
	// -----------------------------------------------------
	// MSDParser Constructors and Destructor
	// -----------------------------------------------------
	// Initialize the MSDParser with an MSD file
	MSDParser::MSDParser(std::string _filename) : filename(_filename), systems(), runTasks(), runActions(), settingsObjects()
	{
	}

	// Destructor
	MSDParser::~MSDParser()
	{
	}

	// -----------------------------------------------------
	// MSDParser public Parse method
	// -----------------------------------------------------
	bool MSDParser::Load()
	{
		// Prepare reading the file
		FileReader reader(this->filename);

		// Check that the file was found/accessible
		if (!reader.IsOpen())
			return false;

		// Read the first object
		MSDFileObject obj(reader.ReadObject());

		// Loop through all objects in the file
		while (obj.Type() != ObjectType::EOFObject)
		{
			// First determine the type of the object group
			if (reader.ObjectGroupType() == ObjectGroup::SpinSystem) // -------------------------- SpinSystem --------------------------
			{
				// Get an iterator to the current SpinSystem
				auto i = this->systems.begin();
				while (i != this->systems.end() && (*i)->Name().compare(reader.ObjectGroupName()) != 0)
				{
					i++;
				}

				// If the SpinSystem was not found in the collection
				if (i == this->systems.end())
				{
					// Push SpinSystem to the back, and get an iterator to it
					this->systems.push_back(std::shared_ptr<SpinAPI::SpinSystem>(std::make_shared<SpinAPI::SpinSystem>(reader.ObjectGroupName())));
					i = this->systems.end();
					i--;
				}

				// Then create the object itself based on its type
				switch (obj.Type())
				{
				case ObjectType::Spin:
					(*i)->Add(std::make_shared<SpinAPI::Spin>(obj.Name(), obj.Contents()));
					break;
				case ObjectType::Interaction:
					(*i)->Add(std::make_shared<SpinAPI::Interaction>(obj.Name(), obj.Contents()));
					break;
				case ObjectType::Transition:
					(*i)->Add(std::make_shared<SpinAPI::Transition>(obj.Name(), obj.Contents(), (*i)));
					break;
				case ObjectType::Operator:
					(*i)->Add(std::make_shared<SpinAPI::Operator>(obj.Name(), obj.Contents()));
					break;
				case ObjectType::State:
					(*i)->Add(std::make_shared<SpinAPI::State>(obj.Name(), obj.Contents()));
					break;
				case ObjectType::Pulse:
					(*i)->Add(std::make_shared<SpinAPI::Pulse>(obj.Name(), obj.Contents()));
					break;
				case ObjectType::SubSystem:
					(*i)->Add(std::make_shared<SpinAPI::SubSystem>(obj.Name(), obj.Contents(), (*i)));
					break;
				case ObjectType::PulseSequence:
					(*i)->Add(std::make_shared<SpinAPI::PulseSequence>(obj.Name(), obj.Contents()));
					break;
				case ObjectType::Properties:
					if (!(*i)->SetProperties(std::make_shared<ObjectParser>(obj.Name(), obj.Contents())))
						std::cout << "Error: A Properties object was already created for SpinSystem \"" << reader.ObjectGroupName() << "\"! Object \"" << obj.Name() << "\" ignored!" << std::endl;
					break;
				default:
					std::cout << "Error: Unknown or illegal type for object \"" << obj.Name() << "\" within SpinSystem \"" << reader.ObjectGroupName() << "\"! Object ignored!" << std::endl;
				}
			}
			else if (reader.ObjectGroupType() == ObjectGroup::Settings) // -------------------------- Settings --------------------------
			{
				// Check for valid object types for the Settings section
				switch (obj.Type())
				{
				case ObjectType::Settings:
					this->settingsObjects.push_back(ObjectParser(obj.Name(), obj.Contents()));
					break;
				case ObjectType::Action:
					this->runActions.push_back(ObjectParser(obj.Name(), obj.Contents()));
					break;
				case ObjectType::Output:
					this->runOutputs.push_back(std::make_shared<SpinAPI::StandardOutput>(obj.Name(), obj.Contents()));
					break;
				default:
					std::cout << "Error: Unknown or illegal type for object \"" << obj.Name() << "\" within the Settings section! Object ignored!" << std::endl;
				}
			}
			else if (reader.ObjectGroupType() == ObjectGroup::Run) // -------------------------- RunSection --------------------------
			{
				// Check for valid object types for the Run section
				switch (obj.Type())
				{
				case ObjectType::Task:
					this->runTasks.push_back(ObjectParser(obj.Name(), obj.Contents()));
					break;
				default:
					std::cout << "Error: Unknown or illegal type for object \"" << obj.Name() << "\" within the Run section! Object ignored!" << std::endl;
				}
			}
			else // -------------------------- Unknown --------------------------
			{
				std::cout << "Found an object outside any recognized object group. Object ignored." << std::endl;
			}

			// Get next object in the file
			obj = reader.ReadObject();
		}

		// Put spins into other objects - this can only be done now that all spins have been loaded
		int failed = 0;
		for (auto i = this->systems.cbegin(); i != this->systems.cend(); i++)
		{
			// Prepare the state objects
			auto failedStates = (*i)->ValidateStates();
			for (auto j = failedStates.cbegin(); j != failedStates.cend(); j++)
			{
				std::cout << "Failed to load state " << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}

			// Load Spins into Interaction objects
			auto failedInteractions = (*i)->ValidateInteractions();
			for (auto j = failedInteractions.cbegin(); j != failedInteractions.cend(); j++)
			{
				std::cout << "Failed to load interaction " << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}

			// Prepare Operator objects
			auto failedOperators = (*i)->ValidateOperators(this->systems);
			for (auto j = failedOperators.cbegin(); j != failedOperators.cend(); j++)
			{
				std::cout << "Failed to load operator object " << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}

			// Load Spins into Pulses objects
			auto failedpulses = (*i)->ValidatePulses();
			for (auto j = failedpulses.cbegin(); j != failedpulses.cend(); j++)
			{
				std::cout << "Failed to load pulse " << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}

		}

		// Put states into other objects - this can only be done now that all states have been loaded
		for (auto i = this->systems.cbegin(); i != this->systems.cend(); i++)
		{
			// Load States into Transition objects
			auto failedTransitions = (*i)->ValidateTransitions(this->systems);
			for (auto j = failedTransitions.cbegin(); j != failedTransitions.cend(); j++)
			{
				std::cout << "Failed to load transition " << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}
		}

		// validating subsystems;
		for (auto i = this->systems.cbegin(); i != this->systems.cend(); i++)
		{
			auto failedsubsystems = (*i)->ValidateSubSystems();
			for (auto j = failedsubsystems.cbegin(); j != failedsubsystems.cend(); j++)
			{
				std::cout << "Failed to load SubSystem " << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}

			SpinAPI::LinkTransitions(*i);
		}

		for (auto i = this->systems.cbegin(); i != this->systems.cend(); i++)
		{
			auto failedpulseseq = (*i)->ValidatePulseSequences();
			for (auto j = failedpulseseq.cbegin(); j != failedpulseseq.cend(); j++)
			{
				std::cout << "Failed to load pulse sequence" << (*j)->Name() << "!" << std::endl;
				failed += 1;
			}
		}

		if(failed > 0)
		{
			#if ASSERT == 1
				throw std::exception();
			#endif
		}

		return true;
	}
	// -----------------------------------------------------
	// Other public methods
	// -----------------------------------------------------
	// Creates tasks and actions for a RunSection object
	void MSDParser::FillRunSection(RunSection::RunSection &_run)
	{
		// Add SpinSystems to the RunSection
		for (auto i = this->systems.cbegin(); i != this->systems.cend(); i++)
			_run.Add((*i));

		// Add tasks to the RunSection
		for (auto i = this->runTasks.cbegin(); i != this->runTasks.cend(); i++)
			_run.Add(ObjectType::Task, (*i));

		// Add the settings object
		for (auto i = this->settingsObjects.cbegin(); i != this->settingsObjects.cend(); i++)
			_run.Add(ObjectType::Settings, (*i));

		// -----------------------------------------------------------------------------
		// NOTE: Make sure that all ActionTargets are created before this point, since
		// the objects added below are using the ActionTargets!
		// -----------------------------------------------------------------------------

		// Add actions to the RunSection
		for (auto i = this->runActions.cbegin(); i != this->runActions.cend(); i++)
			_run.Add(ObjectType::Action, (*i));

		// Add output objects to the RunSection
		for (auto i = this->runOutputs.cbegin(); i != this->runOutputs.cend(); i++)
			_run.Add(*i);
	}
	// -----------------------------------------------------
}
