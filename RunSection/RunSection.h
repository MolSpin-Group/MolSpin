/////////////////////////////////////////////////////////////////////////
// RunSection (RunSection module)
// ------------------
// Class that keeps track of all objects generated from the MSDParser
// module, and is responsible for keeping track of the calculation step
// to run, triggering actions, and running task classes.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_RunSection
#define MOD_RunSection_RunSection

#include <memory>
#include <vector>
#include <map>
#include "MSDParserDefines.h"
#include "MSDParserfwd.h"
#include "RunSectionfwd.h"
#include "ActionTarget.h"
#include "SpinAPIfwd.h"

namespace RunSection
{
	class RunSection
	{
	private:
		// Store pointers to the tasks and actions to avoid the splicing problem
		std::vector<std::shared_ptr<BasicTask>> tasks; // Calculations, save/load, etc.
		std::vector<std::shared_ptr<Action>> actions;  // Actions to be performed at each step
		std::vector<SpinAPI::output_ptr> outputs;	   // StandardOutput objects to provide information about action targets
		std::shared_ptr<Settings> settings;

		// Collections of SpinAPI objects
		std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems;

		// Associate containers of ActionTargets
		std::map<std::string, ActionScalar> actionScalars;
		std::map<std::string, ActionVector> actionVectors;

		// Commandline options
		bool overruleAppend; // "--append"/"-a"
		bool noCalculations; // "--no-calc"/"-z"

		// Method that creates task objects - add new task classes here
		// NOTE: Located in the seperate file "RunSection_CreateTask.cpp"
		std::shared_ptr<BasicTask> CreateTask(const std::string &, const MSDParser::ObjectParser &);

	public:
		// public variables

		// Constructors / Destructors
		RunSection();							 // Normal constructor
		RunSection(const RunSection &) = delete; // Disable Copy-constructor
		~RunSection();							 // Destructor

		// Operators
		const RunSection &operator=(const RunSection &) = delete; // Disable Copy-assignment

		// Public non-const methods
		bool Run(unsigned int);											  // Execute the Run section
		bool Run(std::string, unsigned int);							  // Execute the Run section starting from the task with the given name. Nothing is run if the name was not found
		bool Step(unsigned int);										  // Advance a single step (i.e. calling actions), with the current step number
		bool Add(MSDParser::ObjectType, const MSDParser::ObjectParser &); // Add objects
		bool Add(std::string, const ActionScalar &);
		bool Add(std::string, const ActionVector &);
		bool Add(std::shared_ptr<SpinAPI::SpinSystem>);
		bool Add(const SpinAPI::output_ptr &);

		// Public get-object-by-name methods
		std::shared_ptr<BasicTask> GetTask(const std::string &);
		// std::vector<std::shared_ptr<Action>> GetActions() const { return this->actions; };

		// Methods to get ActionTargets
		std::map<std::string, ActionScalar> GetActionScalars();
		std::map<std::string, ActionVector> GetActionVectors();

		// Public const methods
		std::shared_ptr<const Settings> GetSettings() const { return this->settings; };
		void PrintSystems(bool) const;

		// Set the commandline options
		void SetOverruleAppend(bool _overruleAppend) { this->overruleAppend = _overruleAppend; };	  // "--append"/"-a"
		void SetNoCalculationsMode(bool _noCalculations) { this->noCalculations = _noCalculations; }; // "--no-calc"/"-z"

		// Public output object methods
		bool WriteOutputHeader(std::ostream &) const; // Writes the headers for the output columns
		bool WriteOutput(std::ostream &) const;		  // Writes standard output (information about action targets)

		// void SetThreads(int value)
		//{
		//	threads = value;
		// }

		// int GetThreads()
		//{
		//	return threads;
		// }
		//  BasicTask can access SpinSystems and Settings to pass on to child task classes
		friend class BasicTask;
	};
}

#endif
