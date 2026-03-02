/////////////////////////////////////////////////////////////////////////
// Settings (RunSection module)
// ------------------
// Settings for a RunSection class instance, as specified in the Settings
// object of the MSD input files.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_Settings
#define MOD_RunSection_Settings

#include "MSDParserfwd.h"
#include "RunSectionDefines.h"
#include "ActionTarget.h"

namespace RunSection
{
	class Settings
	{
	private:
		// Implementation
		unsigned int steps;
		unsigned int currentStep;
		char dataDelimiter;
		MessageType notificationLevel;
		double time;
		unsigned int trajectoryStep;
		bool setTrajectoryStepBeforeTime;
		unsigned int dataPrecision;
		unsigned int logPrecision;
		//bool m_Parallelize;

	public:
		// Constructors / Destructors
		Settings();					// Normal constructor
		Settings(const Settings &); // Default Copy-constructor, no need to copy, so better delete than end up with splicing problems
		~Settings();				// Destructor

		// Operators
		const Settings &operator=(const Settings &); // Default Copy-assignment

		// Update method
		void Update(const MSDParser::ObjectParser &);

		// Public methods to change parameters
		void SetCurrentStep(unsigned int);

		// Public const methods
		unsigned int Steps() const { return this->steps; };
		unsigned int CurrentStep() const { return this->currentStep; };
		char DataDelimiter() const { return this->dataDelimiter; };
		MessageType NotificationLevel() const { return this->notificationLevel; };
		double Time() const { return this->time; };
		unsigned int TrajectoryStep() const { return this->trajectoryStep; };
		bool SetTrajectoryStepBeforeTime() const { return this->setTrajectoryStepBeforeTime; };
		unsigned int DataPrecision() const { return this->dataPrecision; };
		unsigned int LogPrecision() const { return this->logPrecision; };
		//bool GetParallel() const { return this->m_Parallelize; };

		// Method to define ActionTargets for the Settings object
		void GetActionTargets(std::map<std::string, ActionScalar> &, std::map<std::string, ActionVector> &);

		// Default values
		const unsigned int DefaultSteps = 1;
		const double DefaultTime = 0.0;
		const unsigned int DefaultTrajectoryStep = 0;
		const unsigned int DefaultDataPrecision = 6;
	};

	// Non-member non-friend functions for ActionTarget validation
	bool CheckActionScalarSettingsTime(const double &);
}

#endif
