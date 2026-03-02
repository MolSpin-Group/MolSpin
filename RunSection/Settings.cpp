/////////////////////////////////////////////////////////////////////////
// Settings implementation (RunSection module)
//
// Settings for a RunSection class instance, as specified in the Settings
// object of the MSD input files.
// Includes default values for all the settings.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "Settings.h"
#include "ObjectParser.h"
#include <limits>

namespace RunSection
{
	// -----------------------------------------------------
	// Settings Constructors and Destructor
	// -----------------------------------------------------
	Settings::Settings() : steps(1),
						   currentStep(1),
						   dataDelimiter(' '),
						   notificationLevel(DefaultNotificationLevel),
						   time(0),
						   trajectoryStep(0),
						   setTrajectoryStepBeforeTime(true),
						   dataPrecision(DefaultDataPrecision),
						   logPrecision(DefaultDataPrecision)
						   //m_Parallelize(false)
	{
		// Note: These cannot be initialized in the initializer list as the default values are initialized after the fields (unless header file is reordered)
		steps = DefaultSteps;
		time = DefaultTime;
		trajectoryStep = DefaultTrajectoryStep;
		dataPrecision = DefaultDataPrecision;
		logPrecision = DefaultDataPrecision;
	}

	Settings::Settings(const Settings &_settings) : steps(_settings.steps), currentStep(_settings.currentStep), dataDelimiter(_settings.dataDelimiter), notificationLevel(_settings.notificationLevel),
													time(_settings.time), trajectoryStep(_settings.trajectoryStep), setTrajectoryStepBeforeTime(_settings.setTrajectoryStepBeforeTime),
													dataPrecision(_settings.dataPrecision), logPrecision(_settings.logPrecision) //, m_Parallelize(_settings.m_Parallelize)
	{
	}

	Settings::~Settings()
	{
	}
	// -----------------------------------------------------
	// Copy-assignment operator
	// -----------------------------------------------------
	const Settings &Settings::operator=(const Settings &_settings)
	{
		this->steps = _settings.steps;
		this->currentStep = _settings.currentStep;
		this->dataDelimiter = _settings.dataDelimiter;
		this->notificationLevel = _settings.notificationLevel;
		this->time = _settings.time;
		this->trajectoryStep = _settings.trajectoryStep;
		this->setTrajectoryStepBeforeTime = _settings.setTrajectoryStepBeforeTime;
		this->dataPrecision = _settings.dataPrecision;
		this->logPrecision = _settings.logPrecision;
		//this->m_Parallelize = _settings.m_Parallelize;

		return (*this);
	}
	// -----------------------------------------------------
	// Settings public methods
	// -----------------------------------------------------
	// Update all settings from an ObjectParser object
	void Settings::Update(const MSDParser::ObjectParser &_settings)
	{
		// Get the number of calculation steps
		_settings.Get("steps", this->steps);

		// Get the trajectory step to use for all tasks (task classes can redefine the trajectory step if they want)
		if (_settings.Get("trajectorystep", this->trajectoryStep))
		{
			this->setTrajectoryStepBeforeTime = false; // If only trajectory step is specified, make sure to set that last (overriding time)
		}

		// Get the time parameter to use for all tasks (task classes can redefine the time if they want)
		double tmpTime;
		if (_settings.Get("time", tmpTime))
		{
			if (std::isfinite(tmpTime))
			{
				this->time = tmpTime;
				this->setTrajectoryStepBeforeTime = true; // Use time over trajectory step unless other behaviour is requested
			}
			else
			{
				std::cout << "Warning: Invalid value specified for time: " << tmpTime << "\nIgnoring time specification!" << std::endl;
			}
		}

		// Allow the user to give the trajectory step priority over time
		_settings.Get("trajectorystepbeforetime", this->setTrajectoryStepBeforeTime);

		// Output precision for data/log streams
		unsigned int prec = 0;
		if (_settings.Get("outputprecision", prec) || _settings.Get("dataprecision", prec) ||
			_settings.Get("data_precision", prec) || _settings.Get("output_precision", prec))
		{
			const unsigned int maxPrec = std::numeric_limits<double>::max_digits10;
			if (prec >= 1 && prec <= maxPrec)
				this->dataPrecision = prec;
		}
		if (_settings.Get("logprecision", prec) || _settings.Get("log_precision", prec))
		{
			const unsigned int maxPrec = std::numeric_limits<double>::max_digits10;
			if (prec >= 1 && prec <= maxPrec)
				this->logPrecision = prec;
		}
		else
		{
			this->logPrecision = this->dataPrecision;
		}

		// TODO: Allow the user to specify dataDelimiter

		// If a general notification level was specified, attempt to parse it
		std::string str;
		if (_settings.Get("notifications", str) || _settings.Get("notificationlevel", str))
		{
			if (str.compare("details") == 0 || str.compare("showdetails") == 0 || str.compare("all") == 0 || str.compare("verbose") == 0)
			{
				this->notificationLevel = MessageType_Details;
			}
			else if (str.compare("normal") == 0 || str.compare("standard") == 0)
			{
				this->notificationLevel = MessageType_Normal;
			}
			else if (str.compare("sparse") == 0 || str.compare("important") == 0 || str.compare("importantonly") == 0)
			{
				this->notificationLevel = MessageType_Important;
			}
			else if (str.compare("quiet") == 0 || str.compare("silent") == 0 || str.compare("critical") == 0 || str.compare("criticalonly") == 0)
			{
				this->notificationLevel = MessageType_Critical;
			}
			else
			{
				std::cout << "Failed to set the general notification level! Unrecognized value \"" << str << "\"!" << std::endl;
			}
		}

		// The possibility to ignore warnings (not recommended, but possible)
		bool flag;
		if (_settings.Get("ignorewarnings", flag) || _settings.Get("nowarn", flag))
		{
			if (flag)
				this->notificationLevel &= ~MessageType_Warning; // Reset the warning flag
			else
				this->notificationLevel |= MessageType_Warning; // Set the warning flag
		}
		else
		{
			this->notificationLevel |= MessageType_Warning;
		}

		// The possibility to ignore errors (not recommended, but possible)
		if (_settings.Get("ignoreerrors", flag) || _settings.Get("noerr", flag))
		{
			if (flag)
				this->notificationLevel &= ~MessageType_Error; // Reset the error flag
			else
				this->notificationLevel |= MessageType_Error; // Set the error flag
		}
		else
		{
			this->notificationLevel |= MessageType_Error;
		}

		//std::string parallelize;
		//if(_settings.Get("parallelize", parallelize))
		//{
		//	if(parallelize.compare("true"))
		//		this->m_Parallelize = true;
		//	else
		//		this->m_Parallelize = false;
		//}
	}

	// Update the current step
	void Settings::SetCurrentStep(unsigned int _currentStep)
	{
		this->currentStep = _currentStep;
	}
	// -----------------------------------------------------
	// Public method to define ActionTargets
	// -----------------------------------------------------
	// Defines ActionTargets for the Settings object
	void Settings::GetActionTargets(std::map<std::string, ActionScalar> &_scalars, std::map<std::string, ActionVector> &_vectors)
	{
		// Prepare vectors for the created ActionTargets
		std::vector<NamedActionScalar> scalars;
		// std::vector<NamedActionVector> vectors;

		// Suppress warning about unused parameter. Remove if _vectors is getting used.
		(void)_vectors;

		// Create ActionScalars
		ActionScalar settingsTimeScalar = ActionScalar(this->time, &CheckActionScalarSettingsTime, false);
		scalars.push_back(NamedActionScalar("settings.time", settingsTimeScalar));

		// Insert all the ActionScalars
		for (auto i = scalars.cbegin(); i != scalars.cend(); i++)
		{
			auto j = _scalars.insert((*i));
			if (!j.second)
				std::cout << "Could not create ActionScalar named \"" << j.first->first << "\" because it already exists!" << std::endl;
		}

		// Insert all the ActionVectors
		/*for(auto i = vectors.cbegin(); i != vectors.cend(); i++)
		{
			auto j = _vectors.insert((*i));
			if(!j.second)
				std::cout << "Could not create ActionVector named \"" << j.first->first << "\" because it already exists!" << std::endl;
		}*/
	}
	// -----------------------------------------------------
	// Non-member non-friend ActionTarget Check functions
	// -----------------------------------------------------
	// Make sure that the time has a valid value (not NaN or infinite)
	bool CheckActionScalarSettingsTime(const double &_d)
	{
		return std::isfinite(_d);
	}
	// -----------------------------------------------------
}
