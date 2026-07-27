/////////////////////////////////////////////////////////////////////////
// BasicTask (RunSection module)
// ------------------
// Base class for task/calculation methods that go in the Run section
// of the input file. I.e. all types of calculation (time integration,
// time propagation, eigenvalue calculation, etc.) is implemented as a
// class inheriting BasicTask.
//
// A RunSection contains a list of BasicTasks whoose Run methods are
// called on every step.
//
// The BasicTask class has access to all the loaded information from
// the input file, i.e. properties of spins, interactions, etc.
// ------------------
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_BasicTask
#define MOD_RunSection_BasicTask

#include <memory>
#include <vector>
#include "OutputHandler.h"
#include "RunSectionfwd.h"
#include "MSDParserfwd.h"
#include "SpinAPIfwd.h"
#include "ActionTarget.h"
#include "Utility.h"
#include "SpinAPIDefines.h"
#include "PowderGrid.h"

namespace RunSection
{

	enum class Propagator
	{
		Default = 0, //whatever the timeevo class chooses
		exp = 1,
		RK4 = 2,
		RK45 = 3
	};

	enum class TaskName //currently used to verify what tasks support SW
	{
		DEFULAT = 0, 
		//add tasknames below here
		STATICSS = 1,
		STATICSS_TIMEVO
	};

	struct TimeEvoProperties
	{
		Propagator prop;
		double TotalTIme;
		double TimeStep;
		double MinTimeStep;
		double MaxTImeStep;
	};

	class BasicTask
	{
	private:
		// Implementation
		std::shared_ptr<MSDParser::ObjectParser> properties; // Use a pointer to the object to minimize compilation dependencies
		const RunSection &runsection;
		OutputHandler output;
		bool isValid;
		bool isValidated;
		const std::map<std::string, ActionScalar> *scalars;
		const std::map<std::string, ActionVector> *vectors;
		std::map<std::string, std::pair<ActionScalar *, double>> usedScalars;
		std::map<std::string, std::pair<ActionVector *, arma::vec>> usedVectors;

		// Private method that calls the pure virtual Validate method, and allows BasicTask to do some extra initialization first
		// (i.e. setting the notification level, as that cannot be done in the constructor since the runsection.settings object must be properly loaded first)
		bool DoValidation();

		// Other private methods
		void ResetTimeAndTrajectoryStep();

	protected:
		// Run methods to be overwritten in derived classes
		virtual bool RunLocal() = 0; // Normal run method for a single workstation
		virtual bool RunMPI();		 // MPI run method to use on a supercomputer cluster (not required to be implemented)
		virtual bool Validate() = 0; // Method to validate the task, i.e. to check that it has the required parameters etc.
		
		virtual void SelectPropagator(std::string str); //Method to choose the Propagator for timeevo tasks

		// Allow access to settings, properties, spin systems, etc. for derived classes
		std::shared_ptr<const Settings> RunSettings() const;
		std::vector<SpinAPI::system_ptr> SpinSystems() const;
		const std::shared_ptr<MSDParser::ObjectParser> &Properties() const;
		std::ostream &Log(const MessageType &_msgtype = MessageType_Normal);
		std::ostream &Data();
		bool WriteStandardOutputHeader(std::ostream &);
		bool WriteStandardOutput(std::ostream &);
		const std::vector<std::shared_ptr<Action>> &Actions() const;
		bool ReadExplicitPowderGrid(SpinAPI::PowderGrid &);

		// ActionTarget access
		bool Scalar(std::string _name, ActionScalar **_scalar = nullptr);
		bool Vector(std::string _name, ActionVector **_vector = nullptr);

		Propagator prop;
		TaskName name;
		
		//semi classical 
		virtual void GetSamples(std::vector<arma::sp_cx_mat>&, arma::sp_cx_mat&, std::vector<SCData>&, std::vector<std::vector<double>>&, std::vector<std::vector<std::vector<double>>>&);
		virtual void SCDirectEvaluation(SpinAPI::system_ptr&, arma::sp_cx_mat&, SCData&, arma::cx_vec&, arma::cx_vec&);
		//virtual void SCTimeIntegration(SpinAPI::system_ptr&, SpinAPI::SpinSpace&, arma::sp_cx_mat&, SCData&, arma::cx_vec&, TimeEvoProperties&);

		struct SCIntegrationProperties
		{
			std::vector<double> maxBondLenght;
			std::vector<int> numSamples;
			std::vector<std::vector<double>> spacing;
		};

		std::pair<arma::cx_vec,double> IntegrateSC(std::vector<std::pair<int,arma::cx_vec>>&, std::vector<std::pair<int,double>>&, SCIntegrationProperties);

	public:
		// Constructors / Destructors
		BasicTask(const MSDParser::ObjectParser &, const RunSection &); // Normal constructor
		BasicTask(const BasicTask &) = delete;							// Default Copy-constructor - no need to copy a BasicTask which could cause splicing problems
		virtual ~BasicTask();											// Destructor

		// Operators
		BasicTask &operator=(const BasicTask &) = delete; // Default Copy-assignment

		// Public methods
		bool Run();
		bool IsValid();
		std::string Name();

		// Method to provide BasicTask with access to ActionTargets
		void SetActionTargets(const std::map<std::string, ActionScalar> &, const std::map<std::string, ActionVector> &);

		// Methods to change the Log and Data stream
		bool SetLogStream(std::ostream &);
		bool SetDataStream(std::ostream &);

	};
}

#endif
