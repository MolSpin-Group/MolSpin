/////////////////////////////////////////////////////////////////////////
// Action (RunSection module)
// ------------------
// Base class for Actions that can be triggered between calculation steps
// in order to change parameters of the spin system.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_Action
#define MOD_RunSection_Action

#include <memory>
#include <map>
#include "MSDParserfwd.h"
#include "ActionTarget.h"

namespace RunSection
{
	class Action
	{
	public:
		enum class MODE
		{
			linespace = 0,
			defualt = 1
		};
	private:
		// Data members
		std::shared_ptr<MSDParser::ObjectParser> properties; // Use a pointer to the object to minimize compilation dependencies
		const std::map<std::string, ActionScalar> &scalars;
		const std::map<std::string, ActionVector> &vectors;
		bool isValid;
		unsigned int period;
		bool m_loop;
		
		//bool m_Parallelize; 

		//addition actions only
	protected:

		unsigned int first;
		unsigned int last;

		double value;
		MODE m_ActionMode;
		double m_targetScalar;
		arma::vec m_targetVec;

		double m_startScaler;
		arma::vec m_startVec;
		bool m_ValidLineSpace;

		int m_num_steps;

	protected:
		// Protected methods
		std::shared_ptr<MSDParser::ObjectParser> Properties() const { return this->properties; };
		bool Scalar(std::string _name, ActionScalar **_scalar = nullptr) const;
		bool Vector(std::string _name, ActionVector **_vector = nullptr) const;

		// Methods to be overwritten in derived classes
		virtual bool DoStep() = 0;
		virtual bool DoValidate() = 0;
		virtual bool Reset() = 0;

	public:
		// Constructors / Destructors
		Action(const MSDParser::ObjectParser &, const std::map<std::string, ActionScalar> &, const std::map<std::string, ActionVector> &); // Normal constructor
		Action(const Action &) = delete;																								   // Default Copy-constructor, no need to copy, so better delete than end up with splicing problems
		virtual ~Action();																												   // Destructor

		// Operators
		Action &operator=(const Action &) = delete; // Default Copy-assignment

		// Public methods
		double Value() const;
		unsigned int First() const;
		unsigned int Last() const;
		unsigned int Period() const;
		void Step(unsigned int);
		bool IsValid() const;
		std::string Name();
		//bool GetParallel() const { return this->m_Parallelize; };

		const std::shared_ptr<MSDParser::ObjectParser> GetProperties() const { return this->properties; };
		const MODE GetActionMode() const {return this->m_ActionMode;};

		// Public method to be overwritten
		bool Validate();
	};
}

#endif
