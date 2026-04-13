/////////////////////////////////////////////////////////////////////////
// ActionAddScalar implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ActionAddScalar.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// ActionAddScalar Constructors and Destructor
	// -----------------------------------------------------
	ActionAddScalar::ActionAddScalar(const MSDParser::ObjectParser &_parser, const std::map<std::string, ActionScalar> &_scalars, const std::map<std::string, ActionVector> &_vectors) : Action(_parser, _scalars, _vectors), actionScalar(nullptr)
	{
	}

	ActionAddScalar::~ActionAddScalar()
	{
	}
	// -----------------------------------------------------
	// ActionAddScalar derived methods
	// -----------------------------------------------------
	// Method to perform a step
	bool ActionAddScalar::DoStep()
	{
		// Make sure we have an ActionScalar to act on
		if (actionScalar == nullptr || !this->IsValid())
		{
			return false;
		}

		// Retrieve the scalar we want to change
		double d = actionScalar->Get();

		// Set the new scalar
		return this->actionScalar->Set(d + this->Value());
	}

	// Method to prepare the action and check whether it is valid
	bool ActionAddScalar::DoValidate()
	{
		// Get the ActionScalar name
		std::string str;
		if (!this->Properties()->Get("actionscalar", str) && !this->Properties()->Get("scalar", str))
		{
			std::cout << "ERROR: No ActionScalar specified for the AddScalar action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to set the ActionScalar
		if (!this->Scalar(str, &(this->actionScalar)))
		{
			std::cout << "ERROR: Could not find ActionScalar \"" << str << "\" specified for the AddScalar action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Readonly ActionTargets cannot be acted on
		if (this->actionScalar->IsReadonly())
		{
			std::cout << "ERROR: Readonly ActionScalar \"" << str << "\" specified for the AddScalar action \"" << this->Name() << "\"! Cannot act on this scalar!" << std::endl;
			return false;
		}

		if(this->m_ActionMode == MODE::linespace)
		{
			this->m_startScaler = this->actionScalar->Get();
			double diff = this->m_targetScalar - this->m_startScaler;
			if(diff < 0.0)
			{
				return false;
			}
			this->value = diff/((double)this->m_num_steps-1.0);
			this->first = 1;
			this->last = this->m_num_steps;
		}

		return true;
	}

	bool ActionAddScalar::Reset()
	{
		this->actionScalar->Reset();
		return true;
	}
	// -----------------------------------------------------
}
