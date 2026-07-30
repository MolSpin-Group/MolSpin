/////////////////////////////////////////////////////////////////////////
// ActionMultiplyScalar implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ActionMultiplyScalar.h"
#include "ObjectParser.h"

#include <cmath>

namespace RunSection
{
	// -----------------------------------------------------
	// ActionMultiplyScalar Constructors and Destructor
	// -----------------------------------------------------
	ActionMultiplyScalar::ActionMultiplyScalar(const MSDParser::ObjectParser &_parser, const std::map<std::string, ActionScalar> &_scalars, const std::map<std::string, ActionVector> &_vectors) : Action(_parser, _scalars, _vectors), actionScalar(nullptr)
	{
	}

	ActionMultiplyScalar::~ActionMultiplyScalar()
	{
	}
	// -----------------------------------------------------
	// ActionMultiplyScalar derived methods
	// -----------------------------------------------------
	// Method to perform a step
	bool ActionMultiplyScalar::DoStep()
	{
		// Make sure we have an ActionScalar to act on
		if (actionScalar == nullptr || !this->IsValid())
		{
			return false;
		}

		// Retrieve the scalar we want to change
		double d = actionScalar->Get();
		const double result = d * this->Value();
		if (!std::isfinite(result))
			return false;

		// Set the new scalar
		return this->actionScalar->Set(result);
	}

	// Method to prepare the action and check whether it is valid
	bool ActionMultiplyScalar::DoValidate()
	{
		// Get the ActionScalar name
		std::string str;
		if (!this->Properties()->Get("actionscalar", str) && !this->Properties()->Get("scalar", str))
		{
			std::cout << "ERROR: No ActionScalar specified for the MultiplyScalar action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to set the ActionScalar
		if (!this->Scalar(str, &(this->actionScalar)))
		{
			std::cout << "ERROR: Could not find ActionScalar \"" << str << "\" specified for the MultiplyScalar action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Readonly ActionTargets cannot be acted on
		if (this->actionScalar->IsReadonly())
		{
			std::cout << "ERROR: Readonly ActionScalar \"" << str << "\" specified for the MultiplyScalar action \"" << this->Name() << "\"! Cannot act on this scalar!" << std::endl;
			return false;
		}

		if (!std::isfinite(this->actionScalar->Get()))
		{
			std::cout << "ERROR: ActionScalar \"" << str << "\" has a non-finite value!" << std::endl;
			return false;
		}

		return true;
	}

	bool ActionMultiplyScalar::Reset()
	{
		this->actionScalar->Reset();
		return true;
	}
	// -----------------------------------------------------
}
