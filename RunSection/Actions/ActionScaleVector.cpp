/////////////////////////////////////////////////////////////////////////
// ActionScaleVector implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ActionScaleVector.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// ActionScaleVector Constructors and Destructor
	// -----------------------------------------------------
	ActionScaleVector::ActionScaleVector(const MSDParser::ObjectParser &_parser, const std::map<std::string, ActionScalar> &_scalars, const std::map<std::string, ActionVector> &_vectors) : Action(_parser, _scalars, _vectors), actionVector(nullptr)
	{
	}

	ActionScaleVector::~ActionScaleVector()
	{
	}
	// -----------------------------------------------------
	// ActionScaleVector derived methods
	// -----------------------------------------------------
	// Method to perform a step
	bool ActionScaleVector::DoStep()
	{
		// Make sure we have an ActionVector to act on
		if (actionVector == nullptr || !this->IsValid())
		{
			return false;
		}

		// Retrieve the vector we want to scale
		arma::vec vec = actionVector->Get();
		if (!vec.is_finite())
			return false;

		// Scale the vector
		vec *= this->Value();
		if (!vec.is_finite())
			return false;

		// Set the new vector
		return this->actionVector->Set(vec);
	}

	// Method to prepare the action and check whether it is valid
	bool ActionScaleVector::DoValidate()
	{
		// Get the ActionVector name
		std::string str;
		if (!this->Properties()->Get("actionvector", str) && !this->Properties()->Get("vector", str))
		{
			std::cout << "ERROR: No ActionVector specified for the ScaleVector action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to set the ActionVector
		if (!this->Vector(str, &(this->actionVector)))
		{
			std::cout << "ERROR: Could not find ActionVector \"" << str << "\" specified for the ScaleVector action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Readonly ActionTargets cannot be acted on
		if (this->actionVector->IsReadonly())
		{
			std::cout << "ERROR: Readonly ActionVector \"" << str << "\" specified for the ScaleVector action \"" << this->Name() << "\"! Cannot act on this vector!" << std::endl;
			return false;
		}

		if (!this->actionVector->Get().is_finite())
		{
			std::cout << "ERROR: ActionVector \"" << str << "\" has non-finite components!" << std::endl;
			return false;
		}

		return true;
	}

	bool ActionScaleVector::Reset()
	{
		this->actionVector->Reset();
		return true;
	}
	// -----------------------------------------------------
}
