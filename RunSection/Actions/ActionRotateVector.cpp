/////////////////////////////////////////////////////////////////////////
// ActionRotateVector implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ActionRotateVector.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// ActionRotateVector Constructors and Destructor
	// -----------------------------------------------------
	ActionRotateVector::ActionRotateVector(const MSDParser::ObjectParser &_parser, const std::map<std::string, ActionScalar> &_scalars, const std::map<std::string, ActionVector> &_vectors) : Action(_parser, _scalars, _vectors), actionVector(nullptr), axis(), refAxis1(), refAxis2()
	{
	}

	ActionRotateVector::~ActionRotateVector()
	{
	}
	// -----------------------------------------------------
	// ActionRotateVector derived methods
	// -----------------------------------------------------
	// Method to perform a step
	bool ActionRotateVector::DoStep()
	{
		// Make sure we have an ActionVector to act on and an axis defined
		if (actionVector == nullptr || !this->IsValid())
		{
			return false;
		}

		// Retrieve the vector we want to rotate
		arma::vec vec = actionVector->Get();

		// Subtract the projection along the rotation axis
		// Hence vec will then lie in the orthogonal plane
		double axisProj = dot(this->axis, vec);
		vec -= this->axis * axisProj;

		// Obtain the length of the projection onto the plane orthogonal to the axis
		double planeLength = norm(vec);

		// Obtain a projection in the orthogonal plane
		double ref1Proj = dot(this->refAxis1, vec);
		double ref2Proj = dot(this->refAxis2, vec);

		// Get the current angle in the plane
		double angle = atan2(ref2Proj, ref1Proj);

		// Do the rotation
		angle += this->Value() / 180.0 * arma::datum::pi;

		// Get the new vector
		vec = planeLength * (this->refAxis1 * cos(angle) + this->refAxis2 * sin(angle)) + this->axis * axisProj;

		// Set the new vector
		return this->actionVector->Set(vec);
	}

	// Method to prepare the action and check whether it is valid
	bool ActionRotateVector::DoValidate()
	{
		// Make sure that an axis is specified
		arma::vec tmp;
		if (!this->Properties()->Get("axis", tmp))
		{
			std::cout << "ERROR: No axis specified for the RotateVector action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to set the axis
		if (!this->SetAxis(tmp))
		{
			std::cout << "ERROR: Invalid axis specified for the RotateVector action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Get the ActionVector name
		std::string str;
		if (!this->Properties()->Get("actionvector", str) && !this->Properties()->Get("vector", str))
		{
			std::cout << "ERROR: No ActionVector specified for the RotateVector action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to set the ActionVector
		if (!this->Vector(str, &(this->actionVector)))
		{
			std::cout << "ERROR: Could not find ActionVector \"" << str << "\" specified for the RotateVector action \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Readonly ActionTargets cannot be acted on
		if (this->actionVector->IsReadonly())
		{
			std::cout << "ERROR: Readonly ActionVector \"" << str << "\" specified for the RotateVector action \"" << this->Name() << "\"! Cannot act on this vector!" << std::endl;
			return false;
		}

		return true;
	}
	// -----------------------------------------------------
	// ActionRotateVector private methods
	// -----------------------------------------------------
	// Sets the rotation axis to a unit vector
	bool ActionRotateVector::SetAxis(const arma::vec &_vec)
	{
		// Check whether we have a valid axis
		if (_vec.n_elem == 3 && _vec.is_finite())
		{
			// Set the axis
			this->axis = normalise(_vec);

			// We also need to set a reference axis, and we will choose the coordinate axis
			// with the lowest projection and subtract the axis projection
			arma::mat A(3, 3, arma::fill::eye);
			arma::uword minIndex;
			arma::vec projections = abs(A * this->axis);		  // Get projections (absolute value!)
			projections.min(minIndex);							  // Get index of minimal projection (i.e. closest to zero!)
			this->refAxis1.zeros(3);							  // Obtain a zero-vector for the reference axis
			this->refAxis1(minIndex) = 1;						  // Set it to 1 along the direction of minimal projection
			this->refAxis1 -= projections(minIndex) * this->axis; // Make it orthogonal to the axis
			this->refAxis1 = normalise(this->refAxis1);			  // Normalize the reference vector

			// Obtain the second reference axis (note: this should already be a unit vector)
			this->refAxis2 = cross(this->axis, this->refAxis1);
			this->refAxis2 = normalise(this->refAxis2); // Just in case of numerical errors

			return true;
		}

		return false;
	}

	bool ActionRotateVector::Reset()
	{
		this->actionVector->Reset();
		return true;
	}
	// -----------------------------------------------------
}
