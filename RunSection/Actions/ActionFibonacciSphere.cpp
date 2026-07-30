/////////////////////////////////////////////////////////////////////////
// ActionFibonacciSphere implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ActionFibonacciSphere.h"
#include "ObjectParser.h"

#include <array>
#include "Utility.h"

namespace RunSection
{
	ActionFibonacciSphere::ActionFibonacciSphere(const MSDParser::ObjectParser &_parser, const std::map<std::string, ActionScalar> &_scalars, const std::map<std::string, ActionVector> &_vectors)
	    : Action(_parser, _scalars, _vectors), actionVector(nullptr), m_Points(nullptr), m_Num(0), m_Step(0), m_Magnitude(1.0)
	{
	}

    ActionFibonacciSphere::~ActionFibonacciSphere()
    {
        if (m_Points != nullptr)
        {
            free(m_Points);
        }
    }

	bool ActionFibonacciSphere::CalculatePoints(int n)
	{
	    if (n < 2)
	        return false;

	    if (m_Points != nullptr)
	    {
	        free(m_Points);
	        m_Points = nullptr;
	    }
	    m_Points = CalculateFibPoints(n);

        if (m_Points == NULL)
        {
            std::cout << "Memory not allocated" << std::endl;
            return false;
        }

        return true;
    }

    bool ActionFibonacciSphere::GetPoint(std::array<double, 3> &arr)
    {
        if (m_Step >= m_Num)
        {
            return false;
        }
        bool PointRetrieved = RetrievePoint(arr,m_Points,m_Step);

        double x = m_Magnitude * arr[0];
        double y = m_Magnitude * arr[1];
        double z = m_Magnitude * arr[2];

        m_Step++;

        arr = {x, y, z};
        return PointRetrieved;
    }

    bool ActionFibonacciSphere::DoStep()
    {

        // Make sure we have an ActionVector to act on
        if (actionVector == nullptr || !this->IsValid())
        {
            return false;
        }

        // Retrieve the vector we want to change
        std::array<double, 3> points = {0, 0, 0};
	    if (!GetPoint(points))
	        return true; // All requested grid points have already been emitted.
	    arma::vec vec = {points[0], points[1], points[2]};

        return this->actionVector->Set(vec);
    }

    bool ActionFibonacciSphere::DoValidate()
    {
        std::string str;
        if (!this->Properties()->Get("actionvector", str) && !this->Properties()->Get("vector", str))
        {
            std::cout << "ERROR: No ActionVector specified for the FibonacciSphere action \"" << this->Name() << "\"!" << std::endl;
            return false;
        }

        int NumPoints = 0;
	    if (!this->Properties()->Get("points", NumPoints))
        {
            std::cout << "ERROR: No Number of points specified for the FibonacciSphere action \"" << this->Name() << "\"!" << std::endl;
	        return false;
	    }

	    if (NumPoints < 2)
	    {
	        std::cout << "ERROR: FibonacciSphere action \"" << this->Name() << "\" requires at least two points." << std::endl;
	        return false;
	    }

	    m_Num = NumPoints;

	    // Attemp to set the ActionVector
        if (!this->Vector(str, &(this->actionVector)))
        {
            std::cout << "ERROR: Could not find ActionVector \"" << str << "\" specified for the FibonacciSphere action \"" << this->Name() << "\"!" << std::endl;
            return false;
        }

        // Readonly ActionTargets cannot be acted on
        if (this->actionVector->IsReadonly())
        {
            std::cout << "ERROR: Read only ActionVector \"" << str << "\" specified for the FibonacciSphere action \"" << this->Name() << "\"! Cannot act on this vector!" << std::endl;
	        return false;
	    }

	    auto vec1 = this->actionVector->Get();
	    if (vec1.n_elem != 3 || !vec1.is_finite())
	    {
	        std::cout << "ERROR: FibonacciSphere action \"" << this->Name() << "\" requires a finite three-component target vector." << std::endl;
	        return false;
	    }

	    m_Magnitude = arma::norm(vec1);
	    if (!std::isfinite(m_Magnitude) || !CalculatePoints(m_Num))
	        return false;

	    m_Step = 0;
	    // Install point zero before calculation step 1 only for the default
	    // schedule. A delayed grid starts when Action::Step reaches first.
	    if (this->first == 1)
	    {
	        std::array<double, 3> points = {0, 0, 0};
	        if (!GetPoint(points))
	            return false;

	        arma::vec vec = {points[0], points[1], points[2]};
	        if (!this->actionVector->Set(vec))
	            return false;
	    }
        return true;
    }

    bool ActionFibonacciSphere::Reset()
    {
	    m_Step = 0;
	    std::array<double, 3> points = {0, 0, 0};
	    if (!GetPoint(points))
	        return false;
	    arma::vec vec = {points[0], points[1], points[2]};
	    return this->actionVector->Set(vec);
	}
}
