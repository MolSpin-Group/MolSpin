/////////////////////////////////////////////////////////////////////////
// ActionFibonacciSphere (RunSection module)
// ------------------
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_ActionFibonacciSphere
#define MOD_RunSection_ActionFibonacciSphere

#include "Action.h"

#include <array>
#include <tuple>
#include <vector>

namespace RunSection
{
	class ActionFibonacciSphere : public Action
	{
	private:
		// Data members
		ActionVector *actionVector;

		// std::vector<std::pair<float,float>> points; //stores y and theta
		typedef std::pair<float, float> point;
		point *m_Points; // stores y and theta
		int m_Num;
		int m_Step;
		double m_Magnitude; // Maginitude of the intial vector;

	private:
		bool CalculatePoints(int n);
		bool GetPoint(std::array<double, 3> &);

	protected:
		// Overwritten protected methods
		bool DoStep() override;
		bool DoValidate() override;
		bool Reset() override;

	public:
		// Constructors / Destructors
		ActionFibonacciSphere(const MSDParser::ObjectParser &, const std::map<std::string, ActionScalar> &, const std::map<std::string, ActionVector> &); // Normal constructor
		~ActionFibonacciSphere();
	};

}

#endif
