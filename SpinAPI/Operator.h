/////////////////////////////////////////////////////////////////////////
// Operator class (SpinAPI Module)
// ------------------
// Special operators to be used in some task types.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Operator
#define MOD_SpinAPI_Operator

#include <memory>
#include <vector>
#include "SpinAPIDefines.h"
#include "MSDParserfwd.h"
#include "SpinAPIfwd.h"

namespace SpinAPI
{
	class Operator
	{
	private:
		// Implementation details
		std::shared_ptr<MSDParser::ObjectParser> properties;
		OperatorType type;
		std::vector<spin_ptr> spins;
		double rate1;
		double rate2;
		double rate3;
		RelaxationFrame relaxationFrame;
		bool isValid;

	public:
		// Constructors / Destructors
		Operator(std::string, std::string); // Normal constructor
		Operator(const Operator &);			// Copy-constructor
		~Operator();						// Destructor

		// Operators
		const Operator &operator=(const Operator &); // Copy-assignment

		// Name and validation
		std::string Name() const;
		bool Validate(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &);
		bool IsValid() const;

		// Public property methods
		OperatorType Type() const;
		std::vector<spin_ptr> Spins() const;
		unsigned int SpinCount() const;
		double Rate1() const;
		double Rate2() const;
		double Rate3() const;
		// Controls whether Cartesian spin-operator axes remain lab-fixed or
		// follow molecular powder orientations before basis transformation.
		RelaxationFrame Frame() const;

		// Allow access to custom properties to be used for custom tasks
		std::shared_ptr<const MSDParser::ObjectParser> Properties() const;
	};

	// Define alias for Operator-pointers
	using operator_ptr = std::shared_ptr<Operator>;

	// Non-member non-friend functions
	bool IsValid(const Operator &);
}

#endif
