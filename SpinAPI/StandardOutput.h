/////////////////////////////////////////////////////////////////////////
// StandardOutput class (SpinAPI Module)
// ------------------
// Specifier of standard output information for a spin system.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_StandardOutput
#define MOD_SpinAPI_StandardOutput

#include <map>
#include <memory>
#include "SpinAPIDefines.h"
#include "ActionTarget.h"

namespace SpinAPI
{
	class StandardOutput
	{
	private:
		// Data members
		std::shared_ptr<MSDParser::ObjectParser> properties; // Use a pointer to the object to minimize compilation dependencies
		char delimiter;
		bool isValid;
		RunSection::ActionScalar *scalar;
		RunSection::ActionVector *vector;
		StandardOutputType type;
		arma::vec reference;
		std::string actionTargetName;
		double prefactor;

		// NOTE: The strings returned here should NOT contain any newline-characters!
		std::string Header() const; // String to write column header(s)
		std::string Data() const;	// String to write a row in data column(s)

		// Private methods
		bool setActionVector(const std::map<std::string, RunSection::ActionVector> &);
		bool setActionScalar(const std::map<std::string, RunSection::ActionScalar> &);

	public:
		// Constructors / Destructors
		StandardOutput(std::string, std::string);		 // Normal constructor
		StandardOutput(const StandardOutput &) = delete; // Copy-constructor
		~StandardOutput();								 // Destructor

		// Operators
		const StandardOutput operator=(const StandardOutput &) = delete; // Copy-assignment

		// Public methods
		std::string Name() const;
		bool IsValid() const;
		bool Validate(const std::map<std::string, RunSection::ActionScalar> &, const std::map<std::string, RunSection::ActionVector> &);
		bool SetDelimiter(char);

		// Public output methods
		bool WriteOutputHeader(std::ostream &) const; // Writes the header for the output column(s)
		bool WriteOutput(std::ostream &) const;		  // Writes output data

		// Global output precision for StandardOutput data
		static void SetDefaultPrecision(unsigned int);
		static unsigned int DefaultPrecision();
	};

	// Define alias for StandardOutput-pointers
	using output_ptr = std::shared_ptr<StandardOutput>;
}

#endif
