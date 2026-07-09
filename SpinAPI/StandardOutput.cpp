/////////////////////////////////////////////////////////////////////////
// StandardOutput implementation (RunSection module)
// ------------------
// Specifier of standard output information for a spin system.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// #include <iostream>
#include <iomanip>
#include <limits>
#include "ObjectParser.h"
#include "StandardOutput.h"

namespace SpinAPI
{
	namespace
	{
		unsigned int &DefaultOutputPrecision()
		{
			static unsigned int precision = 6;
			return precision;
		}
	}
	// -----------------------------------------------------
	// StandardOutput Constructors and Destructor
	// -----------------------------------------------------
	StandardOutput::StandardOutput(std::string _name, std::string _contents)
		: properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents)), delimiter(' '), isValid(false), scalar(nullptr), vector(nullptr), type(StandardOutputType::Undefined), reference(), prefactor(1.0)
	{
	}

	StandardOutput::~StandardOutput()
	{
	}
	// -----------------------------------------------------
	// StandardOutput private methods
	// -----------------------------------------------------
	// Gets the action vector specified for the output object
	bool StandardOutput::setActionVector(const std::map<std::string, RunSection::ActionVector> &_vectors)
	{
		// Get the ActionVector name
		std::string str;
		if (!this->properties->Get("actionvector", str) && !this->properties->Get("vector", str))
		{
			std::cout << "ERROR: No ActionVector specified for the output object \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to find the ActionVector
		auto i = _vectors.find(str);
		if (i == _vectors.cend())
		{
			std::cout << "ERROR: Could not find ActionVector \"" << str << "\" specified for the output object \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Finally set the ActionVector
		this->vector = const_cast<RunSection::ActionVector *>(&(i->second));
		this->actionTargetName = str;

		return true;
	}

	// Gets the action scalar specified for the output object
	bool StandardOutput::setActionScalar(const std::map<std::string, RunSection::ActionScalar> &_scalars)
	{
		// Get the ActionScalar name
		std::string str;
		if (!this->properties->Get("actionscalar", str) && !this->properties->Get("scalar", str))
		{
			std::cout << "ERROR: No ActionScalar specified for the output object \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Attemp to find the ActionScalar
		auto i = _scalars.find(str);
		if (i == _scalars.cend())
		{
			std::cout << "ERROR: Could not find ActionScalar \"" << str << "\" specified for the output object \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Finally set the ActionScalar
		this->scalar = const_cast<RunSection::ActionScalar *>(&(i->second));
		this->actionTargetName = str;

		return true;
	}

	// Returns the header string
	std::string StandardOutput::Header() const
	{
		// Use a stringstream to construct the output
		std::stringstream output;

		// Write the header depending on the type
		switch (this->type)
		{
		case StandardOutputType::VectorXYZ:
			output << this->actionTargetName << ".x" << this->delimiter << this->actionTargetName << ".y" << this->delimiter << this->actionTargetName << ".z";
			break;
		case StandardOutputType::VectorLength:
			output << this->actionTargetName << ".length";
			break;
		case StandardOutputType::VectorAngle:
			output << this->actionTargetName << ".angle(" << this->Name() << ")"; // Use the name of the output object when a reference vector is used
			break;
		case StandardOutputType::VectorDot:
			output << this->actionTargetName << ".dot(" << this->Name() << ")";
			break;
		case StandardOutputType::Scalar:
			output << this->actionTargetName;
			break;
		default:
			output << "";
			break;
		}

		return output.str();
	}

	// Returns the data string
	std::string StandardOutput::Data() const
	{
		// Use a stringstream to construct the output
		std::stringstream output;
		output << std::setprecision(DefaultPrecision());

		// Write the header depending on the type
		// Note: The blocks { } are used to prevent scoping problems with the labels (leads to compilation errors despite breaks)
		switch (this->type)
		{
		case StandardOutputType::VectorXYZ:
		{
			arma::vec v = this->prefactor * this->vector->Get();
			if (v.n_elem == 3)
				output << v[0] << this->delimiter << v[1] << this->delimiter << v[2];
			else
				output << '-' << this->delimiter << '-' << this->delimiter << '-';
		}
		break;
		case StandardOutputType::VectorLength:
		{
			arma::vec v = this->vector->Get();
			if (v.n_elem == 3)
				output << (this->prefactor * arma::norm(v));
			else
				output << '-';
		}
		break;
		case StandardOutputType::VectorAngle:
		{
			arma::vec v = this->vector->Get();
			if (v.n_elem == 3)
				output << (this->prefactor * acos(dot(normalise(v), this->reference)) * 180.0 / arma::datum::pi);
			else
				output << '-';
		}
		break;
		case StandardOutputType::VectorDot:
		{
			arma::vec v = this->vector->Get();
			if (v.n_elem == 3)
				output << (this->prefactor * dot(v, this->reference));
			else
				output << '-';
		}
		break;
		case StandardOutputType::Scalar:
			output << (this->prefactor * this->scalar->Get());
			break;
		default:
			output << "";
			break;
		}

		return output.str();
	}
	// -----------------------------------------------------
	// Precision control
	// -----------------------------------------------------
	void StandardOutput::SetDefaultPrecision(unsigned int _precision)
	{
		const unsigned int maxPrec = std::numeric_limits<double>::max_digits10;
		if (_precision >= 1 && _precision <= maxPrec)
			DefaultOutputPrecision() = _precision;
	}

	unsigned int StandardOutput::DefaultPrecision()
	{
		return DefaultOutputPrecision();
	}
	// -----------------------------------------------------
	// StandardOutput public methods
	// -----------------------------------------------------
	// Returns the name of the output object
	std::string StandardOutput::Name() const
	{
		if (this->properties == nullptr)
			return "undefined";

		return this->properties->Name();
	}

	// Checks whether the action is valid
	bool StandardOutput::IsValid() const
	{
		return this->isValid;
	}

	// Prepares the output object and checks whether it is valid
	bool StandardOutput::Validate(const std::map<std::string, RunSection::ActionScalar> &_scalars, const std::map<std::string, RunSection::ActionVector> &_vectors)
	{
		// Check whether a type was specified
		std::string str;
		if (!this->properties->Get("type", str))
		{
			std::cout << "ERROR: No Type specified for the output object \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Then determine which type of output was specified
		if (str.compare("vector") == 0 || str.compare("vectorxyz") == 0 || str.compare("xyz") == 0 || str.compare("components") == 0)
		{
			this->type = StandardOutputType::VectorXYZ;
		}
		else if (str.compare("vectorangle") == 0 || str.compare("angle") == 0)
		{
			this->type = StandardOutputType::VectorAngle;
		}
		else if (str.compare("vectorlength") == 0 || str.compare("length") == 0)
		{
			this->type = StandardOutputType::VectorLength;
		}
		else if (str.compare("vectordot") == 0 || str.compare("dot") == 0 || str.compare("projection") == 0)
		{
			this->type = StandardOutputType::VectorDot;
		}
		else if (str.compare("scalar") == 0)
		{
			this->type = StandardOutputType::Scalar;
		}
		else
		{
			std::cout << "ERROR: Invalid Type \"" << str << "\" specified for the output object \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Get the prefactor
		this->properties->Get("prefactor", this->prefactor);

		// Get the parameters needed for the output object
		arma::vec tmpvec;
		switch (this->type)
		{
		// Both Angle and Dot need a reference
		case StandardOutputType::VectorAngle:
		case StandardOutputType::VectorDot:
			if (!this->properties->Get("reference", tmpvec))
			{
				std::cout << "ERROR: Failed to set reference vector for output object \"" << this->Name() << "\"!" << std::endl;
				return false;
			}
			this->reference = arma::normalise(tmpvec); // Make sure the reference is normalized

		// All vector outputs need an ActionVector
		case StandardOutputType::VectorLength:
		case StandardOutputType::VectorXYZ:
			if (!this->setActionVector(_vectors))
			{
				std::cout << "ERROR: Failed to set ActionVector for output object \"" << this->Name() << "\"!" << std::endl;
				return false;
			}
			break;

		// Only a single output type is needed for a scalar
		case StandardOutputType::Scalar:
			if (!this->setActionScalar(_scalars))
			{
				std::cout << "ERROR: Failed to set ActionScalar for output object \"" << this->Name() << "\"!" << std::endl;
				return false;
			}
			break;

		// When the type is not recognized
		default:
			return false;
		}

		// The output object was setup correctly
		this->isValid = true;
		return this->isValid;
	}

	// Method to set the delimiter character
	// Returns false if the character is invalid as a delimiter
	bool StandardOutput::SetDelimiter(char _c)
	{
		// Define allowed delimiter characters
		const std::string allowed(".,;|></ \t");

		// Check if the given character is allowed
		if (allowed.find(_c) != std::string::npos)
		{
			this->delimiter = _c;
			return true;
		}

		return false;
	}
	// -----------------------------------------------------
	// StandardOutput public output methods
	// -----------------------------------------------------
	// Writes the text for the header column of the data
	bool StandardOutput::WriteOutputHeader(std::ostream &_stream) const
	{
		// Check whether the object is valid
		if (!this->isValid)
			return false;

		// Write the header
		_stream << this->Header();
		return true;
	}

	// Writes the data
	bool StandardOutput::WriteOutput(std::ostream &_stream) const
	{
		// Check whether the object is valid
		if (!this->isValid)
			return false;

		// Write the data
		_stream << this->Data();
		return true;
	}
	// -----------------------------------------------------
}
