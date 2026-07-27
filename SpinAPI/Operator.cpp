/////////////////////////////////////////////////////////////////////////
// Operator class (SpinAPI Module)
// ------------------
// Special operators to be used in some task types.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "Operator.h"
#include "ObjectParser.h"
#include "SpinSystem.h"
#include <algorithm>
#include <cctype>

namespace SpinAPI
{
	namespace
	{
		RelaxationFrame DefaultRelaxationFrame(OperatorType _type)
		{
			// Dedicated Bloch-style T1/T2 channels are defined relative to the
			// laboratory static-field axis. Raw Cartesian channels retain their
			// historical molecule-fixed default; users can override either
			// convention explicitly with frame=lab or frame=molecular.
			if (_type == OperatorType::RelaxationT1 || _type == OperatorType::RelaxationT2)
				return RelaxationFrame::Lab;
			return RelaxationFrame::Molecular;
		}
	}

	// -----------------------------------------------------
	// Spin Constructors and Destructor
	// -----------------------------------------------------
	Operator::Operator(std::string _name, std::string _contents) : properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents)), type(OperatorType::Unspecified), spins(), rate1(0.0), rate2(0.0), rate3(0.0), relaxationFrame(RelaxationFrame::Molecular), isValid(false)
	{
	}

	Operator::Operator(const Operator &_operator) : properties(std::make_shared<MSDParser::ObjectParser>(*(_operator.properties))), type(_operator.type), spins(_operator.spins), rate1(_operator.rate1), rate2(_operator.rate2), rate3(_operator.rate3), relaxationFrame(_operator.relaxationFrame), isValid(_operator.isValid)
	{
	}

	Operator::~Operator()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	const Operator &Operator::operator=(const Operator &_operator)
	{
		this->properties = std::make_shared<MSDParser::ObjectParser>(*(_operator.properties));
		this->type = _operator.type;
		this->spins = _operator.spins;
		this->rate1 = _operator.rate1;
		this->rate2 = _operator.rate2;
		this->rate3 = _operator.rate3;
		this->relaxationFrame = _operator.relaxationFrame;
		this->isValid = _operator.isValid;

		return (*this);
	}
	// -----------------------------------------------------
	// Public validation methods
	// -----------------------------------------------------
	// Validate the Operator object
	bool Operator::Validate(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &_systems)
	{
		// Validation may be repeated after a system is edited. Rebuild derived
		// fields from the parser rather than accumulating stale spin pointers.
		this->type = OperatorType::Unspecified;
		this->spins.clear();
		this->rate1 = 0.0;
		this->rate2 = 0.0;
		this->rate3 = 0.0;

		// Get the type of the operator
		std::string str;
		if (this->properties->Get("type", str) || this->properties->Get("operatortype", str))
		{
			if (str.compare("relaxationlindblad") == 0 || str.compare("relaxationlindbladsinglespin") == 0 || str.compare("relaxationlindbladsinglespins") == 0)
			{
				this->type = OperatorType::RelaxationLindblad;
			}
			else if (str.compare("relaxationlindbladdoublespin") == 0)
			{
				this->type = OperatorType::RelaxationLindbladDoubleSpin;
			}
			else if (str.compare("relaxationdephasing") == 0)
			{
				this->type = OperatorType::RelaxationDephasing;
			}
			else if (str.compare("relaxationrandomfields") == 0)
			{
				this->type = OperatorType::RelaxationRandomFields;
			}
			else if (str.compare("relaxationt1") == 0)
			{
				this->type = OperatorType::RelaxationT1;
			}
			else if (str.compare("relaxationt2") == 0)
			{
				this->type = OperatorType::RelaxationT2;
			}
			else if (str.compare("relaxationphenomenological") == 0 || str.compare("phenomenologicalrelaxation") == 0 || str.compare("phenomenological") == 0)
			{
				// rate1: population transfer between eigenstates
				// rate2: damping of off-diagonal density-matrix elements
				this->type = OperatorType::RelaxationPhenomenological;
			}
			else if (str.compare("unspecified") == 0)
			{
				this->type = OperatorType::Unspecified;
			}
			else
			{
				std::cout << "Failed to validate operator object \"" << this->Name() << "\": Invalid operator type!" << std::endl;
				this->isValid = false;
				return this->isValid;
			}
		}

		this->relaxationFrame = DefaultRelaxationFrame(this->type);
		if (this->properties->Get("frame", str) || this->properties->Get("relaxationframe", str))
		{
			std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			if (str == "lab" || str == "laboratory" || str == "fixed")
			{
				this->relaxationFrame = RelaxationFrame::Lab;
			}
			else if (str == "molecular" || str == "molecule" || str == "mol" || str == "rotating")
			{
				this->relaxationFrame = RelaxationFrame::Molecular;
			}
			else
			{
				std::cout << "Warning: Unknown relaxation frame \"" << str << "\" for Operator object " << this->Name() << ". Using the default frame for this operator type." << std::endl;
			}
		}

		// Get one or more rates
		double rate;
		if (this->properties->Get("rate", rate))
		{
			if (rate >= 0.0 && std::isfinite(rate))
			{
				this->rate1 = rate;
				this->rate2 = rate;
				this->rate3 = rate;
			}
			else
			{
				std::cout << "Warning: Ignored invalid rate \"" << rate << "\" specified for Operator object " << this->Name() << "!" << std::endl;
			}
		}
		if (this->properties->Get("rate1", rate) || this->properties->Get("ratex", rate))
		{
			if (rate >= 0.0 && std::isfinite(rate))
				this->rate1 = rate;
			else
				std::cout << "Warning: Ignored invalid rate \"" << rate << "\" specified for Operator object " << this->Name() << "!" << std::endl;
		}
		if (this->properties->Get("rate2", rate) || this->properties->Get("ratey", rate))
		{
			if (rate >= 0.0 && std::isfinite(rate))
				this->rate2 = rate;
			else
				std::cout << "Warning: Ignored invalid rate \"" << rate << "\" specified for Operator object " << this->Name() << "!" << std::endl;
		}
		if (this->properties->Get("rate3", rate) || this->properties->Get("ratez", rate))
		{
			if (rate >= 0.0 && std::isfinite(rate))
				this->rate3 = rate;
			else
				std::cout << "Warning: Ignored invalid rate \"" << rate << "\" specified for Operator object " << this->Name() << "!" << std::endl;
		}

		// Get a list of spins that should be affected by the operator
		std::vector<std::string> spinlist;
		if (this->properties->GetList("spins", spinlist) || this->properties->GetList("spin", spinlist) || this->properties->GetList("spinlist", spinlist))
		{
			for (const std::string &s : spinlist)
			{
				// Search through the spin systems for the spin
				for (auto i = _systems.cbegin(); i != _systems.cend(); i++)
				{
					auto tmp = (*i)->spins_find(s);
					if (tmp != nullptr)
					{
						this->spins.push_back(tmp);
						break;
					}
				}
			}
		}

		this->isValid = true;
		return this->isValid;
	}
	// -----------------------------------------------------
	// Other public methods
	// -----------------------------------------------------
	// Returns the name of the Operator object
	std::string Operator::Name() const
	{
		return this->properties->Name();
	}

	// Returns the Operator type
	OperatorType Operator::Type() const
	{
		return this->type;
	}

	// Returns a copy of the collection of spins
	std::vector<spin_ptr> Operator::Spins() const
	{
		return this->spins;
	}

	// Returns the number of spins in the collection
	unsigned int Operator::SpinCount() const
	{
		return this->spins.size();
	}

	// Returns the first rate
	double Operator::Rate1() const
	{
		return this->rate1;
	}

	// Returns the second rate
	double Operator::Rate2() const
	{
		return this->rate2;
	}

	// Returns the third rate
	double Operator::Rate3() const
	{
		return this->rate3;
	}

	// Returns whether Cartesian relaxation axes are lab- or molecule-fixed.
	RelaxationFrame Operator::Frame() const
	{
		return this->relaxationFrame;
	}

	// Checks whether the Operator object was validated successfully
	bool Operator::IsValid() const
	{
		return this->isValid;
	}
	// -----------------------------------------------------
	// Access to custom properties
	// -----------------------------------------------------
	std::shared_ptr<const MSDParser::ObjectParser> Operator::Properties() const
	{
		return this->properties;
	}
	// -----------------------------------------------------
	// Non-member non-friend methods
	// -----------------------------------------------------
	bool IsValid(const Operator &_operator)
	{
		return _operator.IsValid();
	}
	// -----------------------------------------------------
}
