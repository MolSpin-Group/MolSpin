/////////////////////////////////////////////////////////////////////////
// Interaction class (SpinAPI Module)
// ------------------
// Base class for interactions.
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards and Irina Anisimova.
// (c) 2024 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <memory>
#include <iostream>
#include <Spin.h>
#include "ObjectParser.h"
#include "Pulse.h"
#include "Utility.h"

namespace SpinAPI
{
	// -----------------------------------------------------
	// Interaction Constructors and Destructor
	// -----------------------------------------------------
	// The constructor sets up the pulse parameters, but
	// the spin groups are read in the method ParseSpinGroups instead.
	Pulse::Pulse(std::string _name, std::string _contents) : properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents)), type(PulseType::Unspecified), group(), timestep(0.1), rotationaxis({0, 0, 0}), angle(0.0), pulsetime(0.0), field({0, 0, 0}), frequency(0.0), addCommonPrefactorList(), ignoreTensorsList(), prefactorList()
	{
		// Filling required parameter
		std::string str = "";

		// Get the type of the pulse
		this->properties->Get("type", str);
		if (str.compare("instantpulse") == 0 || str.compare("InstantPulse") == 0)
			this->type = PulseType::InstantPulse;
		else if (str.compare("longpulse") == 0 || str.compare("LongPulse") == 0)
			this->type = PulseType::LongPulse;
		else if (str.compare("longpulsestaticfield") == 0 || str.compare("LongPulseStaticField") == 0)
			this->type = PulseType::LongPulseStaticField;
		else if (str.compare("mwpulse") == 0 || str.compare("MWPulse") == 0)
			this->type = PulseType::MWPulse;

		if (this->type == PulseType::InstantPulse)
		{
			// Get rotation axis
			if (!this->properties->Get("rotationaxis", rotationaxis))
			{
				std::cout << "Warning: Failed to obtain input for rotationaxis." << std::endl;
				std::cout << "Using vector: " << rotationaxis << std::endl;
			}

			// Get angle
			if (!this->properties->Get("angle", angle))
			{
				std::cout << "Warning: Failed to obtain input for angle." << std::endl;
				std::cout << "Using vector: " << angle << std::endl;
			}

			// Get Pulse timestep
			if (!this->properties->Get("timestep", timestep))
			{
				std::cout << "Warning: Failed to obtain input for pulse timestep. The timestep will be set to 0.1 ns." << std::endl;
			}
		}
		else if (this->type == PulseType::LongPulse)
		{
			// Get pulse time
			if (!this->properties->Get("pulsetime", pulsetime))
			{
				std::cout << "Warning: Failed to obtain input for pulsetime." << std::endl;
			}

			// Get Pulse timestep
			if (!this->properties->Get("timestep", timestep))
			{
				std::cout << "Warning: Failed to obtain input for pulse timestep. The timestep will be set to 0.1 ns." << std::endl;
			}

			// Get Pulse field
			if (!this->properties->Get("field", field))
			{
				std::cout << "Warning: Failed to obtain input for the field." << std::endl;
				std::cout << "Using vector: " << field << std::endl;
			}

			// Get pulse frequency
			if (!this->properties->Get("frequency", frequency))
			{
				std::cout << "Warning: Failed to obtain input for the pulse frequency." << std::endl;
				std::cout << "Using frequency: " << frequency << std::endl;
			}

			// Get prefactor list
			if (!this->properties->GetList("prefactorlist", prefactorList))
			{
				std::cout << "Warning: Failed to obtain input for the prefactorList." << std::endl;
			}

			// Get prefactor properties
			if (!this->properties->GetList("commonprefactorlist", addCommonPrefactorList))
			{
				std::cout << "Warning: Failed to obtain input for the commonPrefactorList." << std::endl;
			}

			if (!this->properties->GetList("ignoretensorslist", ignoreTensorsList))
			{
				std::cout << "Warning: Failed to obtain input for the ignoreTensorList." << std::endl;
			}
		}
		else if (this->type == PulseType::LongPulseStaticField)
		{
			// Get pulse time
			if (!this->properties->Get("pulsetime", pulsetime))
			{
				std::cout << "Warning: Failed to obtain input for pulsetime." << std::endl;
			}

			// Get Pulse timestep
			if (!this->properties->Get("timestep", timestep))
			{
				std::cout << "Warning: Failed to obtain input for pulse timestep. The timestep will be set to 0.1 ns." << std::endl;
			}

			// Get Pulse field
			if (!this->properties->Get("field", field))
			{
				std::cout << "Warning: Failed to obtain input for the field." << std::endl;
				std::cout << "Using vector: " << field << std::endl;
			}

			// Get prefactor list
			if (!this->properties->GetList("prefactorlist", prefactorList))
			{
				std::cout << "Warning: Failed to obtain input for the prefactorList." << std::endl;
			}

			// Get prefactor properties
			if (!this->properties->GetList("commonprefactorlist", addCommonPrefactorList))
			{
				std::cout << "Warning: Failed to obtain input for the CommonPrefactorList." << std::endl;
			}

			if (!this->properties->GetList("ignoretensorslist", ignoreTensorsList))
			{
				std::cout << "Warning: Failed to obtain input for the ignoreTensorList." << std::endl;
			}
		}
		else if (this->type == PulseType::MWPulse)
		{
			// Get pulse time
			if (!this->properties->Get("pulsetime", pulsetime))
			{
				std::cout << "Warning: Failed to obtain input for pulsetime." << std::endl;
			}

			// Get Pulse timestep
			if (!this->properties->Get("timestep", timestep))
			{
				std::cout << "Warning: Failed to obtain input for pulse timestep. The timestep will be set to 0.1 ns." << std::endl;
			}

			// Get Pulse field
			if (!this->properties->Get("field", field))
			{
				std::cout << "Warning: Failed to obtain input for the field." << std::endl;
				std::cout << "Using vector: " << field << std::endl;
			}

			// Get pulse frequency
			if (!this->properties->Get("frequency", frequency))
			{
				std::cout << "Warning: Failed to obtain input for the pulse frequency." << std::endl;
				std::cout << "Using frequency: " << frequency << std::endl;
			}

			// Get prefactor list
			if (!this->properties->GetList("prefactorlist", prefactorList))
			{
				std::cout << "Warning: Failed to obtain input for the prefactorList." << std::endl;
			}

			// Get prefactor properties
			if (!this->properties->GetList("commonprefactorlist", addCommonPrefactorList))
			{
				std::cout << "Warning: Failed to obtain input for the commonPrefactorList." << std::endl;
			}

			if (!this->properties->GetList("ignoretensorslist", ignoreTensorsList))
			{
				std::cout << "Warning: Failed to obtain input for the ignoreTensorList." << std::endl;
			}
		}
		else if (this->type == PulseType::ShapedPulse)
		{
			std::cout << "ShapedPulse not implemented yet, sorry. " << std::endl;
		}
		else
		{
			std::cout << "Type of pulse is not recognized" << std::endl;
		}
	}

	Pulse::Pulse(const Pulse &_pulse) : properties(_pulse.properties), type(_pulse.type), group(_pulse.group), timestep(_pulse.timestep), rotationaxis(_pulse.rotationaxis), angle(_pulse.angle), pulsetime(_pulse.pulsetime), field(_pulse.field), frequency(_pulse.frequency), addCommonPrefactorList(_pulse.addCommonPrefactorList), ignoreTensorsList(_pulse.ignoreTensorsList), prefactorList(_pulse.prefactorList)
	{
	}

	Pulse::~Pulse()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	const Pulse &Pulse::operator=(const Pulse &_pulse)
	{
		this->properties = std::make_shared<MSDParser::ObjectParser>(*(_pulse.properties));
		this->type = _pulse.type;
		this->group = _pulse.group;
		this->timestep = _pulse.timestep;
		this->rotationaxis = _pulse.rotationaxis;
		this->angle = _pulse.angle;
		this->pulsetime = _pulse.pulsetime;
		this->field = _pulse.field;
		this->frequency = _pulse.frequency;
		this->addCommonPrefactorList = _pulse.addCommonPrefactorList;
		this->ignoreTensorsList = _pulse.ignoreTensorsList;
		this->prefactorList = _pulse.prefactorList;

		return (*this);
	}
	// -----------------------------------------------------
	// Public methods
	// -----------------------------------------------------
	std::string Pulse::Name() const
	{
		return this->properties->Name();
	}

	bool Pulse::IsValid() const
	{
		if (this->type == PulseType::Unspecified && !this->group.empty())
			return true;
		else if (this->type == PulseType::InstantPulse && !this->group.empty())
			return true;
		else if (this->type == PulseType::LongPulse && !this->group.empty())
			return true;
		else if (this->type == PulseType::LongPulseStaticField && !this->group.empty())
			return true;
		else if (this->type == PulseType::MWPulse && !this->group.empty())
			return true;
		else if (this->type == PulseType::ShapedPulse && !this->group.empty())
			return true;

		return false;
	}
	// -----------------------------------------------------
	// Property methods
	// -----------------------------------------------------
	// Returns the rotation axis
	const arma::vec Pulse::Rotationaxis() const
	{
		return (this->rotationaxis / arma::norm(this->rotationaxis)); // The normal vector in direction of rotation
	}

	// Returns the rotation angle
	const double Pulse::Angle() const
	{
		return this->angle;
	}

	// Returns the time of the pulse
	const double Pulse::Pulsetime() const
	{
		return this->pulsetime;
	}

	// Returns the field of the pulse
	const arma::vec Pulse::Field() const
	{
		return this->field;
	}

	// Returns the frequency of the pulse
	const double Pulse::Frequency() const
	{
		return this->frequency;
	}

	// Returns pulse type
	PulseType Type(const Pulse &_pulse)
	{
		return _pulse.Type();
	}

	const arma::vec Pulse::PrefactorList() const
	{
		return this->prefactorList;
	}

	// Returns the timestep of the pulse
	const double Pulse::Timestep() const
	{
		return this->timestep;
	}
	// -----------------------------------------------------
	// Access to custom properties
	// -----------------------------------------------------
	std::shared_ptr<const MSDParser::ObjectParser> Pulse::Properties() const
	{
		return this->properties;
	}

	// -----------------------------------------------------
	// Public method to read spins into group,
	// returns false if some spins were not found
	// -----------------------------------------------------
	bool Pulse::ParseSpinGroups(const std::vector<spin_ptr> &_spinlist)
	{
		bool createdSpinLists = false;

		// Attempt to get a list of spins from the input file
		std::string str;
		if (!this->properties->Get("spins", str) && !this->properties->Get("group", str))
			return false;

		createdSpinLists = this->AddSpinList(str, _spinlist, this->group);

		// Check whether we were successful
		if (!createdSpinLists)
		{
			this->group.clear();
			return false;
		}

		return true;
	}

	// Splits the string into spin names, find the corresponding spins in _spinlist and adds them to _group
	// Private helper method for the ParseSpinGroups method
	bool Pulse::AddSpinList(const std::string &_names, const std::vector<spin_ptr> &_spinlist, std::vector<spin_ptr> &_group, const std::vector<spin_ptr> *_crossCheck)
	{
		// Iterate through the spin name (while-loop) with a ',' delimiter
		std::stringstream ss(_names);
		bool spin_was_found;
		std::string str;
		while (std::getline(ss, str, ','))
		{
			spin_was_found = false;
			str = RunSection::trim(str);

			// Spin names are case-sensitive identifiers throughout SpinAPI.
			// Trim list whitespace, but preserve the name exactly as supplied.
			for (auto i = _spinlist.cbegin(); i != _spinlist.cend(); i++)
			{
				if ((*i)->Name().compare(str) == 0)
				{
					// Perform crosscheck if requested
					if (_crossCheck != nullptr && std::find((*_crossCheck).cbegin(), (*_crossCheck).cend(), (*i)) != (*_crossCheck).cend())
						return false;

					// Make sure we don't have duplicates in the spin group
					if (std::find(_group.cbegin(), _group.cend(), (*i)) != _group.cend())
						return false;

					// Add it to the group
					_group.push_back(*i);
					spin_was_found = true;
					break;
				}
			}

			if (!spin_was_found)
				return false;
		}

		return true;
	}
	// -----------------------------------------------------
	// Public spin list methods
	// -----------------------------------------------------

	// -----------------------------------------------------
	// Methods to create ActionTarget objects
	// -----------------------------------------------------
	// Create ActionVectors
	std::vector<RunSection::NamedActionVector> Pulse::CreateActionVectors(const std::string &_system)
	{
		std::vector<RunSection::NamedActionVector> vectors;

		if (this->IsValid())
		{

			if (this->type == PulseType::InstantPulse)
			{
				// The rotationaxis vector
				RunSection::ActionVector rotationaxisVector = RunSection::ActionVector(this->rotationaxis, &CheckActionVectorPulseAxis);
				vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".rotationaxis", rotationaxisVector));
			}

			if (this->type == PulseType::LongPulse || this->type == PulseType::LongPulseStaticField || this->type == PulseType::MWPulse)
			{
				// The field vector
				RunSection::ActionVector fieldVector = RunSection::ActionVector(this->field, &CheckActionVectorPulseField);
				vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".field", fieldVector));
			}
		}

		return vectors;
	}

	// Create ActionScalars
	std::vector<RunSection::NamedActionScalar> Pulse::CreateActionScalars(const std::string &_system)
	{
		std::vector<RunSection::NamedActionScalar> scalars;

		if (this->IsValid())
		{

			if (this->type == PulseType::InstantPulse)
			{
				// We should always have a scalar for the prefactor
				RunSection::ActionScalar angleScalar = RunSection::ActionScalar(this->angle, &CheckActionScalarPulseScalar);
				scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".angle", angleScalar));
			}

			if (this->type == PulseType::LongPulse)
			{
				// We should always have a scalar for the prefactor
				RunSection::ActionScalar pulsetimeScalar = RunSection::ActionScalar(this->pulsetime, &CheckActionScalarPulseDuration);
				scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".pulsetime", pulsetimeScalar));

				RunSection::ActionScalar frequencyScalar = RunSection::ActionScalar(this->frequency, &CheckActionScalarPulseScalar);
				scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".frequency", frequencyScalar));
			}

			if (this->type == PulseType::MWPulse)
			{
				// We should always have a scalar for the prefactor
				RunSection::ActionScalar pulsetimeScalar = RunSection::ActionScalar(this->pulsetime, &CheckActionScalarPulseDuration);
				scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".pulsetime", pulsetimeScalar));

				RunSection::ActionScalar frequencyScalar = RunSection::ActionScalar(this->frequency, &CheckActionScalarPulseScalar);
				scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".frequency", frequencyScalar));
			}

			if (this->type == PulseType::LongPulseStaticField)
			{
				// We should always have a scalar for the prefactor
				RunSection::ActionScalar pulsetimeScalar = RunSection::ActionScalar(this->pulsetime, &CheckActionScalarPulseDuration);
				scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".pulsetime", pulsetimeScalar));
			}
		}

		return scalars;
	}

	// Method that calls the methods to generate ActionVectors and ActionScalars and inserts them into the given collections
	void Pulse::GetActionTargets(std::vector<RunSection::NamedActionScalar> &_scalars, std::vector<RunSection::NamedActionVector> &_vectors, const std::string &_system)
	{
		// Get ActionTargets from private methods
		auto scalars = this->CreateActionScalars(_system);
		auto vectors = this->CreateActionVectors(_system);

		// Insert them
		_scalars.insert(_scalars.end(), scalars.begin(), scalars.end());
		_vectors.insert(_vectors.end(), vectors.begin(), vectors.end());
	}
	// -----------------------------------------------------

	// -----------------------------------------------------
	// Non-member non-friend ActionTarget Check functions
	// -----------------------------------------------------
	// Checks that a vector set by an Action is valid
	bool CheckActionVectorPulseField(const arma::vec &_v)
	{
		// A field vector must have 3 components
		if (_v.n_elem != 3)
			return false;

		// Make sure we don't have invalid values
		if (_v.has_nan() || _v.has_inf())
			return false;

		return true;
	}

	// Rotation operators normalize this vector, so a zero axis is undefined.
	bool CheckActionVectorPulseAxis(const arma::vec &_v)
	{
		return CheckActionVectorPulseField(_v) && arma::norm(_v) > 0.0;
	}

	// Make sure that a pulse scalar has a finite value.
	bool CheckActionScalarPulseScalar(const double &_d)
	{
		return std::isfinite(_d);
	}

	// Pulse durations may be zero, but propagation with negative time is invalid.
	bool CheckActionScalarPulseDuration(const double &_d)
	{
		return std::isfinite(_d) && _d >= 0.0;
	}
}
