/////////////////////////////////////////////////////////////////////////
// Transition class (SpinAPI Module)
// ------------------
// Describes reactions operators.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ObjectParser.h"
#include "State.h"
#include "Transition.h"

#include <algorithm>
#include <cctype>

namespace SpinAPI
{
	namespace
	{
		std::string LowerProfileName(std::string value)
		{
			std::transform(value.begin(), value.end(), value.begin(),
				[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return value;
		}

		// A Transition can expose ".rate" as an ActionTarget only when changing
		// Transition::Rate() changes the actual physical rate used downstream.
		// General/MultiSS profiles with an explicit peak/fraction/profile table
		// derive k(t) independently of Transition::Rate(), so leaving ".rate"
		// writable would create a silent no-op Action.
		bool ProfileOverridesTransitionRate(
			const std::shared_ptr<MSDParser::ObjectParser> &properties)
		{
			if (properties == nullptr)
				return false;

			std::string kind;
			if (!(properties->Get("rateprofile", kind) ||
				properties->Get("rate_profile", kind) ||
				properties->Get("opticalprofile", kind)))
				return false;

			kind = LowerProfileName(kind);
			double ignored = 0.0;
			if (kind == "gaussian")
			{
				return properties->Get("peakrate", ignored) ||
					properties->Get("peak_rate", ignored) ||
					properties->Get("transferfraction", ignored) ||
					properties->Get("transfer_fraction", ignored);
			}
			if (kind == "rectangular" || kind == "square")
			{
				return properties->Get("peakrate", ignored) ||
					properties->Get("peak_rate", ignored);
			}
			if (kind == "trajectory" ||
				kind == "instantaneous" ||
				kind == "event" ||
				kind == "delta")
				return true;

			// Constant/static profiles and Gaussian/rectangular profiles without
			// an explicit override use Transition::Rate() directly.
			return false;
		}
	}

	// -----------------------------------------------------
	// Spin Constructors and Destructor
	// -----------------------------------------------------
	Transition::Transition(std::string _name, std::string _contents, std::shared_ptr<SpinSystem> _system) : system(_system), target(nullptr), properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents)),
																											rate(1.0), type(TransitionType::Sink), sourcestate(nullptr), targetstate(nullptr),
																											reactionOperators(ReactionOperatorType::Unspecified), isValid(false), trajectory(),
																											trjHasTime(false), trjHasRate(false), trjRateIsLifetime(false), trjTime(0), trjRate(0),
																											Pulsed(false),Active(true),active_time(0.0)
																											

	{
		// Check whether a trajectory was specified
		std::string str;
		
		if (this->properties->Get("duration", this->active_time))
		{
			this->Pulsed = true;
			this->Active = false;
		}

		if (this->properties->Get("trajectory", str))
		{
			// Get the directory
			std::string directory;
			if (!this->properties->Get("FileReaderDirectory", directory))
			{
				std::cout << "Warning: Failed to obtain input file directory while initializing Transition " << this->Name() << "!\n";
				std::cout << "         This may lead to problems when trying to localize the trajectory file." << std::endl;
			}


			// Load the trajectory, and check whether something was loaded
			if (this->trajectory.Load(directory + str))
			{
				if (this->trajectory.Length() > 0)
				{
					// Get column numbers
					this->trjHasTime = this->trajectory.HasColumn("time", trjTime);
					this->trjHasRate = this->trajectory.HasColumn("rate", trjRate);

					// If rate was not specified, check for lifetime
					if (!this->trjHasRate)
					{
						this->trjHasRate = this->trajectory.HasColumn("lifetime", trjRate);
						this->trjRateIsLifetime = this->trjHasRate;
					}
				}

				// If no useful data was loaded, clear the trajectory
				if (!this->trjHasRate)
				{
					this->trajectory.Clear();
					std::cout << "NOTE: No useful data found in trajectory \"" << (directory + str) << "\" for Transition object \"" << this->Name() << "\". No data was loaded." << std::endl;
				}
			}
			else
			{
				std::cout << "ERROR: Failed to load trajectory \"" << (directory + str) << "\" for Transition object \"" << this->Name() << "\"!" << std::endl;
			}
		}

		// Attempt to obtain the rate (or its inverse, the lifetime)
		if (!this->properties->Get("rate", this->rate))
		{
			if (this->properties->Get("lifetime", this->rate))
				this->rate = 1.0 / this->rate; // TODO: Consider the units used here - is this correct?
			else if (!this->HasTrajectory() && !this->trjHasRate)
				std::cout << "Warning: No rate or lifetime specified for transition object \"" << this->Name() << "\"!" << std::endl;
		}

		// The rate/lifetime can only be defined as a positive quantity
		#if NEGATIVERATES == 1
			this->rate = this->rate;
		#else
			this->rate = std::abs(this->rate);
		#endif

		// Parse transition type
		if (this->properties->Get("type", str))
		{
			if (str.compare("sink") == 0 || str.compare("decay") == 0)
			{
				this->type = TransitionType::Sink;
			}
			else if (str.compare("source") == 0 || str.compare("creation") == 0)
			{
				this->type = TransitionType::Source;
			}
			else
			{
				std::cout << "Warning: Unknown transition type \"" << str << "\"! Transition \"" << this->Name() << "\" will be treated as a decay/sink." << std::endl;
				this->type = TransitionType::Sink;
			}
		}

		// Parse reaction operator type
		if (this->properties->Get("reactionoperators", str) || this->properties->Get("reactionoperatortype", str))
		{
			if (str.compare("haberkorn") == 0)
			{
				this->reactionOperators = ReactionOperatorType::Haberkorn;
			}
			else if (str.compare("lindblad") == 0)
			{
				this->reactionOperators = ReactionOperatorType::Lindblad;
			}
			else
			{
				std::cout << "Warning: Unknown reaction operator type \"" << str << "\"! Transition \"" << this->Name() << "\" will use default reaction operator type." << std::endl;
			}
		}
	}

	Transition::Transition(const Transition &_transition) : system(_transition.system), target(_transition.target), properties(std::make_shared<MSDParser::ObjectParser>(*(this->properties))), rate(_transition.rate),
															type(_transition.type), sourcestate(_transition.sourcestate), targetstate(_transition.targetstate), isValid(_transition.isValid),
															trajectory(_transition.trajectory), trjHasTime(_transition.trjHasTime), trjHasRate(_transition.trjHasRate), trjRateIsLifetime(_transition.trjRateIsLifetime), trjTime(_transition.trjTime), trjRate(_transition.trjRate),
															Pulsed(_transition.Pulsed), Active(_transition.Active), active_time(_transition.active_time)
	{
	}

	Transition::~Transition()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	const Transition &Transition::operator=(const Transition &_transition)
	{
		this->system = _transition.system;
		this->target = _transition.target;
		this->properties = std::make_shared<MSDParser::ObjectParser>(*(_transition.properties));
		this->rate = _transition.rate;
		this->type = _transition.type;
		this->sourcestate = _transition.sourcestate;
		this->targetstate = _transition.targetstate;
		this->isValid = _transition.isValid;

		this->trajectory = _transition.trajectory;
		this->trjHasTime = _transition.trjHasTime;
		this->trjHasRate = _transition.trjHasRate;
		this->trjRateIsLifetime = _transition.trjRateIsLifetime;
		this->trjTime = _transition.trjTime;
		this->trjRate = _transition.trjRate;

		this->Active = _transition.Active;
		this->active_time = _transition.active_time;
		this->Pulsed = _transition.Pulsed;

		return (*this);
	}
	// -----------------------------------------------------
	// Public validation methods
	// -----------------------------------------------------
	// Validate the Transition object and get information about State objects
	bool Transition::Validate(const std::vector<std::shared_ptr<SpinAPI::SpinSystem>> &_systems)
	{
		// Make sure that we belong to a SpinSystem
		if (this->system == nullptr)
			return false;

		// Check that a source state has been specified
		std::string str;
		if (!this->properties->Get("sourcestate", str) && !this->properties->Get("state", str) && !this->properties->Get("source", str))
		{
			std::cout << "Error: No source state specified for Transition \"" << this->Name() << "\" in SpinSystem \"" << this->system->Name() << "\"!" << std::endl;
			return false;
		}

		// See if the system has a state with the requested name
		this->sourcestate = this->system->states_find(str);
		if (this->sourcestate == nullptr)
		{
			std::cout << "Error: The SpinSystem \"" << this->system->Name() << "\" has no state \"" << str << "\"! Failed to validate Transition \"" << this->Name() << "\"!" << std::endl;
			return false;
		}

		// Check whether a target system was specified
		if (this->properties->Get("target", str) || this->properties->Get("targetsystem", str))
		{
			// Only sinks/decays can have a target system
			if (this->type != TransitionType::Sink)
			{
				std::cout << "Error: Transition \"" << this->Name() << "\" must be of type sink/decay in order to have a target system!" << std::endl;
				return false;
			}

			// Search for the target system
			for (auto i = _systems.cbegin(); i != _systems.cend(); i++)
			{
				if ((*i)->Name().compare(str) == 0)
				{
					this->target = (*i);
					break;
				}
			}

			// Make sure that a target system was found
			if (this->target == nullptr)
			{
				std::cout << "Error: Could not find target SpinSystem \"" << str << "\" specified for Transition \"" << this->Name() << "\"! Source SpinSystem is \"" << this->system->Name() << "\"!" << std::endl;
				return false;
			}

			// Check that a target state has been specified
			if (!this->properties->Get("targetstate", str))
			{
				std::cout << "Error: No target state specified for Transition \"" << this->Name() << "\" for target SpinSystem \"" << this->target->Name() << "\"! Source SpinSystem is \"" << this->system->Name() << "\"!" << std::endl;
				return false;
			}

			// See if the target system has a state with the requested name
			this->targetstate = this->target->states_find(str);
			if (this->targetstate == nullptr)
			{
				std::cout << "Error: The target SpinSystem \"" << this->target->Name() << "\" has no state \"" << str << "\"! Failed to validate Transition \"" << this->Name() << "\"!" << std::endl;
				return false;
			}
		}

		this->isValid = true;
		return this->isValid;
	}
	// -----------------------------------------------------
	// Other public methods
	// -----------------------------------------------------
	// Returns the name of the Transition object
	std::string Transition::Name() const
	{
		return this->properties->Name();
	}

	// Returns the source system
	std::shared_ptr<SpinSystem> Transition::System() const
	{
		return this->system;
	}

	// Returns the target system (nullptr is none is specified)
	std::shared_ptr<SpinSystem> Transition::Target() const
	{
		return this->target;
	}

	// Returns the rate
	double Transition::Rate() const
	{
		return this->rate;
	}

	// Returns the transition type, e.g. source/decay or sink
	TransitionType Transition::Type() const
	{
		return this->type;
	}

	// Returns the source state
	state_ptr Transition::SourceState() const
	{
		return this->sourcestate;
	}

	// Returns the target state (nullptr if no target system is specified)
	state_ptr Transition::TargetState() const
	{
		return this->targetstate;
	}

	// Returns the reaction operator type specified for the transition
	ReactionOperatorType Transition::GetReactionOperatorType() const
	{
		return this->reactionOperators;
	}

	// Checks whether the Transition object was validated successfully
	bool Transition::IsValid() const
	{
		return this->isValid;
	}

	// Checks whether a trajectory with a Rate column was specified for the transition
	bool Transition::HasTrajectory() const
	{
		if (this->trajectory.Length() > 0 && this->trjHasRate)
			return true;

		return false;
	}

	// Returns the length of the trajectory
	unsigned int Transition::TrajectoryLength() const
	{
		return this->trajectory.Length();
	}

	// Set the current state of the Transition to a specific step in the trajectory (row in the trajectory file)
	bool Transition::SetTrajectoryStep(unsigned int _step)
	{
		// Make sure that we have a trajectory
		if (this->trajectory.Length() < 1 || _step >= this->trajectory.Length())
			return false;

		// Check whether the requested trajectory step is valid
		if (_step < this->trajectory.Length())
		{
			// Get the rate or lifetime at the requested trajectory step
			if (this->trjHasRate)
			{
				if (this->trjRateIsLifetime)
					this->rate = 1.0 / std::abs(this->trajectory.Get(_step, this->trjRate));
				else
					this->rate = std::abs(this->trajectory.Get(_step, this->trjRate));
			}
		}

		return true;
	}

	// Sets the current value of the rate constant to a value from the trajectory (requires a "time" column in the trajectory)
	bool Transition::SetTime(double _time)
	{
		if (!this->HasTrajectory())
			return false;

		// Get the row corresponding to the time right after (or at) the requested time
		unsigned int row = 0;
		double timestepAbove = 0.0;
		if (!this->trajectory.FirstRowEqGreaterThan(_time, this->trjTime, row, timestepAbove))
		{
			// If no such time was found, i.e. we are past the last element of the trajectory, use the last step of the trajectory
			return this->SetTrajectoryStep(this->trajectory.Length() - 1);
		}
		else if (row == 0)
		{
			// If the specified time is before the first element of the trajectory, use the first step of the trajectory
			return this->SetTrajectoryStep((unsigned int)0);
		}
		else
		{
			// Prepare a linear interpolation
			double timestepBelow = this->trajectory.Get(row - 1, trjTime);			  // Get time at previous step
			double l = 1 - (timestepAbove - _time) / (timestepAbove - timestepBelow); // Get linear interpolation factor, i.e. a number between 0 and 1, where 0 = previous time step and 1 = next time step in the trajectory

			// Get the rate or lifetime at the requested trajectory step
			if (this->trjHasRate)
			{
				if (this->trjRateIsLifetime)
					this->rate = 1.0 / (std::abs(this->trajectory.Get(row, this->trjRate)) * l + std::abs(this->trajectory.Get(row - 1, this->trjRate)) * (1 - l));
				else
					this->rate = std::abs(this->trajectory.Get(row, this->trjRate) * l + this->trajectory.Get(row - 1, this->trjRate) * (1 - l));
			}
		}

		return true;
	}
	// -----------------------------------------------------
	// Access to custom properties
	// -----------------------------------------------------
	std::shared_ptr<const MSDParser::ObjectParser> Transition::Properties() const
	{
		return this->properties;
	}
	// -----------------------------------------------------
	// Public spin subspace methods
	// -----------------------------------------------------
	// Checks whether any spins in the source state are coupled/entangled to any spins outside the source state
	bool Transition::IsComplete(const std::vector<spin_ptr> &_subspace) const
	{
		if (this->sourcestate == nullptr)
			return true;

		return this->sourcestate->IsComplete(_subspace);
	}

	// Checks whether any spins in the target state are coupled/entangled to any spins outside the target state
	bool Transition::IsTargetComplete(const std::vector<spin_ptr> &_subspace) const
	{
		if (this->targetstate == nullptr)
			return true;

		return this->targetstate->IsComplete(_subspace);
	}

	// Extends the set such that all spins inside the source state can only be coupled/entangled to other spins inside the source state
	// Returns true if the subspace was extended, false if it was already complete
	bool Transition::CompleteSet(std::vector<spin_ptr> &_subspace) const
	{
		if (this->sourcestate == nullptr)
			return false;

		return this->sourcestate->CompleteSet(_subspace);
	}

	// Extends the set such that all spins inside the target state can only be coupled/entangled to other spins inside the target state
	// Returns true if the subspace was extended, false if it was already complete
	bool Transition::CompleteTargetSet(std::vector<spin_ptr> &_subspace) const
	{
		if (this->targetstate == nullptr)
			return false;

		return this->targetstate->CompleteSet(_subspace);
	}
	// -----------------------------------------------------
	// Methods to create ActionTarget objects
	// -----------------------------------------------------
	// Create ActionScalar for the Rate of the transition
	std::vector<RunSection::NamedActionScalar> Transition::CreateActionScalars(const std::string &_system)
	{
		std::vector<RunSection::NamedActionScalar> scalars;

		if (this->IsValid())
		{
			// Do not expose a writable ActionTarget when the selected time profile
			// ignores Transition::Rate(). A writable-but-ineffective target is
			// worse than an explicit read-only contract because parameter scans can
			// otherwise appear to run while leaving the physical generator unchanged.
			const bool rateReadonly =
				this->trjHasRate || ProfileOverridesTransitionRate(this->properties);
			RunSection::ActionScalar rateScalar = RunSection::ActionScalar(
				this->rate, &CheckActionScalarTransitionRate, rateReadonly);
			scalars.push_back(RunSection::NamedActionScalar(
				_system + "." + this->Name() + ".rate", rateScalar));
		}

		return scalars;
	}

	// Method that calls the methods to generate ActionVectors and ActionScalars and inserts them into the given collections
	void Transition::GetActionTargets(std::vector<RunSection::NamedActionScalar> &_scalars, std::vector<RunSection::NamedActionVector> &_vectors, const std::string &_system)
	{
		// Get ActionTargets from private methods
		auto scalars = this->CreateActionScalars(_system);

		// Insert them
		_scalars.insert(_scalars.end(), scalars.begin(), scalars.end());
	}

	// -----------------------------------------------------
	// Non-member non-friend methods
	// -----------------------------------------------------
	bool IsValid(const Transition &_transition)
	{
		return _transition.IsValid();
	}

	bool IsStatic(const Transition &_transition)
	{
		return !_transition.HasTD();
	}

	bool HasTrajectory(const Transition &_transition)
	{
		if (_transition.HasTrajectory())
			return true;

		return false;
	}

	// -----------------------------------------------------
	// Non-member non-friend ActionTarget Check functions
	// -----------------------------------------------------
	// Make sure that the rate has a valid value (positive, not NaN or infinite)
	bool CheckActionScalarTransitionRate(const double &_d)
	{
		return (std::isfinite(_d) && _d >= 0.0);
	}
	// -----------------------------------------------------
}
