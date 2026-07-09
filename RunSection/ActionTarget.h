/////////////////////////////////////////////////////////////////////////
// ActionTarget (RunSection module)
// ------------------
// A class template describing a target for Action objects to act on.
//
// Defines ActionScalar (double) and ActionVector (arma::vec).
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_ActionTarget
#define MOD_RunSection_ActionTarget

#include <armadillo>

namespace RunSection
{
	template <class T>
	using CheckFunction = bool (*)(const T &);

	template <class T>
	class ActionTarget
	{
	private:
		// Data reference to act on
		T *data;

		// Other data members
		bool readonly;
		T initialValue;

		// Function pointers
		bool (*check)(const T &);

	public:
		// Constructors / Destructors
		ActionTarget(T &_data, CheckFunction<T> _check, bool _readonly = false) : data(&_data), readonly(_readonly), initialValue(_data), check(_check){}; // Normal constructor
		ActionTarget(const ActionTarget<T> &_at) : data(_at.data), readonly(_at.readonly), initialValue(_at.initialValue), check(_at.check){};			  // Copy-constructor
		~ActionTarget(){};																																	  // Destructor

		// Operators
		ActionTarget<T> &operator=(const ActionTarget<T> &_at) // Copy-assignment
		{
			this->data = _at.data;
			this->readonly = _at.readonly;
			this->initialValue = _at.initialValue;
			this->check = _at.check;

			return (*this);
		}

		// Get the value
		// It is guaranteed to be different from nullptr since a reference was passed to the constructor (or to the constructor of the object used to copy construct/assign)
		T Get() const
		{
			return (*data);
		}

		// Set the value, if not a readonly ActionTarget
		// If a check-function was provided, use it to verify the input
		bool Set(const T &_in)
		{
			if (this->readonly || (this->check != nullptr && this->check(_in) == false))
				return false;

			*(this->data) = _in;
			return true;
		}

		// Readonly ActionTargets does not have a Set method, but can be changed in other ways (i.e. the prefactor in an Interaction object may be changed by a trajectory)
		// Such changes in readonly-ActionTargets can be reversed to an initial value through this method.
		void Reset()
		{
			*(this->data) = this->initialValue;
		}

		// Other public methods
		bool HasCheck() const { return (this->check != nullptr); };
		bool IsReadonly() const { return this->readonly; }
	};

	using ActionScalar = ActionTarget<double>;
	using NamedActionScalar = std::pair<std::string, ActionScalar>;

	using ActionVector = ActionTarget<arma::vec>;
	using NamedActionVector = std::pair<std::string, ActionVector>;
}

#endif
