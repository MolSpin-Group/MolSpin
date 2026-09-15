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
#include <functional>
#include <stdexcept>

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

		//functional data source
		std::function<T()> dataSource;
		std::function<bool(const T &)> dataSetter;
		std::function<void()> dataResetter;

		// Other data members
		bool readonly;
		T initialValue;

		// Function pointers
		bool (*check)(const T &);

	public:
		// Constructors / Destructors
		ActionTarget(std::function<T()> _dataSource) :data(nullptr), dataSource(_dataSource), readonly(true), initialValue(T()), check(nullptr){};
		// Writable functional targets can update several underlying values.
		// A custom resetter restores their complete initial state when a single
		// scalar snapshot is insufficient. Getter-only targets stay read-only.
		ActionTarget(std::function<T()> _dataSource, std::function<bool(const T &)> _dataSetter,
			CheckFunction<T> _check = nullptr, std::function<void()> _dataResetter = nullptr)
			: data(nullptr), dataSource(_dataSource), dataSetter(_dataSetter), dataResetter(_dataResetter),
			  readonly(false), initialValue(T()), check(_check)
		{
			if (!dataSource || !dataSetter)
				throw std::invalid_argument("Writable functional ActionTarget requires a getter and setter");
			initialValue = dataSource();
		}
		ActionTarget(T &_data, bool _readonly = false) :data(&_data), dataSource(nullptr), readonly(_readonly), initialValue(_data), check(nullptr){};
		ActionTarget(T &_data, CheckFunction<T> _check, bool _readonly = false) : data(&_data), dataSource(nullptr),readonly(_readonly), initialValue(_data), check(_check){}; // Normal constructor
		ActionTarget(const ActionTarget<T> &_at) : data(_at.data),dataSource(_at.dataSource), dataSetter(_at.dataSetter), dataResetter(_at.dataResetter), readonly(_at.readonly), initialValue(_at.initialValue), check(_at.check){};			  // Copy-constructor
		~ActionTarget(){};																																  // Destructor

		// Operators
		ActionTarget<T> &operator=(const ActionTarget<T> &_at) // Copy-assignment
		{
			this->data = _at.data;
			this->dataSource = _at.dataSource;
			this->dataSetter = _at.dataSetter;
			this->dataResetter = _at.dataResetter;
			this->readonly = _at.readonly;
			this->initialValue = _at.initialValue;
			this->check = _at.check;

			return (*this);
		}

		// Get the value
		// It is guaranteed to be different from nullptr since a reference was passed to the constructor (or to the constructor of the object used to copy construct/assign)
		T Get() const
		{
			if(this->dataSource)
			{
				return this->dataSource();
			}
			return (*data);
		}

		// Set the value, if not a readonly ActionTarget
		// If a check-function was provided, use it to verify the input
		bool Set(const T &_in)
		{
			if (this->readonly || (this->check != nullptr && this->check(_in) == false))
				return false;
			if (this->dataSetter)
				return this->dataSetter(_in);
			if (this->dataSource || this->data == nullptr)
				return false;

			*(this->data) = _in;
			return true;
		}

		// Readonly ActionTargets does not have a Set method, but can be changed in other ways (i.e. the prefactor in an Interaction object may be changed by a trajectory)
		// Such changes in readonly-ActionTargets can be reversed to an initial value through this method.
		void Reset()
		{
			if (this->dataResetter)
			{
				this->dataResetter();
				return;
			}
			if (this->dataSetter)
			{
				this->dataSetter(this->initialValue);
				return;
			}
			if(!this->dataSource && this->data != nullptr)
			{
				*(this->data) = this->initialValue;
			}
		}

		// Other public methods
		bool HasCheck() const { return (this->check != nullptr); };
		bool IsReadonly() const { return this->readonly; }
		bool IsFunctional() const {return (this->dataSource != nullptr);}
	};

	using ActionScalar = ActionTarget<double>;
	using NamedActionScalar = std::pair<std::string, ActionScalar>;

	using ActionVector = ActionTarget<arma::vec>;
	using NamedActionVector = std::pair<std::string, ActionVector>;
}

#endif
