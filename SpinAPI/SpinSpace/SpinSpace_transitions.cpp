#include "SpinSpace.h"
/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// This source file contains methods related to transition objects,
// i.e. reaction operators.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
namespace SpinAPI
{
	// -----------------------------------------------------
	// Transitions/decay operators
	// -----------------------------------------------------
	// Provides the operator "k/2 * P" in Hilbert space, or the
	// anti-commutator operator "k/2 * {P,.}" in superoperator space
	// where P is the projection onto a state
	bool SpinSpace::ReactionOperator(const transition_ptr &_transition, arma::cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		// Make sure that we have a valid transition object
		if (_transition == nullptr || !_transition->IsValid())
			return false;

		// Get a Hilbert space projection operator onto the source state
		arma::cx_mat P;
		if (!this->GetState(_transition->SourceState(), P))
			return false;

		// Multiply by k/2
		P *= _transition->Rate() / 2.0;

		// Check whether we need to convert to superspace
		if (this->useSuperspace)
		{
			// Figure out what type of reaction operator to create (only relevant for superspace)
			ReactionOperatorType ROT = this->reactionOperators;
			if (_forcedReactionOperatorType != ReactionOperatorType::Unspecified)
				ROT = _forcedReactionOperatorType;
			else if (_transition->GetReactionOperatorType() != ReactionOperatorType::Unspecified)
				ROT = _transition->GetReactionOperatorType();

			if (ROT == ReactionOperatorType::Lindblad)
			{
				// ---------------- Lindblad form ----------------
				// Get the first operator (P * . * conj(P))
				arma::cx_mat PF;
				if (!this->SuperoperatorFromOperators(P, P.t(), PF))
					return false;

				// Get the left-hand operator (conj(P)*P * .)
				arma::cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P.t() * P, PL))
					return false;

				// Get the right-hand operator (. * conj(P)*P)
				arma::cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P.t() * P, PR))
					return false;

				// Set the total operator
				_out = 2.0 * PF - (PL + PR);
			}
			else
			{
				// ---------------- Haberkorn form ----------------
				// Get the left-hand operator (P * .)
				arma::cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P, PL))
					return false;

				// Get the right-hand operator (. * P)
				arma::cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P, PR))
					return false;

				// Set the total operator
				_out = (PL + PR);
			}
		}
		else
		{
			_out = P;
		}

		return true;
	}

	// Sparse version of the ReactionOperator method
	bool SpinSpace::ReactionOperator(const transition_ptr &_transition, arma::sp_cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		// Make sure that we have a valid transition object
		if (_transition == nullptr || !_transition->IsValid())
			return false;

		// Get a Hilbert space projection operator onto the source state
		arma::sp_cx_mat P;
		if (!this->GetState(_transition->SourceState(), P))
			return false;

		// Multiply by k/2
		P *= _transition->Rate() / 2.0;

		// Check whether we need to convert to superspace
		if (this->useSuperspace)
		{
			// Figure out what type of reaction operator to create (only relevant for superspace)
			ReactionOperatorType ROT = this->reactionOperators;
			if (_forcedReactionOperatorType != ReactionOperatorType::Unspecified)
				ROT = _forcedReactionOperatorType;
			else if (_transition->GetReactionOperatorType() != ReactionOperatorType::Unspecified)
				ROT = _transition->GetReactionOperatorType();

			if (ROT == ReactionOperatorType::Lindblad)
			{
				// ---------------- Lindblad form ----------------
				// Get the first operator (P * . * conj(P))
				arma::sp_cx_mat PF;
				if (!this->SuperoperatorFromOperators(P, P.t(), PF))
					return false;

				// Get the left-hand operator (conj(P)*P * .)
				arma::sp_cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P.t() * P, PL))
					return false;

				// Get the right-hand operator (. * conj(P)*P)
				arma::sp_cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P.t() * P, PR))
					return false;

				// Set the total operator
				_out = 2.0 * PF - (PL + PR);
			}
			else
			{
				// ---------------- Haberkorn form ----------------
				// Get the left-hand operator (P * .)
				arma::sp_cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P, PL))
					return false;

				// Get the right-hand operator (. * P)
				arma::sp_cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P, PR))
					return false;

				// Set the total operator
				_out = (PL + PR);
			}
		}
		else
		{
			_out = P;
		}

		return true;
	}

	// Sets the dense matrix to the sum of all the reaction operators
	bool SpinSpace::TotalReactionOperator(arma::cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		// If we don't have any transitions, the total reaction operator is zero
		if (this->transitions.size() < 1)
		{
			_out = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Get the first transition contribution
		auto i = this->transitions.cbegin();
		arma::cx_mat tmp;
		arma::cx_mat result;
		if (!this->ReactionOperator((*i), result, _forcedReactionOperatorType))
			return false;

		// We have already used the first transition
		i++;

		// Loop through the rest
		for (; i != this->transitions.cend(); i++)
		{
			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!this->ReactionOperator((*i), tmp, _forcedReactionOperatorType))
				return false;
			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the sum of all the reaction operators
	bool SpinSpace::TotalReactionOperator(arma::sp_cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		// If we don't have any transitions, the total reaction operator is zero
		if (this->transitions.size() < 1)
		{
			_out = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Get the first transition contribution
		auto i = this->transitions.cbegin();
		arma::sp_cx_mat tmp;
		arma::sp_cx_mat result;
		if (!this->ReactionOperator((*i), result, _forcedReactionOperatorType))
			return false;

		// We have already used the first transition
		i++;

		// Loop through the rest
		for (; i != this->transitions.cend(); i++)
		{
			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!this->ReactionOperator((*i), tmp, _forcedReactionOperatorType))
				return false;
			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the total reaction operator that is independent of time or trajectory step
	bool SpinSpace::StaticTotalReactionOperator(arma::cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType, bool NoInterSystem) const
	{
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());

		// If we don't have any transitions, the total reaction operator is zero
		if (this->transitions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
		{
			// Skip any dynamic (time-dependent) transitions
			if (!IsStatic(*(*i)))
				continue;
			
			if(NoInterSystem && (*i)->Target() != nullptr)
				continue;

			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!this->ReactionOperator((*i), tmp, _forcedReactionOperatorType))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the part of the total reaction operator that is independent of time or trajectory step
	bool SpinSpace::StaticTotalReactionOperator(arma::sp_cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());

		// If we don't have any transitions, the total reaction operator is zero
		if (this->transitions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
		{
			// Skip any dynamic (time-dependent) transitions
			if (!IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!this->ReactionOperator((*i), tmp, _forcedReactionOperatorType))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the total reaction operator that is dependent on time and/or trajectory step
	bool SpinSpace::DynamicTotalReactionOperator(arma::cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());

		// If we don't have any transitions, the total reaction operator is zero
		if (this->transitions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
		{
			// Skip static transitions
			if (IsStatic(*(*i)))
				continue;

			if(!(*i)->IsActive())
				continue;

			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!this->ReactionOperator((*i), tmp, _forcedReactionOperatorType))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the part of the total reaction operator that is dependent on time and/or trajectory step
	bool SpinSpace::DynamicTotalReactionOperator(arma::sp_cx_mat &_out, const ReactionOperatorType &_forcedReactionOperatorType) const
	{
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());

		// If we don't have any transitions, the total reaction operator is zero
		if (this->transitions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		for (auto i = this->transitions.cbegin(); i != this->transitions.cend(); i++)
		{
			// Skip static transitions
			if (IsStatic(*(*i)))
				continue;

			if(!(*i)->IsActive())
				continue;

			// Attempt to get the matrix representing the reaction operator in the spin space
			if (!this->ReactionOperator((*i), tmp, _forcedReactionOperatorType))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Returns the reaction operator type, e.g. Haberkorn reaction operators
	ReactionOperatorType SpinSpace::GetReactionOperatorType() const
	{
		return this->reactionOperators;
	}

	// -----------------------------------------------------
	// Transitions/decay operators in the target system
	// -----------------------------------------------------
	// An alternative to the use of the creation operators that are defined below.
	// These operators appear in the diagonal of the multi-system space (the creation operators defined below
	// appear in the off-diagonal blocks).
	bool SpinSpace::ReactionTargetOperator(const transition_ptr &_transition, double _fraction, arma::cx_mat &_out) const
	{
		// Make sure that we have a valid transition object
		if (_transition == nullptr || !_transition->IsValid() || _transition->TargetState() == nullptr)
			return false;

		// Get a Hilbert space projection operator onto the target state
		arma::cx_mat P;
		if (!this->GetState(_transition->TargetState(), P))
			return false;

		// Multiply by k
		// NOTE: The _fraction parameter describes the amount of source state in the source system
		P *= _transition->Rate() * _fraction;

		// Check whether we need to convert to superspace
		if (this->useSuperspace)
		{
			if (this->reactionOperators == ReactionOperatorType::Lindblad)
			{
				// ---------------- Lindblad form ----------------
				// Get the first operator (P * . * conj(P))
				arma::cx_mat PF;
				if (!this->SuperoperatorFromOperators(P, P.t(), PF))
					return false;

				// Get the left-hand operator (conj(P)*P * .)
				arma::cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P.t() * P, PL))
					return false;

				// Get the right-hand operator (. * conj(P)*P)
				arma::cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P.t() * P, PR))
					return false;

				// Set the total operator
				_out = PF - 0.5 * (PL + PR);
			}
			else
			{
				// ---------------- Haberkorn form ----------------
				// Get the left-hand operator (P * .)
				arma::cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P, PL))
					return false;

				// Get the right-hand operator (. * P)
				arma::cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P, PR))
					return false;

				// Set the total operator
				_out = (PL + PR);
			}
		}
		else
		{
			_out = P;
		}

		return true;
	}

	// Sparse version of the ReactionTargetOperator method
	bool SpinSpace::ReactionTargetOperator(const transition_ptr &_transition, double _fraction, arma::sp_cx_mat &_out) const
	{
		// Make sure that we have a valid transition object
		if (_transition == nullptr || !_transition->IsValid() || _transition->TargetState() == nullptr)
			return false;

		// Get a Hilbert space projection operator onto the target state
		arma::sp_cx_mat P;
		if (!this->GetState(_transition->TargetState(), P))
			return false;

		// Multiply by k
		// NOTE: The _fraction parameter describes the amount of source state in the source system
		P *= _transition->Rate() * _fraction;

		// Check whether we need to convert to superspace
		if (this->useSuperspace)
		{
			if (this->reactionOperators == ReactionOperatorType::Lindblad)
			{
				// ---------------- Lindblad form ----------------
				// Get the first operator (P * . * conj(P))
				arma::sp_cx_mat PF;
				if (!this->SuperoperatorFromOperators(P, P.t(), PF))
					return false;

				// Get the left-hand operator (conj(P)*P * .)
				arma::sp_cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P.t() * P, PL))
					return false;

				// Get the right-hand operator (. * conj(P)*P)
				arma::sp_cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P.t() * P, PR))
					return false;

				// Set the total operator
				_out = PF - 0.5 * (PL + PR);
			}
			else
			{
				// ---------------- Haberkorn form ----------------
				// Get the left-hand operator (P * .)
				arma::sp_cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P, PL))
					return false;

				// Get the right-hand operator (. * P)
				arma::sp_cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P, PR))
					return false;

				// Set the total operator
				_out = (PL + PR);
			}
		}
		else
		{
			_out = P;
			//std::cout << _out << std::endl;
		}

		//std::cout << _out << std::endl;

		return true;
	}

    bool SpinSpace::ReactionSourceOperator(const transition_ptr &_transition, double _fraction, arma::cx_mat &_out) const
    {
        // Make sure that we have a valid transition object
		if (_transition == nullptr || !_transition->IsValid())
			return false;

		// Get a Hilbert space projection operator onto the source state
		arma::cx_mat P;
		if (!this->GetState(_transition->SourceState(), P))
			return false;

		// Multiply by k
		// NOTE: The _fraction parameter describes the amount of source state in the source system
		P *= _transition->Rate() * _fraction;

		// Check whether we need to convert to superspace
		if (this->useSuperspace)
		{
			if (this->reactionOperators == ReactionOperatorType::Lindblad)
			{
				// ---------------- Lindblad form ----------------
				// Get the first operator (P * . * conj(P))
				arma::cx_mat PF;
				if (!this->SuperoperatorFromOperators(P, P.t(), PF))
					return false;

				// Get the left-hand operator (conj(P)*P * .)
				arma::cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P.t() * P, PL))
					return false;

				// Get the right-hand operator (. * conj(P)*P)
				arma::cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P.t() * P, PR))
					return false;

				// Set the total operator
				_out = PF - 0.5 * (PL + PR);
			}
			else
			{
				// ---------------- Haberkorn form ----------------
				// Get the left-hand operator (P * .)
				arma::cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P, PL))
					return false;

				// Get the right-hand operator (. * P)
				arma::cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P, PR))
					return false;

				// Set the total operator
				_out = (PL + PR);
			}
		}
		else
		{
			_out = P;
			//std::cout << _out << std::endl;
		}

		return true;
    }

    bool SpinSpace::ReactionSourceOperator(const transition_ptr &_transition, double _fraction, arma::sp_cx_mat &_out) const
    {
        // Make sure that we have a valid transition object
		if (_transition == nullptr || !_transition->IsValid())
			return false;

		// Get a Hilbert space projection operator onto the source state
		arma::sp_cx_mat P;
		if (!this->GetState(_transition->SourceState(), P))
			return false;

		// Multiply by k
		// NOTE: The _fraction parameter describes the amount of source state in the source system
		P *= _transition->Rate() * _fraction;

		// Check whether we need to convert to superspace
		if (this->useSuperspace)
		{
			if (this->reactionOperators == ReactionOperatorType::Lindblad)
			{
				// ---------------- Lindblad form ----------------
				// Get the first operator (P * . * conj(P))
				arma::sp_cx_mat PF;
				if (!this->SuperoperatorFromOperators(P, P.t(), PF))
					return false;

				// Get the left-hand operator (conj(P)*P * .)
				arma::sp_cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P.t() * P, PL))
					return false;

				// Get the right-hand operator (. * conj(P)*P)
				arma::sp_cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P.t() * P, PR))
					return false;

				// Set the total operator
				_out = PF - 0.5 * (PL + PR);
			}
			else
			{
				// ---------------- Haberkorn form ----------------
				// Get the left-hand operator (P * .)
				arma::sp_cx_mat PL;
				if (!this->SuperoperatorFromLeftOperator(P, PL))
					return false;

				// Get the right-hand operator (. * P)
				arma::sp_cx_mat PR;
				if (!this->SuperoperatorFromRightOperator(P, PR))
					return false;

				// Set the total operator
				_out = (PL + PR);
			}
		}
		else
		{
			_out = P;
			//std::cout << _out << std::endl;
		}

		return true;
    }

    // -----------------------------------------------------
    // Creation operators (non-member non-friend functions)
	// -----------------------------------------------------
	// Provides a creation operator that allows transitions between two spin systems
	bool CreationOperator(const transition_ptr &_transition, const SpinSpace &_sourceSpace, const SpinSpace &_targetSpace, arma::cx_mat &_out, bool _useSuperoperatorSpace)
	{
		// Make sure that we have a valid transition object, with a target system specified
		if (_transition == nullptr || !_transition->IsValid() || _transition->TargetState() == nullptr)
			return false;

		// Get a Hilbert space state vector of both the source state and the target state
		arma::cx_vec S;
		arma::cx_vec T;
		if (!_sourceSpace.GetState(_transition->SourceState(), S) || !_targetSpace.GetState(_transition->TargetState(), T))
			return false;

		// Make sure the states are normalized
		S /= arma::norm(S);
		T /= arma::norm(T);

		// Obtain the creation operator
		arma::cx_mat C = T * S.t();

		// Check whether we need to convert to superspace
		if (_useSuperoperatorSpace)
		{
			// Get the Liouville-space representation with C operating both from the left and the right
			// Note that the right-operator is the Hermitian conjugate of C
			arma::cx_mat CL;
			if (!_sourceSpace.SuperoperatorFromOperators(C, C.t(), CL))
				return false;

			// Set the output operator
			_out = CL;
		}
		else
		{
			_out = C;
		}

		return true;
	}

	// Provides a sparse matrix creation operator that allows transitions between two spin systems
	// TODO: Make a more efficient sparse matrix implementation
	bool CreationOperator(const transition_ptr &_transition, const SpinSpace &_sourceSpace, const SpinSpace &_targetSpace, arma::sp_cx_mat &_out, bool _useSuperoperatorSpace)
	{
		// Make sure that we have a valid transition object, with a target system specified
		if (_transition == nullptr || !_transition->IsValid() || _transition->TargetState() == nullptr || _transition->IsActive() == false)
			return false;

		// Get a Hilbert space state vector of both the source state and the target state
		arma::cx_vec S;
		arma::cx_vec T;
		if (!_sourceSpace.GetState(_transition->SourceState(), S) || !_targetSpace.GetState(_transition->TargetState(), T))
			return false;

		// Make sure the states are normalized
		S /= arma::norm(S);
		T /= arma::norm(T);

		// Obtain the creation operator
		arma::sp_cx_mat C = arma::conv_to<arma::sp_cx_mat>::from(T * S.t());

		// Check whether we need to convert to superspace
		if (_useSuperoperatorSpace)
		{
			// Get the Liouville-space representation with C operating both from the left and the right
			// Note that the right-operator is the Hermitian conjugate of C
			arma::sp_cx_mat CL;
			if (!_sourceSpace.SuperoperatorFromOperators(C, C.t(), CL))
				return false;

			// Set the output operator
			_out = CL;
		}
		else
		{
			_out = C;
		}

		return true;
	}
}
