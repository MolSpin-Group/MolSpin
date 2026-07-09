/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// This source file contains methods for creating single-spin operators,
// and for converting operators or vectors to/from Superspace.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
namespace SpinAPI
{
	// -----------------------------------------------------
	// Operator representations in the space
	// ---
	// Returns an operator (matrix) defined in the spin space.
	// Note that the matrix dimensions must fit NxN where N is the multiplicity of the spin.
	// -----------------------------------------------------
	// A block in the diagonal (dense)
	bool SpinSpace::CreateOperator(const arma::cx_mat &_operator, const spin_ptr &_spin, arma::cx_mat &_out) const
	{
		// Validate the input
		if (_spin == nullptr)
			return false;

		// Check that the given operator is valid
		if (_operator.n_rows != static_cast<unsigned int>(_spin->Multiplicity()) || _operator.n_cols != static_cast<unsigned int>(_spin->Multiplicity()))
			return false;

		// If the spin is outside the space, we just have an identity inside the space
		if (!this->Contains(_spin))
		{
			_out = arma::eye<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Prepare by getting the matrix in the space of the first spin
		arma::cx_mat result;
		auto i = this->spins.cbegin();

		if ((*i) != _spin)
			result = arma::eye<arma::cx_mat>((*i)->Multiplicity(), (*i)->Multiplicity());
		else
			result = _operator;

		// Iterate over the remaining spins
		++i; // We used the first one
		arma::cx_mat tmp;
		for (; i != this->spins.cend(); i++)
		{
			if ((*i) != _spin)
				tmp = arma::kron(result, arma::eye<arma::cx_mat>((*i)->Multiplicity(), (*i)->Multiplicity()));
			else
				tmp = arma::kron(result, _operator);

			result = tmp;
		}

		_out = result;
		return true;
	}

	// Create operator in the spin space from a single-spin operator (sparse version)
	bool SpinSpace::CreateOperator(const arma::sp_cx_mat &_operator, const spin_ptr &_spin, arma::sp_cx_mat &_out) const
	{
		// Validate the input
		if (_spin == nullptr)
			return false;

		// Check that the given operator is valid
		if (_operator.n_rows != static_cast<unsigned int>(_spin->Multiplicity()) || _operator.n_cols != static_cast<unsigned int>(_spin->Multiplicity()))
			return false;

		// If the spin is outside the space, we just have an identity inside the space
		if (!this->Contains(_spin))
		{
			_out = arma::speye<arma::sp_cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Prepare by getting the matrix in the space of the first spin
		arma::sp_cx_mat result;
		auto i = this->spins.cbegin();

		if ((*i) != _spin)
			result = arma::speye<arma::sp_cx_mat>((*i)->Multiplicity(), (*i)->Multiplicity());
		else
			result = _operator;

		// Iterate over the remaining spins
		++i; // We used the first one
		arma::sp_cx_mat tmp;
		for (; i != this->spins.cend(); i++)
		{
			if ((*i) != _spin)
				tmp = arma::kron(result, arma::speye<arma::sp_cx_mat>((*i)->Multiplicity(), (*i)->Multiplicity()));
			else
				tmp = arma::kron(result, _operator);

			result = tmp;
		}

		_out = result;
		return true;
	}

	// Converts an operator to a superspace-vector
	bool SpinSpace::OperatorToSuperspace(const arma::cx_mat &_in, arma::cx_vec &_out) const // TODO: Check matrix dimensions
	{
		_out = arma::vectorise(_in.st());
		return true;
	}

	// Sparse version
	// Implemented in terms of the dense version as it produces a "dense vector" anyway
	bool SpinSpace::OperatorToSuperspace(const arma::sp_cx_mat &_in, arma::cx_vec &_out) const
	{
		arma::cx_mat tmp(_in);
		return this->OperatorToSuperspace(tmp, _out);
	}

	// Converts a superspace vector to an operator in the Hilbert space
	bool SpinSpace::OperatorFromSuperspace(const arma::cx_vec &_in, arma::cx_mat &_out) const // TODO: Check matrix dimensions
	{
		_out = arma::cx_mat(_in);
		_out.reshape(sqrt(_in.size()), sqrt(_in.size()));

		// Change by Luca Gerhards -- Needs to be done if matrix is not fully symmetric due to the filling of rows in c++
		_out = _out.st();

		return true;
	}

	// Sparse version
	// Implemented in terms of the dense version as it relies on a "dense vector" anyway
	bool SpinSpace::OperatorFromSuperspace(const arma::cx_vec &_in, arma::sp_cx_mat &_out) const
	{
		arma::cx_mat output;
		auto result = this->OperatorFromSuperspace(_in, output);
		_out = arma::conv_to<arma::sp_cx_mat>::from(output);
		return result;
	}

	// Defines the superoperator A(x) such that A(x) ~ L*x*R^T, where L and R are the left-side and right-side operators acting on an operator x, and ^T means transpose
	// Note: If you use an identity operator for either the left- or right-side operator, it is more efficient to use the other methods defined below.
	bool SpinSpace::SuperoperatorFromOperators(const arma::cx_mat &_leftside, const arma::cx_mat &_rightside, arma::cx_mat &_out) const // TODO: Check matrix dimensions
	{
		_out = kron(_leftside, _rightside.st());
		return true;
	}

	// Sparse version
	bool SpinSpace::SuperoperatorFromOperators(const arma::sp_cx_mat &_leftside, const arma::sp_cx_mat &_rightside, arma::sp_cx_mat &_out) const
	{
		_out = kron(_leftside, _rightside.st());
		return true;
	}

	// Defines the superoperator A(x) such that A(x) ~ L*x*I, where L is the left-side operator acting on an operator x, and I is the identity
	bool SpinSpace::SuperoperatorFromLeftOperator(const arma::cx_mat &_leftside, arma::cx_mat &_out) const // TODO: Implement more efficient algorithm, and check matrix dimensions
	{
		_out = kron(_leftside, arma::eye<arma::cx_mat>(arma::size(_leftside)));
		return true;
	}

	// Sparse version
	bool SpinSpace::SuperoperatorFromLeftOperator(const arma::sp_cx_mat &_leftside, arma::sp_cx_mat &_out) const
	{
		_out = kron(_leftside, arma::speye<arma::sp_cx_mat>(arma::size(_leftside)));
		return true;
	}

	// Defines the superoperator A(x) such that A(x) ~ I*x*R^T, where R is the right-side operator acting on an operator x, I is the identity, and ^T means transpose
	bool SpinSpace::SuperoperatorFromRightOperator(const arma::cx_mat &_rightside, arma::cx_mat &_out) const // TODO: Implement more efficient algorithm, and check matrix dimensions
	{
		_out = kron(arma::eye<arma::cx_mat>(arma::size(_rightside)), _rightside.st());
		return true;
	}

	// Sparse version
	bool SpinSpace::SuperoperatorFromRightOperator(const arma::sp_cx_mat &_rightside, arma::sp_cx_mat &_out) const
	{
		_out = kron(arma::speye<arma::sp_cx_mat>(arma::size(_rightside)), _rightside.st());
		return true;
	}

	// -----------------------------------------------------
	// Methods for reordering the basis, i.e. change the
	// order in which the individual spins appear in the
	// Kronecker products.
	// -----------------------------------------------------
	// Reordering to the native order of the SpinSpace is done in terms of the more general method
	bool SpinSpace::ReorderBasis(arma::cx_vec &_out, const std::vector<spin_ptr> &_oldBasis) const
	{
		return this->ReorderBasis(_out, _oldBasis, this->spins);
	}
	bool SpinSpace::ReorderBasis(arma::cx_mat &_out, const std::vector<spin_ptr> &_oldBasis) const
	{
		return this->ReorderBasis(_out, _oldBasis, this->spins);
	}
	bool SpinSpace::ReorderBasis(arma::sp_cx_mat &_out, const std::vector<spin_ptr> &_oldBasis) const
	{
		return this->ReorderBasis(_out, _oldBasis, this->spins);
	}

	// Reordering by application of the reordering operator (vector)
	bool SpinSpace::ReorderBasis(arma::cx_vec &_out, const std::vector<spin_ptr> &_oldBasis, const std::vector<spin_ptr> &_newBasis) const
	{
		// Obtain the reordering operator
		arma::sp_cx_mat R;
		if (!this->ReorderingOperator(R, _oldBasis, _newBasis))
			return false;

		// Perform the reordering
		arma::cx_vec tmp = R * _out;
		_out = tmp;

		return true;
	}

	// Reordering by application of the reordering operator (dense matrix)
	bool SpinSpace::ReorderBasis(arma::cx_mat &_out, const std::vector<spin_ptr> &_oldBasis, const std::vector<spin_ptr> &_newBasis) const
	{
		// Obtain the reordering operator
		arma::sp_cx_mat R;
		if (!this->ReorderingOperator(R, _oldBasis, _newBasis))
			return false;

		// Perform the reordering
		arma::cx_mat tmp = R * _out * R.t();
		_out = tmp;

		return true;
	}

	// Reordering by application of the reordering operator (sparse matrix)
	bool SpinSpace::ReorderBasis(arma::sp_cx_mat &_out, const std::vector<spin_ptr> &_oldBasis, const std::vector<spin_ptr> &_newBasis) const
	{
		// Obtain the reordering operator
		arma::sp_cx_mat R;
		if (!this->ReorderingOperator(R, _oldBasis, _newBasis))
			return false;

		// Perform the reordering
		arma::sp_cx_mat tmp = R * _out * R.t();
		_out = tmp;

		return true;
	}

	// Calculation of the reordering operator
	bool SpinSpace::ReorderingOperator(arma::sp_cx_mat &_out, const std::vector<spin_ptr> &_oldBasis, const std::vector<spin_ptr> &_newBasis) const
	{
		// Check that the two spin collections contain the same number of spins (we are only reordering here!)
		if (_oldBasis.size() != _newBasis.size())
			return false;

		// The contained spins should also be identical
		for (auto i = _oldBasis.cbegin(); i != _oldBasis.cend(); i++)
			if (std::find(_newBasis.cbegin(), _newBasis.cend(), *i) == _newBasis.cend())
				return false;

		// Get the dimensions of the spin space
		unsigned int dimensions = 1;
		for (auto i = _oldBasis.cbegin(); i != _oldBasis.cend(); i++)
			dimensions *= static_cast<unsigned int>((*i)->Multiplicity());

		// Compute the reordering operator
		bool isFirst = true;
		std::vector<spin_ptr> tempBasis = _oldBasis;
		arma::sp_cx_mat R(dimensions, dimensions);
		arma::sp_cx_mat PrevId;
		arma::sp_cx_mat PostId;
		for (auto i = _newBasis.cbegin(); i != _newBasis.cend(); i++)
		{
			// Get an iterator to the current spin in the old basis, and its position in the old basis
			auto target = std::find(tempBasis.cbegin(), tempBasis.cend(), *i);
			auto targetPosition = target - tempBasis.cbegin();
			auto newPosition = i - _newBasis.cbegin();

			// Check whether both spins have the same position, as no action is required in that case
			while (targetPosition > newPosition)
			{
				// Get the dimensions of the spin subspace consiting of the spins before the target and its previous spin (that we are swapping with)
				unsigned int subsetDimensions = 1;
				for (auto j = tempBasis.cbegin(); j != target - 1; j++)
					subsetDimensions *= static_cast<unsigned int>((*j)->Multiplicity());

				// Get the identity operator of the spin subspace that has been fixed so far
				PrevId = arma::eye<arma::sp_cx_mat>(subsetDimensions, subsetDimensions);

				// Also get the identity operator of the remaining subspace
				subsetDimensions = 1;
				for (auto j = target + 1; j != tempBasis.cend(); j++)
					subsetDimensions *= static_cast<unsigned int>((*j)->Multiplicity());
				PostId = arma::eye<arma::sp_cx_mat>(subsetDimensions, subsetDimensions);

				// Produce swapping operator in the two-spin subspace of the spins to be swapped
				// NOTE: This operator is known as the perfect shuffle of the Kronecker product
				unsigned int p = static_cast<unsigned int>((*(target - 1))->Multiplicity());
				unsigned int q = static_cast<unsigned int>((*target)->Multiplicity());
				unsigned int r = p * q;
				arma::sp_cx_mat S = arma::zeros<arma::sp_cx_mat>(r, r);
				unsigned int s = 0;
				for (unsigned int j = 0; j < q; j++)
				{
					for (unsigned int k = j; k < r; k += q)
						S(s++, k) = 1;
				}

				// Get the total swapping operator
				arma::sp_cx_mat P = kron(PrevId, kron(S, PostId));

				// Append the operator to the total reordering operator
				if (isFirst)
				{
					R = P;
					isFirst = false;
				}
				else
				{
					S = P * R;
					R = S;
				}

				// Swap neighbours in the old basis
				std::iter_swap(tempBasis.begin() + targetPosition, tempBasis.begin() + targetPosition - 1);

				// The target position is now closer to the beginning. Find is used again to make sure the iterator is valid after swap
				target = std::find(tempBasis.cbegin(), tempBasis.cend(), *i);
				targetPosition = target - tempBasis.cbegin();
			}
		}

		// If no reordering is necessary, return an identity
		if (isFirst)
			R = arma::eye<arma::sp_cx_mat>(dimensions, dimensions);

		// Set the result
		_out = R;

		return true;
	}

	// -----------------------------------------------------
	// Spherical tensors
	// -----------------------------------------------------

	//-----------------------------------------------------------------
	//-----------------RANK 1 TENSORS - BILINEAR-----------------------
	//-----------------------------------------------------------------

	// Rank 1 spherical tensors with m=0
	bool SpinSpace::Rk1SphericalTensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{

		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;
		arma::cx_mat S2x;
		arma::cx_mat S2y;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -1.0 * (arma::cx_double(0.0, 1.0) / sqrt(2.0)) * ((S1x * S2y) - (S1y * S2x));
		return true;
	}

	// Rank 1 spherical tensors with m=+1
	bool SpinSpace::Rk1SphericalTensorTp1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;
		arma::cx_mat S2x;
		arma::cx_mat S2y;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -0.5 * (S1z * S2x - S1x * S2y + (arma::cx_double(0.0, 1.0) * (S1z * S2y - S1y * S2z)));

		return true;
	}

	// Rank 1 spherical tensors with m=-1
	bool SpinSpace::Rk1SphericalTensorTm1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;
		arma::cx_mat S2x;
		arma::cx_mat S2y;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -0.5 * (S1z * S2x - S1x * S2y + (arma::cx_double(0.0, 1.0) * (S1z * S2y - S1y * S2z)));

		return true;
	}

	//-----------------------------------------------------------------
	//---------------------LINEAR RANK 2 TENSORS---------------------
	//-----------------------------------------------------------------

	// Rank 0 spherical tensor
	bool SpinSpace::LRk0TensorT0(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.0 / sqrt(3.0)) * (S1x * _field(0) + S1y * _field(1) + S1z * _field(2));
		return true;
	}

	// Rank 0 spherical tensor
	bool SpinSpace::LRk0TensorT0(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.0 / sqrt(3.0)) * (S1x * _field(0) + S1y * _field(1) + S1z * _field(2));
		return true;
	}

	// Rank 2 spherical tensor with m=0
	bool SpinSpace::LRk2SphericalTensorT0(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.00 / sqrt(6.0)) * ((3 * S1z * _field(2)) - (S1x * _field(0) + S1y * _field(1) + S1z * _field(2)));

		//_out = (S1p*S2m + S1m*S2p + 2.0*S1z*S2z) * 0.40824829;	// Factor of 1/sqrt(6) = 0.40824829
		return true;
	}

	// Rank 2 spherical tensor with m=0, sparse version
	bool SpinSpace::LRk2SphericalTensorT0(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.00 / sqrt(6.0)) * ((3 * S1z * _field(2)) - (S1x * _field(0) + S1y * _field(1) + S1z * _field(2)));
		//_out = (S1p*S2m + S1m*S2p + 2.0*S1z*S2z) * 0.40824829;	// Factor of 1/sqrt(6) = 0.40824829
		return true;
	}

	// Rank 2 spherical tensor with m=+1
	bool SpinSpace::LRk2SphericalTensorTp1(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -0.5 * (S1x * _field(2) + S1z * _field(0) + (arma::cx_double(0.0, 1.00) * (S1y * _field(2) + S1z * _field(1))));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=+1, sparse version
	bool SpinSpace::LRk2SphericalTensorTp1(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -0.5 * (S1x * _field(2) + S1z * _field(0) + (arma::cx_double(0.0, 1.00) * (S1y * _field(2) + S1z * _field(1))));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=-1
	bool SpinSpace::LRk2SphericalTensorTm1(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * _field(2) + S1z * _field(0) - (arma::cx_double(0.0, 1.00) * (S1y * _field(2) + S1z * _field(1))));
		//_out = (S1m*S2z + S1z*S2m) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=-1, sparse version
	bool SpinSpace::LRk2SphericalTensorTm1(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * _field(2) + S1z * _field(0) - (arma::cx_double(0.0, 1.00) * (S1y * _field(2) + S1z * _field(1))));
		//_out = (S1m*S2z + S1z*S2m) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=+2
	bool SpinSpace::LRk2SphericalTensorTp2(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * _field(0) - S1y * _field(1) + (arma::cx_double(0.0, 1.0) * (S1x * _field(1) + S1y * _field(0))));
		//_out = S1p*S2p;
		return true;
	}

	// Rank 2 spherical tensor with m=+2, sparse version
	bool SpinSpace::LRk2SphericalTensorTp2(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * _field(0) - S1y * _field(1) + (arma::cx_double(0.0, 1.0) * (S1x * _field(1) + S1y * _field(0))));
		//_out = S1p*S2p;
		return true;
	}

	// Rank 2 spherical tensor with m=-2
	bool SpinSpace::LRk2SphericalTensorTm2(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * _field(0) - S1y * _field(1) - (arma::cx_double(0.0, 1.0) * (S1x * _field(1) + S1y * _field(0))));
		//_out = S1m*S2m;
		return true;
	}

	// Rank 2 spherical tensor with m=-2, sparse version
	bool SpinSpace::LRk2SphericalTensorTm2(const spin_ptr &_spin1, const arma::cx_vec &_field, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * _field(0) - S1y * _field(1) - (arma::cx_double(0.0, 1.0) * (S1x * _field(1) + S1y * _field(0))));
		//_out = S1m*S2m;
		return true;
	}

	//-----------------------------------------------------------------
	//---------------------BILINEAR RANK 2 TENSORS---------------------
	//-----------------------------------------------------------------

	// Rank 0 spherical tensor
	bool SpinSpace::BlRk0TensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S1y;
		arma::cx_mat S1z;
		arma::cx_mat S2x;
		arma::cx_mat S2y;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.0 / sqrt(3.0)) * (S1x * S2x + S1y * S2y + S1z * S2z);
		return true;
	}

	// Rank 0 spherical tensor, sparse version
	bool SpinSpace::BlRk0TensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S1z;
		arma::sp_cx_mat S2x;
		arma::sp_cx_mat S2y;
		arma::sp_cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.0 / sqrt(3.0)) * (S1x * S2x + S1y * S2y + S1z * S2z);
		return true;
	}

	// Rank 2 spherical tensor with m=0
	bool SpinSpace::BlRk2SphericalTensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S2x;
		arma::cx_mat S1y;
		arma::cx_mat S2y;
		arma::cx_mat S1z;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.00 / sqrt(6.00)) * (3.00 * S1z * S2z - (S1x * S2x + S1y * S2y + S1z * S2z));
		//_out = (S1p*S2m + S1m*S2p + 2.0*S1z*S2z) * 0.40824829;	// Factor of 1/sqrt(6) = 0.40824829
		return true;
	}

	// Rank 2 spherical tensor with m=0, sparse version
	bool SpinSpace::BlRk2SphericalTensorT0(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S2x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S2y;
		arma::sp_cx_mat S1z;
		arma::sp_cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = (1.00 / sqrt(6.00)) * (3.00 * S1z * S2z - (S1x * S2x + S1y * S2y + S1z * S2z));
		//_out = (S1p*S2m + S1m*S2p + 2.0*S1z*S2z) * 0.40824829;	// Factor of 1/sqrt(6) = 0.40824829
		return true;
	}

	// Rank 2 spherical tensor with m=+1
	bool SpinSpace::BlRk2SphericalTensorTp1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S2x;
		arma::cx_mat S1y;
		arma::cx_mat S2y;
		arma::cx_mat S1z;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -0.5 * (S1x * S2z + S1z * S2x + (arma::cx_double(0.00, 1.00) * (S1y * S2z + S1z * S2y)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=+1, sparse version
	bool SpinSpace::BlRk2SphericalTensorTp1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S2x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S2y;
		arma::sp_cx_mat S1z;
		arma::sp_cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = -0.5 * (S1x * S2z + S1z * S2x + (arma::cx_double(0.00, 1.00) * (S1y * S2z + S1z * S2y)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=-1
	bool SpinSpace::BlRk2SphericalTensorTm1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S2x;
		arma::cx_mat S1y;
		arma::cx_mat S2y;
		arma::cx_mat S1z;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * S2z + S1z * S2x - (arma::cx_double(0.00, 1.00) * (S1y * S2z + S1z * S2y)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=-1, sparse version
	bool SpinSpace::BlRk2SphericalTensorTm1(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S2x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S2y;
		arma::sp_cx_mat S1z;
		arma::sp_cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * S2z + S1z * S2x - (arma::cx_double(0.00, 1.00) * (S1y * S2z + S1z * S2y)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=+2
	bool SpinSpace::BlRk2SphericalTensorTp2(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S2x;
		arma::cx_mat S1y;
		arma::cx_mat S2y;
		arma::cx_mat S1z;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * S2x - S1y * S2y + (arma::cx_double(0.0, 1.0) * (S1x * S2y + S1y * S2x)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=+2, sparse version
	bool SpinSpace::BlRk2SphericalTensorTp2(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S2x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S2y;
		arma::sp_cx_mat S1z;
		arma::sp_cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * S2x - S1y * S2y + (arma::cx_double(0.0, 1.0) * (S1x * S2y + S1y * S2x)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=-2
	bool SpinSpace::BlRk2SphericalTensorTm2(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::cx_mat &_out) const
	{
		arma::cx_mat S1x;
		arma::cx_mat S2x;
		arma::cx_mat S1y;
		arma::cx_mat S2y;
		arma::cx_mat S1z;
		arma::cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * S2x - S1y * S2y - (arma::cx_double(0.0, 1.0) * (S1x * S2y + S1y * S2x)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Rank 2 spherical tensor with m=-2, sparse version
	bool SpinSpace::BlRk2SphericalTensorTm2(const spin_ptr &_spin1, const spin_ptr &_spin2, arma::sp_cx_mat &_out) const
	{
		arma::sp_cx_mat S1x;
		arma::sp_cx_mat S2x;
		arma::sp_cx_mat S1y;
		arma::sp_cx_mat S2y;
		arma::sp_cx_mat S1z;
		arma::sp_cx_mat S2z;

		// We need all of these operators
		if (!this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sx()), _spin1, S1x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sy()), _spin1, S1y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin1->Sz()), _spin1, S1z) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sx()), _spin2, S2x) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sy()), _spin2, S2y) || !this->CreateOperator(arma::conv_to<arma::sp_cx_mat>::from(_spin2->Sz()), _spin2, S2z))
		{
			return false;
		}

		// Create the spherical tensor
		_out = 0.5 * (S1x * S2x - S1y * S2y - (arma::cx_double(0.0, 1.0) * (S1x * S2y + S1y * S2x)));
		//_out = (S1p*S2z + S1z*S2p) * 0.707106781;	// Factor of 1/sqrt(2) = 0.707106781
		return true;
	}

	// Stochastically generates a SU(Z) spin state
	arma::cx_colvec SpinSpace::SUZstate(const int &spinmult, std::mt19937 &generator)
	{

		arma::cx_colvec state(spinmult);
		std::normal_distribution<double> distr2(0.0, 1.0); // normal distribution with mean = 0 and variance 1

		for (int it = 0; it != spinmult; it++)
		{
			state(it, 0) = arma::cx_double(distr2(generator), distr2(generator));
		}
		state = normalise(state);
		return state;
	}
	// Creates a Coherent spin state
	arma::cx_colvec SpinSpace::CoherentState(std::vector<SpinAPI::system_ptr>::const_iterator i, std::mt19937 &generator)
	{
		arma::cx_colvec coherentstate(1);
		coherentstate(0, 0) = 1;
		std::uniform_real_distribution<double> distr(0, 1); // Uniform distribution between 0 and 1

		// Loops over nuclear spin states
		for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
		{
			std::string spintype;
			(*l)->Properties()->Get("type", spintype);
			if (spintype != "electron")
			{
				double theta = arma::datum::pi * distr(generator);
				double phi = arma::datum::pi * distr(generator);
				arma::cx_colvec tempstate;
				tempstate.zeros((*l)->Multiplicity());
				tempstate(0, 0) = 1;
				arma::cx_mat loweringop((*l)->Sm());
				tempstate = arma::expmat(tan(theta / 2.0) * exp(arma::cx_double(0.0, 1.0) * phi) * loweringop) * tempstate;
				tempstate = pow(cos(theta / 2.0), (*l)->Multiplicity() - 1) * tempstate;
				coherentstate = arma::kron(coherentstate, tempstate);
			}
		}

		return coherentstate;
	}

	arma::cx_mat SpinSpace::HighamProp(arma::sp_cx_mat &H, arma::cx_mat &B, const std::complex<double> t, const std::string precision, arma::mat &M)
	{
		int HilbSize = B.n_rows;
		int lengthB = B.n_cols; // Essentially how many collumns the B vector has. It is a dynamically found variable.

		bool shift = false;

		// Do a shift that increases execution speed.
		std::complex<double> mu = arma::trace(H) / arma::cx_double(HilbSize, 0.0);

		if (abs(mu) != 0)
		{
			H = H - mu * arma::speye(HilbSize, HilbSize);
			shift = true;
		}

		// Choosing Tolerance
		double tol = 0;

		if (precision == "double")
		{
			tol = std::pow(2, -53);
		}
		else if (precision == "single")
		{
			tol = std::pow(2, -24);
		}
		else if (precision == "half")
		{
			tol = std::pow(2, -10);
		}
		else
		{
			std::cerr << "# ERROR: Unsupported precision. Defaulting to double precision." << std::endl;
			tol = std::pow(2, -53);
		}

		// Select Taylor Degree Function

		// int pmax = 8;
		int mmax = 55;

		if (M.is_empty())
		{
			M = SelectTaylorDegree(t * H, precision, lengthB);
		}

		// End of taylor select function

		int m;
		double s = 0;
		if (abs(t) == 0)
		{
			m = 0;
		}
		else
		{
			arma::mat U(mmax, mmax);
			U.zeros(mmax, mmax);
			for (int it = 0; it < mmax; it++)
			{
				U(it, it) = it + 1;
			}

			arma::mat C = trans(abs(t) * ceil(M)) * U;

			C.replace(0, arma::datum::inf);

			int cost = C.min();

			arma::uvec index = ind2sub(size(C), C.index_min());
			m = index(1) + 1;
			arma::vec temp({cost / static_cast<double>(m), 1});
			s = temp.max();
		}
		arma::cx_mat F(B.n_rows, B.n_cols);
		F = B;

		std::complex<double> eta = arma::cx_double(1.0, 0.0);
		if (shift)
		{
			eta = std::exp(t * mu / arma::cx_double(s, 0.0));
		}

		for (int it1 = 0; it1 < s; it1++)
		{
			std::complex<double> c1 = arma::norm(B, "inf");
			for (int it2 = 0; it2 < m; it2++)
			{
				B = t / (s * (it2 + 1)) * H * B;
				std::complex<double> c2 = arma::norm(B, "inf");
				F = F + B;
				if (abs(c1 + c2) <= abs(tol * arma::norm(F, "inf")))
				{
					break;
				}
				c1 = c2;
			}
			F = eta * F;
			B = F;
		}
		if (shift)
		{
			H = H + mu * arma::speye(HilbSize, HilbSize);
		}

		return F;
	}

	// Get the Taylore degree function used in HighamProp
	arma::mat SpinSpace::SelectTaylorDegree(const arma::sp_cx_mat &H, const std::string precision, const int lengthB)
	{

		arma::vec theta;

		if (precision == "double")
		{
			theta = {2.22045e-16, 2.58096e-08, 1.38635e-05, 3.39717e-04, 2.40088e-03, 9.06566e-03, 2.38446e-02, 4.99123e-02, 8.95776e-02, 1.44183e-01, 2.14236e-01, 2.99616e-01, 3.99778e-01, 5.13915e-01, 6.41084e-01, 7.80287e-01, 9.30533e-01, 1.09086e+00, 1.26038e+00, 1.43825e+00, 1.62372e+00, 1.81608e+00, 2.01471e+00, 2.21905e+00, 2.42858e+00, 2.64285e+00, 2.86145e+00, 3.08400e+00, 3.31017e+00, 3.53967e+00, 3.77221e+00, 4.00756e+00, 4.24550e+00, 4.48582e+00, 4.72835e+00, 4.97292e+00, 5.21938e+00, 5.46759e+00, 5.71744e+00, 5.96880e+00, 6.22158e+00, 6.47568e+00, 6.73102e+00, 6.98750e+00, 7.24507e+00, 7.50365e+00, 7.76317e+00, 8.02359e+00, 8.28485e+00, 8.54690e+00, 8.80969e+00, 9.07319e+00, 9.33734e+00, 9.60212e+00, 9.86750e+00, 1.01334e+01, 1.03999e+01, 1.06669e+01, 1.09343e+01, 1.12022e+01, 1.14705e+01, 1.17392e+01, 1.20084e+01, 1.22778e+01, 1.25477e+01, 1.28178e+01, 1.30883e+01, 1.33591e+01, 1.36302e+01, 1.39016e+01, 1.41732e+01, 1.44451e+01, 1.47172e+01, 1.49896e+01, 1.52622e+01, 1.55350e+01, 1.58080e+01, 1.60812e+01, 1.63546e+01, 1.66281e+01, 1.69019e+01, 1.71758e+01, 1.74498e+01, 1.77240e+01, 1.79984e+01, 1.82729e+01, 1.85476e+01, 1.88223e+01, 1.90972e+01, 1.93723e+01, 1.96474e+01, 1.99227e+01, 2.01980e+01, 2.04735e+01, 2.07491e+01, 2.10248e+01, 2.13005e+01, 2.15764e+01, 2.18523e+01, 2.21284e+01};
		}
		else if (precision == "single")
		{
			theta = {1.19209e-07, 5.97886e-04, 1.12339e-02, 5.11662e-02, 1.30849e-01, 2.49529e-01, 4.01458e-01, 5.80052e-01, 7.79511e-01, 9.95184e-01, 1.22348e+00, 1.46166e+00, 1.70765e+00, 1.95985e+00, 2.21704e+00, 2.47828e+00, 2.74282e+00, 3.01007e+00, 3.27956e+00, 3.55093e+00, 3.82386e+00, 4.09810e+00, 4.37347e+00, 4.64978e+00, 4.92690e+00, 5.20471e+00, 5.48311e+00, 5.76201e+00, 6.04136e+00, 6.32108e+00, 6.60113e+00, 6.88146e+00, 7.16204e+00, 7.44283e+00, 7.72380e+00, 8.00493e+00, 8.28620e+00, 8.56759e+00, 8.84908e+00, 9.13065e+00, 9.41230e+00, 9.69402e+00, 9.97579e+00, 1.02576e+01, 1.05394e+01, 1.08213e+01, 1.11032e+01, 1.13852e+01, 1.16671e+01, 1.19490e+01, 1.22310e+01, 1.25129e+01, 1.27949e+01, 1.30769e+01, 1.33588e+01, 1.36407e+01, 1.39226e+01, 1.42046e+01, 1.44865e+01, 1.47684e+01};
		}
		else if (precision == "half")
		{
			theta = {1.95058e-03, 7.44366e-02, 2.66455e-01, 5.24205e-01, 8.10269e-01, 1.10823e+00, 1.41082e+00, 1.71493e+00, 2.01903e+00, 2.32247e+00, 2.62492e+00, 2.92630e+00, 3.22657e+00, 3.52578e+00, 3.82398e+00, 4.12123e+00, 4.41759e+00, 4.71314e+00, 5.00792e+00, 5.30200e+00, 5.59543e+00, 5.88825e+00, 6.18051e+00, 6.47224e+00, 6.76348e+00, 7.05427e+00, 7.34464e+00, 7.63460e+00, 7.92419e+00, 8.21342e+00, 8.50232e+00, 8.79091e+00, 9.07921e+00, 9.36722e+00, 9.65497e+00, 9.94247e+00, 1.02297e+01, 1.05168e+01, 1.08036e+01, 1.10902e+01, 1.13766e+01, 1.16629e+01, 1.19489e+01, 1.22348e+01, 1.25205e+01, 1.28061e+01, 1.30915e+01, 1.33768e+01, 1.36619e+01, 1.39469e+01, 1.42318e+01, 1.45165e+01, 1.48012e+01, 1.50857e+01, 1.53702e+01, 1.56547e+01, 1.59391e+01, 1.62235e+01, 1.65078e+01, 1.67920e+01, 1.70761e+01, 1.73599e+01, 1.76437e+01, 1.79273e+01, 1.82108e+01, 1.84943e+01, 1.87776e+01, 1.90609e+01, 1.93442e+01, 1.96273e+01, 1.99104e+01, 2.01935e+01, 2.04764e+01, 2.07593e+01, 2.10422e+01, 2.13250e+01, 2.16077e+01, 2.18904e+01, 2.21730e+01, 2.24556e+01, 2.27381e+01, 2.30206e+01, 2.33030e+01, 2.35854e+01, 2.38677e+01, 2.41500e+01, 2.44322e+01, 2.47144e+01, 2.49966e+01, 2.52787e+01, 2.55608e+01, 2.58428e+01, 2.61248e+01, 2.64068e+01, 2.66887e+01, 2.69706e+01, 2.72524e+01, 2.75342e+01, 2.78160e+01, 2.80978e+01};
		}
		else
		{
			std::cout << "# ERROR: wrong tolerance. Choose between double, single or half precision." << std::endl;
			arma::mat Err;
			return Err;
		}

		int pmax = 8;
		int mmax = 55;

		std::random_device rand_dev2;		  // random number generator
		std::mt19937 generator2(rand_dev2()); // random number generator

		double c;
		arma::vec alpha;
		double normH;
		normH = arma::norm(H, 1);

		if (normH <= 4 * theta(mmax - 1) * pmax * (pmax + 3) / (mmax * lengthB))
		{
			c = normH;
			alpha.ones(pmax - 1);
			alpha = c * alpha;
		}
		else
		{
			arma::vec eta;
			eta.zeros(pmax);
			alpha.zeros(pmax - 1);
			// arma::sp_cx_mat X(H.n_rows, H.n_cols);
			// X = H;
			for (int p = 0; p < pmax; p++)
			{
				// X = X * H;
				double c = normAmEst(H, p + 2, generator2);
				// double c = arma::norm(X,1); //normAm of a power, you can also an estimator instead
				c = pow(c, 1 / (p + 2));
				eta(p) = c;
			}

			for (int p = 0; p < pmax - 1; p++)
			{
				arma::vec temp = {eta(p), eta(p + 1)};
				alpha(p) = temp.max();
			}
		}
		arma::mat M(mmax, pmax - 1);
		M.zeros(mmax, pmax - 1);
		for (int p = 2; p < pmax + 1; p++)
		{
			for (int m = p * (p - 1) - 1; m < mmax + 1; m++)
			{
				M(m - 1, p - 2) = alpha(p - 2) / theta(m - 1);
			}
		}

		return M;
	}

	// Get the normalization estimator used in SelectTaylorDegree
	double SpinSpace::normAmEst(const arma::sp_cx_mat &H, double m, std::mt19937 &generator)
	{
		double ans = 0;
		arma::cx_colvec X(H.n_cols);
		std::uniform_real_distribution<double> distr(-1, 1); // Uniform distribution between -1 and 1

		for (int it = 0; it != static_cast<int>(H.n_cols); it++)
		{
			X(it, 0) = arma::cx_double(distr(generator), 0.0);
		}

		X = normalise(X);
		int count = 0;
		for (int it = 0; it != 5; it++)
		{
			arma::cx_colvec temp(H.n_cols);
			temp = X;
			for (int it2 = 0; it2 != m; it2++)
			{
				temp = H * temp;
			}
			X = temp / arma::norm(temp, 1);
			double tempans = arma::norm(temp, 1);
			if (std::abs(tempans - ans) < 1.0e-6)
			{
				ans = tempans;
				count += 1;
				break;
			}
			count += 1;
			ans = tempans;
		}
		ans = ans * count;
		return ans;
	}

	// Returns the action of the matrix exponential of sparse general complex matrix H onto complex column vector b, with krylov subspave dimension of KryDim.
	arma::cx_colvec SpinSpace::KrylovExpmGeneral(const arma::sp_cx_mat &H, const arma::cx_colvec &b, const arma::cx_double dt, int KryDim, int HilbSize)
	{
		// Initialize Krylov basis and upper Hessenberg matrix
		arma::cx_mat Hessen; // Upper Hessenberg matrix
		Hessen.zeros(KryDim, KryDim);

		arma::cx_mat KryBasis(HilbSize, KryDim, arma::fill::zeros); // Orthogonal krylov subspace

		KryBasis.col(0) = b / norm(b);

		double h_mplusone_m;

		// Compute upper Hessenberg matrix and krylov basis using Arnoldi process
		ArnoldiProcess(H, b, KryBasis, Hessen, KryDim, h_mplusone_m);

		arma::cx_colvec e1;
		e1.zeros(KryDim);
		e1(0) = 1;

		// Compute the matrix exponential action
		return norm(b) * KryBasis * arma::expmat(Hessen * dt) * e1;
	}
	// Returns the action of the matrix exponential of sparse symmetric complex matrix H onto complex column vector b, with krylov subspave dimension of KryDim.
	arma::cx_colvec SpinSpace::KrylovExpmSymm(const arma::sp_cx_mat &H, const arma::cx_colvec &b, const arma::cx_double dt, int KryDim, int HilbSize)
	{
		// Initialize Krylov basis and upper Hessenberg matrix
		arma::cx_mat Hessen; // Upper Hessenberg matrix
		Hessen.zeros(KryDim, KryDim);

		arma::cx_mat KryBasis(HilbSize, KryDim, arma::fill::zeros); // Orthogonal krylov subspace

		KryBasis.col(0) = b / norm(b);

		double h_mplusone_m;
		// Compute upper Hessenberg matrix and krylov basis using Lanczos process
		LanczosProcess(H, b, KryBasis, Hessen, KryDim, h_mplusone_m);

		arma::cx_colvec e1;
		e1.zeros(KryDim);
		e1(0) = 1;
		// Compute the matrix exponential action
		return norm(b) * KryBasis * arma::expmat(Hessen * dt) * e1;
	}

	// Compute the Arnoldi process for the given sparse complex general matrix H, complex column vector b, and integer KryDim.
	void SpinSpace::ArnoldiProcess(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_mat &KryBasis, arma::cx_mat &Hessen, int KryDim, double &h_mplusone_m)
	{
		// Perform the Arnoldi process for KryDim iterations
		for (int it1 = 0; it1 < KryDim; it1++)
		{
			// Compute the matrix-vector product.
			arma::cx_colvec z = H * KryBasis.col(it1);
			// Compute the elements of the Hessenberg matrix
			for (int it2 = 0; it2 < it1 + 1; it2++)
			{
				arma::cx_vec temp = KryBasis.col(it2);
				Hessen(it2, it1) = cdot(temp, z);
				z = z - Hessen(it2, it1) * temp;
			}

			// Compute the next column of Krylov basis KryBasis.
			if (KryDim - 1 == it1)
			{
				h_mplusone_m = norm(z);
				break;
			}
			Hessen(it1 + 1, it1) = norm(z);
			if (abs(Hessen(it1 + 1, it1)) < pow(10, -14))
			{
				std::cout << "Stopped in Arnoldi Process" << std::endl;
				break;
			}
			KryBasis.col(it1 + 1) = z / Hessen(it1 + 1, it1);
		}
	}

	// Compute the Lanczos process for the given sparse complex symmetric matrix H, complex column vector b, and integer KryDim.
	void SpinSpace::LanczosProcess(const arma::sp_cx_mat &H, const arma::cx_colvec &b, arma::cx_mat &KryBasis, arma::cx_mat &Hessen, int KryDim, double &h_mplusone_m)
	{
		// Perform the Lanczos process for KryDim iterations.
		for (int it1 = 0; it1 < KryDim; it1++)
		{
			// Compute the matrix-vector product.
			arma::cx_colvec z = H * KryBasis.col(it1);

			// Compute the elements of the Hessenberg matrix.
			for (int it2 = it1 - 1; it2 < it1 + 1; it2++)
			{
				if (it2 == -1)
				{
					continue;
				}
				else
				{
					arma::cx_vec temp = KryBasis.col(it2);
					Hessen(it2, it1) = cdot(temp, z);
					z = z - Hessen(it2, it1) * temp;
				}
			}
			// Compute the next column of Krylov basis KryBasis.
			if (KryDim - 1 == it1)
			{
				h_mplusone_m = norm(z);
				break;
			}
			Hessen(it1 + 1, it1) = norm(z);
			if (abs(Hessen(it1 + 1, it1)) < pow(10, -14))
			{
				break;
			}
			KryBasis.col(it1 + 1) = z / Hessen(it1 + 1, it1);
		}
	}

}
