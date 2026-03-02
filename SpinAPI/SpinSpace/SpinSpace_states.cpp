/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// This source file generates matrix and vector representations of state
// objects.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2019 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////

namespace SpinAPI
{
	// -----------------------------------------------------
	// Spin state representations in the space
	// -----------------------------------------------------
	// Returns a projection operator onto the state of the single spin with the given value of the "mz" quantum number
	arma::cx_mat SpinSpace::GetSingleSpinState(const spin_ptr &_spin, int _mz) const
	{
		arma::cx_mat temp;
		arma::cx_mat result;
		bool isFirst = true;

		// Loop through all spins in the space
		for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
		{
			// Check whether we found the spin whose state we are interested in
			if ((*i) == _spin)
			{
				// Get the index along the diagonal that should be set to 1 as the only non-zero element in the matrix
				// Note the division by 2 since both S and _mz is in units of 1/2
				auto index = _spin->Multiplicity() - (_spin->S() + _mz) / 2 - 1;
				temp.zeros((*i)->Multiplicity(), (*i)->Multiplicity());
				temp(index, index) = 1.0;
			}
			else
			{
				// Use an identity for all the other spins
				temp.eye((*i)->Multiplicity(), (*i)->Multiplicity());
			}

			// Update the result-matrix
			if (isFirst)
			{
				result = temp;
				isFirst = false;
			}
			else
			{
				auto tmp = kron(result, temp);
				result = tmp;
			}
		}

		return result;
	}

	// Sets the vector to a representation of the state within the given spin space
	// Returns false if the given state entangles spins within the spin space with spins not contained in the spin space
	bool SpinSpace::GetState(const CompleteState &_cstate, arma::cx_vec &_out, bool _useFullBasis) const
	{
		// Get dimensions of the vector to return
		unsigned int dimensions = 1;
		if (_useFullBasis)
			dimensions = this->HilbertSpaceDimensions();
		else
			for (auto i = _cstate.cbegin(); i != _cstate.cend(); i++)
				dimensions *= static_cast<unsigned int>(i->first->Multiplicity());

		// Helper vectors
		arma::cx_vec tmpvec;
		arma::cx_vec nextvec;
		arma::cx_vec resultvec;
		arma::cx_vec sumresultvec(dimensions);
		double factor_sqsum = 0.0; // Norm square of the CompleteState
		unsigned int index = 0;

		// Get the norm square of the CompleteState

		while (index < _cstate[0].second.size())
		{
			auto factor = _cstate[0].second[index++].second;
			factor_sqsum += std::abs(factor) * std::abs(factor);
		}

		// Don't forget to reset index for the next loop
		index = 0;

		// Loop through all contributions
		while (index < _cstate[0].second.size())
		{
			// Get the normalization factor
			auto factor = _cstate[0].second[index].second / std::sqrt(factor_sqsum);

			// Set the "nextvec" to a scalar "1"
			nextvec = arma::ones<arma::cx_vec>(1);

			// Loop through the spins in the SpinSpace
			for (auto i = this->spins.cbegin(); i != this->spins.cend(); i++)
			{
				// Check whether the spin is in the CompleteState
				auto j = _cstate.cbegin();
				while (j != _cstate.cend() && j->first != (*i))
				{
					++j;
				}

				// Get a vector representation of the single-spin state in the vector space of the spin
				if (j != _cstate.cend())
				{
					// Put "1/N^2" for the state with the correct "mz", and "0" for all other indices
					// "N^2" is the norm square of the CompleteState
					tmpvec = arma::zeros<arma::cx_vec>(static_cast<unsigned int>((*i)->Multiplicity()));
					auto mz = j->second[index].first;
					auto vec_index = (*i)->Multiplicity() - ((*i)->S() + mz) / 2 - 1;
					if (vec_index >= 0 || static_cast<unsigned int>(vec_index) < tmpvec.n_elem)
						tmpvec[vec_index] = 1.0;
				}
				else
				{
					// Check whether we should skip spins outside the CompleteState
					if (!_useFullBasis)
						continue;

					// If the spin was not part of the CompleteState, fill the vector space with ones instead
					tmpvec = arma::ones<arma::cx_vec>(static_cast<unsigned int>((*i)->Multiplicity())) / factor; // / (double)(*i)->Multiplicity();
				}

				// Combine individual spin spaces with Kronecker products
				resultvec = kron(nextvec, tmpvec);
				nextvec = resultvec;
			}

			// Add resultvec
			sumresultvec += resultvec * factor;

			++index;
		}

		_out = sumresultvec;
		return true;
	}

	// Sets the vector to a representation of the state within the given subspace of the whole spin system
	bool SpinSpace::GetStateSubSpace(const state_ptr &_state, arma::cx_vec &_out) const
	{
		// Make sure that the state can be described on the spin space (i.e. not entangled with spins outside the space)
		if (!_state->IsComplete(this->spins))
			return false;

		// Helper variables
		auto spinlist = this->spins; // A list of spins that have yet to be checked for state information
		CompleteState cstate;
		arma::cx_vec result = arma::ones<arma::cx_vec>(1);
		arma::cx_vec cstate_vec;
		arma::cx_vec tmpvec;
		spin_ptr tmpspin;

		// We need to obtain the basis used for creating the state, such that it can be reordered later
		std::vector<spin_ptr> basis;
		basis.reserve(spinlist.size());

		while (spinlist.size() > 0)
		{
			// Get the spin at the end of the list, and remove it from the list
			tmpspin = *(spinlist.begin());
			spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), tmpspin), spinlist.end());

			// Put the spin into the basis vector which tracks the ordering of the spins
			basis.push_back(tmpspin);

			// Check whether the State object has any information about the spin
			// Also puts a CompleteState object into cstate if it returns true
			if (_state->GetCompleteState(tmpspin, cstate))
			{
				// Remove the spins in the CompleteState from the spin list, as they are all handled here
				for (auto i = cstate.cbegin(); i != cstate.cend(); i++)
				{
					spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), i->first), spinlist.end());

					// Also, construct the basis order of the spins
					if (i->first != tmpspin)
						basis.push_back(i->first);
				}

				// Get the state vector for the subset of spins (the CompleteState)
				if (!this->GetState(cstate, cstate_vec, false))
					return false;

				// Expand the vector using the direct product of subspaces
				tmpvec = kron(result, cstate_vec);
				result = tmpvec;
			}
			else
			{
				continue;
			}
		}

		// Reorder the spins in the basis
		// this->ReorderBasis(result, basis);

		_out = result;

		return true;
	}

	// Sets the vector to a representation of the state within the given spin space
	// Returns false if the given state entangles spins within the spin space with spins not contained in the spin space
	bool SpinSpace::GetState(const state_ptr &_state, arma::cx_vec &_out) const
	{
		// Make sure that the state can be described on the spin space (i.e. not entangled with spins outside the space)
		if (!_state->IsComplete(this->spins))
			return false;

		// Helper variables
		auto spinlist = this->spins; // A list of spins that have yet to be checked for state information
		CompleteState cstate;
		arma::cx_vec result = arma::ones<arma::cx_vec>(1);
		arma::cx_vec cstate_vec;
		arma::cx_vec tmpvec;
		spin_ptr tmpspin;

		// We need to obtain the basis used for creating the state, such that it can be reordered later
		std::vector<spin_ptr> basis;
		basis.reserve(spinlist.size());

		while (spinlist.size() > 0)
		{
			// Get the spin at the end of the list, and remove it from the list
			tmpspin = *(spinlist.begin());
			spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), tmpspin), spinlist.end());

			// Put the spin into the basis vector which tracks the ordering of the spins
			basis.push_back(tmpspin);

			// Check whether the State object has any information about the spin
			// Also puts a CompleteState object into cstate if it returns true
			if (_state->GetCompleteState(tmpspin, cstate))
			{
				// Remove the spins in the CompleteState from the spin list, as they are all handled here
				for (auto i = cstate.cbegin(); i != cstate.cend(); i++)
				{
					spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), i->first), spinlist.end());

					// Also, construct the basis order of the spins
					if (i->first != tmpspin)
						basis.push_back(i->first);
				}

				// Get the state vector for the subset of spins (the CompleteState)
				if (!this->GetState(cstate, cstate_vec, false))
					return false;
			}
			else
			{
				// Choose a spin state for the vector - a linear combination of all mz values is chosen, and normalized
				cstate_vec = arma::ones<arma::cx_vec>(static_cast<unsigned int>(tmpspin->Multiplicity())) / std::sqrt(static_cast<double>(tmpspin->Multiplicity()));
			}

			// Expand the vector using the direct product of subspaces
			tmpvec = kron(result, cstate_vec);
			result = tmpvec;
		}

		// Reorder the spins in the basis
		this->ReorderBasis(result, basis);

		_out = result;

		return true;
	}

	// Sets the (dense) matrix to a projection operator onto the state within the given spin space
	// Returns false if the given state entangles spins within the spin space with spins not contained in the spin space
	bool SpinSpace::GetState(const state_ptr &_state, arma::cx_mat &_mat) const
	{
		// Make sure that the state can be described on the spin space (i.e. not entangled with spins outside the space)
		if (!_state->IsComplete(this->spins))
			return false;

		// Helper variables
		auto spinlist = this->spins; // A list of spins that have yet to be checked for state information
		CompleteState cstate;
		arma::cx_mat result = arma::ones<arma::cx_mat>(1, 1);
		arma::cx_vec cstate_vec;
		arma::cx_mat cstate_proj;
		arma::cx_mat prevresult;
		spin_ptr tmpspin;

		// We need to obtain the basis used for creating the state, such that it can be reordered later
		std::vector<spin_ptr> basis;
		basis.reserve(spinlist.size());

		while (spinlist.size() > 0)
		{
			// Get the spin at the end of the list, and remove it from the list
			tmpspin = *(spinlist.begin());
			spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), tmpspin), spinlist.end());

			// Put the spin into the basis vector which tracks the ordering of the spins
			basis.push_back(tmpspin);

			// Make sure that the spin points to something
			if (tmpspin == nullptr)
			{
				continue;
			}

			// Check whether the State object has any information about the spin
			// Also puts a CompleteState object into cstate if it returns true
			if (_state->GetCompleteState(tmpspin, cstate))
			{
				// Remove the spins in the CompleteState from the spin list, as they are all handled here
				for (auto i = cstate.cbegin(); i != cstate.cend(); i++)
				{
					spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), i->first), spinlist.end());

					// Also, construct the basis order of the spins
					if (i->first != tmpspin)
						basis.push_back(i->first);
				}

				// Get the state vector
				if (!this->GetState(cstate, cstate_vec, false))
					return false;

				// Create projection matrix
				cstate_proj = cstate_vec * cstate_vec.t();
			}
			else
			{
				unsigned int M = static_cast<unsigned int>(tmpspin->Multiplicity());
				cstate_proj = arma::eye<arma::cx_mat>(M, M);
			}

			prevresult = result;
			result = kron(prevresult, cstate_proj);
		}

		// Reorder the spins in the basis
		this->ReorderBasis(result, basis);

		_mat = result;

		return true;
	}

	// Sets the (sparse) matrix to a projection operator onto the state within the given spin space
	// Returns false if the given state entangles spins within the spin space with spins not contained in the spin space
	bool SpinSpace::GetState(const state_ptr &_state, arma::sp_cx_mat &_mat) const
	{
		// Make sure that the state can be described on the spin space (i.e. not entangled with spins outside the space)
		if (!_state->IsComplete(this->spins))
			return false;

		// Helper variables
		auto spinlist = this->spins; // A list of spins that have yet to be checked for state information
		CompleteState cstate;
		arma::sp_cx_mat result = arma::conv_to<arma::sp_cx_mat>::from(arma::ones<arma::cx_mat>(1, 1));
		arma::cx_vec cstate_vec;
		arma::sp_cx_mat cstate_proj;
		arma::sp_cx_mat prevresult;
		spin_ptr tmpspin;

		// We need to obtain the basis used for creating the state, such that it can be reordered later
		std::vector<spin_ptr> basis;
		basis.reserve(spinlist.size());

		while (spinlist.size() > 0)
		{
			// Get the spin at the end of the list, and remove it from the list
			tmpspin = *(spinlist.begin());
			spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), tmpspin), spinlist.end());

			// Put the spin into the basis vector which tracks the ordering of the spins
			basis.push_back(tmpspin);

			// Make sure that the spin points to something
			if (tmpspin == nullptr)
			{
				continue;
			}

			// Check whether the State object has any information about the spin
			// Also puts a CompleteState object into cstate if it returns true
			if (_state->GetCompleteState(tmpspin, cstate))
			{
				// Remove the spins in the CompleteState from the spin list, as they are all handled here
				for (auto i = cstate.cbegin(); i != cstate.cend(); i++)
				{
					spinlist.erase(std::remove(spinlist.begin(), spinlist.end(), i->first), spinlist.end());

					// Also, construct the basis order of the spins
					if (i->first != tmpspin)
						basis.push_back(i->first);
				}

				// Get the state vector
				if (!this->GetState(cstate, cstate_vec, false))
					return false;

				// Create projection matrix
				cstate_proj = arma::conv_to<arma::sp_cx_mat>::from(cstate_vec * cstate_vec.t());
			}
			else
			{
				unsigned int M = static_cast<unsigned int>(tmpspin->Multiplicity());
				cstate_proj = arma::eye<arma::sp_cx_mat>(M, M);
			}

			prevresult = result;
			result = kron(prevresult, cstate_proj);
		}

		// Reorder the spins in the basis
		this->ReorderBasis(result, basis);

		_mat = result;

		return true;
	}

	// Produces the thermal state
	bool SpinSpace::GetThermalState(SpinAPI::SpinSpace &_space, double _Temperature, std::vector<std::string> thermalhamiltonian_list, arma::cx_mat &_mat) const
	{
		const bool useSuperspaceBeforeThermal = _space.useSuperspace;
		_space.UseSuperoperatorSpace(false);

		arma::cx_mat H;

		if (!_space.ThermalHamiltonian(thermalhamiltonian_list, H))
		{
			std::cout << "Failed to obtain Static Hamiltonian in superspace." << std::endl;
			_space.UseSuperoperatorSpace(useSuperspaceBeforeThermal);
			return false;
		}

		if (_Temperature <= 0.0)
		{
			std::cout << "Failed to obtain thermal state: temperature must be > 0 K." << std::endl;
			_space.UseSuperoperatorSpace(useSuperspaceBeforeThermal);
			return false;
		}

		// Helper variables
		arma::cx_mat result = arma::ones<arma::cx_mat>(1, 1);
		double Kb = 8.617333262 * std::pow(10, -5);	   // Boltzmann const in eV/K
		double hbar = 6.582119569 * std::pow(10, -16); // Reduced planck constant in eVs/rads
		double beta = hbar / (Kb * _Temperature);
		beta *= std::pow(10, 9);

		// Calculate thermal states here
		result = (-beta) * H;
		result = arma::expmat_sym(result);
		result /= arma::trace(result);
		_mat = result;

		_space.UseSuperoperatorSpace(useSuperspaceBeforeThermal);

		return true;
	}
}
