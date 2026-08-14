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

#include "SpinSpace.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace SpinAPI
{
	bool RotationMatrixToAxisAngle(const arma::mat &_rotation, arma::vec &_axis, double &_angle)
	{
		if (_rotation.n_rows != 3 || _rotation.n_cols != 3)
			return false;

		double qw = 0.0;
		double qx = 0.0;
		double qy = 0.0;
		double qz = 0.0;
		const double trace = _rotation(0, 0) + _rotation(1, 1) + _rotation(2, 2);

		if (trace > 0.0)
		{
			const double s = 2.0 * std::sqrt(std::max(0.0, trace + 1.0));
			if (s == 0.0)
				return false;
			qw = 0.25 * s;
			qx = (_rotation(2, 1) - _rotation(1, 2)) / s;
			qy = (_rotation(0, 2) - _rotation(2, 0)) / s;
			qz = (_rotation(1, 0) - _rotation(0, 1)) / s;
		}
		else if (_rotation(0, 0) > _rotation(1, 1) && _rotation(0, 0) > _rotation(2, 2))
		{
			const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + _rotation(0, 0) - _rotation(1, 1) - _rotation(2, 2)));
			if (s == 0.0)
				return false;
			qw = (_rotation(2, 1) - _rotation(1, 2)) / s;
			qx = 0.25 * s;
			qy = (_rotation(0, 1) + _rotation(1, 0)) / s;
			qz = (_rotation(0, 2) + _rotation(2, 0)) / s;
		}
		else if (_rotation(1, 1) > _rotation(2, 2))
		{
			const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + _rotation(1, 1) - _rotation(0, 0) - _rotation(2, 2)));
			if (s == 0.0)
				return false;
			qw = (_rotation(0, 2) - _rotation(2, 0)) / s;
			qx = (_rotation(0, 1) + _rotation(1, 0)) / s;
			qy = 0.25 * s;
			qz = (_rotation(1, 2) + _rotation(2, 1)) / s;
		}
		else
		{
			const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + _rotation(2, 2) - _rotation(0, 0) - _rotation(1, 1)));
			if (s == 0.0)
				return false;
			qw = (_rotation(1, 0) - _rotation(0, 1)) / s;
			qx = (_rotation(0, 2) + _rotation(2, 0)) / s;
			qy = (_rotation(1, 2) + _rotation(2, 1)) / s;
			qz = 0.25 * s;
		}

		const double qnorm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
		if (qnorm == 0.0)
			return false;

		qw /= qnorm;
		qx /= qnorm;
		qy /= qnorm;
		qz /= qnorm;

		if (qw < 0.0)
		{
			qw = -qw;
			qx = -qx;
			qy = -qy;
			qz = -qz;
		}

		qw = std::max(-1.0, std::min(1.0, qw));
		_angle = 2.0 * std::acos(qw);

		const double sin_half = std::sqrt(std::max(0.0, 1.0 - qw * qw));
		if (sin_half < 1.0e-12 || std::abs(_angle) < 1.0e-12)
		{
			_axis = arma::vec({1.0, 0.0, 0.0});
			_angle = 0.0;
			return true;
		}

		_axis = arma::vec({qx / sin_half, qy / sin_half, qz / sin_half});
		return true;
	}

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

	bool SpinSpace::BuildTraceSamples(const state_ptr &_state,
										 arma::uword _sampleCount,
										 TraceSamplingMethod _method,
										 std::mt19937 &_generator,
										 HilbertTraceSampleSet &_samples,
										 std::string *_error) const
	{
		auto fail = [&](const std::string &_message) {
			_samples = HilbertTraceSampleSet();
			if (_error != nullptr)
				*_error = _message;
			return false;
		};

		if (_state == nullptr)
			return fail("thermal initial states cannot be represented by pure-state trace samples");
		if (_sampleCount == 0)
			return fail("the trace-sample count must be greater than zero");

		arma::cx_mat supportProjector;
		if (!this->GetState(_state, supportProjector))
			return fail("failed to construct the support projector for state \"" + _state->Name() + "\"");

		const arma::uword dimension = this->HilbertSpaceDimensions();
		if (supportProjector.n_rows != dimension || supportProjector.n_cols != dimension)
			return fail("the state support does not match the active Hilbert space");

		// GetState() is pure on explicitly mentioned spins and identity on
		// omitted spins. Its trace is therefore the exact sampled subspace size.
		const arma::cx_double supportTrace = arma::trace(supportProjector);
		const double traceTolerance = 1.0e-10 * std::max(1.0, std::abs(supportTrace));
		if (!std::isfinite(std::real(supportTrace)) || !std::isfinite(std::imag(supportTrace)) ||
			std::abs(std::imag(supportTrace)) > traceTolerance || std::real(supportTrace) <= 0.0)
		{
			return fail("the state support projector has an invalid trace");
		}

		const double roundedTrace = std::round(std::real(supportTrace));
		if (std::abs(std::real(supportTrace) - roundedTrace) > traceTolerance)
			return fail("the state support projector does not have an integer rank");

		_samples.factors.set_size(dimension, _sampleCount);
		_samples.sampledSubspaceDimension = static_cast<arma::uword>(roundedTrace);

		std::normal_distribution<double> gaussian(0.0, 1.0);
		std::uniform_real_distribution<double> uniform(0.0, 1.0);
		const arma::cx_double minusI(0.0, -1.0);
		const double normTolerance = 64.0 * std::numeric_limits<double>::epsilon();

		for (arma::uword sample = 0; sample < _sampleCount; ++sample)
		{
			arma::cx_vec candidate;
			bool accepted = false;
			for (unsigned int attempt = 0; attempt < 16 && !accepted; ++attempt)
			{
				if (_method == TraceSamplingMethod::SUZ)
				{
					candidate.set_size(dimension);
					for (arma::uword index = 0; index < dimension; ++index)
						candidate(index) = arma::cx_double(gaussian(_generator), gaussian(_generator));
				}
				else
				{
					candidate = arma::ones<arma::cx_vec>(1);
					for (const auto &spin : this->spins)
					{
						if (spin == nullptr || spin->Multiplicity() <= 0)
							return fail("the spin space contains an invalid spin");

						const double theta = std::acos(1.0 - 2.0 * uniform(_generator));
						const double phi = 2.0 * arma::datum::pi * uniform(_generator);
						arma::cx_vec local = arma::zeros<arma::cx_vec>(static_cast<arma::uword>(spin->Multiplicity()));
						local(0) = 1.0;
						const arma::cx_mat Sz(spin->Sz());
						const arma::cx_mat Sy(spin->Sy());
						local = arma::expmat(minusI * phi * Sz) *
								arma::expmat(minusI * theta * Sy) * local;
						candidate = arma::kron(candidate, local);
					}
				}

				candidate = supportProjector * candidate;
				const double candidateNorm = arma::norm(candidate, 2);
				if (std::isfinite(candidateNorm) && candidateNorm > normTolerance)
				{
					candidate /= candidateNorm;
					accepted = true;
				}
			}

			if (!accepted)
				return fail("failed to draw a non-zero state from the requested trace-sampling subspace");
			_samples.factors.col(sample) = candidate;
		}

		if (_error != nullptr)
			_error->clear();
		return true;
	}

	bool SpinSpace::FactorizeDensityMatrix(const arma::cx_mat &_density,
										 arma::cx_mat &_factors,
										 std::string *_error,
										 double _tolerance) const
	{
		auto fail = [&](const std::string &_message) {
			_factors.reset();
			if (_error != nullptr)
				*_error = _message;
			return false;
		};

		const arma::uword dimension = this->HilbertSpaceDimensions();
		if (_density.n_rows != dimension || _density.n_cols != dimension)
			return fail("the density matrix does not match the active Hilbert space");
		if (!_density.is_finite())
			return fail("the density matrix contains non-finite values");

		arma::cx_mat normalized = 0.5 * (_density + _density.t());
		const arma::cx_double densityTrace = arma::trace(normalized);
		const double traceScale = std::max(1.0, std::abs(densityTrace));
		if (std::abs(std::imag(densityTrace)) > _tolerance * traceScale ||
			!std::isfinite(std::real(densityTrace)) || std::real(densityTrace) <= 0.0)
		{
			return fail("the density matrix has an invalid trace");
		}
		normalized /= std::real(densityTrace);

		arma::vec eigenvalues;
		arma::cx_mat eigenvectors;
		if (!arma::eig_sym(eigenvalues, eigenvectors, normalized))
			return fail("failed to diagonalize the density matrix");

		const double maxEigenvalue = eigenvalues.is_empty() ? 0.0 : std::abs(eigenvalues.max());
		const double eigenTolerance = std::max(1.0e-14, _tolerance * std::max(1.0, maxEigenvalue));
		if (eigenvalues.is_empty() || eigenvalues.min() < -eigenTolerance)
			return fail("the density matrix is not positive semidefinite");

		const arma::uvec retained = arma::find(eigenvalues > eigenTolerance);
		if (retained.is_empty())
			return fail("the density matrix is numerically rank zero");

		_factors.zeros(dimension, retained.n_elem);
		for (arma::uword column = 0; column < retained.n_elem; ++column)
		{
			const arma::uword index = retained(column);
			_factors.col(column) = std::sqrt(eigenvalues(index)) * eigenvectors.col(index);
		}

		if (_error != nullptr)
			_error->clear();
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

	bool SpinSpace::RotateState(const arma::cx_mat &_state, const arma::mat &_rotation, arma::cx_mat &_out) const
	{
		HilbertStateRotationCache cache;
		if (!this->CreateStateRotationCache(_state, cache))
			return false;
		return this->RotateState(_state, _rotation, cache, _out);
	}

	bool SpinSpace::CreateStateRotationCache(const arma::cx_mat &_state, HilbertStateRotationCache &_cache, double _tolerance) const
	{
		if (_state.n_rows != _state.n_cols ||
			_state.n_rows != this->HilbertSpaceDimensions() ||
			!std::isfinite(_tolerance) ||
			_tolerance < 0.0)
		{
			return false;
		}

		const arma::uword dim = this->HilbertSpaceDimensions();
		_cache.Jx.zeros(dim, dim);
		_cache.Jy.zeros(dim, dim);
		_cache.Jz.zeros(dim, dim);
		for (const auto &spin : this->spins)
		{
			arma::cx_mat Sx;
			arma::cx_mat Sy;
			arma::cx_mat Sz;
			if (!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx) ||
				!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy) ||
				!this->CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz))
			{
				return false;
			}

			_cache.Jx += Sx;
			_cache.Jy += Sy;
			_cache.Jz += Sz;
		}

		// A density matrix is invariant under every global powder rotation iff
		// it commutes with the three generators. This catches singlet projectors,
		// identity factors, and isotropic mixtures without relying on state names.
		const double scale = std::max(1.0, arma::norm(_state, "fro"));
		const double limit = _tolerance * scale;
		_cache.rotationInvariant =
			arma::norm(_cache.Jx * _state - _state * _cache.Jx, "fro") <= limit &&
			arma::norm(_cache.Jy * _state - _state * _cache.Jy, "fro") <= limit &&
			arma::norm(_cache.Jz * _state - _state * _cache.Jz, "fro") <= limit;
		return true;
	}

	bool SpinSpace::RotateState(const arma::cx_mat &_state, const arma::mat &_rotation, const HilbertStateRotationCache &_cache, arma::cx_mat &_out) const
	{
		if (_state.n_rows != _state.n_cols ||
			_state.n_rows != this->HilbertSpaceDimensions() ||
			_cache.Jx.n_rows != _state.n_rows ||
			_cache.Jx.n_cols != _state.n_cols ||
			_cache.Jy.n_rows != _state.n_rows ||
			_cache.Jy.n_cols != _state.n_cols ||
			_cache.Jz.n_rows != _state.n_rows ||
			_cache.Jz.n_cols != _state.n_cols)
		{
			return false;
		}

		if (_cache.rotationInvariant)
		{
			_out = _state;
			return true;
		}

		arma::cx_mat propagator;
		if (!this->CreateStateRotationOperator(_rotation, _cache, propagator))
			return false;

		_out = propagator * _state * propagator.t();
		return true;
	}

	bool SpinSpace::CreateStateRotationOperator(const arma::mat &_rotation, const HilbertStateRotationCache &_cache, arma::cx_mat &_operator) const
	{
		const arma::uword dim = this->HilbertSpaceDimensions();
		if (_cache.Jx.n_rows != dim || _cache.Jx.n_cols != dim ||
			_cache.Jy.n_rows != dim || _cache.Jy.n_cols != dim ||
			_cache.Jz.n_rows != dim || _cache.Jz.n_cols != dim)
		{
			return false;
		}

		// Convert the spatial powder rotation into the corresponding spin
		// rotation using the cached total angular-momentum generators.
		arma::vec axis;
		double angle = 0.0;
		if (!RotationMatrixToAxisAngle(_rotation, axis, angle))
			return false;

		if (std::abs(angle) < 1.0e-12)
		{
			_operator = arma::eye<arma::cx_mat>(dim, dim);
			return true;
		}

		const arma::cx_mat generator = axis(0) * _cache.Jx + axis(1) * _cache.Jy + axis(2) * _cache.Jz;

		const arma::cx_double imaginaryUnit(0.0, 1.0);
		_operator = arma::expmat(-imaginaryUnit * angle * generator);
		return true;
	}

	bool SpinSpace::RotateStateFactors(const arma::cx_mat &_factors, const arma::mat &_rotation, const HilbertStateRotationCache &_cache, arma::cx_mat &_out) const
	{
		if (_factors.n_rows != this->HilbertSpaceDimensions() || _factors.n_cols == 0)
			return false;

		// Do not use rotationInvariant here. Individual Monte-Carlo factors are
		// orientation dependent even when their ensemble density is isotropic.
		arma::cx_mat propagator;
		if (!this->CreateStateRotationOperator(_rotation, _cache, propagator))
			return false;
		_out = propagator * _factors;
		return true;
	}

	bool SpinSpace::PrepareInitialDensityForPowder(const arma::cx_mat &_referenceDensity,
												   const arma::mat &_orientationRotation,
												   StateFrame _stateFrame,
												   bool _discardHamiltonianCoherences,
												   const std::vector<std::string> &_dephasingHamiltonian,
												   const HilbertStateRotationCache *_rotationCache,
												   arma::cx_mat &_orientedDensity)
	{
		// Preserve the historical high-field dephasing path for every existing
		// caller. HSGeneral can use the overload below to request full-Hamiltonian
		// eigenbasis populations explicitly.
		return this->PrepareInitialDensityForPowder(_referenceDensity,
			_orientationRotation, _stateFrame, _discardHamiltonianCoherences,
			_dephasingHamiltonian, HamiltonianApproximation::Secular,
			_rotationCache, _orientedDensity);
	}

	bool SpinSpace::PrepareInitialDensityForPowder(const arma::cx_mat &_referenceDensity,
												   const arma::mat &_orientationRotation,
												   StateFrame _stateFrame,
												   bool _discardHamiltonianCoherences,
												   const std::vector<std::string> &_dephasingHamiltonian,
												   HamiltonianApproximation _dephasingApproximation,
												   const HilbertStateRotationCache *_rotationCache,
												   arma::cx_mat &_orientedDensity)
	{
		_orientedDensity = _referenceDensity;
		const bool previousSuperspaceSetting = this->useSuperspace;
		this->useSuperspace = false;

		if (_stateFrame == StateFrame::Molecular)
		{
			const bool rotated = (_rotationCache != nullptr)
									 ? this->RotateState(_referenceDensity, _orientationRotation, *_rotationCache, _orientedDensity)
									 : this->RotateState(_referenceDensity, _orientationRotation, _orientedDensity);
			if (!rotated)
			{
				this->useSuperspace = previousSuperspaceSetting;
				return false;
			}
		}

		if (_discardHamiltonianCoherences)
		{
			arma::sp_cx_mat H0sp;
			const bool built = _dephasingApproximation == HamiltonianApproximation::Secular
				? this->BaseHamiltonianRotated_SA(_dephasingHamiltonian, _orientationRotation, H0sp)
				: this->BaseHamiltonianRotatedZYZ(_dephasingHamiltonian, _orientationRotation, H0sp);
			if (!built)
			{
				this->useSuperspace = previousSuperspaceSetting;
				return false;
			}

			arma::cx_mat rhoDephased;
			if (!this->DephaseStateInEigenbasis(_orientedDensity, arma::cx_mat(H0sp), rhoDephased))
			{
				this->useSuperspace = previousSuperspaceSetting;
				return false;
			}
			_orientedDensity = std::move(rhoDephased);
		}

		this->useSuperspace = previousSuperspaceSetting;
		return true;
	}

	bool SpinSpace::DephaseStateInEigenbasis(const arma::cx_mat &_state, const arma::cx_mat &_hamiltonian, arma::cx_mat &_out) const
	{
		if (_state.n_rows != _state.n_cols || _state.n_rows != this->HilbertSpaceDimensions())
			return false;
		if (_hamiltonian.n_rows != _hamiltonian.n_cols || _hamiltonian.n_rows != this->HilbertSpaceDimensions())
			return false;

		arma::vec eigenvalues;
		arma::cx_mat eigenvectors;
		if (!arma::eig_sym(eigenvalues, eigenvectors, _hamiltonian))
			return false;

		// The dephasing operation is performed in the eigenbasis of the
		// orientation-specific Hamiltonian. Dropping entries in the lab basis
		// would remove the wrong coherences for anisotropic powder samples.
		arma::cx_mat rhoInEigenbasis = eigenvectors.t() * _state * eigenvectors;
		rhoInEigenbasis %= arma::eye<arma::cx_mat>(rhoInEigenbasis.n_rows, rhoInEigenbasis.n_cols);
		_out = eigenvectors * rhoInEigenbasis * eigenvectors.t();
		return true;
	}

	bool SpinSpace::ThermalStateFromHamiltonian(const arma::cx_mat &_hamiltonian, double _Temperature, arma::cx_mat &_mat) const
	{
		if (_hamiltonian.n_rows != _hamiltonian.n_cols || _hamiltonian.n_rows != this->HilbertSpaceDimensions())
			return false;

		if (_Temperature <= 0.0)
		{
			std::cout << "Failed to obtain thermal state: temperature must be > 0 K." << std::endl;
			return false;
		}

		double Kb = 8.617333262 * std::pow(10, -5);	  // Boltzmann const in eV/K
		double hbar = 6.582119569 * std::pow(10, -16); // Reduced planck constant in eVs/rads
		double beta = hbar / (Kb * _Temperature);
		beta *= std::pow(10, 9);

		arma::cx_mat result = arma::expmat_sym((-beta) * _hamiltonian);
		result /= arma::trace(result);
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

		if (!this->ThermalStateFromHamiltonian(H, _Temperature, _mat))
		{
			_space.UseSuperoperatorSpace(useSuperspaceBeforeThermal);
			return false;
		}

		_space.UseSuperoperatorSpace(useSuperspaceBeforeThermal);

		return true;
	}
}
