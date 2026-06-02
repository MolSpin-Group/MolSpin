/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// This source file contains methods related to relaxation operators,
// which are described by Operator objects.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
namespace SpinAPI
{
	// Phenomenological relaxation in MolSpin superspace ordering. Diagonal
	// density-matrix elements exchange population at rate1, while off-diagonal
	// elements decay independently at rate2. The operator is basis agnostic;
	// callers choose the basis by transforming before or after construction.
	bool PhenomenologicalRelaxationOperator(unsigned int _dimension, double _populationRate, double _coherenceRate, arma::cx_mat &_out)
	{
		if (_dimension == 0 || _populationRate < 0.0 || _coherenceRate < 0.0)
			return false;

		const unsigned int superspaceDimension = _dimension * _dimension;
		_out = arma::zeros<arma::cx_mat>(superspaceDimension, superspaceDimension);

		for (unsigned int row = 0; row < _dimension; ++row)
		{
			for (unsigned int col = 0; col < _dimension; ++col)
			{
				const unsigned int index = row * _dimension + col;
				if (row == col)
				{
					_out(index, index) -= _populationRate * static_cast<double>(_dimension - 1);
					for (unsigned int source = 0; source < _dimension; ++source)
					{
						if (source == row)
							continue;
						const unsigned int sourceIndex = source * _dimension + source;
						_out(index, sourceIndex) += _populationRate;
					}
				}
				else
				{
					_out(index, index) -= _coherenceRate;
				}
			}
		}

		return true;
	}

	// Sparse version of the same rate model. Keeping this separate avoids
	// constructing a dense temporary in larger Hilbert spaces.
	bool PhenomenologicalRelaxationOperator(unsigned int _dimension, double _populationRate, double _coherenceRate, arma::sp_cx_mat &_out)
	{
		if (_dimension == 0 || _populationRate < 0.0 || _coherenceRate < 0.0)
			return false;

		const unsigned int superspaceDimension = _dimension * _dimension;
		_out = arma::sp_cx_mat(superspaceDimension, superspaceDimension);

		for (unsigned int row = 0; row < _dimension; ++row)
		{
			for (unsigned int col = 0; col < _dimension; ++col)
			{
				const unsigned int index = row * _dimension + col;
				if (row == col)
				{
					_out(index, index) -= _populationRate * static_cast<double>(_dimension - 1);
					for (unsigned int source = 0; source < _dimension; ++source)
					{
						if (source == row)
							continue;
						const unsigned int sourceIndex = source * _dimension + source;
						_out(index, sourceIndex) += _populationRate;
					}
				}
				else
				{
					_out(index, index) -= _coherenceRate;
				}
			}
		}

		return true;
	}

	bool TransformSuperoperatorToEigenbasis(const arma::cx_mat &_superoperatorLab, const arma::cx_mat &_eigenvectors, arma::cx_mat &_out)
	{
		if (_eigenvectors.n_rows != _eigenvectors.n_cols)
			return false;
		if (_superoperatorLab.n_rows != _superoperatorLab.n_cols)
			return false;
		if (_superoperatorLab.n_rows != _eigenvectors.n_rows * _eigenvectors.n_rows)
			return false;

		// Same row-major superspace convention as OperatorToSuperspace().
		// The basis change is applied to both sides of the density operator.
		const arma::cx_mat labToEigen = arma::kron(_eigenvectors.t(), _eigenvectors.st());
		const arma::cx_mat eigenToLab = arma::kron(_eigenvectors, arma::conj(_eigenvectors));
		_out = labToEigen * _superoperatorLab * eigenToLab;
		return true;
	}

	bool TransformSuperoperatorFromEigenbasis(const SpinSpace &_space, const arma::cx_mat &_eigenvectors, const arma::sp_cx_mat &_superoperatorEigenbasis, arma::sp_cx_mat &_out)
	{
		// Powder spectra propagate in the lab basis, but relaxation is defined
		// in the orientation-specific Hamiltonian eigenbasis. This converts the
		// relaxation superoperator back without reimplementing the superspace
		// vectorization convention here.
		if (_eigenvectors.n_rows != _eigenvectors.n_cols)
			return false;
		if (_superoperatorEigenbasis.n_rows != _superoperatorEigenbasis.n_cols)
			return false;
		if (_superoperatorEigenbasis.n_rows != _eigenvectors.n_rows * _eigenvectors.n_rows)
			return false;

		arma::cx_mat eigenvectors = _eigenvectors;
		arma::cx_mat eigenvectorsDag = eigenvectors.t();
		arma::cx_mat fromEigenbasis;
		arma::cx_mat toEigenbasis;
		if (!_space.SuperoperatorFromOperators(eigenvectors, eigenvectorsDag, fromEigenbasis) ||
			!_space.SuperoperatorFromOperators(eigenvectorsDag, eigenvectors, toEigenbasis))
		{
			return false;
		}

		_out = arma::sp_cx_mat(fromEigenbasis) * _superoperatorEigenbasis * arma::sp_cx_mat(toEigenbasis);
		return true;
	}

	template <typename MatrixType, typename DaggerExpression>
	bool AddLindbladDissipator(const SpinSpace &_space, const MatrixType &_jump, const DaggerExpression &_jumpDaggerExpression, double _rate, MatrixType &_out)
	{
		if (_rate == 0.0)
			return true;

		// Armadillo represents .t() lazily. Materialising it here keeps one
		// helper usable for both dense and sparse matrix expressions.
		const MatrixType jumpDagger = _jumpDaggerExpression;
		MatrixType PB;
		MatrixType PL;
		MatrixType PR;
		const MatrixType jumpDaggerJump = jumpDagger * _jump;
		if (!_space.SuperoperatorFromOperators(_jump, jumpDagger, PB) ||
			!_space.SuperoperatorFromLeftOperator(jumpDaggerJump, PL) ||
			!_space.SuperoperatorFromRightOperator(jumpDaggerJump, PR))
		{
			return false;
		}

		_out += _rate * (PB - (PL + PR) / 2.0);
		return true;
	}

	bool SpinSpace::CreateSpinOperatorTriplet(const spin_ptr &_spin, arma::cx_mat &_Sx, arma::cx_mat &_Sy, arma::cx_mat &_Sz) const
	{
		return this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin->Sx()), _spin, _Sx) &&
			   this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin->Sy()), _spin, _Sy) &&
			   this->CreateOperator(arma::conv_to<arma::cx_mat>::from(_spin->Sz()), _spin, _Sz);
	}

	bool SpinSpace::CreateSpinOperatorTriplet(const spin_ptr &_spin, arma::sp_cx_mat &_Sx, arma::sp_cx_mat &_Sy, arma::sp_cx_mat &_Sz) const
	{
		return this->CreateOperator(_spin->Sx(), _spin, _Sx) &&
			   this->CreateOperator(_spin->Sy(), _spin, _Sy) &&
			   this->CreateOperator(_spin->Sz(), _spin, _Sz);
	}

	template <typename MatrixType>
	bool SpinSpace::RotateCartesianOperatorTriplet(const arma::mat &_spatialrotation, MatrixType &_Sx, MatrixType &_Sy, MatrixType &_Sz) const
	{
		if (_spatialrotation.n_rows != 3 || _spatialrotation.n_cols != 3)
			return false;

		std::vector<MatrixType> original = {_Sx, _Sy, _Sz};
		std::vector<MatrixType> rotated(3);
		for (unsigned int axis = 0; axis < 3; ++axis)
		{
			rotated[axis].zeros(original[0].n_rows, original[0].n_cols);
			for (unsigned int lab_axis = 0; lab_axis < 3; ++lab_axis)
			{
				rotated[axis] += _spatialrotation(lab_axis, axis) * original[lab_axis];
			}
		}

		_Sx = rotated[0];
		_Sy = rotated[1];
		_Sz = rotated[2];
		return true;
	}

	template <typename MatrixType>
	void SpinSpace::TransformOperatorToBasis(const arma::cx_mat &_basisrotation, MatrixType &_operator) const
	{
		_operator = _basisrotation.t() * _operator * _basisrotation;
	}

	template <typename MatrixType>
	bool SpinSpace::CreateRotatedSpinTripletInBasis(const spin_ptr &_spin, const arma::cx_mat &_basisrotation, const arma::mat &_spatialrotation, MatrixType &_Sx, MatrixType &_Sy, MatrixType &_Sz) const
	{
		if (!this->CreateSpinOperatorTriplet(_spin, _Sx, _Sy, _Sz))
			return false;

		// Spatial rotation is applied before the Hilbert-basis transformation.
		// Callers pass either the current molecular powder rotation or identity,
		// depending on the RelaxationFrame selected by the Operator object.
		if (!this->RotateCartesianOperatorTriplet(_spatialrotation, _Sx, _Sy, _Sz))
			return false;

		this->TransformOperatorToBasis(_basisrotation, _Sx);
		this->TransformOperatorToBasis(_basisrotation, _Sy);
		this->TransformOperatorToBasis(_basisrotation, _Sz);
		return true;
	}

	template <typename MatrixType>
	bool SpinSpace::CreateRotatedSpinPlusMinusInBasis(const spin_ptr &_spin, const arma::cx_mat &_basisrotation, const arma::mat &_spatialrotation, MatrixType &_Splus, MatrixType &_Sminus) const
	{
		MatrixType Sx;
		MatrixType Sy;
		MatrixType Sz;
		if (!this->CreateRotatedSpinTripletInBasis(_spin, _basisrotation, _spatialrotation, Sx, Sy, Sz))
			return false;

		const arma::cx_double im(0.0, 1.0);
		_Splus = Sx + im * Sy;
		_Sminus = Sx - im * Sy;
		return true;
	}

	template <typename MatrixType>
	bool SpinSpace::RelaxationOperatorFrameChangeRotatedInternal(const operator_ptr &_operator, const arma::cx_mat &_basisrotation, const arma::mat &_spatialrotation, MatrixType &_out) const
	{
		// Raw Cartesian channels can follow the molecule. Dedicated Bloch-style
		// T1/T2 channels default to laboratory axes, because experimental T1
		// and T2 are normally defined relative to B0. The explicit frame
		// property allows either convention without duplicating constructors.
		arma::mat channelSpatialRotation;
		if (_operator->Frame() == RelaxationFrame::Molecular)
			channelSpatialRotation = _spatialrotation;
		else
			channelSpatialRotation.eye(3, 3);

		if (_operator->Type() == OperatorType::RelaxationLindblad)
		{
			auto spins = _operator->Spins();

			MatrixType Sx;
			MatrixType Sy;
			MatrixType Sz;
			MatrixType P;
			P.zeros(this->SpaceDimensions(), this->SpaceDimensions());

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				if (!this->CreateRotatedSpinTripletInBasis((*i), _basisrotation, channelSpatialRotation, Sx, Sy, Sz))
					return false;

				if (!AddLindbladDissipator(*this, Sx, Sx.t(), _operator->Rate1(), P) ||
					!AddLindbladDissipator(*this, Sy, Sy.t(), _operator->Rate2(), P) ||
					!AddLindbladDissipator(*this, Sz, Sz.t(), _operator->Rate3(), P))
					return false;
			}

			_out = P;
		}
		else if (_operator->Type() == OperatorType::RelaxationLindbladDoubleSpin)
		{
			auto spins = _operator->Spins();
			std::vector<MatrixType> Sp_operators;
			std::vector<MatrixType> Sm_operators;

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				MatrixType Sptmp;
				MatrixType Smtmp;
				if (!this->CreateRotatedSpinPlusMinusInBasis((*i), _basisrotation, channelSpatialRotation, Sptmp, Smtmp))
					return false;

				Sp_operators.push_back(Sptmp);
				Sm_operators.push_back(Smtmp);
			}

			MatrixType P;
			MatrixType PB;
			MatrixType PL;
			MatrixType PR;
			P.zeros(this->SpaceDimensions(), this->SpaceDimensions());

			for (size_t it1 = 0; it1 < Sp_operators.size(); ++it1)
			{
				for (size_t it2 = 0; it2 < Sp_operators.size(); ++it2)
				{
					if (it1 == it2)
						continue;

					MatrixType L_minus = Sm_operators[it1] * Sp_operators[it2];
					MatrixType L_plus = Sp_operators[it1] * Sm_operators[it2];

					if (!this->SuperoperatorFromOperators(L_minus, L_minus.t(), PB) || !this->SuperoperatorFromLeftOperator(L_minus.t() * L_minus, PL) || !this->SuperoperatorFromRightOperator(L_minus.t() * L_minus, PR))
						return false;
					P += (PB - (PL + PR) / 2.0) * _operator->Rate1();

					if (!this->SuperoperatorFromOperators(L_plus, L_plus.t(), PB) || !this->SuperoperatorFromLeftOperator(L_plus.t() * L_plus, PL) || !this->SuperoperatorFromRightOperator(L_plus.t() * L_plus, PR))
						return false;
					P += (PB - (PL + PR) / 2.0) * _operator->Rate1();
				}
			}

			_out = P;
		}
		else if (_operator->Type() == OperatorType::RelaxationDephasing)
		{
			auto spins = _operator->Spins();

			std::vector<MatrixType> Sx_operators;
			std::vector<MatrixType> Sy_operators;
			std::vector<MatrixType> Sz_operators;

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				MatrixType Sxtmp;
				MatrixType Sytmp;
				MatrixType Sztmp;
				if (!this->CreateRotatedSpinTripletInBasis((*i), _basisrotation, channelSpatialRotation, Sxtmp, Sytmp, Sztmp))
					return false;

				Sx_operators.push_back(Sxtmp);
				Sy_operators.push_back(Sytmp);
				Sz_operators.push_back(Sztmp);
			}

			if (Sx_operators.size() < 2)
				return false;

			MatrixType E;
			E.eye(Sx_operators[0].n_rows, Sx_operators[0].n_cols);
			MatrixType Psinglet = (1.0 / 4.0) * E - (Sx_operators[0] * Sx_operators[1] + Sy_operators[0] * Sy_operators[1] + Sz_operators[0] * Sz_operators[1]);
			MatrixType Ptriplet = E - Psinglet;

			MatrixType PLsinglet;
			MatrixType PRsinglet;
			MatrixType PLtriplet;
			MatrixType PRtriplet;
			if (!this->SuperoperatorFromLeftOperator(Psinglet, PLsinglet) || !this->SuperoperatorFromLeftOperator(Ptriplet, PLtriplet) || !this->SuperoperatorFromRightOperator(Psinglet, PRsinglet) || !this->SuperoperatorFromRightOperator(Ptriplet, PRtriplet))
				return false;

			_out = -1.0 * _operator->Rate1() * (PLsinglet * PRtriplet + PLtriplet * PRsinglet);
		}
		else if (_operator->Type() == OperatorType::RelaxationRandomFields)
		{
			auto spins = _operator->Spins();

			MatrixType Sx;
			MatrixType Sy;
			MatrixType Sz;
			MatrixType P;
			P.zeros(this->SpaceDimensions(), this->SpaceDimensions());

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				if (!this->CreateRotatedSpinTripletInBasis((*i), _basisrotation, channelSpatialRotation, Sx, Sy, Sz))
					return false;

				// Kattnig et al. (2016), DOI: 10.1088/1367-2630/18/6/063007,
				// Appendix C reduces to one Lindblad
				// dissipator per Cartesian random-field component. This form
				// is trace preserving for every spin multiplicity and remains
				// valid when anisotropic component rates are requested.
				if (!AddLindbladDissipator(*this, Sx, Sx.t(), _operator->Rate1(), P) ||
					!AddLindbladDissipator(*this, Sy, Sy.t(), _operator->Rate2(), P) ||
					!AddLindbladDissipator(*this, Sz, Sz.t(), _operator->Rate3(), P))
					return false;
			}

			_out = P;
		}
		else if (_operator->Type() == OperatorType::RelaxationT1)
		{
			auto spins = _operator->Spins();

			MatrixType P;
			MatrixType S_plus;
			MatrixType S_minus;
			P.zeros(this->SpaceDimensions(), this->SpaceDimensions());

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				if (!this->CreateRotatedSpinPlusMinusInBasis((*i), _basisrotation, channelSpatialRotation, S_plus, S_minus))
					return false;

				const double channelRate = 0.5 * _operator->Rate1();
				if (!AddLindbladDissipator(*this, S_plus, S_minus, channelRate, P) ||
					!AddLindbladDissipator(*this, S_minus, S_plus, channelRate, P))
				{
					return false;
				}
			}

			_out = P;
		}
		else if (_operator->Type() == OperatorType::RelaxationT2)
		{
			auto spins = _operator->Spins();

			MatrixType Sx;
			MatrixType Sy;
			MatrixType Sz;
			MatrixType P;
			P.zeros(this->SpaceDimensions(), this->SpaceDimensions());

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				if (!this->CreateRotatedSpinTripletInBasis((*i), _basisrotation, channelSpatialRotation, Sx, Sy, Sz))
					return false;

				// For spin 1/2, D[Sz] damps rho_up,down at half its
				// coefficient. The user-facing rate is the pure-dephasing
				// rate 1/Tphi, so the Lindblad coefficient must be 2/Tphi.
				// For higher spins the resulting Delta-m squared dependence
				// is the expected microscopic Sz-dephasing behaviour.
				if (!AddLindbladDissipator(*this, Sz, Sz.t(), 2.0 * _operator->Rate1(), P))
					return false;
			}

			_out = P;
		}
		else if (_operator->Type() == OperatorType::RelaxationPhenomenological)
		{
			if (!PhenomenologicalRelaxationOperator(this->HilbertSpaceDimensions(), _operator->Rate1(), _operator->Rate2(), _out))
				return false;
		}
		else if (_operator->Type() == OperatorType::Unspecified)
		{
			return false;
		}
		else
		{
			std::cout << "Cannot construct relaxation operator for Operator " << _operator->Name() << "!" << std::endl;
			return false;
		}

		return true;
	}

	// -----------------------------------------------------
	// Relaxation operators
	// -----------------------------------------------------
	// Dense and sparse superspace callers share the same implementation as
	// powder calculations. Identity rotations recover the non-powder case.
	bool SpinSpace::RelaxationOperator(const operator_ptr &_operator, arma::cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid() || !this->useSuperspace)
			return false;

		arma::cx_mat basisRotation;
		basisRotation.eye(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
		arma::mat spatialRotation;
		spatialRotation.eye(3, 3);
		return this->RelaxationOperatorFrameChangeRotatedInternal(_operator, basisRotation, spatialRotation, _out);
	}

	bool SpinSpace::RelaxationOperator(const operator_ptr &_operator, arma::sp_cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid() || !this->useSuperspace)
			return false;

		arma::cx_mat basisRotation;
		basisRotation.eye(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
		arma::mat spatialRotation;
		spatialRotation.eye(3, 3);
		return this->RelaxationOperatorFrameChangeRotatedInternal(_operator, basisRotation, spatialRotation, _out);
	}

	// Hilbert-space relaxation cache builder. Powder tasks pass the current
	// molecular orientation, but only molecule-fixed channels consume it.
	// Lab-frame T1/T2 channels intentionally remain aligned with B0.
	bool SpinSpace::RelaxationOperatorHilbertInternal(const operator_ptr &_operator, const arma::mat *_spatialrotation, HilbertRelaxationCache &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid())
			return false;

		if (this->useSuperspace)
			return false;

		auto add_lindblad = [&](const arma::sp_cx_mat &L, const arma::sp_cx_mat &R, double rate) {
			if (rate == 0.0)
				return false;
			HilbertRelaxationTerm term;
			term.L = L;
			term.R = R;
			term.M = R * L;
			term.rate = rate;
			_out.lindblad_terms.push_back(term);
			return true;
		};
		auto create_triplet = [&](const spin_ptr &spin, arma::sp_cx_mat &Sx, arma::sp_cx_mat &Sy, arma::sp_cx_mat &Sz) {
			if (!this->CreateSpinOperatorTriplet(spin, Sx, Sy, Sz))
				return false;
			// Non-powder callers and lab-frame channels retain Cartesian lab
			// axes. Molecule-fixed powder channels rotate the complete triplet
			// before Lindblad terms are assembled from it.
			return _spatialrotation == nullptr ||
				   _operator->Frame() == RelaxationFrame::Lab ||
				   this->RotateCartesianOperatorTriplet(*_spatialrotation, Sx, Sy, Sz);
		};
		auto create_plus_minus = [&](const spin_ptr &spin, arma::sp_cx_mat &Sp, arma::sp_cx_mat &Sm) {
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;
			if (!create_triplet(spin, Sx, Sy, Sz))
				return false;
			const arma::cx_double imag(0.0, 1.0);
			Sp = Sx + imag * Sy;
			Sm = Sx - imag * Sy;
			return true;
		};

		bool added = false;

		if (_operator->Type() == OperatorType::RelaxationLindblad)
		{
			auto spins = _operator->Spins();
			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				arma::sp_cx_mat Sx;
				arma::sp_cx_mat Sy;
				arma::sp_cx_mat Sz;
				if (!create_triplet(*i, Sx, Sy, Sz))
				{
					continue;
				}

				added = add_lindblad(Sx, Sx.t(), _operator->Rate1()) || added;
				added = add_lindblad(Sy, Sy.t(), _operator->Rate2()) || added;
				added = add_lindblad(Sz, Sz.t(), _operator->Rate3()) || added;
			}
		}
		else if (_operator->Type() == OperatorType::RelaxationLindbladDoubleSpin)
		{
			auto spins = _operator->Spins();
			std::vector<arma::sp_cx_mat> Sp_operators(spins.size());
			std::vector<arma::sp_cx_mat> Sm_operators(spins.size());
			std::vector<bool> valid(spins.size(), false);

			for (size_t idx = 0; idx < spins.size(); ++idx)
			{
				if (!this->Contains(spins[idx]))
					continue;

				arma::sp_cx_mat Sp;
				arma::sp_cx_mat Sm;
				if (!create_plus_minus(spins[idx], Sp, Sm))
				{
					continue;
				}

				Sp_operators[idx] = Sp;
				Sm_operators[idx] = Sm;
				valid[idx] = true;
			}

			for (size_t i = 0; i < spins.size(); ++i)
			{
				if (!valid[i])
					continue;

				for (size_t j = 0; j < spins.size(); ++j)
				{
					if (i == j || !valid[j])
						continue;

					arma::sp_cx_mat L_minus = Sm_operators[i] * Sp_operators[j];
					arma::sp_cx_mat L_plus = Sp_operators[i] * Sm_operators[j];

					added = add_lindblad(L_minus, L_minus.t(), _operator->Rate1()) || added;
					added = add_lindblad(L_plus, L_plus.t(), _operator->Rate1()) || added;
				}
			}
		}
		else if (_operator->Type() == OperatorType::RelaxationDephasing)
		{
			auto spins = _operator->Spins();
			std::vector<arma::sp_cx_mat> Sx_operators;
			std::vector<arma::sp_cx_mat> Sy_operators;
			std::vector<arma::sp_cx_mat> Sz_operators;

			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				arma::sp_cx_mat Sx;
				arma::sp_cx_mat Sy;
				arma::sp_cx_mat Sz;
				if (!create_triplet(*i, Sx, Sy, Sz))
				{
					continue;
				}

				Sx_operators.push_back(Sx);
				Sy_operators.push_back(Sy);
				Sz_operators.push_back(Sz);
			}

			if (Sx_operators.size() >= 2 && _operator->Rate1() != 0.0)
			{
				arma::sp_cx_mat E = arma::speye<arma::sp_cx_mat>(Sx_operators[0].n_rows, Sx_operators[0].n_cols);
				arma::sp_cx_mat Psinglet = (1.0 / 4.0) * E -
										   (Sx_operators[0] * Sx_operators[1] + Sy_operators[0] * Sy_operators[1] + Sz_operators[0] * Sz_operators[1]);
				arma::sp_cx_mat Ptriplet = E - Psinglet;

				HilbertRelaxationDephasingTerm term;
				term.Psinglet = Psinglet;
				term.Ptriplet = Ptriplet;
				term.Psinglet_t = Psinglet.t();
				term.Ptriplet_t = Ptriplet.t();
				term.Psinglet_dag = Psinglet.st();
				term.Ptriplet_dag = Ptriplet.st();
				term.rate = _operator->Rate1();
				_out.dephasing_terms.push_back(term);
				added = true;
			}
		}
		else if (_operator->Type() == OperatorType::RelaxationRandomFields)
		{
			auto spins = _operator->Spins();
			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				arma::sp_cx_mat Sx;
				arma::sp_cx_mat Sy;
				arma::sp_cx_mat Sz;
				if (!create_triplet(*i, Sx, Sy, Sz))
				{
					continue;
				}

				// Random fields are ordinary Lindblad channels. Reusing the
				// common cache representation keeps Hilbert and superspace
				// propagation identical and guarantees trace preservation.
				added = add_lindblad(Sx, Sx.t(), _operator->Rate1()) || added;
				added = add_lindblad(Sy, Sy.t(), _operator->Rate2()) || added;
				added = add_lindblad(Sz, Sz.t(), _operator->Rate3()) || added;
			}
		}
		else if (_operator->Type() == OperatorType::RelaxationPhenomenological)
		{
			if (_operator->Rate1() < 0.0 || _operator->Rate2() < 0.0)
				return false;

			if (_operator->Rate1() != 0.0 || _operator->Rate2() != 0.0)
			{
				// Store rates only. The actual matrix size is fixed by the
				// SpinSpace used during Hilbert-space propagation.
				HilbertRelaxationPhenomenologicalTerm term;
				term.populationRate = _operator->Rate1();
				term.coherenceRate = _operator->Rate2();
				_out.phenomenological_terms.push_back(term);
				added = true;
			}
		}
		else if (_operator->Type() == OperatorType::RelaxationT1)
		{
			auto spins = _operator->Spins();
			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				arma::sp_cx_mat Sp;
				arma::sp_cx_mat Sm;
				if (!create_plus_minus(*i, Sp, Sm))
				{
					continue;
				}

				const double channelRate = 0.5 * _operator->Rate1();
				added = add_lindblad(Sp, Sm, channelRate) || added;
				added = add_lindblad(Sm, Sp, channelRate) || added;
			}
		}
		else if (_operator->Type() == OperatorType::RelaxationT2)
		{
			auto spins = _operator->Spins();
			for (auto i = spins.cbegin(); i != spins.cend(); i++)
			{
				if (!this->Contains(*i))
					continue;

				arma::sp_cx_mat Sx;
				arma::sp_cx_mat Sy;
				arma::sp_cx_mat Sz;
				if (!create_triplet(*i, Sx, Sy, Sz))
				{
					continue;
				}

				// User-facing rate is the pure-dephasing rate 1/Tphi.
				added = add_lindblad(Sz, Sz.t(), 2.0 * _operator->Rate1()) || added;
			}
		}

		return added;
	}

	bool SpinSpace::RelaxationOperator(const operator_ptr &_operator, HilbertRelaxationCache &_out) const
	{
		return this->RelaxationOperatorHilbertInternal(_operator, nullptr, _out);
	}

	bool SpinSpace::PowderRelaxationOperatorHilbert(const operator_ptr &_operator, const arma::mat &_spatialrotation, HilbertRelaxationCache &_out) const
	{
		if (_spatialrotation.n_rows != 3 || _spatialrotation.n_cols != 3)
			return false;
		return this->RelaxationOperatorHilbertInternal(_operator, &_spatialrotation, _out);
	}

	bool SpinSpace::ApplyRelaxationHilbert(const HilbertRelaxationCache &_cache, const arma::cx_mat &_rho, arma::cx_mat &_out) const
	{
		_out.zeros(_rho.n_rows, _rho.n_cols);

		for (const auto &term : _cache.lindblad_terms)
		{
			if (term.rate == 0.0)
				continue;

			arma::cx_mat PB = term.L * _rho * term.R;
			arma::cx_mat PL = term.M * _rho;
			arma::cx_mat PR = _rho * term.M;
			_out += term.rate * (PB - 0.5 * (PL + PR));
		}

		for (const auto &term : _cache.dephasing_terms)
		{
			if (term.rate == 0.0)
				continue;

			_out += -term.rate * (term.Ptriplet_t * _rho * term.Psinglet_dag + term.Psinglet_t * _rho * term.Ptriplet_dag);
		}

		for (const auto &term : _cache.phenomenological_terms)
		{
			const arma::uword dim = _rho.n_rows;
			if (_rho.n_cols != dim)
				continue;

			if (term.populationRate != 0.0)
			{
				for (arma::uword row = 0; row < dim; ++row)
				{
					_out(row, row) -= term.populationRate * static_cast<double>(dim - 1) * _rho(row, row);
					for (arma::uword source = 0; source < dim; ++source)
					{
						if (source == row)
							continue;
						_out(row, row) += term.populationRate * _rho(source, source);
					}
				}
			}

			if (term.coherenceRate != 0.0)
			{
				for (arma::uword row = 0; row < dim; ++row)
				{
					for (arma::uword col = 0; col < dim; ++col)
					{
						if (row != col)
							_out(row, col) -= term.coherenceRate * _rho(row, col);
					}
				}
			}
		}

		return true;
	}

	bool SpinSpace::ApplyPhenomenologicalRelaxationHilbert(const std::vector<HilbertRelaxationPhenomenologicalTerm> &_terms, const arma::cx_mat &_basisEigenvectors, const arma::cx_mat &_rho, arma::cx_mat &_out) const
	{
		if (_rho.n_rows != _rho.n_cols ||
			_basisEigenvectors.n_rows != _rho.n_rows ||
			_basisEigenvectors.n_cols != _rho.n_cols)
		{
			return false;
		}

		_out.zeros(_rho.n_rows, _rho.n_cols);
		if (_terms.empty())
			return true;

		// In the selected Hamiltonian eigenbasis, rate1 exchanges every
		// population with every other population while preserving trace.
		// rate2 damps only off-diagonal coherences. The caller uses this
		// derivative form when explicit relaxation channels require RK steps.
		const arma::uword dim = _rho.n_rows;
		const arma::cx_mat rho_basis = _basisEigenvectors.t() * _rho * _basisEigenvectors;
		arma::cx_mat relax_basis(dim, dim, arma::fill::zeros);

		for (const auto &term : _terms)
		{
			if (term.populationRate != 0.0)
			{
				for (arma::uword row = 0; row < dim; ++row)
				{
					relax_basis(row, row) -= term.populationRate * static_cast<double>(dim - 1) * rho_basis(row, row);
					for (arma::uword source = 0; source < dim; ++source)
					{
						if (source != row)
							relax_basis(row, row) += term.populationRate * rho_basis(source, source);
					}
				}
			}

			if (term.coherenceRate != 0.0)
			{
				for (arma::uword row = 0; row < dim; ++row)
				{
					for (arma::uword col = 0; col < dim; ++col)
					{
						if (row != col)
							relax_basis(row, col) -= term.coherenceRate * rho_basis(row, col);
					}
				}
			}
		}

		_out = _basisEigenvectors * relax_basis * _basisEigenvectors.t();
		return true;
	}

	bool SpinSpace::CreatePhenomenologicalRelaxationMapHilbert(const std::vector<HilbertRelaxationPhenomenologicalTerm> &_terms, const arma::cx_mat &_basisEigenvectors, double _timestep, HilbertPhenomenologicalRelaxationMap &_out) const
	{
		const arma::uword dim = _basisEigenvectors.n_rows;
		if (dim == 0 || _basisEigenvectors.n_cols != dim ||
			!std::isfinite(_timestep) || _timestep < 0.0)
		{
			return false;
		}

		double populationRate = 0.0;
		double coherenceRate = 0.0;
		for (const auto &term : _terms)
		{
			if (!std::isfinite(term.populationRate) || term.populationRate < 0.0 ||
				!std::isfinite(term.coherenceRate) || term.coherenceRate < 0.0)
			{
				return false;
			}
			populationRate += term.populationRate;
			coherenceRate += term.coherenceRate;
		}

		// All phenomenological generators commute in the same eigenbasis, so
		// their rates can be summed before exponentiation. For N populations,
		// deviations from trace(rho)/N decay as exp(-N * rate1 * dt);
		// off-diagonal entries decay as exp(-rate2 * dt).
		_out.basisToLab = _basisEigenvectors;
		_out.labToBasis = _basisEigenvectors.t();
		_out.populationDecay = std::exp(-static_cast<double>(dim) * populationRate * _timestep);
		_out.coherenceDecay = std::exp(-coherenceRate * _timestep);
		return true;
	}

	bool SpinSpace::ApplyPhenomenologicalRelaxationMapHilbert(const HilbertPhenomenologicalRelaxationMap &_map, arma::cx_mat &_rho, arma::cx_mat &_workspace) const
	{
		const arma::uword dim = _rho.n_rows;
		if (dim == 0 || _rho.n_cols != dim ||
			_map.basisToLab.n_rows != dim || _map.basisToLab.n_cols != dim ||
			_map.labToBasis.n_rows != dim || _map.labToBasis.n_cols != dim ||
			!std::isfinite(_map.populationDecay) || _map.populationDecay < 0.0 ||
			!std::isfinite(_map.coherenceDecay) || _map.coherenceDecay < 0.0)
		{
			return false;
		}

		// Move into the orientation-specific H0 eigenbasis, apply the exact
		// finite-step population and coherence decay, then return to the lab
		// propagation basis. The caller supplies workspace to avoid allocations
		// inside long powder/time loops.
		_workspace = _map.labToBasis * _rho;
		_rho = _workspace * _map.basisToLab;
		if (!this->ApplyPhenomenologicalRelaxationMapInBasisHilbert(_map, _rho))
			return false;
		_workspace = _map.basisToLab * _rho;
		_rho = _workspace * _map.labToBasis;
		return true;
	}

	bool SpinSpace::ApplyPhenomenologicalRelaxationMapInBasisHilbert(const HilbertPhenomenologicalRelaxationMap &_map, arma::cx_mat &_rho) const
	{
		const arma::uword dim = _rho.n_rows;
		if (dim == 0 || _rho.n_cols != dim ||
			!std::isfinite(_map.populationDecay) || _map.populationDecay < 0.0 ||
			!std::isfinite(_map.coherenceDecay) || _map.coherenceDecay < 0.0)
		{
			return false;
		}

		// When the caller already propagates in the selected H0 eigenbasis,
		// phenomenological relaxation is an element-wise operation. Avoiding a
		// lab-to-basis round trip for every time step is essential for large
		// powder grids.
		const arma::cx_double equilibriumPopulation = arma::trace(_rho) / static_cast<double>(dim);
		for (arma::uword row = 0; row < dim; ++row)
		{
			_rho(row, row) = equilibriumPopulation + (_rho(row, row) - equilibriumPopulation) * _map.populationDecay;
			for (arma::uword col = 0; col < dim; ++col)
			{
				if (row != col)
					_rho(row, col) *= _map.coherenceDecay;
			}
		}
		return true;
	}

	bool SpinSpace::RelaxationSuperoperatorHilbert(const HilbertRelaxationCache &_cache, arma::cx_mat &_out) const
	{
		int dim = static_cast<int>(this->HilbertSpaceDimensions());
		if (dim <= 0)
			return false;

		arma::cx_mat Iden = arma::eye<arma::cx_mat>(dim, dim);
		_out.zeros(dim * dim, dim * dim);

		for (const auto &term : _cache.lindblad_terms)
		{
			if (term.rate == 0.0)
				continue;

			arma::sp_cx_mat PB;
			arma::sp_cx_mat PL;
			arma::sp_cx_mat PR;
			if (!this->SuperoperatorFromOperators(term.L, term.R, PB) ||
				!this->SuperoperatorFromLeftOperator(term.M, PL) ||
				!this->SuperoperatorFromRightOperator(term.M, PR))
			{
				return false;
			}

			_out += term.rate * arma::cx_mat(PB - 0.5 * (PL + PR));
		}

		for (const auto &term : _cache.dephasing_terms)
		{
			if (term.rate == 0.0)
				continue;

			arma::cx_mat Ps = arma::cx_mat(term.Psinglet);
			arma::cx_mat Pt = arma::cx_mat(term.Ptriplet);
			arma::cx_mat Ps_conj = arma::conj(Ps);
			arma::cx_mat Pt_conj = arma::conj(Pt);

			_out += -term.rate * (arma::kron(Ps_conj, Pt.t()) + arma::kron(Pt_conj, Ps.t()));
		}

		for (const auto &term : _cache.phenomenological_terms)
		{
			arma::cx_mat P;
			if (PhenomenologicalRelaxationOperator(static_cast<unsigned int>(dim), term.populationRate, term.coherenceRate, P))
			{
				_out += P;
			}
		}

		return true;
	}

	bool SpinSpace::PhenomenologicalRelaxationSuperoperatorHilbert(const std::vector<HilbertRelaxationPhenomenologicalTerm> &_terms, const arma::cx_mat &_basisEigenvectors, arma::cx_mat &_out) const
	{
		const arma::uword dim = _basisEigenvectors.n_rows;
		if (dim == 0 || _basisEigenvectors.n_cols != dim)
			return false;

		if (_terms.empty())
		{
			_out.reset();
			return true;
		}

		// timeinf solves require a Liouvillian rather than a finite-step map.
		// Build it in the same H0 eigenbasis used by timeevo and transform it
		// back to lab superspace ordering before returning it.
		arma::cx_mat canonical(dim * dim, dim * dim, arma::fill::zeros);
		for (const auto &term : _terms)
		{
			arma::cx_mat contribution;
			if (!PhenomenologicalRelaxationOperator(static_cast<unsigned int>(dim), term.populationRate, term.coherenceRate, contribution))
				return false;
			canonical += contribution;
		}

		arma::cx_mat basisToLab;
		arma::cx_mat labToBasis;
		if (!this->SuperoperatorFromOperators(_basisEigenvectors, _basisEigenvectors.t(), basisToLab) ||
			!this->SuperoperatorFromOperators(_basisEigenvectors.t(), _basisEigenvectors, labToBasis))
		{
			return false;
		}

		_out = basisToLab * canonical * labToBasis;
		return true;
	}

	// --------------------------------------------------------------
	// Relaxation operators when a unitary transformation is required
	// --------------------------------------------------------------

	bool SpinSpace::RelaxationOperatorFrameChangeRotated(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::mat _spatialrotation, arma::cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid())
			return false;

		if (!this->useSuperspace)
			return false;

		if (_rotationmatrix.n_rows != this->HilbertSpaceDimensions() || _rotationmatrix.n_cols != this->HilbertSpaceDimensions())
			return false;

		return this->RelaxationOperatorFrameChangeRotatedInternal(_operator, _rotationmatrix, _spatialrotation, _out);
	}

	bool SpinSpace::RelaxationOperatorFrameChangeRotated(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::mat _spatialrotation, arma::sp_cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid())
			return false;

		if (!this->useSuperspace)
			return false;

		if (_rotationmatrix.n_rows != this->HilbertSpaceDimensions() || _rotationmatrix.n_cols != this->HilbertSpaceDimensions())
			return false;

		return this->RelaxationOperatorFrameChangeRotatedInternal(_operator, _rotationmatrix, _spatialrotation, _out);
	}

	bool SpinSpace::PowderRelaxationOperatorEigenbasis(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid())
			return false;

		if (!this->useSuperspace)
			return false;

		if (_eigenvectors.n_rows != this->HilbertSpaceDimensions() || _eigenvectors.n_cols != this->HilbertSpaceDimensions())
			return false;

		// Phenomenological relaxation is already defined directly in the
		// caller-supplied Hamiltonian eigenbasis. Spin-operator relaxation is
		// represented in that basis after applying the powder rotation only
		// when the operator explicitly uses molecule-fixed axes.
		if (_operator->Type() == OperatorType::RelaxationPhenomenological)
			return this->RelaxationOperatorFrameChange(_operator, _eigenvectors, _out);

		return this->RelaxationOperatorFrameChangeRotated(_operator, _eigenvectors, _spatialrotation, _out);
	}

	bool SpinSpace::PowderRelaxationOperatorEigenbasis(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::sp_cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid())
			return false;

		if (!this->useSuperspace)
			return false;

		if (_eigenvectors.n_rows != this->HilbertSpaceDimensions() || _eigenvectors.n_cols != this->HilbertSpaceDimensions())
			return false;

		if (_operator->Type() == OperatorType::RelaxationPhenomenological)
			return this->RelaxationOperatorFrameChange(_operator, _eigenvectors, _out);

		return this->RelaxationOperatorFrameChangeRotated(_operator, _eigenvectors, _spatialrotation, _out);
	}

	bool SpinSpace::PowderRelaxationOperator(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::cx_mat &_out) const
	{
		arma::sp_cx_mat sparse_out;
		if (!this->PowderRelaxationOperator(_operator, _eigenvectors, _spatialrotation, sparse_out))
			return false;

		_out = arma::cx_mat(sparse_out);
		return true;
	}

	bool SpinSpace::PowderRelaxationOperator(const operator_ptr &_operator, arma::cx_mat _eigenvectors, arma::mat _spatialrotation, arma::sp_cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid())
			return false;

		if (!this->useSuperspace)
			return false;

		if (_eigenvectors.n_rows != this->HilbertSpaceDimensions() || _eigenvectors.n_cols != this->HilbertSpaceDimensions())
			return false;

		arma::sp_cx_mat eigenbasisRelaxation;
		if (!this->PowderRelaxationOperatorEigenbasis(_operator, _eigenvectors, _spatialrotation, eigenbasisRelaxation))
			return false;

		return TransformSuperoperatorFromEigenbasis(*this, _eigenvectors, eigenbasisRelaxation, _out);
	}

	bool SpinSpace::RelaxationOperatorFrameChange(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid() || !this->useSuperspace)
			return false;
		if (_rotationmatrix.n_rows != this->HilbertSpaceDimensions() || _rotationmatrix.n_cols != this->HilbertSpaceDimensions())
			return false;

		arma::mat spatialRotation;
		spatialRotation.eye(3, 3);
		return this->RelaxationOperatorFrameChangeRotatedInternal(_operator, _rotationmatrix, spatialRotation, _out);
	}

	bool SpinSpace::RelaxationOperatorFrameChange(const operator_ptr &_operator, arma::cx_mat _rotationmatrix, arma::sp_cx_mat &_out) const
	{
		if (_operator == nullptr || !_operator->IsValid() || !this->useSuperspace)
			return false;
		if (_rotationmatrix.n_rows != this->HilbertSpaceDimensions() || _rotationmatrix.n_cols != this->HilbertSpaceDimensions())
			return false;

		arma::mat spatialRotation;
		spatialRotation.eye(3, 3);
		return this->RelaxationOperatorFrameChangeRotatedInternal(_operator, _rotationmatrix, spatialRotation, _out);
	}

	bool SpinSpace::RelaxationOperatorFrameChange(const operator_ptr &_operator, arma::sp_cx_mat _rotationmatrix, arma::sp_cx_mat &_out) const
	{
		return this->RelaxationOperatorFrameChange(_operator, arma::cx_mat(_rotationmatrix), _out);
	}
}
