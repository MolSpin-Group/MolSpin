#include "SpinSpace.h"
/////////////////////////////////////////////////////////////////////////
// SpinSpace class (SpinAPI Module)
// ------------------
// This source files contains methods to create matrix representations of
// interactions, and thus to generate Hamiltonians.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <exception>
namespace SpinAPI
{
	// Interaction "orientation" values are stored as passive ZYZ Euler angles.
	// This helper returns the tensor-frame rotation used before any powder
	// rotation is applied by the calling task.
	arma::mat EulerZYZPassiveFrame(const arma::vec &_angles)
	{
		const double alpha = (_angles.n_elem >= 1) ? _angles(0) : 0.0;
		const double beta = (_angles.n_elem >= 2) ? _angles(1) : 0.0;
		const double gamma = (_angles.n_elem >= 3) ? _angles(2) : 0.0;

		const double ca = std::cos(alpha), sa = std::sin(alpha);
		const double cb = std::cos(beta), sb = std::sin(beta);
		const double cg = std::cos(gamma), sg = std::sin(gamma);

		arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
		arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
		arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
		return Rg * Rb * Ra;
	}

	// Build the zero-field-splitting tensor used in
	// H_ZFS = S^T D S. With energyshift=true this is the conventional
	// D(Sz^2 - S(S+1)/3) + E(Sx^2 - Sy^2) form.
	arma::mat ZfsTensorFromParameters(double _D, double _E, bool _subtractIsotropicShift)
	{
		arma::mat zfsTensor = arma::zeros<arma::mat>(3, 3);
		if (_subtractIsotropicShift)
		{
			zfsTensor(0, 0) = -_D / 3.0 + _E;
			zfsTensor(1, 1) = -_D / 3.0 - _E;
			zfsTensor(2, 2) = 2.0 * _D / 3.0;
		}
		else
		{
			zfsTensor(0, 0) = _E;
			zfsTensor(1, 1) = -_E;
			zfsTensor(2, 2) = _D;
		}
		return zfsTensor;
	}

	// Move an interaction tensor from its own principal-axis frame into the
	// molecular frame. Powder rotations, when present, are applied afterwards.
	arma::mat ApplyInteractionFrame(const arma::mat &_tensor, const interaction_ptr &_interaction)
	{
		const arma::mat frameToLab = EulerZYZPassiveFrame(_interaction->Framelist());
		const arma::mat frameToMolecule = frameToLab.t();
		return frameToMolecule * _tensor * frameToMolecule.t();
	}

	template <typename MatrixT>
	void AddQuadraticSpinTensor(MatrixT &_out,
								const MatrixT &_Sx,
								const MatrixT &_Sy,
								const MatrixT &_Sz,
								const arma::mat &_tensor)
	{
		// Full quadratic form: sum_ab tensor_ab S_a S_b. This is used for
		// quadratic-spin interactions and for the non-secular ZFS Hamiltonian.
		_out += _Sx * _Sx * _tensor(0, 0) + _Sx * _Sy * _tensor(0, 1) + _Sx * _Sz * _tensor(0, 2);
		_out += _Sy * _Sx * _tensor(1, 0) + _Sy * _Sy * _tensor(1, 1) + _Sy * _Sz * _tensor(1, 2);
		_out += _Sz * _Sx * _tensor(2, 0) + _Sz * _Sy * _tensor(2, 1) + _Sz * _Sz * _tensor(2, 2);
	}

	template <typename MatrixT>
	void AddSecularQuadraticSpinTensor(MatrixT &_out,
									   const MatrixT &_Sz,
									   double _s,
									   const arma::mat &_tensor,
									   bool _subtractIsotropicShift)
	{
		// Secular limit around the current z axis. Only the difference between
		// the z component and the transverse average changes transition
		// frequencies; the optional shift keeps the usual ZFS trace convention.
		const double transverseMean = 0.5 * (_tensor(0, 0) + _tensor(1, 1));
		const double coefficient = _tensor(2, 2) - transverseMean;
		MatrixT term = _Sz * _Sz;
		if (_subtractIsotropicShift)
		{
			const double shift = _s * (_s + 1.0) / 3.0;
			term -= shift * arma::speye<MatrixT>(_Sz.n_rows, _Sz.n_cols);
		}
		_out += coefficient * term;
	}

	// -----------------------------------------------------
	// Hamiltonian representations in the space
	// -----------------------------------------------------
	// Returns the matrix representation of the Interaction object on the spin space
	bool SpinSpace::InteractionOperator(const interaction_ptr &_interaction, arma::cx_mat &_out) const
	{
		// Make sure the interaction is valid
		if (_interaction == nullptr)
			return false;

		// Create temporary matrix to hold the result
		arma::cx_mat tmp = arma::zeros<arma::cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

		// Get the interaction tensor
		auto ATensor = _interaction->CouplingTensor();

		if (_interaction->Type() == InteractionType::SingleSpin)
		{
			// Get the field at the current time or trajectory step
			arma::vec field;
			field = _interaction->Field();

			// Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			arma::cx_mat Sx;
			arma::cx_mat Sy;
			arma::cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), Sx);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), Sy);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), Sz);

				arma::mat G = arma::eye<arma::mat>(3, 3);
				if (!_interaction->IgnoreTensors())
				{
					G = arma::conv_to<arma::mat>::from((*i)->GetTensor().LabFrame());
				}
				G = ApplyInteractionFrame(G, _interaction);

				tmp += Sx * (field(0) * G(0, 0) + field(1) * G(0, 1) + field(2) * G(0, 2));
				tmp += Sy * (field(0) * G(1, 0) + field(1) * G(1, 1) + field(2) * G(1, 2));
				tmp += Sz * (field(0) * G(2, 0) + field(1) * G(2, 1) + field(2) * G(2, 2));
			}
		}
		else if (_interaction->Type() == InteractionType::DoubleSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::cx_mat S1x;
			arma::cx_mat S1y;
			arma::cx_mat S1z;
			arma::cx_mat S2x;
			arma::cx_mat S2y;
			arma::cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), S2z);
					}
					else
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Ty()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tz()), (*j), S2z);
					}

					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						arma::mat A = ApplyInteractionFrame(ATensor->LabFrame(), _interaction);

						tmp += S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2);
						tmp += S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2);
						tmp += S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2);
						}
						else
						{
							// If there is no interaction tensor or if the tensor is isotropic, just take the dot product
							tmp += S1x * S2x + S1y * S2y + S1z * S2z;
						}
					}
				}
			}
			else if (_interaction->Type() == InteractionType::QuadraticSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			arma::cx_mat S1x;
			arma::cx_mat S1y;
			arma::cx_mat S1z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				// Quadratic spin tensors are interaction tensors, not spin g-tensors.
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S1x);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S1y);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S1z);

				if (ATensor != nullptr && !IsIsotropic(*ATensor))
				{
					arma::mat A = ApplyInteractionFrame(ATensor->LabFrame(), _interaction);
					AddQuadraticSpinTensor(tmp, S1x, S1y, S1z, A);
				}
				else
				{
					// If there is no interaction tensor or if the tensor is isotropic, just take the dot product
					tmp += S1x * S1x + S1y * S1y + S1z * S1z;
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Zfs)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			arma::cx_mat Sx;
			arma::cx_mat Sy;
			arma::cx_mat Sz;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), Sx);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), Sy);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), Sz);

				arma::mat Dtensor = ZfsTensorFromParameters(_interaction->Dvalue(), _interaction->Evalue(), _interaction->ES());
				Dtensor = ApplyInteractionFrame(Dtensor, _interaction);
				AddQuadraticSpinTensor(tmp, Sx, Sy, Sz, Dtensor);
			}
		}
		else if (_interaction->Type() == InteractionType::Exchange)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::cx_mat S1x;
			arma::cx_mat S1y;
			arma::cx_mat S1z;
			arma::cx_mat S2x;
			arma::cx_mat S2y;
			arma::cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), S2z);
					}
					else
					{
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Ty()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Tz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Ty()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tz()), (*j), S2z);
					}
					// TODO: This is not correct when the user uses a non isotropic exchange which he will not when being smart.
					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();
						tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
						tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
						tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));

						arma::cx_mat half = 0.5 * arma::eye<arma::cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp += half;
					}
					else
					{
						// TODO: Be careful with the sign here!!! Usually you would use a minus sign to at the whole interaction.
						//  If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);
						arma::cx_mat half = 0.5 * arma::eye<arma::cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp += half;
					}
					}
				}
			}
			else if (_interaction->Type() == InteractionType::SemiClassicalField)
		{
			//  Grab orientation parameters
			int n = _interaction->Orientations();
			tmp = arma::zeros<arma::cx_mat>(this->HilbertSpaceDimensions(), n * this->HilbertSpaceDimensions());

			InternalCreateSCCompositeMatrix(_interaction, n, tmp);
		}
		else
		{
			// The interaction type was not recognized
			return false;
		}

		// Multiply by the isotropic value, if the tensor was isotropic
		if (ATensor != nullptr && IsIsotropic(*ATensor))
			tmp *= ATensor->Isotropic();

		// Multiply with the given prefactor (or 1.0 if none was specified)
		tmp *= _interaction->Prefactor();

		// Multiply by common prefactor (bohr magneton / hbar)
		if (_interaction->AddCommonPrefactor())
			tmp *= 8.79410005e+1;

		// Check whether we want a superspace or Hilbert space result
		if (this->useSuperspace)
		{
			arma::cx_mat lhs;
			arma::cx_mat rhs;
			int submats = tmp.n_cols / this->HilbertSpaceDimensions();
			int superoperatorspace = this->HilbertSpaceDimensions() * this->HilbertSpaceDimensions();
			_out = arma::zeros<arma::cx_mat>(superoperatorspace, submats * superoperatorspace);
			for (int i = 0; i < submats; i++)
			{
				auto tmpmat = tmp.submat(0, i * this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions() - 1, (i + 1) * this->HilbertSpaceDimensions() - 1);
				auto result = this->SuperoperatorFromLeftOperator(tmpmat, lhs);
				result &= this->SuperoperatorFromRightOperator(tmpmat, rhs);
				if (result)
				{
					_out.submat(0, i * superoperatorspace, superoperatorspace - 1, (i + 1) * superoperatorspace - 1) = lhs - rhs;
				}
				else
					return false;
			}
		}
		else
		{
			// We already have the result in the Hilbert space
			_out = tmp;
		}

		return true;
	}

	// Returns the matrix representation of the Interaction object on the spins space (sparse matrix version)
	bool SpinSpace::InteractionOperator(const interaction_ptr &_interaction, arma::sp_cx_mat &_out) const
	{
		// Make sure the interaction is valid
		if (_interaction == nullptr)
			return false;

		// Create temporary matrix to hold the result
		arma::sp_cx_mat tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

		// Get the interaction tensor
		auto ATensor = _interaction->CouplingTensor();

		if (_interaction->Type() == InteractionType::SingleSpin)
		{
			// Get the field at the current time or trajectory step
			arma::vec field;
			field = _interaction->Field();

			// Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				this->CreateOperator((*i)->Sx(), (*i), Sx);
				this->CreateOperator((*i)->Sy(), (*i), Sy);
				this->CreateOperator((*i)->Sz(), (*i), Sz);

				arma::mat G = arma::eye<arma::mat>(3, 3);
				if (!_interaction->IgnoreTensors())
				{
					G = arma::conv_to<arma::mat>::from((*i)->GetTensor().LabFrame());
				}
				G = ApplyInteractionFrame(G, _interaction);

				tmp += Sx * (field(0) * G(0, 0) + field(1) * G(0, 1) + field(2) * G(0, 2));
				tmp += Sy * (field(0) * G(1, 0) + field(1) * G(1, 1) + field(2) * G(1, 2));
				tmp += Sz * (field(0) * G(2, 0) + field(1) * G(2, 1) + field(2) * G(2, 2));
			}
		}
		else if (_interaction->Type() == InteractionType::DoubleSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::sp_cx_mat S1x;
			arma::sp_cx_mat S1y;
			arma::sp_cx_mat S1z;
			arma::sp_cx_mat S2x;
			arma::sp_cx_mat S2y;
			arma::sp_cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{

						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{

						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

						if (ATensor != nullptr && !IsIsotropic(*ATensor))
						{
							// Use the tensor to calculate the product S_1 * A * S_2
							arma::mat A = ApplyInteractionFrame(ATensor->LabFrame(), _interaction);
							tmp += S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2);
							tmp += S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2);
							tmp += S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2);
					}
					else
					{
						// If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += S1x * S2x + S1y * S2y + S1z * S2z;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::QuadraticSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			arma::sp_cx_mat S1x;
			arma::sp_cx_mat S1y;
			arma::sp_cx_mat S1z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				// Quadratic spin tensors are interaction tensors, not spin g-tensors.
				this->CreateOperator((*i)->Sx(), (*i), S1x);
				this->CreateOperator((*i)->Sy(), (*i), S1y);
				this->CreateOperator((*i)->Sz(), (*i), S1z);

				if (ATensor != nullptr && !IsIsotropic(*ATensor))
				{
					arma::mat A = ApplyInteractionFrame(ATensor->LabFrame(), _interaction);
					AddQuadraticSpinTensor(tmp, S1x, S1y, S1z, A);

					// std::cout << "ZFS Hamiltonian part matrix (before multiplying prefactor and commonprefactor):" << std::endl;
					// std::cout << tmp << std::endl;
				}
				else
				{
					// If there is no interaction tensor or if the tensor is isotropic, just take the dot product
					tmp += S1x * S1x + S1y * S1y + S1z * S1z;
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Exchange)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::sp_cx_mat S1x;
			arma::sp_cx_mat S1y;
			arma::sp_cx_mat S1z;
			arma::sp_cx_mat S2x;
			arma::sp_cx_mat S2y;
			arma::sp_cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{

						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{

						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

					// TODO: This is not correct when the user uses a non isotropic exchange which he will not when being smart.
					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();
						tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
						tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
						tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));

						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp(arma::span::all, arma::span::all) += half;
					}
					else
					{
						// TODO: Be carfeul with the sign here!!! Usually you would use a minus sign to at the whol interaction.
						//  If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);

						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp(arma::span::all, arma::span::all) += half;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Zfs)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			// Build Sx, Sy, Sz for *each* electron in Group1
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				this->CreateOperator((*i)->Sx(), (*i), Sx);
				this->CreateOperator((*i)->Sy(), (*i), Sy);
				this->CreateOperator((*i)->Sz(), (*i), Sz);

				arma::mat Dtensor = ZfsTensorFromParameters(_interaction->Dvalue(), _interaction->Evalue(), _interaction->ES());
				Dtensor = ApplyInteractionFrame(Dtensor, _interaction);
				AddQuadraticSpinTensor(tmp, Sx, Sy, Sz, Dtensor);
			}
		}
		else if (_interaction->Type() == InteractionType::SemiClassicalField)
		{
			//  Grab orientation parameters
			int n = _interaction->Orientations();
			tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), n * this->HilbertSpaceDimensions());

			InternalCreateSCCompositeMatrix(_interaction, n, tmp);
		}
		else
		{
			// The interaction type was not recognized
			return false;
		}

		// Multiply by the isotropic value, if the tensor was isotropic
		if (ATensor != nullptr && IsIsotropic(*ATensor))
			tmp *= ATensor->Isotropic();

		// Multiply with the given prefactor (or 1.0 if none was specified)
		tmp *= _interaction->Prefactor();

		// Multiply by common prefactor. TODO: Consider which system of units to use
		if (_interaction->AddCommonPrefactor())
			tmp *= 8.79410005e+1;

		// Check whether we want a superspace or Hilbert space result
		if (this->useSuperspace)
		{
			arma::sp_cx_mat lhs;
			arma::sp_cx_mat rhs;
			int submats = tmp.n_cols / this->HilbertSpaceDimensions();
			int superoperatorspace = this->HilbertSpaceDimensions() * this->HilbertSpaceDimensions();
			_out = arma::zeros<arma::sp_cx_mat>(superoperatorspace, submats * superoperatorspace);
			for (int i = 0; i < submats; i++)
			{
				auto tmpmat = tmp.submat(0, i * this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions() - 1, (i + 1) * this->HilbertSpaceDimensions() - 1);
				auto result = this->SuperoperatorFromLeftOperator(tmpmat, lhs);
				result &= this->SuperoperatorFromRightOperator(tmpmat, rhs);
				if (result)
				{
					_out.submat(0, i * superoperatorspace, superoperatorspace - 1, (i + 1) * superoperatorspace - 1) = lhs - rhs;
				}
				else
					return false;
			}
		}
		else
		{
			// We already have the result in the dense matrix to the Hamiltonian at the given time or trajectorye Hilbert space
			_out = tmp;
		}
		return true;
	}

	// Returns the matrix representation of the Interaction object on the spins space (sparse matrix version). Singlespin and double spin interaction are rotated with interactionframe and additionally to the point on the sphere. ZFS and Semiclassical field are currently not modified in any way
	bool SpinSpace::InteractionOperatorRotatedZYZ(const interaction_ptr &_interaction, arma::mat &_rotationmatrix, arma::sp_cx_mat &_out) const
	{
		// Make sure the interaction is valid
		if (_interaction == nullptr)
			return false;

		// Create temporary matrix to hold the result
		arma::sp_cx_mat tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

		// Get the interaction tensor
		auto ATensor = _interaction->CouplingTensor();

		// The supplied powder-rotation matrix follows the same passive
		// molecular-to-lab convention as the powder tasks. The interaction
		// frame is applied first; the powder orientation is applied second.
		const arma::mat Rpowder = _rotationmatrix;

		auto RotateTensorFrameAndPowder = [&](const arma::mat &A) -> arma::mat
		{
			arma::mat Af = ApplyInteractionFrame(A, _interaction);
			arma::mat Al = Rpowder * Af * Rpowder.t();
			if (!this->useFullTensorRotation)
			{
				// Historical spectra used only the diagonal tensor components
				// after rotation. Keep that path available for old inputs.
				Al = Al % arma::eye<arma::mat>(3, 3);
			}
			return Al;
		};

		if (_interaction->Type() == InteractionType::SingleSpin)
		{
			// Get the field (lab frame)
			arma::vec field = _interaction->Field();

			// Obtain the list of spins interacting with the field
			auto spinlist = _interaction->Group1();
			arma::sp_cx_mat Sx, Sy, Sz;

			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				this->CreateOperator((*i)->Sx(), (*i), Sx);
				this->CreateOperator((*i)->Sy(), (*i), Sy);
				this->CreateOperator((*i)->Sz(), (*i), Sz);

				// Spin tensor (typically g-tensor for electron Zeeman)
				arma::mat G = arma::conv_to<arma::mat>::from((*i)->GetTensor().LabFrame());
				if (_interaction->IgnoreTensors())
				{
					// match InteractionOperator(): ignore spin tensors completely
					G = arma::eye<arma::mat>(3, 3);
				}

				// Apply interaction frame and powder orientation
				G = RotateTensorFrameAndPowder(G);

				// H = S · (G · B)
				tmp += Sx * (field(0) * G(0, 0) + field(1) * G(0, 1) + field(2) * G(0, 2));
				tmp += Sy * (field(0) * G(1, 0) + field(1) * G(1, 1) + field(2) * G(1, 2));
				tmp += Sz * (field(0) * G(2, 0) + field(1) * G(2, 1) + field(2) * G(2, 2));
			}
		}
		else if (_interaction->Type() == InteractionType::DoubleSpin)
		{
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();

			arma::sp_cx_mat S1x, S1y, S1z;
			arma::sp_cx_mat S2x, S2y, S2z;

			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Match InteractionOperator(): use S operators or T operators depending on IgnoreTensors
					if (_interaction->IgnoreTensors())
					{
						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{
						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
						A = RotateTensorFrameAndPowder(A);

						// S1 · A · S2
						tmp += S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2);
						tmp += S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2);
						tmp += S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2);
					}
					else
					{
						// isotropic dot product
						tmp += S1x * S2x + S1y * S2y + S1z * S2z;
					}
				}
			}
			}
			else if (_interaction->Type() == InteractionType::QuadraticSpin)
			{
				auto spins1 = _interaction->Group1();
				arma::sp_cx_mat Sx, Sy, Sz;

				for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);

					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
						A = RotateTensorFrameAndPowder(A);
						AddQuadraticSpinTensor(tmp, Sx, Sy, Sz, A);
					}
					else
					{
						tmp += Sx * Sx + Sy * Sy + Sz * Sz;
					}
				}
			}
			else if (_interaction->Type() == InteractionType::Exchange)
			{
				auto spins1 = _interaction->Group1();
				auto spins2 = _interaction->Group2();
			arma::sp_cx_mat S1x, S1y, S1z;
			arma::sp_cx_mat S2x, S2y, S2z;

			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					if (_interaction->IgnoreTensors())
					{
						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{
						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
						A = RotateTensorFrameAndPowder(A);
						// factor 2 and +1/2 as in original implementation
						tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
						tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
						tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));
						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
						tmp(arma::span::all, arma::span::all) += half;
					}
					else
					{
						tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);
						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
						tmp(arma::span::all, arma::span::all) += half;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Zfs)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			// Build Sx, Sy, Sz for *each* electron in Group1
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				this->CreateOperator((*i)->Sx(), (*i), Sx);
				this->CreateOperator((*i)->Sy(), (*i), Sy);
				this->CreateOperator((*i)->Sz(), (*i), Sz);

				arma::mat Dtensor = ZfsTensorFromParameters(_interaction->Dvalue(), _interaction->Evalue(), _interaction->ES());
				Dtensor = RotateTensorFrameAndPowder(Dtensor);
				AddQuadraticSpinTensor(tmp, Sx, Sy, Sz, Dtensor);
			}
		}
		else if (_interaction->Type() == InteractionType::SemiClassicalField)
		{
			//  Grab orientation parameters
			int n = _interaction->Orientations();
			tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), n * this->HilbertSpaceDimensions());

			InternalCreateSCCompositeMatrix(_interaction, n, tmp);
		}
		else
		{
			// The interaction type was not recognized
			return false;
		}

		// Multiply by the isotropic value, if the tensor was isotropic
		if (ATensor != nullptr && IsIsotropic(*ATensor))
			tmp *= ATensor->Isotropic();

		// Multiply with the given prefactor (or 1.0 if none was specified)
		tmp *= _interaction->Prefactor();

		// Multiply by common prefactor. TODO: Consider which system of units to use
		if (_interaction->AddCommonPrefactor())
			tmp *= 8.79410005e+1;

		// Check whether we want a superspace or Hilbert space result
		if (this->useSuperspace)
		{
			arma::sp_cx_mat lhs;
			arma::sp_cx_mat rhs;
			int submats = tmp.n_cols / this->HilbertSpaceDimensions();
			int superoperatorspace = this->HilbertSpaceDimensions() * this->HilbertSpaceDimensions();
			_out = arma::zeros<arma::sp_cx_mat>(superoperatorspace, submats * superoperatorspace);
			for (int i = 0; i < submats; i++)
			{
				auto tmpmat = tmp.submat(0, i * this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions() - 1, (i + 1) * this->HilbertSpaceDimensions() - 1);
				auto result = this->SuperoperatorFromLeftOperator(tmpmat, lhs);
				result &= this->SuperoperatorFromRightOperator(tmpmat, rhs);
				if (result)
				{
					_out.submat(0, i * superoperatorspace, superoperatorspace - 1, (i + 1) * superoperatorspace - 1) = lhs - rhs;
				}
				else
					return false;
			}
		}
		else
		{
			_out = tmp;
		}

		return true;
	}

	// Returns the matrix representation of the Interaction object in a
	// high-field secular approximation (sparse matrix version). Single-spin,
	// double-spin, quadratic-spin, and ZFS interactions are rotated with their
	// interaction frame and with the current powder/crystallite orientation.
	bool SpinSpace::InteractionOperatorRotated_SA(const interaction_ptr &_interaction, arma::mat &_rotationmatrix, arma::sp_cx_mat &_out) const
	{
		// Make sure the interaction is valid
		if (_interaction == nullptr)
			return false;

		// Create temporary matrix to hold the result
		arma::sp_cx_mat tmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

		// Get the interaction tensor
		auto ATensor = _interaction->CouplingTensor();

		// Historical _SA path: rotate from the interaction tensor frame to the
		// molecular frame, then rotate the full crystallite to the lab frame.
		// CreateRotationMatrix is kept here deliberately so this code follows
		// the original implementation structure while preserving newer tensor
		// types below.
		auto ATensorFrame = _interaction->Framelist();
		arma::mat RFrame;
		if (!this->CreateRotationMatrix(ATensorFrame(0), ATensorFrame(1), ATensorFrame(2), RFrame))
		{
			std::cerr << "Failed to construct the rotation matrix for powder averaging in the lab frame." << std::endl;
		}

		auto RotateTensorFrameAndPowder = [&](const arma::mat &A) -> arma::mat
		{
			arma::mat Af = RFrame * A * RFrame.t();
			return _rotationmatrix * Af * _rotationmatrix.t();
		};

		if (_interaction->Type() == InteractionType::SingleSpin)
		{
			// Get the field at the current time or trajectory step
			arma::vec field;
			field = _interaction->Field();

			// Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
			auto spinlist = _interaction->Group1();
			arma::cx_mat Sx;
			arma::cx_mat Sy;
			arma::cx_mat Sz;

			// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				arma::mat g = arma::eye<arma::mat>(3, 3);
				if (!_interaction->IgnoreTensors())
				{
					g = arma::conv_to<arma::mat>::from((*i)->GetTensor().LabFrame());
				}

				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), Sx);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), Sy);
				this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), Sz);

				g = RotateTensorFrameAndPowder(g);

				// Secular approximation
				tmp += Sz * field(2) * sqrt(g(2, 2) * g(2, 2) + g(2, 1) * g(2, 1) + g(2, 0) * g(2, 0));
			}
		}
		else if (_interaction->Type() == InteractionType::DoubleSpin)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::cx_mat S1x;
			arma::cx_mat S1y;
			arma::cx_mat S1z;
			arma::cx_mat S2x;
			arma::cx_mat S2y;
			arma::cx_mat S2z;

			std::string interaction_type;
			bool transpose_tensor_for_electron_second = false;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					transpose_tensor_for_electron_second = false;
					if ((*i)->Type() == SpinAPI::SpinType::Electron)
					{
						// For better understanding: In the secular builder S1 is always the electron.
						// This keeps the electron coherence-order projection independent of
						// whether the user writes group1=electron/group2=nucleus or vice versa.
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), S2z);

						if ((*j)->Type() == SpinAPI::SpinType::Electron)
						{
							interaction_type = "Dipolar";
						}
						else if ((*j)->Type() == SpinAPI::SpinType::Nucleus)
						{
							interaction_type = "Hyperfine";
						}
						else
						{
							std::cerr << "Could not construct hamiltonian in the secular approximation. Unrecognized type of spins. Please specify in the Spin object 'type = electron' or 'type = nucleus'." << std::endl;
						}
					}
					else if ((*j)->Type() == SpinAPI::SpinType::Electron)
					{
						// Correct reversed hyperfine ordering. The old code
						// embedded the nucleus local matrices on the electron spin and vice versa.
						// Here S1 is the electron operator on spin j and S2 is the partner on spin i.
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), S1x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), S1y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), S1z);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sx()), (*i), S2x);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sy()), (*i), S2y);
						this->CreateOperator(arma::conv_to<arma::cx_mat>::from((*i)->Sz()), (*i), S2z);

						if ((*i)->Type() == SpinAPI::SpinType::Nucleus)
						{
							interaction_type = "Hyperfine";
							transpose_tensor_for_electron_second = true;
						}
						else
						{
							std::cerr << "Could not construct hamiltonian in the secular approximation. Unrecognized type of spins. Please specify in the Spin object 'type = electron' or 'type = nucleus'." << std::endl;
						}
					}
					else
					{
						std::cerr << "Could not construct hamiltonian in the secular approximation. Unrecognized type of spins. Please specify in the Spin object 'type = electron' or 'type = nucleus'. Additionally, currently there is no supported nucleus-nucleus interaction in the secular approximation." << std::endl;
					}

					// Project bilinear interactions onto the terms that are stationary
					// in a common electron-z rotating frame. Electron-electron
					// interactions retain total electron coherence order q=0, whereas
					// electron-nuclear hyperfine interactions retain only electron q=0.
					// In particular, isotropic A S.I becomes A Sz Iz in this frame.
					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
						A = RotateTensorFrameAndPowder(A);
						if (transpose_tensor_for_electron_second)
						{
							// For better understanding: group1=nucleus/group2=electron represents
							// I^T A S. After reordering to S^T A' I for electron-secular
							// projection, the tensor must become A' = A^T.
							A = A.t();
						}

						if (interaction_type == "Dipolar")
						{
							// Keep the historical electron-electron secular projection.
							// The q=0 projection tested during the refactor is numerically
							// indistinguishable for dmCry, but this form preserves the old
							// _SA behavior for existing inputs.
							tmp += S1x * S2x * A(0, 0) + S1y * S2y * A(1, 1) + S1z * S2z * A(2, 2);
						}
						else if (interaction_type == "Hyperfine")
						{
							tmp += S1z * (A(2, 0) * S2x + A(2, 1) * S2y + A(2, 2) * S2z);
						}
						else
						{
							std::cerr << "Unrecognized type of interaction. Could not construct hamiltonian in the secular approximation." << std::endl;
						}
					}
					else if (interaction_type == "Dipolar")
					{
						// Isotropic electron-electron coupling is already q=0.
						tmp += S1x * S2x + S1y * S2y + S1z * S2z;
					}
					else if (interaction_type == "Hyperfine")
					{
						// Electron-secular isotropic hyperfine limit.
						tmp += S1z * S2z;
					}
					else
					{
						std::cerr << "Unrecognized type of interaction. Could not construct hamiltonian in the secular approximation." << std::endl;
					}
				}
			}
		}
		else if (_interaction->Type() == InteractionType::QuadraticSpin)
		{
			auto spins1 = _interaction->Group1();
			arma::sp_cx_mat Sx;
			arma::sp_cx_mat Sy;
			arma::sp_cx_mat Sz;

			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				if (ATensor != nullptr && !IsIsotropic(*ATensor))
				{
					this->CreateOperator((*i)->Sz(), (*i), Sz);
					arma::mat A = arma::conv_to<arma::mat>::from(ATensor->LabFrame());
					A = RotateTensorFrameAndPowder(A);
					const double s = 0.5 * static_cast<double>((*i)->S());
					AddSecularQuadraticSpinTensor(tmp, Sz, s, A, false);
				}
				else
				{
					this->CreateOperator((*i)->Sx(), (*i), Sx);
					this->CreateOperator((*i)->Sy(), (*i), Sy);
					this->CreateOperator((*i)->Sz(), (*i), Sz);
					tmp += Sx * Sx + Sy * Sy + Sz * Sz;
				}
			}
		}
		else if (_interaction->Type() == InteractionType::Zfs)
		{
			auto spinlist = _interaction->Group1();
			arma::sp_cx_mat Sz;

			arma::mat Dtensor = ZfsTensorFromParameters(_interaction->Dvalue(), _interaction->Evalue(), _interaction->ES());
			Dtensor = RotateTensorFrameAndPowder(Dtensor);

			for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
			{
				this->CreateOperator((*i)->Sz(), (*i), Sz);
				const double s = 0.5 * static_cast<double>((*i)->S());
				AddSecularQuadraticSpinTensor(tmp, Sz, s, Dtensor, _interaction->ES());
			}
		}
		else if (_interaction->Type() == InteractionType::Exchange)
		{
			// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators
			auto spins1 = _interaction->Group1();
			auto spins2 = _interaction->Group2();
			arma::sp_cx_mat S1x;
			arma::sp_cx_mat S1y;
			arma::sp_cx_mat S1z;
			arma::sp_cx_mat S2x;
			arma::sp_cx_mat S2y;
			arma::sp_cx_mat S2z;

			// Fill the matrix with the sum of all the interactions
			for (auto i = spins1.cbegin(); i != spins1.cend(); i++)
			{
				for (auto j = spins2.cbegin(); j != spins2.cend(); j++)
				{
					// Obtain the magnetic moment operators within the Hilbert space
					if (_interaction->IgnoreTensors())
					{

						this->CreateOperator((*i)->Sx(), (*i), S1x);
						this->CreateOperator((*i)->Sy(), (*i), S1y);
						this->CreateOperator((*i)->Sz(), (*i), S1z);
						this->CreateOperator((*j)->Sx(), (*j), S2x);
						this->CreateOperator((*j)->Sy(), (*j), S2y);
						this->CreateOperator((*j)->Sz(), (*j), S2z);
					}
					else
					{

						this->CreateOperator((*i)->Tx(), (*i), S1x);
						this->CreateOperator((*i)->Ty(), (*i), S1y);
						this->CreateOperator((*i)->Tz(), (*i), S1z);
						this->CreateOperator((*j)->Tx(), (*j), S2x);
						this->CreateOperator((*j)->Ty(), (*j), S2y);
						this->CreateOperator((*j)->Tz(), (*j), S2z);
					}

					// TODO: This is not correct when the user uses a non isotropic exchange which he will not when being smart.
					if (ATensor != nullptr && !IsIsotropic(*ATensor))
					{
						std::cerr << "Exchange interaction can only be isotropic." << std::endl;

						// Use the tensor to calculate the product S_1 * A * S_2
						auto A = ATensor->LabFrame();
						tmp += 2.0 * (S1x * S2x * A(0, 0) + S1x * S2y * A(0, 1) + S1x * S2z * A(0, 2));
						tmp += 2.0 * (S1y * S2x * A(1, 0) + S1y * S2y * A(1, 1) + S1y * S2z * A(1, 2));
						tmp += 2.0 * (S1z * S2x * A(2, 0) + S1z * S2y * A(2, 1) + S1z * S2z * A(2, 2));

						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp(arma::span::all, arma::span::all) += half;
					}
					else
					{
						// TODO: Be carfeul with the sign here!!! Usually you would use a minus sign to at the whol interaction.
						//  If there is no interaction tensor or if the tensor is isotropic, just take the dot product
						tmp += 2.0 * (S1x * S2x + S1y * S2y + S1z * S2z);

						arma::sp_cx_mat half = 0.5 * arma::eye<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());

						tmp(arma::span::all, arma::span::all) += half;
					}
				}
			}
		}
		else
		{
			// The interaction type was not recognized
			std::cerr << "The interaction type is not currently supported in the rotating frame approximation." << std::endl;
			return false;
		}

		// Multiply by the isotropic value, if the tensor was isotropic
		if (ATensor != nullptr && IsIsotropic(*ATensor))
			tmp *= ATensor->Isotropic();

		// Multiply with the given prefactor (or 1.0 if none was specified)
		tmp *= _interaction->Prefactor();

		// Multiply by common prefactor. TODO: Consider which system of units to use
		if (_interaction->AddCommonPrefactor())
			tmp *= 8.79410005e+1;

		// Check whether we want a superspace or Hilbert space result
		if (this->useSuperspace)
		{
			arma::sp_cx_mat lhs;
			arma::sp_cx_mat rhs;
			int submats = tmp.n_cols / this->HilbertSpaceDimensions();
			int superoperatorspace = this->HilbertSpaceDimensions() * this->HilbertSpaceDimensions();
			_out = arma::zeros<arma::sp_cx_mat>(superoperatorspace, submats * superoperatorspace);
			for (int i = 0; i < submats; i++)
			{
				auto tmpmat = tmp.submat(0, i * this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions() - 1, (i + 1) * this->HilbertSpaceDimensions() - 1);
				auto result = this->SuperoperatorFromLeftOperator(tmpmat, lhs);
				result &= this->SuperoperatorFromRightOperator(tmpmat, rhs);
				if (result)
				{
					_out.submat(0, i * superoperatorspace, superoperatorspace - 1, (i + 1) * superoperatorspace - 1) = lhs - rhs;
				}
				else
					return false;
			}
		}
		else
		{
			// We already have the result in the dense matrix to the Hamiltonian at the given time or trajectorye Hilbert space
			_out = tmp;
		}

		return true;
	}

	// Sets the dense matrix to the Hamiltonian at the given time or trajectory step
	bool SpinSpace::Hamiltonian(arma::cx_mat &_out, int TaskNum) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		if (this->interactions.size() < 1)
		{
			_out = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		// Get the first interaction contribution
		bool semiclassical = false;
		auto i = this->interactions.cbegin();
		arma::cx_mat tmp;
		arma::cx_mat result;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		if ((*i)->Type() == InteractionType::SemiClassicalField)
		{
			SemiClassicalInteractions.push_back((*i));
			semiclassical = true;
		}
		else
		{
			if (!this->InteractionOperator((*i), result))
				return false;
		}

		// We have already used the first interaction
		i++;

		// Loop through the rest
		for (; i != this->interactions.cend(); i++)
		{
			if ((*i)->Type() == InteractionType::SemiClassicalField)
			{
				SemiClassicalInteractions.push_back((*i));
				semiclassical = true;
				continue;
			}
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;
			result += tmp;
		}

		if (semiclassical && SCSupportedTasks(TaskNum))
		{
			arma::cx_mat SCout;
			if (!SemiClassicalHamiltonian(SCout, SemiClassicalInteractions))
				return false;
			int width, height = 0;
			width = SCout.n_cols;
			height = SCout.n_rows;
			int Block1h, Block1w = 0;
			Block1h = result.n_rows;
			Block1w = result.n_cols;
			_out = arma::cx_mat(Block1h + height, (width > Block1w) ? width : Block1w);
			_out.submat(0, 0, Block1h - 1, Block1w - 1) = result;
			if (result.is_hermitian() == false)
			{
				std::cin.get();
			}
			_out.submat(Block1h, 0, Block1h + height - 1, width - 1) = SCout;
			return true;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the Hamiltonian at the given time or trajectory step
	bool SpinSpace::Hamiltonian(arma::sp_cx_mat &_out, int TaskNum) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		if (this->interactions.size() < 1)
		{
			_out = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
			return true;
		}

		bool semiclassical = false;
		// Get the first interaction contribution
		auto i = this->interactions.cbegin();
		arma::sp_cx_mat tmp;
		arma::sp_cx_mat result;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		if ((*i)->Type() == InteractionType::SemiClassicalField)
		{
			SemiClassicalInteractions.push_back((*i));
			semiclassical = true;
		}
		else
		{
			if (!this->InteractionOperator((*i), result))
				return false;
		}

		// We have already used the first interaction
		i++;

		// Loop through the rest
		for (; i != this->interactions.cend(); i++)
		{
			if ((*i)->Type() == InteractionType::SemiClassicalField)
			{
				SemiClassicalInteractions.push_back((*i));
				semiclassical = true;
				continue;
			}
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;
			result += tmp;
		}

		arma::sp_cx_mat SCout;
		if (semiclassical && SCSupportedTasks(TaskNum))
		{
			if (!SemiClassicalHamiltonian(SCout, SemiClassicalInteractions))
				return false;

			int width, height = 0;
			width = SCout.n_cols;
			height = SCout.n_rows;
			int Block1h, Block1w = 0;
			Block1h = result.n_rows;
			Block1w = result.n_cols;
			_out = arma::sp_cx_mat(Block1h + height, (width > Block1w) ? width : Block1w);
			_out.submat(0, 0, Block1h - 1, Block1w - 1) = result;
			if (result.is_hermitian() == false)
			{
				std::cin.get();
			}
			if (SCout.n_nonzero != 0)
				_out.submat(Block1h, 0, Block1h + height - 1, width - 1) = SCout;
			return true;
		}
		_out = result;
		return true;
	}

	bool SpinSpace::SemiClassicalHamiltonian(arma::cx_mat &_out, std::vector<interaction_ptr> &interactions) const
	{
		arma::sp_cx_mat _outSP = arma::conv_to<arma::sp_cx_mat>::from(_out);
		bool result = SemiClassicalHamiltonian(_outSP, interactions);
		_out = arma::conv_to<arma::cx_mat>::from(_outSP);
		return result;
	}

	bool SpinSpace::SemiClassicalHamiltonian(arma::sp_cx_mat &_out, std::vector<interaction_ptr> &interactions) const
	{
		arma::sp_cx_mat tmp;
		arma::sp_cx_mat result;

		// obtain number of orientations per each interaction
		int samples = 0;
		for (auto i = interactions.begin(); i != interactions.cend(); i++)
		{
			if (!(*i)->IsValid())
			{
				continue;
			}
			samples += 1;
		}
		int *ori = (int *)malloc(samples * sizeof(int));
		for (auto i = interactions.begin(); i != interactions.cend(); i++)
		{
			ori[i - interactions.begin()] = (*i)->Orientations();
		}
		int MaxOri = 0;
		for (int i = 0; i < samples; i++)
		{
			if (ori[i] > MaxOri)
			{
				MaxOri = ori[i];
			}
		}
		int operatorspace = 0;
		if (this->useSuperspace)
			operatorspace = this->HilbertSpaceDimensions() * this->HilbertSpaceDimensions();
		else
			operatorspace = this->HilbertSpaceDimensions();
		result = arma::sp_cx_mat(samples * operatorspace, MaxOri * operatorspace);

		int op = 0;
		for (auto i = interactions.begin(); i != interactions.cend(); i++)
		{
			if (!(*i)->IsValid())
			{
				continue;
			}
			if (!this->InteractionOperator((*i), tmp))
			{
				return false;
			}
			result.submat(op * operatorspace, 0, (op + 1) * operatorspace - 1, ori[op] * operatorspace - 1) = tmp;
			op += 1;
		}
		_out = result;

		free(ori);
		return true;
	}

	// Sets the dense matrix to the part of the Hamiltonian that is independent of time or trajectory step
	bool SpinSpace::StaticHamiltonian(arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if ((*i)->Type() == InteractionType::SemiClassicalField)
			{
				SemiClassicalInteractions.push_back((*i));
				// semiclassical = true;
				continue;
			}
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;
			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the part of the Hamiltonian that is independent of time or trajectory step
	bool SpinSpace::StaticHamiltonian(arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if ((*i)->Type() == InteractionType::SemiClassicalField)
			{
				SemiClassicalInteractions.push_back((*i));
				// semiclassical = true;
				continue;
			}
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the Hamiltonian that is dependent on time and/or trajectory step
	bool SpinSpace::DynamicHamiltonian(arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip static interactions
			if (IsStatic(*(*i)))
				continue;

			// Attempt to get the matrix representing the Interaction object in the spin space
			if ((*i)->Type() == InteractionType::SemiClassicalField)
			{
				SemiClassicalInteractions.push_back((*i));
				// semiclassical = true;
				continue;
			}
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the sparse matrix to the part of the Hamiltonian that is dependent on time and/or trajectory step
	bool SpinSpace::DynamicHamiltonian(arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip static interactions
			if (IsStatic(*(*i)))
				continue;

			if ((*i)->Type() == InteractionType::SemiClassicalField)
			{
				SemiClassicalInteractions.push_back((*i));
				// semiclassical = true;
				continue;
			}
			// Attempt to get the matrix representing the Interaction object in the spin space
			if (!this->InteractionOperator((*i), tmp))
				return false;

			result += tmp;
		}

		_out = result;
		return true;
	}

	// Sets the dense matrix to the part of the Hamiltonian that is used in the thermal equilibrium (without Zeeman terms)
	bool SpinSpace::ThermalHamiltonian(std::vector<std::string> thermalhamiltonian_list, arma::cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Check if the interaction name is in the thermalhamiltonian_list
			if (std::find(thermalhamiltonian_list.begin(), thermalhamiltonian_list.end(), (*i)->Name()) != thermalhamiltonian_list.end())
			{
				// Attempt to get the matrix representing the Interaction object in the spin space
				if ((*i)->Type() == InteractionType::SemiClassicalField)
				{
					SemiClassicalInteractions.push_back((*i));
					// semiclassical = true;
					continue;
				}
				if (!this->InteractionOperator((*i), tmp))
					return false;

				result += tmp;
			}
		}

		_out = result;
		return true;
	}

	// Sets the sparce matrix to the part of the Hamiltonian that is used in the thermal equilibrium (without Zeeman terms)
	bool SpinSpace::ThermalHamiltonian(std::vector<std::string> thermalhamiltonian_list, arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::cx_mat result = arma::zeros<arma::cx_mat>(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Check if the interaction name is in the thermalhamiltonian_list
			if (std::find(thermalhamiltonian_list.begin(), thermalhamiltonian_list.end(), (*i)->Name()) != thermalhamiltonian_list.end())
			{
				if ((*i)->Type() == InteractionType::SemiClassicalField)
				{
					SemiClassicalInteractions.push_back((*i));
					// semiclassical = true;
					continue;
				}
				// Attempt to get the matrix representing the Interaction object in the spin space
				if (!this->InteractionOperator((*i), tmp))
					return false;

				result += tmp;
			}
		}

		_out = result;
		return true;
	}

	// Sets the rotated sparce matrix to the part of the Hamiltonian in the secular approximation
	bool SpinSpace::BaseHamiltonianRotated_SA(std::vector<std::string> basehamiltonian_list, arma::mat rotmatrix, arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Check if the interaction name is in the basehamiltonian_list
			if (std::find(basehamiltonian_list.begin(), basehamiltonian_list.end(), (*i)->Name()) != basehamiltonian_list.end())
			{
				if ((*i)->Type() == InteractionType::SemiClassicalField)
				{
					SemiClassicalInteractions.push_back((*i));
					// semiclassical = true;
					continue;
				}
				// Attempt to get the matrix representing the Interaction object in the spin space
				if (!this->InteractionOperatorRotated_SA((*i), rotmatrix, tmp))
					return false;

				result += tmp;
			}
		}

		_out = result;
		return true;
	}

	bool SpinSpace::BaseHamiltonianRotatedZYZ(std::vector<std::string> basehamiltonian_list, arma::mat rotmatrix, arma::sp_cx_mat &_out) const
	{
		// If we don't have any interactions, the Hamiltonian is zero
		arma::sp_cx_mat result = arma::sp_cx_mat(this->SpaceDimensions(), this->SpaceDimensions());
		if (this->interactions.size() < 1)
		{
			_out = result;
			return true;
		}

		arma::sp_cx_mat tmp;
		// bool semiclassical = false;
		std::vector<interaction_ptr> SemiClassicalInteractions = {};
		for (auto i = this->interactions.cbegin(); i != this->interactions.cend(); i++)
		{
			// Skip any dynamic interactions (time-dependent or with a trajectory)
			if (!IsStatic(*(*i)))
				continue;

			// Check if the interaction name is in the basehamiltonian_list
			if (std::find(basehamiltonian_list.begin(), basehamiltonian_list.end(), (*i)->Name()) != basehamiltonian_list.end())
			{
				if ((*i)->Type() == InteractionType::SemiClassicalField)
				{
					SemiClassicalInteractions.push_back((*i));
					// semiclassical = true;
					continue;
				}
				// Attempt to get the matrix representing the Interaction object in the spin space
				if (!this->InteractionOperatorRotatedZYZ((*i), rotmatrix, tmp))
					return false;

				result += tmp;
			}
		}

		_out = result;
		return true;
	}

	bool SpinSpace::PowderHamiltonianRotatedSA(const std::vector<std::string> &h0list,
											   const std::vector<std::string> &h1list,
											   const arma::mat &rotmatrix,
											   arma::sp_cx_mat &H0,
											   arma::sp_cx_mat &H1,
											   arma::sp_cx_mat &H) const
	{
		// Powder magnetic-resonance tasks use one crystallite orientation for
		// all tensor-aware terms. H0 is secularized in the static-field frame;
		// H1 is then rotated with the same molecular-to-lab orientation so the
		// microwave interaction does not silently live in a different frame.
		if (!this->BaseHamiltonianRotated_SA(h0list, rotmatrix, H0))
			return false;

		if (!this->BaseHamiltonianRotatedZYZ(h1list, rotmatrix, H1))
			return false;

		H = H0 + H1;
		return true;
	}

	bool SpinSpace::InternalCreateSCCompositeMatrix(const SpinAPI::interaction_ptr &_interaction, int n, arma::sp_cx_mat &tmp) const
	{
		// Obtain lists of interacting spins, coupling tensor, and define matrices to hold the magnetic moment operators

		auto spinlist = _interaction->Group1();
		//  Grab amplitude parameters
		std::vector<double> B = _interaction->VL();
		double BMax = std::reduce(B.begin(), B.end());

		// Build Sx, Sy, Sz for *each* electron in Group1
		arma::sp_cx_mat Sx;
		arma::sp_cx_mat Sy;
		arma::sp_cx_mat Sz;

		std::vector<arma::sp_cx_mat> Samples;

		// Fill the matrix with the sum of all the interactions (i.e. between spin magnetic moment and fields)
		for (auto i = spinlist.cbegin(); i != spinlist.cend(); i++)
		{
			if (_interaction->IgnoreTensors())
			{
				this->CreateOperator((*i)->Sx(), (*i), Sx);
				this->CreateOperator((*i)->Sy(), (*i), Sy);
				this->CreateOperator((*i)->Sz(), (*i), Sz);
			}
			else
			{
				this->CreateOperator((*i)->Tx(), (*i), Sx);
				this->CreateOperator((*i)->Ty(), (*i), Sy);
				this->CreateOperator((*i)->Tz(), (*i), Sz);
			}

			RunSection::MCSpherePoint *points = RunSection::CalculateMCSpherePoints(n, BMax);
			int currentcol = 0;
			typedef std::pair<std::pair<std::array<double, 3>, double>, double> WeightsType;
			std::vector<WeightsType> weights;

			for (int k = 0; k < n; k++)
			{
				std::array<double, 3> NuclearSpinVector;
				RunSection::RetrieveMCPoint(NuclearSpinVector, points, k);
				double x = NuclearSpinVector[0];
				double y = NuclearSpinVector[1];
				double z = NuclearSpinVector[2];
				double r = std::sqrt(x * x + y * y + z * z);
				if (z <= 0)
				{
					r = -r;
				}
				double sampleweight = _interaction->f({x, y, z}); // distribution funcition ptr
				weights.push_back({{{x, y, z}, r}, sampleweight});
			}

			auto weights_sort = [](const WeightsType &a, const WeightsType &b)
			{
				return a.second < b.second;
			};

			std::sort(weights.begin(), weights.end(), weights_sort);

			// seperate the weights into left and right halves
			std::vector<WeightsType> weightsL;
			std::vector<WeightsType> weightsR;
			bool left = true;
			for (auto &weight : weights)
			{
				left = weight.first.first[2] <= 0 ? true : false;
				if (left)
				{
					weightsL.push_back(weight);
				}
				else
				{
					weightsR.push_back(weight);
				}
				left = !left;
			}

			auto weightsR_sort = [](const WeightsType &a, const WeightsType &b)
			{
				return a.second > b.second;
			};
			std::sort(weightsR.begin(), weightsR.end(), weightsR_sort);
			weights.clear();
			weights.insert(weights.end(), weightsL.begin(), weightsL.end());
			weights.insert(weights.end(), weightsR.begin(), weightsR.end());

			for (int k = 0; k < n; k++)
			{
				arma::sp_cx_mat SampleTmp = arma::zeros<arma::sp_cx_mat>(this->HilbertSpaceDimensions(), this->HilbertSpaceDimensions());
				auto [x, y, z] = weights[k].first.first;
				SampleTmp = Sx * x + Sy * y + Sz * z;
				tmp.submat(0, currentcol, this->HilbertSpaceDimensions() - 1, currentcol + this->HilbertSpaceDimensions() - 1) = SampleTmp;
				currentcol += this->HilbertSpaceDimensions();
			}
			free(points);
			double area = 0;
			std::vector<double> spacing;
			std::vector<double> spacingcheck;
			for (unsigned int k = 0; k < weights.size(); k++)
			{
				if (k < weights.size() - 1)
					spacingcheck.push_back(std::abs(weights[k + 1].first.second - weights[k].first.second));
				spacing.push_back(weights[k].first.second);
			}
			// get the normalization factor
			for (unsigned int k = 0; k < weights.size() - 1; k++)
			{
				double t = weights[k].second + weights[k + 1].second;
				area += (0.5 * t) * spacingcheck[k];
			}
			// normalise the weights
			std::vector<double> weights_final;
			for (auto &weight : weights)
			{
				weights_final.push_back(weight.second / area);
			}
			_interaction->GetOriWeights() = weights_final;
			_interaction->GetSpacing() = spacing;
		}
		return true;
	}

	bool SpinSpace::InternalCreateSCCompositeMatrix(const SpinAPI::interaction_ptr &_interaction, int n, arma::cx_mat &tmp) const
	{
		arma::sp_cx_mat tmp2 = arma::conv_to<arma::sp_cx_mat>::from(tmp);
		bool result = InternalCreateSCCompositeMatrix(_interaction, n, tmp2);
		tmp = arma::conv_to<arma::cx_mat>::from(tmp2);
		return result;
	}

	bool SpinSpace::SCSupportedTasks(int tasknum) const
	{
		static std::vector<int> supported = {
			1, // STATICSS
			2  // STATICSS_TIMEEVO
		};

		if (std::find(supported.begin(), supported.end(), tasknum) != supported.end())
			return true;
		std::cout << "[INFO]: Task does not suppport SC approximation" << std::endl;
#if ASSERT == 1
		throw std::exception();
#endif
		return false;
	}

	bool SpinSpace::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		arma::mat R1 = {
			{std::cos(_alpha), -std::sin(_alpha), 0.0},
			{std::sin(_alpha), std::cos(_alpha), 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{std::cos(_beta), 0.0, std::sin(_beta)},
			{0.0, 1.0, 0.0},
			{-std::sin(_beta), 0.0, std::cos(_beta)}};

		arma::mat R3 = {
			{std::cos(_gamma), -std::sin(_gamma), 0.0},
			{std::sin(_gamma), std::cos(_gamma), 0.0},
			{0.0, 0.0, 1.0}};

		_R = R1 * R2 * R3;

		return true;
	}
}
