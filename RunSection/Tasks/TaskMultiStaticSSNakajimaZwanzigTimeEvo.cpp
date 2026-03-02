/////////////////////////////////////////////////////////////////////////
// TaskMultiStaticSSNakajimaZwanzigTimeEvo implementation (RunSection module)

// -- Multi-system version: Allows transitions between SpinSystems --
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskMultiStaticSSNakajimaZwanzigTimeEvo.h"
#include "Settings.h"
#include "ObjectParser.h"

#include <omp.h>
#include <memory>
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "Operator.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskMultiStaticSSNakajimaZwanzigTimeEvo Constructors and Destructor
	// -----------------------------------------------------
	TaskMultiStaticSSNakajimaZwanzigTimeEvo::TaskMultiStaticSSNakajimaZwanzigTimeEvo(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4),
																																							  reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskMultiStaticSSNakajimaZwanzigTimeEvo::~TaskMultiStaticSSNakajimaZwanzigTimeEvo()
	{
	}
	// -----------------------------------------------------
	// TaskMultiStaticSSNakajimaZwanzigTimeEvo protected methods
	// -----------------------------------------------------
	bool TaskMultiStaticSSNakajimaZwanzigTimeEvo::RunLocal()
	{
		this->Log() << "Running method StaticSS-NakajimaZwanzig-MultiSystem." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems to obtain SpinSpace objects
		auto systems = this->SpinSystems();
		std::vector<std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>> spaces;
		unsigned int dimensions = 0;
		for (auto i = systems.cbegin(); i < systems.cend(); i++)
		{
			auto space = std::make_shared<SpinAPI::SpinSpace>(*(*i));

			// We are using hilbert space because of R, later it will be transformed into Liouville space
			space->UseSuperoperatorSpace(false);
			space->SetReactionOperatorType(this->reactionOperators);
			dimensions += space->SpaceDimensions();

			// Make sure to save the newly created spin space
			spaces.push_back(std::pair<std::shared_ptr<SpinAPI::SpinSystem>, std::shared_ptr<SpinAPI::SpinSpace>>(*i, space));
		}

		// Now, create a matrix to hold the Liouvillian superoperator, the initial state and eigenvectors of respective subspaces
		arma::cx_mat L((dimensions * dimensions), (dimensions * dimensions));
		arma::cx_vec rho0((dimensions * dimensions));
		std::vector<arma::cx_mat> eigveclist;

		unsigned int nextDimension = 0; // Keeps track of the dimension where the next spin space starts

		// Loop through the systems again to fill this matrix and vector
		for (auto i = spaces.cbegin(); i < spaces.cend(); i++)
		{
			// Make sure we have an initial state
			auto initial_states = i->first->InitialState();
			arma::cx_mat rho0HS;
			if (initial_states.size() < 1)
			{
				this->Log() << "Note: No initial state specified for spin system \"" << i->first->Name() << "\", setting the initial state to zero." << std::endl;
				rho0HS = arma::zeros<arma::cx_mat>(i->second->HilbertSpaceDimensions(), i->second->HilbertSpaceDimensions());
			}
			else
			{
				// Get the initial state for the current system
				for (auto j = initial_states.cbegin(); j < initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;
					if ((*j) == nullptr) // "Thermal initial state"
					{
						this->Log() << "Initial state = thermal " << std::endl;

						std::vector<std::string> thermalhamiltonian_list = i->first->ThermalHamiltonianList();
						this->Log() << "ThermalHamiltonianList = [";
						for (size_t k = 0; k < thermalhamiltonian_list.size(); k++)
						{
							this->Log() << thermalhamiltonian_list[k];
							if (k < thermalhamiltonian_list.size() - 1)
								this->Log() << ", ";
						}
						this->Log() << "]" << std::endl;

						double temperature = i->first->Temperature();
						this->Log() << "Temperature = " << temperature << "K" << std::endl;

						if (!i->second->GetThermalState(*(i->second), temperature, thermalhamiltonian_list, tmp_rho0))
						{
							this->Log() << "ERROR: Failed to obtain projection matrix onto thermal state, initial state of SpinSystem \"" << i->first->Name() << "\"." << std::endl;
							return false;
						}
					}
					else if (!i->second->GetState(*j, tmp_rho0))
					{
						this->Log() << "ERROR: Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << i->first->Name() << "\"." << std::endl;
						return false;
					}
					if (j == initial_states.cbegin())
						rho0HS = tmp_rho0;
					else
						rho0HS += tmp_rho0;
				}
				rho0HS /= arma::trace(rho0HS); // The density operator should have a trace of 1
			}

			// Next, get the Hamiltonian
			arma::cx_mat H;
			if (!i->second->Hamiltonian(H))
			{
				this->Log() << "ERROR: Failed to obtain the superspace Hamiltonian for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}

			// NakajimaZwanzig tensor
			arma::cx_mat R;
			R.set_size(size(kron(H, H)));
			R.zeros();

			// || i->first->interactions_size() < 1

			if (i->first->Name() == "products")
			{
				// Now put the initial state into the superspace vector
				arma::cx_vec rho0vec;
				if (!i->second->OperatorToSuperspace(rho0HS, rho0vec))
				{
					this->Log() << "ERROR: Failed convert initial state to superspace for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				rho0.rows(nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) = rho0vec;

				// ---------------------------------------------------------------
				// SETUP COMPLETE HAMILTONIAN
				// ---------------------------------------------------------------
				// Transform  H0 into superspace
				arma::cx_mat lhs;
				arma::cx_mat rhs;
				arma::cx_mat H_SS;

				// Transforming into superspace
				i->second->SuperoperatorFromLeftOperator(H, lhs);
				i->second->SuperoperatorFromRightOperator(H, rhs);

				H_SS = lhs - rhs;

				// Conversion into Liouville space of static Hamiltonian
				L.submat(nextDimension, nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) = arma::cx_double(0.0, -1.0) * H_SS;

				// Then get the reaction operators
				arma::cx_mat K;
				if (!i->second->TotalReactionOperator(K))
				{
					this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operators for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				// Transform ReactionOperator into superspace
				arma::cx_mat Klhs;
				arma::cx_mat Krhs;
				arma::cx_mat K_SS;

				i->second->SuperoperatorFromLeftOperator(K, Klhs);
				i->second->SuperoperatorFromRightOperator(K, Krhs);

				K_SS = Klhs + Krhs;

				// Conversion into Liouville space of reaction operator
				L.submat(nextDimension, nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) -= K_SS;
			}
			else
			{

				arma::cx_mat eigen_vec; // To hold eigenvectors
				arma::vec eigen_val;	// To hold eigenvalues
				arma::cx_mat eig_val_mat;

				// ----------------------------------------------------------------
				// DIAGONALIZATION OF H0// We need all of these operators
				// ----------------------------------------------------------------
				this->Log() << "Starting diagonalization..." << std::endl;
				arma::eig_sym(eigen_val, eigen_vec, H);
				this->Log() << "Diagonalization done! Eigenvalues: " << eigen_val.n_elem << ", eigenvectors: " << eigen_vec.n_cols << std::endl;

				eigveclist.push_back(eigen_vec);

				// Rotate density operator in eigenbasis of H0
				rho0HS = (eigen_vec.t() * rho0HS * eigen_vec);

				// Now put the initial state into the superspace vector
				arma::cx_vec rho0vec;
				if (!i->second->OperatorToSuperspace(rho0HS, rho0vec))
				{
					this->Log() << "ERROR: Failed convert initial state to superspace for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				rho0.rows(nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) = rho0vec;

				// ----------------------------------------------------------------
				// CONSTRUCTING TRANSITION MATRIX "lambda" OUT OF EIGENVALUES OF H0 FOR SPECTRAL DENSITIES
				// ----------------------------------------------------------------

				// Constructing diagonal matrix with eigenvalues of H0
				eig_val_mat = arma::diagmat(arma::conv_to<arma::cx_mat>::from(eigen_val));

				// Unit matrix
				arma::cx_mat one;
				one.set_size(arma::size(H));
				one.eye();

				arma::cx_mat lambda;
				lambda = (arma::kron(eig_val_mat, one) - arma::kron(one, eig_val_mat.st()));

				// ---------------------------------------------------------------
				// SETUP RELAXATION OPERATOR
				// ---------------------------------------------------------------

				// Temporary NakajimaZwanzig tensor
				arma::cx_mat tmp_R;
				tmp_R.set_size(size(kron(H, H)));
				tmp_R.zeros();

				// R Tensor array pointer for parallelization
				arma::cx_mat **ptr_R = NULL;
				int threads = 1;

// Get number of threads for tensor pointer arrays
#pragma omp parallel
				{
					threads = omp_get_num_threads();
				}

				this->Log() << "Setting up threads. " << threads << " CPU are used." << std::endl;

				ptr_R = new arma::cx_mat *[threads];

				for (int l = 0; l < threads; l++)
				{
					ptr_R[l] = new arma::cx_mat(tmp_R);
					*ptr_R[l] *= 0.0;
				}

				// Spectral density matrix
				arma::cx_mat SpecDens;
				SpecDens.set_size(arma::size(H));
				SpecDens.zeros();

				// Correlation function set ups
				std::vector<double> tau_c_list;
				std::vector<double> ampl_list;
				arma::cx_double ampl_combined = 0.0;

				// Defining variables for parameter of user
				int terms = 0;
				int def_g = 0;
				int def_multexpo = 0;
				int ops = 0;
				int coeff = 0;

				this->Log() << "Contructing all required pre-matrices." << std::endl;

				// Defining varibales for loops and storage of operators
				int num_op = 0;
				int num_element = 0;
				arma::cx_mat **ptr_Tensors = NULL;

				// Spin-Operators
				arma::cx_mat *Sz1 = new arma::cx_mat;
				arma::cx_mat *Sx1 = new arma::cx_mat;
				arma::cx_mat *Sy1 = new arma::cx_mat;
				arma::cx_mat *Sz2 = new arma::cx_mat;
				arma::cx_mat *Sx2 = new arma::cx_mat;
				arma::cx_mat *Sy2 = new arma::cx_mat;

				// Double-Spin operators
				arma::cx_mat *Sx1Sx2 = new arma::cx_mat;
				arma::cx_mat *Sx1Sy2 = new arma::cx_mat;
				arma::cx_mat *Sx1Sz2 = new arma::cx_mat;
				arma::cx_mat *Sy1Sx2 = new arma::cx_mat;
				arma::cx_mat *Sy1Sy2 = new arma::cx_mat;
				arma::cx_mat *Sy1Sz2 = new arma::cx_mat;
				arma::cx_mat *Sz1Sx2 = new arma::cx_mat;
				arma::cx_mat *Sz1Sy2 = new arma::cx_mat;
				arma::cx_mat *Sz1Sz2 = new arma::cx_mat;

				// T0 for rank 0 & 2 (rank 1 neglected but can be added - see SpinSpace_operators.cpp - space.Rk1SphericalTensorXXX)
				arma::cx_mat *T0_rank_0 = new arma::cx_mat;
				arma::cx_mat *T0_rank_2 = new arma::cx_mat;

				// Tp1 & T1m for rank 2 (rank 1 neglected but can be added - see SpinSpace_operators.cpp - space.Rk1SphericalTensorXXX)
				arma::cx_mat *Tp1 = new arma::cx_mat;
				arma::cx_mat *Tm1 = new arma::cx_mat;

				// Tp2 & Tp2 for rank 2
				arma::cx_mat *Tp2 = new arma::cx_mat;
				arma::cx_mat *Tm2 = new arma::cx_mat;

				// arma::cx_mat ** Tensors_rotating = NULL;
				int k;
				int s;
				int l;

				// ------------------------------------------------------------------
				// STARTING WITH RELAXATION MATRIX CONSTRUCTION - GOOD LUCK
				// ------------------------------------------------------------------

				this->Log() << "Starting with construction of relaxation matrix." << std::endl;
				for (auto interaction = i->first->interactions_cbegin(); interaction < i->first->interactions_cend(); interaction++)
				{
					// Chosen parameter in input file
					this->Log() << "------------------------------------------" << std::endl;
					this->Log() << "Chosen input parameter for interaction:  " << (*interaction)->Name() << std::endl;
					this->Log() << "------------------------------------------" << std::endl;
					(*interaction)->Properties()->Get("terms", terms);
					this->Log() << "terms == " << terms << std::endl;
					(*interaction)->Properties()->Get("def_g", def_g);
					this->Log() << "def_g == " << def_g << std::endl;
					(*interaction)->Properties()->Get("def_multexpo", def_multexpo);
					this->Log() << "def_multexpo == " << def_multexpo << std::endl;
					(*interaction)->Properties()->Get("ops", ops);
					this->Log() << "ops == " << ops << std::endl;
					(*interaction)->Properties()->Get("coeff", coeff);
					this->Log() << "coeff == " << coeff << std::endl;
					this->Log() << "------------------------------------------" << std::endl;

					// Check if def_multexpo keyword is used
					if ((*interaction)->Properties()->Get("def_multexpo", def_multexpo) && def_multexpo == 1)
					{
						this->Log() << "Multiple exponential fits found for different operators" << std::endl;
						arma::cx_mat **ptr_SpecDens = NULL;
						arma::mat tau_c_mat;
						arma::mat ampl_mat;

						// Check if tau_c and g lists are available and get the lists in matrix form (tau_c_mat(m,n) - m -> different correlaton function, n -> element in correlation function)
						if ((*interaction)->Properties()->GetMatrix("tau_c", tau_c_mat))
						{
							if ((*interaction)->Properties()->GetMatrix("g", ampl_mat))
							{
								// Check if SingleSpin or DoubleSpin interaction
								if ((*interaction)->Type() == SpinAPI::InteractionType::SingleSpin)
								{
									// ------------------------------------------------------------------
									// Relaxation Matrix Construction for Interactions of Type::SingleSpin
									// ------------------------------------------------------------------

									// Get groups with respect to interaction
									auto group1 = (*interaction)->Group1();
									// Loop through groups to get all interaction type
									for (auto s1 = group1.cbegin(); s1 < group1.cend(); s1++)
									{
										// Which operator basis was chosen:
										if ((*interaction)->Properties()->Get("ops", ops) && ops == 1)
										{
											// --------------------------------------------------------
											// CREATION OF SPIN OPERATORS (Sz, Sx, Sy)
											// --------------------------------------------------------

											this->Log() << "Sx, Sy and Sz operator basis was chosen - ops == 1" << std::endl;

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
											{
												return false;
											}

											// Build double-spin operators
											*Sx1Sx2 = (*Sx1); // * Bx
											*Sx1Sy2 = (*Sx1); // * By
											*Sx1Sz2 = (*Sx1); // * Bz
											*Sy1Sx2 = (*Sy1); // * Bx
											*Sy1Sy2 = (*Sy1); // * By
											*Sy1Sz2 = (*Sy1); // * Bz
											*Sz1Sx2 = (*Sz1); // * Bx
											*Sz1Sy2 = (*Sz1); // * By
											*Sz1Sz2 = (*Sz1); // * Bz

											// Put all tensors on pointer array
											num_op = 9;
											delete[] ptr_Tensors;
											ptr_Tensors = new arma::cx_mat *[num_op];

											ptr_Tensors[0] = Sx1Sx2;
											ptr_Tensors[1] = Sx1Sy2;
											ptr_Tensors[2] = Sx1Sz2;
											ptr_Tensors[3] = Sy1Sx2;
											ptr_Tensors[4] = Sy1Sy2;
											ptr_Tensors[5] = Sy1Sz2;
											ptr_Tensors[6] = Sz1Sx2;
											ptr_Tensors[7] = Sz1Sy2;
											ptr_Tensors[8] = Sz1Sz2;

											// Bx,Bx (Sx1); Bx,By (Sx1); Bx,Bz (Sx1);By, Bz (Sx1);...; Bx,Bx (Sy1)
										}
										else
										{
											// ----------------------------------------------------------------
											// CREATION OF IRREDUCIBLE SPHERICAL TENSORS
											// ----------------------------------------------------------------

											this->Log() << "Irreducible spherical tensor operator basis (Rank 0 & Rank 2) was chosen - ops == 0" << std::endl;
											this->Log() << "Using magentic field to construct SingleSpin irreducible tensors." << std::endl;

											arma::vec static_field = (*interaction)->Field();
											arma::cx_vec complex_field;

											// Make field complex
											complex_field = arma::conv_to<arma::cx_vec>::from(static_field);

											// Rank 0 tensor with m=0
											if (!i->second->LRk0TensorT0((*s1), complex_field, *T0_rank_0))
											{
												this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
															<< "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=0
											if (!i->second->LRk2SphericalTensorT0((*s1), complex_field, *T0_rank_2))
											{
												this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
															<< "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=1
											if (!i->second->LRk2SphericalTensorTp1((*s1), complex_field, *Tp1))
											{
												this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and Field"
															<< "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=-1
											if (!i->second->LRk2SphericalTensorTm1((*s1), complex_field, *Tm1))
											{
												this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and Field"
															<< "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=2
											if (!i->second->LRk2SphericalTensorTp2((*s1), complex_field, *Tp2))
											{
												this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and Field"
															<< "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=-2
											if (!i->second->LRk2SphericalTensorTm2((*s1), complex_field, *Tm2))
											{
												this->Log() << "Failed to produce Tm2 spherical tensor between spin " << (*s1)->Name() << " and Field"
															<< "! Skipping." << std::endl;
												continue;
											}

											// Put all tensors on pointer array and get the number of indicies for subsequent loops	- adjust number when including rank 1 tensors
											num_op = 6;
											delete[] ptr_Tensors;
											ptr_Tensors = new arma::cx_mat *[num_op];
											ptr_Tensors[0] = T0_rank_0;
											ptr_Tensors[1] = T0_rank_2;
											ptr_Tensors[2] = Tm1;
											ptr_Tensors[3] = Tp1;
											ptr_Tensors[4] = Tm2;
											ptr_Tensors[5] = Tp2;

											// Construct spatial spherical Tensors
											SpinAPI::Tensor inTensor(0);
											if ((*interaction)->Properties()->Get("tensor", inTensor) && (*interaction)->Properties()->Get("coeff", coeff) && coeff == 1)
											{
												this->Log() << "Producing spatial shperical tensors with coupling tensor..." << std::endl;
												auto ATensor = (*interaction)->CouplingTensor();
												arma::cx_mat A(3, 3);
												arma::cx_vec Am(9);

												A.zeros();
												A = arma::conv_to<arma::cx_mat>::from((ATensor->LabFrame()));

												Am(0) = (1.0 / sqrt(3.0)) * (A(0, 0) + A(1, 1) + A(2, 2));
												Am(1) = (1.0 / sqrt(6.0)) * (3.0 * A(2, 2) - (A(0, 0) + A(1, 1) + A(2, 2)));
												Am(2) = 0.5 * (A(0, 2) + A(2, 0) + ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
												Am(3) = -0.5 * (A(0, 2) + A(2, 0) - ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
												Am(4) = 0.5 * (A(0, 0) - A(1, 1) - ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));
												Am(5) = 0.5 * (A(0, 0) - A(1, 1) + ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));

												// Rank 0
												*ptr_Tensors[0] = Am(0) * (*ptr_Tensors[0]);
												// Rank 2
												*ptr_Tensors[1] = Am(1) * (*ptr_Tensors[1]);
												*ptr_Tensors[2] = -Am(3) * (*ptr_Tensors[2]);
												*ptr_Tensors[3] = -Am(2) * (*ptr_Tensors[3]);
												*ptr_Tensors[4] = Am(4) * (*ptr_Tensors[4]);
												*ptr_Tensors[5] = Am(5) * (*ptr_Tensors[5]);
											}
											else
											{
												// Norm
												*ptr_Tensors[0] = (*ptr_Tensors[0]);
												*ptr_Tensors[1] = (*ptr_Tensors[1]);
												*ptr_Tensors[2] = -(*ptr_Tensors[2]);
												*ptr_Tensors[3] = -(*ptr_Tensors[3]);
												*ptr_Tensors[4] = (*ptr_Tensors[4]);
												*ptr_Tensors[5] = (*ptr_Tensors[5]);
											}
										}

// Rotate tensors in eigenbasis of H0
#pragma omp parallel for firstprivate(ptr_Tensors) shared(eigen_vec) num_threads(threads)
										for (k = 0; k < num_op; k++)
										{
											*ptr_Tensors[k] = (eigen_vec.t() * (*ptr_Tensors[k]) * eigen_vec);
										}

										// Number of elments for SpecDens. Important for delete statemant later
										if ((*interaction)->Properties()->Get("terms", terms) && terms == 0)
										{
											num_element = num_op * num_op;
										}
										else if ((*interaction)->Properties()->Get("terms", terms) && terms == 1)
										{
											num_element = num_op;
										}

										// Contructing dimension of ptr_SpecDens
										delete[] ptr_SpecDens;
										ptr_SpecDens = new arma::cx_mat *[num_element];

										for (int l = 0; l < num_element; l++)
										{
											ptr_SpecDens[l] = new arma::cx_mat(lambda);
											*ptr_SpecDens[l] *= 0.0;
										}

										this->Log() << "Calculating R tensor for:" << (*interaction)->Name() << " for spin " << (*s1)->Name() << std::endl;

										if ((*interaction)->Properties()->Get("terms", terms) && terms == 0 && ops == 1)
										{
											this->Log() << "Setting up NakajimaZwanzig tensor for each component (Axxxx, Axxxy, Axxxz, Axxyx, ...) " << std::endl;

											// Counter for inner loop
											int m = 0;
											// Counter to check if some rows are 0 and must not be computed
											double max_row_value;
											// Do the autocorrelation terms for each of the 9 terms in the tensor
											for (k = 0; k < num_op; k++)
											{
												for (s = 0; s < num_op; s++)
												{
													max_row_value = max(ampl_mat.row(m));

													// Check if the maximum value is zero
													if (max_row_value == 0.0)
													{
														this->Log() << "The maximum value of row (correlation function) " << m << " is zero." << std::endl;
													}
													else
													{
														for (int n = 0; n < (int)ampl_mat.n_cols; n++)
														{

															// ----------------------------------------------------------------
															// CONSTRUCTING SPECTRAL DENSITY MATRIX
															// ----------------------------------------------------------------

															SpecDens *= 0.0;

															if (!ConstructSpecDensSpecificTimeEvo(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
															{
																this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
																continue;
															}

															*ptr_SpecDens[m] += SpecDens;
														}

														// -----------------------------------------------------------------
														// CONSTRUCTING R MATRIX
														// -----------------------------------------------------------------

														tmp_R *= 0.0;

														if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[s]), *ptr_SpecDens[m], tmp_R))
														{
															this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
															continue;
														}

														R += tmp_R;
													}

													m = m + 1;
												}
											}
										}
										else if ((*interaction)->Properties()->Get("terms", terms) && terms == 1 && ops == 1)
										{
											this->Log() << "Setting up NakajimaZwanzig tensor just for each operator seperartely. No cross-correlation." << std::endl;

											// Counter for inner loop
											int m = 0;
											// Counter to check if some rows are 0 and must not be computed
											double max_row_value;

											// Do the autocorrelation terms for each of the 3 terms
											for (k = 0; k < num_op; k++)
											{
												max_row_value = max(ampl_mat.row(m));

												// Check if the maximum value is zero
												if (max_row_value == 0.0)
												{
													this->Log() << "The maximum value of row (correlation function) " << m << " is zero." << std::endl;
												}
												else
												{
													for (int n = 0; n < (int)ampl_mat.n_cols; n++)
													{
														// ----------------------------------------------------------------
														// CONSTRUCTING SPECTRAL DENSITY MATRIX
														// ----------------------------------------------------------------

														SpecDens *= 0.0;

														if (!ConstructSpecDensSpecificTimeEvo(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
														{
															this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
															continue;
														}

														*ptr_SpecDens[m] += SpecDens;
													}

													// -----------------------------------------------------------------
													// CONSTRUCTING R MATRIX
													// -----------------------------------------------------------------

													tmp_R *= 0.0;

													if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[k]), *ptr_SpecDens[m], tmp_R))
													{
														this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
														continue;
													}

													R += tmp_R;
												}

												m = m + 1;
											}
										}
										else
										{
											this->Log() << "Your current options are not implemented yet. Please ensure that you use Cartesian operators (ops = 1)." << std::endl;
										}
									}
								}
								else
								{
									// Get groups with respect to interaction
									auto group1 = (*interaction)->Group1();
									auto group2 = (*interaction)->Group2();

									// Loop through groups to get all interaction [group1 = electrons, group2 = particles electrons interact with, commonly nuclei]
									for (auto s1 = group1.cbegin(); s1 < group1.cend(); s1++)
									{
										for (auto s2 = group2.cbegin(); s2 < group2.cend(); s2++)
										{
											if ((*interaction)->Properties()->Get("ops", ops) && ops == 1)
											{

												// --------------------------------------------------------
												// CREATION OF SPIN OPERATORS (Sz, Sx, Sy)
												// --------------------------------------------------------

												this->Log() << "Sx, Sy and Sz operator basis was chosen - ops == 1" << std::endl;

												if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
												{
													return false;
												}

												if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
												{
													return false;
												}

												if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
												{
													return false;
												}

												if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sz()), *s2, *Sz2))
												{
													return false;
												}

												if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sx()), *s2, *Sx2))
												{
													return false;
												}

												if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sy()), *s2, *Sy2))
												{
													return false;
												}

												// Build double-spin operators
												*Sx1Sx2 = (*Sx1) * (*Sx2);
												*Sx1Sy2 = (*Sx1) * (*Sy2);
												*Sx1Sz2 = (*Sx1) * (*Sz2);
												*Sy1Sx2 = (*Sy1) * (*Sx2);
												*Sy1Sy2 = (*Sy1) * (*Sy2);
												*Sy1Sz2 = (*Sy1) * (*Sz2);
												*Sz1Sx2 = (*Sz1) * (*Sx2);
												*Sz1Sy2 = (*Sz1) * (*Sy2);
												*Sz1Sz2 = (*Sz1) * (*Sz2);

												// Put all tensors on pointer array
												num_op = 9;
												delete[] ptr_Tensors;
												ptr_Tensors = new arma::cx_mat *[num_op];

												ptr_Tensors[0] = Sx1Sx2;
												ptr_Tensors[1] = Sx1Sy2;
												ptr_Tensors[2] = Sx1Sz2;
												ptr_Tensors[3] = Sy1Sx2;
												ptr_Tensors[4] = Sy1Sy2;
												ptr_Tensors[5] = Sy1Sz2;
												ptr_Tensors[6] = Sz1Sx2;
												ptr_Tensors[7] = Sz1Sy2;
												ptr_Tensors[8] = Sz1Sz2;
											}
											else
											{
												// -------------------------------------------------------------------
												// Relaxation Matrix Construction for Interactions of Type::DoubleSpin
												// -------------------------------------------------------------------

												this->Log() << "Irreducible spherical tensor operator basis (Rank 0 & Rank 2) was chosen - ops == 0" << std::endl;

												// ----------------------------------------------------------------
												// CREATION OF IRREDUCIBLE SPHERICAL TENSORS
												// ----------------------------------------------------------------

												if (!i->second->BlRk0TensorT0(*s1, *s2, *T0_rank_0))
												{
													this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
													continue;
												}

												// Rank 2 tensor with m=0
												if (!i->second->BlRk2SphericalTensorT0(*s1, *s2, *T0_rank_2))
												{
													this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
													continue;
												}

												// Rank 2 tensor with m=1
												if (!i->second->BlRk2SphericalTensorTp1(*s1, *s2, *Tp1))
												{
													this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
													continue;
												}

												// Rank 2 tensor with m=-1
												if (!i->second->BlRk2SphericalTensorTm1(*s1, *s2, *Tm1))
												{
													this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
													continue;
												}

												// Rank 2 tensor with m=2
												if (!i->second->BlRk2SphericalTensorTp2(*s1, *s2, *Tp2))
												{
													this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
													continue;
												}

												// Rank 2 tensor with m=-2
												if (!i->second->BlRk2SphericalTensorTm2(*s1, *s2, *Tm2))
												{
													this->Log() << "Failed to produce Tm2 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
													continue;
												}

												// Put all tensors on pointer array and get the number of indicies for subsequent loops	- adjust number when including rank 1 tensors
												num_op = 6;
												delete[] ptr_Tensors;
												ptr_Tensors = new arma::cx_mat *[num_op];
												ptr_Tensors[0] = T0_rank_0;
												ptr_Tensors[1] = T0_rank_2;
												ptr_Tensors[2] = Tm1;
												ptr_Tensors[3] = Tp1;
												ptr_Tensors[4] = Tm2;
												ptr_Tensors[5] = Tp2;

												// Construct spatial spherical Tensors
												SpinAPI::Tensor inTensor(0);
												if ((*interaction)->Properties()->Get("tensor", inTensor) && (*interaction)->Properties()->Get("coeff", coeff) && coeff == 1)
												{
													this->Log() << "Producing spatial shperical tensors with coupling tensor..." << std::endl;
													auto ATensor = (*interaction)->CouplingTensor();
													arma::cx_mat A(3, 3);
													arma::cx_vec Am(6);

													A.zeros();
													A = arma::conv_to<arma::cx_mat>::from((ATensor->LabFrame()));

													Am(0) = (1.0 / sqrt(3.0)) * (A(0, 0) + A(1, 1) + A(2, 2));
													Am(1) = (1.0 / sqrt(6.0)) * (3.0 * A(2, 2) - (A(0, 0) + A(1, 1) + A(2, 2)));
													Am(2) = 0.5 * (A(0, 2) + A(2, 0) - ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
													Am(3) = -0.5 * (A(0, 2) + A(2, 0) + ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
													Am(4) = 0.5 * (A(0, 0) - A(1, 1) - ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));
													Am(5) = 0.5 * (A(0, 0) - A(1, 1) + ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));

													// Rank 0
													*ptr_Tensors[0] = Am(0) * (*ptr_Tensors[0]);
													// Rank 2
													*ptr_Tensors[1] = Am(1) * (*ptr_Tensors[1]);
													*ptr_Tensors[2] = -Am(3) * (*ptr_Tensors[2]);
													*ptr_Tensors[3] = -Am(2) * (*ptr_Tensors[3]);
													*ptr_Tensors[4] = Am(4) * (*ptr_Tensors[4]);
													*ptr_Tensors[5] = Am(5) * (*ptr_Tensors[5]);
												}
												else
												{
													// Norm
													*ptr_Tensors[0] = (*ptr_Tensors[0]);
													*ptr_Tensors[1] = (*ptr_Tensors[1]);
													*ptr_Tensors[2] = -(*ptr_Tensors[2]);
													*ptr_Tensors[3] = -(*ptr_Tensors[3]);
													*ptr_Tensors[4] = (*ptr_Tensors[4]);
													*ptr_Tensors[5] = (*ptr_Tensors[5]);
												}
											}

// Rotate tensors in eigenbasis of H0
#pragma omp parallel for firstprivate(ptr_Tensors) shared(eigen_vec) num_threads(threads)
											for (k = 0; k < num_op; k++)
											{
												*ptr_Tensors[k] = eigen_vec.t() * (*ptr_Tensors[k]) * eigen_vec;
											}

											// Number of elments for SpecDens. Important for delete statemant later
											if ((*interaction)->Properties()->Get("terms", terms) && terms == 0)
											{
												num_element = num_op * num_op;
											}
											else if ((*interaction)->Properties()->Get("terms", terms) && terms == 1)
											{
												num_element = num_op;
											}

											// Contructing dimension of ptr_SpecDens
											delete[] ptr_SpecDens;
											ptr_SpecDens = new arma::cx_mat *[num_element];

											for (int l = 0; l < num_element; l++)
											{
												ptr_SpecDens[l] = new arma::cx_mat(lambda);
												*ptr_SpecDens[l] *= 0.0;
											}

											this->Log() << "Calculating R tensor for:" << (*interaction)->Name() << " for spin " << (*s1)->Name() << " and spin " << (*s2)->Name() << std::endl;

											if ((*interaction)->Properties()->Get("terms", terms) && terms == 0 && ops == 1)
											{
												this->Log() << "Setting up NakajimaZwanzig tensor for each component (Axxxx, Axxxy, Axxxz, Axxyx, ...)" << std::endl;

												// Counter for inner loop
												int m = 0;
												// Counter to check if some rows are 0 and must not be computed
												double max_row_value;
												// Do the autocorrelation terms for each of the 9 terms in the tensor
												for (k = 0; k < num_op; k++)
												{
													for (s = 0; s < num_op; s++)
													{
														max_row_value = max(ampl_mat.row(m));

														// Check if the maximum value is zero
														if (max_row_value == 0.0)
														{
															this->Log() << "The maximum value of row (correlation function) " << m << " is zero." << std::endl;
														}
														else
														{
															for (int n = 0; n < (int)ampl_mat.n_cols; n++)
															{
																// ----------------------------------------------------------------
																// CONSTRUCTING SPECTRAL DENSITY MATRIX
																// ----------------------------------------------------------------

																SpecDens *= 0.0;

																if (!ConstructSpecDensSpecificTimeEvo(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
																{
																	this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
																	continue;
																}

																*ptr_SpecDens[m] += SpecDens;
															}

															// -----------------------------------------------------------------
															// CONSTRUCTING R MATRIX
															// -----------------------------------------------------------------

															tmp_R *= 0.0;

															if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[s]), *ptr_SpecDens[m], tmp_R))
															{
																this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
																continue;
															}

															R += tmp_R;
														}

														m = m + 1;
													}
												}
											}
											else if ((*interaction)->Properties()->Get("terms", terms) && terms == 1 && ops == 1)
											{
												this->Log() << "Setting up NakajimaZwanzig tensor just for each operator seperartely. No cross-correlation." << std::endl;

												// Counter for inner loop
												int m = 0;
												// Counter to check if some rows are 0 and must not be computed
												double max_row_value;

												// Do the autocorrelation terms for each of the 3 terms
												for (k = 0; k < num_op; k++)
												{
													max_row_value = max(ampl_mat.row(m));

													// Check if the maximum value is zero
													if (max_row_value == 0.0)
													{
														this->Log() << "The maximum value of row (correlation function) " << m << " is zero." << std::endl;
													}
													else
													{
														for (int n = 0; n < (int)ampl_mat.n_cols; n++)
														{
															// ----------------------------------------------------------------
															// CONSTRUCTING SPECTRAL DENSITY MATRIX
															// ----------------------------------------------------------------

															SpecDens *= 0.0;

															if (!ConstructSpecDensSpecificTimeEvo(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
															{
																this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
																continue;
															}

															*ptr_SpecDens[m] += SpecDens;
														}

														// -----------------------------------------------------------------
														// CONSTRUCTING R MATRIX
														// -----------------------------------------------------------------

														tmp_R *= 0.0;

														if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[k]), *ptr_SpecDens[m], tmp_R))
														{
															this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
															continue;
														}

														R += tmp_R;
													}

													m = m + 1;
												}
											}
											else
											{
												this->Log() << "Your current options are not implemented yet. Please ensure that you use Cartesian operators (ops = 1)." << std::endl;
											}

											this->Log() << "Added relaxation matrix term for interaction " << (*interaction)->Name() << " between spins " << (*s1)->Name() << " and " << (*s2)->Name() << "." << std::endl;
										}
									}
								}
							}
						}

						if (num_element == num_op * num_op)
						{
							for (int i = 0; i < num_op * num_op; i++)
							{
								delete ptr_SpecDens[i]; // Delete individual arma::cx_mat objects
							}
							delete[] ptr_SpecDens; // Delete the array of arma::cx_mat pointers
						}
						else if (num_element == num_op)
						{
							for (int i = 0; i < num_element; i++)
							{
								delete ptr_SpecDens[i]; // Delete individual arma::cx_mat objects
							}
							delete[] ptr_SpecDens; // Delete the array of arma::cx_mat pointers
						}
						else
						{
							// Handle the case when neither l is 9 nor 3, or provide appropriate error handling
							// ...
						}
					}
					// Normal loops without multiple multiexponential lists
					else if ((*interaction)->Properties()->GetList("tau_c", tau_c_list))
					{
						if ((*interaction)->Properties()->GetList("g", ampl_list))
						{
							if ((*interaction)->Type() == SpinAPI::InteractionType::SingleSpin)
							{
								// ------------------------------------------------------------------
								// Relaxation Matrix Construction for Interactions of Type::SingleSpin
								// ------------------------------------------------------------------

								// Get groups with respect to interaction
								auto group1 = (*interaction)->Group1();

								// Loop through groups to get all interaction type
								for (auto s1 = group1.cbegin(); s1 < group1.cend(); s1++)
								{
									// Which operator basis was chosen:
									if ((*interaction)->Properties()->Get("ops", ops) && ops == 1)
									{
										// --------------------------------------------------------
										// CREATION OF SPIN OPERATORS (Sz, Sx, Sy)
										// --------------------------------------------------------

										this->Log() << "Sx, Sy and Sz operator basis was chosen - ops == 1" << std::endl;

										if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
										{
											return false;
										}

										if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
										{
											return false;
										}

										if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
										{
											return false;
										}

										// Build double-spin operators
										*Sx1Sx2 = (*Sx1); // * Bx
										*Sx1Sy2 = (*Sx1); // * By
										*Sx1Sz2 = (*Sx1); // * Bz
										*Sy1Sx2 = (*Sy1); // * Bx
										*Sy1Sy2 = (*Sy1); // * By
										*Sy1Sz2 = (*Sy1); // * Bz
										*Sz1Sx2 = (*Sz1); // * Bx
										*Sz1Sy2 = (*Sz1); // * By
										*Sz1Sz2 = (*Sz1); // * Bz

										// Put all tensors on pointer array
										num_op = 9;
										delete[] ptr_Tensors;
										ptr_Tensors = new arma::cx_mat *[num_op];

										ptr_Tensors[0] = Sx1Sx2;
										ptr_Tensors[1] = Sx1Sy2;
										ptr_Tensors[2] = Sx1Sz2;
										ptr_Tensors[3] = Sy1Sx2;
										ptr_Tensors[4] = Sy1Sy2;
										ptr_Tensors[5] = Sy1Sz2;
										ptr_Tensors[6] = Sz1Sx2;
										ptr_Tensors[7] = Sz1Sy2;
										ptr_Tensors[8] = Sz1Sz2;

										// Bx,Bx (Sx1); Bx,By (Sx1); Bx,Bz (Sx1);By, Bz (Sx1);...; Bx,Bx (Sy1)
									}
									else
									{
										// ----------------------------------------------------------------
										// CREATION OF IRREDUCIBLE SPHERICAL TENSORS
										// ----------------------------------------------------------------

										this->Log() << "Irreducible spherical tensor operator basis (Rank 0 & Rank 2) was chosen - ops == 0" << std::endl;
										this->Log() << "Using magentic field to construct SingleSpin irreducible tensors." << std::endl;

										arma::vec static_field = (*interaction)->Field();
										arma::cx_vec complex_field;

										// Make field complex
										complex_field = arma::conv_to<arma::cx_vec>::from(static_field);

										// Rank 0 tensor with m=0
										if (!i->second->LRk0TensorT0((*s1), complex_field, *T0_rank_0))
										{
											this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=0
										if (!i->second->LRk2SphericalTensorT0((*s1), complex_field, *T0_rank_2))
										{
											this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=1
										if (!i->second->LRk2SphericalTensorTp1((*s1), complex_field, *Tp1))
										{
											this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=-1
										if (!i->second->LRk2SphericalTensorTm1((*s1), complex_field, *Tm1))
										{
											this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=2
										if (!i->second->LRk2SphericalTensorTp2((*s1), complex_field, *Tp2))
										{
											this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=-2
										if (!i->second->LRk2SphericalTensorTm2((*s1), complex_field, *Tm2))
										{
											this->Log() << "Failed to produce Tm2 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Put all tensors on pointer array and get the number of indicies for subsequent loops	- adjust number when including rank 1 tensors
										num_op = 6;
										delete[] ptr_Tensors;
										ptr_Tensors = new arma::cx_mat *[num_op];
										ptr_Tensors[0] = T0_rank_0;
										ptr_Tensors[1] = T0_rank_2;
										ptr_Tensors[2] = Tm1;
										ptr_Tensors[3] = Tp1;
										ptr_Tensors[4] = Tm2;
										ptr_Tensors[5] = Tp2;

										// Construct spatial spherical Tensors
										SpinAPI::Tensor inTensor(0);
										if ((*interaction)->Properties()->Get("tensor", inTensor) && (*interaction)->Properties()->Get("coeff", coeff) && coeff == 1)
										{
											this->Log() << "Producing spatial shperical tensors with coupling tensor..." << std::endl;
											auto ATensor = (*interaction)->CouplingTensor();
											arma::cx_mat A(3, 3);
											arma::cx_vec Am(9);

											A.zeros();
											A = arma::conv_to<arma::cx_mat>::from((ATensor->LabFrame()));

											Am(0) = (1.0 / sqrt(3.0)) * (A(0, 0) + A(1, 1) + A(2, 2));
											Am(1) = (1.0 / sqrt(6.0)) * (3.0 * A(2, 2) - (A(0, 0) + A(1, 1) + A(2, 2)));
											Am(2) = 0.5 * (A(0, 2) + A(2, 0) + ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
											Am(3) = -0.5 * (A(0, 2) + A(2, 0) - ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
											Am(4) = 0.5 * (A(0, 0) - A(1, 1) - ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));
											Am(5) = 0.5 * (A(0, 0) - A(1, 1) + ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));

											// Rank 0
											*ptr_Tensors[0] = Am(0) * (*ptr_Tensors[0]);
											// Rank 2
											*ptr_Tensors[1] = Am(1) * (*ptr_Tensors[1]);
											*ptr_Tensors[2] = Am(3) * (*ptr_Tensors[2]);
											*ptr_Tensors[3] = Am(2) * (*ptr_Tensors[3]);
											*ptr_Tensors[4] = Am(4) * (*ptr_Tensors[4]);
											*ptr_Tensors[5] = Am(5) * (*ptr_Tensors[5]);
										}
										else
										{
											// Norm
											*ptr_Tensors[0] = (*ptr_Tensors[0]);
											*ptr_Tensors[1] = (*ptr_Tensors[1]);
											*ptr_Tensors[2] = -(*ptr_Tensors[2]);
											*ptr_Tensors[3] = -(*ptr_Tensors[3]);
											*ptr_Tensors[4] = (*ptr_Tensors[4]);
											*ptr_Tensors[5] = (*ptr_Tensors[5]);
										}
									}

									// Rotate tensors in eigenbasis of H0
#pragma omp parallel for firstprivate(ptr_Tensors) shared(eigen_vec) num_threads(threads)
									for (k = 0; k < num_op; k++)
									{
										*ptr_Tensors[k] = (eigen_vec.t() * (*ptr_Tensors[k]) * eigen_vec);
									}

									this->Log() << "Calculating R tensor for:" << (*interaction)->Name() << " for spin " << (*s1)->Name() << std::endl;

									if ((*interaction)->Properties()->Get("terms", terms) && terms == 1)
									{
										this->Log() << "No cross-relaxation terms are requested" << std::endl;

										if ((*interaction)->Properties()->Get("def_g", def_g) && def_g == 1)
										{
											this->Log() << "Setting J up for each operator separatly - def_g == 1 " << std::endl;

// loop through all operators and construct R tensor in Liouville space
#pragma omp parallel for num_threads(threads)
											for (int l = 0; l < threads; l++)
											{
												*ptr_R[l] *= 0.0;
											}

#pragma omp parallel for firstprivate(tmp_R, SpecDens, ampl_combined) shared(ampl_list, tau_c_list, num_op, ptr_Tensors, ptr_R, lambda, interaction) num_threads(threads)
											for (k = 0; k < num_op; k++)
											{
												// ----------------------------------------------------------------
												// CONSTRUCTING SPECTRAL DENSITY MATRIX
												// ----------------------------------------------------------------
												ampl_combined = ampl_list[k] * ampl_list[k];

												SpecDens *= 0.0;

												if (!ConstructSpecDensSpecificTimeEvo(static_cast<std::complex<double>>(ampl_combined), static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
												{
													this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
													continue;
												}

												// -----------------------------------------------------------------
												// CONSTRUCTING R MATRIX
												// -----------------------------------------------------------------

												tmp_R *= 0.0;

												if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
												{
													this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
													continue;
												}

												*ptr_R[omp_get_thread_num()] += tmp_R;
											}
										}
										else
										{
											this->Log() << "J is generally constructed for all operators - def_g == 0 " << std::endl;

											// ----------------------------------------------------------------
											// CONSTRUCTING SPECTRAL DENSITY MATRIX
											// ----------------------------------------------------------------
											SpecDens *= 0.0;

											if (!ConstructSpecDensGeneralTimeEvo(ampl_list, tau_c_list, lambda, SpecDens))
											{
												this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
												continue;
											}

#pragma omp parallel for num_threads(threads)
											for (l = 0; l < threads; l++)
											{
												*ptr_R[l] *= 0.0;
											}

#pragma omp parallel for firstprivate(tmp_R) shared(ampl_list, tau_c_list, SpecDens, num_op, ptr_R, ptr_Tensors, eigen_vec) num_threads(threads)
											for (k = 0; k < num_op; k++)
											{
												// -----------------------------------------------------------------
												// CONSTRUCTING R MATRIX
												// -----------------------------------------------------------------

												tmp_R *= 0.0;

												if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
												{
													this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
													continue;
												}

												*ptr_R[omp_get_thread_num()] += tmp_R;
											}
										}
									}
									else
									{
										if ((*interaction)->Properties()->Get("def_g", def_g) && def_g == 1)
										{
											this->Log() << "Setting J up for each operator separatly - def_g == 1 " << std::endl;

#pragma omp parallel for num_threads(threads)
											for (l = 0; l < threads; l++)
											{
												*ptr_R[l] *= 0.0;
											}

// loop through all operators and construct R tensor in Liouville space
#pragma omp parallel for collapse(2) firstprivate(tmp_R, SpecDens, ampl_combined) shared(ampl_list, tau_c_list, num_op, ptr_Tensors, ptr_R, interaction) num_threads(threads)
											for (k = 0; k < num_op; k++)
											{
												for (s = 0; s < num_op; s++)
												{
													// ----------------------------------------------------------------
													// CONSTRUCTING SPECTRAL DENSITY MATRIX
													// ----------------------------------------------------------------

													ampl_combined = ampl_list[k] * ampl_list[s];

													SpecDens *= 0.0;
													if (!ConstructSpecDensSpecificTimeEvo(ampl_combined, static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
													{
														this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
														continue;
													}

													// ----------------------------------------------------------------
													// CONSTRUCTING R MATRIX
													// ----------------------------------------------------------------

													tmp_R *= 0.0;

													if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
													{
														this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
														continue;
													}

													*ptr_R[omp_get_thread_num()] += tmp_R;
												}
											}
										}
										else
										{
											// loop through all operators and construct R tensor in Liouville space
											// ----------------------------------------------------------------
											// CONSTRUCTING SPECTRAL DENSITY MATRIX
											// ----------------------------------------------------------------

											this->Log() << "J is generally constructed for all operators - def_g == 0 " << std::endl;
											SpecDens *= 0.0;

											if (!ConstructSpecDensGeneralTimeEvo(ampl_list, tau_c_list, lambda, SpecDens))
											{
												this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
												continue;
											}

#pragma omp parallel for num_threads(threads)
											for (l = 0; l < threads; l++)
											{
												*ptr_R[l] *= 0.0;
											}

#pragma omp parallel for collapse(2) firstprivate(tmp_R) shared(ampl_list, tau_c_list, SpecDens, num_op, ptr_R, ptr_Tensors) num_threads(threads)
											for (k = 0; k < num_op; k++)
											{
												for (s = 0; s < num_op; s++)
												{
													// ----------------------------------------------------------------
													// CONSTRUCTING R MATRIX
													// ----------------------------------------------------------------

													tmp_R *= 0.0;

													if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
													{
														this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
														continue;
													}

													*ptr_R[omp_get_thread_num()] += tmp_R;
												}
											}
										}
									}

									for (l = 0; l < threads; l++)
									{
										R += *ptr_R[l];
									}

									this->Log() << "Added relaxation matrix term for interaction " << (*interaction)->Name() << " of spin " << (*s1)->Name() << "." << std::endl;
								}
							}
							else
							{
								// Get groups with respect to interaction
								auto group1 = (*interaction)->Group1();
								auto group2 = (*interaction)->Group2();

								// Loop through groups to get all interaction [group1 = electrons, group2 = particles electrons interact with, commonly nuclei]
								for (auto s1 = group1.cbegin(); s1 < group1.cend(); s1++)
								{
									for (auto s2 = group2.cbegin(); s2 < group2.cend(); s2++)
									{
										if ((*interaction)->Properties()->Get("ops", ops) && ops == 1)
										{

											// --------------------------------------------------------
											// CREATION OF SPIN OPERATORS (Sz, Sx, Sy)
											// --------------------------------------------------------

											this->Log() << "Sx, Sy and Sz operator basis was chosen - ops == 1" << std::endl;

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sz()), *s2, *Sz2))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sx()), *s2, *Sx2))
											{
												return false;
											}

											if (!i->second->CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sy()), *s2, *Sy2))
											{
												return false;
											}

											// Build double-spin operators
											*Sx1Sx2 = ((*Sx1) * (*Sx2));
											*Sx1Sy2 = ((*Sx1) * (*Sy2));
											*Sx1Sz2 = ((*Sx1) * (*Sz2));
											*Sy1Sx2 = ((*Sy1) * (*Sx2));
											*Sy1Sy2 = ((*Sy1) * (*Sy2));
											*Sy1Sz2 = ((*Sy1) * (*Sz2));
											*Sz1Sx2 = ((*Sz1) * (*Sx2));
											*Sz1Sy2 = ((*Sz1) * (*Sy2));
											*Sz1Sz2 = ((*Sz1) * (*Sz2));

											// Put all tensors on pointer array
											num_op = 9;
											delete[] ptr_Tensors;
											ptr_Tensors = new arma::cx_mat *[num_op];

											ptr_Tensors[0] = Sx1Sx2;
											ptr_Tensors[1] = Sx1Sy2;
											ptr_Tensors[2] = Sx1Sz2;
											ptr_Tensors[3] = Sy1Sx2;
											ptr_Tensors[4] = Sy1Sy2;
											ptr_Tensors[5] = Sy1Sz2;
											ptr_Tensors[6] = Sz1Sx2;
											ptr_Tensors[7] = Sz1Sy2;
											ptr_Tensors[8] = Sz1Sz2;
										}
										else
										{
											// -------------------------------------------------------------------
											// Relaxation Matrix Construction for Interactions of Type::DoubleSpin
											// -------------------------------------------------------------------

											this->Log() << "Irreducible spherical tensor operator basis (Rank 0 & Rank 2) was chosen - ops == 0" << std::endl;

											// ----------------------------------------------------------------
											// CREATION OF IRREDUCIBLE SPHERICAL TENSORS
											// ----------------------------------------------------------------

											if (!i->second->BlRk0TensorT0(*s1, *s2, *T0_rank_0))
											{
												this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=0
											if (!i->second->BlRk2SphericalTensorT0(*s1, *s2, *T0_rank_2))
											{
												this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=1
											if (!i->second->BlRk2SphericalTensorTp1(*s1, *s2, *Tp1))
											{
												this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=-1
											if (!i->second->BlRk2SphericalTensorTm1(*s1, *s2, *Tm1))
											{
												this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=2
											if (!i->second->BlRk2SphericalTensorTp2(*s1, *s2, *Tp2))
											{
												this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=-2
											if (!i->second->BlRk2SphericalTensorTm2(*s1, *s2, *Tm2))
											{
												this->Log() << "Failed to produce Tm2 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Put all tensors on pointer array and get the number of indicies for subsequent loops	- adjust number when including rank 1 tensors
											num_op = 6;
											delete[] ptr_Tensors;
											ptr_Tensors = new arma::cx_mat *[num_op];
											ptr_Tensors[0] = T0_rank_0;
											ptr_Tensors[1] = T0_rank_2;
											ptr_Tensors[2] = Tm1;
											ptr_Tensors[3] = Tp1;
											ptr_Tensors[4] = Tm2;
											ptr_Tensors[5] = Tp2;

											// Construct spatial spherical Tensors
											SpinAPI::Tensor inTensor(0);
											if ((*interaction)->Properties()->Get("tensor", inTensor) && (*interaction)->Properties()->Get("coeff", coeff) && coeff == 1)
											{
												this->Log() << "Producing spatial shperical tensors with coupling tensor..." << std::endl;
												auto ATensor = (*interaction)->CouplingTensor();
												arma::cx_mat A(3, 3);
												arma::cx_vec Am(9);

												A.zeros();
												A = arma::conv_to<arma::cx_mat>::from((ATensor->LabFrame()));

												Am(0) = (1.0 / sqrt(3.0)) * (A(0, 0) + A(1, 1) + A(2, 2));
												Am(1) = (1.0 / sqrt(6.0)) * (3.0 * A(2, 2) - (A(0, 0) + A(1, 1) + A(2, 2)));
												Am(2) = 0.5 * (A(0, 2) + A(2, 0) + ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
												Am(3) = -0.5 * (A(0, 2) + A(2, 0) - ((arma::cx_double(0.0, 1.0)) * (A(1, 2) + A(2, 1))));
												Am(4) = 0.5 * (A(0, 0) - A(1, 1) - ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));
												Am(5) = 0.5 * (A(0, 0) - A(1, 1) + ((arma::cx_double(0.0, 1.0)) * (A(0, 1) + A(1, 0))));

												// Rank 0
												*ptr_Tensors[0] = Am(0) * (*ptr_Tensors[0]);
												// Rank 2
												*ptr_Tensors[1] = Am(1) * (*ptr_Tensors[1]);
												*ptr_Tensors[2] = Am(3) * (*ptr_Tensors[2]);
												*ptr_Tensors[3] = Am(2) * (*ptr_Tensors[3]);
												*ptr_Tensors[4] = Am(4) * (*ptr_Tensors[4]);
												*ptr_Tensors[5] = Am(5) * (*ptr_Tensors[5]);
											}
											else
											{
												// Norm
												*ptr_Tensors[0] = (*ptr_Tensors[0]);
												*ptr_Tensors[1] = (*ptr_Tensors[1]);
												*ptr_Tensors[2] = -(*ptr_Tensors[2]);
												*ptr_Tensors[3] = -(*ptr_Tensors[3]);
												*ptr_Tensors[4] = (*ptr_Tensors[4]);
												*ptr_Tensors[5] = (*ptr_Tensors[5]);
											}
										}

										// Rotate tensors in eigenbasis of H0
#pragma omp parallel for firstprivate(ptr_Tensors) shared(eigen_vec) num_threads(threads)
										for (k = 0; k < num_op; k++)
										{
											*ptr_Tensors[k] = (eigen_vec.t() * (*ptr_Tensors[k]) * eigen_vec);
										}

										this->Log() << "Calculating R tensor for:" << (*interaction)->Name() << " for spin " << (*s1)->Name() << std::endl;

										if ((*interaction)->Properties()->Get("terms", terms) && terms == 1)
										{
											this->Log() << "No cross-relaxation terms are requested" << std::endl;

											if ((*interaction)->Properties()->Get("def_g", def_g) && def_g == 1)
											{
												this->Log() << "Setting J up for each operator separatly - def_g == 1 " << std::endl;

// loop through all operators and construct R tensor in Liouville space
#pragma omp parallel for num_threads(threads)
												for (int l = 0; l < threads; l++)
												{
													*ptr_R[l] *= 0.0;
												}

#pragma omp parallel for firstprivate(tmp_R, SpecDens, ampl_combined) shared(ampl_list, tau_c_list, num_op, ptr_Tensors, ptr_R, interaction) num_threads(threads)
												for (k = 0; k < num_op; k++)
												{
													// ----------------------------------------------------------------
													// CONSTRUCTING SPECTRAL DENSITY MATRIX
													// ----------------------------------------------------------------
													ampl_combined = ampl_list[k] * ampl_list[k];

													SpecDens *= 0.0;

													if (!ConstructSpecDensSpecificTimeEvo(static_cast<std::complex<double>>(ampl_combined), static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
													{
														this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
														continue;
													}

													// -----------------------------------------------------------------
													// CONSTRUCTING R MATRIX
													// -----------------------------------------------------------------

													tmp_R *= 0.0;

													if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
													{
														this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
														continue;
													}

													*ptr_R[omp_get_thread_num()] += tmp_R;
												}
											}
											else
											{
												this->Log() << "J is generally constructed for all operators - def_g == 0 " << std::endl;

												// ----------------------------------------------------------------
												// CONSTRUCTING SPECTRAL DENSITY MATRIX
												// ----------------------------------------------------------------

												SpecDens *= 0.0;

												if (!ConstructSpecDensGeneralTimeEvo(ampl_list, tau_c_list, lambda, SpecDens))
												{
													this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
													continue;
												}

#pragma omp parallel for num_threads(threads)
												for (l = 0; l < threads; l++)
												{
													*ptr_R[l] *= 0.0;
												}

#pragma omp parallel for firstprivate(tmp_R) shared(ampl_list, tau_c_list, SpecDens, num_op, ptr_R, ptr_Tensors) num_threads(threads)
												for (k = 0; k < num_op; k++)
												{
													// -----------------------------------------------------------------
													// CONSTRUCTING R MATRIX
													// -----------------------------------------------------------------

													tmp_R *= 0.0;

													if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
													{
														this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
														continue;
													}

													*ptr_R[omp_get_thread_num()] += tmp_R;
												}
											}
										}
										else
										{
											if ((*interaction)->Properties()->Get("def_g", def_g) && def_g == 1)
											{
												this->Log() << "Setting J up for each operator separatly - def_g == 1 " << std::endl;

#pragma omp parallel for num_threads(threads)
												for (l = 0; l < threads; l++)
												{
													*ptr_R[l] *= 0.0;
												}

// loop through all operators and construct R tensor in Liouville space
#pragma omp parallel for collapse(2) firstprivate(tmp_R, SpecDens, ampl_combined) shared(ampl_list, tau_c_list, num_op, ptr_Tensors, ptr_R, interaction) num_threads(threads)
												for (k = 0; k < num_op; k++)
												{
													for (s = 0; s < num_op; s++)
													{
														// ----------------------------------------------------------------
														// CONSTRUCTING SPECTRAL DENSITY MATRIX
														// ----------------------------------------------------------------

														ampl_combined = ampl_list[k] * ampl_list[s];

														SpecDens *= 0.0;

														if (!ConstructSpecDensSpecificTimeEvo(ampl_combined, static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
														{
															this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
															continue;
														}

														// ----------------------------------------------------------------
														// CONSTRUCTING R MATRIX
														// ----------------------------------------------------------------

														tmp_R *= 0.0;

														if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
														{
															this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
															continue;
														}

														*ptr_R[omp_get_thread_num()] += tmp_R;
													}
												}
											}
											else
											{
												// loop through all operators and construct R tensor in Liouville space
												// ----------------------------------------------------------------
												// CONSTRUCTING SPECTRAL DENSITY MATRIX
												// ----------------------------------------------------------------

												this->Log() << "J is generally constructed for all operators - def_g == 0 " << std::endl;

												SpecDens *= 0.0;

												if (!ConstructSpecDensGeneralTimeEvo(ampl_list, tau_c_list, lambda, SpecDens))
												{
													this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
													continue;
												}

#pragma omp parallel for num_threads(threads)
												for (l = 0; l < threads; l++)
												{
													*ptr_R[l] *= 0.0;
												}

#pragma omp parallel for collapse(2) firstprivate(tmp_R) shared(ampl_list, tau_c_list, SpecDens, num_op, ptr_R, ptr_Tensors) num_threads(threads)
												for (k = 0; k < num_op; k++)
												{
													for (s = 0; s < num_op; s++)
													{
														// ----------------------------------------------------------------
														// CONSTRUCTING R MATRIX
														// ----------------------------------------------------------------

														tmp_R *= 0.0;

														if (!NakajimaZwanzigtensorTimeEvo((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
														{
															this->Log() << "There are problems with the construction of the NakajimaZwanzig tensor - Please check your input." << std::endl;
															continue;
														}

														*ptr_R[omp_get_thread_num()] += tmp_R;
													}
												}
											}
										}

										for (l = 0; l < threads; l++)
										{
											R += *ptr_R[l];
										}

										this->Log() << "Added relaxation matrix term for interaction " << (*interaction)->Name() << " of spin " << (*s1)->Name() << " and " << (*s2)->Name() << "." << std::endl;
									}
								}
							}
						}
					}
					else
					{
						this->Log() << "No relaxation matrix term requested for interaction " << (*interaction)->Name() << "." << std::endl;
					}

					std::fill(ampl_list.begin(), ampl_list.end(), 0);
					std::fill(tau_c_list.begin(), tau_c_list.end(), 0);
					terms *= 0.0;
					SpecDens *= 0.0;
				}

				for (int l = 0; l < threads; l++)
				{
					delete ptr_R[l];
				}
				delete[] ptr_R;

				for (int l = 0; l < num_op; l++)
				{
					delete ptr_Tensors[l];
				}
				delete[] ptr_Tensors;

				// ---------------------------------------------------------------
				// SETUP COMPLETE HAMILTONIAN
				// ---------------------------------------------------------------
				// Transform  H0 into superspace
				arma::cx_mat lhs;
				arma::cx_mat rhs;
				arma::cx_mat H_SS;

				// Transforming into superspace
				i->second->SuperoperatorFromLeftOperator(eig_val_mat, lhs);
				i->second->SuperoperatorFromRightOperator(eig_val_mat, rhs);

				H_SS = lhs - rhs;

				// Conversion into Liouville space of static Hamiltonian
				L.submat(nextDimension, nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) = arma::cx_double(0.0, -1.0) * H_SS;

				// Then get the reaction operators
				arma::cx_mat K;
				if (!i->second->TotalReactionOperator(K))
				{
					this->Log() << "ERROR: Failed to obtain matrix representation of the reaction operators for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				// Rotation into eigenbasis of H0
				K = eigen_vec.t() * K * eigen_vec;

				// Transform ReactionOperator into superspace
				arma::cx_mat Klhs;
				arma::cx_mat Krhs;
				arma::cx_mat K_SS;

				i->second->SuperoperatorFromLeftOperator(K, Klhs);
				i->second->SuperoperatorFromRightOperator(K, Krhs);

				K_SS = Klhs + Krhs;

				// Conversion into Liouville space of reaction operator
				L.submat(nextDimension, nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) -= K_SS;

				// Get the relaxation terms of other relaxation operators, assuming that they can just be added
				arma::cx_mat O_SS;

				for (auto t = i->first->operators_cbegin(); t != i->first->operators_cend(); t++)
				{
					i->second->UseSuperoperatorSpace(true);
					if (i->second->RelaxationOperatorFrameChange((*t), eigen_vec, O_SS))
					{

						i->second->UseSuperoperatorSpace(false);
						L.submat(nextDimension, nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) += O_SS;

						this->Log() << "Added other relaxation operator \"" << (*t)->Name() << "\" to the Liouvillian.\n";
					}
					else
					{
						this->Log() << "There is a problem with operator \"" << (*t)->Name() << ". Please check.\n";
						i->second->UseSuperoperatorSpace(false);
					}
				}

				// Include R tensor in Liouville operator
				L.submat(nextDimension, nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1) += R;
			}

			// Obtain the creation operators - note that we need to loop through the other SpinSystems again to find transitions leading into the current SpinSystem
			unsigned int nextCDimension = 0; // Similar to nextDimension, but to keep track of first dimension for this other SpinSystem
			for (auto j = spaces.cbegin(); j < spaces.cend(); j++)
			{
				// Creation operators are off-diagonal elements
				if (j != i)
				{
					// Check all transitions whether they should produce a creation operator
					for (auto t = j->first->transitions_cbegin(); t < j->first->transitions_cend(); t++)
					{
						// Does the Transition lead into the current spin space?
						if ((*t)->Target() == i->first)
						{
							// Calculate the index for the j'th element
							size_t j_index = std::distance(spaces.cbegin(), j);

							// Prepare a creation operator
							arma::sp_cx_mat C;
							arma::sp_cx_mat C_SS;

							// Get a Hilbert space state vector of both the source state and the target state
							arma::cx_vec S;
							arma::cx_vec T;
							j->second->GetState((*t)->SourceState(), S);
							i->second->GetState((*t)->TargetState(), T);

							// Make sure the states are normalized
							S /= arma::norm(S);
							T /= arma::norm(T);

							// Rotate NakajimaZwanzig influenced subspace creaction operator into correct frame
							if (j_index < eigveclist.size())
							{
								arma::cx_mat &eigvec_j = eigveclist[j_index];
								S = eigvec_j.t() * S;
							}

							size_t i_index = std::distance(spaces.cbegin(), i);
							if (i_index < eigveclist.size())
							{
								T = eigveclist[i_index].t() * T;
							}

							// Obtain the creation operator
							C = arma::conv_to<arma::sp_cx_mat>::from(T * S.t());

							// Transform into SuperSpace
							C_SS = arma::kron(C, C.t().st());

							// Put it into the total Liouvillian:
							//  - The row should be that of the current spin space (the target space)
							//  - The column should be that of the source spin space (the spin system containing the Transition object)
							L.submat(nextDimension, nextCDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1, nextCDimension + (j->second->SpaceDimensions() * j->second->SpaceDimensions()) - 1) += C_SS * (*t)->Rate();
						}
					}
				}

				// Move on to check next spin system for transitions into the current spin space
				nextCDimension += (j->second->SpaceDimensions() * j->second->SpaceDimensions());
			}

			// Move on to next spin space
			nextDimension += (i->second->SpaceDimensions() * i->second->SpaceDimensions());
		}

		// Write results for initial state as well (i.e. at time 0)
		this->Data() << this->RunSettings()->CurrentStep() << " 0 ";
		this->WriteStandardOutput(this->Data());
		nextDimension = 0;

		for (auto i = spaces.cbegin(); i < spaces.cend(); i++)
		{
			// Get the superspace result vector and convert it back to the native Hilbert space
			arma::cx_mat rho_result;
			arma::cx_vec rho_result_vec;
			rho_result_vec = rho0.rows(nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1);
			if (!i->second->OperatorFromSuperspace(rho_result_vec, rho_result))
			{
				this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
				return false;
			}

			// Loop through all states
			arma::cx_mat Px;
			auto sstates = i->first->States();
			for (auto j = sstates.cbegin(); j < sstates.cend(); j++)
			{
				if (!i->second->GetState((*j), Px))
				{
					this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << i->first->Name() << "\"." << std::endl;
					continue;
				}

				// Calculate the index for the j'th element
				size_t i_index = std::distance(spaces.cbegin(), i);

				// Rotate NakajimaZwanzig influenced subspace creation operator into correct frame
				if (i_index < eigveclist.size())
				{
					arma::cx_mat &eigvec_i = eigveclist[i_index];
					Px = eigvec_i.t() * Px * eigvec_i;
				}

				// Return the yield for this state - note that no reaction rates are included here.
				this->Data() << std::abs(arma::trace(Px * rho_result)) << " ";
			}

			// Move on to next spin space
			nextDimension += (i->second->SpaceDimensions() * i->second->SpaceDimensions());
		}
		this->Data() << std::endl;

		// We need the propagator
		this->Log() << "Calculating the propagator..." << std::endl;
		arma::cx_mat P = arma::expmat(arma::conv_to<arma::cx_mat>::from(L) * this->timestep);

		// Perform the calculation
		this->Log() << "Ready to perform calculation." << std::endl;
		unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
		for (unsigned int n = 1; n <= steps; n++)
		{
			// Write first part of the data output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << (static_cast<double>(n) * this->timestep) << " ";
			this->WriteStandardOutput(this->Data());

			// Propagate (use special scope to be able to dispose of the temporary vector asap)
			{
				arma::cx_vec tmp = P * rho0;
				rho0 = tmp;
			}

			// Retrieve the resulting density matrix for each spin system and output the results
			nextDimension = 0;
			for (auto i = spaces.cbegin(); i < spaces.cend(); i++)
			{
				// Get the superspace result vector and convert it back to the native Hilbert space
				arma::cx_mat rho_result;
				arma::cx_vec rho_result_vec;
				rho_result_vec = rho0.rows(nextDimension, nextDimension + (i->second->SpaceDimensions() * i->second->SpaceDimensions()) - 1);
				if (!i->second->OperatorFromSuperspace(rho_result_vec, rho_result))
				{
					this->Log() << "ERROR: Failed to convert resulting superspace-vector back to native Hilbert space for spin system \"" << i->first->Name() << "\"!" << std::endl;
					return false;
				}

				// Loop through all states
				arma::cx_mat Px;
				auto sstates = i->first->States();
				for (auto j = sstates.cbegin(); j < sstates.cend(); j++)
				{
					if (!i->second->GetState((*j), Px))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << i->first->Name() << "\"." << std::endl;
						continue;
					}

					// Calculate the index for the j'th element
					size_t i_index = std::distance(spaces.cbegin(), i);

					// Rotate NakajimaZwanzig influenced subspace creation operator into correct frame
					if (i_index < eigveclist.size())
					{
						arma::cx_mat &eigvec_i = eigveclist[i_index];
						Px = eigvec_i.t() * Px * eigvec_i;
					}

					// Return the yield for this state - note that no reaction rates are included here.
					this->Data() << std::abs(arma::trace(Px * rho_result)) << " ";
				}

				// Move on to next spin space
				nextDimension += (i->second->SpaceDimensions() * i->second->SpaceDimensions());
			}

			// Terminate the line in the data file after iteration through all spin systems
			this->Data() << std::endl;
		}

		this->Log() << "Done with calculation." << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskMultiStaticSSNakajimaZwanzigTimeEvo::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time(ns) ";
		this->WriteStandardOutputHeader(_stream);

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i < systems.cend(); i++)
		{
			// Write each state name
			auto states = (*i)->States();
			for (auto j = states.cbegin(); j < states.cend(); j++)
				_stream << (*i)->Name() << "." << (*j)->Name() << " ";
		}
		_stream << std::endl;
	}

	// Validation of the required input
	bool TaskMultiStaticSSNakajimaZwanzigTimeEvo::Validate()
	{
		double inputTimestep = 0.0;
		double inputTotaltime = 0.0;

		// Get timestep
		if (this->Properties()->Get("timestep", inputTimestep))
		{
			if (std::isfinite(inputTimestep) && inputTimestep > 0.0)
			{
				this->timestep = inputTimestep;
			}
			else
			{
				// We can run the calculation if an invalid timestep was specified
				return false;
			}
		}

		// Get totaltime
		if (this->Properties()->Get("totaltime", inputTotaltime))
		{
			if (std::isfinite(inputTotaltime) && inputTotaltime > 0.0)
			{
				this->totaltime = inputTotaltime;
			}
			else
			{
				// We can run the calculation if an invalid total time was specified
				return false;
			}
		}

		// Get the reacton operator type
		std::string str;
		if (this->Properties()->Get("reactionoperators", str))
		{
			if (str.compare("haberkorn") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Haberkorn;
				this->Log() << "Setting reaction operator type to Haberkorn." << std::endl;
			}
			else if (str.compare("lindblad") == 0)
			{
				this->reactionOperators = SpinAPI::ReactionOperatorType::Lindblad;
				this->Log() << "Setting reaction operator type to Lindblad." << std::endl;
			}
			else
			{
				this->Log() << "Warning: Unknown reaction operator type specified. Using default reaction operators." << std::endl;
			}
		}

		return true;
	}

	// Construction NZ tensor
	bool TaskMultiStaticSSNakajimaZwanzigTimeEvo::NakajimaZwanzigtensorTimeEvo(const arma::cx_mat &_op1, const arma::cx_mat &_op2, const arma::cx_mat &_specdens, arma::cx_mat &_NakajimaZwanzigtensor)
	{
		_NakajimaZwanzigtensor *= 0.0;

		// -1 *_op1.t() * _eigenvec * _specdens * _eigenvec.t() * _op2
		// J. Chem. Phys. 154, 084121 (2021) https://doi.org/10.1063/5.0040519

		arma::cx_mat _op1_SS = _specdens;
		arma::cx_mat _op2_SS = _specdens;
		arma::cx_mat one = arma::eye<arma::cx_mat>(arma::size(_op1));

		_op1_SS = arma::kron((_op1).t(), one) - arma::kron(one, (_op1.t()).st());
		_op2_SS = arma::kron((_op2), one) - arma::kron(one, (_op2).st());

		_NakajimaZwanzigtensor = arma::cx_double(-1.00, 0.00) * _op1_SS * _specdens.t() * _op2_SS;

		return true;
	}

	bool TaskMultiStaticSSNakajimaZwanzigTimeEvo::ConstructSpecDensGeneralTimeEvo(const std::vector<double> &_ampl_list, const std::vector<double> &_tau_c_list, const arma::cx_mat &_omega, arma::cx_mat &_specdens)
	{
		// Solution of spectral density: S = Ampl/(1/tau_c - i * omega)
		arma::cx_vec spectral_entries = _omega.diag();
		spectral_entries.zeros();

		for (auto ii = 0; ii < (int)_omega.n_cols; ii++)
		{
			for (auto jj = 0; jj < (int)_tau_c_list.size(); jj++)
			{
				spectral_entries(ii) += (static_cast<std::complex<double>>(_ampl_list[jj])) / ((1.00 / (static_cast<std::complex<double>>(_tau_c_list[jj]))) + arma::cx_double(0.0, -1.00) * _omega(ii, ii));
			}
		}

		_specdens = arma::diagmat(arma::conv_to<arma::cx_mat>::from(spectral_entries));

		return true;
	}

	bool TaskMultiStaticSSNakajimaZwanzigTimeEvo::ConstructSpecDensSpecificTimeEvo(const std::complex<double> &_ampl, const std::complex<double> &_tau_c, const arma::cx_mat &_omega, arma::cx_mat &_specdens)
	{
		// Solution of spectral density: S = Ampl/(1/tau_c - i * omega)

		arma::cx_vec spectral_entries = _omega.diag();

		for (auto ii = 0; ii < (int)_omega.n_cols; ii++)
		{
			spectral_entries(ii) = _ampl / ((1.00 / _tau_c) + arma::cx_double(0.0, -1.00) * _omega(ii, ii));
		}

		_specdens = arma::diagmat(arma::conv_to<arma::cx_mat>::from(spectral_entries));

		return true;
	}

	// -----------------------------------------------------
}
