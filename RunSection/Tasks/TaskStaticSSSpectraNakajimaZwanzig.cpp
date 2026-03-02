/////////////////////////////////////////////////////////////////////////
// TaskStaticSSSpectraNakajimaZwanzig implementation (RunSection module)  developed by Luca Gerhards and Irina Anisimova.
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <memory>
#include <sstream>
#include <cmath>
#include "TaskStaticSSSpectraNakajimaZwanzig.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Operator.h"
#include "Pulse.h"

namespace RunSection
{
	namespace
	{
		bool ParseVec3(const std::string &s, arma::vec &v)
		{
			v.set_size(3);
			std::string tmp = s;
			for (char &c : tmp)
			{
				if (c == ',')
					c = ' ';
			}

			std::istringstream iss(tmp);
			double x, y, z;
			if (!(iss >> x >> y >> z))
				return false;
			v(0) = x;
			v(1) = y;
			v(2) = z;
			return true;
		}

		arma::vec SafeNormalise(const arma::vec &v, const arma::vec &fallback)
		{
			double n = arma::norm(v);
			if (!std::isfinite(n) || n < 1e-15)
				return fallback;
			return v / n;
		}
	}

	// -----------------------------------------------------
	// TaskStaticSSSpectraNakajimaZwanzig Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSSpectraNakajimaZwanzig::TaskStaticSSSpectraNakajimaZwanzig(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticSSSpectraNakajimaZwanzig::~TaskStaticSSSpectraNakajimaZwanzig()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSSSpectraNakajimaZwanzig protected methods
	// -----------------------------------------------------
	bool TaskStaticSSSpectraNakajimaZwanzig::RunLocal()
	{
		this->Log() << "Running method StaticSS-Spectra-NakajimaZwanzig." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Obtain spin systems
		auto systems = this->SpinSystems();
		// Decline density matrix and density vector variables
		arma::cx_mat rho0;
		arma::cx_vec rho0vec;
		
		// Loop through all SpinSystems
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Make sure we have an initial state
			auto initial_states = (*i)->InitialState();
			if (initial_states.size() < 1)
			{
				this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no initial state was specified." << std::endl;
				continue;
			}

			this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

			// Obtain a SpinSpace to describe the system
			SpinAPI::SpinSpace space(*(*i));
			space.UseSuperoperatorSpace(false);
			space.SetReactionOperatorType(this->reactionOperators);

			std::vector<double> weights;
			weights = (*i)->Weights();

			// Normalize the weights
			double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
			if (sum_weights > 0)
			{
				for (double &weight : weights)
				{
					weight /= sum_weights;
				}
			}

			// Get the initial state
			if (weights.size() > 1)
			{
				this->Log() << "Using weighted density matrix for initial state. Be sure that the sum of weights equals to 1." << std::endl;
				// Get the initial state
				int counter = 0;
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;
					if ((*j) == nullptr)
					{
						this->Log() << "Failed to obtain weighted thermal initial state for SpinSystem \"" << (*i)->Name() << "\". Weighted thermal initial states are not supported." << std::endl;
						continue;
					}
					if (!space.GetState(*j, tmp_rho0))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
						continue;
					}

					if (j == initial_states.cbegin())
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[0] << "\"." << std::endl;
						rho0 = weights[0] * tmp_rho0;
						counter += 1;
					}
					else
					{
						this->Log() << "State: \"" << (*j)->Name() << "\", Weight:\"" << weights[counter] << "\"." << std::endl;
						rho0 += weights[counter] * tmp_rho0;
						counter += 1;
					}
				}
			}
			else
			// Get the initial state without weights
			{
				for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
				{
					arma::cx_mat tmp_rho0;

					// Get the initial state in thermal equilibrium
					if ((*j) == nullptr) // "Thermal initial state"
					{
						this->Log() << "Initial state = thermal " << std::endl;

						// Get the thermalhamiltonianlist
						std::vector<std::string> thermalhamiltonian_list = (*i)->ThermalHamiltonianList();
						
						this->Log() << "ThermalHamiltonianList = [";
						for (size_t j = 0; j < thermalhamiltonian_list.size(); j++)
						{
							this->Log() << thermalhamiltonian_list[j];
							if (j < thermalhamiltonian_list.size() - 1)
								this->Log() << ", ";  // Add a comma between elements
						}
						this->Log() << "]" << std::endl;
						
						// Get temperature
						double temperature = (*i)->Temperature();
						this->Log() << "Temperature = " << temperature << "K" << std::endl;

						// Get the initial state with thermal equilibrium
						if (!space.GetThermalState(space, temperature, thermalhamiltonian_list, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto thermal state, initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}
					else // Get the initial state without thermal equilibrium
					{
						if (!space.GetState(*j, tmp_rho0))
						{
							this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
							continue;
						}
					}

					// Obtain the initial density matrix
					if (j == initial_states.cbegin())
					{
						rho0 = tmp_rho0;
					}
					else
					{
						rho0 += tmp_rho0;
					}
				}
			}
			rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

			// ----------------------------------------------------------------
			// Get the Hamiltonian
			// ----------------------------------------------------------------
			arma::cx_mat H;
			arma::cx_mat eigen_vec; // To hold eigenvectors
			arma::vec eigen_val; // To hold eigenvalues
			arma::cx_mat eig_val_mat;

			if (!space.Hamiltonian(H))
			{
				this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
				continue;
			}

			// ----------------------------------------------------------------
			// DIAGONALIZATION OF H0// We need all of these operators
			// ----------------------------------------------------------------
			this->Log() << "Starting diagonalization..." << std::endl;
			arma::eig_sym(eigen_val, eigen_vec, (H));
			this->Log() << "Diagonalization done! Eigenvalues: " << eigen_val.n_elem << ", eigenvectors: " << eigen_vec.n_cols << std::endl;

			// Rotate density operator in eigenbasis of H0
			rho0 = (eigen_vec.t() * rho0 * eigen_vec);
			rho0 = rho0 / trace(rho0);

			// Convert initial state to superoperator space
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

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

			// NakajimaZwanzig tensor
			arma::cx_mat R;
			R.set_size(size(kron(H, H)));
			R.zeros();

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
			for (auto interaction = (*i)->interactions_cbegin(); interaction < (*i)->interactions_cend(); interaction++)
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

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
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
										if (!space.LRk0TensorT0((*s1), complex_field, *T0_rank_0))
										{
											this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=0
										if (!space.LRk2SphericalTensorT0((*s1), complex_field, *T0_rank_2))
										{
											this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=1
										if (!space.LRk2SphericalTensorTp1((*s1), complex_field, *Tp1))
										{
											this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=-1
										if (!space.LRk2SphericalTensorTm1((*s1), complex_field, *Tm1))
										{
											this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=2
										if (!space.LRk2SphericalTensorTp2((*s1), complex_field, *Tp2))
										{
											this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and Field"
														<< "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=-2
										if (!space.LRk2SphericalTensorTm2((*s1), complex_field, *Tm2))
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

														if (!ConstructSpecDensSpecificSpectra(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
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

													if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[s]), *ptr_SpecDens[m], tmp_R))
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

													if (!ConstructSpecDensSpecificSpectra(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
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

												if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[k]), *ptr_SpecDens[m], tmp_R))
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

											if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
											{
												return false;
											}

											if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
											{
												return false;
											}

											if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
											{
												return false;
											}

											if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sz()), *s2, *Sz2))
											{
												return false;
											}

											if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sx()), *s2, *Sx2))
											{
												return false;
											}

											if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sy()), *s2, *Sy2))
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

											if (!space.BlRk0TensorT0(*s1, *s2, *T0_rank_0))
											{
												this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=0
											if (!space.BlRk2SphericalTensorT0(*s1, *s2, *T0_rank_2))
											{
												this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=1
											if (!space.BlRk2SphericalTensorTp1(*s1, *s2, *Tp1))
											{
												this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=-1
											if (!space.BlRk2SphericalTensorTm1(*s1, *s2, *Tm1))
											{
												this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=2
											if (!space.BlRk2SphericalTensorTp2(*s1, *s2, *Tp2))
											{
												this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
												continue;
											}

											// Rank 2 tensor with m=-2
											if (!space.BlRk2SphericalTensorTm2(*s1, *s2, *Tm2))
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

															if (!ConstructSpecDensSpecificSpectra(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
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

														if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[s]), *ptr_SpecDens[m], tmp_R))
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

														if (!ConstructSpecDensSpecificSpectra(static_cast<std::complex<double>>(ampl_mat(m, n)), static_cast<std::complex<double>>(tau_c_mat(m, n)), lambda, SpecDens))
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

													if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[k]), *ptr_SpecDens[m], tmp_R))
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

									if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
									{
										return false;
									}

									if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
									{
										return false;
									}

									if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
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
									if (!space.LRk0TensorT0((*s1), complex_field, *T0_rank_0))
									{
										this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
													<< "! Skipping." << std::endl;
										continue;
									}

									// Rank 2 tensor with m=0
									if (!space.LRk2SphericalTensorT0((*s1), complex_field, *T0_rank_2))
									{
										this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and Field"
													<< "! Skipping." << std::endl;
										continue;
									}

									// Rank 2 tensor with m=1
									if (!space.LRk2SphericalTensorTp1((*s1), complex_field, *Tp1))
									{
										this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and Field"
													<< "! Skipping." << std::endl;
										continue;
									}

									// Rank 2 tensor with m=-1
									if (!space.LRk2SphericalTensorTm1((*s1), complex_field, *Tm1))
									{
										this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and Field"
													<< "! Skipping." << std::endl;
										continue;
									}

									// Rank 2 tensor with m=2
									if (!space.LRk2SphericalTensorTp2((*s1), complex_field, *Tp2))
									{
										this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and Field"
													<< "! Skipping." << std::endl;
										continue;
									}

									// Rank 2 tensor with m=-2
									if (!space.LRk2SphericalTensorTm2((*s1), complex_field, *Tm2))
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

											if (!ConstructSpecDensSpecificSpectra(static_cast<std::complex<double>>(ampl_combined), static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
											{
												this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
												continue;
											}

											// -----------------------------------------------------------------
											// CONSTRUCTING R MATRIX
											// -----------------------------------------------------------------

											tmp_R *= 0.0;

											if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
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

										if (!ConstructSpecDensGeneralSpectra(ampl_list, tau_c_list, lambda, SpecDens))
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

											if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
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
												if (!ConstructSpecDensSpecificSpectra(ampl_combined, static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
												{
													this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
													continue;
												}

												// ----------------------------------------------------------------
												// CONSTRUCTING R MATRIX
												// ----------------------------------------------------------------

												tmp_R *= 0.0;

												if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
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

										if (!ConstructSpecDensGeneralSpectra(ampl_list, tau_c_list, lambda, SpecDens))
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

												if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
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

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sz()), *s1, *Sz1))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sx()), *s1, *Sx1))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s1)->Sy()), *s1, *Sy1))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sz()), *s2, *Sz2))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sx()), *s2, *Sx2))
										{
											return false;
										}

										if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*s2)->Sy()), *s2, *Sy2))
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

										if (!space.BlRk0TensorT0(*s1, *s2, *T0_rank_0))
										{
											this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=0
										if (!space.BlRk2SphericalTensorT0(*s1, *s2, *T0_rank_2))
										{
											this->Log() << "Failed to produce T0 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=1
										if (!space.BlRk2SphericalTensorTp1(*s1, *s2, *Tp1))
										{
											this->Log() << "Failed to produce Tp1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=-1
										if (!space.BlRk2SphericalTensorTm1(*s1, *s2, *Tm1))
										{
											this->Log() << "Failed to produce Tm1 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=2
										if (!space.BlRk2SphericalTensorTp2(*s1, *s2, *Tp2))
										{
											this->Log() << "Failed to produce Tp2 spherical tensor between spin " << (*s1)->Name() << " and " << (*s2)->Name() << "! Skipping." << std::endl;
											continue;
										}

										// Rank 2 tensor with m=-2
										if (!space.BlRk2SphericalTensorTm2(*s1, *s2, *Tm2))
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

												if (!ConstructSpecDensSpecificSpectra(static_cast<std::complex<double>>(ampl_combined), static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
												{
													this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
													continue;
												}

												// -----------------------------------------------------------------
												// CONSTRUCTING R MATRIX
												// -----------------------------------------------------------------

												tmp_R *= 0.0;

												if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
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

											if (!ConstructSpecDensGeneralSpectra(ampl_list, tau_c_list, lambda, SpecDens))
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

												if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[k]), SpecDens, tmp_R))
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

													if (!ConstructSpecDensSpecificSpectra(ampl_combined, static_cast<std::complex<double>>(tau_c_list[0]), lambda, SpecDens))
													{
														this->Log() << "There are problems with the construction of the spectral density matrix - Please check your input." << std::endl;
														continue;
													}

													// ----------------------------------------------------------------
													// CONSTRUCTING R MATRIX
													// ----------------------------------------------------------------

													tmp_R *= 0.0;

													if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
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

											if (!ConstructSpecDensGeneralSpectra(ampl_list, tau_c_list, lambda, SpecDens))
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

													if (!NakajimaZwanzigtensorSpectra((*ptr_Tensors[k]), (*ptr_Tensors[s]), SpecDens, tmp_R))
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
			space.SuperoperatorFromLeftOperator(eig_val_mat, lhs);
			space.SuperoperatorFromRightOperator(eig_val_mat, rhs);

			H_SS = lhs - rhs;

			// Get a matrix to collect all the terms (the total Liouvillian)
			arma::cx_mat A = arma::cx_double(0.0, -1.0) * H_SS;

			// Transform ReactionOperator into superspace
			arma::cx_mat K;
			arma::cx_mat Klhs;
			arma::cx_mat Krhs;
			arma::cx_mat K_SS;
			
			if (!space.TotalReactionOperator(K))
			{
				this->Log() << "Warning: Failed to obtain matrix representation of the reaction operators!" << std::endl;
			}

			K = (eigen_vec.t() * K * eigen_vec);

			space.SuperoperatorFromLeftOperator(K, Klhs);
			space.SuperoperatorFromRightOperator(K, Krhs);

			K_SS = Klhs + Krhs;

			A -= K_SS;

			// Get the relaxation terms of other relaxation operators, assuming that they can just be added
			arma::cx_mat O_SS;

			for (auto t = (*i)->operators_cbegin(); t != (*i)->operators_cend(); t++)
			{
				space.UseSuperoperatorSpace(true);

				if (space.RelaxationOperatorFrameChange((*t), eigen_vec, O_SS))
				{

					A += O_SS;

					space.UseSuperoperatorSpace(false);
					this->Log() << "Added other relaxation operator \"" << (*t)->Name() << "\" to the Liouvillian.\n";
				}
				else
				{
					this->Log() << "There is a problem with operator \"" << (*t)->Name() << ". Please check.\n";
					space.UseSuperoperatorSpace(false);
				}
			}
			// Adding R tensor to whole hamiltonian
			A += R;

			// Read in input parameters 

			// Read the method from the input file
			std::string Method;
			if (!this->Properties()->Get("method", Method))
			{
				this->Log() << "Failed to obtain an input for a Method. Please specify method = timeinf or method = timeevo." << std::endl;
			}

			// Read if the result should be integrated or not if method.
			bool integration = false;
			if (!this->Properties()->Get("integration", integration))
			{
				this->Log() << "Failed to obtain an input for an integtation. Plese use integration = true/false. Using integration = false by default. " << std::endl;
			}
			this->Log() << "Integration of the yield in time on a grid  = " << integration << std::endl;

			// Read integrationwindow from the input file
			std::string Integrationwindow;
			if (!this->Properties()->Get("integrationtimeframe", Integrationwindow))
			{
				this->Log() << "Failed to obtain an input for a integrationtimeframe. Please choose integrationtimeframe =  pulse / freeevo / full. Using freeevo propagation evolution window by default" << std::endl;
				Integrationwindow = "freeevo";
			}
			this->Log() << "Timewindow for the propagation integration: " << Integrationwindow << std::endl;

			// Read CIDSP from the input file
			bool CIDSP = false;
			if (!this->Properties()->Get("cidsp", CIDSP))
			{
				this->Log() << "Failed to obtain an input for a CIDSP. Plese use cidsp = true/false. Using cidsp = false by default. " << std::endl;
			}
			ProjectionCache projection_cache;
			if (!this->BuildProjectionCache(*i, space, eigen_vec, CIDSP, projection_cache, this->Log()))
			{
				this->Log() << "Failed to build projection cache for SpinSystem \"" << (*i)->Name() << "\". Skipping system." << std::endl;
				continue;
			}

			// Read printtimeframe from the input file
			std::string Timewindow;
			if (!this->Properties()->Get("printtimeframe", Timewindow))
			{
				this->Log() << "Failed to obtain an input for a printtimeframe. Please choose printtimeframe =  pulse / freeevo / full. Using full propagation evolution window by default" << std::endl;\
				Timewindow = "full";
			}
			this->Log() << "Timewindow for the propagation printing: " << Timewindow << std::endl;

			double Printedtime = 0;
			
			// Initialize a first step
			arma::cx_vec rhovec = rho0vec;

			// Read a pulse sequence from the input
			std::vector<std::tuple<std::string, double>> Pulsesequence;
			if (this->Properties()->GetPulseSequence("pulsesequence", Pulsesequence))
			{
				this->Log() << "Pulsesequence" << std::endl;

				// Loop through all pulse sequences
				for (const auto &seq : Pulsesequence)
				{
					// Write which pulse in pulsesequence is calculating now
					this->Log() << std::get<0>(seq) << ", " << std::get<1>(seq) << std::endl;

					// Save the parameters from the input as variables
					std::string pulse_name = std::get<0>(seq);
					double timerelaxation = std::get<1>(seq);
					space.UseSuperoperatorSpace(true);

					for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
					{
						if ((*pulse)->Name().compare(pulse_name) == 0)
						{

							// Apply a pulse to our density vector
							if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
							{
								// Create a Pulse operator in SS
								arma::cx_mat pulse_operator;
								if (!space.PulseOperatorFrameChange((*pulse), eigen_vec, pulse_operator))
								{
									this->Log() << "Failed to create a pulse operator in SS." << std::endl;
									continue;
								}
								rhovec = pulse_operator * rhovec;
							}
							else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulseStaticField)
							{

								// Create a Pulse operator in SS
								arma::cx_mat pulse_operator;
								if (!space.PulseOperatorFrameChange((*pulse), eigen_vec, pulse_operator))
								{
									this->Log() << "Failed to create a pulse operator in SS." << std::endl;
									continue;
								}

								// Create a holder vector for an averaged density
								arma::cx_vec rhoavg;
								rhoavg.zeros(size(rho0vec));

								int firststep;
								if (Printedtime == 0)
									firststep = 0;
								else
									firststep = 1;

								unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
								bool use_dense_expm = (A.n_rows <= 64);
								arma::cx_mat A_sp_dense;
								if (use_dense_expm)
								{
									try
									{
										A_sp_dense = arma::expmat((A + (arma::cx_double(0.0, -1.0) * pulse_operator)) * (*pulse)->Timestep());
									}
									catch (const std::exception &e)
									{
										this->Log() << "Dense expm LongPulseStaticField propagation failed (" << e.what() << "). Falling back to Krylov propagation." << std::endl;
										use_dense_expm = false;
									}
								}
								if (use_dense_expm)
								{
									// Create array containing a propagator and the current state of each system
									std::pair<arma::cx_mat, arma::cx_vec> G;
									G = std::pair<arma::cx_mat, arma::cx_vec>(A_sp_dense, rhovec);

									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0) // for the very first step just printing is done, no propagation
										{
											// Take a step, "first" is propagator and "second" is current state
											rhovec = G.first * G.second;

											// Integrate the result if needed
											if ((integration) && (Integrationwindow.compare("freeevo") != 0))
											{
												rhoavg += (*pulse)->Timestep() * (G.second + rhovec) / 2;
											}

											// Get the new current state density vector
											G.second = rhovec;

											// Save the result if there were some changes
											if (!rhoavg.is_zero(0))
											{
												rhovec = rhoavg;
											}
										}

										if (Timewindow.compare("freeevo") != 0)
										{
											if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
												this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
										}
									}
								}
								else
								{
									arma::sp_cx_mat A_step = arma::conv_to<arma::sp_cx_mat>::from(A + (arma::cx_double(0.0, -1.0) * pulse_operator));
									arma::cx_vec tmp_rho = rhovec;

									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0)
										{
											arma::cx_vec out_rho;
											if (arma::norm(tmp_rho, 2) < 1.0e-20)
											{
												out_rho = tmp_rho;
											}
											else
											{
												try
												{
													out_rho = space.KrylovExpmGeneral(A_step, tmp_rho, (*pulse)->Timestep(), 16, A_step.n_rows);
												}
												catch (const std::exception &e)
												{
													this->Log() << "Krylov LongPulseStaticField propagation failed (" << e.what() << "). Retrying with smaller substeps." << std::endl;
													const int substeps = 8;
													const double sub_dt = (*pulse)->Timestep() / static_cast<double>(substeps);
													arma::cx_vec sub_rho = tmp_rho;
													bool sub_ok = true;
													for (int sub = 0; sub < substeps; ++sub)
													{
														try
														{
															sub_rho = space.KrylovExpmGeneral(A_step, sub_rho, sub_dt, 16, A_step.n_rows);
														}
														catch (...)
														{
															sub_ok = false;
															break;
														}
													}
													out_rho = sub_ok ? sub_rho : tmp_rho;
												}
											}

											// Integrate the result if needed
											if ((integration) && (Integrationwindow.compare("freeevo") != 0))
											{
												rhoavg += (*pulse)->Timestep() * (tmp_rho + out_rho) / 2;
											}

											// Get the new current state density vector
											tmp_rho = out_rho;
											rhovec = out_rho;

											// Save the result if there were some changes
											if (!rhoavg.is_zero(0))
											{
												rhovec = rhoavg;
											}
										}

										if (Timewindow.compare("freeevo") != 0)
										{
											if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
												this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
										}
									}
								}
							}
							else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
							{
								// Create a Pulse operator in SS
								arma::cx_mat pulse_operator;
								if (!space.PulseOperatorFrameChange((*pulse), eigen_vec, pulse_operator))
								{
									this->Log() << "Failed to create a pulse operator in SS." << std::endl;
									continue;
								}
								
								// Create a holder vector for an averaged density
								arma::cx_vec rhoavg;
								rhoavg.zeros(size(rho0vec));

								int firststep;
								if (Printedtime == 0)
									firststep = 0;
								else
									firststep = 1;

								// Save the density on the current step.
								arma::cx_vec tmp_rho = rhovec;
								unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
								if (A.n_rows <= 64)
								{
									for (unsigned int n = firststep; n <= steps; n++)
										{
											if (n != 0) // for the very first step just printing is done, no propagation
											{
												// Use time-dependent pulse strength at this propagation step.
												double t = n * (*pulse)->Timestep();
												bool used_dense_expm = true;
												try
												{
													arma::cx_mat A_sp = arma::expmat((A + (arma::cx_double(0.0, -1.0) * pulse_operator * std::cos((*pulse)->Frequency() * t))) * (*pulse)->Timestep());
													rhovec = A_sp * tmp_rho;
												}
												catch (const std::exception &e)
												{
													used_dense_expm = false;
													this->Log() << "Dense expm LongPulse propagation failed (" << e.what() << "). Falling back to Krylov propagation for this step." << std::endl;
												}
												if (!used_dense_expm)
												{
													arma::sp_cx_mat A_step = arma::conv_to<arma::sp_cx_mat>::from(
														A + (arma::cx_double(0.0, -1.0) * pulse_operator * std::cos((*pulse)->Frequency() * t)));
													if (arma::norm(tmp_rho, 2) < 1.0e-20)
													{
														rhovec = tmp_rho;
													}
													else
													{
														try
														{
															rhovec = space.KrylovExpmGeneral(A_step, tmp_rho, (*pulse)->Timestep(), 16, A_step.n_rows);
														}
														catch (const std::exception &e2)
														{
															this->Log() << "Krylov LongPulse fallback propagation failed (" << e2.what() << "). Retrying with smaller substeps." << std::endl;
															const int substeps = 8;
															const double sub_dt = (*pulse)->Timestep() / static_cast<double>(substeps);
															arma::cx_vec sub_rho = tmp_rho;
															bool sub_ok = true;
															for (int sub = 0; sub < substeps; ++sub)
															{
																try
																{
																	sub_rho = space.KrylovExpmGeneral(A_step, sub_rho, sub_dt, 16, A_step.n_rows);
																}
																catch (...)
																{
																	sub_ok = false;
																	break;
																}
															}
															rhovec = sub_ok ? sub_rho : tmp_rho;
														}
													}
												}

												// Integrate the result if needed
												if ((integration) && (Integrationwindow.compare("freeevo") != 0))
												{
												rhoavg += (*pulse)->Timestep() * (tmp_rho + rhovec) / 2;
											}

											// Get the new current state density vector
											tmp_rho = rhovec;

											// Save the result if there were some changes
											if (!rhoavg.is_zero(0))
											{
												rhovec = rhoavg;
											}
										}

										if (Timewindow.compare("freeevo") != 0)
										{
											if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
												this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
										}
									}
								}
								else
								{
									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0)
										{
											arma::cx_vec out_rho;
											// Midpoint time gives a better approximation for large systems under harmonic drive.
											double t_eval = n * (*pulse)->Timestep() + 0.5 * (*pulse)->Timestep();
											arma::sp_cx_mat A_step = arma::conv_to<arma::sp_cx_mat>::from(
												A + (arma::cx_double(0.0, -1.0) * pulse_operator * std::cos((*pulse)->Frequency() * t_eval)));
											if (arma::norm(tmp_rho, 2) < 1.0e-20)
											{
												out_rho = tmp_rho;
											}
											else
											{
												try
												{
													out_rho = space.KrylovExpmGeneral(A_step, tmp_rho, (*pulse)->Timestep(), 16, A_step.n_rows);
												}
												catch (const std::exception &e)
												{
													this->Log() << "Krylov LongPulse propagation failed (" << e.what() << "). Retrying with smaller substeps." << std::endl;
													const int substeps = 8;
													const double sub_dt = (*pulse)->Timestep() / static_cast<double>(substeps);
													arma::cx_vec sub_rho = tmp_rho;
													bool sub_ok = true;
													for (int sub = 0; sub < substeps; ++sub)
													{
														try
														{
															sub_rho = space.KrylovExpmGeneral(A_step, sub_rho, sub_dt, 16, A_step.n_rows);
														}
														catch (...)
														{
															sub_ok = false;
															break;
														}
													}
													out_rho = sub_ok ? sub_rho : tmp_rho;
												}
											}

											// Integrate the result if needed
											if ((integration) && (Integrationwindow.compare("freeevo") != 0))
											{
												rhoavg += (*pulse)->Timestep() * (tmp_rho + out_rho) / 2;
											}

											// Get the new current state density vector
											tmp_rho = out_rho;
											rhovec = out_rho;

											// Save the result if there were some changes
											if (!rhoavg.is_zero(0))
											{
												rhovec = rhoavg;
											}
										}

										if (Timewindow.compare("freeevo") != 0)
										{
											if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
												this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
										}
									}
								}
							}
							else if ((*pulse)->Type() == SpinAPI::PulseType::MWPulse)
							{
								// Create a holder vector for an averaged density
								arma::cx_vec rhoavg;
								rhoavg.zeros(size(rho0vec));

								// Define first propagation step
								int firststep;
								if (Printedtime == 0)
									firststep = 0;
								else
									firststep = 1;

								// Save the density on the current step
								arma::cx_vec tmp_rho = rhovec;

								unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0) // for the very first step just printing is done, no propagation
										{
											arma::cx_vec out_rho;
											// Midpoint time gives a better approximation for large systems under harmonic drive.
											double t_eval = (A.n_rows <= 64) ? (n * (*pulse)->Timestep()) : (n * (*pulse)->Timestep() + 0.5 * (*pulse)->Timestep());
											arma::cx_mat pulse_operator;
											if (!space.PulseOperatorFrameChange_mw((*pulse), eigen_vec, pulse_operator, t_eval))
											{
												this->Log() << "Failed to create a frame-changed microwave pulse operator in SS." << std::endl;
												continue;
											}

												if (A.n_rows <= 64)
												{
													bool used_dense_expm = true;
													try
													{
														arma::cx_mat A_step = arma::expmat((A + (arma::cx_double(0.0, -1.0) * pulse_operator)) * (*pulse)->Timestep());
														out_rho = A_step * tmp_rho;
													}
													catch (const std::exception &e)
													{
														used_dense_expm = false;
														this->Log() << "Dense expm MWPulse propagation failed (" << e.what() << "). Falling back to Krylov propagation for this step." << std::endl;
													}
													if (!used_dense_expm)
													{
														arma::sp_cx_mat A_step = arma::conv_to<arma::sp_cx_mat>::from(A + (arma::cx_double(0.0, -1.0) * pulse_operator));
														if (arma::norm(tmp_rho, 2) < 1.0e-20)
														{
															out_rho = tmp_rho;
														}
														else
														{
															try
															{
																out_rho = space.KrylovExpmGeneral(A_step, tmp_rho, (*pulse)->Timestep(), 16, A_step.n_rows);
															}
															catch (const std::exception &e2)
															{
																this->Log() << "Krylov MWPulse fallback propagation failed (" << e2.what() << "). Retrying with smaller substeps." << std::endl;
																const int substeps = 8;
																const double sub_dt = (*pulse)->Timestep() / static_cast<double>(substeps);
																arma::cx_vec sub_rho = tmp_rho;
																bool sub_ok = true;
																for (int sub = 0; sub < substeps; ++sub)
																{
																	try
																	{
																		sub_rho = space.KrylovExpmGeneral(A_step, sub_rho, sub_dt, 16, A_step.n_rows);
																	}
																	catch (...)
																	{
																		sub_ok = false;
																		break;
																	}
																}
																out_rho = sub_ok ? sub_rho : tmp_rho;
															}
														}
													}
												}
											else
											{
												arma::sp_cx_mat A_step = arma::conv_to<arma::sp_cx_mat>::from(A + (arma::cx_double(0.0, -1.0) * pulse_operator));
												if (arma::norm(tmp_rho, 2) < 1.0e-20)
												{
													out_rho = tmp_rho;
												}
												else
												{
													try
													{
														out_rho = space.KrylovExpmGeneral(A_step, tmp_rho, (*pulse)->Timestep(), 16, A_step.n_rows);
													}
													catch (const std::exception &e)
													{
														this->Log() << "Krylov MWPulse propagation failed (" << e.what() << "). Retrying with smaller substeps." << std::endl;
														const int substeps = 8;
														const double sub_dt = (*pulse)->Timestep() / static_cast<double>(substeps);
														arma::cx_vec sub_rho = tmp_rho;
														bool sub_ok = true;
														for (int sub = 0; sub < substeps; ++sub)
														{
															try
															{
																sub_rho = space.KrylovExpmGeneral(A_step, sub_rho, sub_dt, 16, A_step.n_rows);
															}
															catch (...)
															{
																sub_ok = false;
																break;
															}
														}
														out_rho = sub_ok ? sub_rho : tmp_rho;
													}
												}
											}

										// Integrate the result if needed
										if ((integration) && (Integrationwindow.compare("freeevo") != 0))
										{
											rhoavg += (*pulse)->Timestep() * (tmp_rho + out_rho) / 2;
										}

										// Get the new current state density vector
										tmp_rho = out_rho;
										rhovec = out_rho;

										// Save the result if there were some changes
										if (!rhoavg.is_zero(0))
										{
											rhovec = rhoavg;
										}
									}

									if (Timewindow.compare("freeevo") != 0)
									{
										if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
											this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
									}
								}
							}
							else
							{
								this->Log() << "Current pulse type is not implemented. Please use type = InstantPulse / LongPulseStaticField / LongPulse / MWPulse." << std::endl;
							}

							// Update the printed time according to the printtimeframe key
							if (Timewindow.compare("freeevo") != 0)
							{
								Printedtime += (*pulse)->Pulsetime();
							}

							// Get the system relax during the time
								if (timerelaxation != 0)
								{
									// Create a holder vector for an averaged density
									arma::cx_vec rhoavg;
									rhoavg.zeros(size(rho0vec));

									unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / (*pulse)->Timestep()));
									bool use_dense_expm = (A.n_rows <= 64);
									arma::cx_mat A_sp_dense;
									if (use_dense_expm)
									{
										try
										{
											A_sp_dense = arma::expmat(A * (*pulse)->Timestep());
										}
										catch (const std::exception &e)
										{
											this->Log() << "Dense expm free-evolution propagation failed (" << e.what() << "). Falling back to Krylov propagation." << std::endl;
											use_dense_expm = false;
										}
									}
									if (use_dense_expm)
									{
										std::pair<arma::cx_mat, arma::cx_vec> G(A_sp_dense, rhovec);
										for (unsigned int n = 1; n <= steps; n++) // n always starts with one here
										{
											rhovec = G.first * G.second;

											// Integrate the result if needed
											if ((integration) && (Integrationwindow.compare("freeevo") != 0))
											{
												rhoavg += (*pulse)->Timestep() * (G.second + rhovec) / 2;
											}

											// Get the new current state density vector
											G.second = rhovec;

											// Save the result if there were some changes
											if (!rhoavg.is_zero(0))
											{
												rhovec = rhoavg;
											}

											if (Timewindow.compare("freeevo") != 0)
											{
												if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
													this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
											}
										}
									}
									else
									{
										arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(A);
										arma::cx_vec tmp_rho = rhovec;
										for (unsigned int n = 1; n <= steps; n++) // n always starts with one here
										{
											arma::cx_vec out_rho;
											if (arma::norm(tmp_rho, 2) < 1.0e-20)
											{
												out_rho = tmp_rho;
											}
											else
											{
												try
												{
													out_rho = space.KrylovExpmGeneral(A_sp, tmp_rho, (*pulse)->Timestep(), 16, A_sp.n_rows);
												}
												catch (const std::exception &e)
												{
													this->Log() << "Krylov free-evolution propagation failed (" << e.what() << "). Retrying with smaller substeps." << std::endl;
													const int substeps = 8;
													const double sub_dt = (*pulse)->Timestep() / static_cast<double>(substeps);
													arma::cx_vec sub_rho = tmp_rho;
													bool sub_ok = true;
													for (int sub = 0; sub < substeps; ++sub)
													{
														try
														{
															sub_rho = space.KrylovExpmGeneral(A_sp, sub_rho, sub_dt, 16, A_sp.n_rows);
														}
														catch (...)
														{
															sub_ok = false;
															break;
														}
													}
													out_rho = sub_ok ? sub_rho : tmp_rho;
												}
											}
											rhovec = out_rho;

											// Integrate the result if needed
											if ((integration) && (Integrationwindow.compare("freeevo") != 0))
											{
												rhoavg += (*pulse)->Timestep() * (tmp_rho + rhovec) / 2;
											}

											// Get the new current state density vector
											tmp_rho = rhovec;

											// Save the result if there were some changes
											if (!rhoavg.is_zero(0))
											{
												rhovec = rhoavg;
											}

											if (Timewindow.compare("freeevo") != 0)
											{
												if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
													this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
											}
										}
									}
								}
							// Update the printed time according to the printtimeframe key
							if (Timewindow.compare("freeevo") != 0)
							{
								Printedtime += timerelaxation;
							}
						}
					}
					space.UseSuperoperatorSpace(false);
				}
			}
		
			//////////////////////////////////////////////////////

			// Method Propagation to infinity
			if (Method.compare("timeinf") == 0)
			{
				// Perform the calculation
				this->Log() << "Ready to perform calculation." << std::endl;

				this->Log() << "Method = " << Method << std::endl;

				// Print the warning to the user, what integration keyword is used
				if (integration)
				{
					this->Log() << "Warning: steady state method (timeinf) is calculated as an inverse of the Liouvillian operator, instead of the integration on a grid."<<
					"The integration of the pulse sequence timewindow could be added if integration = true and integrationtimeframe = pulse / full." << std::endl;
				}

				arma::cx_vec result = -solve(arma::conv_to<arma::cx_mat>::from(A), rhovec);

				rhovec = result;

				if (Timewindow.compare("pulse") != 0)
				{
					if (!this->ProjectAndPrintOutputLineInf(i, space, projection_cache, rhovec, Printedtime, this->timestep, CIDSP, this->Data(), this->Log()))
						this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
				}

				this->Log() << "Done with calculation." << std::endl;
			}
			// Method TIME EVOLUTION
			else if (Method.compare("timeevo") == 0)
			{

					if (!this->totaltime == 0)
					{
					// Perform the calculation
					this->Log() << "Ready to perform calculation." << std::endl;

					this->Log() << "Method = " << Method << std::endl;

						// Create a holder vector for an averaged density
						arma::cx_vec rhoavg;
						rhoavg.zeros(size(rho0vec));

						// Avoid printing double timesteps
						int firststep;
						if (Printedtime == 0)
							firststep = 0;
						else
							firststep = 1;

						unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
						bool use_dense_expm = (A.n_rows <= 64);
						arma::cx_mat A_sp_dense;
						if (use_dense_expm)
						{
							try
							{
								A_sp_dense = arma::expmat(A * this->timestep);
							}
							catch (const std::exception &e)
							{
								this->Log() << "Dense expm time-evolution propagation failed (" << e.what() << "). Falling back to Krylov propagation." << std::endl;
								use_dense_expm = false;
							}
						}
						if (use_dense_expm)
						{
							// Create array containing a propagator and the current state of each system.
							std::pair<arma::cx_mat, arma::cx_vec> G(A_sp_dense, rhovec);
							for (unsigned int n = firststep; n <= steps; n++)
							{
								if (n != 0)
								{
									// Take a step, "first" is propagator and "second" is current state.
									rhovec = G.first * G.second;

									// Integrate the density vector over the current time interval.
									if ((integration) && (Integrationwindow.compare("pulse") != 0))
									{
										rhoavg += this->timestep * (G.second + rhovec) / 2;
									}

									// Get the new current state density vector.
									G.second = rhovec;

									// Save the result if there were some changes (made so we can include TotalTime=0).
									if (!rhoavg.is_zero(0))
									{
										rhovec = rhoavg;
									}
								}

								if (Timewindow.compare("pulse") != 0)
								{
									if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, this->timestep, n, CIDSP, this->Data(), this->Log()))
										this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
								}
							}
						}
						else
						{
							arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(A);
							arma::cx_vec tmp_rho = rhovec;
							for (unsigned int n = firststep; n <= steps; n++)
							{
								if (n != 0)
								{
									arma::cx_vec out_rho;
									if (arma::norm(tmp_rho, 2) < 1.0e-20)
									{
										out_rho = tmp_rho;
									}
									else
									{
										try
										{
											out_rho = space.KrylovExpmGeneral(A_sp, tmp_rho, this->timestep, 16, A_sp.n_rows);
										}
										catch (const std::exception &e)
										{
											this->Log() << "Krylov time-evolution propagation failed (" << e.what() << "). Retrying with smaller substeps." << std::endl;
											const int substeps = 8;
											const double sub_dt = this->timestep / static_cast<double>(substeps);
											arma::cx_vec sub_rho = tmp_rho;
											bool sub_ok = true;
											for (int sub = 0; sub < substeps; ++sub)
											{
												try
												{
													sub_rho = space.KrylovExpmGeneral(A_sp, sub_rho, sub_dt, 16, A_sp.n_rows);
												}
												catch (...)
												{
													sub_ok = false;
													break;
												}
											}
											out_rho = sub_ok ? sub_rho : tmp_rho;
										}
									}
									rhovec = out_rho;

									// Integrate the density vector over the current time interval.
									if ((integration) && (Integrationwindow.compare("pulse") != 0))
									{
										rhoavg += this->timestep * (tmp_rho + rhovec) / 2;
									}

									// Get the new current state density vector.
									tmp_rho = rhovec;

									// Save the result if there were some changes (made so we can include TotalTime=0).
									if (!rhoavg.is_zero(0))
									{
										rhovec = rhoavg;
									}
								}

								if (Timewindow.compare("pulse") != 0)
								{
									if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, this->timestep, n, CIDSP, this->Data(), this->Log()))
										this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
								}
							}
						}
					}

				this->Log() << "Done with calculation." << std::endl;
			}
			else
			{
				this->Log() << "Undefined spectroscopy method. Please choose between timeinf or timeevo methods." << std::endl;
			}

			this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
		}

		// Terminate the line in the data file after iteration through all spin systems
		this->Data() << std::endl;

		return true;
	}

	// Writes the header of the data file (but can also be passed to other streams)
	void TaskStaticSSSpectraNakajimaZwanzig::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		std::string outputframe = "lab";
		(void)this->Properties()->Get("outputframe", outputframe);
		std::string Lx = ".Ix ";
		std::string Ly = ".Iy ";
		std::string Lz = ".Iz ";
		std::string Lp = ".Ip ";
		std::string Lm = ".Im ";
		if (outputframe == "cw")
		{
			Lx = ".Idrive ";
			Ly = ".Iabs ";
			Lz = ".In ";
			Lp = ".Iplus ";
			Lm = ".Iminus ";
		}

		std::vector<std::string> spinList;
		bool CIDSP = false;
		int m;

		// Get header for each spin system
		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{

			if (this->Properties()->GetList("spinlist", spinList, ','))
			{
				for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
				{
					std::string spintype;

					(*l)->Properties()->Get("type", spintype);

					for (m = 0; m < (int)spinList.size(); m++)
					{

						if ((*l)->Name() == spinList[m])
						{
							if (this->Properties()->Get("cidsp", CIDSP) && CIDSP == true)
							{
								auto transitions = (*i)->Transitions();
								for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
								{
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield" << Lx;
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield" << Ly;
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield" << Lz;
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield" << Lp;
									_stream << (*i)->Name() << "." << (*l)->Name() << "." << (*j)->Name() << ".yield" << Lm;
								}
							}
							else
							{
								_stream << (*i)->Name() << "." << (*l)->Name() << Lx;
								_stream << (*i)->Name() << "." << (*l)->Name() << Ly;
								_stream << (*i)->Name() << "." << (*l)->Name() << Lz;
								_stream << (*i)->Name() << "." << (*l)->Name() << Lp;
								_stream << (*i)->Name() << "." << (*l)->Name() << Lm;
							}
						}
					}
				}
			}
		}
		_stream << std::endl;
	}

	// Validation
	bool TaskStaticSSSpectraNakajimaZwanzig::Validate()
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
				this->Log() << "# WARNING: undefined timestep, using by default 0.1 ns!" << std::endl;
				this->timestep = 0.1;
			}
		}

		// Get totaltime
		if (this->Properties()->Get("totaltime", inputTotaltime))
		{
			if (std::isfinite(inputTotaltime) && inputTotaltime >= 0.0)
			{
				this->totaltime = inputTotaltime;
			}
			else
			{
				this->Log() << "# ERROR: invalid total time!" << std::endl;
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

	// --------------------------------------------------------------------------------------------------------------------------------------
	// Construction NZ tensor
	// --------------------------------------------------------------------------------------------------------------------------------------
	bool TaskStaticSSSpectraNakajimaZwanzig::NakajimaZwanzigtensorSpectra(const arma::cx_mat &_op1, const arma::cx_mat &_op2, const arma::cx_mat &_specdens, arma::cx_mat &_NakajimaZwanzigtensor)
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

	bool TaskStaticSSSpectraNakajimaZwanzig::ConstructSpecDensGeneralSpectra(const std::vector<double> &_ampl_list, const std::vector<double> &_tau_c_list, const arma::cx_mat &_omega, arma::cx_mat &_specdens)
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

	bool TaskStaticSSSpectraNakajimaZwanzig::ConstructSpecDensSpecificSpectra(const std::complex<double> &_ampl, const std::complex<double> &_tau_c, const arma::cx_mat &_omega, arma::cx_mat &_specdens)
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

	bool TaskStaticSSSpectraNakajimaZwanzig::BuildProjectionCache(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space, const arma::cx_mat &_rotationmtx, bool _cidsp, ProjectionCache &_cache, std::ostream &_log_stream)
	{
		_cache = ProjectionCache{};

		std::string outputframe = "lab";
		(void)this->Properties()->Get("outputframe", outputframe);
		std::string b0dir_str, b1dir_str;
		arma::vec b0dir = arma::vec({0.0, 0.0, 1.0});
		arma::vec b1dir = arma::vec({1.0, 0.0, 0.0});
		if (this->Properties()->Get("b0dir", b0dir_str))
		{
			arma::vec tmp;
			if (ParseVec3(b0dir_str, tmp))
				b0dir = SafeNormalise(tmp, b0dir);
		}
		if (this->Properties()->Get("b1dir", b1dir_str))
		{
			arma::vec tmp;
			if (ParseVec3(b1dir_str, tmp))
				b1dir = SafeNormalise(tmp, b1dir);
		}

		std::vector<std::string> spinList;
		_cache.has_spinlist = this->Properties()->GetList("spinlist", spinList, ',');
		if (!_cache.has_spinlist)
		{
			return false;
		}

		_cache.spin_Ix.reserve(spinList.size());
		_cache.spin_Iy.reserve(spinList.size());
		_cache.spin_Iz.reserve(spinList.size());
		_cache.spin_Ip.reserve(spinList.size());
		_cache.spin_Im.reserve(spinList.size());

		for (auto l = _system->spins_cbegin(); l != _system->spins_cend(); l++)
		{
			for (const auto &spin_name : spinList)
			{
				if ((*l)->Name() == spin_name)
				{
					arma::cx_mat Iprojx;
					arma::cx_mat Iprojy;
					arma::cx_mat Iprojz;
					arma::cx_mat Iprojp;
					arma::cx_mat Iprojm;

					if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sx()), (*l), Iprojx))
					{
						return false;
					}

					if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sy()), (*l), Iprojy))
					{
						return false;
					}

					if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sz()), (*l), Iprojz))
					{
						return false;
					}

					if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sp()), (*l), Iprojp))
					{
						return false;
					}

					if (!_space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*l)->Sm()), (*l), Iprojm))
					{
						return false;
					}

					if (outputframe == "cw" && (*l)->Type() == SpinAPI::SpinType::Electron)
					{
						arma::mat G = (*l)->GetTensor().LabFrame();
						arma::vec a = G.t() * b0dir;
						arma::vec n = SafeNormalise(a, b0dir);

						arma::vec v = G.t() * b1dir;
						arma::vec v_perp = v - arma::dot(v, n) * n;
						arma::vec e1 = SafeNormalise(v_perp, b1dir);

						arma::vec e2 = arma::cross(n, e1);
						e2 = SafeNormalise(e2, arma::vec({0.0, 1.0, 0.0}));

						arma::cx_mat I_e1 = e1(0) * Iprojx + e1(1) * Iprojy + e1(2) * Iprojz;
						arma::cx_mat I_e2 = e2(0) * Iprojx + e2(1) * Iprojy + e2(2) * Iprojz;
						arma::cx_mat I_n = n(0) * Iprojx + n(1) * Iprojy + n(2) * Iprojz;

						arma::cx_double I(0.0, 1.0);
						arma::cx_mat I_plus = I_e1 + I * I_e2;
						arma::cx_mat I_minus = I_e1 - I * I_e2;

						Iprojx = std::move(I_e1);
						Iprojy = std::move(I_e2);
						Iprojz = std::move(I_n);
						Iprojp = std::move(I_plus);
						Iprojm = std::move(I_minus);
					}

					// NZ task propagates in H0 eigenbasis, so cache observables in that basis.
					Iprojx = _rotationmtx.t() * Iprojx * _rotationmtx;
					Iprojy = _rotationmtx.t() * Iprojy * _rotationmtx;
					Iprojz = _rotationmtx.t() * Iprojz * _rotationmtx;
					Iprojp = _rotationmtx.t() * Iprojp * _rotationmtx;
					Iprojm = _rotationmtx.t() * Iprojm * _rotationmtx;

					_cache.spin_Ix.push_back(std::move(Iprojx));
					_cache.spin_Iy.push_back(std::move(Iprojy));
					_cache.spin_Iz.push_back(std::move(Iprojz));
					_cache.spin_Ip.push_back(std::move(Iprojp));
					_cache.spin_Im.push_back(std::move(Iprojm));
				}
			}
		}

		if (_cidsp)
		{
			auto transitions = _system->Transitions();
			_cache.transition_proj.reserve(transitions.size());
			_cache.transition_rates.reserve(transitions.size());

			for (auto j = transitions.cbegin(); j != transitions.cend(); j++)
			{
				if ((*j)->SourceState() == nullptr)
					continue;

				arma::cx_mat P;
				if (!_space.GetState((*j)->SourceState(), P))
				{
					_log_stream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << _system->Name() << "\"." << std::endl;
					continue;
				}

				P = _rotationmtx.t() * P * _rotationmtx;
				_cache.transition_proj.push_back(std::move(P));
				_cache.transition_rates.push_back((*j)->Rate());
			}
		}

		_cache.ready = true;
		return true;
	}

	bool TaskStaticSSSpectraNakajimaZwanzig::ProjectAndPrintOutputLine(auto &_i, SpinAPI::SpinSpace &_space, const ProjectionCache &_cache, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		(void)_i;

		arma::cx_mat rho0;
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)) && (_n == 0))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		if (_n == 0)
			_logstream << "CIDSP = " << _cidsp << std::endl;

		_datastream << this->RunSettings()->CurrentStep() << " ";
		_datastream << std::setprecision(12) << _printedtime + (_n * _timestep) << " ";
		this->WriteStandardOutput(_datastream);

		if (!_cache.has_spinlist)
		{
			if (_n == 0)
				_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		if (!_cache.ready)
		{
			return false;
		}

		if (_cidsp == true)
		{
			for (size_t s = 0; s < _cache.spin_Ix.size(); ++s)
			{
				for (size_t t = 0; t < _cache.transition_proj.size(); ++t)
				{
					const double rate = _cache.transition_rates[t];
					const arma::cx_mat &P = _cache.transition_proj[t];
					_datastream << std::real(arma::trace(_cache.spin_Ix[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Iy[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Iz[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Ip[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Im[s] * rate * P * rho0)) << " ";
				}
			}
		}
		else
		{
			for (size_t s = 0; s < _cache.spin_Ix.size(); ++s)
			{
				_datastream << std::real(arma::trace(_cache.spin_Ix[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Iy[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Iz[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Ip[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Im[s] * rho0)) << " ";
			}
		}

		_datastream << std::endl;
		return true;
	}

	bool TaskStaticSSSpectraNakajimaZwanzig::ProjectAndPrintOutputLineInf(auto &_i, SpinAPI::SpinSpace &_space, const ProjectionCache &_cache, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		(void)_i;
		(void)_printedtime;
		(void)_timestep;

		arma::cx_mat rho0;
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		_logstream << "CIDSP = " << _cidsp << std::endl;

		_datastream << this->RunSettings()->CurrentStep() << " ";
		_datastream << "inf" << " ";
		this->WriteStandardOutput(_datastream);

		if (!_cache.has_spinlist)
		{
			_logstream << "No nucleus was specified for projection" << std::endl;
			return false;
		}

		if (!_cache.ready)
		{
			return false;
		}

		if (_cidsp == true)
		{
			for (size_t s = 0; s < _cache.spin_Ix.size(); ++s)
			{
				for (size_t t = 0; t < _cache.transition_proj.size(); ++t)
				{
					const double rate = _cache.transition_rates[t];
					const arma::cx_mat &P = _cache.transition_proj[t];
					_datastream << std::real(arma::trace(_cache.spin_Ix[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Iy[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Iz[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Ip[s] * rate * P * rho0)) << " ";
					_datastream << std::real(arma::trace(_cache.spin_Im[s] * rate * P * rho0)) << " ";
				}
			}
		}
		else
		{
			for (size_t s = 0; s < _cache.spin_Ix.size(); ++s)
			{
				_datastream << std::real(arma::trace(_cache.spin_Ix[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Iy[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Iz[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Ip[s] * rho0)) << " ";
				_datastream << std::real(arma::trace(_cache.spin_Im[s] * rho0)) << " ";
			}
		}

		return true;
	}
}
