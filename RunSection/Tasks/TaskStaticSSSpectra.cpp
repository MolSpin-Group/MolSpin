/////////////////////////////////////////////////////////////////////////
// TaskStaticSSSpectra implementation (RunSection module)  developed by Irina Anisimova and Luca Gerhards.
//
// Molecular Spin Dynamics Software - developed by Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <numeric>
#include <utility>
#include "TaskStaticSSSpectra.h"
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

#include <sstream>
#include <cmath>

namespace RunSection
{

	namespace
	{
		// Parse a 3-vector from a string like "1 0 0" or "1,0,0".
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
	// TaskStaticSSSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticSSSpectra::TaskStaticSSSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(1.0), totaltime(1.0e+4), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn)
	{
	}

	TaskStaticSSSpectra::~TaskStaticSSSpectra()
	{
	}
	// -----------------------------------------------------
	// TaskStaticSSSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticSSSpectra::RunLocal()
	{
		this->Log() << "Running method StaticSS-Spectra." << std::endl;

		// If this is the first step, write first part of header to the data file
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Decline density matrix and density vector variables
		arma::cx_mat rho0;
		arma::cx_vec rho0vec;

		// Obtain spin systems
		auto systems = this->SpinSystems();

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
			space.UseSuperoperatorSpace(true);
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
								this->Log() << ", "; // Add a comma between elements
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

			// Convert initial state to superoperator space
			if (!space.OperatorToSuperspace(rho0, rho0vec))
			{
				this->Log() << "Failed to convert initial state density operator to superspace." << std::endl;
				continue;
			}

			// Get the Hamiltonian
			arma::sp_cx_mat H;
			if (!space.Hamiltonian(H))
			{
				this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
				continue;
			}

			// Get a matrix to collect all the terms (the total Liouvillian)
			arma::sp_cx_mat A = arma::cx_double(0.0, -1.0) * H;

			// Get the reaction operators, and add them to "A"
			arma::sp_cx_mat K;
			if (!space.TotalReactionOperator(K))
			{
				this->Log() << "Warning: Failed to obtain matrix representation of the reaction operators!" << std::endl;
			}
			A -= K;

			// Get the relaxation terms, assuming that they can just be added to the Liouvillian superoperator
			arma::sp_cx_mat R;
			for (auto j = (*i)->operators_cbegin(); j != (*i)->operators_cend(); j++)
			{
				if (space.RelaxationOperator((*j), R))
				{
					A += R;
					this->Log() << "Added relaxation operator \"" << (*j)->Name() << "\" to the Liouvillian.\n";
				}
			}

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
			this->BuildProjectionCache(*i, space, CIDSP, projection_cache, this->Log());

			// Read printtimeframe from the input file
			std::string Timewindow;
			if (!this->Properties()->Get("printtimeframe", Timewindow))
			{
				this->Log() << "Failed to obtain an input for a printtimeframe. Please choose printtimeframe =  pulse / freeevo / full. Using full propagation evolution window by default" << std::endl;
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

					for (auto pulse = (*i)->pulses_cbegin(); pulse < (*i)->pulses_cend(); pulse++)
					{
						if ((*pulse)->Name().compare(pulse_name) == 0)
						{
							// Apply a pulse to our density vector
							if ((*pulse)->Type() == SpinAPI::PulseType::InstantPulse)
							{
								// Create a Pulse operator in SS. Here pulse_operator is exp[-i * angle * (Vx * Sx  + Vy * Sy  + Vz * Sz )]
								arma::sp_cx_mat pulse_operator;
								if (!space.PulseOperator((*pulse), pulse_operator))
								{
									this->Log() << "Failed to create a pulse operator in SS." << std::endl;
									continue;
								}
								rhovec = pulse_operator * rhovec;

								// Since there is no time propagation, we don't print the result here
							}
							else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulseStaticField)
							{
								// Create a Pulse operator in SS . Here pulse_operator is Bx * Sx * gamma + By * Sy * gamma  + Bz * Sz * gamma
								arma::sp_cx_mat pulse_operator;
								if (!space.PulseOperator((*pulse), pulse_operator))
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

								if (A.n_rows <= 64) // for the systems not bigger than 2 electrons and 1 spin 1/2 nuclei
								{
									// Create array containing a propagator and the current state of each system
									std::pair<arma::sp_cx_mat, arma::cx_vec> G;
									// Get the propagator and put it into the array together with the initial state
									arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator)) * (*pulse)->Timestep())));
									G = std::pair<arma::sp_cx_mat, arma::cx_vec>(A_sp, rhovec);

									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
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
								else // Use Krylov propagation
								{
									// assign number of steps
									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));

									// Get the propagator
									arma::sp_cx_mat A_sp = A + (arma::cx_double(0.0, -1.0) * pulse_operator);

									// save the density on the current step
									arma::cx_vec tmp_rho = rhovec;

									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0)
										{
											rhovec = space.KrylovExpmGeneral(A_sp, tmp_rho, (*pulse)->Timestep(), 30, A_sp.n_rows); // dimension of the Krylov matrix is m=30.

											// Integrate the density vector over the current time interval
											if ((integration) && (Integrationwindow.compare("pulse") != 0))
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
							}
							else if ((*pulse)->Type() == SpinAPI::PulseType::LongPulse)
							{
								// Create a Pulse operator in SS. Here pulse_operator is Bx * Sx * gamma + By * Sy * gamma  + Bz * Sz * gamma
								arma::sp_cx_mat pulse_operator;
								if (!space.PulseOperator((*pulse), pulse_operator))
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

								// Here we don't save tupple anymore, because the A_sp needs to be constructed on every step

								// save the density on the current step
								arma::cx_vec tmp_rho = rhovec;

								if (A.n_rows <= 64) // for the systems not bigger than 2 electrons and 1 spin 1/2 nuclei
								{
									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0) // for the very first step just printing is done, no propagation
										{
											// Get current time
											double t = n * (*pulse)->Timestep();

											// Get propagator
											arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator * std::cos((*pulse)->Frequency() * t))) * (*pulse)->Timestep())));

											// Take a step
											rhovec = A_sp * tmp_rho;

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
								else // Use Krylov propagation
								{
									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0) // for the very first step just printing is done, no propagation
										{
											// Get current time
											double t = n * (*pulse)->Timestep();

											// Midpoint time (better approximation fos cos(w * t) )
											double t_mid = t + 0.5 * (*pulse)->Timestep();

											// Build the propagation matrix
											arma::sp_cx_mat A_sp = A + (arma::cx_double(0.0, -1.0) * std::cos((*pulse)->Frequency() * t_mid) * pulse_operator);

											// Take a step
											rhovec = space.KrylovExpmGeneral(A_sp, tmp_rho, (*pulse)->Timestep(), 30, A_sp.n_rows); // dimension of the Krylov matrix is m=30.

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
							}
							else if ((*pulse)->Type() == SpinAPI::PulseType::MWPulse)
							{
								// Create a holder vector for an averaged density
								arma::cx_vec rhoavg;
								rhoavg.zeros(size(rho0vec));

								// Define first propagarion step
								int firststep;
								if (Printedtime == 0)
									firststep = 0;
								else
									firststep = 1;

								// save the density on the current step
								arma::cx_vec tmp_rho = rhovec;

								if (A.n_rows <= 64) // for the systems not bigger than two electrons and one spin 1/2 nuclei
								{
									// Here we don't save tupple anymore, because the A_sp  and pulse need to be constructed on every step
									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0) // for the very first step just printing is done, no propagation
										{
											// Get current time
											double t = n * (*pulse)->Timestep();

											// Create a Pulse operator in SS. Here pulse_operator is exactly Bx * Sx * gamma * cos (omega * t) + By * Sy * gamma * sin (omega * t) + Bz * sz
											arma::sp_cx_mat pulse_operator;
											if (!space.PulseOperator_mw((*pulse), pulse_operator, t))
											{
												this->Log() << "Failed to create a pulse operator in SS." << std::endl;
												continue;
											}

											// Get the propagator
											arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from((A + (arma::cx_double(0.0, -1.0) * pulse_operator)) * (*pulse)->Timestep())));

											// Take a step
											rhovec = A_sp * tmp_rho;

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
								else // Use Krylov propagation
								{
									unsigned int steps = static_cast<unsigned int>(std::abs((*pulse)->Pulsetime() / (*pulse)->Timestep()));
									for (unsigned int n = firststep; n <= steps; n++)
									{
										if (n != 0) // for the very first step just printing is done, no propagation
										{
											// Get current time
											double t = n * (*pulse)->Timestep();

											// Midpoint time (better approximation fos cos(w * t) )
											double t_mid = t + 0.5 * (*pulse)->Timestep();

											// Create a Pulse operator in SS. Here pulse_operator is exactly Bx * Sx * gamma * cos (omega * t) + By * Sy * gamma * sin (omega * t) + Bz * sz
											arma::sp_cx_mat pulse_operator;
											if (!space.PulseOperator_mw((*pulse), pulse_operator, t_mid))
											{
												this->Log() << "Failed to create a pulse operator in SS." << std::endl;
												continue;
											}

											// Build the propagation matrix
											arma::sp_cx_mat A_sp = A + (arma::cx_double(0.0, -1.0) * pulse_operator);

											// Take a step
											rhovec = space.KrylovExpmGeneral(A_sp, tmp_rho, (*pulse)->Timestep(), 30, A_sp.n_rows); // dimension of the Krylov matrix is m=30.

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

								if (A.n_rows <= 64) // for the systems not bigger than 2 electrons and 1 spin 1/2 nuclei
								{

									// Create array containing a propagator and the current state of each system
									std::pair<arma::sp_cx_mat, arma::cx_vec> G;
									arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from(A * (*pulse)->Timestep())));
									// Get the propagator and put it into the array together with the initial state
									G = std::pair<arma::sp_cx_mat, arma::cx_vec>(A_sp, rhovec);

									unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / (*pulse)->Timestep()));
									for (unsigned int n = 1; n <= steps; n++) // n always starts with one here, because this part cannot be done without the pulse part on top
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

										if (Timewindow.compare("freeevo") != 0)
										{
											if (!this->ProjectAndPrintOutputLine(i, space, projection_cache, rhovec, Printedtime, (*pulse)->Timestep(), n, CIDSP, this->Data(), this->Log()))
												this->Log() << "Could not project the state vector and print the result into a file" << std::endl;
										}
									}
								}
								else // Use Krylov propagation
								{
									// assign number of steps
									unsigned int steps = static_cast<unsigned int>(std::abs(timerelaxation / (*pulse)->Timestep()));

									// save the density on the current step
									arma::cx_vec tmp_rho = rhovec;

									for (unsigned int n = 1; n <= steps; n++) //n always starts with one here, because this part cannot be done without the pulse part on top
									{
										if (n != 0)
										{
											rhovec = space.KrylovExpmGeneral(A, tmp_rho, (*pulse)->Timestep(), 30, A.n_rows); // dimension of the Krylov matrix is m=30.

											// Integrate the density vector over the current time interval
											if ((integration) && (Integrationwindow.compare("pulse") != 0))
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
							}
							// Update the printed time according to the printtimeframe key
							if (Timewindow.compare("freeevo") != 0)
							{
								Printedtime += timerelaxation;
							}
						}
					}
				}
			}

			// Free time evolution propagation after the pulse sequience

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

					if (A.n_rows <= 64) // for the systems not bigger than 2 electrons and 1 spin 1/2 nuclei
					{
						// Create array containing a propagator and the current state of each system
						std::pair<arma::sp_cx_mat, arma::cx_vec> G;
						arma::sp_cx_mat A_sp = arma::conv_to<arma::sp_cx_mat>::from(arma::expmat(arma::conv_to<arma::cx_mat>::from(A * this->timestep)));
						// Get the propagator and put it into the array together with the initial state
						G = std::pair<arma::sp_cx_mat, arma::cx_vec>(A_sp, rhovec);

						unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));
						for (unsigned int n = firststep; n <= steps; n++)
						{
							if (n != 0)
							{
								// Take a step, "first" is propagator and "second" is current state
								rhovec = G.first * G.second;

								// Integrate the density vector over the current time interval
								if ((integration) && (Integrationwindow.compare("pulse") != 0))
								{
									rhoavg += this->timestep * (G.second + rhovec) / 2;
								}

								// Get the new current state density vector
								G.second = rhovec;

								// Save the result if there were some changes
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
					else // Use Krylov propagation
					{
						// assign number of steps
						unsigned int steps = static_cast<unsigned int>(std::abs(this->totaltime / this->timestep));

						// save the density on the current step
						arma::cx_vec tmp_rho = rhovec;

						for (unsigned int n = firststep; n <= steps; n++)
						{
							if (n != 0)
							{
								rhovec = space.KrylovExpmGeneral(A, tmp_rho, this->timestep, 30, A.n_rows); // dimension of the Krylov matrix is m=30.

								// Integrate the density vector over the current time interval
								if ((integration) && (Integrationwindow.compare("pulse") != 0))
								{
									rhoavg += this->timestep * (tmp_rho + rhovec) / 2;
								}

								// Get the new current state density vector
								tmp_rho = rhovec;

								// Save the result if there were some changes
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
	void TaskStaticSSSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		// Match the column labels to the operator frame used in BuildProjectionCache.
		std::string outputframe = "lab";
		(void)this->Properties()->Get("outputframe", outputframe);
		std::string Lx = ".Ix ";
		std::string Ly = ".Iy ";
		std::string Lz = ".Iz ";
		std::string Lp = ".Ip ";
		std::string Lm = ".Im ";
		if (outputframe == "cw")
		{
			// In CW-RWA mode we interpret the printed operators as:
			//   x -> drive axis (e1)
			//   y -> absorptive quadrature axis (e2)
			//   z -> quantization axis (n)
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
							// Yields are written per transition
							// bool CIDSP = false;
							if (this->Properties()->Get("cidsp", CIDSP) && CIDSP == true)
							{
								// Write each transition name
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
								// Write each state name
								auto states = (*i)->States();
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
	bool TaskStaticSSSpectra::Validate()
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

	bool TaskStaticSSSpectra::GetEigenvectors_H0(SpinAPI::SpinSpace &_space, arma::vec &_eigen_val, arma::sp_cx_mat &_eigen_vec_sp) const
	{
		_space.UseSuperoperatorSpace(false);

		arma::cx_mat H;

		if (!_space.Hamiltonian(H))
		{
			// this->Log() << "Failed to obtain Static Hamiltonian in Hilbert Space." << std::endl;
		}

		arma::cx_mat _eigen_vec;

		// ----------------------------------------------------------------
		// DIAGONALIZATION OF H0// We need all of these operators
		// ----------------------------------------------------------------
		// this->Log() << "Starting diagonalization..." << std::endl;
		arma::eig_sym(_eigen_val, _eigen_vec, (H));
		// this->Log() << "Diagonalization done! Eigenvalues: " << _eigen_val.n_elem << ", eigenvectors: " << _eigen_vec.n_cols << std::endl;

		_eigen_vec_sp = arma::conv_to<arma::sp_cx_mat>::from(_eigen_vec);

		_space.UseSuperoperatorSpace(true);

		return true;
	}

	arma::cx_vec TaskStaticSSSpectra::ComputeRhoDot(double t, arma::sp_cx_mat &L, arma::cx_vec &K, arma::cx_vec RhoNaught)
	{
		arma::cx_vec ReturnVec(L.n_rows);
		RhoNaught = RhoNaught + K;
		ReturnVec = L * RhoNaught;
		return ReturnVec;
	}

	bool TaskStaticSSSpectra::BuildProjectionCache(const SpinAPI::system_ptr &_system, SpinAPI::SpinSpace &_space, bool _cidsp, ProjectionCache &_cache, std::ostream &_log_stream)
	{
		_cache = ProjectionCache{};

		// Optional: output observables in an orientation-dependent CW-RWA frame.
		// This is a PURE operator-frame change (linear combinations of Sx,Sy,Sz) and therefore
		// stays consistent with any Hamiltonian tensor rotation that was performed externally.
		//
		// outputframe = lab | cw
		//   lab: prints <Sx>,<Sy>,<Sz>,<S+>,<S-> (current behaviour)
		//   cw : prints <S_e1>,<S_e2>,<S_n>,<S+_(e1,e2)>,<S-_(e1,e2)>
		//        where n is the Zeeman quantization axis derived from the (already rotated) g-tensor,
		//        e1 is the transverse drive axis, and e2 = n x e1 is the absorptive quadrature axis.
		//
		// b0dir / b1dir are lab-frame directions (defaults: z and x)
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

					// If requested, rotate the OUTPUT OPERATORS into the CW-RWA frame.
					// Note: We do NOT rotate the density matrix here; Tr(O_rot rho) = Tr(O rho_rot)
					// so rotating the observable operators is sufficient and avoids costly basis transforms.
					//
					// Applied to ELECTRONS only: the microwave rotating-frame construction is defined for electron spins.
					if (outputframe == "cw" && (*l)->Type() == SpinAPI::SpinType::Electron)
					{
						// Quantization axis from (already rotated) g-tensor.
						// n ~ G^T * b0dir
						arma::mat G = (*l)->GetTensor().LabFrame();
						arma::vec a = G.t() * b0dir;
						arma::vec n = SafeNormalise(a, b0dir);

						// Drive axis from transverse component of v ~ G^T * b1dir
						arma::vec v = G.t() * b1dir;
						arma::vec v_perp = v - arma::dot(v, n) * n;
						arma::vec e1 = SafeNormalise(v_perp, b1dir);

						// Absorptive quadrature axis
						arma::vec e2 = arma::cross(n, e1);
						e2 = SafeNormalise(e2, arma::vec({0.0, 1.0, 0.0}));

						// Build rotated spin operators: S_e1, S_e2, S_n.
						arma::cx_mat I_e1 = e1(0) * Iprojx + e1(1) * Iprojy + e1(2) * Iprojz;
						arma::cx_mat I_e2 = e2(0) * Iprojx + e2(1) * Iprojy + e2(2) * Iprojz;
						arma::cx_mat I_n = n(0) * Iprojx + n(1) * Iprojy + n(2) * Iprojz;

						// Raising/lowering operators in the (e1,e2) transverse plane
						arma::cx_double I(0.0, 1.0);
						arma::cx_mat I_plus = I_e1 + I * I_e2;
						arma::cx_mat I_minus = I_e1 - I * I_e2;

						Iprojx = std::move(I_e1);
						Iprojy = std::move(I_e2);
						Iprojz = std::move(I_n);
						Iprojp = std::move(I_plus);
						Iprojm = std::move(I_minus);
					}

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
				// Make sure that there is a state object
				if ((*j)->SourceState() == nullptr)
					continue;

				arma::cx_mat P;
				if (!_space.GetState((*j)->SourceState(), P))
				{
					_log_stream << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\" of SpinSystem \"" << _system->Name() << "\"." << std::endl;
					continue;
				}

				_cache.transition_proj.push_back(std::move(P));
				_cache.transition_rates.push_back((*j)->Rate());
			}
		}

		_cache.ready = true;
		return true;
	}

	bool TaskStaticSSSpectra::ProjectAndPrintOutputLine(SystemIterator &_i, SpinAPI::SpinSpace &_space, const ProjectionCache &_cache, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, unsigned int &_n, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		(void)_i;

		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)) && (_n == 0))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		if (_n == 0)
			_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
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

		// There are two result modes - either write results per transition if CIDSP is true or for each defined state if CIDSP is false.
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

	bool TaskStaticSSSpectra::ProjectAndPrintOutputLineInf(SystemIterator &_i, SpinAPI::SpinSpace &_space, const ProjectionCache &_cache, arma::cx_vec &_rhovec, double &_printedtime, double _timestep, bool &_cidsp, std::ostream &_datastream, std::ostream &_logstream)
	{
		(void)_i;

		arma::cx_mat rho0;

		// Convert the resulting density operator back to its Hilbert space representation
		if ((!_space.OperatorFromSuperspace(_rhovec, rho0)))
		{
			_logstream << "Failed to convert resulting superspace-vector back to native Hilbert space." << std::endl;
			return false;
		}

		_logstream << "CIDSP = " << _cidsp << std::endl;

		// Save the current step
		_datastream << this->RunSettings()->CurrentStep() << " ";
		// Save the current time
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

		// There are two result modes - either write results per transition if CIDSP is true or for each defined state if CIDSP is false.
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

	// -----------------------------------------------------
}
