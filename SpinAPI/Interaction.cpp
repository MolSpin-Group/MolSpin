/////////////////////////////////////////////////////////////////////////
// Interaction class (SpinAPI Module)
// ------------------
// Base class for interactions.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
///////////////////////////////////////////////////////////////////////////
#include <memory>
#include <iostream>
#include <cmath>
#include <Spin.h>
#include "ObjectParser.h"
#include "Interaction.h"

namespace SpinAPI
{
	// -----------------------------------------------------
	// Interaction Constructors and Destructor
	// -----------------------------------------------------
	// The constructor sets up the interaction parameters, but
	// the spin groups are read in the method ParseSpinGroups instead.
	Interaction::Interaction(std::string _name, std::string _contents) : properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents)), couplingTensor(nullptr),
																		 field({0, 0, 0}), dvalue(0.0), evalue(0.0), EnergyShift(true), group1(), group2(), type(InteractionType::Undefined), fieldType(InteractionFieldType::Static), prefactor(1.0), addCommonPrefactor(true), ignoreTensors(false), isValid(true),
																		 trjHasTime(false), trjHasField(false), trjHasTensor(false), trjHasPrefactor(false), trjTime(0), trjFieldX(0), trjFieldY(0), trjFieldZ(0), trjPrefactor(0),
																		 tdFrequency(1.0), tdPhase(0.0), tdAxis("0 0 1"), tdPerpendicularOscillation(false), tdInitialField({0, 0, 0}), tensorType(InteractionTensorType::Static), tdTimestep(0),
																		 tdInitialTensor(3, 3, arma::fill::zeros), tdMinFreq(0.0), tdMaxFreq(0.0), tdFreqs(), tdAmps(), tdPhases(), tdComponents(0), tdRandOrients(false), tdThetas(), tdPhis(), tdCorrTime(0.0),
																		 tdPrintTensor(false), tdPrintField(false), hffield(), orientations(0), OriWeights(), BondLengths(), Spacing(), tau(0.0), dist(), tdSeed(0), tdAutoseed(false), tdGenerator(1), framelist({0, 0, 0}), f() 
	{
		// Is a trajectory specified?
		std::string str;
		if (this->properties->Get("trajectory", str))
		{
			// Get the directory
			std::string directory;
			if (!this->properties->Get("FileReaderDirectory", directory))
			{
				std::cout << "Warning: Failed to obtain input file directory while initializing Interaction " << this->Name() << "!\n";
				std::cout << "         This may lead to problems when trying to localize the trajectory file." << std::endl;
			}

			// Load the trajectory, and check whether something was loaded
			if (this->trajectory.Load(directory + str))
			{
				if (this->trajectory.Length() > 0)
				{
					// Get column numbers
					this->trjHasTime = this->trajectory.HasColumn("time", trjTime);
					this->trjHasPrefactor = this->trajectory.HasColumn("prefactor", trjPrefactor);
					this->trjHasField = this->trajectory.HasColumn("field.x", trjFieldX);
					this->trjHasField &= this->trajectory.HasColumn("field.y", trjFieldY);
					this->trjHasField &= this->trajectory.HasColumn("field.z", trjFieldZ);
				}
			}
			else
			{
				std::cout << "ERROR: Failed to load trajectory \"" << (directory + str) << "\" for Interaction object \"" << this->Name() << "\"!" << std::endl;
			}
		}

		// Get the type of the interaction
		if (this->properties->Get("type", str))
		{

			if (str.compare("onespin") == 0 || str.compare("singlespin") == 0 || str.compare("zeeman") == 0)
			{

				// A onespin interaction should have a field attached, either directly or through a trajectory
				arma::vec inField = arma::zeros<arma::vec>(3);

				this->properties->Get("field", inField);

				if (this->properties->Get("field", inField) || this->trjHasField)
				{
					this->field = inField;
					this->type = InteractionType::SingleSpin;
				}
				std::vector<double> _framelist;
				if (this->Properties()->GetList("orientation", _framelist))
				{
					this->framelist = _framelist;
				}
			}
			else if (str.compare("twospin") == 0 || str.compare("doublespin") == 0 || str.compare("hyperfine") == 0 || str.compare("dipole") == 0)
			{
				this->type = InteractionType::DoubleSpin;

				std::vector<double> _framelist;
				if (this->Properties()->GetList("orientation", _framelist))
				{
					this->framelist = _framelist;
				}
			}
			else if (str.compare("quadraticspin") == 0)
			{
				this->type = InteractionType::QuadraticSpin;
			}
			else if (str.compare("exchange") == 0)
			{
				this->type = InteractionType::Exchange;
			}
			else if (str.compare("zfs") == 0)
			{
				this->type = InteractionType::Zfs;

				double indvalue, inevalue;
				this->Properties()->Get("dvalue", indvalue);
				this->Properties()->Get("evalue", inevalue);
				std::string shift;
				if (this->Properties()->Get("energyshift", shift))
				{
					if (shift == "true")
						this->EnergyShift = true;
					else if (shift == "false")
						this->EnergyShift = false;
				}

				this->dvalue = indvalue;
				this->evalue = inevalue;
			}
			else if (str.compare("semiclassicalfield") == 0)
			{
				this->type = InteractionType::SemiClassicalField;

				int inorientationsvalue;
				this->Properties()->Get("orientations", inorientationsvalue);
				this->orientations = inorientationsvalue;
				std::string HFfieldStr;
				this->Properties()->Get("hyperfinefield", HFfieldStr);
				std::vector<std::tuple<SCMatrix3x3, int, double>> HFfield;
				bool InTuple = false;
				int element = 0;
				std::string CurrentString;
				SCMatrix3x3 ZeroTensor = {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
				std::tuple<SCMatrix3x3, int, double> CurrentHfTensor = std::make_tuple(ZeroTensor, -1, -1.0); //{{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}, -1, -1.0};
				auto c = HFfieldStr.begin();
				int depth = 0;
				int validelements = 0;
				while (c != HFfieldStr.end())
				{
					if ((*c) == '(' || (*c) == ')')
					{
						if ((*c) == '(')
						{
							depth += 1;
							if (depth == 1)
							{
								InTuple = true;
								CurrentString = "";
							}
							else
							{
								CurrentString += (*c);
							}
						}
						else
						{
							depth -= 1;
							if (depth == 0)
							{
								InTuple = false;
							}
							else
							{
								CurrentString += (*c);
							}
						}
						if (!InTuple && CurrentString != "")
						{
							c = c - 1;
							(*c) = ',';
							InTuple = true;
							depth = 1;
							continue;
						}
						if (!InTuple && validelements < 3)
						{
							std::cout << "[INFO]: Not enough valid elements in hyperfine field definition for the semi classical interaction." << this->Name();
							auto [a, n, s] = CurrentHfTensor;
							std::cout << " Ignoring the field defined as: " << a << " , " << n << " , " << s << "\n -1 is just used as a placeholder value " << std::endl;
							CurrentHfTensor = std::make_tuple(ZeroTensor, -1, -1.0);
							validelements = 0;
						}
						if (!InTuple && validelements == 3)
						{
							HFfield.push_back(CurrentHfTensor);
							CurrentHfTensor = std::make_tuple(ZeroTensor, -1, -1.0);
							validelements = 0;
						}
						element = 0;
						c++;
					}
					if (!InTuple && (*c) == ',')
					{
						c++;
						continue;
					}
					if (!InTuple && (*c) == ' ')
					{
						c++;
						continue;
					}
					if (InTuple)
					{
						// depth check i.e if depth is 2 we are in tensor
						if (depth == 2)
						{
							CurrentString += (*c);
						}
						else if ((*c) != ',' && (*c) != ' ')
						{
							CurrentString += (*c);
						}
						if (depth == 1 && (*c) == ',')
						{
							// evaluate strings
							if (element == 0)
							{
								// parse tensor
								SCMatrix3x3 parsedtensor;
								bool valid = Parse3x3TensorLikeArrayFromString(CurrentString, parsedtensor);
								std::get<0>(CurrentHfTensor) = parsedtensor;
								if (valid)
								{
									validelements += 1;
								}
							}
							else if (element == 1)
							{
								int num = std::stoi(CurrentString);
								std::get<1>(CurrentHfTensor) = num;
								validelements += 1;
							}
							else if (element == 2)
							{

								auto frac = [](std::string str)
								{
									std::string p1, p2;
									std::string *cstr = &p1;
									auto c2 = str.begin();
									while (c2 != str.end())
									{
										if ((*c2) == '/' || (*c2) == '\\')
										{
											cstr = &p2;
											c2++;
											continue;
										}
										(*cstr) += (*c2);
										c2++;
									}
									return std::stod(p1) / std::stod(p2);
								};
								double sn = 0.0;
								if (std::find(CurrentString.begin(), CurrentString.end(), '/') != CurrentString.end())
								{
									sn = frac(CurrentString);
								}
								else
								{
									sn = std::stod(CurrentString);
								}
								std::get<2>(CurrentHfTensor) = sn;
								validelements += 1;
							}
							else
							{
								std::cout << "[INFO]: Too many elements in hyperfine field definition for the semi classical interaction " << this->Name() << " ignoring value: " << CurrentString << std::endl;
							}
							CurrentString = "";
							element += 1;
							c++;
							continue;
						}
						c++;
					}
				}
				this->hffield = HFfield;

				std::string distribution; // slighty overkill as we only have one distribution
				if (!this->Properties()->Get("distribution", distribution) && !this->Properties()->Get("dist", distribution))
				{
					this->dist = SCDistribution::DEFUALT;
				}
				else
				{
					if (distribution == "fjc" || distribution == "freelyjointedchain")
					{
						this->dist = SCDistribution::FJC;
					}
					else
					{
						this->dist = SCDistribution::DEFUALT;
					}
				}

				if (this->dist == SCDistribution::FJC || this->dist == SCDistribution::DEFUALT)
				{
					FreelyJointedPolymerBL(this->BondLengths, this->hffield, this->tau, this->orientations);

					double prefactor = this->tau * 0.25 * M_1_PI; // tua^2 / 4pi
					prefactor = std::pow(prefactor, 3.0 / 2.0);
					double exp1 = -0.25 * this->tau;
					this->f = FreelyJointedPolymerD(prefactor, exp1);
				}
			}
		}

		// If we have a valid interaction type, read the other parameters
		if (this->type != InteractionType::Undefined)
		{

			Tensor inTensor(0);
			if (this->properties->Get("tensor", inTensor))
			{
				this->couplingTensor = std::make_shared<Tensor>(inTensor);
			}

			this->properties->Get("prefactor", this->prefactor);
			this->properties->Get("commonprefactor", this->addCommonPrefactor);
			this->properties->Get("ignoretensors", this->ignoreTensors);
		}

		// One-spin interactions can have a time-dependent field
		if (this->type == InteractionType::SingleSpin)
		{

			// Do we have a trajectory entry for field? Note that the trajectory may not have a time column, in which case there will be no time dependence
			if (this->trjHasField)
			{
				this->fieldType = InteractionFieldType::Trajectory;
			}
			// do we have a sepcified fieldtype?
			if (this->properties->Get("fieldtype", str) || this->properties->Get("timedependence", str))
			{

				// some fieldtypes require a seed to be specified for random number generation
				if (str.compare("broadband") == 0 || str.compare("ougeneral") == 0)
				{

					// Random Number Generator Preparation
					std::random_device rand_dev;						 // random number generator
					this->tdGenerator.seed(rand_dev());					 // automatic seed
					this->properties->Get("autoseed", this->tdAutoseed); // seed can also be specified by the user
					if (!this->tdAutoseed)
					{
						std::cout << "Autoseed is off." << std::endl;
						this->properties->Get("seed", this->tdSeed);
						if (this->tdSeed != 0)
						{
							this->tdGenerator.seed(this->tdSeed);
							std::cout << "Seed number is " << this->tdSeed << "." << std::endl;
						}
						else
						{
							std::cout << "# ERROR: undefined seed number in Interaction " << this->Name() << ". Setting to default of 1." << std::endl;
							this->tdSeed = 1;
						}
					}
				}

				if (this->trjHasField)
				{
					std::cout << "Warning: Ignored fieldtype for Interaction \"" << this->Name() << "\"! Using trajectory instead." << std::endl;
				}
				// Figure out what type of time-dependence should be used
				else if (str.compare("linearpolarized") == 0 || str.compare("linearpolarization") == 0 || str.compare("oscillating") == 0 || str.compare("monochromatic") == 0)
				{
					this->fieldType = InteractionFieldType::LinearPolarization;
					this->properties->Get("frequency", this->tdFrequency);
					this->properties->Get("phase", this->tdPhase);
					this->properties->Get("printfield", this->tdPrintField);
				}
				else if (str.compare("circularpolarized") == 0 || str.compare("circularpolarization") == 0)
				{
					this->fieldType = InteractionFieldType::CircularPolarization;
					this->properties->Get("frequency", this->tdFrequency);
					this->properties->Get("phase", this->tdPhase);
					this->properties->Get("axis", this->tdAxis);
					this->properties->Get("perpendicularoscillations", this->tdPerpendicularOscillation);
					this->properties->Get("printfield", this->tdPrintField);
				}
				else if (str.compare("broadband") == 0)
				{
					this->fieldType = InteractionFieldType::Broadband;
					this->properties->Get("minfreq", this->tdMinFreq);
					this->properties->Get("maxfreq", this->tdMaxFreq);
					this->properties->Get("components", this->tdComponents);
					this->properties->Get("printfield", this->tdPrintField);
					this->properties->Get("randomorientations", this->tdRandOrients);

					// define all the distributions for broadband noise and fill out vectors
					std::normal_distribution<double> amp_dist(0.0, 1.0);
					std::uniform_real_distribution<double> phase_dist(0, 2.0 * M_PI);
					std::uniform_real_distribution<double> freq_dist(this->tdMinFreq, this->tdMaxFreq);

					this->tdAmps.set_size(this->tdComponents, 3);
					this->tdFreqs.set_size(this->tdComponents, 3);
					this->tdPhases.set_size(this->tdComponents, 3);

					// generate all the required random numbers for the 6 unique tensor components
					for (int j_tens = 0; j_tens < 3; j_tens++)
					{
						for (int i_comp = 0; i_comp < this->tdComponents; i_comp++)
						{
							double phase = phase_dist(this->tdGenerator);
							double freq = freq_dist(this->tdGenerator);
							double amp = amp_dist(this->tdGenerator);

							this->tdPhases(i_comp, j_tens) = phase;
							this->tdFreqs(i_comp, j_tens) = freq;
							this->tdAmps(i_comp, j_tens) = amp;
						}
					}

					// generate random angles if required
					if (this->tdRandOrients == true)
					{

						this->tdThetas.set_size(this->tdComponents, 3);
						this->tdPhis.set_size(this->tdComponents, 3);
						std::uniform_real_distribution<double> cos_theta_dist(-1.0, 1.0);
						std::uniform_real_distribution<double> phi_dist(0.0, 2 * M_PI);
						for (int j_tens = 0; j_tens < 3; j_tens++)
						{
							for (int i_comp = 0; i_comp < this->tdComponents; i_comp++)
							{
								double theta_BB = std::acos(cos_theta_dist(this->tdGenerator));
								double phi_BB = phi_dist(this->tdGenerator);

								this->tdThetas(i_comp, j_tens) = theta_BB;
								this->tdPhis(i_comp, j_tens) = phi_BB;
							}
						}
					}
				}
				else if (str.compare("ougeneral") == 0)
				{
					this->fieldType = InteractionFieldType::OUGeneral;
					this->properties->Get("correlationtime", this->tdCorrTime);
					this->properties->Get("timestep", this->tdTimestep);
					this->properties->Get("printfield", this->tdPrintField);
				}

				else
				{
					std::cout << "Warning: Unknown fieldtype for Interaction \"" << this->Name() << "\"! Assuming static field." << std::endl;
				}
			}

			// Save the initial field - used only for time-dependent interactions where Actions cannot change the field
			if (this->fieldType == InteractionFieldType::OUGeneral)
			{
				this->tdInitialField = this->field;
				this->field = arma::zeros<arma::vec>(3);
			}
			else
			{
				this->tdInitialField = this->field;
			}
			// Set the time to 0 by default
			if (this->HasFieldTimeDependence())
			{
				this->SetTime(0.0);
			}
		}

		// Double-spin interactions can have a time-dependent tensor
		if (this->type == InteractionType::DoubleSpin)
		{
			// do we have a specified tensortype?
			if (this->properties->Get("tensortype", str))
			{
				// some tensortypes require a seed to be specified for random number generation
				if (str.compare("broadband") == 0 || str.compare("ougeneral") == 0)
				{
					// Random Number Generator Preparation
					std::random_device rand_dev;						 // random number generator
					this->tdGenerator.seed(rand_dev());					 // automatic seed
					this->properties->Get("autoseed", this->tdAutoseed); // seed can also be specified by the user

					if (!this->tdAutoseed)
					{
						std::cout << "Autoseed is off." << std::endl;
						this->properties->Get("seed", this->tdSeed);
						if (this->tdSeed != 0)
						{
							this->tdGenerator.seed(this->tdSeed);
							std::cout << "Seed number is " << this->tdSeed << "." << std::endl;
						}
						else
						{
							std::cout << "# ERROR: undefined seed number in Interaction " << this->Name() << ". Setting to default of 1." << std::endl;
							this->tdSeed = 1;
						}
					}
				}

				// Do we have a trajectory entry for tensor? Note that the trajectory may not have a time column, in which case there will be no time dependence
				if (this->trjHasTensor)
				{
					this->tensorType = InteractionTensorType::Trajectory;
					std::cout << "Warning: Ignored tensortype for Interaction \"" << this->Name() << "\"! Using trajectory instead." << std::endl;
				}

				// Figure out what type of time-dependence should be used
				else if (str.compare("monochromatic") == 0)
				{
					this->tensorType = InteractionTensorType::Monochromatic;
					this->properties->Get("frequency", this->tdFrequency);
					this->properties->Get("phase", this->tdPhase);
					this->properties->Get("printtensor", this->tdPrintTensor);
				}
				else if (str.compare("broadband") == 0)
				{
					this->tensorType = InteractionTensorType::Broadband;
					this->properties->Get("minfreq", this->tdMinFreq);
					this->properties->Get("maxfreq", this->tdMaxFreq);
					this->properties->Get("components", this->tdComponents);
					this->properties->Get("printtensor", this->tdPrintTensor);

					// define all the distributions for broadband noise and fill out vectors
					std::normal_distribution<double> amp_dist(0.0, 1.0);
					std::uniform_real_distribution<double> phase_dist(0, 2.0 * M_PI);
					std::uniform_real_distribution<double> freq_dist(this->tdMinFreq, this->tdMaxFreq);

					this->tdAmps.set_size(this->tdComponents, 6);
					this->tdFreqs.set_size(this->tdComponents, 6);
					this->tdPhases.set_size(this->tdComponents, 6);

					// generate all the required random numbers for the 6 unique tensor components
					for (int j_tens = 0; j_tens < 6; j_tens++)
					{
						for (int i_comp = 0; i_comp < this->tdComponents; i_comp++)
						{
							double phase = phase_dist(this->tdGenerator);
							double freq = freq_dist(this->tdGenerator);
							double amp = amp_dist(this->tdGenerator);

							this->tdPhases(i_comp, j_tens) = phase;
							this->tdFreqs(i_comp, j_tens) = freq;
							this->tdAmps(i_comp, j_tens) = amp;
						}
					}
				}

				else if (str.compare("ougeneral") == 0)
				{
					this->tensorType = InteractionTensorType::OUGeneral;
					this->properties->Get("correlationtime", this->tdCorrTime);
					this->properties->Get("timestep", this->tdTimestep);
					this->properties->Get("printtensor", this->tdPrintTensor);
				}

				else
				{
					std::cout << "Warning: Unknown tensortype for Interaction \"" << this->Name() << "\"! Assuming static field." << std::endl;
				}
			}

			// Set the time to 0 by default
			if (this->HasTensorTimeDependence())
			{
				if (this->tensorType == InteractionTensorType::OUGeneral)
				{
					this->tdInitialTensor = couplingTensor->LabFrame();
					arma::mat zeros(3, 3, arma::fill::zeros);
					this->couplingTensor->SetTensor(zeros);
				}
				else
				{
					this->tdInitialTensor = couplingTensor->LabFrame();
				}
				this->SetTime(0.0);
			}
		}

		// Quadratic-spin interactions can have a time-dependent tensor
		if (this->type == InteractionType::QuadraticSpin)
		{
			// do we have a specified tensortype?
			if (this->properties->Get("tensortype", str))
			{
				// some tensortypes require a seed to be specified for random number generation
				if (str.compare("broadband") == 0 || str.compare("ougeneral") == 0)
				{
					// Random Number Generator Preparation
					std::random_device rand_dev;						 // random number generator
					this->tdGenerator.seed(rand_dev());					 // automatic seed
					this->properties->Get("autoseed", this->tdAutoseed); // seed can also be specified by the user

					if (!this->tdAutoseed)
					{
						std::cout << "Autoseed is off." << std::endl;
						this->properties->Get("seed", this->tdSeed);
						if (this->tdSeed != 0)
						{
							this->tdGenerator.seed(this->tdSeed);
							std::cout << "Seed number is " << this->tdSeed << "." << std::endl;
						}
						else
						{
							std::cout << "# ERROR: undefined seed number in Interaction " << this->Name() << ". Setting to default of 1." << std::endl;
							this->tdSeed = 1;
						}
					}
				}

				// Do we have a trajectory entry for tensor? Note that the trajectory may not have a time column, in which case there will be no time dependence
				if (this->trjHasTensor)
				{
					this->tensorType = InteractionTensorType::Trajectory;
					std::cout << "Warning: Ignored tensortype for Interaction \"" << this->Name() << "\"! Using trajectory instead." << std::endl;
				}

				// Figure out what type of time-dependence should be used
				else if (str.compare("monochromatic") == 0)
				{
					this->tensorType = InteractionTensorType::Monochromatic;
					this->properties->Get("frequency", this->tdFrequency);
					this->properties->Get("phase", this->tdPhase);
					this->properties->Get("printtensor", this->tdPrintTensor);
				}
				else if (str.compare("broadband") == 0)
				{
					this->tensorType = InteractionTensorType::Broadband;
					this->properties->Get("minfreq", this->tdMinFreq);
					this->properties->Get("maxfreq", this->tdMaxFreq);
					this->properties->Get("components", this->tdComponents);
					this->properties->Get("printtensor", this->tdPrintTensor);

					// define all the distributions for broadband noise and fill out vectors
					std::normal_distribution<double> amp_dist(0.0, 1.0);
					std::uniform_real_distribution<double> phase_dist(0, 2.0 * M_PI);
					std::uniform_real_distribution<double> freq_dist(this->tdMinFreq, this->tdMaxFreq);

					this->tdAmps.set_size(this->tdComponents, 6);
					this->tdFreqs.set_size(this->tdComponents, 6);
					this->tdPhases.set_size(this->tdComponents, 6);

					// generate all the required random numbers for the 6 unique tensor components
					for (int j_tens = 0; j_tens < 6; j_tens++)
					{
						for (int i_comp = 0; i_comp < this->tdComponents; i_comp++)
						{
							double phase = phase_dist(this->tdGenerator);
							double freq = freq_dist(this->tdGenerator);
							double amp = amp_dist(this->tdGenerator);

							this->tdPhases(i_comp, j_tens) = phase;
							this->tdFreqs(i_comp, j_tens) = freq;
							this->tdAmps(i_comp, j_tens) = amp;
						}
					}
				}

				else if (str.compare("ougeneral") == 0)
				{
					this->tensorType = InteractionTensorType::OUGeneral;
					this->properties->Get("correlationtime", this->tdCorrTime);
					this->properties->Get("timestep", this->tdTimestep);
					this->properties->Get("printtensor", this->tdPrintTensor);
				}

				else
				{
					std::cout << "Warning: Unknown tensortype for Interaction \"" << this->Name() << "\"! Assuming static field." << std::endl;
				}
			}

			// Set the time to 0 by default
			if (this->HasTensorTimeDependence())
			{
				if (this->tensorType == InteractionTensorType::OUGeneral)
				{
					this->tdInitialTensor = couplingTensor->LabFrame();
					arma::mat zeros(3, 3, arma::fill::zeros);
					this->couplingTensor->SetTensor(zeros);
				}
				else
				{
					this->tdInitialTensor = couplingTensor->LabFrame();
				}
				this->SetTime(0.0);
			}
		}
	}

	std::ostream &operator<<(std::ostream &stream, const SCMatrix3x3 &mat)
	{
		stream << "\n[[" << mat[0][0] << ", " << mat[0][1] << ", " << mat[0][2] << "]\n "
			   << "[" << mat[1][0] << ", " << mat[1][1] << ", " << mat[1][2] << "]\n "
			   << "[" << mat[2][0] << ", " << mat[2][1] << ", " << mat[2][2] << "]]\n";
		return stream;
	}

	Interaction::Interaction(const Interaction &_interaction) : properties(_interaction.properties), couplingTensor(_interaction.couplingTensor), field(_interaction.field), dvalue(_interaction.dvalue), evalue(_interaction.evalue), EnergyShift(_interaction.EnergyShift),
																group1(_interaction.group1), group2(_interaction.group2), type(_interaction.type), fieldType(_interaction.fieldType),
																prefactor(_interaction.prefactor), addCommonPrefactor(_interaction.addCommonPrefactor), ignoreTensors(_interaction.ignoreTensors), isValid(_interaction.isValid),
																trjHasTime(_interaction.trjHasTime), trjHasField(_interaction.trjHasField), trjHasTensor(_interaction.trjHasTensor), trjHasPrefactor(_interaction.trjHasPrefactor),
																trjTime(_interaction.trjTime), trjFieldX(_interaction.trjFieldX), trjFieldY(_interaction.trjFieldY), trjFieldZ(_interaction.trjFieldZ),
																trjPrefactor(_interaction.trjPrefactor), tdFrequency(_interaction.tdFrequency), tdPhase(_interaction.tdPhase), tdAxis(_interaction.tdAxis), tdPerpendicularOscillation(_interaction.tdPerpendicularOscillation),
																tdInitialField(_interaction.tdInitialField), tensorType(_interaction.tensorType),
																tdTimestep(_interaction.tdTimestep), tdInitialTensor(_interaction.tdInitialTensor),
																tdMinFreq(_interaction.tdMinFreq), tdMaxFreq(_interaction.tdMaxFreq), tdFreqs(_interaction.tdFreqs), tdAmps(_interaction.tdAmps), tdPhases(_interaction.tdPhases),
																tdComponents(_interaction.tdComponents), tdRandOrients(_interaction.tdRandOrients), tdThetas(_interaction.tdThetas), tdPhis(_interaction.tdPhis), tdCorrTime(_interaction.tdCorrTime),
																tdPrintTensor(_interaction.tdPrintTensor), tdPrintField(_interaction.tdPrintField), hffield(_interaction.hffield), orientations(_interaction.orientations),
																OriWeights(_interaction.OriWeights), BondLengths(_interaction.BondLengths), Spacing(_interaction.Spacing), tau(_interaction.tau), dist(_interaction.dist),
																tdSeed(_interaction.tdSeed), tdAutoseed(_interaction.tdAutoseed), tdGenerator(_interaction.tdGenerator), framelist(_interaction.framelist), f(_interaction.f)

	{
	}

	Interaction::~Interaction()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	const Interaction &Interaction::operator=(const Interaction &_interaction)
	{
		this->properties = std::make_shared<MSDParser::ObjectParser>(*(_interaction.properties));
		this->couplingTensor = _interaction.couplingTensor;
		this->field = _interaction.field;
		this->dvalue = _interaction.dvalue;
		this->evalue = _interaction.evalue;
		this->EnergyShift = _interaction.EnergyShift;
		this->hffield = _interaction.hffield;
		this->orientations = _interaction.orientations;
		this->BondLengths = _interaction.BondLengths;
		this->tau = _interaction.tau;
		this->OriWeights = _interaction.OriWeights;
		this->Spacing = _interaction.Spacing;
		this->f = _interaction.f;
		this->dist = _interaction.dist;
		this->type = _interaction.type;
		this->fieldType = _interaction.fieldType;
		this->prefactor = _interaction.prefactor;
		this->addCommonPrefactor = _interaction.addCommonPrefactor;
		this->ignoreTensors = _interaction.ignoreTensors;
		this->isValid = _interaction.isValid;
		this->trjHasTime = _interaction.trjHasTime;
		this->trjHasField = _interaction.trjHasField;
		this->trjHasTensor = _interaction.trjHasTensor;
		this->trjHasPrefactor = _interaction.trjHasPrefactor;
		this->trjTime = _interaction.trjTime;
		this->trjFieldX = _interaction.trjFieldX;
		this->trjFieldY = _interaction.trjFieldY;
		this->trjFieldZ = _interaction.trjFieldZ;
		this->trjPrefactor = _interaction.trjPrefactor;
		// member variables added for time-dependent interactions
		this->tdFrequency = _interaction.tdFrequency;
		this->tdPhase = _interaction.tdPhase;
		this->tdAxis = _interaction.tdAxis;
		this->tdPerpendicularOscillation = _interaction.tdPerpendicularOscillation;
		this->tdInitialField = _interaction.tdInitialField;
		this->tensorType = _interaction.tensorType;
		this->tdTimestep = _interaction.tdTimestep;
		this->tdInitialTensor = _interaction.tdInitialTensor;
		this->tdMinFreq = _interaction.tdMinFreq;
		this->tdMaxFreq = _interaction.tdMaxFreq;
		this->tdFreqs = _interaction.tdFreqs;
		this->tdAmps = _interaction.tdAmps;
		this->tdPhases = _interaction.tdPhases;
		this->tdComponents = _interaction.tdComponents;
		this->tdRandOrients = _interaction.tdRandOrients;
		this->tdThetas = _interaction.tdThetas;
		this->tdPhis = _interaction.tdPhis;
		this->tdCorrTime = _interaction.tdCorrTime;
		this->tdPrintTensor = _interaction.tdPrintTensor;
		this->tdPrintField = _interaction.tdPrintField;
		this->tdSeed = _interaction.tdSeed;
		this->tdAutoseed = _interaction.tdAutoseed;
		this->tdGenerator = _interaction.tdGenerator;
		this->framelist = _interaction.framelist;

		return (*this);
	}
	// -----------------------------------------------------
	// Public methods
	// -----------------------------------------------------
	std::string Interaction::Name() const
	{
		return this->properties->Name();
	}

	bool Interaction::IsValid() const
	{
		if (!isValid)
		{
			return false;
		} // by default this function should never return at this point; the only time it should ever return is if the interaction hasn't been assigned a subsystem in a task where subsystems are used

		if (this->type == InteractionType::SingleSpin && !this->group1.empty())
			return true;
		else if (this->type == InteractionType::DoubleSpin && !this->group1.empty() && !this->group2.empty())
			return true;
		else if (this->type == InteractionType::QuadraticSpin && !this->group1.empty())
            		return true;
		else if (this->type == InteractionType::Exchange && !this->group1.empty() && !this->group2.empty())
			return true;
		else if (this->type == InteractionType::Zfs && !this->group1.empty())
			return true;
		else if (this->type == InteractionType::SemiClassicalField && !this->group1.empty() && !this->hffield.empty())
			return true;

		return false;
	}
	// -----------------------------------------------------
	// Property methods
	// -----------------------------------------------------
	// Returns the field vector
	const arma::vec Interaction::Field() const
	{
		return this->field;
	}

	// Sets the field vector
	bool Interaction::SetField(const arma::vec &_field)
	{
		if (!CheckActionVectorInteractionField(_field))
			return false;
		this->field = _field;
		return true;
	}

	// Returns the D value for Zfs
	const double Interaction::Dvalue() const
	{
		return this->dvalue;
	}

	// Returns the E value for Zfs
	const double Interaction::Evalue() const
	{
		return this->evalue;
	}

	// Returns the HFI amplitude for SCI
	const std::vector<SCHyperfineField> Interaction::Hfiamplitude() const
	{
		return this->hffield;
	}

	// Returns the orientations for SCOI
	const int Interaction::Orientations() const
	{
		return this->orientations;
	}

	// Returns the prefactor value
	const double Interaction::Prefactor() const
	{
		return this->prefactor;
	}

	// Returns orientation of the coupling tensor as an euler angles
	const arma::vec Interaction::Framelist() const
	{
		return this->framelist;
	}

	// Checks whether the interaction field is time-dependent
	bool Interaction::HasFieldTimeDependence() const
	{
		if (this->fieldType != InteractionFieldType::Static)
			return true;

		return false;
	}

	// Checks whether the interaction tensor is time-dependent
	bool Interaction::HasTensorTimeDependence() const
	{
		if (this->tensorType != InteractionTensorType::Static)
			return true;

		return false;
	}

	// Checks whether the interaction is time-dependent
	bool Interaction::HasTimeDependence() const
	{
		if ((this->type == InteractionType::SingleSpin && this->HasFieldTimeDependence()) || (this->trjHasTime && this->trjHasPrefactor) || (this->HasTensorTimeDependence()))
			return true;

		if (this->trjHasTime)
		{
			return true;
		}

		return false;
	}

	// -----------------------------------------------------
	// Trajectory-related methods
	// -----------------------------------------------------
	// Returns the length of the trajectory, if any
	unsigned int Interaction::TrajectoryLength() const
	{
		return this->trajectory.Length();
	}

	// Set the current state of the Interaction to a specific step in the trajectory (row in the trajectory file)
	bool Interaction::SetTrajectoryStep(unsigned int _step)
	{
		// Make sure that we have a trajectory
		if (this->trajectory.Length() < 1 || _step >= this->trajectory.Length())
		{
			if (this->couplingTensor != nullptr)
				return this->couplingTensor->SetTrajectoryStep(_step); // The tensor might also have a trajectory, so don't return false without asking the tensor...
			else
				return false;
		}

		// Check whether the requested trajectory step is valid
		if (_step < this->trajectory.Length())
		{
			// Get the prefactor at the requested trajectory step
			if (this->trjHasPrefactor)
				this->prefactor = this->trajectory.Get(_step, this->trjPrefactor);

			// Get the field at the requested trajectory step
			if (this->trjHasField && this->fieldType == InteractionFieldType::Trajectory)
			{
				arma::vec tmp = arma::zeros<arma::vec>(3);
				tmp(0) = this->trajectory.Get(_step, this->trjFieldX);
				tmp(1) = this->trajectory.Get(_step, this->trjFieldY);
				tmp(2) = this->trajectory.Get(_step, this->trjFieldZ);
				this->field = tmp;
			}

			// Also set the trajectory step for the tensor, if possible
			// TODO: Consider what should happen if tensor trajectory length is different from interaction object trajectory length
			if (this->couplingTensor != nullptr)
				this->couplingTensor->SetTrajectoryStep(_step);
		}

		return true;
	}

	// Set the current state of the tensor to a specific time defined either in the trajectory (requires a "time" column in the trajectory) or the time-dependence function of the field
	bool Interaction::SetTime(double _time)
	{
		// Set the time for the tensor; the tensor trajectory is independent of the trajectory/time dependence of the interaction object
		if (this->couplingTensor != nullptr && this->trjHasTime)
			this->couplingTensor->SetTime(_time);

		// If we have a trajectory, which has a "time" column
		if (this->trajectory.Length() > 0 && this->trjHasTime)
		{
			// Get the row corresponding to the time right after (or at) the requested time
			unsigned int row = 0;
			double timestepAbove = 0.0;
			if (!this->trajectory.FirstRowEqGreaterThan(_time, this->trjTime, row, timestepAbove))
			{
				// If no such time was found, i.e. we are past the last element of the trajectory, use the last step of the trajectory
				this->SetTrajectoryStep(this->trajectory.Length() - 1);
			}
			else if (row == 0)
			{
				// If the specified time is before the first element of the trajectory, use the first step of the trajectory
				this->SetTrajectoryStep((unsigned int)0);
			}
			else
			{
				// Prepare a linear interpolation
				double timestepBelow = this->trajectory.Get(row - 1, trjTime);			  // Get time at previous step
				double l = 1 - (timestepAbove - _time) / (timestepAbove - timestepBelow); // Get linear interpolation factor, i.e. a number between 0 and 1, where 0 = previous time step and 1 = next time step in the trajectory

				// Get a linear interpolation for the prefactor
				if (this->trjHasPrefactor)
					this->prefactor = this->trajectory.Get(row, this->trjPrefactor) * l + this->trajectory.Get(row - 1, this->trjPrefactor) * (1 - l);

				// Get a linear interpolation for the field vector
				if (this->trjHasField && this->fieldType == InteractionFieldType::Trajectory)
				{
					arma::vec tmp = arma::zeros<arma::vec>(3);
					tmp(0) = this->trajectory.Get(row, this->trjFieldX) * l + this->trajectory.Get(row - 1, this->trjFieldX) * (1 - l);
					tmp(1) = this->trajectory.Get(row, this->trjFieldY) * l + this->trajectory.Get(row - 1, this->trjFieldY) * (1 - l);
					tmp(2) = this->trajectory.Get(row, this->trjFieldZ) * l + this->trajectory.Get(row - 1, this->trjFieldZ) * (1 - l);
					this->field = tmp;
				}
			}
		}

		std::string tdFilename = this->Name() + ".mst";

		// output the headings of the time-dependent field mst file if required
		if (_time == 0)
		{
			if (this->tdPrintField)
			{
				std::ofstream file;
				file.open(tdFilename);
				file << "time, field.x, field.y, field.z" << std::endl;
				file.close();
			}
		}

		// Apply the right time-dependence function if not using a trajectory or static field
		if (this->fieldType == InteractionFieldType::LinearPolarization)
			this->field = FieldTimeDependenceLinearPolarization(this->tdInitialField, _time, this->tdFrequency, this->tdPhase);
		else if (this->fieldType == InteractionFieldType::CircularPolarization)
			this->field = FieldTimeDependenceCircularPolarization(this->tdInitialField, _time, this->tdFrequency, this->tdPhase, this->tdAxis, this->tdPerpendicularOscillation);
		else if (this->fieldType == InteractionFieldType::Broadband)
			this->field = FieldTimeDependenceBroadband(this->tdInitialField, _time, this->tdFreqs, this->tdAmps, this->tdPhases, this->tdThetas, this->tdPhis, this->tdRandOrients, this->tdComponents);
		else if (this->fieldType == InteractionFieldType::OUGeneral)
			this->field = FieldTimeDependenceOUGeneral(this->tdInitialField, this->field, _time, this->tdTimestep, this->tdCorrTime, this->tdGenerator);

		// output the value of the field at the specified time if required
		if (this->tdPrintField)
		{
			std::ofstream file;
			file.open(tdFilename, std::ofstream::app);
			file << _time << " " << this->field(0) << " " << this->field(1) << " " << this->field(2) << std::endl;
			file.close();
		}

		// output the headings of the time-dependent tensor mst file if required
		if (_time == 0)
		{
			if (this->tdPrintTensor)
			{
				std::ofstream file;
				file.open(tdFilename);
				file << "time " << "mat.xx " << "mat.xy " << "mat.xz " << "mat.yx " << "mat.yy " << "mat.yz " << "mat.zx " << "mat.zy " << "mat.zz" << std::endl;
				file.close();
			}
		}

		// TODO: IMPLEMENT TENSOR TIME-DEPENDENCE AND FIELD TIME-DEPENDENCE IN THE SAME WAY
		if (this->tensorType == InteractionTensorType::Monochromatic)
		{
			TensorTimeDependenceMonochromatic(this->tdInitialTensor, _time, this->tdFrequency, this->tdPhase);
		}
		else if (this->tensorType == InteractionTensorType::Broadband)
		{
			TensorTimeDependenceBroadband(this->tdInitialTensor, _time, this->tdFreqs, this->tdAmps, this->tdPhases, this->tdComponents);
		}
		else if (this->tensorType == InteractionTensorType::OUGeneral)
		{
			TensorTimeDependenceOUGeneral(this->tdInitialTensor, _time, this->tdTimestep, this->tdCorrTime);
		}

		return true;
	}

	// -----------------------------------------------------
	// Access to custom properties
	// -----------------------------------------------------
	std::shared_ptr<const MSDParser::ObjectParser> Interaction::Properties() const
	{
		return this->properties;
	}

	// -----------------------------------------------------
	// Public method to read spins into group1 and group2,
	// returns false if some spins were not found
	// -----------------------------------------------------
	bool Interaction::ParseSpinGroups(const std::vector<spin_ptr> &_spinlist)
	{
		bool createdSpinLists = false;

		if (this->type == InteractionType::SingleSpin || this->type == InteractionType::Zfs || this->type == InteractionType::SemiClassicalField || this->type == InteractionType::QuadraticSpin)
		{
			// Attempt to get a list of spins from the input file
			std::string str;
			if (!this->properties->Get("spins", str) && !this->properties->Get("group1", str))
				return false;

			createdSpinLists = this->AddSpinList(str, _spinlist, this->group1);
		}
		else if (this->type == InteractionType::DoubleSpin || this->type == InteractionType::Exchange )
		{
			// Attempt to get a list of spins from the input file
			std::string str1;
			std::string str2;
			if (!this->properties->Get("group1", str1) || !this->properties->Get("group2", str2))
				return false;

			createdSpinLists = this->AddSpinList(str1, _spinlist, this->group1);
			createdSpinLists &= this->AddSpinList(str2, _spinlist, this->group2, &(this->group1)); // Cross-check with group1 as a spin can be in only one of the groups
		}

		// Check whether we were successful
		if (!createdSpinLists)
		{
			this->group1.clear();
			this->group2.clear();
			return false;
		}

		return true;
	}

	// Splits the string into spin names, find the corresponding spins in _spinlist and adds them to _group
	// Private helper method for the ParseSpinGroups method
	bool Interaction::AddSpinList(const std::string &_names, const std::vector<spin_ptr> &_spinlist, std::vector<spin_ptr> &_group, const std::vector<spin_ptr> *_crossCheck)
	{
		// Iterate through the spin name (while-loop) with a ',' delimiter
		std::stringstream ss(_names);
		bool spin_was_found;
		std::string str;
		while (std::getline(ss, str, ','))
		{
			spin_was_found = false;

			// Get the spin with the name held by str
			for (auto i = _spinlist.cbegin(); i != _spinlist.cend(); i++)
			{
				if ((*i)->Name().compare(str) == 0)
				{
					// Perform crosscheck if requested
					if (_crossCheck != nullptr && std::find((*_crossCheck).cbegin(), (*_crossCheck).cend(), (*i)) != (*_crossCheck).cend())
						return false;

					// Make sure we don't have duplicates in the spin group
					if (std::find(_group.cbegin(), _group.cend(), (*i)) != _group.cend())
						return false;

					// Add it to the group
					_group.push_back(*i);
					spin_was_found = true;
					break;
				}
			}

			if (!spin_was_found)
				return false;
		}

		return true;
	}
	// -----------------------------------------------------
	// Public spin list methods
	// -----------------------------------------------------
	// Returns a list of spins such that no spins in the list interacts with a spin outside the list (via the current Interaction object)
	std::vector<spin_ptr> Interaction::CompleteSet(const spin_ptr &_spin) const
	{
		std::vector<spin_ptr> result(1); // Only allocate space for a single spin_ptr by default

		// Only the DoubleSpin interaction type couples spins.
		// If the spin is found in either group, return a list containing the set-theoretic union of the groups
		if (this->type == InteractionType::DoubleSpin &&
			(std::find(this->group1.cbegin(), this->group1.cend(), _spin) != this->group1.cend() || std::find(this->group2.cbegin(), this->group2.cend(), _spin) != this->group2.cend()))
		{
			result.reserve(this->group1.size() + this->group2.size());				   // Reserve space for both groups to avoid more than 1 reallocation
			result.insert(result.begin(), this->group1.cbegin(), this->group1.cend()); // Insert at the beginning of the vector
			result.insert(result.end(), this->group2.cbegin(), this->group2.cend());   // Insert after the previously inserted spins
		}
		else if (this->type == InteractionType::Exchange &&
				 (std::find(this->group1.cbegin(), this->group1.cend(), _spin) != this->group1.cend() || std::find(this->group2.cbegin(), this->group2.cend(), _spin) != this->group2.cend()))
		{
			result.reserve(this->group1.size() + this->group2.size());				   // Reserve space for both groups to avoid more than 1 reallocation
			result.insert(result.begin(), this->group1.cbegin(), this->group1.cend()); // Insert at the beginning of the vector
			result.insert(result.end(), this->group2.cbegin(), this->group2.cend());   // Insert after the previously inserted spins
		}
		else if (this->type == InteractionType::Zfs &&
				 (std::find(this->group1.cbegin(), this->group1.cend(), _spin) != this->group1.cend() || std::find(this->group2.cbegin(), this->group2.cend(), _spin) != this->group2.cend()))
		{
			result.reserve(this->group1.size() + this->group2.size());				   // Reserve space for both groups to avoid more than 1 reallocation
			result.insert(result.begin(), this->group1.cbegin(), this->group1.cend()); // Insert at the beginning of the vector
			result.insert(result.end(), this->group2.cbegin(), this->group2.cend());   // Insert after the previously inserted spins
		}
		else if (this->type == InteractionType::SemiClassicalField &&
				 (std::find(this->group1.cbegin(), this->group1.cend(), _spin) != this->group1.cend() || std::find(this->group2.cbegin(), this->group2.cend(), _spin) != this->group2.cend()))
		{
			result.reserve(this->group1.size() + this->group2.size());				   // Reserve space for both groups to avoid more than 1 reallocation
			result.insert(result.begin(), this->group1.cbegin(), this->group1.cend()); // Insert at the beginning of the vector
			result.insert(result.end(), this->group2.cbegin(), this->group2.cend());   // Insert after the previously inserted spins
		}
		else
		{
			// If the spin is not in either group, the spin itself forms a complete set with respect to the current interaction object
			result.push_back(_spin);
		}

		return result;
	}

	// Extends the list of spins such that no spins in the list interacts with a spin outside the list (via the current Interaction object)
	// NOTE: The order of the spins in the list is changed if and only if the list is extended.
	bool Interaction::CompleteSet(std::vector<spin_ptr> &_list) const
	{
		bool was_extended = false;

		// Only the DoubleSpin interaction type couples spins
		if (this->type == InteractionType::DoubleSpin || this->type == InteractionType::Exchange )
		{
			bool containsInteractingSpins = false;

			// Loop through all elements in the list
			for (auto i = _list.cbegin(); i != _list.cend(); i++)
			{
				// Check whether any of the spins in the list are coupled to another spin via the current Interaction object
				if (std::find(this->group1.cbegin(), this->group1.cend(), (*i)) != this->group1.cend() || std::find(this->group2.cbegin(), this->group2.cend(), (*i)) != this->group2.cend())
				{
					containsInteractingSpins = true;
					break;
				}
			}

			// There are interacting spins in the list, so we may need to extend it
			if (containsInteractingSpins)
			{
				// First get a temporary list that is the set-theoretic completion of the given list
				std::vector<spin_ptr> tmpvec;											  // Make a temporary list...
				tmpvec.reserve(_list.size() + this->group1.size() + this->group2.size()); // ...and reserve memory to hold the following three inserts
				tmpvec.insert(tmpvec.end(), _list.cbegin(), _list.cend());				  // Insert all elements from the list
				tmpvec.insert(tmpvec.end(), this->group1.cbegin(), this->group1.cend());  // Insert all elements from group1
				tmpvec.insert(tmpvec.end(), this->group2.cbegin(), this->group2.cend());  // Insert all elements from group2
				std::sort(tmpvec.begin(), tmpvec.end());								  // Sort the elements
				auto newEnd = std::unique(tmpvec.begin(), tmpvec.end());				  // Remove duplicates (requires list to be sorted)
				tmpvec.resize(std::distance(tmpvec.begin(), newEnd));					  // Shrink vector size to get rid of removed objects (std::unique cannot do that)

				// Then check if we added new elements in the process
				if (tmpvec.size() > _list.size())
				{
					// Replace the list with the temporary one, if the temporary list was an extension
					_list = tmpvec;
					was_extended = true;
				}
			}
		}

		return was_extended;
	}

	// Public method to check whether a list of spins is complete,
	// i.e. does not interact with any spins outside the list
	bool Interaction::IsComplete(const std::vector<spin_ptr> &_list) const
	{
		unsigned int group1Elements = 0;
		unsigned int group2Elements = 0;

		// Loop through all elements in the list
		for (auto i = _list.cbegin(); i != _list.cend(); i++)
		{
			// Count number of elements from group1 found in the list
			if (std::find(this->group1.cbegin(), this->group1.cend(), (*i)) != this->group1.cend())
				++group1Elements;

			// Count number of elements from group2 found in the list
			if (std::find(this->group2.cbegin(), this->group2.cend(), (*i)) != this->group2.cend())
				++group2Elements;
		}

		// If an element from group1 is present, all elements from group2 must be present, and vice versa
		if (group1Elements > 0 && group2Elements < this->group2.size())
			return false;
		else if (group2Elements > 0 && group1Elements < this->group1.size())
			return false;

		return true;
	}
	// -----------------------------------------------------
	// Methods to create ActionTarget objects
	// -----------------------------------------------------
	// Create ActionVectors
	std::vector<RunSection::NamedActionVector> Interaction::CreateActionVectors(const std::string &_system)
	{
		std::vector<RunSection::NamedActionVector> vectors;

		if (this->IsValid())
		{
			if (this->type == InteractionType::SingleSpin)
			{
				// The field vector
				RunSection::ActionVector fieldVector = RunSection::ActionVector(this->field, &CheckActionVectorInteractionField, this->HasFieldTimeDependence());
				vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".field", fieldVector));

				// Allows changing the base field used by time-dependence functions
				if (this->HasFieldTimeDependence() && this->fieldType != InteractionFieldType::Trajectory)
				{
					RunSection::ActionVector initialfieldVector = RunSection::ActionVector(this->tdInitialField, &CheckActionVectorInteractionField);
					vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".basefield", initialfieldVector));
				}

				// The axis of circular polarized oscillating fields
				if (this->fieldType == InteractionFieldType::CircularPolarization)
				{
					RunSection::ActionVector oscaxisVector = RunSection::ActionVector(this->tdAxis, &CheckActionVectorInteractionField);
					vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".axis", oscaxisVector));
				}
			}
		}

		return vectors;
	}

	// Create ActionScalars
	std::vector<RunSection::NamedActionScalar> Interaction::CreateActionScalars(const std::string &_system)
	{
		std::vector<RunSection::NamedActionScalar> scalars;

		if (this->IsValid())
		{
			// We should always have a scalar for the prefactor
			RunSection::ActionScalar prefactorScalar = RunSection::ActionScalar(this->prefactor, &CheckActionScalarInteractionPrefactor, this->trjHasPrefactor);
			scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".prefactor", prefactorScalar));

			// There may be additional scalars for singlespin interactions
			if (this->type == InteractionType::SingleSpin)
			{
				// Frequency and phase
				if (this->fieldType == InteractionFieldType::LinearPolarization || this->fieldType == InteractionFieldType::CircularPolarization)
				{
					RunSection::ActionScalar frequencyScalar = RunSection::ActionScalar(this->tdFrequency, nullptr);
					scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".frequency", frequencyScalar));

					RunSection::ActionScalar phaseScalar = RunSection::ActionScalar(this->tdPhase, nullptr);
					scalars.push_back(RunSection::NamedActionScalar(_system + "." + this->Name() + ".phase", phaseScalar));
				}
			}
		}

		return scalars;
	}

	// Applies monochromatic noise to each tensor component, scaled by the static value
	void Interaction::TensorTimeDependenceMonochromatic(arma::mat _m, double _time, double _frequency, double _phase)
	{
		double A_xx = _m(0, 0);
		double A_yy = _m(1, 1);
		double A_zz = _m(2, 2);
		double A_xy = _m(0, 1);
		double A_xz = _m(0, 2);
		double A_yz = _m(1, 2);

		A_xx = A_xx * std::sin(_frequency * _time + _phase);
		A_yy = A_yy * std::sin(_frequency * _time + _phase);
		A_zz = A_zz * std::sin(_frequency * _time + _phase);
		A_xy = A_xy * std::sin(_frequency * _time + _phase);
		A_xz = A_xz * std::sin(_frequency * _time + _phase);
		A_yz = A_yz * std::sin(_frequency * _time + _phase);

		arma::mat tdTensor = {{A_xx, A_xy, A_xz},
							  {A_xy, A_yy, A_yz},
							  {A_xz, A_yz, A_zz}};
		// output the current tensor to file if required
		if (this->tdPrintTensor == true)
		{
			std::ofstream file;
			std::string tdFilename = this->Name() + ".mst";
			file.open(tdFilename, std::ofstream::app);
			file << _time << " " << A_xx << " " << A_xy << " " << A_xz << " " << A_xy << " " << A_yy << " " << A_yz << " " << A_xz << " " << A_yz << " " << A_zz << std::endl;
			file.close();
		}
		// set the new diagonalised matrix within the Tensor object
		this->couplingTensor->SetTensor(tdTensor);
	}

	// Applies broadband noise to the specified tensor components
	void Interaction::TensorTimeDependenceBroadband(arma::mat _m, double _time, arma::mat _freqs, arma::mat _amps, arma::mat _phases, int _comps)
	{

		// components of initial tensor
		double A_xx = _m(0, 0);
		double A_yy = _m(1, 1);
		double A_zz = _m(2, 2);
		double A_xy = _m(0, 1);
		double A_xz = _m(0, 2);
		double A_yz = _m(1, 2);

		arma::vec A_j = {A_xx, A_xy, A_xz, A_yy, A_yz, A_zz};
		arma::vec noise_j = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

		// loop through all of the noise components to generate independent noise for each tensor component
		if (_comps > 0)
		{
			for (int j_tens = 0; j_tens < 6; j_tens++)
			{
				for (int comp = 0; comp < _comps; comp++)
				{
					double phase = _phases(comp, j_tens);
					double freq = _freqs(comp, j_tens);
					double amp = _amps(comp, j_tens);

					noise_j(j_tens) += amp / sqrt(_comps) * std::sin(freq * _time + phase);
				}
				A_j(j_tens) = noise_j(j_tens) * A_j(j_tens);
			}
		}

		// set the new tensor
		A_xx = A_j(0);
		A_xy = A_j(1);
		A_xz = A_j(2);
		A_yy = A_j(3);
		A_yz = A_j(4);
		A_zz = A_j(5);

		// output the current tensor to file if required
		if (this->tdPrintTensor == true)
		{
			std::ofstream file;
			std::string tdFilename = this->Name() + ".mst";
			file.open(tdFilename, std::ofstream::app);
			file << _time << " " << A_xx << " " << A_xy << " " << A_xz << " " << A_xy << " " << A_yy << " " << A_yz << " " << A_xz << " " << A_yz << " " << A_zz << std::endl;
			file.close();
		}

		arma::mat tdTensor = {{A_xx, A_xy, A_xz},
							  {A_xy, A_yy, A_yz},
							  {A_xz, A_yz, A_zz}};
		// set the new diagonalised matrix within the Tensor object
		this->couplingTensor->SetTensor(tdTensor);
	}

	// Applies noise generated using the Ornstein-Uhlenbeck process to tensor components
	void Interaction::TensorTimeDependenceOUGeneral(arma::mat _m, double _time, double _timestep, double _corrtime)
	{
		// get the interaction tensor in the lab frame
		arma::mat labTensor = this->couplingTensor->GetTensor();

		std::normal_distribution<double> dist(0.0, 1.0);

		double noise_xx = dist(this->tdGenerator);
		double noise_yy = dist(this->tdGenerator);
		double noise_zz = dist(this->tdGenerator);
		double noise_xy = dist(this->tdGenerator);
		double noise_xz = dist(this->tdGenerator);
		double noise_yz = dist(this->tdGenerator);

		if (_time == 0)
		{
			// initial condition
			double A_xx = noise_xx * _m(0, 0);
			double A_xy = noise_xy * _m(0, 1);
			double A_xz = noise_xz * _m(0, 2);
			double A_yy = noise_yy * _m(1, 1);
			double A_yz = noise_yz * _m(1, 2);
			double A_zz = noise_zz * _m(2, 2);

			arma::mat tdTensor = {{A_xx, A_xy, A_xz},
								  {A_xy, A_yy, A_yz},
								  {A_xz, A_yz, A_zz}};
			// set the new diagonalised matrix within the Tensor object
			this->couplingTensor->SetTensor(tdTensor);

			// output the tensor at time=0 to file if required
			if (this->tdPrintTensor == true)
			{
				std::ofstream file;
				std::string tdFilename = this->Name() + ".mst";
				file.open(tdFilename, std::ofstream::app);
				file << _time << " " << A_xx << " " << A_xy << " " << A_xz << " " << A_xy << " " << A_yy << " " << A_yz << " " << A_xz << " " << A_yz << " " << A_zz << std::endl;
				file.close();
			}
		}
		else
		{
			double A_xx = labTensor(0, 0);
			double A_yy = labTensor(1, 1);
			double A_zz = labTensor(2, 2);
			double A_xy = labTensor(0, 1);
			double A_xz = labTensor(0, 2);
			double A_yz = labTensor(1, 2);

			A_xx = A_xx * (1 - _timestep / _corrtime) + _m(0, 0) * std::sqrt(2.0 * _timestep / _corrtime) * noise_xx;
			A_yy = A_yy * (1 - _timestep / _corrtime) + _m(1, 1) * std::sqrt(2.0 * _timestep / _corrtime) * noise_yy;
			A_zz = A_zz * (1 - _timestep / _corrtime) + _m(2, 2) * std::sqrt(2.0 * _timestep / _corrtime) * noise_zz;
			A_xy = A_xy * (1 - _timestep / _corrtime) + _m(0, 1) * std::sqrt(2.0 * _timestep / _corrtime) * noise_xy;
			A_xz = A_xz * (1 - _timestep / _corrtime) + _m(0, 2) * std::sqrt(2.0 * _timestep / _corrtime) * noise_xz;
			A_yz = A_yz * (1 - _timestep / _corrtime) + _m(1, 2) * std::sqrt(2.0 * _timestep / _corrtime) * noise_yz;

			arma::mat tdTensor = {{A_xx, A_xy, A_xz},
								  {A_xy, A_yy, A_yz},
								  {A_xz, A_yz, A_zz}};
			// output the current tensor to file if required
			if (this->tdPrintTensor == true)
			{
				std::ofstream file;
				std::string tdFilename = this->Name() + ".mst";
				file.open(tdFilename, std::ofstream::app);
				file << _time << " " << A_xx << " " << A_xy << " " << A_xz << " " << A_xy << " " << A_yy << " " << A_yz << " " << A_xz << " " << A_yz << " " << A_zz << std::endl;
				file.close();
			}
			// set the new diagonalised matrix within the Tensor object
			this->couplingTensor->SetTensor(tdTensor);
		}
	}

	// Method that calls the methods to generate ActionVectors and ActionScalars and inserts them into the given collections
	void Interaction::GetActionTargets(std::vector<RunSection::NamedActionScalar> &_scalars, std::vector<RunSection::NamedActionVector> &_vectors, const std::string &_system)
	{
		// Get ActionTargets from private methods
		auto scalars = this->CreateActionScalars(_system);
		auto vectors = this->CreateActionVectors(_system);

		// Insert them
		_scalars.insert(_scalars.end(), scalars.begin(), scalars.end());
		_vectors.insert(_vectors.end(), vectors.begin(), vectors.end());

		// If we have a tensor assigned, also add the ActionTargets of the Tensor
		if (this->couplingTensor != nullptr)
			this->couplingTensor->GetActionTargets(_scalars, _vectors, _system + "." + this->Name());
	}

	// -----------------------------------------------------
	// Non-member non-friend methods
	// -----------------------------------------------------
	bool IsValid(const Interaction &_interaction)
	{
		return _interaction.IsValid();
	}

	bool IsStatic(const Interaction &_interaction)
	{
		return !(HasTrajectory(_interaction) | _interaction.HasTimeDependence());
	}

	bool HasTensor(const Interaction &_interaction)
	{
		return (_interaction.CouplingTensor() != nullptr);
	}

	bool HasTrajectory(const Interaction &_interaction)
	{
		if (_interaction.TrajectoryLength() > 0)
			return true;

		return false;
	}

	InteractionType Type(const Interaction &_interaction)
	{
		return _interaction.Type();
	}

	// -----------------------------------------------------
	// Non-member non-friend time-dependent field functions
	// -----------------------------------------------------
	// Linear polarized oscillating magnetic field
	arma::vec FieldTimeDependenceLinearPolarization(const arma::vec &_v, double _time, double _frequency, double _phase)
	{
		return (_v * cos(_frequency * _time + _phase));
	}

	// Circularly polarized oscillating magnetic field
	arma::vec FieldTimeDependenceCircularPolarization(const arma::vec &_v, double _time, double _frequency, double _phase, const arma::vec &_axis, bool _perpendicularOscillations)
	{
		// Get the angle for the rotation matrix
		double angle = _time * _frequency + _phase;

		// Helper matrix
		arma::mat W(3, 3, arma::fill::zeros);
		W(0, 1) = -_axis(2);
		W(0, 2) = _axis(1);
		W(1, 2) = -_axis(0);
		W(1, 0) = _axis(2);
		W(2, 0) = -_axis(1);
		W(2, 1) = _axis(0);

		// The Rodrigues rotation matrix
		arma::mat R(3, 3, arma::fill::eye);
		R += sin(angle) * W + (2 * sin(angle / 2.0) * sin(angle / 2.0) * W * W);

		if (_perpendicularOscillations)
		{
			// Get the projection of the field vector in the plane perpendicular to the axis, and apply the rotation
			return R * (_v - dot(_axis, _v) * _axis);
		}

		// Just apply the rotation matrix to the field without projecting onto perpendicular plane
		return R * _v;
	}

	// Broadband field noise
	arma::vec FieldTimeDependenceBroadband(const arma::vec &_v, double _time, arma::mat _freqs, arma::mat _amps, arma::mat _phases, arma::mat _thetas, arma::mat _phis, bool _randorients, int _comps)
	{
		arma::vec new_field = arma::zeros<arma::vec>(3);
		// if we include random orientations - each component of the time-dependent field has independent broadband noise, and the specified field is ignored.
		// If not, a scalar function applies noise to the field terms defined in the interaction
		if (_randorients == true)
		{
			// generate broadband noise where each component has a random orientation
			for (int comp = 0; comp < _comps; comp++)
			{
				double phase = _phases(comp, 0);
				double freq = _freqs(comp, 0);
				double amp = _amps(comp, 0);
				double theta = _thetas(comp, 0);
				double phi = _phis(comp, 0);

				arma::vec orient = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
				new_field += orient * amp / sqrt(_comps) * std::sin(freq * _time + phase);
			}
		}

		else
		{
			// components of initial tensor
			double B_x = _v(0);
			double B_y = _v(1);
			double B_z = _v(2);
			arma::vec B_j = {B_x, B_y, B_z};
			arma::vec noise_j = {0.0, 0.0, 0.0};

			// loop through all of the noise components to generate independent noise for each tensor component
			if (_comps > 0)
			{
				for (int j_tens = 0; j_tens < 3; j_tens++)
				{
					for (int comp = 0; comp < _comps; comp++)
					{
						double phase = _phases(comp, j_tens);
						double freq = _freqs(comp, j_tens);
						double amp = _amps(comp, j_tens);
						noise_j(j_tens) += amp / sqrt(_comps) * std::sin(freq * _time + phase);
					}
					B_j(j_tens) = noise_j(j_tens) * B_j(j_tens);
				}
			}
			new_field = B_j;
		}
		return new_field;
	}

	// Applies noise generated using the Ornstein-Uhlenbeck process to field components
	arma::vec FieldTimeDependenceOUGeneral(const arma::vec &_v_init, arma::vec &_v_prev, double _time, double _timestep, double _corrtime, std::mt19937 &_generator)
	{

		std::normal_distribution<double> dist(0.0, 1.0);
		double noise_x = dist(_generator);
		double noise_y = dist(_generator);
		double noise_z = dist(_generator);

		double field_x = _v_prev(0) * (1 - _timestep / _corrtime) + _v_init(0) * std::sqrt(2.0 * _timestep / _corrtime) * noise_x;
		double field_y = _v_prev(1) * (1 - _timestep / _corrtime) + _v_init(1) * std::sqrt(2.0 * _timestep / _corrtime) * noise_y;
		double field_z = _v_prev(2) * (1 - _timestep / _corrtime) + _v_init(2) * std::sqrt(2.0 * _timestep / _corrtime) * noise_z;

		arma::vec newField;
		if (_time == 0)
		{
			newField = {_v_init(0) * noise_x, _v_init(1) * noise_y, _v_init(2) * noise_z};
		}
		else
		{
			newField = {field_x, field_y, field_z};
		}
		return newField;
	}

	// -----------------------------------------------------
	// Non-member non-friend ActionTarget Check functions
	// -----------------------------------------------------
	// Checks that a vector set by an Action is valid
	bool CheckActionVectorInteractionField(const arma::vec &_v)
	{
		// A field vector must have 3 components
		if (_v.n_elem != 3)
			return false;

		// Make sure we don't have invalid values
		if (_v.has_nan() || _v.has_inf())
			return false;

		return true;
	}

	// Make sure that the prefactor has a valid value (not NaN or infinite)
	bool CheckActionScalarInteractionPrefactor(const double &_d)
	{
		return std::isfinite(_d);
	}

	void FreelyJointedPolymerBL(std::vector<double> &BondLengths, std::vector<SCHyperfineField> &Fields, double &tau, int ori)
	{
		std::vector<double> tau_sum;
		for (auto f = Fields.begin(); f != Fields.end(); f++)
		{
			auto [a, n, sn] = (*f);
			// int N = n * ori;
			int N = n;
			for (int i = 0; i < N; i++)
			{
				double bondlength = a[0][0] * std::sqrt(sn * (sn + 1));
				BondLengths.push_back(bondlength);
				tau_sum.push_back(std::pow(bondlength, 2));
			}
		}
		tau = std::reduce(tau_sum.begin(), tau_sum.end());
		if (tau == 0)
		{
			tau = 0;
		}
		else
		{
			tau = 6.0 / tau;
		}
	}

	SCDistributionF FreelyJointedPolymerD(double prefactor, double exp1)
	{
		auto f = [=](std::array<double, 3> vector) mutable
		{
			double dot = 0;
			dot += vector[0] * vector[0];
			dot += vector[1] * vector[1];
			dot += vector[2] * vector[2];
			double weight = prefactor * std::exp(exp1 * dot);
			return weight;
		};
		return f;
	}
	// -----------------------------------------------------
}
