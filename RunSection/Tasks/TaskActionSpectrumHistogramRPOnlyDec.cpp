/////////////////////////////////////////////////////////////////////////
// TaskActionSpectrumHistogramRPOnlyDec implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// Task implemented by Siu Ying Wong and Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <array>
#include <cmath>
#include "ActionSpectrumHistogramUtils.h"
#include "TaskActionSpectrumHistogramRPOnlyDec.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"

namespace RunSection
{
	// -----------------------------------------------------
	// TaskActionSpectrumHistogramRPOnlyDec Constructors and Destructor
	// -----------------------------------------------------
	TaskActionSpectrumHistogramRPOnlyDec::TaskActionSpectrumHistogramRPOnlyDec(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection)
	{
	}

	TaskActionSpectrumHistogramRPOnlyDec::~TaskActionSpectrumHistogramRPOnlyDec()
	{
	}

	// -----------------------------------------------------
	// TaskActionSpectrumHistogramRPOnlyDec protected methods
	// -----------------------------------------------------
	double conversionfactortoMHzRPOnlyDec = 5e2 / arma::datum::pi;

	bool TaskActionSpectrumHistogramRPOnlyDec::RunLocal()
	{
		this->Log() << "Running method ActionSpectrumHistogramRPOnlyDec." << std::endl;

		// If this is the first step, write first part of header to the data file
		const auto bincenters = this->GetHistogramBinCenters();
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}
		// ----------------------------------------------------------------
		// SETTING UP SPIN SYSTEM AND GETTING DENSITY OPERATOR
		// ----------------------------------------------------------------

		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			// Get a list of subspaces, make sure that we have a pair of uncoupled radicals
			auto subspaces = SpinAPI::CompleteSubspaces(*(*i));
			if (subspaces.size() < 2)
			{
				this->Log() << "Failed to obtain radicals. The spin system does not have two uncoupled subspaces! Skipping SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				continue;
			}

			// Objects to hold the two unpaired electrons and their subspaces
			SpinAPI::spin_ptr radical[2] = {nullptr, nullptr};
			std::vector<SpinAPI::spin_ptr> subspace1;
			std::vector<SpinAPI::spin_ptr> subspace2;

			// Find two subspaces with an electron
			for (auto j = subspaces.cbegin(); j != subspaces.cend(); j++)
			{
				for (auto k = j->cbegin(); k != j->cend(); k++)
				{
					if ((*k)->Type() == SpinAPI::SpinType::Electron)
					{
						if (radical[0] == nullptr)
						{
							radical[0] = (*k);
							subspace1 = (*j);
						}
						else if (radical[1] == nullptr)
						{
							radical[1] = (*k);
							subspace2 = (*j);
						}

						break;
					}
				}
			}

			// Check whether we found a radical pair
			if (radical[0] == nullptr || radical[1] == nullptr)
			{
				this->Log() << "Failed to obtain radicals. Did not find two uncoupled electronic spins! Skipping SpinSystem \"" << (*i)->Name() << "\"." << std::endl;
				continue;
			}

			// Obtain SpinSpaces to describe the two subsystems
			SpinAPI::SpinSpace spaces[2] = {SpinAPI::SpinSpace(subspace1), SpinAPI::SpinSpace(subspace2)};
			spaces[0].UseSuperoperatorSpace(false);
			spaces[1].UseSuperoperatorSpace(false);
			spaces[0].Add((*i)->Interactions());
			spaces[1].Add((*i)->Interactions());

			// Provide information about the radicals
			this->Log() << "---------------------------------------" << std::endl;
			this->Log() << "Found radical 1 with " << subspace1.size() << " spins:" << std::endl;
			this->Log() << " - Unpaired Electron: " << radical[0]->Name() << std::endl;
			this->Log() << " - Subspace dimensions: " << spaces[0].SpaceDimensions() << std::endl;

			if (subspace1.size() > 1)
			{
				this->Log() << " - Other spins:" << std::endl;
				for (auto j = subspace1.cbegin(); j != subspace1.cend(); j++)
					if ((*j) != radical[0])
						this->Log() << "   - " << (*j)->Name() << std::endl;
			}
			else
			{
				this->Log() << " - There are no other spins." << std::endl;
			}

			this->Log() << "\nFound radical 2 with " << subspace2.size() << " spins:" << std::endl;
			this->Log() << " - Unpaired Electron: " << radical[1]->Name() << std::endl;
			this->Log() << " - Subspace dimensions: " << spaces[1].SpaceDimensions() << std::endl;

			if (subspace2.size() > 1)
			{
				this->Log() << " - Other spins:" << std::endl;
				for (auto j = subspace2.cbegin(); j != subspace2.cend(); j++)
					if ((*j) != radical[1])
						this->Log() << "   - " << (*j)->Name() << std::endl;
			}
			else
			{
				this->Log() << " - There are no other spins." << std::endl;
			}

			this->Log() << "\nSpins not included in radicals: " << ((*i)->spins_size() - subspace1.size() - subspace2.size()) << " / " << (*i)->spins_size() << std::endl;
			this->Log() << "---------------------------------------" << std::endl;
			this->Log() << "Spins that are not included in the radicals are not considered in the calculations." << std::endl;

			// Matrices to hold spin operators, eigenvectors and eigenvalue-differences
			arma::cx_mat eigenvectors[2]; // To hold eigenvectors
			arma::vec eigenvalues[2];	  // To hold eigenvalues

			arma::cx_mat Id1;
			Id1.eye(spaces[0].HilbertSpaceDimensions(), spaces[0].HilbertSpaceDimensions());
			arma::cx_mat Id2;
			Id2.eye(spaces[1].HilbertSpaceDimensions(), spaces[1].HilbertSpaceDimensions());

			arma::sp_cx_mat transitionhamiltonancomplete = arma::kron(arma::conv_to<arma::sp_cx_mat>::from(Id1), arma::conv_to<arma::sp_cx_mat>::from(Id2));
			transitionhamiltonancomplete.zeros();

			arma::cx_mat tmptransitionhamiltonian[2];
			std::array<arma::cx_mat, 3> electron_operators[2];
			bool radical_ready = true;
			bool perpendicular_ready = false;
			arma::vec perpendicular_rf_field;

			// Fill the matrices (get Hamiltonian, diagonalize it, get spin operators...)
			for (unsigned int r = 0; r < 2; r++)
			{
				// Get the Hamiltonian of the first radical subspace
				arma::cx_mat H;

				if (!spaces[r].Hamiltonian(H))
				{
					this->Log() << "Failed to obtain Hamiltonian for radical " << r << "." << std::endl;
					radical_ready = false;
					break;
				}

				// Diagonalize Hamiltonian
				if (!arma::eig_sym(eigenvalues[r], eigenvectors[r], H))
				{
					this->Log() << "Failed to diagonalize Hamiltonian for radical " << r << "." << std::endl;
					radical_ready = false;
					break;
				}

				// Embed the unpaired-electron operators using the actual spin ordering.
				if (!spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from(radical[r]->Sx()), radical[r], electron_operators[r][0]) ||
				    !spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from(radical[r]->Sy()), radical[r], electron_operators[r][1]) ||
				    !spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from(radical[r]->Sz()), radical[r], electron_operators[r][2]))
				{
					this->Log() << "Failed to construct electronic spin operators for radical " << r << "." << std::endl;
					radical_ready = false;
					break;
				}

				// Make the RF transition Hamiltonian.
				arma::cx_mat transitionhamiltonian = arma::zeros<arma::cx_mat>(arma::size(H));

				for (auto interaction = (*i)->interactions_cbegin(); interaction != (*i)->interactions_cend(); interaction++)
				{
					// Produce the transition Hamiltonian for each radical subsystem
					bool isZeemaninteraction = (*interaction)->Type() == SpinAPI::InteractionType::SingleSpin;
					bool containedinsubspace = spaces[r].Contains(*interaction);

					if (isZeemaninteraction && containedinsubspace)
					{
						// Inherit some properties of Zeeman interaction
						arma::vec staticfield = (*interaction)->Field();
						auto ATensor = (*interaction)->CouplingTensor();
						bool ignoretensors = (*interaction)->IgnoreTensors();
						auto spinlist = (*interaction)->Group1();
						arma::cx_mat contribution(H.n_rows, H.n_cols, arma::fill::zeros);

						// Get RF Field
						std::string rffielduserinput;
						arma::vec rffield(3);
						this->Properties()->Get("rf_field", rffielduserinput);

						bool useperpendicularfield = rffielduserinput == "perpendicular";

						if (useperpendicularfield)
						{
							if (!perpendicular_ready)
							{
								if (arma::norm(staticfield) <= 0.0)
								{
									this->Log() << "Cannot define a perpendicular RF field for zero static field." << std::endl;
									return false;
								}
								perpendicular_rf_field = this->GetPerpendicular3DVectorRPOnlyDec(staticfield, true);
								perpendicular_ready = true;
							}
							rffield = perpendicular_rf_field;
						}
						else if (!this->Properties()->Get("rf_field", rffield) || rffield.n_elem != 3)
						{
							this->Log() << "Action histogram requires rf_field as a 3-vector or 'perpendicular'." << std::endl;
							return false;
						}

						// Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
						arma::cx_mat Sx;
						arma::cx_mat Sy;
						arma::cx_mat Sz;

						// Loop through list of spins and append to the transition hamiltonian
						for (auto j = spinlist.cbegin(); j != spinlist.cend(); j++)
						{
							if (ignoretensors)
							{
								spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), Sx);
								spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), Sy);
								spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), Sz);
							}
							else
							{
								spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tx()), (*j), Sx);
								spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Ty()), (*j), Sy);
								spaces[r].CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tz()), (*j), Sz);
							}

							if (ATensor != nullptr && !IsIsotropic(*ATensor))
							{
								// Use the tensor to calculate the product S * A * B
								auto A = ATensor->LabFrame();
								contribution += Sx * rffield(0) * A(0, 0) + Sx * rffield(1) * A(0, 1) + Sx * rffield(2) * A(0, 2);
								contribution += Sy * rffield(0) * A(1, 0) + Sy * rffield(1) * A(1, 1) + Sy * rffield(2) * A(1, 2);
								contribution += Sz * rffield(0) * A(2, 0) + Sz * rffield(1) * A(2, 1) + Sz * rffield(2) * A(2, 2);
							}
							else
							{
								contribution += Sx * rffield(0) + Sy * rffield(1) + Sz * rffield(2);
							}
						}
						// Multiply by the isotropic value, if the tensor was isotropic
						if (ATensor != nullptr && IsIsotropic(*ATensor))
							contribution *= ATensor->Isotropic();

						// Multiply with the given prefactor (or 1.0 if none was specified)
						contribution *= (*interaction)->Prefactor();
						// Multiply by common prefactor (bohr magneton / hbar)
						if ((*interaction)->AddCommonPrefactor())
							contribution *= 8.79410005e+1;
						transitionhamiltonian += contribution;
					}
				}

				transitionhamiltonian = eigenvectors[r].t() * transitionhamiltonian * eigenvectors[r];
				// Sum up both transiton Hamiltonians
				tmptransitionhamiltonian[r] = transitionhamiltonian;
			}


			// ----------------------------------------------------------------
			// GET ARRAY OF ENERGY GAPS, SIMILAR TO THE "domega" matrix
			// ----------------------------------------------------------------

			if (!radical_ready)
				continue;

			transitionhamiltonancomplete = arma::kron(arma::conv_to<arma::sp_cx_mat>::from(tmptransitionhamiltonian[0]), arma::conv_to<arma::sp_cx_mat>::from(Id2)) + arma::kron(arma::conv_to<arma::sp_cx_mat>::from(Id1), arma::conv_to<arma::sp_cx_mat>::from(tmptransitionhamiltonian[1]));

			// Get the all eigenvalues into one big space matrix
			arma::vec totaleigenvalues;
			totaleigenvalues = arma::kron(eigenvalues[0], arma::ones(spaces[1].HilbertSpaceDimensions())) + arma::kron(arma::ones(spaces[0].HilbertSpaceDimensions()), eigenvalues[1]);

			arma::vec energygaps = this->GetDifferencesRPOnlyDec(totaleigenvalues);

			// ----------------------------------------------------------------
			// EVALUATE RESONANCE EFFECTS USING EIGENVECTORS OF H0
			// ----------------------------------------------------------------

			// Produce the intial state
			arma::cx_vec rho0vec = this->GetPopulationVector(eigenvectors[0], eigenvectors[1], electron_operators[0], electron_operators[1]);

			arma::vec resonanceeffects = this->GetResonanceEffectsRPOnlyDec(rho0vec, transitionhamiltonancomplete);
			//  ----------------------------------------------------------------
			//  BINNING PROCESS FOR HISTOGRAM
			//  ----------------------------------------------------------------
			double upperlimit;
			this->Properties()->Get("upper_limit", upperlimit);
			double binwidth;
			this->Properties()->Get("bin_width", binwidth);
			bool useMHz = true;
			this->Properties()->Get("units_in_MHz", useMHz);

			const int numbins = static_cast<int>(bincenters.n_elem);
			auto normalisedheights = GetNormalisedHistogramHeightsRPOnlyDec(energygaps, resonanceeffects, binwidth, numbins, useMHz);
			this->Data() << this->RunSettings()->CurrentStep() << " ";

			this->Data() << normalisedheights.st(); // already contains endl
		}
		return true;
	}

	// --------------------------------------------------------------------------------------------------------------------------------------

	long long TaskActionSpectrumHistogramRPOnlyDec::GetNumDifferencesRPOnlyDec(long long dimension)
	{
		long long numgaps = dimension * (dimension - 1) / 2;
		return numgaps;
	}

	arma::vec TaskActionSpectrumHistogramRPOnlyDec::GetDifferencesRPOnlyDec(arma::vec values)
	{
		long long dimension = values.size();
		long long numgaps = GetNumDifferencesRPOnlyDec(dimension);
		arma::vec differences(numgaps);
		int ind = 0;
		for (int k = 0; k < dimension - 1; k++)
		{
			for (int s = k + 1; s < dimension; s++)
			{
				differences(ind) = std::abs(values(k) - values(s));
				ind++;
			}
		}

		return differences;
	}

	arma::cx_vec TaskActionSpectrumHistogramRPOnlyDec::GetPopulationVector(
        const arma::cx_mat &V1, const arma::cx_mat &V2,
        const std::array<arma::cx_mat, 3> &S1,
        const std::array<arma::cx_mat, 3> &S2)
	{
		return ActionSpectrumHistogramUtils::SingletPopulations(V1, V2, S1, S2);
	}

    arma::vec TaskActionSpectrumHistogramRPOnlyDec::GetPerpendicular3DVectorRPOnlyDec(arma::vec vector, bool normalise)
	{
		arma::vec perp(3);
		perp.zeros();
		int mutuallyperpaxis = (arma::abs(vector)).index_min();
		if (mutuallyperpaxis == 0)
		{
			perp(2) = vector(1);
			perp(1) = -1 * vector(2);
		}
		if (mutuallyperpaxis == 1)
		{
			perp(2) = vector(0);
			perp(0) = -1 * vector(2);
		}
		if (mutuallyperpaxis == 2)
		{
			perp(1) = vector(0);
			perp(0) = -1 * vector(1);
		}
		if (normalise)
		{
			if (arma::norm(perp) <= 0.0)
				throw std::invalid_argument("Cannot normalize an RF vector perpendicular to zero field.");
			perp /= arma::norm(perp);
		}
		return perp;
	}

	arma::vec TaskActionSpectrumHistogramRPOnlyDec::GetResonanceEffectsRPOnlyDec(arma::cx_vec initialstatepopulation, arma::sp_cx_mat transitionhamiltonian)
	{
		long long dim = initialstatepopulation.n_elem;
		long long numresonances = GetNumDifferencesRPOnlyDec(dim);
		arma::vec resonanceeffects(numresonances);

		// Fill the array of resonance effects one element at the time
		int ind_resonance = 0;
		for (int k = 0; k < dim - 1; k++)
		{
			for (int s = k + 1; s < dim; s++)
			{
				double sqrtoftransprob = std::abs(static_cast<std::complex<double>>(transitionhamiltonian(k, s)));
				double value = std::abs(initialstatepopulation(k) - initialstatepopulation(s)) * (sqrtoftransprob * sqrtoftransprob);
				resonanceeffects(ind_resonance) = value;
				ind_resonance++;
			}
		}
		return resonanceeffects;
	}

	double TaskActionSpectrumHistogramRPOnlyDec::GetHistogramPrefactorRPOnlyDec(bool useMHz_units)
	{
		double prefactor;
		if (useMHz_units)
			prefactor = conversionfactortoMHzRPOnlyDec;
		else
			prefactor = 1e9;
		return prefactor;
	}

	arma::vec TaskActionSpectrumHistogramRPOnlyDec::GetNormalisedHistogramHeightsRPOnlyDec(arma::vec energygaps, arma::vec resonanceeffects, double binwidth, int numbins, bool useMHz_units)
	{
		return ActionSpectrumHistogramUtils::NormalizedHeights(energygaps, resonanceeffects,
		                                                        binwidth, numbins,
		                                                        GetHistogramPrefactorRPOnlyDec(useMHz_units));
	}

    	// Writes the header of the data file (but can also be passed to other streams)
	void TaskActionSpectrumHistogramRPOnlyDec::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		this->WriteStandardOutputHeader(_stream);
		auto bincenters = TaskActionSpectrumHistogramRPOnlyDec::GetHistogramBinCenters();
		_stream << bincenters.st(); // already contains endl
	}

	// Validation
	bool TaskActionSpectrumHistogramRPOnlyDec::Validate()
	{
		double upperlimit = 0.0;
		double binwidth = 0.0;
		const bool valid = this->Properties()->Get("upper_limit", upperlimit) &&
		                   this->Properties()->Get("bin_width", binwidth) &&
		                   std::isfinite(upperlimit) && std::isfinite(binwidth) &&
		                   binwidth > 0.0 && upperlimit >= binwidth / 2.0;
		if (!valid)
			this->Log() << "Action histogram requires upper_limit >= bin_width/2 and bin_width > 0." << std::endl;
		return valid;
	}

    arma::vec TaskActionSpectrumHistogramRPOnlyDec::GetHistogramBinCenters()
	{
		double upperlimit;
		this->Properties()->Get("upper_limit", upperlimit);
		double binwidth;
		this->Properties()->Get("bin_width", binwidth);
		return ActionSpectrumHistogramUtils::BinCenters(upperlimit, binwidth);
	}
}
