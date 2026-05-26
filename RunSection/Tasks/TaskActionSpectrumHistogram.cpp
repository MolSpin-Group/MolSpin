/////////////////////////////////////////////////////////////////////////
// TaskActionSpectrumHistogram implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// Task implemented by Siu Ying Wong and Luca Gerhards.
// (c) 2022 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskActionSpectrumHistogram.h"
#include "Transition.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Spin.h"
#include "Interaction.h"
#include "ObjectParser.h"

//need to modify this to include time dependent hamiltonians 
namespace RunSection
{
    // -----------------------------------------------------
    // TaskActionSpectrumHistogram Constructors and Destructor
    // -----------------------------------------------------
    TaskActionSpectrumHistogram::TaskActionSpectrumHistogram(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection)
    {
    }

    TaskActionSpectrumHistogram::~TaskActionSpectrumHistogram()
    {
    }

    // -----------------------------------------------------
    // TaskActionSpectrumHistogram protected methods
    // -----------------------------------------------------
    double conversion_factor_to_MHz = 5e2 / arma::datum::pi;

    int GetNumDifferences(int dimension)
    {
        int num_gaps = dimension * (dimension - 1) / 2;
        return num_gaps;
    }

    arma::vec GetDifferences(arma::vec values)
    {
        int dimension = values.size();
        int num_gaps = GetNumDifferences(dimension);
        arma::vec differences(num_gaps);
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

    arma::vec GetEigenstatePopulations(arma::cx_mat eigenvectors_matrix, arma::cx_mat initial_state)
    {
        arma::cx_mat eigenbasis_initial_state = eigenvectors_matrix.t() * initial_state * eigenvectors_matrix;
        arma::vec populations = arma::real(eigenbasis_initial_state.diag());
        return populations;
    }

    arma::vec GetPerpendicular3DVector(arma::vec vector, bool normalise)
    {
        arma::vec perp(3);
        perp.zeros();
        int mutually_perp_axis = (arma::abs(vector)).index_min();
        if (mutually_perp_axis == 0)
        {
            perp(2) = vector(1);
            perp(1) = -1 * vector(2);
        }
        if (mutually_perp_axis == 1)
        {
            perp(2) = vector(0);
            perp(0) = -1 * vector(2);
        }
        if (mutually_perp_axis == 2)
        {
            perp(1) = vector(0);
            perp(0) = -1 * vector(1);
        }
        if (normalise)
            perp /= arma::norm(perp);
        return perp;
    }

    arma::vec GetResonanceEffects(arma::cx_mat eigenvectors_matrix, arma::cx_mat initial_state, arma::cx_mat transition_hamiltonian)
    {
        int dim = size(initial_state)(0);
        int num_resonances = GetNumDifferences(dim);
        arma::vec resonance_effects(num_resonances);
        arma::vec eigenstate_populations = GetEigenstatePopulations(eigenvectors_matrix, initial_state);
        arma::cx_mat eigenbasis_transition_hamiltonian = eigenvectors_matrix.t() * transition_hamiltonian * eigenvectors_matrix;

        // Fill the array of resonance effects one element at the time
        int ind_resonance = 0;
        for (int k = 0; k < dim - 1; k++)
        {
            for (int s = k + 1; s < dim; s++)
            {
                auto sqrt_of_transprob = std::abs(eigenbasis_transition_hamiltonian(k, s));
                auto value = std::abs(eigenstate_populations(k) - eigenstate_populations(s)) * (sqrt_of_transprob * sqrt_of_transprob);
                resonance_effects(ind_resonance) = value;
                ind_resonance++;
            }
        }
        return resonance_effects;
    }

    double GetHistogramPrefactor(bool use_MHz_units)
    {
        double prefactor;
        if (use_MHz_units)
            prefactor = conversion_factor_to_MHz;
        else
            prefactor = 1e9;
        return prefactor;
    }

    arma::vec GetNormalisedHistogramHeights(arma::vec energy_gaps, arma::vec resonance_effects, double bin_width, int num_bins, bool use_MHz_units)
    {
        double prefactor = GetHistogramPrefactor(use_MHz_units);
        int num_resonances = energy_gaps.size();
        arma::vec heights(num_bins);
        heights.zeros();
        for (int s = 0; s < num_resonances; s++)
        {
            auto gap = std::abs(energy_gaps(s)) * prefactor;
            auto effect = std::abs(resonance_effects(s));
            auto bin_number = floor(gap / bin_width); // only works for uniform bin_width
            heights(bin_number) += effect;
        }
        return heights / arma::sum(heights);
    }

    bool TaskActionSpectrumHistogram::RunLocal()
    {
        this->Log() << "Running method ActionSpectrumHistogram." << std::endl;

        // If this is the first step, write first part of header to the data file
        auto bin_centers = this->GetHistogramBinCenters();
        if (this->RunSettings()->CurrentStep() == 1)
        {
            this->WriteHeader(this->Data());
        }
        // ----------------------------------------------------------------
        // SETTING UP SPIN SYSTEM AND GETTING DENSITY OPERATOR
        // ----------------------------------------------------------------

        auto systems = this->SpinSystems();
        for (auto system = systems.cbegin(); system != systems.cend(); system++)
        {
            // Make sure we have an initial state
            auto initial_states = (*system)->InitialState();
            if (initial_states.size() < 1)
            {
                this->Log() << "Skipping SpinSystem \"" << (*system)->Name() << "\" as no initial state was specified." << std::endl;
                continue;
            }

            this->Log() << "\nStarting with SpinSystem \"" << (*system)->Name() << "\"." << std::endl;

            // Obtain a SpinSpace to describe the system
            SpinAPI::SpinSpace space(*(*system));
            arma::cx_mat rho0;
            space.UseSuperoperatorSpace(false);

            // Get the initial state
            for (auto j = initial_states.cbegin(); j != initial_states.cend(); j++)
            {
                arma::cx_mat tmp_rho0;
                if (!space.GetState(*j, tmp_rho0))
                {
                    this->Log() << "Failed to obtain projection matrix onto state \"" << (*j)->Name() << "\", initial state of SpinSystem \"" << (*system)->Name() << "\"." << std::endl;
                    continue;
                }
                if (j == initial_states.cbegin())
                    rho0 = tmp_rho0;
                else
                    rho0 += tmp_rho0;
            }
            rho0 /= arma::trace(rho0); // The density operator should have a trace of 1

            // ----------------------------------------------------------------
            // Get the Hamiltonian
            // ----------------------------------------------------------------

            arma::cx_mat H;
            if (!space.Hamiltonian(H))
            {
                this->Log() << "Failed to obtain Hamiltonian in superspace." << std::endl;
                continue;
            }

            // ----------------------------------------------------------------
            // DIAGONALIZATION OF H0//
            // ----------------------------------------------------------------
            arma::cx_mat eigenvectors; // To hold eigenvectors
            arma::vec eigenvalues;     // To hold eigenvalues

            this->Log() << "Starting diagonalization..." << std::endl;
            arma::eig_sym(eigenvalues, eigenvectors, H);
            this->Log() << "Diagonalization done! Eigenvalues: " << eigenvalues.n_elem << ", eigenvectors: " << eigenvectors.n_cols << std::endl;
            // ----------------------------------------------------------------
            // GET ARRAY OF ENERGY GAPS, SIMILAR TO THE "domega" matrix
            // ----------------------------------------------------------------
            arma::vec energy_gaps = GetDifferences(eigenvalues);
            // ----------------------------------------------------------------
            // EVALUATE RESONANCE EFFECTS USING EIGENVECTORS OF H0
            // ----------------------------------------------------------------
            // Make transition hamiltonian with copy and pasted code.
            arma::cx_mat transition_hamiltonian = arma::zeros<arma::cx_mat>(arma::size(H));
            transition_hamiltonian.zeros();
            bool zeeman_parameters_already_found = false;
            for (auto interaction = (*system)->interactions_cbegin(); interaction != (*system)->interactions_cend(); interaction++)
            {
                bool is_Zeeman_interaction = (*interaction)->Type() == SpinAPI::InteractionType::SingleSpin;

                if (is_Zeeman_interaction)
                {
                    // Inherit some properties of Zeeman interaction
                    arma::vec static_field = (*interaction)->Field();
                    auto ATensor = (*interaction)->CouplingTensor();
                    bool ignore_tensors = (*interaction)->IgnoreTensors();
                    auto spinlist = (*interaction)->Group1();

                    // Get RF Field
                    std::string rf_field_user_input;
                    arma::vec rf_field(3);
                    this->Properties()->Get("rf_field", rf_field_user_input);

                    bool use_perpendicular_field = rf_field_user_input == "perpendicular";

                    if (use_perpendicular_field)
                        rf_field = GetPerpendicular3DVector(static_field, true);
                    else
                        this->Properties()->Get("rf_field", rf_field);

                    // Check if a set of Zeeman interaction parameters have been found and parsed previously
                    if (use_perpendicular_field and zeeman_parameters_already_found)
                        throw std::logic_error("RF field vector must be supplied by user if multiple static field Zeeman interactions are provided.");
                    zeeman_parameters_already_found = true;

                    // Obtain the list of spins interacting with the field, and define matrices to hold the magnetic moment operators
                    arma::cx_mat Sx;
                    arma::cx_mat Sy;
                    arma::cx_mat Sz;

                    // Loop through list of spins and append to the transition hamiltonian
                    for (auto j = spinlist.cbegin(); j != spinlist.cend(); j++)
                    {
                        if (ignore_tensors)
                        {
                            space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sx()), (*j), Sx);
                            space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sy()), (*j), Sy);
                            space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Sz()), (*j), Sz);
                        }
                        else
                        {
                            space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tx()), (*j), Sx);
                            space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Ty()), (*j), Sy);
                            space.CreateOperator(arma::conv_to<arma::cx_mat>::from((*j)->Tz()), (*j), Sz);
                        }

                        if (ATensor != nullptr && !IsIsotropic(*ATensor))
                        {
                            // Use the tensor to calculate the product S * A * B
                            auto A = ATensor->LabFrame();
                            transition_hamiltonian += Sx * rf_field(0) * A(0, 0) + Sx * rf_field(1) * A(0, 1) + Sx * rf_field(2) * A(0, 2);
                            transition_hamiltonian += Sy * rf_field(0) * A(1, 0) + Sy * rf_field(1) * A(1, 1) + Sy * rf_field(2) * A(1, 2);
                            transition_hamiltonian += Sz * rf_field(0) * A(2, 0) + Sz * rf_field(1) * A(2, 1) + Sz * rf_field(2) * A(2, 2);
                        }
                        else
                        {
                            transition_hamiltonian += Sx * rf_field(0) + Sy * rf_field(1) + Sz * rf_field(2);
                        }
                    }
                    // Multiply by the isotropic value, if the tensor was isotropic
                    if (ATensor != nullptr && IsIsotropic(*ATensor))
                        transition_hamiltonian *= ATensor->Isotropic();

                    // Multiply with the given prefactor (or 1.0 if none was specified)
                    transition_hamiltonian *= (*interaction)->Prefactor();
                    // Multiply by common prefactor (bohr magneton / hbar)
                    if ((*interaction)->AddCommonPrefactor())
                        transition_hamiltonian *= 8.79410005e+1;
                }
            }
            arma::vec resonance_effects = GetResonanceEffects(eigenvectors, rho0, transition_hamiltonian);
            // ----------------------------------------------------------------
            // BINNING PROCESS FOR HISTOGRAM
            // ----------------------------------------------------------------
            double upper_limit;
            this->Properties()->Get("upper_limit", upper_limit);
            double bin_width;
            this->Properties()->Get("bin_width", bin_width);
            bool use_MHz;
            this->Properties()->Get("units_in_MHz", use_MHz);

            int num_bins = arma::regspace(bin_width / 2, bin_width, upper_limit).size();
            auto normalised_heights = GetNormalisedHistogramHeights(energy_gaps, resonance_effects, bin_width, num_bins, use_MHz);
            this->Data() << this->RunSettings()->CurrentStep() << " ";
            this->Data() << normalised_heights.st(); // already contains endl
        }
        return true;
    }

    // --------------------------------------------------------------------------------------------------------------------------------------

    // Writes the header of the data file (but can also be passed to other streams)
    void TaskActionSpectrumHistogram::WriteHeader(std::ostream &_stream)
    {
        _stream << "Step ";
        this->WriteStandardOutputHeader(_stream);
        auto bin_centers = TaskActionSpectrumHistogram::GetHistogramBinCenters();
        _stream << bin_centers.st(); // already contains endl
    }

    // Validation
    bool TaskActionSpectrumHistogram::Validate()
    {

        return true;
    }

    arma::vec TaskActionSpectrumHistogram::GetHistogramBinCenters()
    {
        double upper_limit;
        this->Properties()->Get("upper_limit", upper_limit);
        double bin_width;
        this->Properties()->Get("bin_width", bin_width);
        auto bin_centers = arma::regspace(bin_width / 2, bin_width, upper_limit);
        return bin_centers;
    }

}