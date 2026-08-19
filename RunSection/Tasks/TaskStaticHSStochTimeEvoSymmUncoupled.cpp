/////////////////////////////////////////////////////////////////////////
// TaskStaticHSStochTimeEvoSymmUncoupled implementation (RunSection module) by Gediminas Pazera and Luca Gerhards
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskStaticHSStochTimeEvoSymmUncoupled.h"
#include "Transition.h"
#include "Operator.h"
#include "Settings.h"
#include "State.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "Interaction.h"
#include <iomanip> // std::setprecision

namespace RunSection
{
        // -----------------------------------------------------
        // TaskStaticHSStochTimeEvoSymmUncoupled Constructors and Destructor
        // -----------------------------------------------------
        TaskStaticHSStochTimeEvoSymmUncoupled::TaskStaticHSStochTimeEvoSymmUncoupled(const MSDParser::ObjectParser &_parser, const RunSection &_runsection) : BasicTask(_parser, _runsection), timestep(0.1), totaltime(1000), reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn), productYieldsOnly(false)
        {
        }

        TaskStaticHSStochTimeEvoSymmUncoupled::~TaskStaticHSStochTimeEvoSymmUncoupled()
        {
        }
        // -----------------------------------------------------
        // TaskStaticHSStochTimeEvoSymmUncoupled protected methods
        // -----------------------------------------------------
        bool TaskStaticHSStochTimeEvoSymmUncoupled::RunLocal()
        {
                this->Log() << "Running method StaticHS_Stoch_TimeEvo_Symm_Uncoupled." << std::endl;

                // If this is the first step, write first part of header to the data file
                if (this->RunSettings()->CurrentStep() == 1)
                {
                        this->WriteHeader(this->Data());
                }

                // Loop through all SpinSystems
                auto systems = this->SpinSystems();
                for (auto i = systems.cbegin(); i != systems.cend(); i++) // iteration through all spin systems, in this case (or usually), this is one
                {
                        this->Log() << "\nStarting with SpinSystem \"" << (*i)->Name() << "\"." << std::endl;

                        // Gyromagnetic constant
                        double gamma_e = 176.0859644; // gyromagnetic ratio of free electron spin in rad mT^-1 mus^-1
                        // If no rate was specified, take the rate of the first transition in the SpinSystem, if any

                        double kmin = 1e-3;

                        auto transitions = (*i)->Transitions();
                        if (transitions.size() > 0)
                        {
                                // Use the value of the first transition object found
                                kmin = transitions[0]->Rate();
                                this->Log() << "Using rate constant from first transition object (\"" << transitions[0]->Name() << "\") in the SpinSystem \"" << (*i)->Name() << "\"!\nFound rate constant value: " << kmin << std::endl;
                        }
                        else
                        {
                                // Use the default value if no value could be found
                                this->Log() << "No rate constant was found for SpinSystem \"" << (*i)->Name() << "\"!\nUsing default rate constant value: " << kmin << std::endl;
                        }

                        kmin = kmin * 1e3 / gamma_e;

                        // Count the number of nuclear spins
                        int nucspins = 0;
                        std::vector<int> SpinNumbers;
                        for (auto l = (*i)->spins_cbegin(); l != (*i)->spins_cend(); l++)
                        {

                                std::string spintype;
                                (*l)->Properties()->Get("type", spintype);
                                if (spintype != "electron")
                                {
                                        nucspins += 1;
                                }
                                if (spintype == "electron")
                                {
                                        // Throws an error if the spins are not spin 1/2
                                        if ((*l)->Multiplicity() != 2)
                                        {
                                                std::cout << (*l)->Multiplicity() << std::endl;
                                                this->Log() << "SkippingSpin System \"" << (*i)->Name() << "\" as electron spins have the wrong multiplicity." << std::endl;
                                                std::cout << "# ERROR: electron spins have to be spin 1/2! Skipping the SpinSystem." << std::endl;
                                                return 1;
                                        }
                                }
                        }

                        // Check if there are any nuclear spins
                        if (nucspins == 0)
                        {
                                this->Log() << "Skipping SpinSystem \"" << (*i)->Name() << "\" as no nuclear spins were specified." << std::endl;
                                std::cout << "# ERROR: no nuclear spins were specified, skipping the system" << std::endl;
                                return 1;
                        }

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

                        double dim1 = spaces[0].SpaceDimensions();
                        std::cout << "# The size of the first subspace: " << dim1 << " x " << dim1 << std::endl;

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

                        double dim2 = spaces[1].SpaceDimensions();
                        std::cout << "# The size of the second subspace: " << dim2 << " x " << dim2 << std::endl;

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

                        this->Log() << "Full Hilbert space size " << dim1 * dim2 << " x " << dim1 * dim2 << std::endl;
                        std::cout << "# Full Hilbert space size: " << dim1 * dim2 << " x " << dim1 * dim2 << std::endl;

                        double Z1 = dim1 / 2; // Size of the nuclear spin space for radical 1
                        double Z2 = dim2 / 2; // Size of the nuclear spin space for radical 2

                        // Get the Hamiltonian of the first radical subspace
                        arma::sp_cx_mat H1;
                        if (!spaces[0].Hamiltonian(H1))
                        {
                                this->Log() << "Failed to obtain Hamiltonian for radical " << 1 << "." << std::endl;
                                continue;
                        }

                        H1 = H1 * 1000;

                        // Get the Hamiltonian of the second radical subspace
                        arma::sp_cx_mat H2;
                        if (!spaces[1].Hamiltonian(H2))
                        {
                                this->Log() << "Failed to obtain Hamiltonian for radical " << 2 << "." << std::endl;
                                continue;
                        }

                        H2 = H2 * 1000;

                        // Set up states for time-propagation
                        arma::cx_mat InitialStateVector(4, 1);
                        std::string InitialState;
                        this->Properties()->Get("initialstate", InitialState);
                        std::string InitialStateLower;
                        InitialStateLower.resize(InitialState.size());
                        std::transform(InitialState.begin(), InitialState.end(), InitialStateLower.begin(), ::tolower);
                        InitialState = InitialStateLower;
                        if (InitialStateLower == "singlet")
                        {
                                this->Log() << "Singlet initial state." << std::endl;
                        }
                        else if (InitialStateLower == "tripletminus")
                        {
                                this->Log() << "Triplet minus initial state." << std::endl;
                        }
                        else if (InitialStateLower == "tripletzero")
                        {
                        }
                        else if (InitialStateLower == "tripletplus")
                        {
                                this->Log() << "Triplet plus initial state." << std::endl;
                        }
                        else
                        {
                                std::cout << "# ERROR: Invalid initial state value! It is set to a Singlet state." << std::endl;
                                this->Log() << "Initial state is undefined. Setting it to a Singlet state" << std::endl;
                                InitialState = "singlet";
                        }

                        // Defining the number of Monte Carlo samples
                        int mc_samples1;
                        this->Properties()->Get("montecarlosamples1", mc_samples1);

                        if (mc_samples1 > 0)
                        {
                                this->Log() << "Number of Monte Carlo samples for the first radical is " << mc_samples1 << "." << std::endl;
                        }
                        else
                        {
                                this->Log() << "Undefined number of Monte Carlo samples for the first radical. Setting it to default of 100." << std::endl;
                                std::cout << "# ERROR: undefined number of Monte Carlo samples for the first radical! Setting the number to default of 100." << std::endl;
                                mc_samples1 = 100;
                        }

                        int mc_samples2;
                        this->Properties()->Get("montecarlosamples2", mc_samples2);

                        if (mc_samples2 > 0)
                        {
                                this->Log() << "Number of Monte Carlo samples for the second radical is " << mc_samples2 << "." << std::endl;
                        }
                        else
                        {
                                this->Log() << "Undefined number of Monte Carlo samples for the second radical. Setting it to default of 100." << std::endl;
                                std::cout << "# ERROR: undefined number of Monte Carlo samples for the second radical! Setting the number to default of 100." << std::endl;
                                mc_samples2 = 100;
                        }

                        // Random Number Generator Preparation
                        std::random_device rand_dev;        // random number generator
                        std::mt19937 generator(rand_dev()); // random number generator
                        bool autoseed;
                        this->Properties()->Get("autoseed", autoseed);

                        if (!autoseed)
                        {
                                this->Log() << "Autoseed is off." << std::endl;
                                double seednumber;
                                this->Properties()->Get("seed", seednumber);
                                if (seednumber != 0)
                                {
                                        generator.seed(seednumber);
                                        this->Log() << "Seed number is " << seednumber << "." << std::endl;
                                }
                                else
                                {
                                        this->Log() << "Undefined seed number! Setting to default of 1." << std::endl;
                                        std::cout << "# ERROR: undefined seed number! Setting to default of 1." << std::endl;
                                        seednumber = 1;
                                }
                        }
                        else
                        {
                                this->Log() << "Autoseed is on." << std::endl;
                        }

                        // Obtain the sampling method and set up states for time-propagation
                        this->Log() << "Using SU(Z) spin states for Monte Carlo sampling." << std::endl;

                        arma::cx_mat Alpha(2, 1);
                        Alpha(0) = 1.0;
                        Alpha(1) = 0.0;
                        arma::cx_mat Beta(2, 1);
                        Beta(0) = 0.0;
                        Beta(1) = 1.0;
                        arma::cx_mat B1;
                        arma::cx_mat B2;
                        if (InitialState == "singlet" || InitialState == "tripletzero")
                        {
                                B1.resize(2 * Z1, 2 * mc_samples1);
                                B2.resize(2 * Z2, 2 * mc_samples2);
                                arma::cx_colvec temp;
                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                {
                                        temp = spaces[0].SUZstate(Z1, generator);
                                        B1.col(it1) = arma::kron(Alpha, temp);
                                        B1.col(it1 + mc_samples1) = arma::kron(Beta, temp);
                                }
                                arma::cx_colvec temp2;
                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                {
                                        temp2 = spaces[0].SUZstate(Z2, generator);
                                        B2.col(it2) = arma::kron(Alpha, temp2);
                                        B2.col(it2 + mc_samples2) = arma::kron(Beta, temp2);
                                }
                        }
                        else if (InitialState == "tripletplus")
                        {
                                B1.resize(2 * Z1, mc_samples1);
                                B2.resize(2 * Z2, mc_samples2);
                                arma::cx_colvec temp;
                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                {
                                        temp = spaces[0].SUZstate(Z1, generator);
                                        B1.col(it1) = arma::kron(Alpha, temp);
                                }
                                arma::cx_colvec temp2;
                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                {
                                        temp2 = spaces[0].SUZstate(Z2, generator);
                                        B2.col(it2) = arma::kron(Alpha, temp2);
                                }
                        }
                        else if (InitialState == "tripletminus")
                        {
                                B1.resize(2 * Z1, mc_samples1);
                                B2.resize(2 * Z2, mc_samples2);
                                arma::cx_colvec temp;
                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                {
                                        temp = spaces[0].SUZstate(Z1, generator);
                                        B1.col(it1) = arma::kron(Beta, temp);
                                }
                                arma::cx_colvec temp2;
                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                {
                                        temp2 = spaces[0].SUZstate(Z2, generator);
                                        B2.col(it2) = arma::kron(Beta, temp2);
                                }
                        }

                        // Setting or calculating total time.
                        double ttotal;
                        double totaltime;
                        this->Properties()->Get("totaltime", totaltime);
                        ttotal = totaltime;
                        double epsilon;
                        epsilon = 0;
                        this->Properties()->Get("epsilon", epsilon);
                        if (ttotal > std::pow(2, -53))
                        {
                                this->Log() << "Total time is chosen as " << ttotal << " ns." << std::endl;
                                ttotal = ttotal * 1e-3 * gamma_e;
                                this->Log() << "Total time is chosen as " << ttotal << " rad mT^-1." << std::endl;
                        }
                        else
                        {
                                this->Log() << "No total time is given or it is incorrect: checking if epsilon is defined." << std::endl;
                                if (epsilon > 0)
                                {
                                        this->Log() << "Epsilon is " << epsilon << "." << std::endl;
                                        ttotal = std::log(1 / epsilon) / kmin;
                                        this->Log() << "Estimated total time is " << ttotal / 1e-3 / gamma_e << " ns." << std::endl;
                                        this->Log() << "Estimated total time is " << ttotal << " rad mT^-1." << std::endl;
                                }
                                else
                                {
                                        this->Log() << "Both total time and epsilon are not given or not appropriately defined." << std::endl;
                                        std::cout << "# ERROR: total time and/or epsilon are not defined/given!" << std::endl;
                                        return 1;
                                }
                        }

                        // Setting timestep
                        double dt;
                        double timestep;
                        this->Properties()->Get("timestep", timestep);
                        dt = timestep;
                        if (dt > std::pow(2, -53))
                        {
                                this->Log() << "Time step is chosen as " << dt << " ns." << std::endl;
                                dt = dt * 1e-3 * gamma_e;
                                this->Log() << "Time step is chosen as " << dt << " rad mT^-1." << std::endl;
                        }
                        else
                        {
                                this->Log() << "Time step is undefined." << std::endl;
                                std::cout << "# ERROR: undefined time step!" << std::endl;
                                return 1;
                        }

                        // Number of time propagation steps
                        int num_steps = std::ceil(ttotal / dt);
                        this->Log() << "Number of time propagation steps: " << num_steps << "." << std::endl;

                        // Choose Propagation Method and other parameters
                        std::string precision;
                        this->Properties()->Get("precision", precision);

                        this->Log() << "This method uses autoexpm for time-propagation" << std::endl;
                        if (precision == "double")
                        {
                                this->Log() << "Double precision is chosen for the autoexpm method." << std::endl;
                        }
                        else if (precision == "single")
                        {
                                this->Log() << "Single precision is chosen for the autoexpm method." << std::endl;
                        }
                        else if (precision == "half")
                        {
                                this->Log() << "Half precision is chosen for the autoexpm method." << std::endl;
                        }
                        else
                        {
                                std::cout << "# ERROR: undefined precision. Using single digit precision!" << std::endl;
                                this->Log() << "No precision for autoexpm method was defined. Using single digit precision." << std::endl;
                                precision = "single";
                        }

                        // Current step
                        this->WriteStandardOutput(this->Data());

                        // Defining Pauli spin matrices
                        arma::sp_cx_mat Sx(2, 2); // x pauli matrix
                        Sx(0, 1) = 0.5;
                        Sx(1, 0) = 0.5;

                        arma::sp_cx_mat Sy(2, 2); // y pauli matrix
                        Sy(0, 1) = arma::cx_double(0.0, -0.5);
                        Sy(1, 0) = arma::cx_double(0.0, 0.5);

                        arma::sp_cx_mat Sz(2, 2); // z pauli matrix
                        Sz(0, 0) = 0.5;
                        Sz(1, 1) = -0.5;

                        arma::sp_cx_mat Id1 = arma::speye<arma::sp_cx_mat>(Z1, Z1);
                        arma::sp_cx_mat Id2 = arma::speye<arma::sp_cx_mat>(Z2, Z2);

                        arma::sp_cx_mat Sx1 = arma::kron(Sx, Id1);
                        arma::sp_cx_mat Sx2 = arma::kron(Sx, Id2);
                        arma::sp_cx_mat Sy1 = arma::kron(Sy, Id1);
                        arma::sp_cx_mat Sy2 = arma::kron(Sy, Id2);
                        arma::sp_cx_mat Sz1 = arma::kron(Sz, Id1);
                        arma::sp_cx_mat Sz2 = arma::kron(Sz, Id2);

                        arma::sp_cx_mat Sp1 = Sx1 + arma::cx_double(0.0, 1.0) * Sy1;
                        arma::sp_cx_mat Sm1 = Sx1 - arma::cx_double(0.0, 1.0) * Sy1;
                        arma::sp_cx_mat Sp2 = Sx2 + arma::cx_double(0.0, 1.0) * Sy2;
                        arma::sp_cx_mat Sm2 = Sx2 - arma::cx_double(0.0, 1.0) * Sy2;

                        arma::sp_cx_mat Sp1Sm1 = Sp1 * Sm1;
                        arma::sp_cx_mat Sm1Sp1 = Sm1 * Sp1;
                        arma::sp_cx_mat Sp2Sm2 = Sp2 * Sm2;
                        arma::sp_cx_mat Sm2Sp2 = Sm2 * Sp2;

                        arma::mat M1; // used for variable estimation
                        arma::mat M2; // used for variable estimation

                        if (InitialState == "singlet")
                        {
                                arma::cx_vec AlphaKet1(2 * Z1, 1);
                                arma::cx_vec BetaKet1(2 * Z1, 1);
                                arma::cx_vec AlphaBra1(1, 2 * Z1);
                                arma::cx_vec BetaBra1(1, 2 * Z1);
                                arma::cx_vec AlphaKet2(2 * Z2, 1);
                                arma::cx_vec BetaKet2(2 * Z2, 1);
                                arma::cx_vec AlphaBra2(1, 2 * Z2);
                                arma::cx_vec BetaBra2(1, 2 * Z2);

                                arma::cx_double Sx1AA;
                                arma::cx_double Sx1AB;
                                arma::cx_double Sx1BB;
                                arma::cx_double Sy1AA;
                                arma::cx_double Sy1AB;
                                arma::cx_double Sy1BB;
                                arma::cx_double Sz1AA;
                                arma::cx_double Sz1AB;
                                arma::cx_double Sz1BB;

                                arma::cx_double Sx2AA;
                                arma::cx_double Sx2BA;
                                arma::cx_double Sx2BB;
                                arma::cx_double Sy2AA;
                                arma::cx_double Sy2BA;
                                arma::cx_double Sy2BB;
                                arma::cx_double Sz2AA;
                                arma::cx_double Sz2BA;
                                arma::cx_double Sz2BB;

                                arma::cx_double Sp1Sm1AA;
                                arma::cx_double Sp1Sm1BB;
                                arma::cx_double Sp1Sm1AB;
                                arma::cx_double Sp2Sm2AA;
                                arma::cx_double Sp2Sm2BB;
                                arma::cx_double Sp2Sm2AB;

                                arma::cx_double Sm1Sp1AA;
                                arma::cx_double Sm1Sp1BB;
                                arma::cx_double Sm1Sp1AB;
                                arma::cx_double Sm2Sp2AA;
                                arma::cx_double Sm2Sp2BB;
                                arma::cx_double Sm2Sp2AB;

                                arma::cx_mat result(1, 1);

                                arma::cx_mat B1t(2 * mc_samples1, 2 * Z1);
                                arma::cx_mat B2t(2 * mc_samples2, 2 * Z2);

                                for (int indx = 0; indx < num_steps; indx++)
                                {
                                        // Set the current time
                                        double current_time = indx * dt;

                                        this->Data() << current_time / (1e-3 * gamma_e);

                                        arma::cx_double Ps;
                                        arma::cx_double Ptplus;
                                        arma::cx_double Ptzero;
                                        arma::cx_double Ptminus;
                                        arma::cx_double temp1;
                                        arma::cx_double temp2;
                                        arma::cx_double temp3;
                                        arma::cx_double temp4;
                                        B1t = B1.t();
                                        B2t = B2.t();

                                        if (mc_samples1 < mc_samples2)
                                        {

                                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                                {

                                                        AlphaKet2 = B2.col(it2);
                                                        BetaKet2 = B2.col(it2 + mc_samples2);
                                                        AlphaBra2 = B2t.row(it2);
                                                        BetaBra2 = B2t.row(it2 + mc_samples2);

                                                        // For P_S
                                                        result = AlphaBra2 * Sx2 * AlphaKet2;
                                                        Sx2AA = result(0, 0);
                                                        result = BetaBra2 * Sx2 * AlphaKet2;
                                                        Sx2BA = result(0, 0);
                                                        result = BetaBra2 * Sx2 * BetaKet2;
                                                        Sx2BB = result(0, 0);

                                                        result = AlphaBra2 * Sy2 * AlphaKet2;
                                                        Sy2AA = result(0, 0);
                                                        result = BetaBra2 * Sy2 * AlphaKet2;
                                                        Sy2BA = result(0, 0);
                                                        result = BetaBra2 * Sy2 * BetaKet2;
                                                        Sy2BB = result(0, 0);

                                                        result = AlphaBra2 * Sz2 * AlphaKet2;
                                                        Sz2AA = result(0, 0);
                                                        result = BetaBra2 * Sz2 * AlphaKet2;
                                                        Sz2BA = result(0, 0);
                                                        result = BetaBra2 * Sz2 * BetaKet2;
                                                        Sz2BB = result(0, 0);

                                                        // For P_Tplus

                                                        result = AlphaBra2 * Sp2Sm2 * AlphaKet2;
                                                        Sp2Sm2AA = result(0, 0);
                                                        result = BetaBra2 * Sp2Sm2 * BetaKet2;
                                                        Sp2Sm2BB = result(0, 0);
                                                        result = AlphaBra2 * Sp2Sm2 * BetaKet2;
                                                        Sp2Sm2AB = result(0, 0);

                                                        // For P_Tminus

                                                        result = AlphaBra2 * Sm2Sp2 * AlphaKet2;
                                                        Sm2Sp2AA = result(0, 0);
                                                        result = BetaBra2 * Sm2Sp2 * BetaKet2;
                                                        Sm2Sp2BB = result(0, 0);
                                                        result = AlphaBra2 * Sm2Sp2 * BetaKet2;
                                                        Sm2Sp2AB = result(0, 0);

                                                        for (int it1 = 0; it1 < mc_samples1; it1++)
                                                        {
                                                                AlphaKet1 = B1.col(it1);
                                                                BetaKet1 = B1.col(it1 + mc_samples1);
                                                                AlphaBra1 = B1t.row(it1);
                                                                BetaBra1 = B1t.row(it1 + mc_samples1);

                                                                // P_S

                                                                result = AlphaBra1 * Sx1 * AlphaKet1;
                                                                Sx1AA = result(0, 0);
                                                                result = AlphaBra1 * Sx1 * BetaKet1;
                                                                Sx1AB = result(0, 0);
                                                                result = BetaBra1 * Sx1 * BetaKet1;
                                                                Sx1BB = result(0, 0);

                                                                result = AlphaBra1 * Sy1 * AlphaKet1;
                                                                Sy1AA = result(0, 0);
                                                                result = AlphaBra1 * Sy1 * BetaKet1;
                                                                Sy1AB = result(0, 0);
                                                                result = BetaBra1 * Sy1 * BetaKet1;
                                                                Sy1BB = result(0, 0);

                                                                result = AlphaBra1 * Sz1 * AlphaKet1;
                                                                Sz1AA = result(0, 0);
                                                                result = AlphaBra1 * Sz1 * BetaKet1;
                                                                Sz1AB = result(0, 0);
                                                                result = BetaBra1 * Sz1 * BetaKet1;
                                                                Sz1BB = result(0, 0);

                                                                // For P_Tplus

                                                                result = AlphaBra1 * Sp1Sm1 * AlphaKet1;
                                                                Sp1Sm1AA = result(0, 0);
                                                                result = BetaBra1 * Sp1Sm1 * BetaKet1;
                                                                Sp1Sm1BB = result(0, 0);
                                                                result = AlphaBra1 * Sp1Sm1 * BetaKet1;
                                                                Sp1Sm1AB = result(0, 0);

                                                                // For P_Tminus

                                                                result = AlphaBra1 * Sm1Sp1 * AlphaKet1;
                                                                Sm1Sp1AA = result(0, 0);
                                                                result = BetaBra1 * Sm1Sp1 * BetaKet1;
                                                                Sm1Sp1BB = result(0, 0);
                                                                result = AlphaBra1 * Sm1Sp1 * BetaKet1;
                                                                Sm1Sp1AB = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - arma::cx_double(0.5, 0) * (Sx1AA * Sx2BB - 2 * (Sx1AB * Sx2BA).real() + Sx1BB * Sx2AA + Sy1AA * Sy2BB - 2 * (Sy1AB * Sy2BA).real() + Sy1BB * Sy2AA + Sz1AA * Sz2BB - 2 * (Sz1AB * Sz2BA).real() + Sz1BB * Sz2AA);
                                                                temp2 = arma::cx_double(0.5, 0.0) * (Sp1Sm1AA * Sp2Sm2BB - 2 * (Sp1Sm1AB * Sp2Sm2AB).real() + Sp1Sm1BB * Sp2Sm2AA);
                                                                temp3 = arma::cx_double(0.5, 0.0) * (Sm1Sp1AA * Sm2Sp2BB - 2 * (Sm1Sp1AB * Sm2Sp2AB).real() + Sm1Sp1BB * Sm2Sp2AA);

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }
                                        else
                                        {

                                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                                {
                                                        AlphaKet1 = B1.col(it1);
                                                        BetaKet1 = B1.col(it1 + mc_samples1);
                                                        AlphaBra1 = B1t.row(it1);
                                                        BetaBra1 = B1t.row(it1 + mc_samples1);

                                                        // For P_S

                                                        result = AlphaBra1 * Sx1 * AlphaKet1;
                                                        Sx1AA = result(0, 0);
                                                        result = AlphaBra1 * Sx1 * BetaKet1;
                                                        Sx1AB = result(0, 0);
                                                        result = BetaBra1 * Sx1 * BetaKet1;
                                                        Sx1BB = result(0, 0);

                                                        result = AlphaBra1 * Sy1 * AlphaKet1;
                                                        Sy1AA = result(0, 0);
                                                        result = AlphaBra1 * Sy1 * BetaKet1;
                                                        Sy1AB = result(0, 0);
                                                        result = BetaBra1 * Sy1 * BetaKet1;
                                                        Sy1BB = result(0, 0);

                                                        result = AlphaBra1 * Sz1 * AlphaKet1;
                                                        Sz1AA = result(0, 0);
                                                        result = AlphaBra1 * Sz1 * BetaKet1;
                                                        Sz1AB = result(0, 0);
                                                        result = BetaBra1 * Sz1 * BetaKet1;
                                                        Sz1BB = result(0, 0);

                                                        // For P_Tplus

                                                        result = AlphaBra1 * Sp1Sm1 * AlphaKet1;
                                                        Sp1Sm1AA = result(0, 0);
                                                        result = BetaBra1 * Sp1Sm1 * BetaKet1;
                                                        Sp1Sm1BB = result(0, 0);
                                                        result = AlphaBra1 * Sp1Sm1 * BetaKet1;
                                                        Sp1Sm1AB = result(0, 0);

                                                        // For P_Tminus

                                                        result = AlphaBra1 * Sm1Sp1 * AlphaKet1;
                                                        Sm1Sp1AA = result(0, 0);
                                                        result = BetaBra1 * Sm1Sp1 * BetaKet1;
                                                        Sm1Sp1BB = result(0, 0);
                                                        result = AlphaBra1 * Sm1Sp1 * BetaKet1;
                                                        Sm1Sp1AB = result(0, 0);

                                                        for (int it2 = 0; it2 < mc_samples2; it2++)
                                                        {
                                                                AlphaKet2 = B2.col(it2);
                                                                BetaKet2 = B2.col(it2 + mc_samples2);
                                                                AlphaBra2 = B2t.row(it2);
                                                                BetaBra2 = B2t.row(it2 + mc_samples2);

                                                                // For P_S

                                                                result = AlphaBra2 * Sx2 * AlphaKet2;
                                                                Sx2AA = result(0, 0);
                                                                result = BetaBra2 * Sx2 * AlphaKet2;
                                                                Sx2BA = result(0, 0);
                                                                result = BetaBra2 * Sx2 * BetaKet2;
                                                                Sx2BB = result(0, 0);

                                                                result = AlphaBra2 * Sy2 * AlphaKet2;
                                                                Sy2AA = result(0, 0);
                                                                result = BetaBra2 * Sy2 * AlphaKet2;
                                                                Sy2BA = result(0, 0);
                                                                result = BetaBra2 * Sy2 * BetaKet2;
                                                                Sy2BB = result(0, 0);

                                                                result = AlphaBra2 * Sz2 * AlphaKet2;
                                                                Sz2AA = result(0, 0);
                                                                result = BetaBra2 * Sz2 * AlphaKet2;
                                                                Sz2BA = result(0, 0);
                                                                result = BetaBra2 * Sz2 * BetaKet2;
                                                                Sz2BB = result(0, 0);

                                                                // For P_Tplus

                                                                result = AlphaBra2 * Sp2Sm2 * AlphaKet2;
                                                                Sp2Sm2AA = result(0, 0);
                                                                result = BetaBra2 * Sp2Sm2 * BetaKet2;
                                                                Sp2Sm2BB = result(0, 0);
                                                                result = AlphaBra2 * Sp2Sm2 * BetaKet2;
                                                                Sp2Sm2AB = result(0, 0);

                                                                // For P_Tminus

                                                                result = AlphaBra2 * Sm2Sp2 * AlphaKet2;
                                                                Sm2Sp2AA = result(0, 0);
                                                                result = BetaBra2 * Sm2Sp2 * BetaKet2;
                                                                Sm2Sp2BB = result(0, 0);
                                                                result = AlphaBra2 * Sm2Sp2 * BetaKet2;
                                                                Sm2Sp2AB = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - arma::cx_double(0.5, 0) * (Sx1AA * Sx2BB - 2 * (Sx1AB * Sx2BA).real() + Sx1BB * Sx2AA + Sy1AA * Sy2BB - 2 * (Sy1AB * Sy2BA).real() + Sy1BB * Sy2AA + Sz1AA * Sz2BB - 2 * (Sz1AB * Sz2BA).real() + Sz1BB * Sz2AA);
                                                                temp2 = arma::cx_double(0.5, 0.0) * (Sp1Sm1AA * Sp2Sm2BB - 2 * (Sp1Sm1AB * Sp2Sm2AB).real() + Sp1Sm1BB * Sp2Sm2AA);
                                                                temp3 = arma::cx_double(0.5, 0.0) * (Sm1Sp1AA * Sm2Sp2BB - 2 * (Sm1Sp1AB * Sm2Sp2AB).real() + Sm1Sp1BB * Sm2Sp2AA);

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }

                                        // Calculate the expected value

                                        double expected_value1 = std::exp(-kmin * current_time) * Ps.real() / mc_samples1 / mc_samples2;
                                        double expected_value2 = std::exp(-kmin * current_time) * Ptplus.real() / mc_samples1 / mc_samples2;
                                        double expected_value3 = std::exp(-kmin * current_time) * Ptzero.real() / mc_samples1 / mc_samples2;
                                        double expected_value4 = std::exp(-kmin * current_time) * Ptminus.real() / mc_samples1 / mc_samples2;

                                        this->Data() << " " << expected_value1;
                                        this->Data() << " " << expected_value2;
                                        this->Data() << " " << expected_value3;
                                        this->Data() << " " << expected_value4;
                                        this->Data() << std::endl;

                                        // Update B using the Higham propagator
                                        B1 = spaces[0].HighamProp(H1, B1, -dt * arma::cx_double(0.0, 1.0), precision, M1);
                                        B2 = spaces[0].HighamProp(H2, B2, -dt * arma::cx_double(0.0, 1.0), precision, M2);
                                }
                        }
                        else if (InitialState == "tripletzero")
                        {
                                arma::cx_vec AlphaKet1(2 * Z1, 1);
                                arma::cx_vec BetaKet1(2 * Z1, 1);
                                arma::cx_vec AlphaBra1(1, 2 * Z1);
                                arma::cx_vec BetaBra1(1, 2 * Z1);
                                arma::cx_vec AlphaKet2(2 * Z2, 1);
                                arma::cx_vec BetaKet2(2 * Z2, 1);
                                arma::cx_vec AlphaBra2(1, 2 * Z2);
                                arma::cx_vec BetaBra2(1, 2 * Z2);

                                arma::cx_double Sx1AA;
                                arma::cx_double Sx1AB;
                                arma::cx_double Sx1BB;
                                arma::cx_double Sy1AA;
                                arma::cx_double Sy1AB;
                                arma::cx_double Sy1BB;
                                arma::cx_double Sz1AA;
                                arma::cx_double Sz1AB;
                                arma::cx_double Sz1BB;

                                arma::cx_double Sx2AA;
                                arma::cx_double Sx2BA;
                                arma::cx_double Sx2BB;
                                arma::cx_double Sy2AA;
                                arma::cx_double Sy2BA;
                                arma::cx_double Sy2BB;
                                arma::cx_double Sz2AA;
                                arma::cx_double Sz2BA;
                                arma::cx_double Sz2BB;

                                arma::cx_double Sp1Sm1AA;
                                arma::cx_double Sp1Sm1BB;
                                arma::cx_double Sp1Sm1AB;
                                arma::cx_double Sp2Sm2AA;
                                arma::cx_double Sp2Sm2BB;
                                arma::cx_double Sp2Sm2AB;

                                arma::cx_double Sm1Sp1AA;
                                arma::cx_double Sm1Sp1BB;
                                arma::cx_double Sm1Sp1AB;
                                arma::cx_double Sm2Sp2AA;
                                arma::cx_double Sm2Sp2BB;
                                arma::cx_double Sm2Sp2AB;

                                arma::cx_mat result(1, 1);

                                arma::cx_mat B1t(2 * mc_samples1, 2 * Z1);
                                arma::cx_mat B2t(2 * mc_samples2, 2 * Z2);

                                for (int indx = 0; indx < num_steps; indx++)
                                {
                                        // Set the current time
                                        double current_time = indx * dt;

                                        this->Data() << current_time / (1e-3 * gamma_e);

                                        arma::cx_double Ps;
                                        arma::cx_double Ptplus;
                                        arma::cx_double Ptzero;
                                        arma::cx_double Ptminus;
                                        arma::cx_double temp1;
                                        arma::cx_double temp2;
                                        arma::cx_double temp3;
                                        arma::cx_double temp4;
                                        B1t = B1.t();
                                        B2t = B2.t();

                                        if (mc_samples1 < mc_samples2)
                                        {

                                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                                {

                                                        AlphaKet2 = B2.col(it2);
                                                        BetaKet2 = B2.col(it2 + mc_samples2);
                                                        AlphaBra2 = B2t.row(it2);
                                                        BetaBra2 = B2t.row(it2 + mc_samples2);

                                                        // For P_S
                                                        result = AlphaBra2 * Sx2 * AlphaKet2;
                                                        Sx2AA = result(0, 0);
                                                        result = BetaBra2 * Sx2 * AlphaKet2;
                                                        Sx2BA = result(0, 0);
                                                        result = BetaBra2 * Sx2 * BetaKet2;
                                                        Sx2BB = result(0, 0);

                                                        result = AlphaBra2 * Sy2 * AlphaKet2;
                                                        Sy2AA = result(0, 0);
                                                        result = BetaBra2 * Sy2 * AlphaKet2;
                                                        Sy2BA = result(0, 0);
                                                        result = BetaBra2 * Sy2 * BetaKet2;
                                                        Sy2BB = result(0, 0);

                                                        result = AlphaBra2 * Sz2 * AlphaKet2;
                                                        Sz2AA = result(0, 0);
                                                        result = BetaBra2 * Sz2 * AlphaKet2;
                                                        Sz2BA = result(0, 0);
                                                        result = BetaBra2 * Sz2 * BetaKet2;
                                                        Sz2BB = result(0, 0);

                                                        // For P_Tplus

                                                        result = AlphaBra2 * Sp2Sm2 * AlphaKet2;
                                                        Sp2Sm2AA = result(0, 0);
                                                        result = BetaBra2 * Sp2Sm2 * BetaKet2;
                                                        Sp2Sm2BB = result(0, 0);
                                                        result = AlphaBra2 * Sp2Sm2 * BetaKet2;
                                                        Sp2Sm2AB = result(0, 0);

                                                        // For P_Tminus

                                                        result = AlphaBra2 * Sm2Sp2 * AlphaKet2;
                                                        Sm2Sp2AA = result(0, 0);
                                                        result = BetaBra2 * Sm2Sp2 * BetaKet2;
                                                        Sm2Sp2BB = result(0, 0);
                                                        result = AlphaBra2 * Sm2Sp2 * BetaKet2;
                                                        Sm2Sp2AB = result(0, 0);

                                                        for (int it1 = 0; it1 < mc_samples1; it1++)
                                                        {
                                                                AlphaKet1 = B1.col(it1);
                                                                BetaKet1 = B1.col(it1 + mc_samples1);
                                                                AlphaBra1 = B1t.row(it1);
                                                                BetaBra1 = B1t.row(it1 + mc_samples1);

                                                                // P_S

                                                                result = AlphaBra1 * Sx1 * AlphaKet1;
                                                                Sx1AA = result(0, 0);
                                                                result = AlphaBra1 * Sx1 * BetaKet1;
                                                                Sx1AB = result(0, 0);
                                                                result = BetaBra1 * Sx1 * BetaKet1;
                                                                Sx1BB = result(0, 0);

                                                                result = AlphaBra1 * Sy1 * AlphaKet1;
                                                                Sy1AA = result(0, 0);
                                                                result = AlphaBra1 * Sy1 * BetaKet1;
                                                                Sy1AB = result(0, 0);
                                                                result = BetaBra1 * Sy1 * BetaKet1;
                                                                Sy1BB = result(0, 0);

                                                                result = AlphaBra1 * Sz1 * AlphaKet1;
                                                                Sz1AA = result(0, 0);
                                                                result = AlphaBra1 * Sz1 * BetaKet1;
                                                                Sz1AB = result(0, 0);
                                                                result = BetaBra1 * Sz1 * BetaKet1;
                                                                Sz1BB = result(0, 0);

                                                                // For P_Tplus

                                                                result = AlphaBra1 * Sp1Sm1 * AlphaKet1;
                                                                Sp1Sm1AA = result(0, 0);
                                                                result = BetaBra1 * Sp1Sm1 * BetaKet1;
                                                                Sp1Sm1BB = result(0, 0);
                                                                result = AlphaBra1 * Sp1Sm1 * BetaKet1;
                                                                Sp1Sm1AB = result(0, 0);

                                                                // For P_Tminus

                                                                result = AlphaBra1 * Sm1Sp1 * AlphaKet1;
                                                                Sm1Sp1AA = result(0, 0);
                                                                result = BetaBra1 * Sm1Sp1 * BetaKet1;
                                                                Sm1Sp1BB = result(0, 0);
                                                                result = AlphaBra1 * Sm1Sp1 * BetaKet1;
                                                                Sm1Sp1AB = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - arma::cx_double(0.5, 0) * (Sx1AA * Sx2BB + 2 * (Sx1AB * Sx2BA).real() + Sx1BB * Sx2AA + Sy1AA * Sy2BB + 2 * (Sy1AB * Sy2BA).real() + Sy1BB * Sy2AA + Sz1AA * Sz2BB + 2 * (Sz1AB * Sz2BA).real() + Sz1BB * Sz2AA);
                                                                temp2 = arma::cx_double(0.5, 0.0) * (Sp1Sm1AA * Sp2Sm2BB + 2 * (Sp1Sm1AB * Sp2Sm2AB).real() + Sp1Sm1BB * Sp2Sm2AA);
                                                                temp3 = arma::cx_double(0.5, 0.0) * (Sm1Sp1AA * Sm2Sp2BB + 2 * (Sm1Sp1AB * Sm2Sp2AB).real() + Sm1Sp1BB * Sm2Sp2AA);

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }
                                        else
                                        {
                                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                                {
                                                        AlphaKet1 = B1.col(it1);
                                                        BetaKet1 = B1.col(it1 + mc_samples1);
                                                        AlphaBra1 = B1t.row(it1);
                                                        BetaBra1 = B1t.row(it1 + mc_samples1);

                                                        // For P_S
                                                        result = AlphaBra1 * Sx1 * AlphaKet1;
                                                        Sx1AA = result(0, 0);
                                                        result = AlphaBra1 * Sx1 * BetaKet1;
                                                        Sx1AB = result(0, 0);
                                                        result = BetaBra1 * Sx1 * BetaKet1;
                                                        Sx1BB = result(0, 0);

                                                        result = AlphaBra1 * Sy1 * AlphaKet1;
                                                        Sy1AA = result(0, 0);
                                                        result = AlphaBra1 * Sy1 * BetaKet1;
                                                        Sy1AB = result(0, 0);
                                                        result = BetaBra1 * Sy1 * BetaKet1;
                                                        Sy1BB = result(0, 0);

                                                        result = AlphaBra1 * Sz1 * AlphaKet1;
                                                        Sz1AA = result(0, 0);
                                                        result = AlphaBra1 * Sz1 * BetaKet1;
                                                        Sz1AB = result(0, 0);
                                                        result = BetaBra1 * Sz1 * BetaKet1;
                                                        Sz1BB = result(0, 0);

                                                        // For P_Tplus

                                                        result = AlphaBra1 * Sp1Sm1 * AlphaKet1;
                                                        Sp1Sm1AA = result(0, 0);
                                                        result = BetaBra1 * Sp1Sm1 * BetaKet1;
                                                        Sp1Sm1BB = result(0, 0);
                                                        result = AlphaBra1 * Sp1Sm1 * BetaKet1;
                                                        Sp1Sm1AB = result(0, 0);

                                                        // For P_Tminus

                                                        result = AlphaBra1 * Sm1Sp1 * AlphaKet1;
                                                        Sm1Sp1AA = result(0, 0);
                                                        result = BetaBra1 * Sm1Sp1 * BetaKet1;
                                                        Sm1Sp1BB = result(0, 0);
                                                        result = AlphaBra1 * Sm1Sp1 * BetaKet1;
                                                        Sm1Sp1AB = result(0, 0);

                                                        for (int it2 = 0; it2 < mc_samples2; it2++)
                                                        {
                                                                AlphaKet2 = B2.col(it2);
                                                                BetaKet2 = B2.col(it2 + mc_samples2);
                                                                AlphaBra2 = B2t.row(it2);
                                                                BetaBra2 = B2t.row(it2 + mc_samples2);

                                                                // For P_S

                                                                result = AlphaBra2 * Sx2 * AlphaKet2;
                                                                Sx2AA = result(0, 0);
                                                                result = BetaBra2 * Sx2 * AlphaKet2;
                                                                Sx2BA = result(0, 0);
                                                                result = BetaBra2 * Sx2 * BetaKet2;
                                                                Sx2BB = result(0, 0);
                                                                result = AlphaBra2 * Sy2 * AlphaKet2;
                                                                Sy2AA = result(0, 0);
                                                                result = BetaBra2 * Sy2 * AlphaKet2;
                                                                Sy2BA = result(0, 0);
                                                                result = BetaBra2 * Sy2 * BetaKet2;
                                                                Sy2BB = result(0, 0);

                                                                result = AlphaBra2 * Sz2 * AlphaKet2;
                                                                Sz2AA = result(0, 0);
                                                                result = BetaBra2 * Sz2 * AlphaKet2;
                                                                Sz2BA = result(0, 0);
                                                                result = BetaBra2 * Sz2 * BetaKet2;
                                                                Sz2BB = result(0, 0);

                                                                // For P_Tplus

                                                                result = AlphaBra2 * Sp2Sm2 * AlphaKet2;
                                                                Sp2Sm2AA = result(0, 0);
                                                                result = BetaBra2 * Sp2Sm2 * BetaKet2;
                                                                Sp2Sm2BB = result(0, 0);
                                                                result = AlphaBra2 * Sp2Sm2 * BetaKet2;
                                                                Sp2Sm2AB = result(0, 0);

                                                                // For P_Tminus

                                                                result = AlphaBra2 * Sm2Sp2 * AlphaKet2;
                                                                Sm2Sp2AA = result(0, 0);
                                                                result = BetaBra2 * Sm2Sp2 * BetaKet2;
                                                                Sm2Sp2BB = result(0, 0);
                                                                result = AlphaBra2 * Sm2Sp2 * BetaKet2;
                                                                Sm2Sp2AB = result(0, 0);

                                                                temp1 = arma::cx_double(0.25, 0.0) - arma::cx_double(0.5, 0) * (Sx1AA * Sx2BB + 2 * (Sx1AB * Sx2BA).real() + Sx1BB * Sx2AA + Sy1AA * Sy2BB + 2 * (Sy1AB * Sy2BA).real() + Sy1BB * Sy2AA + Sz1AA * Sz2BB + 2 * (Sz1AB * Sz2BA).real() + Sz1BB * Sz2AA);
                                                                temp2 = arma::cx_double(0.5, 0.0) * (Sp1Sm1AA * Sp2Sm2BB + 2 * (Sp1Sm1AB * Sp2Sm2AB).real() + Sp1Sm1BB * Sp2Sm2AA);
                                                                temp3 = arma::cx_double(0.5, 0.0) * (Sm1Sp1AA * Sm2Sp2BB + 2 * (Sm1Sp1AB * Sm2Sp2AB).real() + Sm1Sp1BB * Sm2Sp2AA);

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }

                                        double expected_value1 = std::exp(-kmin * current_time) * Ps.real() / mc_samples1 / mc_samples2;
                                        double expected_value2 = std::exp(-kmin * current_time) * Ptplus.real() / mc_samples1 / mc_samples2;
                                        double expected_value3 = std::exp(-kmin * current_time) * Ptzero.real() / mc_samples1 / mc_samples2;
                                        double expected_value4 = std::exp(-kmin * current_time) * Ptminus.real() / mc_samples1 / mc_samples2;

                                        this->Data() << " " << expected_value1;
                                        this->Data() << " " << expected_value2;
                                        this->Data() << " " << expected_value3;
                                        this->Data() << " " << expected_value4;
                                        this->Data() << std::endl;

                                        // Update B using the Higham propagator
                                        B1 = spaces[0].HighamProp(H1, B1, -dt * arma::cx_double(0.0, 1.0), precision, M1);
                                        B2 = spaces[0].HighamProp(H2, B2, -dt * arma::cx_double(0.0, 1.0), precision, M2);
                                }
                        }
                        else if (InitialState == "tripletplus")
                        {
                                arma::cx_vec AlphaKet1(2 * Z1, 1);
                                arma::cx_vec AlphaBra1(1, 2 * Z1);
                                arma::cx_vec AlphaKet2(2 * Z2, 1);
                                arma::cx_vec AlphaBra2(1, 2 * Z2);

                                arma::cx_double Sx1AA;
                                arma::cx_double Sy1AA;
                                arma::cx_double Sz1AA;

                                arma::cx_double Sx2AA;
                                arma::cx_double Sy2AA;
                                arma::cx_double Sz2AA;

                                arma::cx_double Sp1Sm1AA;
                                arma::cx_double Sp2Sm2AA;

                                arma::cx_double Sm1Sp1AA;
                                arma::cx_double Sm2Sp2AA;

                                arma::cx_mat result(1, 1);

                                arma::cx_mat B1t(mc_samples1, 2 * Z1);
                                arma::cx_mat B2t(mc_samples2, 2 * Z2);

                                for (int indx = 0; indx < num_steps; indx++)
                                {
                                        // Set the current time
                                        double current_time = indx * dt;

                                        this->Data() << current_time / (1e-3 * gamma_e);

                                        arma::cx_double Ps;
                                        arma::cx_double Ptplus;
                                        arma::cx_double Ptzero;
                                        arma::cx_double Ptminus;
                                        arma::cx_double temp1;
                                        arma::cx_double temp2;
                                        arma::cx_double temp3;
                                        arma::cx_double temp4;
                                        B1t = B1.t();
                                        B2t = B2.t();

                                        if (mc_samples1 < mc_samples2)
                                        {

                                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                                {

                                                        AlphaKet2 = B2.col(it2);
                                                        AlphaBra2 = B2t.row(it2);

                                                        // For P_S
                                                        result = AlphaBra2 * Sx2 * AlphaKet2;
                                                        Sx2AA = result(0, 0);
                                                        result = AlphaBra2 * Sy2 * AlphaKet2;
                                                        Sy2AA = result(0, 0);
                                                        result = AlphaBra2 * Sz2 * AlphaKet2;
                                                        Sz2AA = result(0, 0);

                                                        // For P_Tplus

                                                        result = AlphaBra2 * Sp2Sm2 * AlphaKet2;
                                                        Sp2Sm2AA = result(0, 0);

                                                        // For P_Tminus

                                                        result = AlphaBra2 * Sm2Sp2 * AlphaKet2;
                                                        Sm2Sp2AA = result(0, 0);

                                                        for (int it1 = 0; it1 < mc_samples1; it1++)
                                                        {
                                                                AlphaKet1 = B1.col(it1);
                                                                AlphaBra1 = B1t.row(it1);

                                                                // P_S

                                                                result = AlphaBra1 * Sx1 * AlphaKet1;
                                                                Sx1AA = result(0, 0);
                                                                result = AlphaBra1 * Sy1 * AlphaKet1;
                                                                Sy1AA = result(0, 0);
                                                                result = AlphaBra1 * Sz1 * AlphaKet1;
                                                                Sz1AA = result(0, 0);

                                                                // For P_Tplus

                                                                result = AlphaBra1 * Sp1Sm1 * AlphaKet1;
                                                                Sp1Sm1AA = result(0, 0);

                                                                // For P_Tminus

                                                                result = AlphaBra1 * Sm1Sp1 * AlphaKet1;
                                                                Sm1Sp1AA = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - (Sx1AA * Sx2AA + Sy1AA * Sy2AA + Sz1AA * Sz2AA);
                                                                temp2 = Sp1Sm1AA * Sp2Sm2AA;
                                                                temp3 = Sm1Sp1AA * Sm2Sp2AA;

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }
                                        else
                                        {
                                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                                {
                                                        AlphaKet1 = B1.col(it1);
                                                        AlphaBra1 = B1t.row(it1);

                                                        // P_S

                                                        result = AlphaBra1 * Sx1 * AlphaKet1;
                                                        Sx1AA = result(0, 0);
                                                        result = AlphaBra1 * Sy1 * AlphaKet1;
                                                        Sy1AA = result(0, 0);
                                                        result = AlphaBra1 * Sz1 * AlphaKet1;
                                                        Sz1AA = result(0, 0);

                                                        // For P_Tplus

                                                        result = AlphaBra1 * Sp1Sm1 * AlphaKet1;
                                                        Sp1Sm1AA = result(0, 0);

                                                        // For P_Tminus

                                                        result = AlphaBra1 * Sm1Sp1 * AlphaKet1;
                                                        Sm1Sp1AA = result(0, 0);
                                                        for (int it2 = 0; it2 < mc_samples2; it2++)
                                                        {
                                                                AlphaKet2 = B2.col(it2);
                                                                AlphaBra2 = B2t.row(it2);

                                                                // For P_S
                                                                result = AlphaBra2 * Sx2 * AlphaKet2;
                                                                Sx2AA = result(0, 0);
                                                                result = AlphaBra2 * Sy2 * AlphaKet2;
                                                                Sy2AA = result(0, 0);
                                                                result = AlphaBra2 * Sz2 * AlphaKet2;
                                                                Sz2AA = result(0, 0);

                                                                // For P_Tplus

                                                                result = AlphaBra2 * Sp2Sm2 * AlphaKet2;
                                                                Sp2Sm2AA = result(0, 0);

                                                                // For P_Tminus

                                                                result = AlphaBra2 * Sm2Sp2 * AlphaKet2;
                                                                Sm2Sp2AA = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - (Sx1AA * Sx2AA + Sy1AA * Sy2AA + Sz1AA * Sz2AA);
                                                                temp2 = Sp1Sm1AA * Sp2Sm2AA;
                                                                temp3 = Sm1Sp1AA * Sm2Sp2AA;

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }

                                        double expected_value1 = std::exp(-kmin * current_time) * Ps.real() / mc_samples1 / mc_samples2;
                                        double expected_value2 = std::exp(-kmin * current_time) * Ptplus.real() / mc_samples1 / mc_samples2;
                                        double expected_value3 = std::exp(-kmin * current_time) * Ptzero.real() / mc_samples1 / mc_samples2;
                                        double expected_value4 = std::exp(-kmin * current_time) * Ptminus.real() / mc_samples1 / mc_samples2;

                                        this->Data() << " " << expected_value1;
                                        this->Data() << " " << expected_value2;
                                        this->Data() << " " << expected_value3;
                                        this->Data() << " " << expected_value4;
                                        this->Data() << std::endl;

                                        // Update B using the Higham propagator
                                        B1 = spaces[0].HighamProp(H1, B1, -dt * arma::cx_double(0.0, 1.0), precision, M1);
                                        B2 = spaces[0].HighamProp(H2, B2, -dt * arma::cx_double(0.0, 1.0), precision, M2);
                                }
                        }
                        else if (InitialState == "tripletminus")
                        {
                                arma::cx_vec BetaKet1(2 * Z1, 1);
                                arma::cx_vec BetaBra1(1, 2 * Z1);
                                arma::cx_vec BetaKet2(2 * Z2, 1);
                                arma::cx_vec BetaBra2(1, 2 * Z2);

                                arma::cx_double Sx1BB;
                                arma::cx_double Sy1BB;
                                arma::cx_double Sz1BB;

                                arma::cx_double Sx2BB;
                                arma::cx_double Sy2BB;
                                arma::cx_double Sz2BB;

                                arma::cx_double Sp1Sm1BB;
                                arma::cx_double Sp2Sm2BB;

                                arma::cx_double Sm1Sp1BB;
                                arma::cx_double Sm2Sp2BB;

                                arma::cx_mat result(1, 1);

                                arma::cx_mat B1t(mc_samples1, 2 * Z1);
                                arma::cx_mat B2t(mc_samples2, 2 * Z2);

                                for (int indx = 0; indx < num_steps; indx++)
                                {
                                        // Set the current time
                                        double current_time = indx * dt;

                                        this->Data() << current_time / (1e-3 * gamma_e);

                                        arma::cx_double Ps;
                                        arma::cx_double Ptplus;
                                        arma::cx_double Ptzero;
                                        arma::cx_double Ptminus;
                                        arma::cx_double temp1;
                                        arma::cx_double temp2;
                                        arma::cx_double temp3;
                                        arma::cx_double temp4;
                                        B1t = B1.t();
                                        B2t = B2.t();
                                        if (mc_samples1 < mc_samples2)
                                        {

                                                for (int it2 = 0; it2 < mc_samples2; it2++)
                                                {
                                                        BetaKet2 = B2.col(it2);
                                                        BetaBra2 = B2t.row(it2);

                                                        // For P_S
                                                        result = BetaBra2 * Sx2 * BetaKet2;
                                                        Sx2BB = result(0, 0);
                                                        result = BetaBra2 * Sy2 * BetaKet2;
                                                        Sy2BB = result(0, 0);
                                                        result = BetaBra2 * Sz2 * BetaKet2;
                                                        Sz2BB = result(0, 0);

                                                        // For P_Tplus

                                                        result = BetaBra2 * Sp2Sm2 * BetaKet2;
                                                        Sp2Sm2BB = result(0, 0);

                                                        // For P_Tminus

                                                        result = BetaBra2 * Sm2Sp2 * BetaKet2;
                                                        Sm2Sp2BB = result(0, 0);

                                                        for (int it1 = 0; it1 < mc_samples1; it1++)
                                                        {
                                                                BetaKet1 = B1.col(it1);
                                                                BetaBra1 = B1t.row(it1);

                                                                // P_S

                                                                result = BetaBra1 * Sx1 * BetaKet1;
                                                                Sx1BB = result(0, 0);
                                                                result = BetaBra1 * Sy1 * BetaKet1;
                                                                Sy1BB = result(0, 0);
                                                                result = BetaBra1 * Sz1 * BetaKet1;
                                                                Sz1BB = result(0, 0);

                                                                // For P_Tplus

                                                                result = BetaBra1 * Sp1Sm1 * BetaKet1;
                                                                Sp1Sm1BB = result(0, 0);

                                                                // For P_Tminus

                                                                result = BetaBra1 * Sm1Sp1 * BetaKet1;
                                                                Sm1Sp1BB = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - (Sx1BB * Sx2BB + Sy1BB * Sy2BB + Sz1BB * Sz2BB);
                                                                temp2 = Sp1Sm1BB * Sp2Sm2BB;
                                                                temp3 = Sm1Sp1BB * Sm2Sp2BB;

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }
                                        else
                                        {
                                                for (int it1 = 0; it1 < mc_samples1; it1++)
                                                {
                                                        BetaKet1 = B1.col(it1);
                                                        BetaBra1 = B1t.row(it1);

                                                        // P_S

                                                        result = BetaBra1 * Sx1 * BetaKet1;
                                                        Sx1BB = result(0, 0);
                                                        result = BetaBra1 * Sy1 * BetaKet1;
                                                        Sy1BB = result(0, 0);
                                                        result = BetaBra1 * Sz1 * BetaKet1;
                                                        Sz1BB = result(0, 0);

                                                        // For P_Tplus

                                                        result = BetaBra1 * Sp1Sm1 * BetaKet1;
                                                        Sp1Sm1BB = result(0, 0);

                                                        // For P_Tminus

                                                        result = BetaBra1 * Sm1Sp1 * BetaKet1;
                                                        Sm1Sp1BB = result(0, 0);

                                                        for (int it2 = 0; it2 < mc_samples2; it2++)
                                                        {
                                                                BetaKet2 = B2.col(it2);
                                                                BetaBra2 = B2t.row(it2);

                                                                // For P_S
                                                                result = BetaBra2 * Sx2 * BetaKet2;
                                                                Sx2BB = result(0, 0);
                                                                result = BetaBra2 * Sy2 * BetaKet2;
                                                                Sy2BB = result(0, 0);
                                                                result = BetaBra2 * Sz2 * BetaKet2;
                                                                Sz2BB = result(0, 0);

                                                                // For P_Tplus

                                                                result = BetaBra2 * Sp2Sm2 * BetaKet2;
                                                                Sp2Sm2BB = result(0, 0);

                                                                // For P_Tminus

                                                                result = BetaBra2 * Sm2Sp2 * BetaKet2;
                                                                Sm2Sp2BB = result(0, 0);

                                                                // Calculating expectation values

                                                                temp1 = arma::cx_double(0.25, 0.0) - (Sx1BB * Sx2BB + Sy1BB * Sy2BB + Sz1BB * Sz2BB);
                                                                temp2 = Sp1Sm1BB * Sp2Sm2BB;
                                                                temp3 = Sm1Sp1BB * Sm2Sp2BB;

                                                                Ps += temp1;
                                                                Ptplus += temp2;
                                                                Ptminus += temp3;
                                                                Ptzero += arma::cx_double(1.0, 0.0) - temp1 - temp2 - temp3;
                                                        }
                                                }
                                        }

                                        double expected_value1 = std::exp(-kmin * current_time) * Ps.real() / mc_samples1 / mc_samples2;
                                        double expected_value2 = std::exp(-kmin * current_time) * Ptplus.real() / mc_samples1 / mc_samples2;
                                        double expected_value3 = std::exp(-kmin * current_time) * Ptzero.real() / mc_samples1 / mc_samples2;
                                        double expected_value4 = std::exp(-kmin * current_time) * Ptminus.real() / mc_samples1 / mc_samples2;

                                        // Obtain results
                                        // this->Data() << this->RunSettings()->CurrentStep() << " ";
                                        this->WriteStandardOutput(this->Data());

                                        this->Data() << " " << expected_value1;
                                        this->Data() << " " << expected_value2;
                                        this->Data() << " " << expected_value3;
                                        this->Data() << " " << expected_value4;
                                        this->Data() << std::endl;

                                        // Update B using the Higham propagator
                                        B1 = spaces[0].HighamProp(H1, B1, -dt * arma::cx_double(0.0, 1.0), precision, M1);
                                        B2 = spaces[0].HighamProp(H2, B2, -dt * arma::cx_double(0.0, 1.0), precision, M2);
                                }
                        }

                        this->Log() << "\nDone with SpinSystem \"" << (*i)->Name() << "\"" << std::endl;
                }
                this->Data() << std::endl;
                return true;
        }

        // Writes the header of the data file (but can also be passed to other streams)
        void TaskStaticHSStochTimeEvoSymmUncoupled::WriteHeader(std::ostream &_stream)
        {
                _stream << "Step ";
                this->WriteStandardOutputHeader(_stream);
                _stream << "P_S P_Tplus P_Tzero P_Tminus" << std::endl;
        }

        // Validation
        bool TaskStaticHSStochTimeEvoSymmUncoupled::Validate()
        {
                this->Properties()->Get("transitionyields", this->productYieldsOnly);

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

}
