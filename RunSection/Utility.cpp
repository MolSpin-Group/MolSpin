/////////////////////////////////////////////////////////////////////////
// Utility implementation (RunSection module)
//
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////

#include "Utility.h"
#include <random>
#include <thread>
#include "PulseSequence.h"
#include "Pulse.h"
#include <sstream>
#include "Interaction.h"
#include "Transition.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static double timestep_floor = 1e-12;

namespace RunSection
{

    struct ArnoldiResult 
    {
        arma::field<arma::cx_vec> V;
        arma::sp_cx_mat Hessian;
        double Beta;
        std::complex<double> h_res;
    };

    arma::cx_mat CubicSplineInterpolation(Buffer<arma::cx_mat>& buff, double T)
    {
        arma::cx_vec timeList = buff.time();
        MatrixSpline ms;
        ms.build(timeList, buff.rhoPoints);
        arma::cx_mat rho_Tk = ms.Eval(T);
        return rho_Tk;
    }

    FibSpherePoint *CalculateFibPoints(int n)
    {
        FibSpherePoint* TempPointArray = (FibSpherePoint*)malloc(n * sizeof(FibSpherePoint));
        if(TempPointArray == NULL)
        {
            std::cout << "Memory not allocated" << std::endl;
            return nullptr;
        }

        double phi = M_PI * (3.0 - std::sqrt(5.0)); //Golden angle in radians
        for (int i = 0; i < n; i++)
        {
            double y = 1.0 - ((double)i / (double)(n-1)) * 2;
            double theta = phi * (double)i;

            TempPointArray[i] = {y,theta};
        }
        return TempPointArray;
    }

    bool RetrievePoint(std::array<double, 3> &arr, FibSpherePoint* ptr, int num)
    {
        FibSpherePoint p =  ptr[num];

        float y = p.first;
        float theta = p.second;
        
        double r = std::sqrt(1.0 - (y * y));
        double x = std::cos(theta) * r;
        double z = std::sin(theta) * r;

        double yd = (double)y;

        arr = {x, yd, z};
        return true;
    }

    //std::tuple<std::vector<SpinAPI::pulse_ptr>,std::vector<double>> EvaluatePulseSequence(std::vector<SpinAPI::pulse_ptr> pulses, SpinAPI::PulseSequence PulseSeq)
    //{
    //    std::vector<SpinAPI::pulse_ptr> pulse_sequence;
    //    std::vector<double> gaps;
//
    //    pulse_sequence.reserve(PulseSeq.size());
    //    gaps.reserve(PulseSeq.size());
//
    //    std::unordered_map<std::string, SpinAPI::pulse_ptr> pulse_map;
    //    pulse_map.reserve(pulses.size())
    //    for(const auto& p : pulses)
    //    {
    //        if (p) 
    //        {
    //            pulse_map[p->Name()] = p;
    //        }
    //    }
//
    //    for(const auto &seq : PulseSeq)
    //    {
    //        auto[pulse,tau] = seq;
    //        auto it = pulse_map.find(pulse);
    //        if(it != pulse_map.end())
    //        {
    //            pulse_sequence.push_back(it->second);
    //            gaps.push_back(tau);
    //        }
    //        else
    //        {
    //            std::cerr << "Pulse: " << pulse << " not found in pulse list";
    //        }
    //    }
//
    //}

    std::vector<block> GenerateTimeEvoBlocking(std::vector<SpinAPI::PulseSequence_ptr>& seq, std::pair<double,double> MinMaxTimesteps, double TotalEvoTime)
    {

        std::vector<std::vector<PulseEvent>> timelines(seq.size());
        std::vector<double> critical_time_points;
        //double global_start = (offset > 0.0) ? offset : 0.0;
        double global_start = 0.0;
        critical_time_points.push_back(0.0);

        for(size_t ch = 0; ch < seq.size(); ch++)
        {
            const auto& track = seq[ch];
            double track_time = 0.0;
            double ch_offset = track->Get_offset();
            if(ch_offset > 0.0) 
            {
                PulseEvent o_event;
                o_event.time = 0.0;
                o_event.is_pulse = false;
                o_event.pulse = SpinAPI::SequenceObject();
                timelines[ch].push_back(o_event);
                critical_time_points.push_back(ch_offset);
                track_time = ch_offset;
            }
            for(auto step = track->begin(); step != track->end(); step++)
            {
                auto[seq_step, tau_key] = *step; //need to change this to handle sequence steps
                auto[pulse_event, interaction_event, transition_event] = seq_step.get();
                double pulse_duration = 0.0;
                if (pulse_event)
                    pulse_duration = (pulse_event->Type() == SpinAPI::PulseType::InstantPulse) ? 0.0 : pulse_event->Pulsetime();
                else
                    pulse_duration = (interaction_event) ? interaction_event->ActiveTime() : transition_event->ActiveTime();
                PulseEvent p_event;
                p_event.time = track_time;
                p_event.pulse = seq_step;
                p_event.is_pulse = true;
                timelines[ch].push_back(p_event);

                track_time += pulse_duration;
                critical_time_points.push_back(track_time);

                double gap_duration = 0.0;
                auto it = track->Get_tau_list().find(tau_key);
                if (it != track->Get_tau_list().end()) 
                {
                    gap_duration = it->second;
                }

                if (gap_duration > 0.0) 
                {
                    PulseEvent g_event;
                    g_event.time = track_time;
                    g_event.is_pulse = false;
                    g_event.pulse = SpinAPI::SequenceObject();
                    timelines[ch].push_back(g_event);
                    
                    track_time += gap_duration;
                    critical_time_points.push_back(track_time);
                }
            }
        }

        std::sort(critical_time_points.begin(), critical_time_points.end());
        critical_time_points.erase(std::unique(critical_time_points.begin(), critical_time_points.end(), [](double a, double b) { return std::abs(a - b) < 1e-14; }), critical_time_points.end());

        std::vector<block> blocks = {};
        bool empty = seq.empty();
        if (empty)
        {
            block first_block;
            first_block.start = 0.0;
            first_block.end = TotalEvoTime;
            first_block.max_timestep = MinMaxTimesteps.second;
            first_block.min_timestep = MinMaxTimesteps.first;
            first_block.free_evolution = true;
            blocks.push_back(first_block);
            return blocks;
        }

        for(size_t i = 0; i < critical_time_points.size() - 1; i++)
        {
            double b_start = critical_time_points[i];
            double b_end = critical_time_points[i+1];

            if(std::abs(b_end - b_start) < 1e-14) continue;

            bool global_free = true;
            double block_min = MinMaxTimesteps.first;
            double block_max = MinMaxTimesteps.second;

            for (size_t ch = 0; ch < seq.size(); ch++)
            {
                const auto& timeline = timelines[ch];
                if(timeline.empty()) continue;
                auto it = std::upper_bound(timeline.begin(), timeline.end(), b_start, [](double val, const PulseEvent& ev) { return val < ev.time;});
                if (it != timeline.begin())
                {
                    auto active_ev = *(it-1);
                    if(active_ev.is_pulse && !active_ev.pulse.IsNullptr()) 
                    {
                        global_free = false;
                        auto[pulse_event, interaction_event, transition_event] = active_ev.pulse.get();
                        double ps,ts = 0.0;
                        if(pulse_event)
                        {
                            ps = (pulse_event->Type() == SpinAPI::PulseType::InstantPulse) ? 0.0 : pulse_event->Pulsetime();
                            ts = pulse_event->Timestep();
                        }
                        else
                        {
                            ps = (interaction_event) ? interaction_event->ActiveTime() : transition_event->ActiveTime();
                        }

                        if(pulse_event)
                        {
                            if(pulse_event->Type() != SpinAPI::PulseType::InstantPulse)
                            {
                                double max_allowed = std::min(ps / 20.0, ts*1e3);
                                double pulse_max = std::min(max_allowed, MinMaxTimesteps.second);
                                double pulse_min = std::min(MinMaxTimesteps.first, ps / 1e3);
                            
                                
                                block_max = std::min(block_max, pulse_max);
                                block_min = std::min(block_min, pulse_min);
                            }
                            else
                            {
                                block_max = MinMaxTimesteps.first;
                                block_min = MinMaxTimesteps.second;
                            }
                        }
                        else
                        {
                            double max_allowed = ps / 20.0;
                            double pulse_max = std::min(max_allowed, MinMaxTimesteps.second);
                            double pulse_min = std::min(MinMaxTimesteps.first, ps / 1e3);
                            
                                
                            block_max = std::min(block_max, pulse_max);
                            block_min = std::min(block_min, pulse_min);
                        }
                    }
                }
            }

            block new_block;
            new_block.start = b_start;
            new_block.end = b_end;
            new_block.max_timestep = block_max;
            new_block.min_timestep = block_min;
            new_block.free_evolution = global_free;
            blocks.push_back(new_block);
        }

        return blocks;

    }

    std::string PrintOutBlockStructure(const std::vector<block>& blocks)
    {
        std::ostringstream ss;
        ss << "\n" << std::string(90, '=') << "\n";
        ss << "\t\t\t\t Generated time evolution block sections\n";
        ss << std::string(90, '=') << "\n";

        ss << std::left 
           << std::setw(8)  << "Block section "
           << std::setw(18) << "Start Time (ns)"
           << std::setw(18) << "End Time (ns)"
           << std::setw(16) << "Duration (ns)"
           << std::setw(16) << "Max Timestep"
           << std::setw(16) << "Min Timestep"
           << "Evolution Type\n";
        ss << std::string(90, '-') << "\n";
        ss << std::scientific << std::setprecision(6);

        for (size_t i = 0; i < blocks.size(); ++i)
        {
            const auto& b = blocks[i];
            double duration = b.end - b.start;
            
            std::string evo_type = b.free_evolution ? "FREE EVOLUTION" : "ACTIVE PULSE(S)";

            ss << std::left
               << std::setw(8)  << i
               << std::setw(18) << b.start
               << std::setw(18) << b.end
               << std::setw(16) << duration
               << std::setw(16) << b.max_timestep
               << std::setw(16) << b.min_timestep
               << evo_type << "\n";
        }
        ss << std::string(90, '=') << "\n\n";
        std::string return_str = ss.str();
        return return_str;
    }

    void ClampTimeEvolution(double ctime, double ttime, const std::vector<block>& blocks, size_t& current_block, double& block_timestep, SpinAPI::SpinSpace::PropParam& params)
    {
        while(current_block < blocks.size() && ctime >= blocks[current_block].end)
        {
            current_block++;
        }
        const block& active_block = (current_block < blocks.size())
                                  ? blocks[current_block]
                                  : blocks.back();
        params.max = active_block.max_timestep;
        params.min = active_block.min_timestep;

        if(block_timestep > active_block.max_timestep)
        {
            block_timestep = active_block.max_timestep;
        }
        if(block_timestep < active_block.min_timestep)
        {
            block_timestep = active_block.min_timestep;
        }

        if(ctime + block_timestep > active_block.end)
        {
            block_timestep = active_block.end - ctime;
        }
        if(ctime + block_timestep > ttime)
        {
            block_timestep = ttime - ctime;
        }
    }

    MCSpherePoint* CalculateMCSpherePoints(int n, double rmax_x, double rmax_y, double rmax_z)
    {
        MCSpherePoint* TempPointArray = (MCSpherePoint*)malloc(n * sizeof(MCSpherePoint));
        if(TempPointArray == NULL)
        {
            std::cout << "Memory not allocated" << std::endl;
            return nullptr;
        }
        std::random_device RandDev;
        std::mt19937 Generator(RandDev());
        std::uniform_real_distribution<double> distPhi(0,2.0*M_PI);
        std::uniform_real_distribution<double> distTheta(0,M_PI);
        std::uniform_real_distribution<double> distScale(0,1);
        
        for(int i = 0; i < n; i++)
        {
            double phi = distPhi(Generator);
            double theta = distTheta(Generator);
            double scalefactor = std::pow(distScale(Generator), 1.0/3.0);
            
            double r_x = scalefactor * rmax_x;
            double r_y = scalefactor * rmax_y;
            double r_z = scalefactor * rmax_z;
            TempPointArray[i] = {theta, phi, {r_x, r_y, r_z}};
            //TempPointArray[i] = {theta,phi,r};
        }

        std::vector<MCSpherePoint> UniquePoints;
        UniquePoints.push_back(TempPointArray[0]);  
        for (int i = 1; i < n; i++)
        {
            if(std::find(UniquePoints.begin(), UniquePoints.end(), TempPointArray[i]) == UniquePoints.end())
            {
                UniquePoints.push_back(TempPointArray[i]);
            }
            else
            {
                std::cin.get();
            }
        }
        return TempPointArray;
    }

    MCSpherePoint* CalculateMCSpherePoints(int n, double rmax)
    {
        return CalculateMCSpherePoints(n, rmax, rmax, rmax);
    }


    bool RetrieveMCPoint(std::array<double, 3> &arr, MCSpherePoint *ptr, int num)
    {
        MCSpherePoint p = ptr[num];
        
        double theta = p.theta;
        double phi = p.phi;
        auto r = p.r;

        double x = r[0] * std::sin(theta) * std::cos(phi);
        double y = r[1] * std::sin(theta) * std::sin(phi);
        double z = r[2] * std::cos(theta);

        arr = {x,y,z};
        return true;
    }

    SCData GetHamiltonian(arma::sp_cx_mat& CompositeMatrix, int Dimension)
    {
        SCData container;
        arma::sp_cx_mat H = CompositeMatrix.submat(0,0,Dimension-1,Dimension-1);
        arma::sp_cx_mat SampleMatrix = CompositeMatrix.submat(Dimension,0,CompositeMatrix.n_rows-1,CompositeMatrix.n_cols-1);
        for (unsigned int i = 0; i < SampleMatrix.n_rows; i += Dimension)
        {
            int samples = 0;
            for (unsigned int e = 0; e < SampleMatrix.n_cols; e+= Dimension)
            {
                arma::sp_cx_mat SubMat = SampleMatrix.submat(i,e,i+Dimension-1,e+Dimension-1);
                if(SubMat.n_nonzero == 0)
                    break;
                
                samples += 1;
            }
            container.samples.push_back(samples);
        }
        container.H = H;
        container.SamplesMatrix = SampleMatrix;
        container.BlockSize = Dimension;
        
        return container;
    }

    arma::sp_cx_mat GetHamiltonian(const arma::sp_cx_mat H0, int S, const std::vector<arma::sp_cx_mat> HSC, std::vector<std::vector<int>> samples)
    {
        int Dimension = H0.n_rows / S;
        arma::sp_cx_mat H = H0;
        for (int i = 0; i < S; i++)
        {
           int interactions = samples[i].size();
           for (int e = 0; e < interactions; e++)
           {
                int col = samples[i][e] * Dimension;
                int row = e * Dimension;

                arma::sp_cx_mat sample = HSC[i].submat(row, col, row + Dimension - 1, col + Dimension -1);
                H.submat(i * Dimension, i * Dimension, (i+1)*Dimension -1, (i+1)*Dimension -1) += arma::cx_double(0.0,-1.0) * sample;
           } 
        }
        return H;
    }

    std::vector<SampleCombination> GenerateCombinationsNI(const std::vector<std::vector<int>>& orientations , int startpoint, int endpoint)
    {
        int num = 1;
        std::vector<int> steps;
        std::vector<int> NumInteractionsPerSpinSystem;
        std::vector<SampleCombination> samples;
        int samplelength = 0;
        for(unsigned int i = 0; i < orientations.size(); i++)
        {
            samplelength += orientations[i].size();
            NumInteractionsPerSpinSystem.push_back(orientations[i].size());
            for(unsigned int e = 0; e < orientations[i].size(); e++)
            {
                steps.push_back(orientations[i][e]);
                num = num * steps[samplelength-orientations[i].size()+e];
            }
        }

        //generate num samples of length samplelength
        std::vector<int> Combination;
        for(int i = 0; i < samplelength; i++)
        {
            Combination.push_back(0);
        }
        int depth = steps.size();
        Combination[depth-1] = -1;
        if(endpoint == 0)
        {
            endpoint = num;
        }
        if(startpoint != 0)
        {
            //calculate start combination
            std::vector<int> remainders = {};
            int r = startpoint % steps[depth-1];
            remainders.insert(remainders.begin(),r);
            int temp = startpoint - r;
            int m = temp / steps[depth-1];
            for (int i = depth-2; i >= 0; i--)
            {
                r = m % steps[i];
                temp = m - r;
                remainders.insert(remainders.begin(),r);
                m = temp / steps[i];
            }
            Combination = remainders;
            startpoint +=1;
        }
        for (int i = startpoint; i < endpoint; i++)
        {
            Combination[depth-1] += 1;
            for (int e = depth-2; e >= 0; e--)
            {
                if(Combination[e+1] == steps[e+1])
                {
                    Combination[e] += 1;
                    Combination[e+1] = 0;
                    continue;
                }
            }

            int s = 0;
            SampleCombination s1;
            for(unsigned int a = 0; a < NumInteractionsPerSpinSystem.size(); a++)
            {
                std::vector<int> temp;
                for(int e = 0; e < NumInteractionsPerSpinSystem[a]; e++)
                {
                    temp.push_back(Combination[e+s]);
                }
                s1.push_back(temp);
                s += NumInteractionsPerSpinSystem[a];
            }
            samples.push_back(s1);
        }
        return samples;
        
    }

    typedef arma::sp_cx_mat MatrixArma;
    typedef arma::cx_vec VecType;

    double EstimateStiffnessArmadillo(arma::sp_cx_mat &L)
    {
        static double tol = 1e-3;
        static int max_iterations = 1000;

        //get symmetric part of L
        arma::sp_cx_mat Ls = 0.5 * (L + arma::trans(L));
        Ls = -1 * Ls;
        //create random wavefunction 
        arma::cx_vec psi(L.n_rows);
        size_t n = L.n_rows;
        #pragma omp parallel
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, 1.0);
            #pragma omp for
            for(size_t i = 0; i < n; i++)
            {
                double re = dist(gen);
                double img = dist(gen);
                psi(i) = std::complex<double>(re,img);
            }
        }

        double norm = arma::norm(psi,2);
        //std::cout << psi << std::endl;
        psi = psi / norm;
        arma::cx_vec psi2 = psi;

        std::vector<std::complex<double>> max_eigenvalue = {0.0,0.0};
        std::vector<std::complex<double>>old_eigenvalue = {0.0,0.0};
        arma::cx_vec phi(L.n_rows);
        arma::cx_vec phi2(L.n_rows);

        for(int i = 0; i < max_iterations; i++)
        {
            phi = Ls * psi;
            phi2 = L * psi2;
            max_eigenvalue[0] = arma::norm(phi,2);
            max_eigenvalue[1] = arma::norm(phi2,2);
            psi = phi / max_eigenvalue[0];
            psi2 = phi2 / max_eigenvalue[1];

            if(std::abs(max_eigenvalue[0] - old_eigenvalue[0]) <= tol && std::abs(max_eigenvalue[1] - old_eigenvalue[1]) <= tol)
            {
                break;
            }

            old_eigenvalue = max_eigenvalue;
        }

        double stiffness = std::abs(max_eigenvalue[1]) / std::abs(max_eigenvalue[0]);
        return stiffness;
    }

    SpinAPI::SpinSpace::TimePropReturnInfo RungeKutta45Armadillo(arma::sp_cx_mat &L, arma::cx_vec &rho0, arma::cx_vec &drhodt, double dumpstep, RungeKuttaFuncArma func, double time, SpinAPI::SpinSpace::PropParam params)
    {
        VecType k0(rho0.n_rows);
        double used = 0.0;

        std::vector<std::pair<float, std::vector<float>>> ButcherTable = {{0.0, {}},
                                                                          {0.25, {0.25}},
                                                                          {3.0 / 8.0, {3.0 / 32.0, 9.0 / 32.0}},
                                                                          {12.0 / 13.0, {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0}},
                                                                          {1.0, {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0}},
                                                                          {1.0 / 2.0, {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0}},
                                                                          {0.0, {16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0}},
                                                                          {0.0, {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0}}};
        auto RungeKutta45 = [&k0, &ButcherTable, &time](MatrixArma &L1, VecType &rho01, double t, RungeKuttaFuncArma func1)
        {
            VecType k1(rho01.n_rows);
            VecType k2(rho01.n_rows);
            VecType k3(rho01.n_rows);
            VecType k4(rho01.n_rows);
            VecType k5(rho01.n_rows);
            VecType k6(rho01.n_rows);

            std::vector<VecType> kvec = {k1, k2, k3, k4, k5, k6};

            auto GetK = [&ButcherTable](int index, std::vector<VecType> kv)
            {
                VecType temp(kv[0].n_rows);
                for (int e = 0; e < int(ButcherTable[index].second.size()); e++)
                {
                    temp = temp + (ButcherTable[index].second[e] * kv[e]);
                }
                return temp;
            };

            int i = 0;
            kvec[0] = t * func1(time + ButcherTable[i].first, L1, k0, rho01);
            // std::cout << kvec[0] << std::endl;
            i += 1;
            for (; i < 6; i++)
            {
                VecType temp = GetK(i, kvec);
                // std::cout << temp << std::endl;
                kvec[i] = t * func1(time + ButcherTable[i].first, L1, temp, rho01);
            }

            VecType ReturnVecRK4 = rho01;
            for (i = 0; i < int(ButcherTable[7].second.size()); i++)
            {
                // std::cout << i << std::endl;
                ReturnVecRK4 += (ButcherTable[7].second[i] * kvec[i]);
            }

            VecType ReturnVecRK5 = rho01;
            for (i = 0; i < int(ButcherTable[6].second.size()); i++)
            {
                ReturnVecRK5 += (ButcherTable[6].second[i] * kvec[i]);
            }

            return std::make_tuple(ReturnVecRK4, ReturnVecRK5);
        };

        bool keep_step = false;
        bool first_step = true;
        while(!keep_step)
        {
            auto [RK4, RK5] = RungeKutta45(L, rho0, dumpstep, func);

            double relative_error = 0.0;
            //auto[atol, rtol, min_step, max_step, safety, f1, f2,t1,t2,ct, i1, i2, i3, i4, i5,i6] = params;
            double atol, rtol, min_step, max_step, safety, f1, f2;
            atol = params.atol;
            rtol = params.rtol;
            min_step = params.min;
            max_step = params.max;
            safety = params.safety;
            f1 = params.f1;
            f2 = params.f2;

            double change = 0;
            {
                VecType diff = RK5 - RK4;
                double sum = 0;
                double relative_error_sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
                for (int i = 0; i < int(diff.n_rows); i++)
                {
                    sum += std::pow(std::abs(diff[i]), 2);
                    relative_error_sum += rtol * std::pow(std::abs(RK5[i]), 2); 
                }

                change = std::sqrt(sum);
                relative_error = std::sqrt(relative_error_sum);
            }

            auto Adjusth = [&](double error_ratio, double tol, double ch, double safety, double f1, double f2)
            {
                return dumpstep * std::min(f2, std::max(f1, safety * std::pow(error_ratio, -1.0/5.0)));
            };

            double NewStepSize = 0.0;

            double max_change = atol + relative_error;
            double error_ratio = change / max_change;
            NewStepSize = Adjusth(error_ratio, max_change, change, safety, f1, f2);
            if (error_ratio <= params.reject_limit)
            {
                drhodt = RK4;
                keep_step = true;
                used = dumpstep;
                //dumpstep = NewStepSize;
            }
            if(NewStepSize < params.min && error_ratio >= params.reject_limit)
            {
                params.min = NewStepSize;
                first_step = false;
            }
            if(NewStepSize > params.max)
            {
                NewStepSize = params.max;
                keep_step = true;
                used = dumpstep;
            }
            if(error_ratio >= params.reject_limit)
            {
                first_step = false;
            }
            dumpstep = NewStepSize;

        }
        return {dumpstep,used,first_step,drhodt};
    }

    //TimePropReturnInfo AdaptiveDirectKrylovArmadillo(arma::sp_cx_mat &L, arma::cx_vec &rho0, arma::cx_vec &drhodt, double dumpstep, double time, PropParam PropParams, HamiltonainTimeDepFuncArma GetTDH)
    //{
//
    //    auto ArnoldiIteration = [=](const arma::sp_cx_mat&L, const arma::cx_vec& rho, int m_max) {
    //        const int N = rho.n_rows;
    //        
    //        ArnoldiResult result;
    //        result.V.set_size(m_max);
    //        result.Hessian = arma::sp_cx_mat(m_max,m_max);
    //        
    //        result.Beta = arma::norm(rho,2);
    //        arma::cx_vec rhoi = rho / result.Beta;
    //        result.V(0) = rhoi;
//
    //        arma::cx_vec AV(N);
    //        std::complex<double> h_next = 0.0;
    //        int m = m_max;
    //        for(int j = 0; j < m_max; j++)
    //        {
    //            AV = L * result.V(j);
    //            for(int i = 0; i <= j; i++)
    //            {
    //                std::complex<double> h_ij = arma::cdot(result.V(i), AV);
    //                result.Hessian(i,j) = h_ij;
    //                arma::cx_vec temp = h_ij * result.V(i);
    //                AV -= temp;
    //            }
//
    //            h_next = arma::norm(AV,2);
//
    //            if(j == m_max - 1) {
    //                result.h_res = h_next;
    //                break;
    //            }
//
    //            if(std::abs(h_next) < 1e-14)
    //            {
    //                result.h_res = std::complex<double>(0.0, 0.0);
    //                m = j + 1;
    //                result.Hessian = result.Hessian.submat(0,0,m-1,m-1);
    //                result.V.set_size(m);
    //                break;
    //            }
//
    //            result.V(j+1) = AV / h_next;
    //            result.Hessian(j+1,j) = h_next;
    //        }
//
    //        return result;
    //    };
//
    //    struct StepReturnStruct
    //    {
    //        arma::cx_vec rho_new;
    //        double err;
    //    };
//
    //    auto MagnusExpansion2ndOrder = [&](const arma::sp_cx_mat& L, double h) {
    //        arma::sp_cx_mat At = GetTDH(PropParams.CurrentTime, L);
    //        arma::sp_cx_mat Atpto2 = GetTDH(PropParams.CurrentTime + h/2.0, L);
//
    //        arma::sp_cx_mat Omega = (h/2.0) * (Atpto2 * At - At * Atpto2);
    //        return Omega;
    //    };
//
    //    auto step = [&](const arma::sp_cx_mat& L, const arma::cx_vec& rho, double h, int m_krylov)  {
//
    //        ArnoldiResult ar;
    //        if(GetTDH != nullptr)
    //        {
    //            ar = ArnoldiIteration(MagnusExpansion2ndOrder(L,h), rho, m_krylov);
    //        }
    //        else
    //        {
    //            ar = ArnoldiIteration(L, rho, m_krylov);
    //        }
    //        int m = ar.Hessian.n_rows;
    //        arma::sp_cx_mat Hm = h * ar.Hessian;
    //        
    //        arma::cx_mat Exponent = arma::expmat(arma::conv_to<arma::cx_mat>::from(Hm));
    //        arma::cx_vec e1(m,arma::fill::zeros);
    //        e1(0) = std::complex<double>(1.0, 0.0);
    //        arma::cx_vec w = Exponent * e1;
//
    //        arma::cx_vec rho_new(rho.n_rows, arma::fill::zeros);
    //        for(int j = 0; j < m; j++)
    //        {
    //            arma::cx_vec temp = ar.Beta * w(j) * ar.V(j);
    //            rho_new += temp;
    //        }
//
    //        std::complex<double> error_val = Exponent(m-1,0);
    //        double err = std::abs(ar.Beta * ar.h_res * error_val);
//
    //        StepReturnStruct return_struct;
    //        return_struct.rho_new = rho_new;
    //        return_struct.err = err;
//
    //        return return_struct;
    //    };
//
    //    bool keep_step = false;
    //    bool first_attempt = true;
    //    while(!keep_step)
    //    {
    //        auto KrylovStep = step(L,rho0,dumpstep,PropParams.max_krylov_iterations);
//
    //        double ynorm = arma::norm(KrylovStep.rho_new,2);
    //        double tol = PropParams.atol + PropParams.rtol * ynorm;
    //        double R = KrylovStep.err / tol;
//
    //        auto Adjusth = [&](double R, double safety, double f1, double f2, double h) {
    //            return h * std::min(f2, std::max(f1, safety * std::pow(R, -1.0/5.0)));
    //        };
//
    //        dumpstep = Adjusth(R, PropParams.safety, PropParams.f1, PropParams.f2, dumpstep);
    //        if(R <= PropParams.reject_limit)
    //        {
    //            drhodt = KrylovStep.rho_new;
    //            keep_step = true;
    //        }
//
    //        if(dumpstep < PropParams.min && R > PropParams.reject_limit)
    //        {
    //            PropParams.min = dumpstep;
    //        }
    //        if(dumpstep > PropParams.max)
    //        {
    //            dumpstep = PropParams.max;
    //        }
    //        if(R >= PropParams.reject_limit)
    //        {
    //            first_attempt = false;
    //        }
    //    }
//
    //    return {dumpstep, first_attempt};
//
    //}

    unsigned int GetNumThreads()
    {
#ifdef _OPENMP
        int omp_threads = omp_get_max_threads();
        if (omp_threads > 0)
        {
            return static_cast<unsigned int>(omp_threads);
        }
#endif
        auto processor_count = std::thread::hardware_concurrency();
        return (processor_count == 0) ? 1U : processor_count;
    }

    arma::cx_vec ThomasBlockSolver(arma::sp_cx_mat &A, arma::cx_vec &b, int block_size, std::vector<arma::sp_cx_mat>CachedBlocks)
    {
        int n_blocks = A.n_rows / block_size; //the total number of blocks in the matrix (including those that are zero)
        std::vector<arma::sp_cx_mat> A_blocks; 
        std::vector<arma::cx_vec> B_blocks;
        bool Cached = false;
        if(CachedBlocks.size() != 0)
            Cached = true;
        else
        {
            if(!IsBlockTridiagonal(A,block_size))
            {
                return arma::cx_vec();
            }
        }

        //number of blocks needed
        //A is tridigonal, so we only need to store the blocks on the diagonal and the blocks above and below it
        int TridiagonalBlocks = (n_blocks) * 3; //3 blocks for the middle rows and 4 to account for the first and last rows (e.g 2x2 - 4 blocks, 3x3 - 7 blocks, 4x4 - 10 blocks, etc...)
        A_blocks.reserve(TridiagonalBlocks);
        B_blocks.reserve(n_blocks);

        arma::sp_cx_mat ZeroBlock(block_size,block_size);

        //Get A and B blocks
        int count = 0;
        for (int i = 0; i < n_blocks; i++)
        {
            arma::sp_cx_mat L,D,U = arma::sp_cx_mat(block_size,block_size);
            //B block
            arma::cx_vec B_subblock = b.rows(i * block_size, (i + 1) * block_size - 1);
            B_blocks.insert(B_blocks.begin() + i, B_subblock);

            if(Cached)
                continue;

            if(i==0)
            {
                L = ZeroBlock;
            }
            else
            {
                L = A.submat(i * block_size, (i-1) * block_size, (i+1) * block_size - 1, i * block_size - 1);
            }

            D = A.submat(i * block_size, i * block_size, (i+1) * block_size - 1, (i+1) * block_size - 1);

            if(i < n_blocks -1)
            {
                U = A.submat(i * block_size, (i+1) * block_size, (i+1) * block_size - 1, (i+2) * block_size - 1);
            }
            else
            {
                U = ZeroBlock;
            }

            A_blocks.insert(A_blocks.begin() + count + 0, L);
            A_blocks.insert(A_blocks.begin() + count + 1, D);
            A_blocks.insert(A_blocks.begin() + count + 2, U);
            count = count + 3;
        }

        //O(n) method so can loop through with a range of n_blocks
        /*
        |D_1 U_1 0   0                  ... 0 | |x_1|     |b_1|
        |L_2 D_2 U_2 0                  ... 0 | |x_2|     |b_2|
        |0  L_3 D_3 U_3                 ... 0 | |x_3|     |b_3|
        |...            ...             ...   |  ...       ... 
        |0            L_n-2 D_n_2 U_n-2   0   | |x_n-2|   |b_n-2|
        |0        ...    0  L_n-1 D_n-1 U_n-1 | |x_n-1|   |b_n-1|
        |0        ...         0   L_n   D_n   | |x_n|     |b_n|
        */
        if(Cached)
            A_blocks = CachedBlocks;

        for (int i = 1; i < n_blocks; i++)
        {
            int PrevIndex = 3*(i-1);
            int CurrIndex = 3*i;
            //Get the blocks
            arma::sp_cx_mat D_prev = A_blocks[PrevIndex+1];
            arma::sp_cx_mat U_prev = A_blocks[PrevIndex+2];
            arma::cx_vec B_prev = B_blocks[i-1];

            arma::sp_cx_mat D = A_blocks[CurrIndex+1];
            arma::cx_mat L = arma::cx_mat(A_blocks[CurrIndex]);
            arma::cx_vec B = B_blocks[i];

            //form augmented matrix (U_prev | B_prev)
            arma::cx_mat UB_prev = AugmentedMatrix(arma::conv_to<arma::cx_mat>::from(U_prev), B_prev);
            arma::cx_mat UB_prev_modified = arma::solve(arma::cx_mat(D_prev), UB_prev);
            
            //A_blocks[(3*(i-1)) + 1] = arma::sp_cx_mat(U_old);
            //B_blocks[i-1] = B_old;

            arma::cx_mat DB = AugmentedMatrix(arma::conv_to<arma::cx_mat>::from(D), B);
            arma::cx_mat RHS = L * UB_prev_modified;
            DB = DB - RHS;
            //Update D and B blocks
            auto [D_new, B_new] = UndoAugmentedMatrix(DB);
            A_blocks[CurrIndex+1] = arma::sp_cx_mat(D_new);
            B_blocks[i] = B_new;

        }

        
        //Back substitution
        std::vector<arma::cx_vec> X_blocks; //Solution blocks - this is reversed 


        for (int i = n_blocks-1; i >= 0; i--)
        {
            int index = 3*i;
            arma::cx_mat D_curr = arma::cx_mat(A_blocks[index+1]);
            arma::cx_vec B_curr = B_blocks[i];
            
            if (i == (n_blocks-1))
            {
                arma::cx_vec X_last = arma::solve(D_curr, B_curr);
                X_blocks.insert(X_blocks.begin(), X_last);
            }
            else
            {
                arma::sp_cx_mat U_curr = A_blocks[index + 2];
                arma::cx_vec X_next = X_blocks[0];
                arma::cx_vec LHS = B_curr - U_curr * X_next;
                arma::cx_vec X_last = arma::solve(D_curr, LHS);
                X_blocks.insert(X_blocks.begin(), X_last);
            }
        }

        //Reconstruct solution vector
        arma::cx_vec x(arma::size(b), arma::fill::zeros);
        for (int i = 0; i < n_blocks; i++)
        {
            x.rows(i * block_size, (i +1) * block_size -1) = X_blocks[i];
        }

        return x;
        
    }

    bool BlockSolver(arma::sp_cx_mat &A, arma::cx_vec &b, std::vector<int> block_sizes, arma::cx_vec &x)
    {
        bool inverted = false;
        //if(IsBlockTridiagonal(A,block_sizes[0]))
        //{
        //    x = ThomasBlockSolver(A, b, block_sizes[0]);
        //    return true;
        //}

        arma::cx_mat A_inv = BlockMatrixInverse(A, block_sizes, inverted);
        if(!inverted)
        {
            x = arma::cx_vec(arma::size(b), arma::fill::zeros);
            return false;
        }
        x = A_inv * b;
        return true;
    }

    //DONT USE THESE FUNCTIONS THEY ARE SLOW
    arma::cx_mat BlockMatrixInverse(arma::sp_cx_mat &A, std::vector<int> block_sizes, bool &Invertible)
    {
        //Matrix Partitions
        arma::sp_cx_mat A11, A12, A21, A22;
        //A11 is square
        //A12 is rectangular - wide
        //A21 is rectangular - tall
        //A22 is square
        auto sum = [](std::vector<int> v, int start, int perodicity) {
            int s = 0;
            for(int i = start; i < (int)v.size(); i += perodicity)
            {
                s= s + v[i];            
            }
            return s;
        };
        std::pair<int, int> A11_size = {block_sizes[0], block_sizes[0]};
        std::pair<int, int> A12_size = {block_sizes[0], sum(block_sizes,1,1)};
        std::pair<int, int> A21_size = {sum(block_sizes,1,1), block_sizes[0]};
        std::pair<int, int> A22_size = {sum(block_sizes,1,1), sum(block_sizes,1,1)};
        A11 = A.submat(0, 0, A11_size.first -1, A11_size.second -1);
        A12 = A.submat(0, A11_size.second, A12_size.first -1, A.n_cols -1);
        A21 = A.submat(A11_size.first, 0, A.n_rows -1, A21_size.second -1);
        A22 = A.submat(A11_size.first, A11_size.second, A.n_rows -1, A.n_cols -1);

        //Check if A11 and A22 are invertible
        arma::cx_mat A11_inv, A22_inv; //The inverse of a sparse matrix is usually dense, so we use a dense matrix here
        //Check invertibility wihtin a scope, that way if not invertable we don't keep the failed inverse
        bool A11_invertible, A22_invertible;
        bool SComplementA, SComplementB, BComplements;
        {
            //A11_invertible = arma::inv(A11_inv, arma::cx_mat(A11));
            arma::cx_mat id(A11.n_rows,A11.n_cols,arma::fill::eye);
            A11_invertible = arma::solve(A11_inv,arma::cx_mat(A11),id);
            //A22_invertible = arma::inv(A22_inv, arma::cx_mat(A22));
            if (block_sizes.size() > 2)
            {
                bool Invertible2;
                auto block_sizes_sub = std::vector<int>(block_sizes.begin() +1, block_sizes.end());
                A22_inv = BlockMatrixInverse(A22, block_sizes_sub, Invertible2);
                A22_invertible = Invertible2;
            }
            else
            {
                arma::cx_mat id(A22.n_rows,A22.n_cols,arma::fill::eye);
                A22_invertible = arma::solve(A22_inv,arma::cx_mat(A22),id);
            }
            //std::cout << A22 << std::endl;

            if(!A11_invertible)
            {
                A11_inv = arma::cx_mat(); 
            }
            if(!A22_invertible)
            {
                A22_inv = arma::cx_mat(); 
            }
        }

        SComplementA = A11_invertible;// && !A22_invertible;
        SComplementB = A22_invertible;
        BComplements = A11_invertible && A22_invertible;
        Invertible = true;

        if(SComplementA)
        {
            return SchurComplementA(A11_inv, A12, A21, A22, Invertible);
        }
        else if(SComplementB)
        {
            return SchurComplementB(A11, A12, A21, A22_inv, Invertible);
        }
        else if(BComplements)
        {
            return BothSchurComponents(A11, A11_inv, A12, A21, A22, A22_inv, Invertible);
        }
        else
        {
            Invertible = false;
            return arma::cx_mat();
        }

    }

    arma::cx_mat SchurComplementA(arma::cx_mat &A11_inv, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::sp_cx_mat &A22, bool &Invertible)
    {
        arma::cx_mat S = A22 - A21 * A11_inv * A12;
        arma::cx_mat S_inv = arma::cx_mat(S.n_rows, S.n_cols);
        arma::cx_mat id(S.n_rows,S.n_cols,arma::fill::eye);
        bool S_invertible = arma::solve(S_inv,S,id);
        if(!S_invertible)
        {
            Invertible = false;
            return arma::cx_mat();
        }
        //Construct the inverse matrix using the Schur complement
        arma::cx_mat P11 = A11_inv + A11_inv * A12 * S_inv * A21 * A11_inv;
        arma::cx_mat P12 = -1 * A11_inv * A12 * S_inv;
        arma::cx_mat P21 = -1 * S_inv * A21 * A11_inv;
        arma::cx_mat P22 = S_inv;

        arma::cx_mat Inv = arma::cx_mat(P11.n_rows + P21.n_rows, P11.n_cols + P12.n_cols);
        Inv.submat(0, 0, P11.n_rows -1, P11.n_cols -1) = P11;
        Inv.submat(0, P11.n_cols, P12.n_rows -1, Inv.n_cols -1) = P12;
        Inv.submat(P11.n_rows, 0, Inv.n_rows -1, P21.n_cols -1) = P21;
        Inv.submat(P11.n_rows, P11.n_cols, Inv.n_rows -1, Inv.n_cols -1) = P22;

        Invertible = true;
        return Inv;
    }

    arma::cx_mat SchurComplementB(arma::sp_cx_mat &A11, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::cx_mat &A22_inv, bool &invertible)
    {
        arma::cx_mat S = A11 - A12 * A22_inv * A21;
        arma::cx_mat S_inv = arma::cx_mat(S.n_rows, S.n_cols);
        arma::cx_mat id(S.n_rows,S.n_cols,arma::fill::eye);
        bool S_invertible  = arma::solve(S_inv,S,id);
        if(!S_invertible)
        {
            invertible = false;
            return arma::cx_mat();
        }
        //Construct the inverse matrix using the Schur complement
        arma::cx_mat P11 = S_inv;
        arma::cx_mat P12 = -1 * S_inv * A12 * A22_inv;
        arma::cx_mat P21 = -1 * A22_inv * A21 * S_inv;
        arma::cx_mat P22 = A22_inv + A22_inv * A21 * S_inv * A12 * A22_inv;

        arma::cx_mat Inv = arma::cx_mat(P11.n_rows + P21.n_rows, P11.n_cols + P12.n_cols);
        Inv.submat(0, 0, P11.n_rows -1, P11.n_cols -1) = P11;
        Inv.submat(0, P11.n_cols, P12.n_rows -1, Inv.n_cols -1) = P12;
        Inv.submat(P11.n_rows, 0, Inv.n_rows -1, P21.n_cols -1) = P21;
        Inv.submat(P11.n_rows, P11.n_cols, Inv.n_rows -1, Inv.n_cols -1) = P22;

        invertible = true;
        return Inv;
    }

    arma::cx_mat BothSchurComponents(arma::sp_cx_mat&A11, arma::cx_mat &A11_inv, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::sp_cx_mat &A22, arma::cx_mat &A22_inv, bool &invertible)
    {
        arma::cx_mat S1 = A11 - A12 * A22_inv * A21;
        arma::cx_mat S2 = A22 - A21 * A11_inv * A12;
        arma::cx_mat S1_inv, S2_inv;
        arma::cx_mat id1(S1.n_rows,S1.n_cols,arma::fill::eye);
        arma::cx_mat id2(S2.n_rows,S2.n_cols,arma::fill::eye);
        bool S1_invertible  = arma::solve(S1_inv,S1,id1);
        bool S2_invertible  = arma::solve(S2_inv,S2,id2);
        if(!S1_invertible || !S2_invertible)
        {
            invertible = false;
            return arma::cx_mat();
        }
        //Construct the inverse matrix using the Schur complement
        arma::cx_mat P11 = S1_inv;
        arma::cx_mat P12 = -1 * S1_inv * A12 * A22_inv;
        arma::cx_mat P21 = -1 * S2_inv * A21 * A11_inv;
        arma::cx_mat P22 = S2_inv;

        arma::cx_mat Inv = arma::cx_mat(P11.n_rows + P21.n_rows, P11.n_cols + P12.n_cols);
        Inv.submat(0, 0, P11.n_rows -1, P11.n_cols -1) = P11;
        Inv.submat(0, P11.n_cols, P12.n_rows -1, Inv.n_cols -1) = P12;
        Inv.submat(P11.n_rows, 0, Inv.n_rows -1, P21.n_cols -1) = P21;
        Inv.submat(P11.n_rows, P11.n_cols, Inv.n_rows -1, Inv.n_cols -1) = P22;
        
        invertible = true;
        return Inv;
    }

    arma::sp_cx_mat AugmentedMatrix(arma::sp_cx_mat Mat, arma::cx_vec b)
    {
        int rows = Mat.n_rows;
        int cols = Mat.n_cols;

        arma::sp_cx_mat AugMat(rows, cols+1);
        AugMat.submat(0, 0, rows-1, cols-1) = Mat;
        AugMat.submat(0, cols, rows-1, cols) = b;
        return AugMat;
    }

    arma::cx_mat AugmentedMatrix(arma::cx_mat Mat, arma::cx_vec b)
    {
        int rows = Mat.n_rows;
        int cols = Mat.n_cols;

        arma::sp_cx_mat AugMat(rows, cols+1);
        //std::cout << arma::sp_cx_mat(Mat) << std::endl;
        AugMat.submat(0, 0, rows-1, cols-1) = Mat;
        //std::cout << AugMat << std::endl;
        //std::cout << arma::sp_cx_mat(b) << std::endl;
        AugMat.submat(0, cols, rows-1, cols) = b;
        //std::cout << AugMat << std::endl;
        return arma::cx_mat(AugMat);
    }

    std::pair<arma::cx_mat, arma::cx_vec> UndoAugmentedMatrix(arma::cx_mat AugMat)
    {
        int rows = AugMat.n_rows;
        int cols = AugMat.n_cols;

        arma::cx_mat Mat = AugMat.submat(0, 0, rows-1, cols-2);
        arma::cx_vec b = AugMat.submat(0, cols-1, rows-1, cols-1);
        return std::make_pair(Mat, b);
    }

    bool IsBlockTridiagonal(arma::sp_cx_mat &A, int block_size)
    {
        int n_blocks = A.n_rows / block_size;
        if (n_blocks == 2)
        {
            return true;   
        }

        for (int row = 0; row < n_blocks; row++)
        {
            for (int col = 0; col < n_blocks; col++)
            {
                if(std::abs(row-col) > 1)
                {
                    arma::sp_cx_mat block = A.submat(row*block_size, col*block_size, (row+1)*block_size -1, (col+1)*block_size -1);
                    int non_zero = block.n_nonzero; //should be very efficient for sparse matrices
                    if(non_zero > 0)
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    arma::cx_vec BiCGSTAB(arma::sp_cx_mat &A, arma::cx_vec &b, PreconditionerType preconditoner ,arma::sp_cx_mat K, double tol, int max_iter, int max_preconditoner_iter)
    {

        if(preconditoner == PreconditionerType::None)
        {
            K = arma::sp_cx_mat(arma::size(A));
            K.eye(); // No preconditioner, use identity matrix
        }
        else if(preconditoner == PreconditionerType::IncompleteBiCGSTAB)
        {
            if (max_preconditoner_iter < 0)
            {
                max_preconditoner_iter = 5; 
            }
            K = IncompleteBiCGSTAB(A, max_preconditoner_iter);
        }
        else if(preconditoner == PreconditionerType::SPAI)
        {
            if (max_preconditoner_iter < 0)
            {
                max_preconditoner_iter = 50; 
            }
            
            K = SPAI(A, max_preconditoner_iter);
        }
        else if(preconditoner == PreconditionerType::JACOBI)
        {
            K = JACOBI(A);
        }


        arma::cx_mat K_1, K_2;
        //auto P = LUDecomposition(K);
        arma::cx_mat DenseK = arma::conv_to<arma::cx_mat>::from(K);
        arma::lu(K_1, K_2, DenseK);
        //arma::lu(K_1, K_2, K);
        //arma::lu()

        arma::cx_vec x = arma::cx_vec(arma::size(b), arma::fill::zeros);
        arma::cx_vec r_naught = b - A * x;
        arma::cx_vec r_naught_hat = r_naught;
        arma::cx_vec r = r_naught;
        arma::cx_vec rho_naught = r_naught;
        arma::cx_vec rho_prev = rho_naught;
        arma::cx_double rho_k_1 = arma::dot(rho_naught, rho_naught);

        for(int k = 1; k <= max_iter; k++)
        {
            //arma::cx_vec y = LUSolve(K,P, rho_prev); //too slow
            arma::cx_vec y = arma::cx_vec(arma::size(rho_prev), arma::fill::zeros);
            arma::solve(y,DenseK, rho_prev);
            arma::cx_vec v = A * y;
            arma::cx_double alpha = rho_k_1 / arma::dot(r_naught_hat, v);
            arma::cx_vec h = x + alpha * y;
            arma::cx_vec s = r - alpha * v;
            double norms = arma::norm(s);
            if(norms < tol)
            {
                std::cout << "Converged in " << k << " iterations." << std::endl;
                return x;
            }
            //arma::cx_vec z = LUSolve(K,P,s);
            arma::cx_vec z = arma::cx_vec(arma::size(s), arma::fill::zeros);
            arma::solve(z, DenseK, s);
            arma::cx_vec t = A * z;
            arma::cx_vec K_1_invt = arma::cx_vec(arma::size(t), arma::fill::zeros);
            arma::solve(K_1_invt, K_1, t);
            arma::cx_vec K_1_invs = arma::cx_vec(arma::size(s), arma::fill::zeros);
            arma::solve(K_1_invs, K_1, s);
            arma::cx_double omega = arma::dot(K_1_invt, K_1_invs) / arma::dot(K_1_invt, K_1_invt);
            x = h + omega * z;
            r = s - omega * t;
            double normr = arma::norm(r);
            if(normr < tol)
            {
                std::cout << "Converged in " << k << " iterations." << std::endl;
                return x;
            }
            arma::cx_double rho_k = arma::dot(r_naught_hat, r);
            arma::cx_double beta = (rho_k / rho_k_1) * (alpha / omega);
            rho_prev = r + beta * (rho_prev - omega * v);
            rho_k_1 = rho_k;
        }
        std::cout << "Did not converge in " << max_iter << " iterations." << std::endl;
        return x; // Return the last computed x, even if it did not converge
    }

    arma::sp_cx_mat IncompleteBiCGSTAB(arma::sp_cx_mat &A, int max_iter)
    {
        int n = A.n_rows;
        arma::sp_cx_mat I = arma::sp_cx_mat(arma::size(A));
        I.eye();
        arma::sp_cx_mat x = arma::sp_cx_mat(arma::size(A));
        for(int i = 0; i < n; i++)
        {
            arma::cx_vec col = arma::cx_vec(n, arma::fill::zeros);
            for (int j = 0; j < n; j++)
            {
                col(j) = A(i,j);
            }
            arma::cx_vec b = BiCGSTAB(A, col, PreconditionerType::None, arma::sp_cx_mat(), max_iter = max_iter);
            {
                x.col(i) = b;
            }
        }
        return x;
    }

    arma::sp_cx_mat SPAI(arma::sp_cx_mat &A, int max_iter)
    {
        arma::sp_cx_mat I = arma::sp_cx_mat(arma::size(A));
        I.eye();
        arma::cx_double alpha = 2.0 / arma::norm(A * arma::trans(A), 1);
        arma::sp_cx_mat M = alpha * A;
        for(int i = 0; i < max_iter; i++)
        {
            arma::sp_cx_mat C = A * M;
            arma::sp_cx_mat G = I - C;
            arma::sp_cx_mat AG = A * G;
            arma::cx_double trace = arma::trace(arma::trans(G) * AG);
            arma::cx_double norm = arma::norm(AG, 1);
            alpha = trace / std::pow(norm,2);
            M = M + alpha * G;
        }
        return M;
    }

    arma::sp_cx_mat JACOBI(arma::sp_cx_mat &A)
    {
        arma::sp_cx_mat K = arma::sp_cx_mat(arma::size(A));
        K = A.diag();
        return K;
    }

    std::vector<int> LUDecomposition(arma::sp_cx_mat &A)
    {
        arma::sp_cx_mat L, U;
        int n = A.n_rows;
        std::vector<int> permuation;
        for(int i = 0; i <= n; i++)
        {
            permuation.push_back(i);
        }

        for (int i = 0; i < n; i++)
        {
            double max_val = 0.0;
            int max_index = i;

            for (int k = i; k < n; k++)
            {
                std::complex<double> ki = A(k,i);
                double val = std::abs(ki);
                if (val > max_val)
                {
                    max_val = val;
                    max_index = k;
                }
            }

            if (max_index != i)
            {
                int j = permuation[i];
                permuation[i] = permuation[max_index];
                permuation[max_index] = j;
                A.swap_rows(i, max_index);
                permuation[n] = permuation[n] + 1; // Increment the permutation count
            }

            for (int j = i + 1; j < n; j++)
            {
                std::complex<double> ii,ji,ik,jk;
                ii = A(i,i);
                ji = A(j,i);
                ik = A(i,j);
                if (ii == std::complex<double>(0, 0))
                {
                    throw std::runtime_error("Matrix is singular, cannot perform LU decomposition.");
                }
                A(j,i) = ji / ii;
                for (int k = i + 1; k < n; k++)
                {
                    jk = A(j,k);
                    A(j,k) = jk - (ji * ik);
                }
            }
        }
        return permuation;
    }

    arma::cx_vec LUSolve(arma::sp_cx_mat &K, std::vector<int> &P, arma::cx_vec &b)
    {
        int n = K.n_rows;
        arma::cx_vec x = arma::cx_vec(n, arma::fill::zeros);
        for (int i = 0; i < n; i++)
        {
            x(i) = b(P[i]);
            for (int j = 0; j < i; j++)
            {
                std::complex<double> xi, xj,ij;
                xi = x(i);
                xj = x(j);
                ij = K(i,j);
                x(i) = xi - ij * xj;
            }
        }

        for (int i = n - 1; i >= 0; i--)
        {
            for(int j = i + 1; j < n; j++)
            {
                std::complex<double> xi, xj, ij;
                xi = x(i);
                xj = x(j);
                ij = K(i,j);
                x(i) = x(i) - ij * xj;
            }
            std::complex<double> xi, ii;
            xi = x(i);
            ii = K(i,i);
            x(i) = x(i) / ii;
        }
        return x;
    }
}
