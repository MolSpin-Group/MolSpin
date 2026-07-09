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
#include <omp.h>

namespace RunSection
{

    FibSpherePoint *CalculateFibPoints(int n)
    {
        FibSpherePoint* TempPointArray = (FibSpherePoint*)malloc(n * sizeof(FibSpherePoint));
        if(TempPointArray == NULL)
        {
            std::cout << "Memory not allocated" << std::endl;
            return nullptr;
        }

        double phi = arma::datum::pi * (3.0 - std::sqrt(5.0)); //Golden angle in radians
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
        std::uniform_real_distribution<double> distPhi(0, 2.0 * arma::datum::pi);
        std::uniform_real_distribution<double> distTheta(0, arma::datum::pi);
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

    double RungeKutta45Armadillo(arma::sp_cx_mat &L, arma::cx_vec &rho0, arma::cx_vec &drhodt, double dumpstep, RungeKuttaFuncArma func, std::pair<double, double> tolerance, double MinTimeStep, double MaxTimeStep, double time)
    {
        VecType k0(rho0.n_rows);

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

        auto [RK4, RK5] = RungeKutta45(L, rho0, dumpstep, func);

        double change = 0;
        {
            VecType diff = RK5 - RK4;
            double sum = 0;

#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < int(diff.n_rows); i++)
            {
                sum += std::pow(std::abs(diff[i]), 2);
            }

            change = std::sqrt(sum);
        }

        auto Adjusth = [](double tol, double ch)
        {
            double h4 = (tol / (2 * ch));
            return std::sqrt(std::sqrt(h4));
        };

        double NewStepSize = 0.0;
        if (change < tolerance.first && dumpstep < MaxTimeStep)
        {
            NewStepSize = dumpstep * Adjusth(tolerance.first, change);
            if (NewStepSize > MaxTimeStep)
            {
                NewStepSize = MaxTimeStep;
            }
        }
        else if (change > tolerance.second && dumpstep > MinTimeStep)
        {
            NewStepSize = dumpstep * Adjusth(tolerance.second, change);
            if (NewStepSize < MinTimeStep)
            {
                NewStepSize = MinTimeStep;
            }
        }
        else
        {
            NewStepSize = dumpstep;
        }

        drhodt = RK4;
        return NewStepSize;
    }

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

    bool BlockSolver(arma::sp_cx_mat &A, arma::cx_vec &b, int block_size, arma::cx_vec &x)
    {
        bool inverted = false;
        if(IsBlockTridiagonal(A,block_size))
        {
            x = ThomasBlockSolver(A, b, block_size);
            return true;
        }

        arma::cx_mat A_inv = BlockMatrixInverse(A, block_size, inverted);
        if(!inverted)
        {
            x = arma::cx_vec(arma::size(b), arma::fill::zeros);
            return false;
        }
        x = A_inv * b;
        return true;
    }

    //DONT USE THESE FUNCTIONS THEY ARE SLOW
    arma::cx_mat BlockMatrixInverse(arma::sp_cx_mat &A, int block_size, bool &Invertible)
    {
        //Matrix Partitions
        arma::sp_cx_mat A11, A12, A21, A22;
        A11 = A.submat(0, 0, block_size -1, block_size -1);
        A12 = A.submat(0, block_size, block_size -1, A.n_cols -1);
        A21 = A.submat(block_size, 0, A.n_rows-1, block_size -1);
        A22 = A.submat(block_size, block_size, A.n_rows -1, A.n_cols -1);

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
            if (A22.n_rows > (unsigned int)block_size)
            {
                bool Invertible2;
                A22_inv = BlockMatrixInverse(A22, block_size, Invertible2);
                A22_invertible = Invertible2;
            }
            else
            {
                arma::cx_mat id(A11.n_rows,A11.n_cols,arma::fill::eye);
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
        arma::cx_mat S_inv;
        arma::cx_mat id(A12.n_rows,A12.n_cols,arma::fill::eye);
        bool S_invertible  = arma::solve(S_inv,S,id);
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

        arma::cx_mat Inv = arma::cx_mat(A11_inv.n_rows * 2, A11_inv.n_cols * 2);
        Inv.submat(0, 0, A11_inv.n_rows -1, A11_inv.n_cols -1) = P11;
        Inv.submat(0, A11_inv.n_cols, A11_inv.n_rows -1, Inv.n_cols -1) = P12;
        Inv.submat(A11_inv.n_rows, 0, Inv.n_rows -1, A11_inv.n_cols -1) = P21;
        Inv.submat(A11_inv.n_rows, A11_inv.n_cols, Inv.n_rows -1, Inv.n_cols -1) = P22;

        Invertible = true;
        return Inv;
    }

    arma::cx_mat SchurComplementB(arma::sp_cx_mat &A11, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::cx_mat &A22_inv, bool &invertible)
    {
        arma::cx_mat S = A11 - A12 * A22_inv * A21;
        arma::cx_mat id(A11.n_rows,A11.n_cols,arma::fill::eye);
        arma::cx_mat S_inv;
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

        arma::cx_mat Inv = arma::cx_mat(A11.n_rows * 2, A11.n_cols * 2);
        Inv.submat(0, 0, A11.n_rows -1, A11.n_cols -1) = P11;
        Inv.submat(0, A11.n_cols, A11.n_rows -1, Inv.n_cols -1) = P12;
        Inv.submat(A11.n_rows, 0, Inv.n_rows -1, A11.n_cols -1) = P21;
        Inv.submat(A11.n_rows, A11.n_cols, Inv.n_rows -1, Inv.n_cols -1) = P22;

        invertible = true;
        return Inv;
    }

    arma::cx_mat BothSchurComponents(arma::sp_cx_mat&A11, arma::cx_mat &A11_inv, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::sp_cx_mat &A22, arma::cx_mat &A22_inv, bool &invertible)
    {
        arma::cx_mat S1 = A11 - A12 * A22_inv * A21;
        arma::cx_mat S2 = A22 - A21 * A11_inv * A12;
        arma::cx_mat S1_inv, S2_inv;
        arma::cx_mat id(A11.n_rows,A11.n_cols,arma::fill::eye);
        bool S1_invertible  = arma::solve(S1_inv,S1,id);
        bool S2_invertible  = arma::solve(S2_inv,S2,id);
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

        arma::cx_mat Inv = arma::cx_mat(A11.n_rows * 2, A11.n_cols * 2);
        Inv.submat(0, 0, A11.n_rows -1, A11.n_cols -1) = P11;
        Inv.submat(0, A11.n_cols, A11.n_rows -1, Inv.n_cols -1) = P12;
        Inv.submat(A11.n_rows, 0, Inv.n_rows -1, A11.n_cols -1) = P21;
        Inv.submat(A11.n_rows, A11.n_cols, Inv.n_rows -1, Inv.n_cols -1) = P22;
        
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
