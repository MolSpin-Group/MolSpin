/////////////////////////////////////////////////////////////////////////
// Utility (RunSection module)
// ------------------
// Utility functions
// functions for multiple tasks
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_Utility
#define MOD_RunSection_Utility

// #include <Eigen/Sparse>
// #include <Eigen/Core>
#include "SpinAPIfwd.h"
#include <utility>
#include <array>

namespace RunSection
{
// region FibSphere
    typedef std::pair<float, float> FibSpherePoint;
    FibSpherePoint* CalculateFibPoints(int n);
    bool RetrievePoint (std::array<double, 3> &arr, FibSpherePoint* ptr, int num);
// endregion FibSphere

// region MonteCarloSphere
    
    struct MCSpherePoint
    {
        double theta;
        double phi;
        std::array<double, 3> r;

        bool operator==(const MCSpherePoint &other) const
        {
            return (theta == other.theta) && (phi == other.phi) && (r[0] == other.r[0]) && (r[1] == other.r[1]) && (r[2] == other.r[2]);
        }
    };
    ///Function to uniformly generate points within a elipsoid
    /// @param n number of points to sample
    /// @param rmax_x maximum radius in the x direction
    /// @param rmax_y maximum radius in the y direction
    /// @param rmax_z maximum radius in the z direction
    MCSpherePoint* CalculateMCSpherePoints(int n, double rmax_x, double rmax_y, double rmax_z);
    ///Overload of the full CalculateMCSpherePoints function for sphere
    /// @param n number of points
    /// @param rmax_n maxiumum radius
    MCSpherePoint* CalculateMCSpherePoints(int n, double rmax);
    bool RetrieveMCPoint (std::array<double, 3> &arr, MCSpherePoint* ptr, int num);
// endregion MonteCarloSphere

// region SemiClassical
    struct SCData
    {
        arma::sp_cx_mat H; //Hamiltonian
        arma::sp_cx_mat SamplesMatrix; 
        
        std::vector<int> samples;
        int BlockSize; 
    };
    SCData GetHamiltonian(arma::sp_cx_mat&, int); //rename function so not to confuse
    ///Function to gernerate the sampled hamiltonian
    /// @param H0: the base hamiltonian without the sampled semi classical operator
    /// @param S: number of spin systems
    /// @param HSC: the collection of sampled semiclassical interactions 
    /// @param config: which samples to use, from each spin system
    arma::sp_cx_mat GetHamiltonian(const arma::sp_cx_mat, int, const std::vector<arma::sp_cx_mat>, std::vector<std::vector<int>>);

    typedef std::vector<std::vector<int>> SampleCombination;
    std::vector<SampleCombination> GenerateCombinationsNI(const std::vector<std::vector<int>>&, int startpoint = 0, int endpont = 0); //non independent spin systems 
    //std::vector<std::vector<int>> GenerateCombinationsI(const std::vector<std::vector<int>>&); //independent spin systems

// endregion SemiClassical
// region TimeEvo
    typedef arma::cx_vec (*RungeKuttaFuncArma)(double t, arma::sp_cx_mat &, arma::cx_vec &, arma::cx_vec);
    
    /// Runge-Kutta-Fehlberg method (4th and 5th order) with adaptive time step control
    ///     @param L: Liouvillian superoperator (sparse complex matrix)
    ///     @param rho0: Initial density matrix (complex vector)
    ///     @param drhodt: Time derivative of the density matrix (complex vector)
    ///     @param dumpstep: Initial Time step (double)
    ///     @param func: Right hand side of the master equation (function pointer - see RungeKuttaFuncArma)
    ///     @param tolerance: Pair of tolerances for adaptive step size control (pair of doubles)
    ///     @param MinTimeStep: Minimum allowed time step (double) - Optional, default = 1e-6
    ///     @param MaxTimeStep: Maximum allowed time step (double) - Optional, default = 1e6
    ///     @param time: Current time (double) - Optional, default = 0
    ///     @return New time step (double)
    double RungeKutta45Armadillo(arma::sp_cx_mat &, arma::cx_vec &, arma::cx_vec &, double, RungeKuttaFuncArma, std::pair<double, double>, double MinTimeStep = 1e-6, double MaxTimeStep = 1e6, double time = 0);

// endregion TimeEvo 
    unsigned int GetNumThreads();

// region BlockMatrixInversionSolvers
    //With these solvers there is the potential for a large amount of matrix fill-in during the solution process.
    //Block thomas has less fill in for larger systems with a block tridiagonal structure, than a general block solver.
    
    /// The thomas algorithm for solving Ax = b where A is a block tridiagonal matrix
    /// This algorithm assumes that A is made up of square blocks of size block_size x block_size
    /// And will use a conventional solver on the blocks 
    /// @param A The block tridiagonal matrix
    /// @param b The right hand side vector
    /// @param block_size The size of the blocks in the block tridiagonal matrix
    /// @return The solution vector x
    arma::cx_vec ThomasBlockSolver(arma::sp_cx_mat &A, arma::cx_vec &b, int block_size, std::vector<arma::sp_cx_mat>CachedBlocks = {});

    /// If the matrix is not tridiaognal a traditional block solver can be used.
    /// This function checks if the matrix is block tridiagonal and if so uses the thomas algorithm.
    /// A block solver for solving Ax = b where A is a block matrix.
    /// This algorithm assumes that A is made up of square blocks of size block_size x block
    /// and will recursively call itself on the blocks getting the blocks down to a smaller size before using a conventional solver.
    /// @param A The block matrix
    /// @param b The right hand side vector
    /// @param block_size The size of the blocks in the block matrix
    /// @return The solution vector x
    bool BlockSolver(arma::sp_cx_mat &A, arma::cx_vec &b, int block_size, arma::cx_vec &x); //TODO: work on the error handling

    //Internal functions used by the block solvers
    bool IsBlockTridiagonal(arma::sp_cx_mat &A, int block_size);
    arma::cx_mat BlockMatrixInverse(arma::sp_cx_mat &A, int block_size, bool &Invertible);
    arma::cx_mat SchurComplementA(arma::cx_mat &A11_inv, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::sp_cx_mat &A22, bool &invertible);
    arma::cx_mat SchurComplementB(arma::sp_cx_mat &A11, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::cx_mat &A22_inv, bool &invertible);
    arma::cx_mat BothSchurComponents(arma::sp_cx_mat&A11, arma::cx_mat &A11_inv, arma::sp_cx_mat &A12, arma::sp_cx_mat &A21, arma::sp_cx_mat &A22, arma::cx_mat &A22_inv, bool &invertible);
    arma::sp_cx_mat AugmentedMatrix(arma::sp_cx_mat Mat, arma::cx_vec b);
    arma::cx_mat AugmentedMatrix(arma::cx_mat Mat, arma::cx_vec b);
    std::pair<arma::cx_mat, arma::cx_vec> UndoAugmentedMatrix(arma::cx_mat AugMat);

// endregion BlockMatrixInversionSolvers

// region SparseMatrixSolvers
    //DONT USE THESE FUNCTIONS THEY ARE SLOW 
    //SparseMatrixSolvers
    //Preconditioned BiCGSTAB solver

    enum class PreconditionerType
    {
        None,
        IncompleteBiCGSTAB,
        SPAI,
        JACOBI,
        CUSTOM
    };

    arma::cx_vec BiCGSTAB(arma::sp_cx_mat &A, arma::cx_vec &b, PreconditionerType preconditioner, arma::sp_cx_mat K = arma::sp_cx_mat(), double tol = 1e-6, int max_iter = 1000, int max_preconditioner_iter = -1); //BiCGSTAB solver with preconditioner
    arma::sp_cx_mat IncompleteBiCGSTAB(arma::sp_cx_mat &A, int max_iter = 5); //Incomplete BiCGSTAB solver used to gererate a preconditioner
    arma::sp_cx_mat SPAI(arma::sp_cx_mat &A, int max_iter = 5); //Sparse Approximate Inverse preconditioner
    arma::sp_cx_mat JACOBI(arma::sp_cx_mat &A); //Jacobi preconditioner - leading  diagonal of A

    std::vector<int> LUDecomposition(arma::sp_cx_mat &K);
    arma::cx_vec LUSolve(arma::sp_cx_mat &K, std::vector<int> &P, arma::cx_vec &b); //LU decomposition and solve

// endregion SparseMatrixSolvers
}

#endif
