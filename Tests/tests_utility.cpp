//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Tests the Utility functions
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
// Add all the test cases
#include "Utility.h"
#include <random>
#include <vector>

bool test_utility_block_solver()
{
    const int bs = 3;
    const int nblk = 3;
    arma::cx_mat D0 = {{5.0,0.1,0.2},{0.0,6.0,0.3},{0.4,0.0,5.5}};
    arma::cx_mat U0 = {{0.1,0.0,0.0},{0.0,0.2,0.1},{0.0,0.0,0.05}};
    arma::cx_mat L1 = {{0.05,0.0,0.0},{0.0,0.1,0.0},{0.02,0.0,0.0}};
    arma::cx_mat D1 = {{4.5,0.05,0.0},{0.0,5.5,0.1},{0.0,0.0,6.0}};
    arma::cx_mat U1 = {{0.02,0.0,0.0},{0.0,0.03,0.0},{0.0,0.0,0.04}};
    arma::cx_mat L2 = {{0.01,0.0,0.0},{0.0,0.02,0.0},{0.0,0.0,0.03}};
    arma::cx_mat D2 = {{5.2,0.0,0.0},{0.0,4.8,0.1},{0.0,0.0,5.0}};

    std::vector<arma::cx_mat> blk(9);
    // row 0
    blk[0] = D0; blk[1] = U0; blk[2] = arma::zeros<arma::cx_mat>(bs,bs);
    // row 1
    blk[3] = L1; blk[4] = D1; blk[5] = U1;
    // row 2
    blk[6] = arma::zeros<arma::cx_mat>(bs,bs); blk[7] = L2; blk[8] = D2;

    arma::sp_cx_mat A(bs*nblk, bs*nblk);
    for (int bi=0; bi<nblk; ++bi) {
        for (int bj=0; bj<nblk; ++bj) {
            arma::cx_mat B = blk[bi*nblk + bj];
            if (arma::norm(B, "fro") == 0.0) continue;
            A.submat(bi*bs, bj*bs, (bi+1)*bs-1, (bj+1)*bs-1) = arma::sp_cx_mat(B);
        }
    }
    arma::vec temp = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0};
    arma::cx_vec b = arma::cx_vec(temp, arma::vec(temp));      

	

	arma::cx_vec result = arma::cx_vec(b.n_rows);
	arma::cx_vec result2 = arma::cx_vec(b.n_rows);
	result = RunSection::ThomasBlockSolver(A,b,3);
	result2 = arma::solve(arma::cx_mat(A), b);

	double diff = arma::norm(result - result2);
	if (diff < 1e-10)
		return true;
	return false;
}

bool test_utility_get_hamiltonian_without_semiclassical_samples()
{
	arma::sp_cx_mat hamiltonian(2, 2);
	hamiltonian(0, 0) = 1.0;
	hamiltonian(1, 1) = -1.0;

	RunSection::SCData data = RunSection::GetHamiltonian(hamiltonian, 2);
	return data.BlockSize == 2 &&
		   data.samples.empty() &&
		   data.SamplesMatrix.n_rows == 0 &&
		   data.SamplesMatrix.n_cols == 2 &&
		   arma::norm(data.H - hamiltonian, "fro") < 1e-14;
}

void AddUtiltiyTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("Utility test 1", test_utility_block_solver));
	_cases.push_back(test_case(
		"Utility GetHamiltonian without semiclassical samples",
		test_utility_get_hamiltonian_without_semiclassical_samples));
}
//////////////////////////////////////////////////////////////////////////////
