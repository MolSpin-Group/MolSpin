/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Optional exact Mz block diagonalization; no physical approximation.
#ifndef MOLSPIN_RESONANCEDIAGONALIZATION_H
#define MOLSPIN_RESONANCEDIAGONALIZATION_H
#include <armadillo>
#include <vector>
#include "SpinAPIfwd.h"

namespace RunSection::General::Resonance
{
    struct MzBlocks
    {
        std::vector<int> mz2;           // total Mz in units of 1/2
        std::vector<arma::uvec> blocks; // basis indices grouped by Mz
    };
    namespace Diagonalization
    {
        MzBlocks BuildMzBlocks(const std::vector<SpinAPI::spin_ptr> &spins);
        bool IsBlockDiagonalMz(const arma::sp_cx_mat &H, const std::vector<int> &mz2, double relTol);
        bool EigSymBlockMz(const arma::sp_cx_mat &H, const std::vector<arma::uvec> &blocks, arma::vec &eigval,
                           arma::cx_mat &eigvec);
    } // namespace Diagonalization
} // namespace RunSection::General::Resonance
#endif
