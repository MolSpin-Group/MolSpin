/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceDiagonalization.h"
#include "Spin.h"
#include <map>
#include <algorithm>

namespace RunSection::General::Resonance
{
    namespace Diagonalization
    {
        MzBlocks BuildMzBlocks(const std::vector<SpinAPI::spin_ptr> &spins)
        {
            MzBlocks result;
            if (spins.empty())
                return result;

            const size_t nspins = spins.size();
            std::vector<size_t> mult(nspins);
            std::vector<int> svals(nspins);
            size_t dim = 1;
            for (size_t i = 0; i < nspins; ++i)
            {
                mult[i] = static_cast<size_t>(spins[i]->Multiplicity());
                svals[i] = spins[i]->S();
                dim *= mult[i];
            }

            std::vector<size_t> stride(nspins, 1);
            for (size_t i = nspins; i-- > 0;)
            {
                if (i + 1 < nspins)
                    stride[i] = stride[i + 1] * mult[i + 1];
            }

            result.mz2.resize(dim);
            for (size_t idx = 0; idx < dim; ++idx)
            {
                int total = 0;
                for (size_t i = 0; i < nspins; ++i)
                {
                    const size_t local = (idx / stride[i]) % mult[i];
                    const int m = svals[i] - 2 * static_cast<int>(local);
                    total += m;
                }
                result.mz2[idx] = total;
            }

            std::map<int, std::vector<arma::uword>> groups;
            for (arma::uword i = 0; i < result.mz2.size(); ++i)
                groups[result.mz2[i]].push_back(i);

            result.blocks.reserve(groups.size());
            for (auto &kv : groups)
            {
                arma::uvec idx(static_cast<arma::uword>(kv.second.size()));
                for (size_t i = 0; i < kv.second.size(); ++i)
                    idx(static_cast<arma::uword>(i)) = kv.second[i];
                result.blocks.push_back(std::move(idx));
            }

            return result;
        }
        bool IsBlockDiagonalMz(const arma::sp_cx_mat &H, const std::vector<int> &mz2, double relTol)
        {
            if (H.n_nonzero == 0)
                return true;
            double maxAbs = 0.0;
            for (auto it = H.begin(); it != H.end(); ++it)
                maxAbs = std::max(maxAbs, std::abs(*it));
            if (maxAbs == 0.0)
                return true;
            const double thresh = maxAbs * relTol;
            for (auto it = H.begin(); it != H.end(); ++it)
            {
                if (std::abs(*it) <= thresh)
                    continue;
                if (mz2[it.row()] != mz2[it.col()])
                    return false;
            }
            return true;
        }
        bool EigSymBlockMz(const arma::sp_cx_mat &H, const std::vector<arma::uvec> &blocks, arma::vec &eigval,
                           arma::cx_mat &eigvec)
        {
            const arma::uword dim = H.n_rows;
            eigval.set_size(dim);
            eigvec.zeros(dim, dim);

            struct Entry
            {
                double val;
                size_t block;
                arma::uword local;
            };

            std::vector<Entry> entries;
            entries.reserve(dim);
            std::vector<arma::cx_mat> block_vecs(blocks.size());

            std::vector<int> block_id(static_cast<size_t>(dim), -1);
            std::vector<arma::uword> local_pos(static_cast<size_t>(dim), 0);
            for (size_t b = 0; b < blocks.size(); ++b)
            {
                const arma::uvec &idx = blocks[b];
                for (arma::uword i = 0; i < idx.n_elem; ++i)
                {
                    block_id[idx(i)] = static_cast<int>(b);
                    local_pos[idx(i)] = i;
                }
            }

            std::vector<arma::cx_mat> block_mats(blocks.size());
            for (size_t b = 0; b < blocks.size(); ++b)
            {
                const arma::uvec &idx = blocks[b];
                block_mats[b].zeros(idx.n_elem, idx.n_elem);
            }

            for (auto it = H.begin(); it != H.end(); ++it)
            {
                const int b = block_id[it.row()];
                if (b < 0)
                    continue;
                if (block_id[it.col()] != b)
                    continue;
                block_mats[static_cast<size_t>(b)](local_pos[it.row()], local_pos[it.col()]) = *it;
            }

            for (size_t b = 0; b < blocks.size(); ++b)
            {
                const arma::uvec &idx = blocks[b];
                if (idx.n_elem == 0)
                    continue;
                arma::vec evals;
                arma::cx_mat evecs;
                if (!arma::eig_sym(evals, evecs, block_mats[b]))
                    return false;
                block_vecs[b] = std::move(evecs);
                for (arma::uword k = 0; k < evals.n_elem; ++k)
                    entries.push_back({evals(k), b, k});
            }

            if (entries.size() != static_cast<size_t>(dim))
                return false;

            std::sort(entries.begin(), entries.end(),
                      [](const Entry &a, const Entry &b) { return a.val < b.val; });

            for (arma::uword col = 0; col < dim; ++col)
            {
                const auto &e = entries[col];
                eigval(col) = e.val;
                const arma::uvec &idx = blocks[e.block];
                for (arma::uword i = 0; i < idx.n_elem; ++i)
                {
                    eigvec(idx(i), col) = block_vecs[e.block](i, e.local);
                }
            }

            return true;
        }
    } // namespace Diagonalization
} // namespace RunSection::General::Resonance
