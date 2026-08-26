/////////////////////////////////////////////////////////////////////////
// MultiSSSystemPreparation (RunSection::General::MultiSS)
// ------------------
// Lays out independently prepared General/SS blocks in a direct-sum
// Liouville vector.  It owns bookkeeping, not local spin physics.
//
// For kinetic coupling between electronic manifolds the represented state is
//
//       rho_global = vec(rho_1) (+) vec(rho_2) (+) ...,
//       D_global = sum_i d_i^2.
//
// This is the efficient block-diagonal sector of an enlarged electronic
// Hilbert space.  It is valid when inter-manifold coherences are intentionally
// absent (kinetic population transfer).  If a future coherent optical drive
// creates S1-CSS coherences, an enlarged Hilbert-space mode is required instead.
// The block-diagonal kinetic representation is consistent with the reversible
// S1/CSS hierarchy discussed in DOI: 10.1039/D6CP00916F.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_MultiSS_MultiSSSystemPreparation
#define MOD_RunSection_General_MultiSS_MultiSSSystemPreparation

#include <armadillo>
#include <memory>
#include <string>
#include <vector>
#include "MultiSSExecutionPlan.h"
#include "MultiSSOrientationSampler.h"
#include "SSLiouvillianBuilder.h"

namespace RunSection::General::MultiSS
{
    struct MultiSSSystemContext
    {
        ::RunSection::General::SS::SSPreparedSystem local;
        arma::uword offset = 0;
        arma::uword superDimension = 0;
        arma::uword hilbertDimension = 0;
    };

    struct MultiSSPreparedSystems
    {
        std::vector<MultiSSSystemContext> contexts;
        arma::uword globalDimension = 0;
        arma::sp_cx_mat internalGenerator;
        arma::cx_vec initialState;
        arma::cx_vec traceFunctional;
    };

    class MultiSSSystemPreparation
    {
    public:
        static bool Prepare(const std::vector<SpinAPI::system_ptr> &_systems,
            const MultiSSExecutionPlan &_plan,
            const MultiSSOrientation &_orientation,
            MultiSSPreparedSystems &_prepared, std::string &_error);

        static const MultiSSSystemContext *FindContext(
            const MultiSSPreparedSystems &_prepared,
            const SpinAPI::system_ptr &_system);
        static MultiSSSystemContext *FindContext(
            MultiSSPreparedSystems &_prepared,
            const SpinAPI::system_ptr &_system);
    };
}
#endif
