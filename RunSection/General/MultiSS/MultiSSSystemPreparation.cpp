/////////////////////////////////////////////////////////////////////////
// MultiSSSystemPreparation implementation (RunSection::General::MultiSS)
// ----------------------------------------------------------------------
// REPRESENTATION
//   The global kinetic state is a direct sum of local Liouville vectors,
//
//       rho_global = vec(rho_1) (+) ... (+) vec(rho_N),
//       D_global   = sum_i d_i^2.
//
//   It is not the tensor product of unrelated electronic manifolds.  Under
//   kinetic S1<->CSS coupling with no coherent S1/CSS superpositions, this is
//   exactly the block-diagonal sector of the enlarged Hilbert-space model and
//   avoids storing physically absent inter-manifold coherences.  The reversible
//   S1/CSS hierarchy motivating this representation is discussed in
//       DOI: 10.1039/D6CP00916F
//
// OWNERSHIP
//   General/SS prepares each local block.  This class only assigns offsets,
//   assembles the block-diagonal internal generator, vectorizes initial
//   densities, and constructs the global trace functional.  Network edges are
//   added later by MultiSSNetworkBuilder.
//
// INITIAL-POPULATION CONTRACT
//   Each local initial density is normalized internally and may then be scaled
//   by SpinSystem property `initialpopulation`.  Users must therefore specify
//   the intended global initial distribution explicitly; a typical pump-probe
//   calculation initializes only the ground/S1 manifold and leaves the other
//   manifolds at zero population.
/////////////////////////////////////////////////////////////////////////
#include "MultiSSSystemPreparation.h"
#include "SpinSpace.h"
#include "SpinSystem.h"

#include <cmath>

namespace RunSection::General::MultiSS
{
    const MultiSSSystemContext *MultiSSSystemPreparation::FindContext(
        const MultiSSPreparedSystems &prepared,const SpinAPI::system_ptr &system)
    {
        for(const auto &c:prepared.contexts) if(c.local.system==system) return &c;
        return nullptr;
    }
    MultiSSSystemContext *MultiSSSystemPreparation::FindContext(
        MultiSSPreparedSystems &prepared,const SpinAPI::system_ptr &system)
    {
        for(auto &c:prepared.contexts) if(c.local.system==system) return &c;
        return nullptr;
    }

    bool MultiSSSystemPreparation::Prepare(const std::vector<SpinAPI::system_ptr> &systems,
        const MultiSSExecutionPlan &plan,const MultiSSOrientation &orientation,
        MultiSSPreparedSystems &prepared,std::string &error)
    {
        prepared=MultiSSPreparedSystems(); error.clear();
        if(systems.empty()){error="MultiSSGeneral requires at least one SpinSystem";return false;}

        for(const auto &system:systems)
        {
            if(system==nullptr){error="MultiSSGeneral cannot prepare a null SpinSystem";return false;}
            MultiSSSystemContext context;
            context.offset=prepared.globalDimension;
            if(!::RunSection::General::SS::SSLiouvillianBuilder::Prepare(system,
                plan.hamiltonianMode,orientation.frameToLab,context.local,error,plan.historicalNZ)) return false;
            context.hilbertDimension=context.local.space->HilbertSpaceDimensions();
            context.superDimension=context.hilbertDimension*context.hilbertDimension;
            prepared.globalDimension+=context.superDimension;
            prepared.contexts.push_back(std::move(context));
        }

        prepared.internalGenerator.zeros(prepared.globalDimension,prepared.globalDimension);
        prepared.initialState.zeros(prepared.globalDimension);
        prepared.traceFunctional.zeros(prepared.globalDimension);
        for(auto &context:prepared.contexts)
        {
            prepared.internalGenerator.submat(context.offset,context.offset,
                context.offset+context.superDimension-1,
                context.offset+context.superDimension-1)=context.local.internalLiouvillian;

            context.local.space->UseSuperoperatorSpace(true);
            arma::cx_vec vector;
            if(!context.local.space->OperatorToSuperspace(context.local.initialDensity,vector))
            {error="failed to vectorize initial density for \""+context.local.system->Name()+"\"";return false;}
            prepared.initialState.subvec(context.offset,context.offset+context.superDimension-1)=vector;

            // Trace functional in MolSpin's row-major Liouville convention.
            // For an identity operator, OperatorToSuperspace(I) has ones only
            // at the d diagonal coordinates and therefore evaluates Tr(rho).
            arma::cx_vec traceBlock;
            if(!context.local.space->OperatorToSuperspace(
                arma::eye<arma::cx_mat>(context.hilbertDimension,context.hilbertDimension),traceBlock))
            {error="failed to construct direct-sum trace functional";return false;}
            prepared.traceFunctional.subvec(context.offset,context.offset+context.superDimension-1)=traceBlock;
        }
        return true;
    }
}
