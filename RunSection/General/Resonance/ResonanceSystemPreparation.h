/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Spin-system selection and common field/state/Hamiltonian preparation.
#ifndef MOLSPIN_RESONANCESYSTEMPREPARATION_H
#define MOLSPIN_RESONANCESYSTEMPREPARATION_H
#include "ResonanceExecutionPlan.h"
#include "HSStatePreparation.h"
#include "ResonanceMagneticMomentBuilder.h"
#include <utility>

namespace RunSection::General::Resonance
{
    struct FieldSyncGuard
    {
        FieldSyncGuard() = default;
        FieldSyncGuard(const FieldSyncGuard &) = delete;
        FieldSyncGuard &operator=(const FieldSyncGuard &) = delete;
        std::vector<std::pair<SpinAPI::interaction_ptr, arma::vec>> saved;

        void Apply(const std::vector<SpinAPI::interaction_ptr> &_interactions, const arma::vec &_field);
        ~FieldSyncGuard();
    };
    namespace SystemPreparation
    {
        bool IsZeemanInteraction(const SpinAPI::interaction_ptr &inter);
        std::vector<SpinAPI::interaction_ptr> CollectZeemanInteractions(
            const SpinAPI::system_ptr &system, const std::vector<std::string> &h0list);
        SpinAPI::interaction_ptr FindZeemanForSpin(const SpinAPI::spin_ptr &spin,
                                                   const std::vector<SpinAPI::interaction_ptr> &zeemanList);
        bool ResolveFieldInteraction(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &_system,
                                     SpinAPI::interaction_ptr &_fieldInteraction);
        bool ResolveDetectionSpins(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &_system,
                                   const SpinAPI::interaction_ptr &_fieldInteraction,
                                   std::vector<SpinAPI::spin_ptr> &_spins,
                                   std::vector<std::string> &_spinNames);
        bool ResolveHamiltonian(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                                std::vector<std::string> &names, std::string &error);
        bool BuildHamiltonian(SpinAPI::SpinSpace &space, const std::vector<std::string> &names,
                              const arma::mat &rotation, arma::sp_cx_mat &h, std::string &error);
        bool PrepareState(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                          SpinAPI::SpinSpace &space, HS::HSPreparedState &state, std::string &error);
        bool OrientState(SpinAPI::SpinSpace &space, const HS::HSPreparedState &state,
                         const arma::mat &rotation, arma::cx_mat &density, std::string &error);
        bool PrepareMoments(const std::vector<SpinAPI::spin_ptr> &spins,
                            const std::vector<SpinAPI::interaction_ptr> &zeeman,
                            std::vector<ResonanceMagneticMomentTerm> &terms, std::string &error);
    } // namespace SystemPreparation

} // namespace RunSection::General::Resonance
#endif
