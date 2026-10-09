/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Action validation and affine field-sweep description.
#ifndef MOLSPIN_RESONANCEFIELDSWEEP_H
#define MOLSPIN_RESONANCEFIELDSWEEP_H
#include "ResonanceExecutionPlan.h"
#include "SpinAPIfwd.h"
#include "RunSectionfwd.h"
#include <armadillo>
#include <map>
#include <memory>

namespace RunSection::General::Resonance
{

    struct ResonanceFieldSweep
    {
        arma::vec field0, fieldStep;
        unsigned int steps = 0;
    };
    namespace FieldSweep
    {
        bool IsParallel(const arma::vec &a, const arma::vec &b, double tol);
        bool CollectAddVectorSteps(const std::vector<std::shared_ptr<Action>> &actions, unsigned int steps,
                                   std::map<std::string, arma::vec> &stepsOut, std::string &error);
        bool GetLinearFieldSweep(const std::vector<std::shared_ptr<Action>> &actions, unsigned int steps,
                                 const SpinAPI::system_ptr &_system,
                                 const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0,
                                 arma::vec &_fieldStep);
        bool Prepare(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                     const SpinAPI::interaction_ptr &field,
                     const std::vector<std::shared_ptr<Action>> &actions, unsigned int steps,
                     ResonanceFieldSweep &sweep, std::string &error);
    } // namespace FieldSweep

} // namespace RunSection::General::Resonance
#endif
