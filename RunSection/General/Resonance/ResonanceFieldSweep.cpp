/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceFieldSweep.h"
#include "ResonanceSystemPreparation.h"
#include "ActionAddVector.h"
#include "ObjectParser.h"
#include "SpinSystem.h"
#include "Interaction.h"
#include <cmath>

namespace RunSection::General::Resonance
{
    namespace FieldSweep
    {
        bool Prepare(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                     const SpinAPI::interaction_ptr &field,
                     const std::vector<std::shared_ptr<Action>> &actions, unsigned int steps,
                     ResonanceFieldSweep &sweep, std::string &error)
        {
            error.clear();
            sweep = ResonanceFieldSweep();
            if (steps < 2 ||
                !GetLinearFieldSweep(actions, steps, system, field, sweep.field0, sweep.fieldStep))
            {
                error = "a linear AddVector field sweep with at least two steps is required";
                return false;
            }
            const arma::vec last = sweep.field0 + (steps - 1) * sweep.fieldStep;
            if (!(arma::norm(sweep.field0) > 0) || !last.is_finite() || arma::dot(sweep.field0, last) <= 0)
            {
                error = "cached resonance field sweeps must stay on one side of zero";
                return false;
            }
            if (plan.detectionHarmonic > 0 && (steps < 3 || !(arma::norm(sweep.fieldStep) > 0)))
            {
                error = "a field derivative requires at least three distinct sweep fields";
                return false;
            }
            std::map<std::string, arma::vec> increments;
            if (!CollectAddVectorSteps(actions, steps, increments, error))
                return false;
            std::vector<std::string> hnames;
            if (!SystemPreparation::ResolveHamiltonian(plan, system, hnames, error))
                return false;
            const auto zeeman = SystemPreparation::CollectZeemanInteractions(system, hnames);
            for (const auto &entry : increments)
            {
                bool allowed = false;
                for (const auto &z : zeeman)
                    if (entry.first == system->Name() + "." + z->Name() + ".field")
                        allowed = true;
                // Other SpinSystems can have their own independent field sweeps.
                if (entry.first.rfind(system->Name() + ".", 0) != 0)
                    allowed = true;
                if (!allowed)
                {
                    error = "a cached resonance sweep cannot include changes to other system parameters";
                    return false;
                }
            }
            for (const auto &z : zeeman)
            {
                if (plan.enforceZeemanSync || z == field)
                    continue;
                const auto value = z->Field();
                const auto increment = increments.find(system->Name() + "." + z->Name() + ".field");
                if (value.n_elem != 3 || !value.is_finite() || arma::norm(value - sweep.field0) > 1e-8 ||
                    increment == increments.end() || arma::norm(increment->second - sweep.fieldStep) > 1e-8)
                {
                    error = "cached resonance requires synchronized Zeeman fields and sweep increments";
                    return false;
                }
            }
            sweep.steps = steps;
            return true;
        }
        bool IsParallel(const arma::vec &a, const arma::vec &b, double tol)
        {
            if (a.n_elem != 3 || b.n_elem != 3)
                return false;
            const double na = arma::norm(a);
            const double nb = arma::norm(b);
            if (!std::isfinite(na) || !std::isfinite(nb) || na == 0.0 || nb == 0.0)
                return false;
            return (arma::norm(arma::cross(a / na, b / nb)) <= tol);
        }
        bool CollectAddVectorSteps(const std::vector<std::shared_ptr<Action>> &actions, unsigned int steps,
                                   std::map<std::string, arma::vec> &stepsOut, std::string &error)
        {
            stepsOut.clear();
            error.clear();

            for (const auto &action : actions)
            {
                auto add = std::dynamic_pointer_cast<ActionAddVector>(action);
                if (!add)
                {
                    error = "Non-AddVector action present.";
                    return false;
                }

                std::string targetName;
                if (!add->GetProperties()->Get("vector", targetName))
                    add->GetProperties()->Get("actionvector", targetName);
                if (targetName.empty())
                {
                    error = "AddVector action missing target vector.";
                    return false;
                }

                arma::vec direction;
                if (!add->GetProperties()->Get("direction", direction) || direction.n_elem != 3 ||
                    !direction.is_finite())
                {
                    error = "AddVector action has invalid direction.";
                    return false;
                }
                direction = arma::normalise(direction);

                if (add->Period() != 1 || add->First() != 1)
                {
                    error = "AddVector action has non-unit period or does not start at step 1.";
                    return false;
                }
                if (add->Last() != 0 && add->Last() < steps)
                {
                    error = "AddVector action terminates before the end of the run.";
                    return false;
                }

                const arma::vec step = add->Value() * direction;
                auto it = stepsOut.find(targetName);
                if (it != stepsOut.end())
                {
                    // RunSection applies every action. Multiple increments for one
                    // target therefore compose into one net sweep step.
                    it->second += step;
                }
                else
                {
                    stepsOut.emplace(targetName, step);
                }
            }

            return true;
        }
        bool GetLinearFieldSweep(const std::vector<std::shared_ptr<Action>> &actions, unsigned int steps,
                                 const SpinAPI::system_ptr &_system,
                                 const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0,
                                 arma::vec &_fieldStep)
        {
            if (_system == nullptr || _fieldInteraction == nullptr)
                return false;

            _field0 = _fieldInteraction->Field();
            if (_field0.n_elem != 3 || !_field0.is_finite())
                return false;

            const std::string target = _system->Name() + "." + _fieldInteraction->Name() + ".field";

            std::map<std::string, arma::vec> stepsByTarget;
            std::string error;
            if (!CollectAddVectorSteps(actions, steps, stepsByTarget, error))
                return false;

            auto it = stepsByTarget.find(target);
            if (it == stepsByTarget.end())
                return false;

            _fieldStep = it->second;
            if (arma::norm(_field0) > 0.0 && arma::norm(_fieldStep) > 0.0)
            {
                arma::vec dir0 = arma::normalise(_field0);
                arma::vec dirStep = arma::normalise(_fieldStep);
                if (arma::norm(arma::cross(dir0, dirStep)) > 1e-6)
                    return false;
            }

            return true;
        }
    } // namespace FieldSweep
} // namespace RunSection::General::Resonance
