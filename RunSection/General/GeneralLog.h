/////////////////////////////////////////////////////////////////////////
// GeneralLog (RunSection::General)
// ------------------
// Shared diagnostic/logging helpers for the General execution hierarchy.
//
// DESIGN CONTRACT
//   * General tasks should report the resolved physics/configuration, not merely
//     announce that a task started.
//   * The logger is representation-agnostic: HS, SS and MultiSS can use the same
//     SpinSystem inventory and powder/progress summaries.
//   * Logging must not own physics or mutate SpinAPI objects.
//   * Expensive orientation loops are throttled; diagnostics=true remains the
//     default in General execution plans, but useful startup information is
//     concise enough to keep normal log files readable.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_GeneralLog
#define MOD_RunSection_General_GeneralLog

#include "Interaction.h"
#include "Operator.h"
#include "Pulse.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace RunSection::General::Log
{
    inline const char *InteractionTypeName(SpinAPI::InteractionType type)
    {
        switch (type)
        {
        case SpinAPI::InteractionType::SingleSpin: return "single-spin";
        case SpinAPI::InteractionType::DoubleSpin: return "double-spin";
        case SpinAPI::InteractionType::QuadraticSpin: return "quadratic-spin";
        case SpinAPI::InteractionType::Exchange: return "exchange";
        case SpinAPI::InteractionType::Zfs: return "zfs";
        case SpinAPI::InteractionType::SemiClassicalField: return "semi-classical-field";
        case SpinAPI::InteractionType::Strain: return "strain";
        default: return "undefined";
        }
    }

    inline const char *FieldTypeName(SpinAPI::InteractionFieldType type)
    {
        switch (type)
        {
        case SpinAPI::InteractionFieldType::Static: return "static";
        case SpinAPI::InteractionFieldType::LinearPolarization: return "linear-polarized";
        case SpinAPI::InteractionFieldType::CircularPolarization: return "circular-polarized";
        case SpinAPI::InteractionFieldType::Broadband: return "broadband";
        case SpinAPI::InteractionFieldType::OUGeneral: return "ornstein-uhlenbeck";
        case SpinAPI::InteractionFieldType::Trajectory: return "trajectory";
        default: return "unknown";
        }
    }

    inline const char *PulseTypeName(SpinAPI::PulseType type)
    {
        switch (type)
        {
        case SpinAPI::PulseType::InstantPulse: return "instant";
        case SpinAPI::PulseType::LongPulse: return "long";
        case SpinAPI::PulseType::LongPulseStaticField: return "long-static-field";
        case SpinAPI::PulseType::MWPulse: return "MW";
        case SpinAPI::PulseType::ShapedPulse: return "shaped";
        default: return "unspecified";
        }
    }

    inline const char *TransitionTypeName(SpinAPI::TransitionType type)
    {
        return type == SpinAPI::TransitionType::Source ? "source" : "sink";
    }

    inline const char *ReactionTypeName(SpinAPI::ReactionOperatorType type)
    {
        switch (type)
        {
        case SpinAPI::ReactionOperatorType::Haberkorn: return "Haberkorn";
        case SpinAPI::ReactionOperatorType::Lindblad: return "Lindblad";
        default: return "task/default";
        }
    }

    inline const char *OperatorTypeName(SpinAPI::OperatorType type)
    {
        switch (type)
        {
        case SpinAPI::OperatorType::RelaxationLindblad: return "Lindblad(single-spin)";
        case SpinAPI::OperatorType::RelaxationLindbladDoubleSpin: return "Lindblad(double-spin)";
        case SpinAPI::OperatorType::RelaxationDephasing: return "dephasing";
        case SpinAPI::OperatorType::RelaxationRandomFields: return "random-fields";
        case SpinAPI::OperatorType::RelaxationT1: return "T1";
        case SpinAPI::OperatorType::RelaxationT2: return "T2";
        case SpinAPI::OperatorType::RelaxationPhenomenological: return "phenomenological";
        default: return "unspecified";
        }
    }

    inline std::string SpinNames(const std::vector<SpinAPI::spin_ptr> &spins)
    {
        std::ostringstream out;
        out << "{";
        for (size_t i = 0; i < spins.size(); ++i)
        {
            if (i != 0) out << ",";
            out << (spins[i] ? spins[i]->Name() : std::string("<null>"));
        }
        out << "}";
        return out.str();
    }

    inline std::string Vector3(const arma::vec &v)
    {
        if (v.n_elem < 3) return "<unset>";
        std::ostringstream out;
        out << std::setprecision(8) << "[" << v(0) << "," << v(1) << "," << v(2) << "]";
        return out.str();
    }

    inline void PrintInitialState(std::ostream &log, const SpinAPI::system_ptr &system)
    {
        const auto states = system->InitialState();
        if (states.empty())
        {
            log << "    initial state: <none configured>" << std::endl;
            return;
        }
        std::vector<double> weights = system->Weights();
        if (weights.empty()) weights.assign(states.size(), 1.0);
        const double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (sum > 0.0)
            for (double &w : weights) w /= sum;

        log << "    initial state: ";
        for (size_t i = 0; i < states.size(); ++i)
        {
            if (i != 0) log << " + ";
            const std::string name = states[i] ? states[i]->Name() : std::string("thermal");
            const double weight = i < weights.size() ? weights[i] : 1.0;
            log << std::setprecision(8) << weight << "*" << name;
        }
        log << "." << std::endl;
    }

    inline void PrintSystemInventory(std::ostream &log,
        const std::vector<SpinAPI::system_ptr> &systems,
        const std::string &heading = "General physics objects")
    {
        log << "\n--- " << heading << " ---" << std::endl;
        for (const auto &system : systems)
        {
            if (system == nullptr)
            {
                log << "SpinSystem <null>." << std::endl;
                continue;
            }

            SpinAPI::SpinSpace space(system);
            const arma::uword d = space.HilbertSpaceDimensions();
            const arma::uword liouville = d * d;
            const auto spins = system->Spins();
            const auto states = system->States();
            const auto interactions = system->Interactions();
            const auto transitions = system->Transitions();
            const auto operators = system->Operators();
            const auto pulses = system->Pulses();

            log << "SpinSystem \"" << system->Name() << "\":" << std::endl;
            log << "    spins=" << spins.size() << ", states=" << states.size()
                << ", interactions=" << interactions.size() << ", transitions=" << transitions.size()
                << ", relaxation operators=" << operators.size() << ", pulses=" << pulses.size()
                << ", Hilbert dimension=" << d << ", Liouville dimension=" << liouville << "." << std::endl;
            log << "    spin objects=" << SpinNames(spins) << std::endl;
            PrintInitialState(log, system);

            for (const auto &interaction : interactions)
            {
                if (!interaction) continue;
                log << "  Interaction \"" << interaction->Name() << "\":" << std::endl;
                log << "    type=" << InteractionTypeName(interaction->Type())
                    << ", group1=" << SpinNames(interaction->Group1());
                if (!interaction->Group2().empty())
                    log << ", group2=" << SpinNames(interaction->Group2());
                log << std::endl;
                log << "    field treatment=" << FieldTypeName(interaction->FieldType());
                const arma::vec field = interaction->Field();
                if (field.n_elem >= 3) log << ", field(T)=" << Vector3(field);
                if (interaction->HasTimeDependence())
                    log << ", time-dependent=yes";
                log << std::endl;
                if (interaction->HasFieldTimeDependence())
                    log << "    carrier frequency=" << std::setprecision(10)
                        << interaction->GetTDFrequency() << "." << std::endl;
            }

            for (const auto &pulse : pulses)
            {
                if (!pulse) continue;
                log << "  Pulse \"" << pulse->Name() << "\":" << std::endl;
                log << "    type=" << PulseTypeName(pulse->Type())
                    << ", affected spins=" << SpinNames(pulse->Group())
                    << ", duration(ns)=" << std::setprecision(10) << pulse->Pulsetime()
                    << ", frequency=" << pulse->Frequency()
                    << ", field(T)=" << Vector3(pulse->Field()) << std::endl;
                if (pulse->Type() == SpinAPI::PulseType::MWPulse)
                    log << "    treatment=Pulse-defined MW/RWA semantics." << std::endl;
            }

            for (const auto &transition : transitions)
            {
                if (!transition) continue;
                log << "  Transition \"" << transition->Name() << "\":" << std::endl;
                log << "    type=" << TransitionTypeName(transition->Type())
                    << ", source=" << (transition->SourceState() ? transition->SourceState()->Name() : std::string("<none>"))
                    << ", target system=" << (transition->Target() ? transition->Target()->Name() : std::string("<terminal>"))
                    << ", target state=" << (transition->TargetState() ? transition->TargetState()->Name() : std::string("<none>"))
                    << ", rate=" << std::setprecision(10) << transition->Rate()
                    << ", reaction operator=" << ReactionTypeName(transition->GetReactionOperatorType())
                    << "." << std::endl;
            }

            for (const auto &op : operators)
            {
                if (!op) continue;
                log << "  Relaxation Operator \"" << op->Name() << "\":" << std::endl;
                log << "    type=" << OperatorTypeName(op->Type())
                    << ", affected spins=" << SpinNames(op->Spins())
                    << ", rates={" << std::setprecision(10) << op->Rate1() << ","
                    << op->Rate2() << "," << op->Rate3() << "}." << std::endl;
            }
        }
    }

    inline void PrintPowderSummary(std::ostream &log, const std::string &mode,
        size_t count, double normalizedWeightSum)
    {
        log << "\nPowder/orientation sampling:" << std::endl;
        log << "    mode=" << mode << std::endl;
        log << "    orientations=" << count << std::endl;
        log << "    normalized weight sum=" << std::setprecision(16) << normalizedWeightSum << std::endl;
        if (std::abs(normalizedWeightSum - 1.0) > 1.0e-10)
            log << "WARNING: orientation weights do not sum to unity; powder-averaged observables may be mis-normalized." << std::endl;
    }

    inline bool ShouldReportOrientation(size_t index, size_t total, size_t desiredUpdates = 10)
    {
        if (total <= 1) return false;
        if (total <= desiredUpdates + 1) return true;
        const size_t stride = std::max<size_t>(1, total / desiredUpdates);
        return index == 0 || index + 1 == total || ((index + 1) % stride == 0);
    }

    inline void PrintOrientationProgress(std::ostream &log, size_t index, size_t total)
    {
        if (!ShouldReportOrientation(index, total)) return;
        const double percent = total > 0 ? 100.0 * static_cast<double>(index + 1) / static_cast<double>(total) : 100.0;
        log << "General orientation progress: " << (index + 1) << "/" << total
            << " (" << std::fixed << std::setprecision(1) << percent << "%)." << std::defaultfloat << std::endl;
    }
}

#endif
