/////////////////////////////////////////////////////////////////////////
// HybridNuclearShiftKernel (RunSection::General::Resonance)
// ------------------
// Deterministic composition of independent conditional-nuclear transition
// factors.  Hamiltonian construction and state tracking remain in
// HybridNuclearResonanceSolver; this class only convolves their additive
// frequency/field-response descriptors.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_HybridNuclearShiftKernel
#define MOD_RunSection_General_Resonance_HybridNuclearShiftKernel

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace RunSection::General::Resonance
{
    enum class HybridNuclearCompositionMode
    {
        Auto,
        Explicit,
        Compressed
    };

    inline const char *HybridNuclearCompositionModeName(
        HybridNuclearCompositionMode mode)
    {
        switch (mode)
        {
        case HybridNuclearCompositionMode::Explicit:
            return "explicit";
        case HybridNuclearCompositionMode::Compressed:
            return "compressed";
        default:
            return "auto";
        }
    }

    struct HybridNuclearShiftComponent
    {
        double omegaRadNs = 0.0;
        double slopeH_RadNsT = 0.0;
        double slopeHH_RadNsT = 0.0;
        double weight = 0.0;
    };

    struct HybridNuclearShiftKernelOptions
    {
        HybridNuclearCompositionMode mode =
            HybridNuclearCompositionMode::Auto;

        // Maximum displacement of a merged line on the field axis.  Zero is
        // the exact (uncompressed) limit.
        double fieldTolerance_mT = 0.0;
        double linewidth_mT = 0.0;
        double microwaveOmegaRadNs = 0.0;
        double minimumSlopeRadNsT = 1.0e-15;
        double referenceSlopeRadNsT = 0.0;
        std::size_t maximumComponents = 65536;
    };

    struct HybridNuclearShiftKernelReport
    {
        HybridNuclearCompositionMode backend =
            HybridNuclearCompositionMode::Explicit;
        std::size_t formalCartesianComponents = 1;
        bool formalCartesianOverflow = false;
        std::size_t generatedCandidates = 0;
        std::size_t mergedComponents = 0;
        std::size_t maximumIntermediateComponents = 1;
        std::vector<std::size_t> componentsAfterStage;
        double inputWeight = 1.0;
        double outputWeight = 1.0;
        double relativeWeightError = 0.0;
    };

    class HybridNuclearShiftKernel
    {
    private:
        struct BinKey
        {
            long long field = 0;
            long long slopeH = 0;
            long long slopeHH = 0;

            bool operator<(const BinKey &other) const
            {
                return std::tie(field,slopeH,slopeHH) <
                    std::tie(other.field,other.slopeH,other.slopeHH);
            }
        };

        struct Accumulator
        {
            long double weight = 0.0L;
            long double omegaWeight = 0.0L;
            long double slopeHWeight = 0.0L;
            long double slopeHHWeight = 0.0L;
            std::size_t count = 0;
        };

        static bool CheckedProduct(
            std::size_t a,
            std::size_t b,
            std::size_t &result)
        {
            if (a != 0 &&
                b > std::numeric_limits<std::size_t>::max()/a)
            {
                result = std::numeric_limits<std::size_t>::max();
                return false;
            }
            result = a*b;
            return true;
        }

        static bool Quantize(
            double value,
            double width,
            long long &bin)
        {
            if (!std::isfinite(value) ||
                !std::isfinite(width) || width <= 0.0)
                return false;
            const long double scaled =
                static_cast<long double>(value)/
                static_cast<long double>(width);
            if (scaled <
                    static_cast<long double>(
                        std::numeric_limits<long long>::min())+1.0L ||
                scaled >
                    static_cast<long double>(
                        std::numeric_limits<long long>::max())-1.0L)
                return false;
            bin = static_cast<long long>(std::llround(scaled));
            return true;
        }

        static bool MakeKey(
            const HybridNuclearShiftComponent &component,
            const HybridNuclearShiftKernelOptions &options,
            double slopeScale,
            BinKey &key)
        {
            // Use the fixed exact-core slope scale for the frequency axis.
            // This keeps the convolution grid additive and well defined even
            // if an intermediate nuclear correction nearly cancels a slope.
            const double fieldCoordinate_mT =
                1.0e3*(component.omegaRadNs-
                    options.microwaveOmegaRadNs)/slopeScale;

            // A slope difference changes the field displacement across a
            // line profile.  The relative bin width therefore scales with
            // the requested field tolerance divided by one linewidth.
            const double profileScale_mT =
                std::max(options.linewidth_mT,
                    options.fieldTolerance_mT);
            const double relativeSlopeTolerance =
                std::min(0.25,
                    options.fieldTolerance_mT/profileScale_mT);
            const double slopeFloor =
                std::max(options.minimumSlopeRadNsT,
                    slopeScale*relativeSlopeTolerance);
            const double transformedSlopeH=std::asinh(
                component.slopeH_RadNsT/slopeFloor);
            const double transformedSlopeHH=std::asinh(
                component.slopeHH_RadNsT/slopeFloor);

            return
                Quantize(fieldCoordinate_mT,
                    options.fieldTolerance_mT,key.field) &&
                Quantize(transformedSlopeH,
                    relativeSlopeTolerance,key.slopeH) &&
                Quantize(transformedSlopeHH,
                    relativeSlopeTolerance,key.slopeHH);
        }

        static bool IsFinite(
            const HybridNuclearShiftComponent &component)
        {
            return std::isfinite(component.omegaRadNs) &&
                std::isfinite(component.slopeH_RadNsT) &&
                std::isfinite(component.slopeHH_RadNsT) &&
                std::isfinite(component.weight) &&
                component.weight >= 0.0;
        }

    public:
        static bool EstimateFormalSize(
            const std::vector<
                std::vector<HybridNuclearShiftComponent>> &factors,
            std::size_t &formalSize,
            bool &overflow)
        {
            formalSize = 1;
            overflow = false;
            for (const auto &factor:factors)
            {
                std::size_t next = 0;
                if (!CheckedProduct(
                        formalSize,factor.size(),next))
                    overflow = true;
                formalSize = next;
            }
            return !overflow;
        }

        static bool Compose(
            const HybridNuclearShiftComponent &core,
            const std::vector<
                std::vector<HybridNuclearShiftComponent>> &factors,
            const HybridNuclearShiftKernelOptions &options,
            std::vector<HybridNuclearShiftComponent> &components,
            HybridNuclearShiftKernelReport &report,
            std::string &error)
        {
            components.clear();
            report = HybridNuclearShiftKernelReport{};
            error.clear();

            if (!IsFinite(core) || core.weight <= 0.0 ||
                !std::isfinite(options.fieldTolerance_mT) ||
                options.fieldTolerance_mT < 0.0 ||
                !std::isfinite(options.linewidth_mT) ||
                options.linewidth_mT < 0.0 ||
                !std::isfinite(options.microwaveOmegaRadNs) ||
                !std::isfinite(options.minimumSlopeRadNsT) ||
                options.minimumSlopeRadNsT <= 0.0 ||
                !std::isfinite(options.referenceSlopeRadNsT) ||
                options.referenceSlopeRadNsT < 0.0 ||
                options.maximumComponents == 0)
            {
                error = "invalid hybrid nuclear shift-kernel request";
                return false;
            }
            for (const auto &factor:factors)
            {
                if (factor.empty())
                {
                    error = "hybrid nuclear shift factor is empty";
                    return false;
                }
                for (const auto &component:factor)
                {
                    if (!IsFinite(component))
                    {
                        error = "hybrid nuclear shift factor contains invalid values";
                        return false;
                    }
                }
            }

            EstimateFormalSize(
                factors,report.formalCartesianComponents,
                report.formalCartesianOverflow);

            HybridNuclearCompositionMode backend = options.mode;
            if (backend == HybridNuclearCompositionMode::Auto)
            {
                backend =
                    report.formalCartesianOverflow ||
                    report.formalCartesianComponents >
                        options.maximumComponents
                    ? HybridNuclearCompositionMode::Compressed
                    : HybridNuclearCompositionMode::Explicit;
            }
            report.backend = backend;

            if (backend == HybridNuclearCompositionMode::Explicit &&
                (report.formalCartesianOverflow ||
                 report.formalCartesianComponents>
                    options.maximumComponents))
            {
                error =
                    "explicit hybrid nuclear component cap exceeded before allocation";
                return false;
            }

            if (backend == HybridNuclearCompositionMode::Compressed &&
                options.fieldTolerance_mT == 0.0)
            {
                error =
                    "compressed hybrid composition requires a positive field tolerance";
                return false;
            }

            components.push_back(core);
            report.inputWeight = core.weight;
            report.outputWeight = core.weight;

            const double slopeScale = std::max(
                std::max(options.minimumSlopeRadNsT,
                    options.referenceSlopeRadNsT),
                std::max(std::abs(core.slopeH_RadNsT),
                    std::abs(core.slopeHH_RadNsT)));

            for (const auto &factor:factors)
            {
                long double factorWeight = 0.0L;
                for (const auto &local:factor)
                    factorWeight += local.weight;
                report.inputWeight *=
                    static_cast<double>(factorWeight);

                if (components.size() >
                    std::numeric_limits<std::size_t>::max()/
                        factor.size())
                {
                    error =
                        "hybrid nuclear convolution candidate count overflow";
                    return false;
                }
                const std::size_t candidates =
                    components.size()*factor.size();
                if (report.generatedCandidates >
                    std::numeric_limits<std::size_t>::max()-candidates)
                    report.generatedCandidates =
                        std::numeric_limits<std::size_t>::max();
                else
                    report.generatedCandidates += candidates;

                if (backend == HybridNuclearCompositionMode::Explicit)
                {
                    if (candidates > options.maximumComponents)
                    {
                        error =
                            "explicit hybrid nuclear component cap exceeded before allocation";
                        return false;
                    }
                    std::vector<HybridNuclearShiftComponent> next;
                    next.reserve(candidates);
                    for (const auto &partial:components)
                    {
                        for (const auto &local:factor)
                        {
                            HybridNuclearShiftComponent combined;
                            combined.omegaRadNs =
                                partial.omegaRadNs+local.omegaRadNs;
                            combined.slopeH_RadNsT =
                                partial.slopeH_RadNsT+
                                local.slopeH_RadNsT;
                            combined.slopeHH_RadNsT =
                                partial.slopeHH_RadNsT+
                                local.slopeHH_RadNsT;
                            combined.weight =
                                partial.weight*local.weight;
                            if (!IsFinite(combined))
                            {
                                error =
                                    "non-finite explicit hybrid convolution component";
                                return false;
                            }
                            next.push_back(combined);
                        }
                    }
                    components.swap(next);
                }
                else
                {
                    std::map<BinKey,Accumulator> bins;
                    for (const auto &partial:components)
                    {
                        for (const auto &local:factor)
                        {
                            HybridNuclearShiftComponent combined;
                            combined.omegaRadNs =
                                partial.omegaRadNs+local.omegaRadNs;
                            combined.slopeH_RadNsT =
                                partial.slopeH_RadNsT+
                                local.slopeH_RadNsT;
                            combined.slopeHH_RadNsT =
                                partial.slopeHH_RadNsT+
                                local.slopeHH_RadNsT;
                            combined.weight =
                                partial.weight*local.weight;
                            if (!IsFinite(combined))
                            {
                                error =
                                    "non-finite compressed hybrid convolution component";
                                return false;
                            }
                            if (combined.weight == 0.0)
                                continue;

                            BinKey key;
                            if (!MakeKey(
                                    combined,options,
                                    slopeScale,key))
                            {
                                error =
                                    "hybrid convolution component cannot be mapped to the compression grid";
                                return false;
                            }
                            auto inserted = bins.emplace(
                                key,Accumulator{});
                            if (inserted.second &&
                                bins.size() > options.maximumComponents)
                            {
                                error =
                                    "compressed hybrid nuclear component cap exceeded; increase tolerance or the component limit";
                                return false;
                            }
                            auto &accumulator = inserted.first->second;
                            const long double weight =
                                combined.weight;
                            accumulator.weight += weight;
                            accumulator.omegaWeight +=
                                weight*combined.omegaRadNs;
                            accumulator.slopeHWeight +=
                                weight*combined.slopeH_RadNsT;
                            accumulator.slopeHHWeight +=
                                weight*combined.slopeHH_RadNsT;
                            ++accumulator.count;
                        }
                    }

                    std::vector<HybridNuclearShiftComponent> next;
                    next.reserve(bins.size());
                    for (const auto &entry:bins)
                    {
                        const auto &accumulator=entry.second;
                        if (!(accumulator.weight>0.0L))
                            continue;
                        HybridNuclearShiftComponent combined;
                        combined.weight =
                            static_cast<double>(accumulator.weight);
                        combined.omegaRadNs = static_cast<double>(
                            accumulator.omegaWeight/accumulator.weight);
                        combined.slopeH_RadNsT = static_cast<double>(
                            accumulator.slopeHWeight/accumulator.weight);
                        combined.slopeHH_RadNsT = static_cast<double>(
                            accumulator.slopeHHWeight/accumulator.weight);
                        if (!IsFinite(combined))
                        {
                            error =
                                "invalid moment-preserving hybrid compression result";
                            return false;
                        }
                        next.push_back(combined);
                        if (accumulator.count > 1)
                            report.mergedComponents +=
                                accumulator.count-1;
                    }
                    components.swap(next);
                }

                report.maximumIntermediateComponents = std::max(
                    report.maximumIntermediateComponents,
                    components.size());
                report.componentsAfterStage.push_back(
                    components.size());
            }

            long double outputWeight = 0.0L;
            for (const auto &component:components)
                outputWeight += component.weight;
            report.outputWeight =
                static_cast<double>(outputWeight);
            report.relativeWeightError =
                report.inputWeight == 0.0
                ? std::abs(report.outputWeight)
                : std::abs(report.outputWeight-report.inputWeight)/
                    std::abs(report.inputWeight);
            if (!std::isfinite(report.relativeWeightError) ||
                report.relativeWeightError > 1.0e-11)
            {
                error =
                    "hybrid nuclear convolution did not conserve spectral weight";
                return false;
            }
            return true;
        }
    };
}

#endif
