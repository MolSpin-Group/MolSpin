/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Field-domain channels, accumulation and detection harmonics.
#ifndef MOLSPIN_RESONANCESPECTRUMPROCESSING_H
#define MOLSPIN_RESONANCESPECTRUMPROCESSING_H
#include "ResonanceExecutionPlan.h"
#include "ResonanceTypes.h"
#include <vector>
#include <string>
#include <iosfwd>

namespace RunSection::General::Resonance
{
    struct ResonanceSpectrum
    {
        unsigned int steps = 0;
        std::vector<double> field_mT;
        std::vector<double> total_x;
        std::vector<double> total_y;
        std::vector<double> total_perp;
        std::vector<double> cross_x;
        std::vector<double> cross_y;
        std::vector<std::string> spin_names;
        std::vector<std::vector<double>> spin_x;
        std::vector<std::vector<double>> spin_y;
        std::vector<std::vector<double>> spin_perp;
        std::vector<std::vector<double>> spin_p;
        std::vector<std::vector<double>> spin_m;
    };
    struct ResonanceSample
    {
        double field_mT = 0;
        SpectrumPoint point;
        std::vector<std::string> spin_names;
    };
    class ResonanceSpectrumProcessing
    {
      public:
        static double LineshapeValue(const ResonanceExecutionPlan &plan, double delta, double width);
        static std::vector<double> ApplyFieldHarmonic(const ResonanceExecutionPlan &plan,
                                                      const std::vector<double> &field,
                                                      const std::vector<double> &channel);
        static void ApplyDetectionHarmonic(const ResonanceExecutionPlan &plan, ResonanceSpectrum &spectrum);
        static void Accumulate(SpectrumPoint &total, const SpectrumPoint &point, double weight);
        static ResonanceSample Sample(const ResonanceSpectrum &spectrum, size_t index);
        static void WriteHeader(const std::string &system, const std::vector<std::string> &spins,
                                std::ostream &out);
        static void WriteSample(const ResonanceSample &sample, std::ostream &out);
    };

} // namespace RunSection::General::Resonance
#endif
