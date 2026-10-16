#ifndef MOD_SpinAPI_OpticalProfile
#define MOD_SpinAPI_OpticalProfile

#include "TimeProfile.h"
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace MSDParser { class ObjectParser; }

namespace SpinAPI
{
    // Raw measured samples in MolSpin's ns time unit. Rates and molecular
    // parameters never live here, so independently calibrated transitions may
    // share the same immutable waveform.
    class OpticalWaveform
    {
    public:
        OpticalWaveform(std::string path, std::string column, std::string timeUnit,
            std::vector<double> times, std::vector<double> intensities);
        double Intensity(double time) const;
        const std::string &Path() const { return path; }
        const std::string &Column() const { return column; }
        const std::string &TimeUnit() const { return timeUnit; }
        const std::vector<double> &Times() const { return times; }
        const std::vector<double> &Intensities() const { return intensities; }
        double PeakIntensity() const;
    private:
        std::string path, column, timeUnit;
        std::vector<double> times, intensities;
    };

    class OpticalWaveformCache
    {
    public:
        // Every call checks the current file bytes. A changed file is rejected
        // within one run, so no graph mixes old and new waveform versions.
        bool Load(const std::string &path, const std::string &column,
            const std::string &timeUnit,
            std::shared_ptr<const OpticalWaveform> &waveform, std::string &error);
    private:
        struct Entry { std::string content; std::shared_ptr<const OpticalWaveform> waveform; };
        std::mutex mutex;
        std::map<std::string, Entry> entries;
        std::map<std::string, std::string> fileContents;
    };

    // Central, extensible portfolio of calibrated intensity-to-rate laws.
    class OpticalRateLaw
    {
    public:
        enum class Kind { Absorption, Linear, PowerLaw, Saturating };
        static bool Parse(const MSDParser::ObjectParser &properties,
            OpticalRateLaw &law, std::string &error);
        double Rate(double rawIntensity) const;
        // Exact action for an absorption, linear, or power-law response over
        // one interval of linearly interpolated intensity. Returns false for
        // laws that use the adaptive quadrature fallback.
        bool ExactLinearSegmentIntegral(double rawStart, double rawEnd,
            double duration, double &action) const;
        Kind Type() const { return kind; }
        const std::string &Name() const { return name; }
        const std::string &IntensityUnitName() const { return intensityUnit; }
        double WavelengthNm() const { return wavelengthNm; }
        double AbsorptionCrossSectionCm2() const { return crossSectionCm2; }
        double IntensityScale() const { return intensityScale; }
        double IrradianceToWm2() const { return irradianceToWm2; }
    private:
        Kind kind = Kind::Linear;
        std::string name = "linear";
        std::string intensityUnit = "relative";
        double intensityScale = 1.0;
        double ratescale = 0.0;
        double order = 1.0;
        double halfIntensity = 1.0;
        double wavelengthNm = 0.0;
        double crossSectionCm2 = 0.0;
        double irradianceToWm2 = 0.0;
    };

    class ExperimentalOpticalTimeProfile final : public TimeProfile
    {
    public:
        ExperimentalOpticalTimeProfile(std::shared_ptr<const OpticalWaveform> waveform,
            OpticalRateLaw law);
        double Intensity(double time) const;
        double Value(double time) const override;
        double ValueLeft(double time) const override;
        double ValueRight(double time) const override;
        double Integral(double start, double end) const override;
        bool IsConstant() const override { return peakRate==0.0; }
        void AppendBreakpoints(std::vector<double> &points) const override;
        TimeProfileKind Kind() const override { return TimeProfileKind::ExperimentalOptical; }
        std::string Description() const override;
        double PeakRate() const { return peakRate; }
        const std::shared_ptr<const OpticalWaveform> &Waveform() const { return waveform; }
        const std::string &RateLawName() const { return law.Name(); }
        const std::string &IntensityUnitName() const { return law.IntensityUnitName(); }
        double IntensityScale() const { return law.IntensityScale(); }
        double IrradianceToWm2() const { return law.IrradianceToWm2(); }
        double WavelengthNm() const { return law.WavelengthNm(); }
        double AbsorptionCrossSectionCm2() const { return law.AbsorptionCrossSectionCm2(); }
    private:
        double SegmentIntegral(double start, double end) const;
        double PrefixAt(double time) const;
        std::shared_ptr<const OpticalWaveform> waveform;
        OpticalRateLaw law;
        double peakRate;
        std::vector<double> prefixRateArea;
    };
}
#endif
