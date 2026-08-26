/////////////////////////////////////////////////////////////////////////
// TimeProfile (SpinAPI Module)
// ------------------
// Immutable scalar time profiles used by reusable physical channels.
//
// HIERARCHY CONTRACT
//   SpinAPI::TimeProfile is a low-level physics primitive.  It knows nothing
//   about RunSection tasks, direct-sum reaction networks, output formatting,
//   or legacy task classes.  RunSection::General::MultiSS composes a
//   TimeProfile with TransferChannel objects; TaskMultiSSGeneral only
//   orchestrates that higher layer.
//
// PHYSICAL SCOPE
//   A TimeProfile describes a *classical scalar coefficient* k(t).  In the
//   MultiSS optical use case this is appropriate for incoherent optical
//   pumping / population transfer after optical coherences have been removed
//   from the model.  A coherent resonant laser pulse must instead enter a
//   Hamiltonian through an electric-field/Rabi envelope and requires the
//   relevant optical states (and their coherences) in one Hilbert space.
//
//   Finite Gaussian time-dependent optical rates are standard in level-rate
//   descriptions of solid-state emitters; an explicit NV-centre example uses
//   a Gaussian temporal pulse profile in the optical transition rate:
//       DOI: 10.1038/ncomms14000
//
// Molecular Spin Dynamics Software.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_TimeProfile
#define MOD_SpinAPI_TimeProfile

#include <memory>
#include <string>
#include <vector>

namespace SpinAPI
{
    enum class TimeProfileKind
    {
        Constant,
        Gaussian,
        Rectangular,
        Trajectory
    };

    class TimeProfile
    {
    public:
        virtual ~TimeProfile() = default;
        virtual double Value(double _time) const = 0;
        virtual double Integral(double _start, double _end) const = 0;
        virtual bool IsConstant() const { return false; }
        virtual TimeProfileKind Kind() const = 0;
        virtual std::string Description() const = 0;
    };

    using time_profile_ptr = std::shared_ptr<const TimeProfile>;

    class ConstantTimeProfile final : public TimeProfile
    {
    public:
        explicit ConstantTimeProfile(double _value);
        double Value(double _time) const override;
        double Integral(double _start, double _end) const override;
        bool IsConstant() const override { return true; }
        TimeProfileKind Kind() const override { return TimeProfileKind::Constant; }
        std::string Description() const override;
        double ConstantValue() const { return value; }
    private:
        double value;
    };

    class GaussianTimeProfile final : public TimeProfile
    {
    public:
        // peak * exp[-4 ln(2) (t-center)^2 / FWHM^2]
        GaussianTimeProfile(double _center, double _fwhm, double _peak);
        double Value(double _time) const override;
        double Integral(double _start, double _end) const override;
        TimeProfileKind Kind() const override { return TimeProfileKind::Gaussian; }
        std::string Description() const override;

        double Center() const { return center; }
        double FWHM() const { return fwhm; }
        double Peak() const { return peak; }
        double FullArea() const;

        // For a one-way whole-source process dp/dt=-k(t)p, a desired
        // asymptotic transferred fraction f obeys f=1-exp[-integral k(t)dt].
        // This helper converts (f,FWHM) into the required Gaussian peak rate.
        static double PeakForTransferredFraction(double _fraction, double _fwhm);
    private:
        double center;
        double fwhm;
        double peak;
    };

    class RectangularTimeProfile final : public TimeProfile
    {
    public:
        RectangularTimeProfile(double _start, double _end, double _value);
        double Value(double _time) const override;
        double Integral(double _start, double _end) const override;
        TimeProfileKind Kind() const override { return TimeProfileKind::Rectangular; }
        std::string Description() const override;
    private:
        double start;
        double end;
        double value;
    };

    class TrajectoryTimeProfile final : public TimeProfile
    {
    public:
        // Piecewise-linear interpolation.  Outside the tabulated time range
        // the end values are held constant, matching the established MolSpin
        // trajectory convention but without mutating a Transition object.
        TrajectoryTimeProfile(std::vector<double> _times, std::vector<double> _values);
        double Value(double _time) const override;
        double Integral(double _start, double _end) const override;
        TimeProfileKind Kind() const override { return TimeProfileKind::Trajectory; }
        std::string Description() const override;
        const std::vector<double> &Times() const { return times; }
        const std::vector<double> &Values() const { return values; }
    private:
        std::vector<double> times;
        std::vector<double> values;
    };
}

#endif
