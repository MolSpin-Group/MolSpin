#include "OpticalProfile.h"

#include "ObjectParser.h"
#include "Trajectory.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SpinAPI
{
    namespace
    {
        constexpr double h = 6.62607015e-34; // exact SI J s
        constexpr double c = 299792458.0;    // exact SI m/s
        constexpr double na = 6.02214076e23; // exact SI mol^-1

        std::string Trim(std::string s)
        {
            const auto first = std::find_if_not(s.begin(), s.end(), [](unsigned char x){ return std::isspace(x); });
            const auto last = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char x){ return std::isspace(x); }).base();
            return first >= last ? std::string() : std::string(first, last);
        }
        std::string Lower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char x){ return static_cast<char>(std::tolower(x)); });
            return s;
        }
        bool ReadNumber(const MSDParser::ObjectParser &p, const char *key,
            bool &present, double &value, std::string &error)
        {
            std::string raw;
            present = p.Get(key, raw);
            if (!present) return true;
            raw = Trim(raw);
            try
            {
                size_t used = 0;
                value = std::stod(raw, &used);
                if (used != raw.size() || !std::isfinite(value)) throw std::invalid_argument("invalid number");
                return true;
            }
            catch (const std::exception &)
            {
                error = std::string(key) + " must be a finite number";
                return false;
            }
        }
        bool TimeFactor(const std::string &unit, double &factor)
        {
            const std::string u = Lower(Trim(unit));
            if (u == "ps") factor = 1.0e-3;
            else if (u == "ns") factor = 1.0;
            else if (u == "us") factor = 1.0e3;
            else if (u == "ms") factor = 1.0e6;
            else if (u == "s") factor = 1.0e9;
            else return false;
            return true;
        }
        bool IrradianceFactor(const std::string &unit, double &factor)
        {
            // Prefix case is significant: MW is mega, mW is milli.
            if (unit == "W/m2" || unit == "W/m^2" || unit == "w/m2" || unit == "w/m^2") factor = 1.0;
            else if (unit == "W/cm2" || unit == "W/cm^2" || unit == "w/cm2" || unit == "w/cm^2") factor = 1.0e4;
            else if (unit == "mW/cm2" || unit == "mW/cm^2") factor = 10.0;
            else if (unit == "kW/cm2" || unit == "kW/cm^2" || unit == "kw/cm2" || unit == "kw/cm^2") factor = 1.0e7;
            else if (unit == "MW/cm2" || unit == "MW/cm^2") factor = 1.0e10;
            else return false;
            return true;
        }
        double Checked(double value, const char *what)
        {
            if (!std::isfinite(value) || value < 0.0)
                throw std::overflow_error(std::string(what) + " must be finite and non-negative");
            return value;
        }
        double Simpson(double a, double b, double fa, double fm, double fb)
        { return (b-a)*(fa+4.0*fm+fb)/6.0; }
        template<class F> double AdaptiveSimpson(const F &f, double a, double b,
            double fa, double fm, double fb, double whole, double tolerance, int depth)
        {
            const double m = a + 0.5*(b-a);
            const double lm = a + 0.5*(m-a), rm = m + 0.5*(b-m);
            if (lm == a || rm == m) return whole;
            const double flm = f(lm), frm = f(rm);
            const double left = Simpson(a,m,fa,flm,fm), right = Simpson(m,b,fm,frm,fb);
            const double delta = left+right-whole;
            if (std::abs(delta) <= 15.0*tolerance)
                return left+right+delta/15.0;
            if (depth == 0)
                throw std::runtime_error("experimental optical rate integral failed to converge");
            return AdaptiveSimpson(f,a,m,fa,flm,fm,left,0.5*tolerance,depth-1)+
                   AdaptiveSimpson(f,m,b,fm,frm,fb,right,0.5*tolerance,depth-1);
        }
    }

    OpticalWaveform::OpticalWaveform(std::string p, std::string col, std::string unit,
        std::vector<double> t, std::vector<double> i)
        : path(std::move(p)), column(std::move(col)), timeUnit(std::move(unit)),
          times(std::move(t)), intensities(std::move(i))
    {
        if (times.size() < 2 || times.size() != intensities.size())
            throw std::invalid_argument("optical waveform requires at least two paired samples");
        for (size_t n=0; n<times.size(); ++n)
        {
            if (!std::isfinite(times[n]) || !std::isfinite(intensities[n]) || intensities[n] < 0.0 ||
                (n && !(times[n] > times[n-1])))
                throw std::invalid_argument("optical waveform requires increasing finite times and non-negative finite intensity");
        }
    }
    double OpticalWaveform::Intensity(double time) const
    {
        if (!std::isfinite(time)) throw std::invalid_argument("optical profile time must be finite");
        if (time < times.front() || time > times.back()) return 0.0;
        if (time == times.front()) return intensities.front();
        if (time == times.back()) return intensities.back();
        const auto upper = std::upper_bound(times.begin(),times.end(),time);
        const size_t hi = static_cast<size_t>(upper-times.begin()), lo=hi-1;
        const double u=(time-times[lo])/(times[hi]-times[lo]);
        return (1.0-u)*intensities[lo]+u*intensities[hi];
    }
    double OpticalWaveform::PeakIntensity() const
    { return *std::max_element(intensities.begin(),intensities.end()); }

    bool OpticalWaveformCache::Load(const std::string &path, const std::string &column,
        const std::string &timeUnit, std::shared_ptr<const OpticalWaveform> &waveform,
        std::string &error)
    {
        waveform.reset(); error.clear();
        double timeFactor=0.0;
        if (!TimeFactor(timeUnit,timeFactor))
        { error="profiletimeunit must be ps, ns, us, ms, or s"; return false; }
        if (column.empty())
        { error="profilecolumn cannot be empty"; return false; }
        std::error_code ec;
        const auto canonical=std::filesystem::canonical(std::filesystem::path(path),ec);
        if (ec)
        { error="cannot resolve experimental optical profilefile \""+path+"\": "+ec.message(); return false; }
        std::ifstream input(canonical,std::ios::binary);
        if (!input)
        { error="cannot open experimental optical profilefile \""+canonical.string()+"\""; return false; }
        std::ostringstream buffer; buffer << input.rdbuf();
        if (input.bad())
        { error="failed to read experimental optical profilefile \""+canonical.string()+"\""; return false; }
        const std::string content=buffer.str();
        const std::string unit=Lower(Trim(timeUnit));
        const std::string key=canonical.string()+std::string(1,'\0')+column+std::string(1,'\0')+unit;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto fileFound=fileContents.find(canonical.string());
            if (fileFound!=fileContents.end() && fileFound->second!=content)
            { error="experimental optical profilefile changed during this calculation; rerun"; return false; }
            const auto found=entries.find(key);
            if (found!=entries.end())
            {
                if (found->second.content!=content)
                { error="experimental optical profilefile changed during this calculation; rerun"; return false; }
                waveform=found->second.waveform; return true;
            }
        }
        std::istringstream text(content);
        Trajectory table;
        if (!table.LoadStrict(text,error)) return false;
        unsigned int timeColumn=0, intensityColumn=0;
        if (!table.HasColumn("time",timeColumn))
        { error="experimental optical profilefile lacks time column"; return false; }
        if (!table.HasColumn(column,intensityColumn))
        { error="experimental optical profilefile lacks intensity column \""+column+"\""; return false; }
        if (timeColumn==intensityColumn)
        { error="profilecolumn must differ from time"; return false; }
        std::vector<double> times, intensities;
        times.reserve(table.Length()); intensities.reserve(table.Length());
        for (unsigned int row=0; row<table.Length(); ++row)
        {
            const double t=table.Get(row,timeColumn)*timeFactor;
            const double i=table.Get(row,intensityColumn);
            if (!std::isfinite(t) || (row && !(t>times.back())))
            { error="experimental optical times must be finite and strictly increasing after conversion to ns"; return false; }
            if (!std::isfinite(i) || i<0.0)
            { error="experimental optical intensity must be finite and non-negative"; return false; }
            times.push_back(t); intensities.push_back(i);
        }
        try
        {
            waveform=std::make_shared<const OpticalWaveform>(canonical.string(),column,unit,
                std::move(times),std::move(intensities));
        }
        catch (const std::exception &e)
        { error=e.what(); return false; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto fileFound=fileContents.find(canonical.string());
            if (fileFound!=fileContents.end() && fileFound->second!=content)
            { waveform.reset(); error="experimental optical profilefile changed during this calculation; rerun"; return false; }
            const auto found=entries.find(key);
            if (found!=entries.end())
            {
                if (found->second.content!=content)
                { waveform.reset(); error="experimental optical profilefile changed during this calculation; rerun"; return false; }
                waveform=found->second.waveform;
            }
            else entries[key]={content,waveform};
            fileContents.emplace(canonical.string(),content);
        }
        return true;
    }

    bool OpticalRateLaw::Parse(const MSDParser::ObjectParser &p, OpticalRateLaw &out,
        std::string &error)
    {
        error.clear();
        OpticalRateLaw law;
        std::string name, unit;
        if (!p.Get("ratelaw",name))
        { error="rateprofile=experimental requires ratelaw"; return false; }
        name=Lower(Trim(name));
        if (name=="absorption") law.kind=Kind::Absorption;
        else if (name=="linear") law.kind=Kind::Linear;
        else if (name=="powerlaw") law.kind=Kind::PowerLaw;
        else if (name=="saturating") law.kind=Kind::Saturating;
        else { error="unknown experimental optical ratelaw \""+name+"\""; return false; }
        law.name=name;
        if (!p.Get("intensityunit",unit))
        { error="rateprofile=experimental requires intensityunit"; return false; }
        unit=Trim(unit);
        if (!unit.empty() && (unit.front()=='"' || unit.front()=='\''))
        {
            if (unit.size()<2 || unit.back()!=unit.front())
            { error="intensityunit has unmatched quotes"; return false; }
            unit=unit.substr(1,unit.size()-2);
        }
        else if (!unit.empty() && (unit.back()=='"' || unit.back()=='\''))
        { error="intensityunit has unmatched quotes"; return false; }
        if (law.kind==Kind::Absorption)
        {
            if (!IrradianceFactor(unit,law.irradianceToWm2))
            { error="absorption requires intensityunit=W/m2, W/cm2, kW/cm2; in .msd, quote case-sensitive \"mW/cm2\" or \"MW/cm2\""; return false; }
            if (law.irradianceToWm2==1.0) unit="W/m2";
            else if (law.irradianceToWm2==1.0e4) unit="W/cm2";
            else if (law.irradianceToWm2==10.0) unit="mW/cm2";
            else if (law.irradianceToWm2==1.0e7) unit="kW/cm2";
            else unit="MW/cm2";
        }
        else if (Lower(unit)!="relative")
        { error="effective optical ratelaws require intensityunit=relative"; return false; }
        else unit="relative";
        law.intensityUnit=unit;

        bool scale=false, rate=false, power=false, half=false, wavelength=false, sigma=false, epsilon=false;
        double sigmaValue=0.0, epsilonValue=0.0;
        if (!ReadNumber(p,"intensityscale",scale,law.intensityScale,error) ||
            !ReadNumber(p,"ratescale",rate,law.ratescale,error) ||
            !ReadNumber(p,"order",power,law.order,error) ||
            !ReadNumber(p,"halfintensity",half,law.halfIntensity,error) ||
            !ReadNumber(p,"wavelength",wavelength,law.wavelengthNm,error) ||
            !ReadNumber(p,"crosssection",sigma,sigmaValue,error) ||
            !ReadNumber(p,"extinctioncoefficient",epsilon,epsilonValue,error)) return false;
        if (law.intensityScale<0.0)
        { error="intensityscale must be non-negative"; return false; }
        if (law.kind==Kind::Absorption)
        {
            if (rate || power || half)
            { error="absorption does not accept ratescale, order, or halfintensity"; return false; }
            if (!wavelength || !(law.wavelengthNm>0.0) || sigma==epsilon ||
                (sigma && !(sigmaValue>0.0)) || (epsilon && !(epsilonValue>0.0)))
            { error="absorption requires positive wavelength and exactly one positive crosssection or extinctioncoefficient"; return false; }
            law.crossSectionCm2=sigma ? sigmaValue : 1000.0*std::log(10.0)*epsilonValue/na;
            if (!std::isfinite(law.crossSectionCm2) || !(law.crossSectionCm2>0.0))
            { error="absorption cross-section conversion overflow or underflow"; return false; }
        }
        else
        {
            if (wavelength || sigma || epsilon)
            { error="effective optical ratelaws do not accept wavelength, crosssection, or extinctioncoefficient"; return false; }
            if (!rate || law.ratescale<0.0)
            { error="effective optical ratelaws require non-negative ratescale in ns^-1"; return false; }
            if (law.kind==Kind::PowerLaw)
            {
                if (!power || !(law.order>0.0) || half)
                { error="powerlaw requires positive order and does not accept halfintensity"; return false; }
            }
            else if (law.kind==Kind::Saturating)
            {
                if (!half || !(law.halfIntensity>0.0) || power)
                { error="saturating requires positive halfintensity and does not accept order"; return false; }
            }
            else if (power || half)
            { error="linear does not accept order or halfintensity"; return false; }
        }
        out=law;
        return true;
    }

    double OpticalRateLaw::Rate(double rawIntensity) const
    {
        Checked(rawIntensity,"optical intensity");
        const double intensity=Checked(rawIntensity*intensityScale,"scaled optical intensity");
        double rateValue=0.0;
        switch (kind)
        {
        case Kind::Absorption:
            // sigma[cm2]->m2, lambda[nm]->m, I[declared]->W/m2,
            // photon energy h*c/lambda, and s^-1 -> ns^-1.
            rateValue=(crossSectionCm2*1.0e-4)*
                Checked(intensity*irradianceToWm2,"irradiance in W/m2")*
                (wavelengthNm*1.0e-9)/(h*c)*1.0e-9;
            break;
        case Kind::Linear: rateValue=ratescale*intensity; break;
        case Kind::PowerLaw: rateValue=ratescale*std::pow(intensity,order); break;
        case Kind::Saturating:
            rateValue=intensity==0.0 ? 0.0 : ratescale/(1.0+halfIntensity/intensity);
            break;
        }
        return Checked(rateValue,"optical excitation rate");
    }

    bool OpticalRateLaw::ExactLinearSegmentIntegral(double rawStart,double rawEnd,
        double duration,double &action) const
    {
        if (kind==Kind::Saturating) return false;
        if (kind==Kind::Absorption || kind==Kind::Linear)
        {
            action=Checked(duration*(0.5*Rate(rawStart)+0.5*Rate(rawEnd)),
                "integrated optical rate");
            return true;
        }
        const double lo=std::min(rawStart,rawEnd), hi=std::max(rawStart,rawEnd);
        if (hi==lo)
        { action=Checked(duration*Rate(hi),"integrated optical rate"); return true; }
        const double peak=Rate(hi);
        const double u=1.0-lo/hi;
        if (u==0.0)
        { action=Checked(duration*peak,"integrated optical rate"); return true; }
        // Mean of x^order over a linearly varying non-negative x. Using
        // log1p/expm1 avoids cancellation when endpoint intensities are close.
        const double q=order+1.0;
        const double ratio=-std::expm1(q*std::log1p(-u))/(q*u);
        action=Checked((peak*ratio)*duration,"integrated optical rate");
        return true;
    }

    ExperimentalOpticalTimeProfile::ExperimentalOpticalTimeProfile(
        std::shared_ptr<const OpticalWaveform> w, OpticalRateLaw l)
        : waveform(std::move(w)), law(std::move(l)), peakRate(0.0)
    {
        if (!waveform) throw std::invalid_argument("experimental optical profile requires a waveform");
        peakRate=law.Rate(waveform->PeakIntensity());
        for (double i : waveform->Intensities()) (void)law.Rate(i);
        prefixRateArea.resize(waveform->Times().size(),0.0);
        for (size_t j=1; j<waveform->Times().size(); ++j)
            prefixRateArea[j]=Checked(prefixRateArea[j-1]+SegmentIntegral(
                waveform->Times()[j-1],waveform->Times()[j]),"cumulative optical rate action");
    }
    double ExperimentalOpticalTimeProfile::Intensity(double time) const
    { return waveform->Intensity(time); }
    double ExperimentalOpticalTimeProfile::Value(double time) const
    { return law.Rate(waveform->Intensity(time)); }
    double ExperimentalOpticalTimeProfile::ValueLeft(double time) const
    { return time==waveform->Times().front() ? 0.0 : Value(time); }
    double ExperimentalOpticalTimeProfile::ValueRight(double time) const
    { return time==waveform->Times().back() ? 0.0 : Value(time); }
    void ExperimentalOpticalTimeProfile::AppendBreakpoints(std::vector<double> &points) const
    { points.insert(points.end(),waveform->Times().begin(),waveform->Times().end()); }
    double ExperimentalOpticalTimeProfile::SegmentIntegral(double a,double b) const
    {
        if (a==b) return 0.0;
        double exact=0.0;
        if (law.ExactLinearSegmentIntegral(waveform->Intensity(a),waveform->Intensity(b),
            b-a,exact)) return exact;
        const double m=a+0.5*(b-a);
        const auto f=[this](double t){ return Value(t); };
        const double fa=f(a), fm=f(m), fb=f(b);
        const double whole=Simpson(a,b,fa,fm,fb);
        const double tolerance=1.0e-12*std::max(1.0,peakRate*(b-a));
        return Checked(AdaptiveSimpson(f,a,b,fa,fm,fb,whole,tolerance,24),
            "integrated optical rate");
    }
    double ExperimentalOpticalTimeProfile::PrefixAt(double time) const
    {
        const auto &times=waveform->Times();
        if (time<=times.front()) return 0.0;
        if (time>=times.back()) return prefixRateArea.back();
        const auto upper=std::upper_bound(times.begin(),times.end(),time);
        const size_t lo=static_cast<size_t>(upper-times.begin())-1;
        return Checked(prefixRateArea[lo]+SegmentIntegral(times[lo],time),
            "cumulative optical rate action");
    }
    double ExperimentalOpticalTimeProfile::Integral(double start,double end) const
    {
        if (!std::isfinite(start) || !std::isfinite(end))
            throw std::invalid_argument("optical rate integration bounds must be finite");
        if (start==end) return 0.0;
        const double sign=end>=start ? 1.0 : -1.0;
        const double lo=std::max(std::min(start,end),waveform->Times().front());
        const double hi=std::min(std::max(start,end),waveform->Times().back());
        if (!(hi>lo)) return 0.0;
        const auto &times=waveform->Times();
        const auto first=std::upper_bound(times.begin(),times.end(),lo);
        const auto second=std::lower_bound(times.begin(),times.end(),hi);
        if (first==second) return sign*SegmentIntegral(lo,hi);
        return sign*Checked(PrefixAt(hi)-PrefixAt(lo),"integrated optical rate");
    }
    std::string ExperimentalOpticalTimeProfile::Description() const
    {
        std::ostringstream out;
        out<<"experimental(file="<<waveform->Path()<<",column="<<waveform->Column()
           <<",timeunit="<<waveform->TimeUnit()<<",intensityunit="<<law.IntensityUnitName()
           <<",intensityscale="<<law.IntensityScale()
           <<",ratelaw="<<law.Name()<<",support_ns=["<<waveform->Times().front()
           <<","<<waveform->Times().back()<<"],peakrate_ns^-1="<<peakRate;
        if (law.Type()==OpticalRateLaw::Kind::Absorption)
            out<<",wavelength_nm="<<law.WavelengthNm()
               <<",crosssection_cm2="<<law.AbsorptionCrossSectionCm2()
               <<",irradiance_to_W_m2="<<law.IrradianceToWm2();
        out<<")";
        return out.str();
    }
}
