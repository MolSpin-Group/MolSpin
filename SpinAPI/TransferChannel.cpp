/////////////////////////////////////////////////////////////////////////
// TransferChannel implementation (SpinAPI Module)
// ----------------------------------------------------------------------
// ROLE IN THE NEW HIERARCHY
//   parsed SpinAPI::Transition
//       -> immutable SpinAPI::TransferChannel            [this file]
//       -> RunSection::General::MultiSS::MultiSSNetworkBuilder
//       -> TaskMultiSSGeneral.
//
//   TransferChannel is reusable physics, not a task. Existing
//   RunSection/Tasks implementations are intentionally untouched and remain
//   independent numerical references during migration.
//
// OPEN-SYSTEM PHYSICS
//   A kinetic transfer channel is represented by rectangular maps
//   C_mu : H_source -> H_target.  With G=sum C_mu^dagger C_mu,
//
//       d rho_source/dt = -k(t)/2 {G,rho_source},
//       d rho_target/dt =  k(t) sum_mu C_mu rho_source C_mu^dagger.
//
//   This is the block-diagonal restriction of a completely-positive Markovian
//   transfer process.  The general semigroup/GKSL structure is grounded in:
//       DOI: 10.1007/BF01608499
//       DOI: 10.1063/1.522979
//
// PRESERVED-SPIN SEMANTICS
//   `preservespins` means exactly the opposite of "spins driven by the pulse".
//   Each listed source/target spin pair is transported by an identity map.  If
//   N is preserved, the intended structure is
//
//       C = C_active (x) I_N,
//
//   so arbitrary N populations *and coherences* survive transfer.  This is the
//   appropriate memory-preserving kinetic structure for reversible S1/CSS
//   models such as DOI: 10.1039/D6CP00916F.  Replacing it with one independent
//   Transition per m_I value would in general prepare a mixture and destroy
//   pre-existing nuclear coherence.
//
// IMPORTANT CONVENTIONS
//   * Existing Transition is parsed once, then compiled; the compiled channel
//     never relies on mutating Transition::SetTime().
//   * The rank-one path is used only when source/target supports are genuinely
//     pure.  Omitted preserved spins must use `preservespins` so they become an
//     identity sector rather than an equal-amplitude pure ket.
//   * The deterministic eigenvector phase convention is essential: arbitrary
//     independent phases in preserved sectors would create a diagonal phase
//     operation instead of I_preserved and would corrupt coherences.
//   * BuildSourceLossSuperoperator / BuildTargetGainSuperoperator return
//     *unit-rate* blocks.  General/MultiSS multiplies them by k(t) exactly once.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "TransferChannel.h"

#include "ObjectParser.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace SpinAPI
{
    namespace
    {
        std::string Lower(std::string value)
        {
            std::transform(value.begin(), value.end(), value.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return value;
        }

        bool FiniteNonNegative(double value) { return std::isfinite(value) && value >= 0.0; }

        bool BuildProjector(const system_ptr &system, const state_ptr &state,
            arma::cx_mat &projector, std::string &error)
        {
            if (system == nullptr || state == nullptr)
            {
                error = "transfer channel requires a non-null source State";
                return false;
            }
            SpinSpace space(system);
            space.UseSuperoperatorSpace(false);
            if (!space.GetState(state, projector))
            {
                error = "failed to construct State projector \"" + state->Name() + "\" in SpinSystem \"" + system->Name() + "\"";
                return false;
            }
            projector = 0.5 * (projector + projector.t());
            return true;
        }

        bool RankOneVector(const arma::cx_mat &projector, arma::cx_vec &vector,
            std::string &error, double tolerance)
        {
            if (projector.n_rows == 0 || projector.n_rows != projector.n_cols)
            {
                error = "rank-one State support has invalid dimensions";
                return false;
            }
            arma::vec eigval;
            arma::cx_mat eigvec;
            if (!arma::eig_sym(eigval, eigvec, projector))
            {
                error = "failed to diagonalize a State support projector";
                return false;
            }
            const double maxEval = eigval.max();
            if (!(maxEval > tolerance))
            {
                error = "State support projector has zero rank";
                return false;
            }
            arma::uword rank = 0;
            for (double value : eigval)
                if (value > tolerance * std::max(1.0, maxEval)) ++rank;
            if (rank != 1)
            {
                std::ostringstream out;
                out << "transfer State support has rank " << rank
                    << "; use preservespins for an identity-preserved omitted spin sector or define a pure active State";
                error = out.str();
                return false;
            }
            const arma::uword index = eigval.index_max();
            vector = eigvec.col(index);
            const double norm = arma::norm(vector, 2);
            if (!(norm > tolerance))
            {
                error = "rank-one State vector has zero norm";
                return false;
            }
            vector /= norm;

            // Deterministic phase convention.  This matters when the same
            // active ket is copied across several preserved-spin basis sectors:
            // independent arbitrary eigensolver phases would otherwise turn
            // I_preserved into an unintended diagonal phase operator and alter
            // preserved coherences.
            arma::uword pivot = 0;
            double pivotAbs = 0.0;
            for (arma::uword i = 0; i < vector.n_elem; ++i)
            {
                const double a = std::abs(vector(i));
                if (a > pivotAbs) { pivotAbs = a; pivot = i; }
            }
            if (pivotAbs > tolerance)
                vector /= vector(pivot) / pivotAbs;
            return true;
        }

        std::vector<spin_ptr> SpinSpaceBasis(const system_ptr &system)
        {
            // SpinSpace(system_ptr) adds a collection through SpinSpace::Add,
            // which sorts the spin pointers.  State projectors and all General
            // SS/MultiSS matrices therefore use this order, not the insertion
            // order returned by SpinSystem::Spins().  Tensor-index decoding
            // must follow the matrix basis exactly.
            auto result = system->Spins();
            std::sort(result.begin(), result.end());
            return result;
        }

        std::vector<unsigned int> Multiplicities(const std::vector<spin_ptr> &spins)
        {
            std::vector<unsigned int> result;
            for (const auto &spin : spins)
                result.push_back(static_cast<unsigned int>(spin->Multiplicity()));
            return result;
        }

        bool SpinPosition(const std::vector<spin_ptr> &spins, const std::string &name,
            size_t &position, unsigned int &multiplicity)
        {
            for (size_t i = 0; i < spins.size(); ++i)
            {
                if (spins[i] != nullptr && spins[i]->Name() == name)
                {
                    position = i;
                    multiplicity = static_cast<unsigned int>(spins[i]->Multiplicity());
                    return true;
                }
            }
            return false;
        }

        std::vector<unsigned int> DecodeIndex(arma::uword index,
            const std::vector<unsigned int> &multiplicities)
        {
            std::vector<unsigned int> digits(multiplicities.size(), 0);
            for (size_t r = multiplicities.size(); r-- > 0; )
            {
                digits[r] = static_cast<unsigned int>(index % multiplicities[r]);
                index /= multiplicities[r];
            }
            return digits;
        }

        std::vector<unsigned int> DecodeTuple(arma::uword index,
            const std::vector<unsigned int> &multiplicities)
        {
            return DecodeIndex(index, multiplicities);
        }

        bool ParsePreservedMap(const transition_ptr &transition,
            std::vector<std::pair<std::string, std::string>> &mapping,
            std::string &error)
        {
            mapping.clear();
            std::vector<std::string> entries;
            const auto props = transition->Properties();
            if (props == nullptr || !props->GetList("preservespins", entries, ','))
                return true;

            for (std::string item : entries)
            {
                item.erase(std::remove_if(item.begin(), item.end(),
                    [](unsigned char c) { return std::isspace(c); }), item.end());
                if (item.empty()) continue;
                const size_t colon = item.find(':');
                if (colon == std::string::npos)
                    mapping.emplace_back(item, item);
                else
                {
                    if (item.find(':', colon + 1) != std::string::npos || colon == 0 || colon + 1 == item.size())
                    {
                        error = "preservespins entries must be NAME or SOURCE_NAME:TARGET_NAME";
                        return false;
                    }
                    mapping.emplace_back(item.substr(0, colon), item.substr(colon + 1));
                }
            }
            return true;
        }

        bool BuildPreservedIsometry(const system_ptr &sourceSystem,
            const state_ptr &sourceState, const system_ptr &targetSystem,
            const state_ptr &targetState,
            const std::vector<std::pair<std::string, std::string>> &mapping,
            arma::sp_cx_mat &jump, arma::sp_cx_mat &sourceEffect,
            std::string &error, double tolerance)
        {
            arma::cx_mat Ps, Pt;
            if (!BuildProjector(sourceSystem, sourceState, Ps, error) ||
                !BuildProjector(targetSystem, targetState, Pt, error)) return false;

            const auto srcSpins = SpinSpaceBasis(sourceSystem);
            const auto dstSpins = SpinSpaceBasis(targetSystem);
            const auto srcMult = Multiplicities(srcSpins);
            const auto dstMult = Multiplicities(dstSpins);
            const arma::uword ds = Ps.n_rows;
            const arma::uword dt = Pt.n_rows;

            std::vector<size_t> srcPos, dstPos;
            std::vector<unsigned int> preservedMult;
            for (const auto &pair : mapping)
            {
                size_t ps = 0, pt = 0;
                unsigned int ms = 0, mt = 0;
                if (!SpinPosition(srcSpins, pair.first, ps, ms))
                {
                    error = "preservespins source spin \"" + pair.first + "\" was not found in SpinSystem \"" + sourceSystem->Name() + "\"";
                    return false;
                }
                if (!SpinPosition(dstSpins, pair.second, pt, mt))
                {
                    error = "preservespins target spin \"" + pair.second + "\" was not found in SpinSystem \"" + targetSystem->Name() + "\"";
                    return false;
                }
                if (ms != mt)
                {
                    error = "preservespins requires equal source/target multiplicities for " + pair.first + ":" + pair.second;
                    return false;
                }
                if (std::find(srcPos.begin(), srcPos.end(), ps) != srcPos.end() ||
                    std::find(dstPos.begin(), dstPos.end(), pt) != dstPos.end())
                {
                    error = "preservespins contains a duplicate source or target spin";
                    return false;
                }
                srcPos.push_back(ps); dstPos.push_back(pt); preservedMult.push_back(ms);
            }

            arma::uword preservedDimension = 1;
            for (unsigned int m : preservedMult) preservedDimension *= m;
            arma::cx_mat C(dt, ds, arma::fill::zeros);

            for (arma::uword tupleIndex = 0; tupleIndex < preservedDimension; ++tupleIndex)
            {
                const auto tuple = DecodeTuple(tupleIndex, preservedMult);
                std::vector<arma::uword> srcIndices, dstIndices;
                for (arma::uword i = 0; i < ds; ++i)
                {
                    const auto digits = DecodeIndex(i, srcMult);
                    bool match = true;
                    for (size_t k = 0; k < srcPos.size(); ++k)
                        if (digits[srcPos[k]] != tuple[k]) { match = false; break; }
                    if (match) srcIndices.push_back(i);
                }
                for (arma::uword i = 0; i < dt; ++i)
                {
                    const auto digits = DecodeIndex(i, dstMult);
                    bool match = true;
                    for (size_t k = 0; k < dstPos.size(); ++k)
                        if (digits[dstPos[k]] != tuple[k]) { match = false; break; }
                    if (match) dstIndices.push_back(i);
                }
                if (srcIndices.empty() || dstIndices.empty())
                {
                    error = "failed to enumerate a preserved-spin basis sector";
                    return false;
                }

                arma::uvec is(srcIndices.size()), it(dstIndices.size());
                for (size_t k = 0; k < srcIndices.size(); ++k) is(k) = srcIndices[k];
                for (size_t k = 0; k < dstIndices.size(); ++k) it(k) = dstIndices[k];
                arma::cx_vec vs, vt;
                if (!RankOneVector(Ps.submat(is, is), vs, error, tolerance))
                {
                    error = "preservespins source sector " + std::to_string(tupleIndex) +
                        " is not a pure active-state support: " + error;
                    return false;
                }
                if (!RankOneVector(Pt.submat(it, it), vt, error, tolerance))
                {
                    error = "preservespins target sector " + std::to_string(tupleIndex) +
                        " is not a pure active-state support: " + error;
                    return false;
                }

                arma::cx_vec fullS(ds, arma::fill::zeros), fullT(dt, arma::fill::zeros);
                for (size_t k = 0; k < srcIndices.size(); ++k) fullS(srcIndices[k]) = vs(k);
                for (size_t k = 0; k < dstIndices.size(); ++k) fullT(dstIndices[k]) = vt(k);
                C += fullT * fullS.t();
            }

            const arma::cx_mat Gs = C.t() * C;
            const arma::cx_mat Gt = C * C.t();
            const double srcScale = std::max(1.0, arma::norm(Ps, "fro"));
            const double dstScale = std::max(1.0, arma::norm(Pt, "fro"));
            if (arma::norm(Gs - Ps, "fro") > 50.0 * tolerance * srcScale ||
                arma::norm(Gt - Pt, "fro") > 50.0 * tolerance * dstScale)
            {
                error = "preservespins map failed C^dagger C=P_source / C C^dagger=P_target consistency; check omitted State spins and mapping";
                return false;
            }

            jump = arma::sp_cx_mat(C);
            sourceEffect = arma::sp_cx_mat(Gs);
            return true;
        }

        bool BuildRankOneIsometry(const system_ptr &sourceSystem, const state_ptr &sourceState,
            const system_ptr &targetSystem, const state_ptr &targetState,
            arma::sp_cx_mat &jump, arma::sp_cx_mat &sourceEffect,
            std::string &error, double tolerance)
        {
            arma::cx_mat Ps, Pt;
            if (!BuildProjector(sourceSystem, sourceState, Ps, error) ||
                !BuildProjector(targetSystem, targetState, Pt, error)) return false;
            arma::cx_vec s, t;
            if (!RankOneVector(Ps, s, error, tolerance) || !RankOneVector(Pt, t, error, tolerance))
                return false;
            const arma::cx_mat C = t * s.t();
            jump = arma::sp_cx_mat(C);
            sourceEffect = arma::sp_cx_mat(C.t() * C);
            return true;
        }

        bool CompileProfile(const transition_ptr &transition, time_profile_ptr &profile,
            bool &instantaneous, double &eventTime, double &eventFraction, std::string &error)
        {
            instantaneous = false; eventTime = 0.0; eventFraction = 0.0;
            const auto props = transition->Properties();
            std::string kind = "constant";
            if (props != nullptr)
            {
                if (!(props->Get("rateprofile", kind) || props->Get("rate_profile", kind) ||
                    props->Get("opticalprofile", kind))) kind = "constant";
            }
            kind = Lower(kind);

            try
            {
                if (kind == "constant" || kind == "static")
                {
                    if (!FiniteNonNegative(transition->Rate()))
                    { error = "constant transfer rate must be finite and non-negative"; return false; }
                    profile = std::make_shared<ConstantTimeProfile>(transition->Rate());
                }
                else if (kind == "gaussian")
                {
                    double center = 0.0, fwhm = 0.0, peak = transition->Rate();
                    bool haveCenter = props && (props->Get("pulsecenter", center) || props->Get("center", center));
                    bool haveFwhm = props && (props->Get("pulsefwhm", fwhm) || props->Get("fwhm", fwhm));
                    bool havePeak = props && (props->Get("peakrate", peak) || props->Get("peak_rate", peak));
                    double fraction = 0.0;
                    bool haveFraction = props && (props->Get("transferfraction", fraction) || props->Get("transfer_fraction", fraction));
                    if (!haveCenter || !haveFwhm)
                    { error = "rateprofile=gaussian requires pulsecenter and pulsefwhm"; return false; }
                    if (havePeak && haveFraction)
                    { error = "Gaussian profile must specify either peakrate or transferfraction, not both"; return false; }
                    if (haveFraction)
                        peak = GaussianTimeProfile::PeakForTransferredFraction(fraction, fwhm);
                    profile = std::make_shared<GaussianTimeProfile>(center, fwhm, peak);
                }
                else if (kind == "rectangular" || kind == "square")
                {
                    double start = 0.0, end = 0.0, value = transition->Rate();
                    if (!(props && (props->Get("pulsestart", start) || props->Get("start", start))) ||
                        !(props && (props->Get("pulseend", end) || props->Get("end", end))))
                    { error = "rateprofile=rectangular requires pulsestart and pulseend"; return false; }
                    if (props) { props->Get("peakrate", value); props->Get("peak_rate", value); }
                    profile = std::make_shared<RectangularTimeProfile>(start, end, value);
                }
                else if (kind == "trajectory")
                {
                    std::vector<double> times, rates;
                    if (!(props && props->GetList("profiletimes", times, ',')) ||
                        !(props && props->GetList("profilerates", rates, ',')))
                    { error = "rateprofile=trajectory requires profiletimes and profilerates lists"; return false; }
                    profile = std::make_shared<TrajectoryTimeProfile>(times, rates);
                }
                else if (kind == "instantaneous" || kind == "event" || kind == "delta")
                {
                    if (!(props && (props->Get("eventtime", eventTime) || props->Get("pulsetime", eventTime))))
                    { error = "rateprofile=instantaneous requires eventtime"; return false; }
                    if (!(props && (props->Get("transferfraction", eventFraction) || props->Get("transfer_fraction", eventFraction))))
                    { error = "rateprofile=instantaneous requires transferfraction"; return false; }
                    if (!std::isfinite(eventTime) || eventTime < 0.0 ||
                        !std::isfinite(eventFraction) || eventFraction < 0.0 || eventFraction > 1.0)
                    { error = "instantaneous event requires eventtime>=0 and 0<=transferfraction<=1"; return false; }
                    instantaneous = true;
                    profile = std::make_shared<ConstantTimeProfile>(0.0);
                }
                else
                {
                    error = "unknown rateprofile \"" + kind + "\"";
                    return false;
                }
            }
            catch (const std::exception &e)
            {
                error = e.what();
                return false;
            }
            return true;
        }
    }

    bool TransferChannel::Compile(const transition_ptr &_transition,
        TransferChannel &_channel, std::string &_error, double _tolerance)
    {
        _channel = TransferChannel();
        _error.clear();
        if (_transition == nullptr || _transition->System() == nullptr || _transition->SourceState() == nullptr)
        {
            _error = "cannot compile a null/incomplete Transition as a TransferChannel";
            return false;
        }
        if (!std::isfinite(_tolerance) || !(_tolerance > 0.0))
        {
            _error = "TransferChannel tolerance must be finite and positive";
            return false;
        }

        _channel.transition = _transition;
        _channel.sourceSystem = _transition->System();
        _channel.targetSystem = _transition->Target();
        _channel.sourceState = _transition->SourceState();
        _channel.targetState = _transition->TargetState();

        if (!CompileProfile(_transition, _channel.profile, _channel.instantaneous,
            _channel.eventTime, _channel.eventFraction, _error)) return false;
        if (!ParsePreservedMap(_transition, _channel.preservedSpinMap, _error)) return false;

        arma::cx_mat sourceProjector;
        if (!BuildProjector(_channel.sourceSystem, _channel.sourceState, sourceProjector, _error)) return false;

        if (_channel.targetSystem == nullptr)
        {
            // Terminal sink: no target gain is represented.  A source projector
            // may have arbitrary rank; its Haberkorn/GKSL no-jump loss is still
            // unambiguous.  preservespins has no meaning for a terminal sink.
            if (!_channel.preservedSpinMap.empty())
            {
                _error = "preservespins is only meaningful for a Transition with a targetsystem";
                return false;
            }
            _channel.sourceEffect = arma::sp_cx_mat(sourceProjector);
            return true;
        }

        if (_channel.targetState == nullptr)
        {
            _error = "targeted TransferChannel requires targetstate";
            return false;
        }
        if (_channel.instantaneous && !_channel.HasTarget())
        {
            _error = "instantaneous transfer events currently require a represented target manifold";
            return false;
        }

        arma::sp_cx_mat jump;
        if (_channel.preservedSpinMap.empty())
        {
            if (!BuildRankOneIsometry(_channel.sourceSystem, _channel.sourceState,
                _channel.targetSystem, _channel.targetState, jump,
                _channel.sourceEffect, _error, _tolerance)) return false;
        }
        else
        {
            if (!BuildPreservedIsometry(_channel.sourceSystem, _channel.sourceState,
                _channel.targetSystem, _channel.targetState,
                _channel.preservedSpinMap, jump, _channel.sourceEffect,
                _error, _tolerance)) return false;
        }
        _channel.kraus.push_back(std::move(jump));
        return true;
    }

    bool TransferChannel::BuildSourceLossSuperoperator(arma::sp_cx_mat &_loss) const
    {
        if (sourceSystem == nullptr || sourceEffect.n_rows == 0) return false;
        const arma::uword d = sourceEffect.n_rows;
        const arma::sp_cx_mat I = arma::speye<arma::sp_cx_mat>(d, d);
        // Row-major MolSpin vectorization: left multiplication is kron(G,I),
        // right multiplication is kron(I,G^T).  G is Hermitian but we retain
        // the explicit simple-transpose convention used by SpinSpace.
        _loss = 0.5 * (arma::kron(sourceEffect, I) + arma::kron(I, sourceEffect.st()));
        return true;
    }

    bool TransferChannel::BuildTargetGainSuperoperator(arma::sp_cx_mat &_gain) const
    {
        if (!HasTarget() || kraus.empty()) return false;
        const arma::uword dt = kraus.front().n_rows;
        const arma::uword ds = kraus.front().n_cols;
        _gain.zeros(dt * dt, ds * ds);
        for (const auto &C : kraus)
        {
            // vec_row(C rho C^dagger) = (C kron C*) vec_row(rho).
            _gain += arma::kron(C, arma::conj(C));
        }
        return true;
    }
}
