#ifndef MOLSPIN_RESONANCE_FIELD_ROOTS_H
#define MOLSPIN_RESONANCE_FIELD_ROOTS_H

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace RunSection::General::Resonance
{
    struct ResonanceFieldRoot
    {
        double fieldT;
        arma::uword lower, upper;
    };

    // Locate isolated gap crossings for a complete affine Hamiltonian.
    // Ordered eigenvalue gaps are Lipschitz continuous, with bound equal to
    // the spectral diameter of dH/dB. This permits adaptive exclusion of
    // intervals even when both endpoints have the same detuning sign: a
    // narrow pair of crossings at an avoided crossing must not be missed.
    // A tangency at exactly zero slope is not a finite-weight field line;
    // that limiting problem requires finite frequency-domain broadening.
    class ResonanceFieldRoots
    {
    public:
        static bool Locate(const arma::cx_mat &h0, const arma::cx_mat &dHdB,
            double fieldMin, double fieldMax, double omega,
            std::vector<ResonanceFieldRoot> &roots, std::string &error,
            double fieldTolerance = 1e-9)
        {
            roots.clear(); error.clear();
            if (h0.n_rows < 2 || h0.n_rows != h0.n_cols ||
                dHdB.n_rows != h0.n_rows || dHdB.n_cols != h0.n_cols ||
                !h0.is_finite() || !dHdB.is_finite() ||
                !std::isfinite(fieldMin) || !std::isfinite(fieldMax) ||
                !(fieldMax > fieldMin) || !(fieldTolerance > 0) ||
                !std::isfinite(fieldTolerance) || !(omega > 0) || !std::isfinite(omega))
            { error = "invalid affine resonance-root request"; return false; }
            if (arma::norm(h0-h0.t(),"fro") > 1e-11*std::max(1.,arma::norm(h0,"fro")) ||
                arma::norm(dHdB-dHdB.t(),"fro") > 1e-11*std::max(1.,arma::norm(dHdB,"fro")))
            { error = "resonance-root Hamiltonian must be Hermitian"; return false; }
            arma::vec derivativeEigenvalues;
            if (!arma::eig_sym(derivativeEigenvalues,dHdB))
            { error = "failed to bound resonance-gap slopes"; return false; }
            const double bound = derivativeEigenvalues.max()-derivativeEigenvalues.min();
            if (bound <= 0) return true;
            std::size_t evaluations = 0;
            auto energies = [&](double b, arma::vec &e) {
                if (++evaluations > 200000)
                { error = "resonance-root refinement exceeded its limit (flat or nearly tangent gaps)"; return false; }
                if (!arma::eig_sym(e, h0+b*dHdB))
                { error = "resonance-root diagonalization failed"; return false; }
                return true;
            };
            arma::vec leftE,rightE;
            if (!energies(fieldMin,leftE) || !energies(fieldMax,rightE)) return false;
            using Pair=std::pair<arma::uword,arma::uword>;
            std::vector<Pair> pairs;
            for (arma::uword a=0;a<h0.n_rows;++a)
                for (arma::uword b=a+1;b<h0.n_rows;++b) pairs.emplace_back(a,b);
            std::function<bool(double,const arma::vec&,double,const arma::vec&,const std::vector<Pair>&)> visit;
            visit = [&](double left,const arma::vec &el,double right,const arma::vec &er,const std::vector<Pair> &candidates) {
                std::vector<Pair> possible;
                const double width=right-left;
                const double gapLeft=arma::diff(el).min(), gapRight=arma::diff(er).min();
                const double isolatedGap=.5*(gapLeft+gapRight-bound*width);
                // For separated eigenvalues, second-order perturbation theory
                // bounds each gap curvature by diameter(dH/dB)^2 / minGap.
                // The linear-interpolation error is then <= curvature*w^2/8.
                // This sharper certificate prevents exhaustive refinement of
                // almost flat but nonresonant avoided-crossing minima.
                const double interpolationError=isolatedGap>0
                    ? bound*bound/isolatedGap*width*width/8.
                    : std::numeric_limits<double>::infinity();
                for (const auto &pair:candidates)
                {
                    const auto [a,b]=pair;
                    const double fl=el(b)-el(a)-omega, fr=er(b)-er(a)-omega;
                    // If same-sign endpoint exclusion cones cover the interval,
                    // the gap cannot reach zero anywhere between them.
                    if (fl*fr>0 && std::abs(fl)+std::abs(fr)>bound*(right-left)*(1+1e-12)) continue;
                    if (fl*fr>0 && std::min(std::abs(fl),std::abs(fr))>interpolationError*(1+1e-12)) continue;
                    // A curvature bound also certifies strict monotonicity:
                    // the derivative differs from the chord slope by at most
                    // curvature*width/2. Refine that unique crossing directly,
                    // with a residual/minimum-slope bound on its field error.
                    const double minimumSlope=std::abs(fr-fl)/width-4*interpolationError/width;
                    if (fl*fr<=0 && minimumSlope>0)
                    {
                        double xl=left,xr=right,yl=fl,yr=fr;
                        bool found=false;
                        for (unsigned iteration=0;iteration<64;++iteration)
                        {
                            const double x=xl-yl*(xr-xl)/(yr-yl);
                            arma::vec ex;
                            if (!energies(x,ex)) return false;
                            const double y=ex(b)-ex(a)-omega;
                            if (std::abs(y)<=minimumSlope*fieldTolerance)
                            { roots.push_back({x,a,b}); found=true; break; }
                            if (!(x>xl && x<xr)) break;
                            if (yl*y<=0) { xr=x;yr=y; } else { xl=x;yl=y; }
                        }
                        if (found) continue;
                    }
                    if (right-left<=fieldTolerance)
                    {
                        if (fl*fr<=0 && !(fl==0 && fr==0))
                        {
                            const double fraction=fl/(fl-fr);
                            roots.push_back({left+fraction*(right-left),a,b});
                        }
                    }
                    else possible.push_back(pair);
                }
                if (possible.empty()) return true;
                const double mid=.5*(left+right);
                arma::vec em;
                if (!energies(mid,em)) return false;
                return visit(left,el,mid,em,possible) && visit(mid,em,right,er,possible);
            };
            if (!visit(fieldMin,leftE,fieldMax,rightE,pairs)) { roots.clear(); return false; }
            std::sort(roots.begin(),roots.end(),[](const auto &a,const auto &b) {
                if (a.lower!=b.lower) return a.lower<b.lower;
                if (a.upper!=b.upper) return a.upper<b.upper;
                return a.fieldT<b.fieldT;
            });
            roots.erase(std::unique(roots.begin(),roots.end(),[&](const auto &a,const auto &b) {
                return a.lower==b.lower && a.upper==b.upper && std::abs(a.fieldT-b.fieldT)<4*fieldTolerance;
            }),roots.end());
            return true;
        }
    };
}
#endif
