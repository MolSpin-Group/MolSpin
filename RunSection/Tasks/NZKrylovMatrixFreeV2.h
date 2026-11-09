#pragma once
// Matrix-free representation of V9's time-local NZ stationary generator.
// Preserves its non-secular double-sided action and row-major superspace.
// Does not change correlation functions, Hamiltonian, reaction operators,
// cross-correlation terms, or independently supplied relaxation operators.
#include "NZKrylovStationary.h"
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

namespace RunSection { namespace NZKrylovMatrixFreeV2 {

struct Term {
    arma::cx_mat left_adjoint; // O1^H
    arma::cx_mat right;        // O2
    arma::cx_mat spectral_conjugate; // conj(S_ij), indexed by eigenstate pair
    Term(const arma::cx_mat &a, const arma::cx_mat &b, const arma::cx_mat &s)
      : left_adjoint(a.t()), right(b), spectral_conjugate(arma::conj(s)) {}
};

inline bool Valid(const Term &t, arma::uword d) {
    return t.left_adjoint.n_rows==d && t.left_adjoint.n_cols==d
        && t.right.n_rows==d && t.right.n_cols==d
        && t.spectral_conjugate.n_rows==d && t.spectral_conjugate.n_cols==d
        && t.left_adjoint.is_finite() && t.right.is_finite()
        && t.spectral_conjugate.is_finite();
}

// Apply exactly the V9 NZ term -L_op1 * diag(conj(spec)) * R_op2
// for MolSpin's row-major vectorization vectorise(rho.st()).
inline arma::cx_mat ApplyTerm(const Term &t, const arma::cx_mat &x) {
    arma::cx_mat z = t.right*x - x*t.right;
    z %= t.spectral_conjugate;
    return -(t.left_adjoint*z - z*t.left_adjoint);
}

class Operator {
public:
    Operator(const arma::cx_mat &eigen_h,
             const arma::cx_mat &reaction_k,
             const std::vector<Term> &terms,
             const arma::cx_mat &other_relaxation)
      : h_(eigen_h), k_(reaction_k), terms_(terms), other_(other_relaxation) {}

    bool Valid() const {
        const arma::uword d=h_.n_rows, n=d*d;
        if(d==0 || h_.n_cols!=d || k_.n_rows!=d || k_.n_cols!=d ||
           !h_.is_finite() || !k_.is_finite() ||
           (other_.n_elem && (other_.n_rows!=n || other_.n_cols!=n || !other_.is_finite())))
            return false;
        for(const auto &t:terms_) if(!NZKrylovMatrixFreeV2::Valid(t,d)) return false;
        return true;
    }

    bool Apply(const arma::cx_vec &x, arma::cx_vec &y) const {
        const arma::uword d=h_.n_rows;
        if(x.n_elem!=d*d || !x.is_finite()) return false;
        const arma::cx_mat rho = arma::reshape(x,d,d).st();
        // Matches precisely V9's -i(L_H-R_H) -(L_K+R_K).
        arma::cx_mat result = arma::cx_double(0.0,-1.0)*(h_*rho-rho*h_)
                            -k_*rho-rho*k_;
        for (const auto &t:terms_) result += ApplyTerm(t,rho);
        if(other_.n_elem) {
            const arma::cx_vec extra = other_*x;
            result += arma::reshape(extra,d,d).st();
        }
        y = arma::vectorise(result.st());
        return y.is_finite();
    }

private:
    const arma::cx_mat &h_, &k_;
    const std::vector<Term> &terms_;
    const arma::cx_mat &other_;
};

template<class Op>
inline bool Solve(const Op &op, const arma::cx_vec &rhs,
                  const arma::cx_mat &preconditioner_q,
                  arma::cx_vec &solution,
                  NZKrylovStationary::Report &report,
                  const NZKrylovStationary::Options &options) {
    report=NZKrylovStationary::Report{};
    if(!op.Valid() || rhs.n_elem==0 ||
       preconditioner_q.n_rows==0 || preconditioner_q.n_rows!=preconditioner_q.n_cols ||
       rhs.n_elem!=preconditioner_q.n_rows*preconditioner_q.n_rows ||
       options.restart<2 || options.restart>256 ||
       options.max_iterations<1 || options.max_iterations>10000 ||
       !std::isfinite(options.relative_tolerance) ||
       options.relative_tolerance<1e-14 || options.relative_tolerance>1e-3 ||
       !rhs.is_finite() || !preconditioner_q.is_finite()) return false;

    const arma::uword n=rhs.n_elem;
    solution.zeros(n);
    const double rhs_norm=arma::norm(rhs,2);
    if(!std::isfinite(rhs_norm)) return false;
    if(rhs_norm==0.0) {
        report.converged=true; report.relative_residual=0.0; return true;
    }
    const unsigned restart=std::min(options.restart,options.max_iterations);
    std::vector<arma::cx_vec> v(restart+1),z(restart);
    arma::cx_mat h(restart+1,restart,arma::fill::zeros);
    while(report.iterations < options.max_iterations) {
        arma::cx_vec ax;
        if(!op.Apply(solution,ax)) return false;
        const arma::cx_vec residual=rhs-ax;
        const double beta=arma::norm(residual,2);
        report.relative_residual=beta/rhs_norm;
        if(!std::isfinite(beta)) return false;
        if(report.relative_residual<=options.relative_tolerance) {
            report.converged=true; return true;
        }
        h.zeros(); v[0]=residual/beta;
        const unsigned cycle=std::min(restart,options.max_iterations-report.iterations);
        bool updated=false;
        for(unsigned j=0;j<cycle;++j) {
            if(!NZKrylovStationary::ApplySylvesterInverse(preconditioner_q,v[j],z[j])) {
                report.preconditioner_failed=true; return false;
            }
            arma::cx_vec w;
            if(!op.Apply(z[j],w)) return false;
            for(unsigned pass=0;pass<2;++pass) {
                for(unsigned k=0;k<=j;++k) {
                    const arma::cx_double c=arma::cdot(v[k],w);
                    h(k,j)+=c; w-=c*v[k];
                }
            }
            const double norm_next=arma::norm(w,2);
            if(!std::isfinite(norm_next)) return false;
            h(j+1,j)=norm_next;
            const bool breakdown=norm_next<1e-14;
            if(!breakdown)v[j+1]=w/norm_next;
            const arma::cx_mat hj=h.submat(0,0,j+1,j);
            arma::cx_vec g(j+2,arma::fill::zeros);
            g(0)=beta;
            arma::cx_vec coefficients;
            try { if(!arma::solve(coefficients,hj,g)) return false; }
            catch(...) { return false; }
            if(coefficients.n_elem!=j+1 || !coefficients.is_finite()) return false;
            const double projected=arma::norm(g-hj*coefficients,2);
            if(!std::isfinite(projected)) return false;
            ++report.iterations;
            if(projected/rhs_norm<=options.relative_tolerance || breakdown || j+1==cycle) {
                for(unsigned k=0;k<=j;++k)solution+=coefficients(k)*z[k];
                ++report.restarts; updated=true; break;
            }
        }
        if(!updated)return false;
    }
    arma::cx_vec ax;
    if(!op.Apply(solution,ax))return false;
    report.relative_residual=arma::norm(rhs-ax,2)/rhs_norm;
    report.converged=std::isfinite(report.relative_residual) &&
                     report.relative_residual<=options.relative_tolerance;
    return report.converged;
}

}} // namespace RunSection::NZKrylovMatrixFreeV2
