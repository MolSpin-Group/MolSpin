/////////////////////////////////////////////////////////////////////////
// NakajimaZwanzig implementation (SpinAPI Module)
// ------------------
// THEORY CONTRACT
//   This file extracts the established MolSpin algebra from task-local NZ code
//   into a reusable SpinAPI primitive. It must not silently change the
//   published model into a different NZ formulation.
//
//   Established MolSpin/JACS workflow:
//     1. diagonalize the spin Hamiltonian H;
//     2. form transition frequencies omega_ab=E_a-E_b;
//     3. evaluate the chosen spectral density J(omega_ab);
//     4. form the task's commutator-superoperator contraction;
//     5. add reaction loss K separately in the task/global generator.
//
//   This form underlies the MolSpin workflow used in
//       DOI: 10.1021/jacs.5c06173
//
//   The formal second-order Markovian reactive Nakajima-Zwanzig construction
//   discussed by Fay, Lindoy and Manolopoulos propagates the memory kernel with
//   a reactive L0 that already contains reaction loss.  See
//       DOI: 10.1063/5.0040519
//
//   Those are not algebraically interchangeable. A reactive-L0 variant may be
//   added later under an explicit model name and separate regression tests.
//
// HIERARCHY
//   SpinAPI::NakajimaZwanzig provides reusable tensor algebra -> General/SS will
//   own one-manifold relaxation assembly -> General/MultiSS composes manifolds.
//   No network topology is encoded here.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "NakajimaZwanzig.h"
#include <cmath>

namespace SpinAPI::NakajimaZwanzig
{
    namespace
    {
        bool Fail(std::string *error,const std::string &message)
        { if(error)*error=message; return false; }
    }

    bool FrequencyMatrix(const arma::vec &eigenvalues,
        arma::cx_mat &omega,std::string *error)
    {
        if(error)error->clear();
        if(eigenvalues.empty()||!eigenvalues.is_finite())
            return Fail(error,"NZ frequency matrix requires finite Hamiltonian eigenvalues");
        const arma::cx_mat E=arma::diagmat(arma::conv_to<arma::cx_vec>::from(eigenvalues));
        const arma::cx_mat I=arma::eye<arma::cx_mat>(E.n_rows,E.n_cols);
        // MolSpin uses lambda = kron(E,I)-kron(I,E^T).
        omega=arma::kron(E,I)-arma::kron(I,E.st());
        return true;
    }

    bool ExponentialSpectralDensity(const std::complex<double> &amplitude,
        const std::complex<double> &tauC,const arma::cx_mat &omega,
        arma::cx_mat &spectralDensity,std::string *error)
    {
        if(error)error->clear();
        if(omega.n_rows==0||omega.n_rows!=omega.n_cols||
            !std::isfinite(amplitude.real())||!std::isfinite(amplitude.imag())||
            !std::isfinite(tauC.real())||!std::isfinite(tauC.imag())||std::abs(tauC)==0.0)
            return Fail(error,"invalid NZ exponential spectral-density inputs");
        arma::cx_vec entries=omega.diag();
        for(arma::uword i=0;i<entries.n_elem;++i)
            entries(i)=amplitude/((1.0/tauC)+arma::cx_double(0.0,-1.0)*omega(i,i));
        spectralDensity=arma::diagmat(entries);
        return spectralDensity.is_finite();
    }

    bool MultiExponentialSpectralDensity(const std::vector<double> &amplitudes,
        const std::vector<double> &tauC,const arma::cx_mat &omega,
        arma::cx_mat &spectralDensity,std::string *error)
    {
        if(error)error->clear();
        if(amplitudes.empty()||amplitudes.size()!=tauC.size())
            return Fail(error,"NZ multi-exponential lists must be equally sized and non-empty");
        arma::cx_vec entries(omega.n_rows,arma::fill::zeros);
        for(arma::uword i=0;i<entries.n_elem;++i)
        {
            for(size_t j=0;j<tauC.size();++j)
            {
                if(!std::isfinite(amplitudes[j])||!std::isfinite(tauC[j]))
                    return Fail(error,"NZ amplitudes/tau_c must be finite");
                // Matrix fits use zero-amplitude entries as padding. Their
                // tau_c value has no physical effect and may also be zero.
                if(amplitudes[j]==0.0) continue;
                if(!(tauC[j]>0.0))
                    return Fail(error,"every nonzero NZ exponential requires tau_c > 0");
                entries(i)+=amplitudes[j]/((1.0/tauC[j])+arma::cx_double(0.0,-1.0)*omega(i,i));
            }
        }
        spectralDensity=arma::diagmat(entries); return spectralDensity.is_finite();
    }

    bool SpectralDensity(
        const std::vector<SpinAPI::Relaxation::ExponentialTerm> &terms,
        const arma::cx_mat &omega, arma::cx_mat &spectralDensity,
        std::string *error)
    {
        return SpinAPI::Relaxation::EvaluateSpectralDensity(terms,omega,
            SpinAPI::Relaxation::SpectralDensityFunction::ComplexOneSided,
            true,spectralDensity,error);
    }

    bool RelaxationTensor(const arma::cx_mat &op1,const arma::cx_mat &op2,
        const arma::cx_mat &spectralDensity,arma::cx_mat &tensor,std::string *error)
    {
        if(error)error->clear();
        if(op1.n_rows==0||op1.n_rows!=op1.n_cols||arma::size(op1)!=arma::size(op2))
            return Fail(error,"NZ operators must be equally sized square matrices");
        const arma::uword d=op1.n_rows;
        if(spectralDensity.n_rows!=d*d||spectralDensity.n_cols!=d*d)
            return Fail(error,"NZ spectral density has incompatible superspace dimension");
        const arma::cx_mat I=arma::eye<arma::cx_mat>(d,d);
        // Exact legacy algebra (TaskStaticSSNakajimaZwanzigTimeEvo):
        // op1_SS = kron(op1^dagger,I) - kron(I,(op1^dagger)^T)
        // op2_SS = kron(op2,I)        - kron(I,op2^T)
        // R       = -op1_SS * J^T * op2_SS
        const arma::cx_mat op1SS=arma::kron(op1.t(),I)-arma::kron(I,op1.t().st());
        const arma::cx_mat op2SS=arma::kron(op2,I)-arma::kron(I,op2.st());
        tensor=-op1SS*spectralDensity.t()*op2SS;
        return tensor.is_finite();
    }
}
