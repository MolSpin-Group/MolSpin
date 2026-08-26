/////////////////////////////////////////////////////////////////////////
// SSNakajimaZwanzigBuilder implementation.
/////////////////////////////////////////////////////////////////////////
#include "SSNakajimaZwanzigBuilder.h"

#include "Interaction.h"
#include "NakajimaZwanzig.h"
#include "ObjectParser.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace RunSection::General::SS
{
    namespace
    {
        bool Fail(std::string &error,const std::string &message)
        { error=message; return false; }

        bool ReadRelaxationLists(const SpinAPI::interaction_ptr &interaction,
            std::vector<double> &g,std::vector<double> &tau,std::string &error)
        {
            g.clear(); tau.clear();
            int multi=0;
            if(interaction->Properties()->Get("def_multexpo",multi) && multi==1)
                return Fail(error,"General historical NZ does not yet accept def_multexpo=1 matrix-valued correlation fits; use the historical NZ task for that input until parity is migrated");
            const bool hasG=interaction->Properties()->GetList("g",g);
            const bool hasTau=interaction->Properties()->GetList("tau_c",tau);
            if(hasG!=hasTau)
                return Fail(error,"historical NZ Interaction \""+interaction->Name()+"\" must specify both g and tau_c");
            if(!hasG) return true; // not an NZ source interaction
            if(g.empty()||tau.empty())
                return Fail(error,"historical NZ Interaction \""+interaction->Name()+"\" has empty g/tau_c lists");
            for(double x:g) if(!std::isfinite(x))
                return Fail(error,"historical NZ Interaction \""+interaction->Name()+"\" has non-finite g amplitude");
            for(double x:tau) if(!std::isfinite(x)||x==0.0)
                return Fail(error,"historical NZ Interaction \""+interaction->Name()+"\" requires finite nonzero tau_c values");
            return true;
        }

        bool CartesianOperators(SpinAPI::SpinSpace &space,
            const SpinAPI::spin_ptr &s1,const SpinAPI::spin_ptr &s2,
            std::vector<arma::cx_mat> &ops,std::string &error)
        {
            arma::cx_mat sx1,sy1,sz1;
            if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(s1->Sx()),s1,sx1) ||
               !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(s1->Sy()),s1,sy1) ||
               !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(s1->Sz()),s1,sz1))
                return Fail(error,"failed to create historical NZ Cartesian operators for spin \""+s1->Name()+"\"");
            if(!s2)
            {
                // Exact historical ordering: Sx repeated for x/y/z spatial
                // components, followed by Sy and Sz in the same pattern.
                ops={sx1,sx1,sx1,sy1,sy1,sy1,sz1,sz1,sz1};
                return true;
            }
            arma::cx_mat sx2,sy2,sz2;
            if(!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(s2->Sx()),s2,sx2) ||
               !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(s2->Sy()),s2,sy2) ||
               !space.CreateOperator(arma::conv_to<arma::cx_mat>::from(s2->Sz()),s2,sz2))
                return Fail(error,"failed to create historical NZ Cartesian operators for spin \""+s2->Name()+"\"");
            ops={sx1*sx2,sx1*sy2,sx1*sz2,
                 sy1*sx2,sy1*sy2,sy1*sz2,
                 sz1*sx2,sz1*sy2,sz1*sz2};
            return true;
        }

        bool SphericalOperators(SpinAPI::SpinSpace &space,
            const SpinAPI::interaction_ptr &interaction,
            const SpinAPI::spin_ptr &s1,const SpinAPI::spin_ptr &s2,
            std::vector<arma::cx_mat> &ops,std::string &error)
        {
            arma::cx_mat t00,t20,tm1,tp1,tm2,tp2;
            bool ok=false;
            if(!s2)
            {
                const arma::cx_vec field=arma::conv_to<arma::cx_vec>::from(interaction->Field());
                ok=space.LRk0TensorT0(s1,field,t00) &&
                   space.LRk2SphericalTensorT0(s1,field,t20) &&
                   space.LRk2SphericalTensorTm1(s1,field,tm1) &&
                   space.LRk2SphericalTensorTp1(s1,field,tp1) &&
                   space.LRk2SphericalTensorTm2(s1,field,tm2) &&
                   space.LRk2SphericalTensorTp2(s1,field,tp2);
            }
            else
            {
                ok=space.BlRk0TensorT0(s1,s2,t00) &&
                   space.BlRk2SphericalTensorT0(s1,s2,t20) &&
                   space.BlRk2SphericalTensorTm1(s1,s2,tm1) &&
                   space.BlRk2SphericalTensorTp1(s1,s2,tp1) &&
                   space.BlRk2SphericalTensorTm2(s1,s2,tm2) &&
                   space.BlRk2SphericalTensorTp2(s1,s2,tp2);
            }
            if(!ok) return Fail(error,"failed to create historical NZ spherical operators for Interaction \""+interaction->Name()+"\"");
            ops={t00,t20,tm1,tp1,tm2,tp2};

            int coeff=0;
            SpinAPI::Tensor parsed(0);
            if(interaction->Properties()->Get("tensor",parsed) &&
               interaction->Properties()->Get("coeff",coeff) && coeff==1)
            {
                const auto tensor=interaction->CouplingTensor();
                if(!tensor) return Fail(error,"historical NZ coeff=1 requires a coupling tensor for Interaction \""+interaction->Name()+"\"");
                const arma::cx_mat A=arma::conv_to<arma::cx_mat>::from(tensor->LabFrame());
                arma::cx_vec am(6,arma::fill::zeros);
                am(0)=(A(0,0)+A(1,1)+A(2,2))/std::sqrt(3.0);
                am(1)=(3.0*A(2,2)-(A(0,0)+A(1,1)+A(2,2)))/std::sqrt(6.0);
                am(2)=0.5*(A(0,2)+A(2,0)+arma::cx_double(0.0,1.0)*(A(1,2)+A(2,1)));
                am(3)=-0.5*(A(0,2)+A(2,0)-arma::cx_double(0.0,1.0)*(A(1,2)+A(2,1)));
                am(4)=0.5*(A(0,0)-A(1,1)-arma::cx_double(0.0,1.0)*(A(0,1)+A(1,0)));
                am(5)=0.5*(A(0,0)-A(1,1)+arma::cx_double(0.0,1.0)*(A(0,1)+A(1,0)));
                // Exact simple-list legacy convention.
                ops[0]*=am(0); ops[1]*=am(1); ops[2]*=am(3);
                ops[3]*=am(2); ops[4]*=am(4); ops[5]*=am(5);
            }
            else
            {
                // Legacy phase convention in the no-coefficient branch.
                ops[2]*=-1.0; ops[3]*=-1.0;
            }
            return true;
        }

        bool AddOperatorSet(const std::vector<arma::cx_mat> &labOps,
            const arma::cx_mat &eigenvectors,const arma::cx_mat &omega,
            const std::vector<double> &g,const std::vector<double> &tau,
            int terms,int defG,arma::cx_mat &total,std::string &error)
        {
            std::vector<arma::cx_mat> ops; ops.reserve(labOps.size());
            for(const auto &op:labOps) ops.push_back(eigenvectors.t()*op*eigenvectors);
            arma::cx_mat commonJ;
            if(defG!=1)
            {
                if(g.size()!=tau.size())
                    return Fail(error,"historical NZ general g and tau_c lists must have identical lengths when def_g=0");
                if(!SpinAPI::NakajimaZwanzig::HistoricalMultiExponentialSpectralDensity(g,tau,omega,commonJ,&error)) return false;
            }
            else
            {
                if(tau.size()!=1 || g.size()!=ops.size())
                    return Fail(error,"historical NZ def_g=1 requires one tau_c and one g coefficient per operator");
            }

            const bool diagonalOnly=(terms==1);
            for(size_t i=0;i<ops.size();++i)
            {
                const size_t jBegin=diagonalOnly?i:0;
                const size_t jEnd=diagonalOnly?i+1:ops.size();
                for(size_t j=jBegin;j<jEnd;++j)
                {
                    arma::cx_mat J=commonJ;
                    if(defG==1)
                    {
                        const double amplitude=g[i]*g[j];
                        if(!SpinAPI::NakajimaZwanzig::HistoricalExponentialSpectralDensity(
                            {amplitude,0.0},{tau[0],0.0},omega,J,&error)) return false;
                    }
                    arma::cx_mat R;
                    if(!SpinAPI::NakajimaZwanzig::HistoricalTensor(ops[i],ops[j],J,R,&error)) return false;
                    total+=R;
                }
            }
            return true;
        }
    }

    bool SSNakajimaZwanzigBuilder::BuildHistorical(const SpinAPI::system_ptr &system,
        SpinAPI::SpinSpace &space,const arma::sp_cx_mat &hamiltonian,
        arma::sp_cx_mat &relaxation,std::string &error)
    {
        error.clear(); relaxation.zeros(hamiltonian.n_rows*hamiltonian.n_rows,
                                       hamiltonian.n_rows*hamiltonian.n_rows);
        if(!system) return Fail(error,"cannot build historical NZ relaxation for a null SpinSystem");
        if(hamiltonian.n_rows==0 || hamiltonian.n_rows!=hamiltonian.n_cols)
            return Fail(error,"historical NZ requires a non-empty square static Hamiltonian");

        arma::vec eigenvalues; arma::cx_mat eigenvectors;
        if(!arma::eig_sym(eigenvalues,eigenvectors,arma::cx_mat(hamiltonian)))
            return Fail(error,"failed to diagonalize the static Hamiltonian for historical NZ relaxation");
        arma::cx_mat omega;
        if(!SpinAPI::NakajimaZwanzig::HistoricalFrequencyMatrix(eigenvalues,omega,&error)) return false;
        arma::cx_mat total(omega.n_rows,omega.n_cols,arma::fill::zeros);
        bool any=false;

        space.UseSuperoperatorSpace(false);
        for(const auto &interaction:system->Interactions())
        {
            if(!interaction) continue;
            std::vector<double> g,tau;
            if(!ReadRelaxationLists(interaction,g,tau,error)) return false;
            if(g.empty()) continue;
            any=true;
            int opsMode=0,terms=0,defG=0;
            interaction->Properties()->Get("ops",opsMode);
            interaction->Properties()->Get("terms",terms);
            interaction->Properties()->Get("def_g",defG);
            if(opsMode!=0 && opsMode!=1)
                return Fail(error,"historical NZ ops must be 0 (rank-0/2 spherical) or 1 (Cartesian)");
            if(terms!=0 && terms!=1)
                return Fail(error,"historical NZ terms must be 0 (cross terms) or 1 (autocorrelation only)");

            const auto group1=interaction->Group1();
            const auto group2=interaction->Group2();
            if(group1.empty())
                return Fail(error,"historical NZ Interaction \""+interaction->Name()+"\" has empty group1");
            if(interaction->Type()==SpinAPI::InteractionType::SingleSpin)
            {
                for(const auto &s1:group1)
                {
                    std::vector<arma::cx_mat> localOps;
                    const bool ok=opsMode==1?CartesianOperators(space,s1,nullptr,localOps,error):
                                             SphericalOperators(space,interaction,s1,nullptr,localOps,error);
                    if(!ok || !AddOperatorSet(localOps,eigenvectors,omega,g,tau,terms,defG,total,error)) return false;
                }
            }
            else if(interaction->Type()==SpinAPI::InteractionType::DoubleSpin)
            {
                if(group2.empty()) return Fail(error,"historical NZ double-spin Interaction \""+interaction->Name()+"\" has empty group2");
                for(const auto &s1:group1) for(const auto &s2:group2)
                {
                    std::vector<arma::cx_mat> localOps;
                    const bool ok=opsMode==1?CartesianOperators(space,s1,s2,localOps,error):
                                             SphericalOperators(space,interaction,s1,s2,localOps,error);
                    if(!ok || !AddOperatorSet(localOps,eigenvectors,omega,g,tau,terms,defG,total,error)) return false;
                }
            }
            else return Fail(error,"historical NZ only supports single- and double-spin Interactions");
        }
        // A MultiSS network may contain optical/bookkeeping manifolds with no
        // spin relaxation source.  In that case the local NZ contribution is
        // exactly zero; other manifolds may still carry NZ-enabled interactions.
        if(!any) return true;
        if(!total.is_finite()) return Fail(error,"historical NZ relaxation contains non-finite entries");
        relaxation=arma::sp_cx_mat(total);
        return true;
    }
}
