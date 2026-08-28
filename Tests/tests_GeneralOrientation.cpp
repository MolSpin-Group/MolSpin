//////////////////////////////////////////////////////////////////////////////
// Shared General orientation semantics across HSGeneral / SSGeneral / MultiSSGeneral.
//////////////////////////////////////////////////////////////////////////////
#include "HSOrientationSampler.h"
#include "SSOrientationSampler.h"
#include "MultiSSOrientationSampler.h"
#include "HSExecutionPlan.h"
#include "SSExecutionPlan.h"
#include "MultiSSExecutionPlan.h"
#include "ObjectParser.h"
#include <cmath>
#include <sstream>

namespace
{
    template <typename A, typename B>
    bool SameOrientation(const A &a,const B &b,double tol=1.0e-13)
    {
        return std::abs(a.alpha-b.alpha)<tol &&
            std::abs(a.beta-b.beta)<tol &&
            std::abs(a.gamma-b.gamma)<tol &&
            std::abs(a.weight-b.weight)<tol &&
            arma::norm(a.frameToLab-b.frameToLab,"fro")<tol;
    }

    bool BuildCommon2D(std::vector<RunSection::General::HS::HSOrientation> &hs,
        std::vector<RunSection::General::SS::SSOrientation> &ss,
        std::vector<RunSection::General::MultiSS::MultiSSOrientation> &ms)
    {
        RunSection::General::HS::HSExecutionPlan hp;
        hp.orientation=RunSection::General::HS::OrientationMode::Powder2D;
        hp.powderGridType=SpinAPI::PowderGridType::Uniform;
        hp.powderDomain=SpinAPI::PowderGridDomain::FullSphere;
        hp.powderPoints=9;

        RunSection::General::SS::SSExecutionPlan sp;
        sp.orientation=RunSection::General::SS::SSOrientationMode::Powder2D;
        sp.powderGridType=hp.powderGridType; sp.powderDomain=hp.powderDomain; sp.powderPoints=hp.powderPoints;

        RunSection::General::MultiSS::MultiSSExecutionPlan mp;
        mp.orientation=RunSection::General::MultiSS::MultiSSOrientationMode::Powder2D;
        mp.powderGridType=hp.powderGridType; mp.powderDomain=hp.powderDomain; mp.powderPoints=hp.powderPoints;

        std::ostringstream log;std::string error;
        return RunSection::General::HS::HSOrientationSampler::Build(hp,hs,log,error)&&
            RunSection::General::SS::SSOrientationSampler::Build(sp,ss,log,error)&&
            RunSection::General::MultiSS::MultiSSOrientationSampler::Build(mp,ms,log,error);
    }
}

bool test_general_orientation_hs_ss_multiss_2d_equivalence()
{
    std::vector<RunSection::General::HS::HSOrientation> hs;
    std::vector<RunSection::General::SS::SSOrientation> ss;
    std::vector<RunSection::General::MultiSS::MultiSSOrientation> ms;
    if(!BuildCommon2D(hs,ss,ms)||hs.size()!=ss.size()||hs.size()!=ms.size())return false;
    for(size_t i=0;i<hs.size();++i)if(!SameOrientation(hs[i],ss[i])||!SameOrientation(hs[i],ms[i]))return false;
    return true;
}

bool test_general_orientation_hs_ss_multiss_so3_equivalence()
{
    RunSection::General::HS::HSExecutionPlan hp;
    hp.orientation=RunSection::General::HS::OrientationMode::PowderSO3;
    hp.powderGridType=SpinAPI::PowderGridType::Sophe;hp.powderGridSize=4;hp.powderSymmetry="ci";
    hp.powderGammaPoints=3;hp.powderGammaOffset=0.17;

    RunSection::General::SS::SSExecutionPlan sp;
    sp.orientation=RunSection::General::SS::SSOrientationMode::PowderSO3;
    sp.powderGridType=hp.powderGridType;sp.powderGridSize=hp.powderGridSize;sp.powderSymmetry=hp.powderSymmetry;
    sp.powderGammaPoints=hp.powderGammaPoints;sp.powderGammaOffset=hp.powderGammaOffset;

    RunSection::General::MultiSS::MultiSSExecutionPlan mp;
    mp.orientation=RunSection::General::MultiSS::MultiSSOrientationMode::PowderSO3;
    mp.powderGridType=hp.powderGridType;mp.powderGridSize=hp.powderGridSize;mp.powderSymmetry=hp.powderSymmetry;
    mp.powderGammaPoints=hp.powderGammaPoints;mp.powderGammaOffset=hp.powderGammaOffset;

    std::vector<RunSection::General::HS::HSOrientation> hs;
    std::vector<RunSection::General::SS::SSOrientation> ss;
    std::vector<RunSection::General::MultiSS::MultiSSOrientation> ms;
    std::ostringstream log;std::string error;
    if(!RunSection::General::HS::HSOrientationSampler::Build(hp,hs,log,error)||
       !RunSection::General::SS::SSOrientationSampler::Build(sp,ss,log,error)||
       !RunSection::General::MultiSS::MultiSSOrientationSampler::Build(mp,ms,log,error)||
       hs.size()!=ss.size()||hs.size()!=ms.size())return false;
    for(size_t i=0;i<hs.size();++i)if(!SameOrientation(hs[i],ss[i])||!SameOrientation(hs[i],ms[i]))return false;
    return true;
}

bool test_general_orientation_explicit_zyz_equivalence()
{
    const double a=0.41,b=0.73,g=-0.19,w=0.37;
    RunSection::General::HS::HSExecutionPlan hp;hp.orientation=RunSection::General::HS::OrientationMode::Explicit;
    hp.powderGammaOffset=a;hp.explicitTheta=b;hp.explicitPhi=g;hp.explicitWeight=w;
    RunSection::General::SS::SSExecutionPlan sp;sp.orientation=RunSection::General::SS::SSOrientationMode::Explicit;
    sp.explicitAlpha=a;sp.explicitBeta=b;sp.explicitGamma=g;sp.explicitWeight=w;
    RunSection::General::MultiSS::MultiSSExecutionPlan mp;mp.orientation=RunSection::General::MultiSS::MultiSSOrientationMode::Explicit;
    mp.explicitAlpha=a;mp.explicitBeta=b;mp.explicitGamma=g;mp.explicitWeight=w;
    std::vector<RunSection::General::HS::HSOrientation> hs;std::vector<RunSection::General::SS::SSOrientation> ss;
    std::vector<RunSection::General::MultiSS::MultiSSOrientation> ms;std::ostringstream log;std::string error;
    if(!RunSection::General::HS::HSOrientationSampler::Build(hp,hs,log,error)||
       !RunSection::General::SS::SSOrientationSampler::Build(sp,ss,log,error)||
       !RunSection::General::MultiSS::MultiSSOrientationSampler::Build(mp,ms,log,error)||
       hs.size()!=1||ss.size()!=1||ms.size()!=1)return false;
    return SameOrientation(hs[0],ss[0])&&SameOrientation(hs[0],ms[0])&&std::abs(hs[0].weight-w)<1e-14;
}

bool test_general_orientation_parser_explicit_zyz_equivalence()
{
    const std::string orientation="powderorientation=0.41 0.73 -0.19 0.37;";
    MSDParser::ObjectParser hpParser("hs",
        "type=HSGeneral;calculation=timeevolution;hamiltonianh0list=H0;"+orientation);
    MSDParser::ObjectParser spParser("ss","type=SSGeneral;"+orientation);
    MSDParser::ObjectParser mpParser("ms","type=MultiSSGeneral;"+orientation);

    RunSection::General::HS::HSExecutionPlan hp;
    RunSection::General::SS::SSExecutionPlan sp;
    RunSection::General::MultiSS::MultiSSExecutionPlan mp;
    std::string error;
    if(!RunSection::General::HS::ResolveExecutionPlan(hpParser,hp,error) ||
       !RunSection::General::SS::ResolveSSExecutionPlan(spParser,sp,error) ||
       !RunSection::General::MultiSS::ResolveMultiSSExecutionPlan(mpParser,mp,error))
        return false;

    std::vector<RunSection::General::HS::HSOrientation> hs;
    std::vector<RunSection::General::SS::SSOrientation> ss;
    std::vector<RunSection::General::MultiSS::MultiSSOrientation> ms;
    std::ostringstream log;
    if(!RunSection::General::HS::HSOrientationSampler::Build(hp,hs,log,error) ||
       !RunSection::General::SS::SSOrientationSampler::Build(sp,ss,log,error) ||
       !RunSection::General::MultiSS::MultiSSOrientationSampler::Build(mp,ms,log,error) ||
       hs.size()!=1 || ss.size()!=1 || ms.size()!=1)
        return false;

    return SameOrientation(hs[0],ss[0]) &&
           SameOrientation(hs[0],ms[0]) &&
           std::abs(hs[0].alpha-0.41)<1.0e-14 &&
           std::abs(hs[0].beta-0.73)<1.0e-14 &&
           std::abs(hs[0].gamma+0.19)<1.0e-14 &&
           std::abs(hs[0].weight-0.37)<1.0e-14;
}

bool test_general_orientation_generated_weights_are_normalized()
{
    std::vector<RunSection::General::HS::HSOrientation> hs;
    std::vector<RunSection::General::SS::SSOrientation> ss;
    std::vector<RunSection::General::MultiSS::MultiSSOrientation> ms;
    if(!BuildCommon2D(hs,ss,ms))return false;
    auto sum=[](const auto &v){double s=0.0;for(const auto&o:v)s+=o.weight;return s;};
    if(std::abs(sum(hs)-1.0)>1e-13||std::abs(sum(ss)-1.0)>1e-13||std::abs(sum(ms)-1.0)>1e-13)return false;

    // SpinAPI keeps its raw solid-angle contract; General normalizes only its
    // generated ensemble samples and must not mutate the underlying API.
    SpinAPI::PowderGrid raw;
    if(!SpinAPI::CreateUniformPowderGrid(9,SpinAPI::PowderGridDomain::FullSphere,raw))return false;
    double rawSum=0.0;for(const auto&p:raw)rawSum+=p.weight;
    return std::abs(rawSum-4.0*arma::datum::pi)<1e-12;
}

void AddGeneralOrientationTests(std::vector<test_case>&cases)
{
    cases.push_back({"General orientation HS/SS/MultiSS 2D equivalence",test_general_orientation_hs_ss_multiss_2d_equivalence});
    cases.push_back({"General orientation HS/SS/MultiSS SO3 equivalence",test_general_orientation_hs_ss_multiss_so3_equivalence});
    cases.push_back({"General orientation explicit ZYZ equivalence",test_general_orientation_explicit_zyz_equivalence});
    cases.push_back({"General orientation parser explicit ZYZ equivalence",test_general_orientation_parser_explicit_zyz_equivalence});
    cases.push_back({"General orientation generated grid normalization",test_general_orientation_generated_weights_are_normalized});
}
