// Stochastic Hilbert relaxation: fixed-seed physics and integration regressions.
#include "Operator.h"
#include "PowderGrid.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include "ObjectParser.h"
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "RunSection.h"
#include "BasicTask.h"
#include "Interaction.h"
#include "Pulse.h"
#include "MSDParser.h"
#include "HSPropagator.h"
#include "HSStatePreparation.h"

namespace st_relaxation_tests
{
	SpinAPI::system_ptr Pair()
	{
		auto system = std::make_shared<SpinAPI::SpinSystem>("pair");
		system->Add(std::make_shared<SpinAPI::Spin>("E1", "spin=1/2;type=electron;"));
		system->Add(std::make_shared<SpinAPI::Spin>("E2", "spin=1/2;type=electron;"));
		return system;
	}
	SpinAPI::operator_ptr Channel(const SpinAPI::system_ptr &system,
		const std::string &options = "spins=E1,E2;rate=0.7;")
	{
		auto op = std::make_shared<SpinAPI::Operator>("ST", "type=relaxationdephasing;" + options);
		if (!op->Validate({system})) return nullptr;
		return op;
	}
	bool Validation()
	{
		auto system = Pair();
		system->Add(std::make_shared<SpinAPI::Spin>("T", "spin=1;"));
		for (const std::string spins : {"E1", "E1,E2,T", "E1,E1", "E1,T", "E1,missing", "E1,E2,missing", ""})
			if (Channel(system, "spins=" + spins + ";rate=1;")) return false;
		return Channel(system) != nullptr && Channel(system, "spins=E1,E2;rate=0;") != nullptr;
	}
	bool Algebra()
	{
		auto system = Pair(); auto op = Channel(system);
		SpinAPI::SpinSpace space(system);
		arma::sp_cx_mat ps, pt;
		if (!space.SingletTripletProjectors(op, ps, pt)) return false;
		const arma::cx_mat I = arma::eye<arma::cx_mat>(4,4), S(ps), T(pt), U(S-T);
		if (arma::norm(S*S-S,"fro") > 1e-13 || arma::norm(T*T-T,"fro") > 1e-13 ||
			arma::norm(S*T,"fro") > 1e-13 || arma::norm(T*S,"fro") > 1e-13 ||
			arma::norm(S+T-I,"fro") > 1e-13 || arma::norm(U*U-I,"fro") > 1e-13 ||
			arma::norm(U.t()*U-I,"fro") > 1e-13) return false;
		// Complex, non-Hermitian matrix tests the whole linear map, not only populations.
		arma::cx_mat rho(4,4);
		for (arma::uword i=0; i<16; ++i) rho(i)=arma::cx_double(0.13*i-0.5,0.07*i+0.3);
		SpinAPI::HilbertRelaxationCache density;
		arma::cx_mat drho, ssrho;
		if (!space.RelaxationOperator(op,density) || !space.ApplyRelaxationHilbert(density,rho,drho)) return false;
		const arma::cx_mat expected=0.35*(U*rho*U.t()-rho);
		space.UseSuperoperatorSpace(true); arma::cx_mat R; arma::cx_vec v;
		if (!space.RelaxationOperator(op,R) || !space.OperatorToSuperspace(rho,v) ||
			!space.OperatorFromSuperspace(R*v,ssrho)) return false;
		return arma::norm(expected-drho,"fro")<1e-12 && arma::norm(expected-ssrho,"fro")<1e-12;
	}
	bool AnalyticAndNorm()
	{
		auto system=Pair(); SpinAPI::SpinSpace space(system); auto op=Channel(system);
		SpinAPI::HilbertStochasticRelaxationCache cache; std::string error;
		if (!space.PrepareStochasticRelaxationHilbert({op},cache,error)) return false;
		const unsigned int N=32768; const double t=1.3, k=0.7;
		arma::cx_mat factors(4,N,arma::fill::zeros); factors.row(1).fill(0.6);
		const arma::cx_mat before=factors; std::mt19937 rng(519);
		if (!SpinAPI::ApplyStochasticRelaxationHilbert(cache,t,factors,rng,error)) return false;
		const arma::cx_mat rho=factors*factors.t()/static_cast<double>(N);
		const double p=-std::expm1(-k*t)/2.0;
		const double sigma=0.36*std::sqrt(p*(1-p)/N);
		if (std::abs(rho(2,2).real()-0.36*p)>6*sigma || std::abs(arma::trace(rho).real()-0.36)>1e-12) return false;
		for (arma::uword col=0;col<N;++col)
			if (std::abs(arma::cdot(factors.col(col),factors.col(col)).real()-0.36)>1e-13) return false;
		arma::cx_vec singlet={0,1/std::sqrt(2.0),-1/std::sqrt(2.0),0};
		arma::cx_vec triplet={0,1/std::sqrt(2.0),1/std::sqrt(2.0),0};
		return std::abs(arma::cdot(singlet,rho*triplet).real()-0.18*std::exp(-k*t))<6*sigma &&
			std::abs(arma::cdot(singlet,rho*singlet).real()-0.18)<1e-12 &&
			std::abs(arma::cdot(triplet,rho*triplet).real()-0.18)<1e-12;
	}
	bool MultipleChannelsAndPowder()
	{
		auto system=Pair(); system->Add(std::make_shared<SpinAPI::Spin>("E3","spin=1/2;"));
		auto a=Channel(system), b=Channel(system,"spins=E2,E3;rate=1.1;");
		SpinAPI::SpinSpace space(system); SpinAPI::HilbertStochasticRelaxationCache cache, rotated;
		std::string error; arma::mat rotation;
		if (!SpinAPI::CreateZYZRotationMatrix(0.7,0.9,-0.2,rotation) ||
			!space.PrepareStochasticRelaxationHilbert({a,b},cache,error) ||
			!space.PrepareStochasticRelaxationHilbert({a,b},rotated,error,&rotation)) return false;
		if (arma::norm(cache.terms[0].U*cache.terms[1].U-cache.terms[1].U*cache.terms[0].U,"fro")<0.1) return false;
		for (size_t j=0;j<2;++j) if(arma::norm(cache.terms[j].U-rotated.terms[j].U,"fro")>1e-12) return false;
		const unsigned int N=32768; const double dt=2.0;
		arma::cx_vec psi(8,arma::fill::zeros); psi(1)=1;
		arma::cx_mat factors=arma::repmat(psi,1,N);
		std::mt19937 rng(219); if(!SpinAPI::ApplyStochasticRelaxationHilbert(cache,dt,factors,rng,error)) return false;
		space.UseSuperoperatorSpace(true); arma::cx_mat R1,R2; arma::cx_vec v; arma::cx_mat reference;
		if(!space.RelaxationOperator(a,R1)||!space.RelaxationOperator(b,R2)||
			!space.OperatorToSuperspace(psi*psi.t(),v)||!space.OperatorFromSuperspace(arma::expmat(dt*(R1+R2))*v,reference)) return false;
		return arma::norm(factors*factors.t()/N-reference,"fro") < 6.0/std::sqrt(N);
	}
	bool ReproducibilityAndNoop()
	{
		auto system=Pair(); SpinAPI::SpinSpace space(system); std::string error;
		SpinAPI::HilbertStochasticRelaxationCache cache, zero;
		if(!space.PrepareStochasticRelaxationHilbert({Channel(system)},cache,error)||
			!space.PrepareStochasticRelaxationHilbert({Channel(system,"spins=E1,E2;rate=0;")},zero,error)||!zero.Empty())return false;
		std::mt19937 source(12), copy=source;
		auto a=SpinAPI::StochasticRelaxationGenerator(source,3), b=SpinAPI::StochasticRelaxationGenerator(source,3);
		if(source!=copy)return false;
		arma::cx_mat A(4,256,arma::fill::ones); A.row(1).zeros(); arma::cx_mat B=A, original=A;
		if(!SpinAPI::ApplyStochasticRelaxationHilbert(zero,1.0,A,source,error)||source!=copy||arma::norm(A-original,"fro")!=0)return false;
		if(!SpinAPI::ApplyStochasticRelaxationHilbert(cache,0,A,source,error)||source!=copy)return false;
		if(!SpinAPI::ApplyStochasticRelaxationHilbert(cache,1,A,a,error)||!SpinAPI::ApplyStochasticRelaxationHilbert(cache,1,B,b,error))return false;
		auto unsupported=std::make_shared<SpinAPI::Operator>("T1","type=relaxationt1;spins=E1;rate=1;");
		if(!unsupported->Validate({system})||space.PrepareStochasticRelaxationHilbert({unsupported},zero,error)||error.find("T1")==std::string::npos)return false;
		return arma::norm(A-B,"fro")==0 && a==b;
	}
	// A dummy uncoupled nucleus exercises the historical radical-pair layout.
	// H=0.8 Sz(E1) does not commute with the joint ST unitary.
	SpinAPI::system_ptr Fixture(const std::string &relaxation="type=relaxationdephasing;spins=E1,E2;rate=0.7;",
		bool dynamic=false, bool updown=false, bool field=true)
	{
		auto system=Pair();
		system->Add(std::make_shared<SpinAPI::Spin>("N","type=nucleus;spin=1/2;"));
		const std::vector<std::pair<std::string,std::string>> states={
			{"S","|1/2,-1/2>-|-1/2,1/2>"}, {"T0","|1/2,-1/2>+|-1/2,1/2>"},
			{"Tp","|1/2,1/2>"}, {"Tm","|-1/2,-1/2>"}, {"UD","|1/2,-1/2>"}};
		for(const auto &item:states)
		{
			auto state=std::make_shared<SpinAPI::State>(item.first,"spins(E1,E2)="+item.second+";");
			system->Add(state); if(!state->ParseFromSystem(*system)) return nullptr;
			if(item.first!="UD") system->Add(std::make_shared<SpinAPI::Transition>(
				"sink"+item.first,"type=sink;sourcestate="+item.first+";rate=0.2;",system));
		}
		if(field) system->Add(std::make_shared<SpinAPI::Interaction>("Z",
			"type=zeeman;spins=E1;field=0 0 0.8;ignoretensors=true;commonprefactor=false;prefactor=1;"));
		if(dynamic) system->Add(std::make_shared<SpinAPI::Interaction>("drive",
			"type=zeeman;spins=E1;field=1.1 0 0;ignoretensors=true;commonprefactor=false;prefactor=1;"
			"fieldtype=linearpolarized;frequency=1.3;phase=0.2;"));
		if(!relaxation.empty()) system->Add(std::make_shared<SpinAPI::Operator>("relax",relaxation));
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",
			std::string("initialstate=")+(updown?"UD":"S")+";initialstatecoherences=keep;"));
		if(!system->ValidateOperators({system}).empty() || !system->ValidateInteractions().empty() ||
			!system->ValidateTransitions({system}).empty()) return nullptr;
		return system;
	}
	struct Output { bool success=false; std::string data, log; std::vector<std::vector<double>> rows; };
	Output Run(const SpinAPI::system_ptr &system,const std::string &type,const std::string &options)
	{
		Output out; if(!system)return out;
		RunSection::RunSection run; run.Add(system);
		MSDParser::ObjectParser parser("task","type="+type+";"+options);
		if(!run.Add(MSDParser::ObjectType::Task,parser))return out;
		std::ostringstream data,log; data<<std::setprecision(16);
		auto task=run.GetTask("task"); task->SetDataStream(data);task->SetLogStream(log);
		out.success=run.Run(1);out.data=data.str();out.log=log.str();
		std::istringstream input(out.data);std::string line;
		while(std::getline(input,line))
		{
			std::istringstream row(line);double value;std::vector<double> values;
			while(row>>value) values.push_back(value);
			if(!values.empty())out.rows.push_back(values);
		}
		if(!out.success)std::cout<<"Task "<<type<<" failed:\n"<<out.log;
		return out;
	}
	const std::string common="totaltime=2;timestep=0.025;autoseed=false;seed=431;montecarlosamples=8192;precision=double;";
	bool Close(const Output &a,const Output &b,double tolerance,size_t skip=2)
	{
		if(!a.success||!b.success||a.rows.empty()||a.rows.size()!=b.rows.size())return false;
		double maxError=0;
		for(size_t r=0;r<a.rows.size();++r)
		{
			if(a.rows[r].size()!=b.rows[r].size())return false;
			for(size_t c=skip;c<a.rows[r].size();++c)
			{
				if(!std::isfinite(a.rows[r][c])||!std::isfinite(b.rows[r][c]))return false;
				maxError=std::max(maxError,std::abs(a.rows[r][c]-b.rows[r][c]));
			}
		}
		std::cout<<"max observable error="<<maxError<<", bound="<<tolerance<<std::endl;
		return maxError<tolerance;
	}
	bool GeneralStatic()
	{
		const std::string options=common+"dynamics=static;calculation=timeevolution;";
		auto direct=Run(Fixture(),"HSGeneral",options+"sampling=direct;propagationmethod=rk4;");
		auto coherent=Run(Fixture(""),"HSGeneral",options+"sampling=direct;propagationmethod=normal;");
		if(!direct.success||!coherent.success||direct.rows.empty()||
			std::abs(direct.rows.back()[2]-coherent.rows.back()[2])<0.07)return false;
		for(const std::string method:{"normal","autoexpm","krylov","rk4"})
		{
			auto stoch=Run(Fixture(),"HSGeneral",options+"sampling=stochastic;propagationmethod="+method+";krylovsize=8;");
			if(!Close(stoch,direct,6.0/(2*std::sqrt(8192.0))) ||
				stoch.log.find("no density propagation")==std::string::npos)return false;
			for(const auto &row:stoch.rows)
			{
				if(row.size()!=7)return false;
				double survival=0;for(size_t c=2;c<6;++c)survival+=row[c];
				if(std::abs(survival-std::exp(-0.2*row[1]))>2e-6)return false;
			}
		}
		return true;
	}
	bool GeneralDynamic()
	{
		const std::string options=common+"dynamics=dynamic;calculation=timeevolution;";
		auto direct=Run(Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",true),"HSGeneral",options+"sampling=direct;propagationmethod=rk4;");
		for(const std::string method:{"normal","rk4"})
		{
			auto stochastic=Run(Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",true),"HSGeneral",options+"sampling=stochastic;propagationmethod="+method+";");
			if(!Close(stochastic,direct,6.0/(2*std::sqrt(8192.0))))return false;
		}
		return true;
	}
	bool LegacyTasks()
	{
		for(const std::string dynamics:{"Static","Dynamic"})
			for(const std::string calculation:{"TimeEvo","Yields"})
			{
				const bool dynamic=dynamics=="Dynamic";
				const std::string options=common+"initialstate=singlet;propagationmethod=autoexpm;";
				auto fixture=[&](){return Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",dynamic);};
				auto direct=Run(fixture(),dynamics+"HS-Direct-"+calculation,options);
				auto stochastic=Run(fixture(),dynamics+"HS-Stoch-"+calculation,options);
				if(!Close(stochastic,direct,6.0/(2*std::sqrt(8192.0)),calculation=="Yields"?1:2))return false;
			}
		return true;
	}
	bool Spectra()
	{
		const std::string options=common+"method=timeevo;spinlist=E1,E2;integration=false;cidsp=false;powdersamplingpoints=0;";
		auto fixture=[](){return Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",false,true);};
		auto direct=Run(fixture(),"StaticHS-Direct-Spectra",options+"sampling=direct;propagationmethod=normal;");
		for(const std::string method:{"normal","autoexpm","krylov"})
		{
			auto stoch=Run(fixture(),"StaticHS-Direct-Spectra",options+"sampling=stochastic;propagationmethod="+method+";krylovsize=8;");
			if(!Close(stoch,direct,6.0/(2*std::sqrt(8192.0))))return false;
		}
		return true;
	}
	bool PowderAndRotation()
	{
		auto system=Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",false,true);
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",
			"initialstate=UD;initialstateframe=molecular;initialstatecoherences=keep;"));
		SpinAPI::SpinSpace space(system);arma::cx_mat density;std::string error;
		if(!RunSection::General::HS::HSStatePreparation::BuildInitialDensity(system,space,density,error))return false;
		SpinAPI::HilbertStateRotationCache cache;
		if(!space.CreateStateRotationCache(density,cache))return false;
		arma::cx_mat B;if(!space.FactorizeDensityMatrix(density,B,&error))return false;
		for(double theta:{0.0,0.3,1.7,3.141592653589793})
		{
			arma::mat rotation;arma::cx_mat sparse,dense;
			if(!SpinAPI::CreateZYZRotationMatrix(0.2,theta,0.4,rotation)||
				!space.RotateStateFactors(B,rotation,sparse)||!space.RotateStateFactors(B,rotation,cache,dense)||
				arma::norm(sparse*sparse.t()-dense*dense.t(),"fro")>1e-12)return false;
		}
		const std::string options=common+"spinlist=E1,E2;powdersamplingpoints=3;powdergammapoints=2;hamiltonianh0list=Z;";
		for(const std::string type:{"HSGeneral","StaticHS-Direct-Spectra"})
		{
			auto direct=Run(system,type,options+"sampling=direct;propagationmethod=normal;");
			auto stochastic=Run(system,type,options+"sampling=stochastic;propagationmethod=normal;");
			if(!Close(stochastic,direct,6.0/(2*std::sqrt(8192.0))))return false;
		}
		return true;
	}
	bool PulseTimeline()
	{
		// Zero drive isolates exact elapsed relaxation/reaction time, including
		// the 0.05-ns final intervals. Instant identity pulse adds no time.
		for(const std::string type:{"HSGeneral","StaticHS-Direct-Spectra"})
		{
			auto system=Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",false,true,false);
			system->Add(std::make_shared<SpinAPI::Pulse>("finite",
				"type=LongPulseStaticField;field=0 0 0;pulsetime=0.25;timestep=0.1;group=E1;"
				"prefactorlist=1,1,1;commonprefactorlist=false;ignoretensorslist=true;"));
			system->Add(std::make_shared<SpinAPI::Pulse>("instant",
				"type=InstantPulse;angle=0;rotationaxis=1 0 0;group=E1;"));
			if(!system->ValidatePulses().empty())return false;
			const auto out=Run(system,type,"totaltime=0.1;timestep=0.1;sampling=stochastic;spinlist=E1;"
				"autoseed=false;seed=519;montecarlosamples=32768;propagationmethod=normal;"
				"pulsesequence=[\"instant 0\",\"finite 0.15\"];printtimeframe=full;integration=false;cidsp=false;");
			if(!out.success||out.rows.empty())return false;
			bool partial=false,delay=false;
			for(const auto &row:out.rows)
			{
				if(row.size()!=5)return false;
				const double t=row[1];partial|=std::abs(t-0.25)<1e-12;delay|=std::abs(t-0.4)<1e-12;
				if(std::abs(row[4]-0.5*std::exp(-0.9*t))>6.0/(2*std::sqrt(32768.0)))return false;
			}
			if(!partial||!delay)return false;
		}
		return true;
	}
	bool Rejections()
	{
		const std::vector<std::string> types={"HSGeneral","StaticHS-Stoch-TimeEvo","StaticHS-Stoch-Yields",
			"DynamicHS-Stoch-TimeEvo","DynamicHS-Stoch-Yields","StaticHS-Direct-Spectra"};
		for(const auto &type:types)
		{
			auto out=Run(Fixture("type=relaxationt1;spins=E1;rate=0.7;"),type,
				common+"sampling=stochastic;spinlist=E1;initialstate=singlet;propagationmethod=autoexpm;");
			if(out.success||out.log.find("unraveling")==std::string::npos)return false;
			if(type!="HSGeneral" && type!="StaticHS-Direct-Spectra")
			{
				auto adaptive=Run(Fixture(),type,common+"initialstate=singlet;propagationmethod=krylov;");
				if(adaptive.success||adaptive.log.find("adaptive legacy Krylov")==std::string::npos)return false;
			}
		}
		for(const std::string type:{"StaticHS-Stoch-TimeEvo-Symm-Uncoupled","StaticHS-Stoch-Yields-Symm-Uncoupled",
			"StaticHS-Direct-TimeEvo-Symm-Uncoupled","StaticHS-Direct-Yields-Symm-Uncoupled"})
		{
			auto out=Run(Fixture(),type,common);
			if(out.success||out.log.find("separated-radical")==std::string::npos)return false;
		}
		for(const std::string type:{"HSGeneral","StaticHS-Direct-Spectra"})
		{
			auto out=Run(Fixture(),type,common+"calculation=yields;sampling=stochastic;method=timeinf;spinlist=E1;");
			if(out.success||out.log.find("timeinf")==std::string::npos)return false;
		}
		return true;
	}
	bool TaskNoopAndReproducibility()
	{
		for(const std::string type:{"HSGeneral","StaticHS-Direct-Spectra","StaticHS-Stoch-TimeEvo",
			"StaticHS-Stoch-Yields","DynamicHS-Stoch-TimeEvo","DynamicHS-Stoch-Yields"})
		{
			const std::string options="totaltime=0.2;timestep=0.05;autoseed=false;seed=71;montecarlosamples=32;"
				"sampling=stochastic;spinlist=E1;initialstate=singlet;propagationmethod=autoexpm;precision=double;";
			const bool dynamic=type.find("Dynamic")==0;
			// Reuse the same State objects: separately allocated entangled States
			// can differ in roundoff from their internal pointer-ordered factors.
			auto unchanged=Fixture("",dynamic);
			auto none=Run(unchanged,type,options);
			unchanged->Add(Channel(unchanged,"spins=E1,E2;rate=0;"));
			auto zero=Run(unchanged,type,options);
			auto relaxing=Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",dynamic);
			auto first=Run(relaxing,type,options);
			auto second=Run(relaxing,type,options);
			if(!none.success||!zero.success||none.rows.empty()||none.data!=zero.data||
				!first.success||!second.success||first.rows.empty()||first.data!=second.data)
			{
				std::cout<<"Reproducibility failure in "<<type<<"\nnone:\n"<<none.data<<"zero:\n"<<zero.data
					<<"first:\n"<<first.data<<"second:\n"<<second.data<<none.log<<std::endl;
				return false;
			}
		}
		return true;
	}
	bool ParserRejectsInvalidOperator()
	{
		const std::string path="/tmp/molspin-st-operator-"+std::to_string(getpid())+".msd";
		std::ofstream file(path);file<<"SpinSystem pair { Spin E1 { spin=1/2; } Spin E2 { spin=1/2; } "
			"Operator bad { type=relaxationdephasing;spins=E1,E2,missing;rate=1; } }\n";file.close();
		MSDParser::MSDParser parser(path);const bool rejected=!parser.Load();std::remove(path.c_str());return rejected;
	}
	bool SplitConvergence()
	{
		auto system=Pair();SpinAPI::SpinSpace space(system);auto op=Channel(system);
		arma::sp_cx_mat ps,pt,sz;space.SingletTripletProjectors(op,ps,pt);
		if(!space.CreateOperator(system->Spins()[0]->Sz(),system->Spins()[0],sz))return false;
		const arma::cx_mat H=1.9*arma::cx_mat(sz),U(ps-pt);
		space.UseSuperoperatorSpace(true);arma::cx_mat R;
		if(!space.RelaxationOperator(op,R))return false;
		const arma::cx_mat I=arma::eye<arma::cx_mat>(4,4);
		const arma::cx_mat L=arma::cx_double(0,-1)*(arma::kron(H,I)-arma::kron(I,H.st()))+R;
		arma::cx_vec psi={0,1/std::sqrt(2.0),-1/std::sqrt(2.0),0},v;space.OperatorToSuperspace(psi*psi.t(),v);
		arma::cx_mat exact;space.OperatorFromSuperspace(arma::expmat(L)*v,exact);
		std::vector<double> errors;
		for(int steps:{5,10,20,40})
		{
			const double dt=1.0/steps,p=-0.5*std::expm1(-0.7*dt/2);
			arma::cx_mat rho=psi*psi.t(),G=arma::expmat(arma::cx_double(0,-dt)*H);
			for(int step=0;step<steps;++step)
			{
				rho=(1-p)*rho+p*U*rho*U.t();rho=G*rho*G.t();rho=(1-p)*rho+p*U*rho*U.t();
			}
			errors.push_back(arma::norm(rho-exact,"fro"));
			std::cout<<"dt="<<dt<<" split error="<<errors.back()<<std::endl;
		}
		for(size_t j=1;j<errors.size();++j)if(errors[j-1]/errors[j]<3.8||errors[j-1]/errors[j]>4.2)return false;
		return true;
	}
	bool MonteCarloConvergence()
	{
		auto system=Pair();SpinAPI::SpinSpace space(system);SpinAPI::HilbertStochasticRelaxationCache cache;std::string error;
		if(!space.PrepareStochasticRelaxationHilbert({Channel(system)},cache,error))return false;
		const double p=-0.5*std::expm1(-0.7);std::vector<double> rms;
		for(unsigned int n:{256U,1024U,4096U})
		{
			double mse=0;
			for(unsigned int seed=1;seed<=64;++seed)
			{
				std::mt19937 rng(seed);arma::cx_mat B(4,n,arma::fill::zeros);B.row(1).ones();
				if(!SpinAPI::ApplyStochasticRelaxationHilbert(cache,1,B,rng,error))return false;
				const double estimate=arma::accu(arma::square(arma::abs(B.row(2))))/n;
				mse+=(estimate-p)*(estimate-p);
			}
			rms.push_back(std::sqrt(mse/64));const double expected=std::sqrt(p*(1-p)/n);
			std::cout<<"N="<<n<<" RMS="<<rms.back()<<" binomial sigma="<<expected<<std::endl;
			if(rms.back()<0.5*expected||rms.back()>1.5*expected)return false;
		}
		return rms.front()/rms.back()>2.5;
	}
	bool LargeSparseFactors()
	{
		auto system=Pair();for(int j=0;j<11;++j)system->Add(std::make_shared<SpinAPI::Spin>("N"+std::to_string(j),"spin=1/2;type=nucleus;"));
		auto state=std::make_shared<SpinAPI::State>("UD","spins(E1,E2)=|1/2,-1/2>;");system->Add(state);
		if(!state->ParseFromSystem(*system))return false;
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=UD;initialstateframe=molecular;initialstatecoherences=keep;"));
		SpinAPI::SpinSpace space(system);SpinAPI::HilbertStochasticRelaxationCache cache;std::string error;
		if(!space.PrepareStochasticRelaxationHilbert({Channel(system)},cache,error))return false;
		using namespace RunSection::General::HS;
		HSExecutionPlan plan;plan.sampling=Sampling::Stochastic;plan.monteCarloSamples=4;plan.orientation=OrientationMode::PowderSO3;
		HSPreparedState prepared;std::mt19937 rng(12);std::ostringstream log;
		if(!HSStatePreparation::Prepare(plan,system,space,prepared,rng,log,error))return false;
		if(!prepared.density.is_empty()||prepared.factors.n_rows!=8192||prepared.factors.n_cols!=4||
			cache.terms[0].U.n_nonzero>2*8192)return false;
		arma::mat rotation;arma::cx_mat B;SpinAPI::CreateZYZRotationMatrix(0.1,0.7,0.3,rotation);
		return space.RotateStateFactors(prepared.factors,rotation,B)&&
			SpinAPI::ApplyStochasticRelaxationHilbert(cache,1,B,rng,error)&&B.is_finite();
	}

	bool UnequalReactionsAndFiniteDrive()
	{
		for(const std::string type:{"HSGeneral","StaticHS-Direct-Spectra"})
		{
			auto fixture=[]() {
				auto system=Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",false,true);
				system->Add(std::make_shared<SpinAPI::Transition>("extra","type=sink;sourcestate=S;rate=0.4;",system));
				system->Add(std::make_shared<SpinAPI::Pulse>("drive",
					"type=LongPulseStaticField;field=1.2 0 0;pulsetime=0.4;timestep=0.025;group=E1;"
					"prefactorlist=1,1,1;commonprefactorlist=false;ignoretensorslist=true;"));
				if(!system->ValidatePulses().empty()||!system->ValidateTransitions({system}).empty())return SpinAPI::system_ptr();
				return system;
			};
			const std::string options=common+"spinlist=E1,E2;pulsesequence=[\"drive 0.2\"];printtimeframe=full;";
			auto direct=Run(fixture(),type,options+"sampling=direct;propagationmethod=normal;");
			auto stochastic=Run(fixture(),type,options+"sampling=stochastic;propagationmethod=normal;");
			if(!Close(stochastic,direct,6.0/(2*std::sqrt(8192.0))))return false;
		}
		// Both static legacy exponential paths must apply the same channel.
		for(const std::string calculation:{"TimeEvo","Yields"})
		{
			const std::string options=common+"initialstate=singlet;propagationmethod=normal;yieldcorrections=false;";
			if(!Close(Run(Fixture(),"StaticHS-Stoch-"+calculation,options),
				Run(Fixture(),"StaticHS-Direct-"+calculation,options),
				6.0/(2*std::sqrt(8192.0)),calculation=="Yields"?1:2))return false;
		}
		return true;
	}
	bool NoncommutingMonteCarloConvergence()
	{
		using namespace RunSection::General::HS;
		auto system=Pair();SpinAPI::SpinSpace space(system);std::string error;auto op=Channel(system);
		HSRelaxationContext context;
		if(!space.PrepareStochasticRelaxationHilbert({op},context.stochasticCache,error))return false;
		arma::sp_cx_mat H,K(4,4),ps,pt;
		if(!space.CreateOperator(system->Spins()[0]->Sz(),system->Spins()[0],H) ||
			!space.SingletTripletProjectors(op,ps,pt))return false;
		H*=1.9;const arma::cx_mat U(ps-pt);
		if(arma::norm(H*U-U*H,"fro")<0.1)return false;
		const double dt=0.05,p=-0.5*std::expm1(-0.7*dt/2);
		arma::cx_vec psi={0,1/std::sqrt(2.0),-1/std::sqrt(2.0),0};
		arma::cx_mat rho=psi*psi.t(),G=arma::expmat(arma::cx_double(0,-dt)*arma::cx_mat(H));
		for(int step=0;step<20;++step){rho=(1-p)*rho+p*U*rho*U.t();rho=G*rho*G.t();rho=(1-p)*rho+p*U*rho*U.t();}
		const double expected=arma::cdot(psi,rho*psi).real();
		HSExecutionPlan plan;plan.propagation=PropagationMethod::Exponential;HSPropagator propagator(plan,space);
		std::vector<double> errors;
		for(unsigned int n:{256U,1024U,4096U})
		{
			double mse=0;
			for(unsigned int seed=1;seed<=32;++seed)
			{
				std::mt19937 rng(seed);arma::cx_mat B=arma::repmat(psi,1,n);
				for(int step=0;step<20;++step)if(!propagator.StepStochastic(H,K,dt,B,context,rng,error))return false;
				const double estimate=arma::accu(arma::square(arma::abs(psi.t()*B)))/n;
				mse+=(estimate-expected)*(estimate-expected);
			}
			errors.push_back(std::sqrt(mse/32));
			std::cout<<"noncommuting N="<<n<<" RMS="<<errors.back()<<std::endl;
			if(errors.back()>3/std::sqrt(static_cast<double>(n)))return false;
		}
		return errors.front()/errors.back()>2.5;
	}

	bool PowderThreadReproducibility()
	{
#ifdef _OPENMP
		const int previous=omp_get_max_threads();
		auto system=Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",false,true);
		const std::string options="sampling=stochastic;montecarlosamples=128;autoseed=false;seed=71;"
			"totaltime=0.3;timestep=0.05;spinlist=E1;hamiltonianh0list=Z;powdersamplingpoints=4;"
			"powdergammapoints=2;propagationmethod=normal;";
		omp_set_num_threads(1);const auto serial=Run(system,"StaticHS-Direct-Spectra",options);
		omp_set_num_threads(2);const auto parallel=Run(system,"StaticHS-Direct-Spectra",options);
		omp_set_num_threads(previous);
		return Close(serial,parallel,1e-12);
#else
		return true;
#endif
	}

	bool SUZInterleavedEntangledSupport()
	{
		auto system=std::make_shared<SpinAPI::SpinSystem>("interleaved");
		for(const std::string name:{"A","B","C","N"})
			system->Add(std::make_shared<SpinAPI::Spin>(name,"spin=1/2;"));
		auto state=std::make_shared<SpinAPI::State>("initial",
			"spins(A,C)=|1/2,-1/2>-|-1/2,1/2>;spin(B)=|1/2>;");
		system->Add(state);if(!state->ParseFromSystem(*system))return false;
		// The shared-system constructor sorts pointers; request this exact basis
		// explicitly so the interleaved layout does not depend on heap allocation.
		SpinAPI::SpinSpace space(system->Spins());arma::sp_cx_mat support;
		if(!space.GetState(state,support))return false;
		// Independent explicit ket oracle in A,B,C,N order.
		arma::cx_mat basis(16,2,arma::fill::zeros);
		basis(2,0)=basis(3,1)=1/std::sqrt(2.0);
		basis(8,0)=basis(9,1)=-1/std::sqrt(2.0);
		const arma::cx_mat expectedSupport=basis*basis.t();
		if(arma::norm(arma::cx_mat(support)-expectedSupport,"fro")>1e-12)return false;
		SpinAPI::HilbertTraceSampleSet samples;std::mt19937 rng(71);std::string error;
		if(!space.BuildTraceSamples(state,8192,SpinAPI::TraceSamplingMethod::SUZ,rng,samples,&error)||
			samples.sampledSubspaceDimension!=2)return false;
		const double supportError=arma::norm(support*samples.factors-samples.factors,"fro");
		const double densityError=arma::norm(samples.factors*samples.factors.t()/8192-expectedSupport/2,"fro");
		std::cout<<"interleaved SU(Z): support error="<<supportError<<", density error="<<densityError<<std::endl;
		return supportError<1e-12 && densityError<6/std::sqrt(8192.0);
	}
	bool SUZEntangledBasisPermutations()
	{
		auto system=std::make_shared<SpinAPI::SpinSystem>("permutations");
		for(const std::string name:{"A","B","C","N"})
			system->Add(std::make_shared<SpinAPI::Spin>(name,"spin=1/2;"));
		auto state=std::make_shared<SpinAPI::State>("initial",
			"spins(A,B,C)=|1/2,1/2,-1/2>-|-1/2,-1/2,1/2>;");
		system->Add(state);if(!state->ParseFromSystem(*system))return false;
		const auto spins=system->Spins();
		std::vector<size_t> permutation{0,1,2,3};
		do
		{
			std::vector<SpinAPI::spin_ptr> ordered;
			for(auto index:permutation)ordered.push_back(spins[index]);
			SpinAPI::SpinSpace space(ordered);
			// Explicit ket oracle: bit 0 is up, bit 1 is down, independently
			// construct (+,+,-) - (-,-,+) and both nuclear basis states.
			arma::cx_mat basis(16,2,arma::fill::zeros);
			for(size_t nucleus=0;nucleus<2;++nucleus)
				for(size_t term=0;term<2;++term)
				{
					const std::vector<size_t> bits{term,term,1-term,nucleus};
					size_t row=0;for(auto index:permutation)row=2*row+bits[index];
					basis(row,nucleus)=(term==0?1:-1)/std::sqrt(2.0);
				}
			const arma::cx_mat expected=basis*basis.t();
			arma::cx_mat dense;arma::sp_cx_mat sparse;arma::cx_vec pure;
			if(!space.GetState(state,dense)||!space.GetState(state,sparse)||!space.GetState(state,pure))return false;
			if(arma::norm(dense-expected,"fro")>1e-12 ||
				arma::norm(arma::cx_mat(sparse)-expected,"fro")>1e-12 ||
				arma::norm(pure-(basis.col(0)+basis.col(1))/std::sqrt(2.0),2)>1e-12)return false;
			SpinAPI::HilbertTraceSampleSet samples;std::mt19937 rng(71);std::string error;
			if(!space.BuildTraceSamples(state,2048,SpinAPI::TraceSamplingMethod::SUZ,rng,samples,&error)||
				samples.sampledSubspaceDimension!=2)return false;
			if(arma::norm(expected*samples.factors-samples.factors,"fro")>1e-12 ||
				arma::norm(samples.factors*samples.factors.t()/2048-expected/2,"fro")>6/std::sqrt(2048.0))return false;
		}while(std::next_permutation(permutation.begin(),permutation.end()));
		return true;
	}
	bool SUZCoupledNucleiAndRelaxation()
	{
		for(bool dynamic:{false,true})
		{
			auto system=Fixture("type=relaxationdephasing;spins=E1,E2;rate=0.7;",dynamic);
			system->Add(std::make_shared<SpinAPI::Spin>("N2","type=nucleus;spin=1;"));
			system->Add(std::make_shared<SpinAPI::Interaction>("hyperfine1",
				"type=hyperfine;group1=E1;group2=N;tensor=anisotropic(0.4,0.7,1.1);"
				"commonprefactor=false;prefactor=1;ignoretensors=true;"));
			system->Add(std::make_shared<SpinAPI::Interaction>("hyperfine2",
				"type=hyperfine;group1=E2;group2=N2;tensor=anisotropic(0.3,0.2,0.5);"
				"commonprefactor=false;prefactor=1;ignoretensors=true;"));
			// Fixture already validated Z/drive. Parse only the newly added groups.
			for (const auto &interaction : system->Interactions())
				if (!interaction->IsValid() && (!interaction->ParseSpinGroups(system->Spins()) || !interaction->IsValid())) return false;
			const std::string options=common+std::string("dynamics=")+(dynamic?"dynamic;":"static;")+"samplingmethod=suz;";
			auto direct=Run(system,"HSGeneral",options+"sampling=direct;propagationmethod=rk4;");
			auto stochastic=Run(system,"HSGeneral",options+"sampling=stochastic;propagationmethod=autoexpm;");
			if(!Close(stochastic,direct,6.0/(2*std::sqrt(8192.0))))return false;
			// This exercises Z=6 rather than an uncoupled spectator trace.
			if(stochastic.log.find("subspace dimension 6")==std::string::npos)return false;
			for(const auto &row:stochastic.rows)
			{
				double survival=0;for(size_t c=2;c<6;++c)survival+=row[c];
				if(std::abs(survival-std::exp(-0.2*row[1]))>2e-6)return false;
			}
		}
		return true;
	}

}

void AddStochasticRelaxationTests(std::vector<test_case> &cases)
{
	using namespace st_relaxation_tests;
	cases.push_back({"ST relaxation validates its spin domain",Validation});
	cases.push_back({"ST projectors and three generator representations agree",Algebra});
	cases.push_back({"Stochastic ST analytic coherence law and norm",AnalyticAndNorm});
	cases.push_back({"Noncommuting stochastic channels and powder invariance",MultipleChannelsAndPowder});
	cases.push_back({"Stochastic relaxation independent seeded RNG and no-op",ReproducibilityAndNoop});
	cases.push_back({"HSGeneral stochastic ST static propagation and Haberkorn survival",GeneralStatic});
	cases.push_back({"HSGeneral stochastic ST dynamic propagation",GeneralDynamic});
	cases.push_back({"Four legacy stochastic tasks match density ST references",LegacyTasks});
	cases.push_back({"DirectSpectra stochastic ST three propagators",Spectra});
	cases.push_back({"ST molecular powder and sparse factor rotations",PowderAndRotation});
	cases.push_back({"ST finite pulse, partial step, delay and instant timeline",PulseTimeline});
	cases.push_back({"Stochastic unsupported operators and representations reject",Rejections});
	cases.push_back({"Six stochastic task families seeded repeat and zero-op regression",TaskNoopAndReproducibility});
	cases.push_back({"MSD parser rejects invalid ST instead of deleting physics",ParserRejectsInvalidOperator});
	cases.push_back({"ST symmetric split second-order convergence",SplitConvergence});
	cases.push_back({"ST Monte Carlo RMS convergence over independent seeds",MonteCarloConvergence});
	cases.push_back({"ST large Hilbert sparse factor preparation and rotation",LargeSparseFactors});
	cases.push_back({"ST unequal reaction rates and finite microwave drive",UnequalReactionsAndFiniteDrive});
	cases.push_back({"ST noncommuting Hamiltonian trajectory convergence",NoncommutingMonteCarloConvergence});
	cases.push_back({"ST powder RNG independent of OpenMP thread scheduling",PowderThreadReproducibility});
	cases.push_back({"SU(Z) interleaved entangled State preserves physical support",SUZInterleavedEntangledSupport});
	cases.push_back({"SU(Z) three-spin entangled State in all basis permutations",SUZEntangledBasisPermutations});
	cases.push_back({"HSGeneral SU(Z) with coupled nuclei and ST relaxation",SUZCoupledNucleiAndRelaxation});
}
