/////////////////////////////////////////////////////////////////////////
// HSExecutionPlan implementation (RunSection::General::HS)
// ------------------
// Input normalization and validation for the modular Hilbert-space execution plan.
// Unsupported physics is rejected explicitly rather than mapped to another model.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSExecutionPlan.h"
#include "ObjectParser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace RunSection::General::HS
{
	namespace
	{
		std::string Lower(std::string value)
		{
			std::transform(value.begin(), value.end(), value.begin(),
				[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return value;
		}

		bool ReadString(const MSDParser::ObjectParser &properties,
			const std::initializer_list<const char *> &keys, std::string &value)
		{
			for (const char *key : keys)
			{
				if (properties.Get(key, value))
				{
					value = Lower(value);
					return true;
				}
			}
			return false;
		}

		bool OneOf(const std::string &value, const std::initializer_list<const char *> &choices)
		{
			for (const char *choice : choices)
				if (value == choice)
					return true;
			return false;
		}

		bool ParseExplicitOrientation(const std::string &value,
			double &alpha, double &beta, double &gamma, double &weight)
		{
			std::string normalized = value;
			for (char &c : normalized)
				if (c == ',' || c == '[' || c == ']' || c == '(' || c == ')') c = ' ';
			std::istringstream stream(normalized);
			if (!(stream >> alpha >> beta)) return false;
			if (!(stream >> gamma)) gamma = 0.0;
			if (!(stream >> weight)) weight = 1.0;
			return std::isfinite(alpha) && std::isfinite(beta) &&
				std::isfinite(gamma) && std::isfinite(weight);
		}
	}

	bool ResolveExecutionPlan(const MSDParser::ObjectParser &properties,
		HSExecutionPlan &plan, std::string &error)
	{
		plan = HSExecutionPlan();
		error.clear();

		std::string value = "static";
		ReadString(properties, {"dynamics", "timedependence", "time_dependence"}, value);
		if (OneOf(value, {"timeindependent", "time-independent"})) value = "static";
		if (OneOf(value, {"timedependent", "time-dependent"})) value = "dynamic";
		if (value == "static") plan.dynamics = Dynamics::Static;
		else if (value == "dynamic") plan.dynamics = Dynamics::Dynamic;
		else { error = "dynamics must be static or dynamic"; return false; }

		value = "timeevolution";
		ReadString(properties, {"calculation", "calculationmode", "calculation_mode"}, value);
		if (OneOf(value, {"timeevo", "time-evolution", "time_evolution"})) value = "timeevolution";
		if (OneOf(value, {"yield", "quantumyield", "quantumyields"})) value = "yields";
		if (OneOf(value, {"spectra", "spectrum", "resonance"}))
		{
			error = "spectroscopy is a standalone task; use StaticHS-Direct-Spectra (or StaticHS-Resonance-Spectra for the dedicated resonance algorithm)";
			return false;
		}
		if (value == "timeevolution") plan.calculation = Calculation::TimeEvolution;
		else if (value == "yields") plan.calculation = Calculation::Yields;
		else { error = "calculation must be timeevolution or yields"; return false; }

		value = "direct";
		ReadString(properties, {"sampling", "tracesampling", "trace_sampling"}, value);
		if (OneOf(value, {"trace", "montecarlo", "monte-carlo", "mc"})) value = "stochastic";
		if (value == "direct") plan.sampling = Sampling::Direct;
		else if (value == "stochastic") plan.sampling = Sampling::Stochastic;
		else { error = "sampling must be direct or stochastic"; return false; }

		std::string approximation = "full";
		ReadString(properties,
			{"approximation", "hamiltonianapproximation", "hamiltonian_approximation"}, approximation);
		bool secular = false;
		if (properties.Get("secularization", secular) || properties.Get("secular", secular))
		{
			approximation = secular ? "secular" : "full";
		}
		if (OneOf(approximation, {"rwa", "rotatingwave", "rotating-wave"}))
		{
			error = "approximation=rwa is not a static Hamiltonian secular approximation; configure the oscillatory drive Interaction/Pulse explicitly and use approximation=full or secular/highfield";
			return false;
		}
		if (OneOf(approximation, {"highfield", "high-field"})) approximation = "secular";
		if (OneOf(approximation, {"nonsecular", "non-secular", "exact"})) approximation = "full";
		if (approximation == "full") plan.approximation = SpinAPI::HamiltonianApproximation::Full;
		else if (approximation == "secular") plan.approximation = SpinAPI::HamiltonianApproximation::Secular;
		else { error = "approximation must be full or secular"; return false; }

		// Powder-grid input follows the shared SpinAPI vocabulary. `powdergrid`
		// is canonical for General tasks; `powdergridtype` is retained only as a
		// compatibility alias for existing resonance inputs. `uniform` is the
		// established golden-angle/Fibonacci constructor in SpinAPI.
		std::string powderGridName;
		const bool powderGridSpecified = ReadString(properties,
			{"powdergrid", "powder_grid", "powdergridtype"}, powderGridName);
		if (powderGridSpecified)
		{
			if (OneOf(powderGridName, {"uniform", "golden", "fibonacci"}))
				plan.powderGridType = SpinAPI::PowderGridType::Uniform;
			else if (powderGridName == "sophe")
				plan.powderGridType = SpinAPI::PowderGridType::Sophe;
			else if (powderGridName == "octant")
				plan.powderGridType = SpinAPI::PowderGridType::Octant;
			else
			{
				error = "powdergrid must be uniform, sophe, or octant";
				return false;
			}
		}

		const bool powderPointsSpecified = properties.Get("powdersamplingpoints", plan.powderPoints);
		const bool powderGridSizeSpecified = properties.Get("powdergridsize", plan.powderGridSize);
		const bool powderSymmetrySpecified = ReadString(properties,
			{"powdersymmetry", "powdergridsymmetry"}, plan.powderSymmetry);

		std::string powderDomainName;
		const bool powderDomainSpecified = ReadString(properties, {"powderdomain"}, powderDomainName);
		if (powderDomainSpecified)
		{
			if (OneOf(powderDomainName, {"upper", "hemisphere", "upperhemisphere"}))
				plan.powderDomain = SpinAPI::PowderGridDomain::UpperHemisphere;
			else if (OneOf(powderDomainName, {"full", "fullsphere", "sphere"}))
				plan.powderDomain = SpinAPI::PowderGridDomain::FullSphere;
			else
			{
				error = "powderdomain must be upper or full";
				return false;
			}
		}

		bool powderFullSphere = false;
		const bool powderFullSphereSpecified =
			properties.Get("powderfullsphere", powderFullSphere) ||
			properties.Get("powder_full_sphere", powderFullSphere);
		if (powderFullSphereSpecified)
		{
			const SpinAPI::PowderGridDomain legacyDomain = powderFullSphere
				? SpinAPI::PowderGridDomain::FullSphere
				: SpinAPI::PowderGridDomain::UpperHemisphere;
			if (powderDomainSpecified && plan.powderDomain != legacyDomain)
			{
				error = "powderdomain conflicts with powderfullsphere";
				return false;
			}
			plan.powderDomain = legacyDomain;
		}

		properties.Get("powdergammapoints", plan.powderGammaPoints);
		const bool powderGammaSpecified =
			properties.Get("powdergamma", plan.powderGammaOffset) ||
			properties.Get("powder_gamma", plan.powderGammaOffset);
		if (plan.powderGammaPoints < 1)
		{
			error = "powdergammapoints must be at least one";
			return false;
		}

		std::string explicitOrientation;
		bool explicitOrientationSpecified = false;
		if (properties.Get("powderorientation", explicitOrientation) ||
			properties.Get("powder_orientation", explicitOrientation))
		{
			if (powderGammaSpecified)
			{
				error = "canonical powderorientation already contains the ZYZ gamma angle; do not combine it with powdergamma";
				return false;
			}
			// Internally HS stores the canonical ZYZ triplet as
			// (powderGammaOffset, explicitTheta, explicitPhi). The shared
			// orientation adapter maps these to (alpha,beta,gamma).
			if (!ParseExplicitOrientation(explicitOrientation, plan.powderGammaOffset,
				plan.explicitTheta, plan.explicitPhi, plan.explicitWeight))
			{
				error = "powderorientation must contain alpha beta [gamma [weight]]";
				return false;
			}
			explicitOrientationSpecified = true;
		}
		else
		{
			double theta = 0.0;
			double phi = 0.0;
			double weight = 1.0;
			const bool thetaSpecified = properties.Get("powdertheta", theta) ||
				properties.Get("powder_theta", theta);
			const bool phiSpecified = properties.Get("powderphi", phi) ||
				properties.Get("powder_phi", phi);
			const bool weightSpecified = properties.Get("powderweight", weight) ||
				properties.Get("powder_weight", weight);
			if (thetaSpecified || phiSpecified || weightSpecified)
			{
				if (!(thetaSpecified && phiSpecified) || !std::isfinite(theta) ||
					!std::isfinite(phi) || !std::isfinite(weight))
				{
					error = "explicit powder orientation requires finite powdertheta and powderphi [powderweight]";
					return false;
				}
				plan.explicitTheta = theta;
				plan.explicitPhi = phi;
				plan.explicitWeight = weight;
				explicitOrientationSpecified = true;
			}
		}

		if (explicitOrientationSpecified)
		{
			if (powderGridSpecified || powderPointsSpecified || powderGridSizeSpecified ||
				powderSymmetrySpecified || powderDomainSpecified || powderFullSphereSpecified ||
				plan.powderGammaPoints > 1)
			{
				error = "explicit powderorientation cannot be combined with generated powder-grid settings";
				return false;
			}
			plan.orientation = OrientationMode::Explicit;
		}
		else
		{
			bool generatedPowderGrid = false;
			switch (plan.powderGridType)
			{
			case SpinAPI::PowderGridType::Uniform:
				if (powderGridSizeSpecified || powderSymmetrySpecified)
				{
					error = "powdergridsize/powdersymmetry are only valid with powdergrid=sophe";
					return false;
				}
				if (powderGridSpecified && !powderPointsSpecified)
				{
					error = "powdergrid=uniform requires powdersamplingpoints";
					return false;
				}
				if (powderPointsSpecified && plan.powderPoints < 1)
				{
					error = "powdersamplingpoints must be at least one";
					return false;
				}
				generatedPowderGrid = (powderGridSpecified && powderPointsSpecified) || plan.powderPoints > 1;
				break;

			case SpinAPI::PowderGridType::Sophe:
				if (powderPointsSpecified)
				{
					error = "powdersamplingpoints is not used by powdergrid=sophe; use powdergridsize";
					return false;
				}
				if (powderDomainSpecified || powderFullSphereSpecified)
				{
					error = "powderdomain/powderfullsphere do not apply to powdergrid=sophe; powdersymmetry defines the SOPHE domain";
					return false;
				}
				if (plan.powderGridSize < 1)
				{
					error = "powdergridsize must be at least one for powdergrid=sophe";
					return false;
				}
				{
					SpinAPI::SopheGridParameters parameters;
					if (!SpinAPI::GetSopheGridParameters(plan.powderSymmetry, parameters))
					{
						error = "invalid powdersymmetry for powdergrid=sophe";
						return false;
					}
				}
				generatedPowderGrid = powderGridSpecified;
				break;

			case SpinAPI::PowderGridType::Octant:
				if (!powderGridSpecified)
				{
					error = "powdergrid=octant must be selected explicitly";
					return false;
				}
				if (!powderPointsSpecified || plan.powderPoints < 1)
				{
					error = "powdergrid=octant requires positive powdersamplingpoints";
					return false;
				}
				if (powderGridSizeSpecified || powderSymmetrySpecified)
				{
					error = "powdergridsize/powdersymmetry are only valid with powdergrid=sophe";
					return false;
				}
				if (powderDomainSpecified || powderFullSphereSpecified)
				{
					error = "powdergrid=octant has a fixed symmetry-reduced octant domain";
					return false;
				}
				generatedPowderGrid = true;
				break;
			}

			if ((powderDomainSpecified || powderFullSphereSpecified) && !generatedPowderGrid)
			{
				error = "powderdomain/powderfullsphere requires a generated powder grid";
				return false;
			}
			if (plan.powderGammaPoints > 1 && !generatedPowderGrid)
			{
				error = "powdergammapoints greater than one requires a generated theta/phi powder grid";
				return false;
			}
			if (plan.powderGridType == SpinAPI::PowderGridType::Uniform &&
				plan.powderGammaPoints > 1 && plan.powderPoints <= 1)
			{
				error = "uniform SO(3) powder sampling requires powdersamplingpoints greater than one";
				return false;
			}
			if (plan.powderGridType == SpinAPI::PowderGridType::Uniform &&
				plan.powderGammaPoints > 1 && !powderDomainSpecified &&
				!powderFullSphereSpecified)
			{
				// Gamma resolves the third Euler angle. Without an explicit symmetry
				// reduction, the associated theta/phi directions must cover the full
				// sphere for a complete SO(3) integral.
				plan.powderDomain = SpinAPI::PowderGridDomain::FullSphere;
				plan.powderDomainAutoExpanded = true;
			}

			if (plan.powderGammaPoints > 1)
				plan.orientation = OrientationMode::PowderSO3;
			else if (generatedPowderGrid)
				plan.orientation = OrientationMode::Powder2D;
		}

		bool powderFlag = false;
		const bool powderSpecified = properties.Get("powderaveraging", powderFlag) ||
			properties.Get("powder", powderFlag);
		if (powderSpecified && powderFlag && plan.orientation == OrientationMode::Identity)
		{
			error = "powderaveraging=true requires a generated powder grid or an explicit powderorientation";
			return false;
		}
		if (powderSpecified && !powderFlag && plan.orientation != OrientationMode::Identity)
		{
			error = "powderaveraging=false conflicts with explicit powder sampling keywords";
			return false;
		}

		properties.Get("totaltime", plan.totalTime);
		properties.Get("timestep", plan.timeStep);
		if (!std::isfinite(plan.totalTime) || plan.totalTime < 0.0) { error = "totaltime must be finite and non-negative"; return false; }
		if (!std::isfinite(plan.timeStep) || plan.timeStep <= 0.0) { error = "timestep must be finite and positive"; return false; }

		value = "normal";
		ReadString(properties, {"propagationmethod", "propagator"}, value);
		if (OneOf(value, {"normal", "exp", "exponential", "matrixexponential", "matrix-exponential"}))
			plan.propagation = PropagationMethod::Exponential;
		else if (value == "autoexpm") plan.propagation = PropagationMethod::AutoExpm;
		else if (value == "krylov") plan.propagation = PropagationMethod::Krylov;
		else if (OneOf(value, {"rk4", "explicit"})) plan.propagation = PropagationMethod::RK4;
		else { error = "propagationmethod must be normal/exp, autoexpm, krylov, or rk4"; return false; }
		properties.Get("precision", plan.precision);
		properties.Get("krylovsize", plan.krylovSize);
		plan.krylovToleranceSpecified = properties.Get("krylovtol", plan.requestedKrylovTolerance);
		if (plan.krylovSize < 1)
			plan.krylovSize = 16;
		if (plan.krylovToleranceSpecified &&
			(!std::isfinite(plan.requestedKrylovTolerance) || !(plan.requestedKrylovTolerance > 0.0)))
		{
			error = "krylovtol must be finite and positive when supplied";
			return false;
		}

		int mcSamples = static_cast<int>(plan.monteCarloSamples);
		properties.Get("montecarlosamples", mcSamples);
		if (mcSamples < 1) { error = "montecarlosamples must be at least one"; return false; }
		plan.monteCarloSamples = static_cast<arma::uword>(mcSamples);
		if (!ReadString(properties, {"samplingmethod"}, plan.samplingMethod))
			plan.samplingMethod = "suz";
		if (!OneOf(plan.samplingMethod, {"suz", "coherent"}))
		{
			error = "samplingmethod must be suz or coherent";
			return false;
		}
		properties.Get("autoseed", plan.autoSeed);
		properties.Get("seed", plan.seed);

		plan.hasH0List = properties.GetList("hamiltonianh0list", plan.h0List, ',') && !plan.h0List.empty();
		plan.hasH1List = properties.GetList("hamiltonianh1list", plan.h1List, ',') && !plan.h1List.empty();
		plan.hasInitialStateHamiltonian =
			properties.GetList("initialstatehamiltonian", plan.initialStateHamiltonian, ',') &&
			!plan.initialStateHamiltonian.empty();
		const bool transitionYieldsSpecified = properties.Get("transitionyields", plan.transitionYields);
		properties.Get("yieldcorrections", plan.yieldCorrections);
		properties.GetList("spinlist", plan.spinList, ',');
		const bool cidspSpecified = properties.Get("cidsp", plan.cidsp);
		bool cidnpAlias = false;
		const bool cidnpSpecified = properties.Get("cidnp", cidnpAlias);
		if (cidnpSpecified && cidnpAlias) plan.cidsp = true;

		std::string observableMode;
		const bool observableModeSpecified = ReadString(properties,
			{"observables", "observablemode", "observable_mode"}, observableMode);
		if (observableModeSpecified)
		{
			if (OneOf(observableMode, {"states", "state", "populations", "statepopulations", "state-populations"}))
				plan.observableMode = ObservableMode::StatePopulations;
			else if (OneOf(observableMode, {"spins", "spin", "polarization", "polarizations", "spinpolarization", "spin-polarization"}))
				plan.observableMode = ObservableMode::SpinPolarization;
			else if (OneOf(observableMode, {"transitionyields", "transition-yields", "yields", "quantumyields", "quantum-yields"}))
				plan.observableMode = ObservableMode::TransitionYields;
			else if (OneOf(observableMode, {"cidsp", "cidnp", "productpolarization", "product-polarization"}))
				plan.observableMode = ObservableMode::ProductPolarization;
			else if (observableMode == "auto")
				plan.observableMode = ObservableMode::Auto;
			else
			{
				error = "observables must be auto, states, spins, transitionyields, or cidsp";
				return false;
			}
		}

		if (plan.observableMode == ObservableMode::Auto)
		{
			// DirectSpectra compatibility: an unqualified spinlist selects spin
			// polarization unless transitionyields was explicitly requested.
			if (!transitionYieldsSpecified && !plan.spinList.empty()) plan.transitionYields = false;
		}
		else if (plan.observableMode == ObservableMode::StatePopulations)
		{
			if (plan.calculation != Calculation::TimeEvolution)
			{ error = "observables=states requires calculation=timeevolution"; return false; }
			if (!plan.spinList.empty() || plan.cidsp)
			{ error = "observables=states cannot be combined with spinlist or cidsp/cidnp"; return false; }
			if (transitionYieldsSpecified && plan.transitionYields)
			{ error = "observables=states conflicts with transitionyields=true"; return false; }
			plan.transitionYields = false;
		}
		else if (plan.observableMode == ObservableMode::SpinPolarization)
		{
			if (plan.spinList.empty())
			{ error = "observables=spins requires spinlist"; return false; }
			if ((cidspSpecified && plan.cidsp) || (cidnpSpecified && cidnpAlias))
			{ error = "observables=spins conflicts with cidsp/cidnp=true"; return false; }
			if (transitionYieldsSpecified && plan.transitionYields)
			{ error = "observables=spins conflicts with transitionyields=true"; return false; }
			plan.transitionYields = false;
			plan.cidsp = false;
		}
		else if (plan.observableMode == ObservableMode::TransitionYields)
		{
			if (plan.calculation != Calculation::Yields)
			{ error = "observables=transitionyields requires calculation=yields"; return false; }
			if (!plan.spinList.empty() || plan.cidsp)
			{ error = "observables=transitionyields cannot be combined with spinlist or cidsp/cidnp"; return false; }
			if (transitionYieldsSpecified && !plan.transitionYields)
			{ error = "observables=transitionyields conflicts with transitionyields=false"; return false; }
			plan.transitionYields = true;
		}
		else
		{
			if (plan.spinList.empty())
			{ error = "observables=cidsp requires spinlist"; return false; }
			if (transitionYieldsSpecified && plan.transitionYields)
			{ error = "observables=cidsp conflicts with transitionyields=true"; return false; }
			plan.transitionYields = false;
			plan.cidsp = true;
		}
		plan.hasPulseSequence = properties.GetPulseSequence("pulsesequence", plan.pulseSequence);

		auto readTimelineWindow = [&](const std::string &key, TimelineWindow &window) -> bool
		{
			std::string value;
			if (!properties.Get(key, value)) return true;
			value = Lower(value);
			if (OneOf(value, {"pulse", "pulses", "preparation"})) window = TimelineWindow::Pulse;
			else if (OneOf(value, {"freeevo", "free-evolution", "free_evolution", "free"})) window = TimelineWindow::FreeEvolution;
			else if (OneOf(value, {"full", "all"})) window = TimelineWindow::Full;
			else { error = key + " must be pulse, freeevo, or full"; return false; }
			return true;
		};
		if (!readTimelineWindow("printtimeframe", plan.printWindow) ||
			!readTimelineWindow("integrationtimeframe", plan.integrationWindow)) return false;
		properties.Get("integration", plan.integrateTimeEvolution);
		if (!plan.hasPulseSequence && plan.printWindow == TimelineWindow::Pulse)
		{
			error = "printtimeframe=pulse requires a task-level pulsesequence";
			return false;
		}
		if (plan.calculation != Calculation::TimeEvolution && plan.integrateTimeEvolution)
		{
			error = "integration=true is a time-evolution output option; use calculation=yields for integrated product/polarization observables";
			return false;
		}

		std::string reactionOperators = "haberkorn";
		ReadString(properties, {"reactionoperators", "reactionoperator", "reaction_operator"}, reactionOperators);
		if (reactionOperators == "haberkorn")
		{
			// Haberkorn is the only Hilbert-space reaction model represented by
			// HSGeneral. The actual kP/2 operator is constructed in SpinAPI.
		}
		else if (reactionOperators == "lindblad")
		{
			error = "reactionoperators=lindblad is a superspace reaction-superoperator model; HSGeneral supports Haberkorn sink loss in Hilbert space";
			return false;
		}
		else { error = "reactionoperators must be haberkorn in HSGeneral"; return false; }

		std::string relaxationModel;
		if (ReadString(properties, {"relaxationmodel", "relaxation_model"}, relaxationModel) &&
			OneOf(relaxationModel, {"nakajimazwanzig", "nakajima-zwanzig", "nz"}))
		{
			error = "Nakajima-Zwanzig relaxation is not available in HSGeneral; use the superspace NZ framework";
			return false;
		}

		// Secular/RWA propagation is defined by an explicit H0/H1 split.  The
		// same SpinSpace construction is used for identity, explicit, 2D powder
		// and SO(3) orientations; H0 is secularized while H1 remains full.
		if (plan.approximation == SpinAPI::HamiltonianApproximation::Secular && !plan.hasH0List)
		{
			error = "approximation=secular/RWA requires hamiltonianh0list so H0 is explicit; hamiltonianh1list is optional and remains non-secular";
			return false;
		}
		if (plan.IsPowder() && !plan.hasH0List)
		{
			error = "HSGeneral powder propagation requires hamiltonianh0list so the orientation-specific H0 is explicit";
			return false;
		}

		if (plan.cidsp && plan.spinList.empty())
		{
			error = "cidsp/cidnp output requires spinlist";
			return false;
		}
		if (plan.calculation == Calculation::Yields && !plan.transitionYields && plan.spinList.empty())
		{
			error = "calculation=yields with transitionyields=false requires spinlist for integrated polarization/CIDSP output";
			return false;
		}

		std::string method;
		if (ReadString(properties, {"method", "yieldmethod", "yield_method"}, method))
		{
			if (plan.calculation == Calculation::Yields)
			{
				if (OneOf(method, {"timeevo", "finite", "finitetime", "finite-time"}))
					plan.yieldMode = YieldMode::FiniteTime;
				else if (OneOf(method, {"timeinf", "infinite", "timeinfinity", "time-infinity"}))
					plan.yieldMode = YieldMode::TimeInfinity;
				else { error = "yield method must be timeevo/finite or timeinf"; return false; }
			}
			else if (plan.calculation == Calculation::TimeEvolution && method != "timeevo")
			{
				error = "calculation=timeevolution requires method=timeevo when method is specified";
				return false;
			}
		}
		if (plan.IsDynamic() && plan.UsesTimeInfinityYields())
		{
			error = "dynamic yields require finite-time propagation; method=timeinf is only defined for static Hamiltonians";
			return false;
		}

		return true;
	}

	const char *ToString(Dynamics value) { return value == Dynamics::Static ? "static" : "dynamic"; }
	const char *ToString(Calculation value)
	{
		switch (value) { case Calculation::TimeEvolution: return "timeevolution"; case Calculation::Yields: return "yields"; }
		return "unknown";
	}
	const char *ToString(Sampling value) { return value == Sampling::Direct ? "direct" : "stochastic"; }
	const char *ToString(OrientationMode value)
	{
		switch (value) { case OrientationMode::Identity: return "identity"; case OrientationMode::Powder2D: return "theta/phi"; case OrientationMode::PowderSO3: return "theta/phi/gamma"; case OrientationMode::Explicit: return "explicit"; }
		return "unknown";
	}
	const char *ToString(ObservableMode value)
	{
		switch (value)
		{
		case ObservableMode::Auto: return "auto";
		case ObservableMode::StatePopulations: return "states";
		case ObservableMode::SpinPolarization: return "spins";
		case ObservableMode::TransitionYields: return "transitionyields";
		case ObservableMode::ProductPolarization: return "cidsp";
		}
		return "unknown";
	}
	const char *ToString(PropagationMethod value)
	{
		switch (value) { case PropagationMethod::Exponential: return "exp"; case PropagationMethod::AutoExpm: return "autoexpm"; case PropagationMethod::Krylov: return "krylov"; case PropagationMethod::RK4: return "rk4"; }
		return "unknown";
	}
	const char *ToString(YieldMode value) { return value == YieldMode::FiniteTime ? "finite" : "timeinf"; }
	const char *ToString(TimelineWindow value)
	{
		switch (value) { case TimelineWindow::Pulse: return "pulse"; case TimelineWindow::FreeEvolution: return "freeevo"; case TimelineWindow::Full: return "full"; }
		return "unknown";
	}
}
