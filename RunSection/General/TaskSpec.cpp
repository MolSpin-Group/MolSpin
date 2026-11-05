#include "TaskSpec.h"
#include "HS/HSExecutionPlan.h"
#include "SS/SSExecutionPlan.h"
#include "MultiSS/MultiSSExecutionPlan.h"

#include <algorithm>
#include <map>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace RunSection::General
{
	namespace
	{
		struct Constraint
		{
			std::string kind;
			std::vector<std::string> keys;
			std::string message;
			std::string whenJson;
		};

		struct Task
		{
			std::vector<std::string> aliases;
			std::map<std::string, TaskSpecProperty> properties;
			std::vector<Constraint> constraints;
		};

		TaskSpecProperty P(std::string type, std::string defaultJson, std::string description,
			std::vector<std::string> aliases = {}, std::vector<std::string> choices = {})
		{
			return {std::move(aliases), std::move(type), std::move(defaultJson),
				std::move(choices), std::move(description)};
		}

		std::string Json(bool value)
		{
			return value ? "true" : "false";
		}

		std::string Json(int value)
		{
			return std::to_string(value);
		}

		std::string Json(double value)
		{
			std::ostringstream output;
			output << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
			return output.str();
		}

		std::string Json(const std::string &value)
		{
			std::ostringstream output;
			output << '"';
			for (char c : value)
			{
				if (c == '"' || c == '\\') output << '\\';
				output << c;
			}
			output << '"';
			return output.str();
		}

		const char *GridName(SpinAPI::PowderGridType type)
		{
			switch (type)
			{
			case SpinAPI::PowderGridType::Uniform: return "uniform";
			case SpinAPI::PowderGridType::Sophe: return "sophe";
			case SpinAPI::PowderGridType::Octant: return "octant";
			}
			return "uniform";
		}

		const char *DomainName(SpinAPI::PowderGridDomain domain)
		{
			return domain == SpinAPI::PowderGridDomain::FullSphere ? "full" : "upper";
		}

		const char *ApproximationName(SpinAPI::HamiltonianApproximation approximation)
		{
			return approximation == SpinAPI::HamiltonianApproximation::Secular
				? "secular" : "full";
		}

		const char *HamiltonianModeName(SS::SSHamiltonianMode mode)
		{
			switch (mode)
			{
			case SS::SSHamiltonianMode::FixedFull: return "full";
			case SS::SSHamiltonianMode::RotatedFull: return "rotated_zyz";
			case SS::SSHamiltonianMode::RotatedSecular: return "rotated_sa";
			}
			return "full";
		}

		const char *ReactionOperatorName(SpinAPI::ReactionOperatorType type)
		{
			return type == SpinAPI::ReactionOperatorType::Lindblad ? "lindblad" : "haberkorn";
		}

		void SetDefault(Task &task, const std::string &propertyName, std::string defaultJson)
		{
			task.properties.at(propertyName).defaultJson = std::move(defaultJson);
		}

		void SetHSDefaults(Task &task)
		{
			const HS::HSExecutionPlan plan;
			SetDefault(task, "dynamics", Json(std::string(HS::ToString(plan.dynamics))));
			SetDefault(task, "calculation", Json(std::string(HS::ToString(plan.calculation))));
			SetDefault(task, "sampling", Json(std::string(HS::ToString(plan.sampling))));
			SetDefault(task, "approximation", Json(std::string(ApproximationName(plan.approximation))));
			SetDefault(task, "autoseed", Json(plan.autoSeed));
			SetDefault(task, "cidsp", Json(plan.cidsp));
			SetDefault(task, "stochasticstandarddeviation", Json(plan.stochasticStandardDeviation));
			SetDefault(task, "totaltime", Json(plan.totalTime));
			SetDefault(task, "timestep", Json(plan.timeStep));
			SetDefault(task, "precision", Json(plan.precision));
			SetDefault(task, "krylovsize", Json(plan.krylovSize));
			SetDefault(task, "montecarlosamples", Json(static_cast<int>(plan.monteCarloSamples)));
			SetDefault(task, "samplingmethod", Json(plan.samplingMethod));
			SetDefault(task, "seed", Json(plan.seed));
			SetDefault(task, "transitionyields", Json(plan.transitionYields));
			SetDefault(task, "yieldcorrections", Json(plan.yieldCorrections));
			SetDefault(task, "observables", Json(std::string(HS::ToString(plan.observableMode))));
			SetDefault(task, "powdergamma", Json(plan.powderGammaOffset));
			SetDefault(task, "powdergammapoints", Json(plan.powderGammaPoints));
			SetDefault(task, "powdergrid", Json(std::string(GridName(plan.powderGridType))));
			SetDefault(task, "powdergridsize", Json(plan.powderGridSize));
			SetDefault(task, "powdersamplingpoints", Json(plan.powderPoints));
			SetDefault(task, "powdersymmetry", Json(plan.powderSymmetry));
			SetDefault(task, "powderdomain", Json(std::string(DomainName(plan.powderDomain))));
			SetDefault(task, "powderfullsphere", Json(false));
			SetDefault(task, "propagationmethod", Json(std::string(HS::ToString(plan.propagation))));
			SetDefault(task, "secularization",
				Json(plan.approximation == SpinAPI::HamiltonianApproximation::Secular));
			SetDefault(task, "printtimeframe", Json(std::string(HS::ToString(plan.printWindow))));
			SetDefault(task, "integrationtimeframe", Json(std::string(HS::ToString(plan.integrationWindow))));
			SetDefault(task, "integration", Json(plan.integrateTimeEvolution));
		}

		void SetSSDefaults(Task &task)
		{
			const SS::SSExecutionPlan plan;
			SetDefault(task, "calculation", Json(std::string(SS::ToString(plan.calculation))));
			SetDefault(task, "propagationmethod", Json(std::string(SS::ToString(plan.propagation))));
			SetDefault(task, "observables", Json(std::string(SS::ToString(plan.observables))));
			SetDefault(task, "hamiltonianmode", Json(std::string(HamiltonianModeName(plan.hamiltonianMode))));
			SetDefault(task, "relaxationmodel", Json(std::string(SS::ToString(plan.relaxationModel))));
			SetDefault(task, "reactionoperators", Json(std::string(ReactionOperatorName(plan.reactionOperator))));
			SetDefault(task, "totaltime", Json(plan.totalTime));
			SetDefault(task, "timestep", Json(plan.timeStep));
			SetDefault(task, "solverresidualtolerance", Json(plan.solverResidualTolerance));
			SetDefault(task, "diagnostics", Json(plan.diagnostics));
			SetDefault(task, "powdergammapoints", Json(plan.powderGammaPoints));
			SetDefault(task, "powdergamma", Json(plan.powderGammaOffset));
			SetDefault(task, "powdergrid", Json(std::string(GridName(plan.powderGridType))));
			SetDefault(task, "powdergridsize", Json(plan.powderGridSize));
			SetDefault(task, "powdersamplingpoints", Json(plan.powderPoints));
			SetDefault(task, "powdersymmetry", Json(plan.powderSymmetry));
			SetDefault(task, "powderdomain", Json(std::string(DomainName(plan.powderDomain))));
		}

		void SetMultiSSDefaults(Task &task)
		{
			const MultiSS::MultiSSExecutionPlan plan;
			SetDefault(task, "calculation", Json(std::string(MultiSS::ToString(plan.calculation))));
			SetDefault(task, "propagationmethod", Json(std::string(MultiSS::ToString(plan.propagation))));
			SetDefault(task, "observables", Json(std::string(MultiSS::ToString(plan.observables))));
			SetDefault(task, "hamiltonianmode", Json(std::string(HamiltonianModeName(plan.hamiltonianMode))));
			SetDefault(task, "relaxationmodel", Json(std::string(SS::ToString(plan.relaxationModel))));
			SetDefault(task, "totaltime", Json(plan.totalTime));
			SetDefault(task, "timestep", Json(plan.timeStep));
			SetDefault(task, "solverresidualtolerance", Json(plan.solverResidualTolerance));
			SetDefault(task, "krylovdimension", Json(plan.krylovDimension));
			SetDefault(task, "krylovtolerance", Json(plan.krylovTolerance));
			SetDefault(task, "diagnostics", Json(plan.diagnostics));
			SetDefault(task, "transitionfluxes", Json(plan.transitionFluxes));
			SetDefault(task, "powdergammapoints", Json(plan.powderGammaPoints));
			SetDefault(task, "powdergamma", Json(plan.powderGammaOffset));
			SetDefault(task, "powdergrid", Json(std::string(GridName(plan.powderGridType))));
			SetDefault(task, "powdergridsize", Json(plan.powderGridSize));
			SetDefault(task, "powdersamplingpoints", Json(plan.powderPoints));
			SetDefault(task, "powdersymmetry", Json(plan.powderSymmetry));
			SetDefault(task, "powderdomain", Json(std::string(DomainName(plan.powderDomain))));
		}

		Constraint C(std::string kind, std::vector<std::string> keys,
			std::string message, std::string whenJson)
		{
			return {std::move(kind), std::move(keys), std::move(message), std::move(whenJson)};
		}

		Task MakeHS()
		{
			Task task;
			task.aliases = {"hsgeneral", "HSGeneral", "hs-general", "HS-General"};
			task.properties = {
				{"autoseed", P("bool", "true", "Generate a random seed when no seed is supplied.")},
				{"approximation", P("string", "\"full\"", "Hamiltonian approximation mode.",
					{"hamiltonianapproximation", "hamiltonian_approximation"},
					{"full", "secular", "highfield", "high-field", "nonsecular", "non-secular", "exact", "rwa", "rotatingwave", "rotating-wave"})},
				{"calculation", P("string", "\"timeevolution\"", "Calculation mode.",
					{"calculationmode", "calculation_mode"},
					{"timeevolution", "timeevo", "time-evolution", "time_evolution", "yields", "yield", "quantumyield", "quantumyields"})},
				{"cidnp", P("bool", "false", "Enable the legacy CIDNP alias for CIDSP.")},
				{"cidsp", P("bool", "false", "Enable product spin polarization output.")},
				{"dynamics", P("string", "\"static\"", "Hamiltonian time-dependence mode.",
					{"timedependence", "time_dependence"},
					{"static", "dynamic", "timeindependent", "time-independent", "timedependent", "time-dependent"})},
				{"hamiltonianh0list", P("list", "null", "Comma-separated Hamiltonian interaction names used for H0.")},
				{"hamiltonianh1list", P("list", "null", "Comma-separated Hamiltonian interaction names used for H1.")},
				{"initialstatehamiltonian", P("list", "null", "Hamiltonian names used for initial-state eigenbasis dephasing.")},
				{"integration", P("bool", "false", "Integrate time-evolution observables.")},
				{"integrationtimeframe", P("string", "\"freeevolution\"", "Timeline segment used for integration.",
					{}, {"pulse", "pulses", "preparation", "freeevo", "free-evolution", "free_evolution", "free", "full", "all"})},
				{"krylovsize", P("int", "16", "Krylov subspace dimension.")},
				{"krylovtol", P("float", "null", "Deprecated compatibility tolerance; not used by the current Krylov propagator.")},
				{"method", P("string", "null", "Yield integration method.",
					{"yieldmethod", "yield_method"},
					{"timeevo", "finite", "finitetime", "finite-time", "timeinf", "infinite", "timeinfinity", "time-infinity"})},
				{"montecarlosamples", P("int", "1", "Number of stochastic trace samples.")},
				{"observables", P("string", "\"auto\"", "Observable output mode.",
					{"observablemode", "observable_mode"},
					{"auto", "states", "state", "populations", "statepopulations", "state-populations", "spins", "spin", "polarization", "polarizations", "spinpolarization", "spin-polarization", "transitionyields", "transition-yields", "yields", "quantumyields", "quantum-yields", "cidsp", "cidnp", "productpolarization", "product-polarization"})},
				{"powderaveraging", P("bool", "null", "Enable or disable powder averaging.", {"powder"})},
				{"powderdomain", P("string", "null", "Powder integration domain.",
					{}, {"upper", "hemisphere", "upperhemisphere", "full", "fullsphere", "sphere"})},
				{"powderfullsphere", P("bool", "null", "Legacy full-sphere powder-domain setting.",
					{"powder_full_sphere"})},
				{"powdergamma", P("float", "0.0", "Offset of the Euler gamma-angle grid.", {"powder_gamma"})},
				{"powdergammapoints", P("int", "1", "Number of Euler gamma-angle samples.")},
				{"powdergrid", P("string", "null", "Powder grid family.",
					{"powder_grid", "powdergridtype"},
					{"uniform", "golden", "fibonacci", "sophe", "octant"})},
				{"powdergridsize", P("int", "4", "SOPHE powder-grid size.")},
				{"powderorientation", P("string", "null", "Explicit orientation as alpha beta [gamma [weight]].",
					{"powder_orientation"})},
				{"powdersamplingpoints", P("int", "null", "Number of powder sampling points.")},
				{"powdersymmetry", P("string", "\"c1\"", "SOPHE powder-grid symmetry.",
					{"powdergridsymmetry"})},
				{"powdertheta", P("float", "null", "Explicit powder orientation theta angle.", {"powder_theta"})},
				{"powderphi", P("float", "null", "Explicit powder orientation phi angle.", {"powder_phi"})},
				{"powderweight", P("float", "null", "Weight for an explicit powder orientation.", {"powder_weight"})},
				{"precision", P("string", "\"single\"", "Numerical precision setting.")},
				{"printtimeframe", P("string", "\"freeevolution\"", "Timeline segment written to time-evolution output.",
					{}, {"pulse", "pulses", "preparation", "freeevo", "free-evolution", "free_evolution", "free", "full", "all"})},
				{"propagationmethod", P("string", "\"normal\"", "Propagation algorithm.",
					{"propagator"},
					{"normal", "exp", "exponential", "matrixexponential", "matrix-exponential", "autoexpm", "krylov", "rk4", "explicit"})},
				{"pulsesequence", P("list", "null", "Task-level pulse sequence.")},
				{"reactionoperators", P("string", "\"haberkorn\"", "Reaction-operator model.",
					{"reactionoperator", "reaction_operator"}, {"haberkorn", "lindblad"})},
				{"relaxationmodel", P("string", "null", "Relaxation model.",
					{"relaxation_model"})},
				{"sampling", P("string", "\"direct\"", "Trace-sampling method.",
					{"tracesampling", "trace_sampling"},
					{"direct", "stochastic", "trace", "montecarlo", "monte-carlo", "mc"})},
				{"samplingmethod", P("string", "\"suz\"", "Stochastic sampling estimator.", {},
					{"suz", "coherent"})},
				{"secularization", P("bool", "null", "Select secular or full approximation.", {"secular"})},
				{"seed", P("float", "1.0", "Random seed.")},
				{"spinlist", P("list", "null", "Comma-separated spin names for spin observables.")},
				{"stochasticstandarddeviation", P("bool", "false", "Report standard deviations for stochastic samples.")},
				{"timestep", P("float", "1.0", "Time step in nanoseconds.")},
				{"totaltime", P("float", "10000.0", "Total propagation time in nanoseconds.")},
				{"transitionyields", P("bool", "true", "Include transition yields in output.")},
				{"yieldcorrections", P("bool", "false", "Apply yield corrections.")}
			};
			task.constraints = {
				C("requires", {"powdergrid", "powdersamplingpoints"}, "powdergrid=uniform requires powdersamplingpoints",
					R"({"key":"powdergrid","equals":"uniform"})"),
				C("scoped", {"powdergridsize", "powdersymmetry"},
					"powdergridsize/powdersymmetry are only valid with powdergrid=sophe",
					R"({"key":"powdergrid","not_equals":"sophe"})"),
				C("requires", {"stochasticstandarddeviation", "sampling"}, "stochasticstandarddeviation=true requires sampling=stochastic",
					R"({"key":"stochasticstandarddeviation","equals":true})"),
				C("requires", {"observables", "calculation"}, "observables=transitionyields requires calculation=yields",
					R"({"key":"observables","equals":"transitionyields"})"),
				C("conflicts", {"powderdomain", "powderfullsphere"},
					"powderdomain conflicts with powderfullsphere",
					R"({"any":[{"all":[{"key":"powderdomain","in":["upper","hemisphere","upperhemisphere"]},{"key":"powderfullsphere","equals":true}]},{"all":[{"key":"powderdomain","in":["full","fullsphere","sphere"]},{"key":"powderfullsphere","equals":false}]}]})")
			};
			SetHSDefaults(task);
			return task;
		}

		Task MakeSS()
		{
			Task task;
			task.aliases = {"ssgeneral", "SSGeneral", "ss-general", "SS-General"};
			task.properties = {
				{"calculation", P("string", "\"timeintegrated\"", "Calculation mode.",
					{"calculationmode", "calculation_mode", "method"},
					{"timeevolution", "timeevo", "time-evolution", "timeintegrated", "time-integrated", "timeinf", "integrated", "static", "steadystate", "steady-state", "stationary"})},
				{"diagnostics", P("bool", "true", "Write solver diagnostics.")},
				{"hamiltonianmode", P("string", "\"full\"", "Hamiltonian approximation mode.",
					{"hamiltonian_mode", "approximation"},
					{"full", "fixed", "exact", "nonsecular", "rotated_zyz", "rotatedzyz", "secular", "highfield", "high-field", "rotated_sa", "rotatedsa"})},
				{"observables", P("string", "\"states\"", "Observable output mode.",
					{"observablemode"}, {"states", "statepopulations", "populations", "transitions", "transitionyields", "products", "yields", "both", "all"})},
				{"powderdomain", P("string", "null", "Powder integration domain.",
					{}, {"full", "sphere", "fullsphere", "upper", "hemisphere", "upperhemisphere"})},
				{"powdergamma", P("float", "0.0", "Offset of the Euler gamma-angle grid.")},
				{"powdergammapoints", P("int", "1", "Number of Euler gamma-angle samples.")},
				{"powdergrid", P("string", "null", "Powder grid family.",
					{"powder_grid"}, {"uniform", "golden", "fibonacci", "octant", "sophe"})},
				{"powdergridsize", P("int", "4", "SOPHE powder-grid size.")},
				{"powderorientation", P("string", "null", "Explicit orientation as alpha beta [gamma [weight]].",
					{"orientation"})},
				{"powdersamplingpoints", P("int", "1", "Number of powder sampling points.")},
				{"powdersymmetry", P("string", "\"c1\"", "SOPHE powder-grid symmetry.")},
				{"propagationmethod", P("string", "\"exponential\"", "Propagation algorithm.",
					{"propagator"}, {"exponential", "exp", "normal", "auto", "rk4", "explicit"})},
				{"reactionoperators", P("string", "\"haberkorn\"", "Reaction-operator model.",
					{"reactionoperator", "reaction_operator"}, {"haberkorn", "hab", "lindblad", "gksl"})},
				{"relaxationmodel", P("string", "\"operators\"", "Relaxation model.",
					{"relaxation_model"}, {"operators", "operator", "local", "nakajima_zwanzig", "nakajima-zwanzig", "nakajimazwanzig", "nz", "redfield"})},
				{"solverresidualtolerance", P("float", "1.0e-8", "Residual tolerance for stationary linear solvers.")},
				{"timestep", P("float", "1.0", "Time step in nanoseconds.")},
				{"totaltime", P("float", "1000.0", "Total propagation time in nanoseconds.")},
			};
			task.constraints = {
				C("requires", {"powdergrid", "powdersamplingpoints"}, "powdergrid=uniform requires powdersamplingpoints",
					R"({"key":"powdergrid","equals":"uniform"})"),
				C("scoped", {"powdergridsize", "powdersymmetry"},
					"powdergridsize/powdersymmetry are only valid with powdergrid=sophe",
					R"({"key":"powdergrid","not_equals":"sophe"})")
			};
			SetSSDefaults(task);
			return task;
		}

		Task MakeMultiSS()
		{
			Task task;
			task.aliases = {"multissgeneral", "MultiSSGeneral", "multiss-general", "MultiSS-General"};
			task.properties = {
				{"calculation", P("string", "\"timeevolution\"", "Calculation mode.",
					{"calculationmode", "calculation_mode"},
					{"timeevolution", "timeevo", "time-evolution", "timeintegrated", "time-integrated", "timeinf", "integrated", "steadystate", "steady-state", "stationary"})},
				{"diagnostics", P("bool", "true", "Write solver diagnostics.")},
				{"hamiltonianmode", P("string", "\"full\"", "Local Hamiltonian approximation mode.",
					{"hamiltonian_mode"},
					{"full", "fixed", "legacy", "rotated_zyz", "rotatedzyz", "exact", "nonsecular", "rotated_sa", "rotatedsa", "secular", "highfield"})},
				{"krylovdimension", P("int", "30", "Krylov subspace dimension.")},
				{"krylovtolerance", P("float", "1.0e-10", "Krylov propagation tolerance.")},
				{"observables", P("string", "\"both\"", "Observable output mode.",
					{"observablemode"}, {"populations", "population", "manifolds", "states", "statepopulations", "both", "all"})},
				{"powderdomain", P("string", "null", "Powder integration domain.",
					{}, {"full", "sphere", "fullsphere", "upper", "hemisphere", "upperhemisphere"})},
				{"powdergamma", P("float", "0.0", "Offset of the Euler gamma-angle grid.")},
				{"powdergammapoints", P("int", "1", "Number of Euler gamma-angle samples.")},
				{"powdergrid", P("string", "null", "Powder grid family.",
					{"powder_grid"}, {"uniform", "golden", "fibonacci", "octant", "sophe"})},
				{"powdergridsize", P("int", "4", "SOPHE powder-grid size.")},
				{"powderorientation", P("string", "null", "Explicit orientation as alpha beta [gamma [weight]].",
					{"orientation"})},
				{"powdersamplingpoints", P("int", "1", "Number of powder sampling points.")},
				{"powdersymmetry", P("string", "\"c1\"", "SOPHE powder-grid symmetry.")},
				{"propagationmethod", P("string", "\"rk4\"", "Propagation algorithm.",
					{"propagator"}, {"rk4", "explicit", "exponential", "exp", "normal", "krylov", "arnoldi", "expmv"})},
				{"relaxationmodel", P("string", "\"operators\"", "Relaxation model.",
					{"relaxation_model"}, {"operators", "operator", "local", "nakajima_zwanzig", "nakajima-zwanzig", "nakajimazwanzig", "nz", "redfield"})},
				{"solverresidualtolerance", P("float", "1.0e-8", "Residual tolerance for stationary linear solvers.")},
				{"timestep", P("float", "1.0", "Time step in nanoseconds.")},
				{"totaltime", P("float", "1000.0", "Total propagation time in nanoseconds.")},
				{"transitionfluxes", P("bool", "false", "Include transition fluxes in output.")},
			};
			task.constraints = {
				C("requires", {"powdergrid", "powdersamplingpoints"}, "powdergrid=uniform requires powdersamplingpoints",
					R"({"key":"powdergrid","equals":"uniform"})"),
				C("scoped", {"powdergridsize", "powdersymmetry"},
					"powdergridsize/powdersymmetry are only valid with powdergrid=sophe",
					R"({"key":"powdergrid","not_equals":"sophe"})")
			};
			SetMultiSSDefaults(task);
			return task;
		}

		const std::map<std::string, Task> &Tasks()
		{
			static const std::map<std::string, Task> tasks = {
				{"HSGeneral", MakeHS()},
				{"MultiSSGeneral", MakeMultiSS()},
				{"SSGeneral", MakeSS()}
			};
			return tasks;
		}

		const Task *FindTask(const std::string &name)
		{
			const auto &tasks = Tasks();
			const auto task = tasks.find(name);
			return task == tasks.end() ? nullptr : &task->second;
		}

		std::string Escape(const std::string &value)
		{
			std::string escaped;
			for (unsigned char c : value)
			{
				switch (c)
				{
				case '"': escaped += "\\\""; break;
				case '\\': escaped += "\\\\"; break;
				case '\b': escaped += "\\b"; break;
				case '\f': escaped += "\\f"; break;
				case '\n': escaped += "\\n"; break;
				case '\r': escaped += "\\r"; break;
				case '\t': escaped += "\\t"; break;
				default:
					if (c < 0x20)
					{
						const char hex[] = "0123456789abcdef";
						escaped += "\\u00";
						escaped += hex[(c >> 4) & 0x0f];
						escaped += hex[c & 0x0f];
					}
					else
						escaped += static_cast<char>(c);
				}
			}
			return escaped;
		}

		void WriteString(std::ostream &output, const std::string &value)
		{
			output << '"' << Escape(value) << '"';
		}

		void WriteStrings(std::ostream &output, const std::vector<std::string> &values)
		{
			output << '[';
			for (size_t i = 0; i < values.size(); ++i)
			{
				if (i != 0) output << ", ";
				WriteString(output, values[i]);
			}
			output << ']';
		}

		void WriteTask(std::ostream &output, const std::string &name, const Task &task)
		{
			output << "    ";
			WriteString(output, name);
			output << ": {\n      \"type\": ";
			WriteString(output, name);
			output << ",\n      \"aliases\": ";
			WriteStrings(output, task.aliases);
			output << ",\n      \"properties\": {\n";
			size_t propertyIndex = 0;
			for (const auto &entry : task.properties)
			{
				const auto &property = entry.second;
				output << "        ";
				WriteString(output, entry.first);
				output << ": {\"name\": ";
				WriteString(output, entry.first);
				output << ", \"aliases\": ";
				WriteStrings(output, property.aliases);
				output << ", \"value_type\": ";
				WriteString(output, property.type);
				output << ", \"default\": " << property.defaultJson << ", \"choices\": ";
				WriteStrings(output, property.choices);
				output << ", \"required\": false, \"description\": ";
				WriteString(output, property.description);
				output << '}';
				if (++propertyIndex != task.properties.size()) output << ',';
				output << '\n';
			}
			output << "      },\n      \"constraints\": [";
			for (size_t i = 0; i < task.constraints.size(); ++i)
			{
				const auto &constraint = task.constraints[i];
				output << (i == 0 ? "\n        {" : ",\n        {")
					<< "\"kind\": ";
				WriteString(output, constraint.kind);
				output << ", \"keys\": ";
				WriteStrings(output, constraint.keys);
				output << ", \"message\": ";
				WriteString(output, constraint.message);
				output << ", \"when\": " << constraint.whenJson << '}';
			}
			if (!task.constraints.empty()) output << '\n';
			output << "      ]\n    }";
		}
	}

	const char *TaskSpecVersion()
	{
		return "2.3";
	}

	const std::map<std::string, TaskSpecProperty> &TaskSpecProperties(const std::string &taskName)
	{
		static const std::map<std::string, TaskSpecProperty> empty;
		const Task *task = FindTask(taskName);
		return task == nullptr ? empty : task->properties;
	}

	std::vector<std::string> TaskSpecInputKeys(const std::string &taskName,
		const std::string &propertyName)
	{
		std::vector<std::string> keys;
		const auto &properties = TaskSpecProperties(taskName);
		const auto property = properties.find(propertyName);
		if (property == properties.end())
			throw std::logic_error("Undeclared " + taskName + " task property: " + propertyName);
		keys.reserve(property->second.aliases.size() + 1);
		keys.push_back(propertyName);
		keys.insert(keys.end(), property->second.aliases.begin(), property->second.aliases.end());
		return keys;
	}

	bool IsTaskSpecAlias(const std::string &taskName, const std::string &alias)
	{
		const Task *task = FindTask(taskName);
		return task != nullptr &&
			std::find(task->aliases.begin(), task->aliases.end(), alias) != task->aliases.end();
	}

	bool WriteTaskSpec(std::ostream &output, const std::string &taskName)
	{
		const auto &tasks = Tasks();
		if (!taskName.empty() && tasks.find(taskName) == tasks.end())
			return false;

		output << "{\n  \"schema_version\": 1,\n  \"molspin_version\": ";
		WriteString(output, TaskSpecVersion());
		output << ",\n  \"tasks\": {\n";

		bool first = true;
		for (const auto &entry : tasks)
		{
			if (!taskName.empty() && entry.first != taskName)
				continue;
			if (!first) output << ",\n";
			WriteTask(output, entry.first, entry.second);
			first = false;
		}

		output << "\n  }\n}\n";
		return static_cast<bool>(output);
	}
}
