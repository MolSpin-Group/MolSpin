/////////////////////////////////////////////////////////////////////////
// HSGeneralConfiguration implementation (RunSection module)
/////////////////////////////////////////////////////////////////////////
#include "HSGeneralConfiguration.h"
#include "ObjectParser.h"
#include "SpinSystem.h"
#include "State.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>

namespace RunSection
{
	namespace
	{
		std::string Lower(std::string _value)
		{
			std::transform(_value.begin(), _value.end(), _value.begin(),
				[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return _value;
		}

		bool ReadString(const MSDParser::ObjectParser &_properties,
						const std::initializer_list<const char *> &_keys,
						std::string &_value)
		{
			for (const char *key : _keys)
			{
				if (_properties.Get(key, _value))
				{
					_value = Lower(_value);
					return true;
				}
			}
			return false;
		}

		bool IsOneOf(const std::string &_value, const std::initializer_list<const char *> &_choices)
		{
			for (const char *choice : _choices)
				if (_value == choice)
					return true;
			return false;
		}
	}

	bool ResolveHSGeneralConfiguration(const MSDParser::ObjectParser &_properties,
									 HSGeneralConfiguration &_configuration,
									 std::string &_error)
	{
		_configuration = HSGeneralConfiguration();
		_error.clear();

		ReadString(_properties, {"dynamics", "timedependence", "time_dependence"}, _configuration.dynamics);
		if (IsOneOf(_configuration.dynamics, {"timeindependent", "time-independent"}))
			_configuration.dynamics = "static";
		else if (IsOneOf(_configuration.dynamics, {"timedependent", "time-dependent"}))
			_configuration.dynamics = "dynamic";
		if (!IsOneOf(_configuration.dynamics, {"static", "dynamic"}))
		{
			_error = "dynamics must be static or dynamic";
			return false;
		}

		ReadString(_properties, {"calculation", "calculationmode", "calculation_mode"}, _configuration.calculation);
		if (IsOneOf(_configuration.calculation, {"timeevo", "time-evolution", "time_evolution"}))
			_configuration.calculation = "timeevolution";
		else if (IsOneOf(_configuration.calculation, {"yield", "quantumyield", "quantumyields"}))
			_configuration.calculation = "yields";
		else if (_configuration.calculation == "resonance")
			_configuration.calculation = "spectra";
		if (!IsOneOf(_configuration.calculation, {"timeevolution", "yields", "spectra"}))
		{
			_error = "calculation must be timeevolution, yields, or spectra";
			return false;
		}

		ReadString(_properties, {"sampling", "tracesampling", "trace_sampling"}, _configuration.sampling);
		if (IsOneOf(_configuration.sampling, {"trace", "montecarlo", "monte-carlo", "mc"}))
			_configuration.sampling = "stochastic";
		if (!IsOneOf(_configuration.sampling, {"direct", "stochastic"}))
		{
			_error = "sampling must be direct or stochastic";
			return false;
		}

		bool approximationSpecified = ReadString(
			_properties, {"approximation", "hamiltonianapproximation", "hamiltonian_approximation"},
			_configuration.approximation);
		bool secularization = false;
		if (_properties.Get("secularization", secularization) || _properties.Get("secular", secularization))
		{
			_configuration.approximation = secularization ? "secular" : "full";
			approximationSpecified = true;
		}
		// Preserve the established high-field/RWA default for spectroscopy.
		// Other HSGeneral calculations use the full Hamiltonian by default.
		if (!approximationSpecified && _configuration.calculation == "spectra")
			_configuration.approximation = "secular";
		if (IsOneOf(_configuration.approximation, {"rwa", "rotatingwave", "rotating-wave", "highfield", "high-field"}))
			_configuration.approximation = "secular";
		else if (IsOneOf(_configuration.approximation, {"nonsecular", "non-secular", "exact"}))
			_configuration.approximation = "full";
		if (!IsOneOf(_configuration.approximation, {"full", "secular"}))
		{
			_error = "approximation must be full or secular";
			return false;
		}

		bool powder = false;
		bool powderKeywordSpecified = false;
		if (_properties.Get("powderaveraging", powder) || _properties.Get("powder", powder))
		{
			_configuration.powderAveraging = powder;
			powderKeywordSpecified = true;
		}
		int powderPoints = 0;
		const bool powderPointsSpecified = _properties.Get("powdersamplingpoints", powderPoints);
		if (powderPointsSpecified && powderPoints > 1)
			_configuration.powderAveraging = true;
		std::string powderOrientation;
		const bool explicitPowderOrientation =
			_properties.Get("powderorientation", powderOrientation) ||
			_properties.Get("powder_orientation", powderOrientation);
		if (explicitPowderOrientation)
		{
			_configuration.powderAveraging = true;
		}
		if (powderKeywordSpecified && powder &&
			(!powderPointsSpecified || powderPoints <= 1) && !explicitPowderOrientation)
		{
			_error = "powderaveraging=true requires powdersamplingpoints greater than one or an explicit powderorientation";
			return false;
		}

		std::string relaxationModel;
		if (ReadString(_properties, {"relaxationmodel", "relaxation_model"}, relaxationModel) &&
			IsOneOf(relaxationModel, {"nakajimazwanzig", "nakajima-zwanzig", "nz"}))
		{
			_error = "Nakajima-Zwanzig relaxation is not yet available in HSGeneral; use the existing superspace NZ task";
			return false;
		}

		if (_configuration.calculation == "spectra")
		{
			if (_configuration.dynamics != "static")
			{
				_error = "spectra currently require dynamics=static";
				return false;
			}
			if (_configuration.sampling == "stochastic" && _configuration.approximation != "secular")
			{
				_error = "full-Hamiltonian stochastic spectra are not yet available; use sampling=direct or approximation=secular";
				return false;
			}
			_configuration.target = (_configuration.approximation == "secular")
				? HSGeneralTarget::StaticDirectSpectra
				: HSGeneralTarget::StaticResonanceSpectra;
			return true;
		}

		if (_configuration.powderAveraging)
		{
			std::string method;
			if (ReadString(_properties, {"method"}, method))
			{
				// A periodic Hamiltonian has no time-independent timeinf operator.
				// Dynamic yields are therefore finite-time integrals of a timeevo
				// propagation, as in the established DynamicHS yield tasks.
				const std::string requiredMethod = _configuration.dynamics == "dynamic"
					? "timeevo"
					: (_configuration.calculation == "yields" ? "timeinf" : "timeevo");
				if (method != requiredMethod)
				{
					_error = "powder calculation=" + _configuration.calculation +
						" requires method=" + requiredMethod + "; remove the conflicting method keyword or use the required value";
					return false;
				}
			}

			// Powder propagation shares one orientation-aware engine. Static yields
			// use the exact timeinf solve; dynamic yields use a finite propagation and
			// trapezoidal integration. The approximation is selected in SpinSpace.
			_configuration.target = HSGeneralTarget::StaticDirectSpectra;
			return true;
		}
		if (_configuration.approximation != "full")
		{
			_error = "secular/RWA selection currently applies only to calculation=spectra";
			return false;
		}

		const bool dynamic = _configuration.dynamics == "dynamic";
		const bool stochastic = _configuration.sampling == "stochastic";
		const bool yields = _configuration.calculation == "yields";

		if (!dynamic && !stochastic && !yields)
			_configuration.target = HSGeneralTarget::StaticDirectTimeEvolution;
		else if (!dynamic && !stochastic && yields)
			_configuration.target = HSGeneralTarget::StaticDirectYields;
		else if (!dynamic && stochastic && !yields)
			_configuration.target = HSGeneralTarget::StaticStochasticTimeEvolution;
		else if (!dynamic && stochastic && yields)
			_configuration.target = HSGeneralTarget::StaticStochasticYields;
		else if (dynamic && !stochastic && !yields)
			_configuration.target = HSGeneralTarget::DynamicDirectTimeEvolution;
		else if (dynamic && !stochastic && yields)
			_configuration.target = HSGeneralTarget::DynamicDirectYields;
		else if (dynamic && stochastic && !yields)
			_configuration.target = HSGeneralTarget::DynamicStochasticTimeEvolution;
		else
			_configuration.target = HSGeneralTarget::DynamicStochasticYields;

		return true;
	}

	bool IsHSGeneralTask(const MSDParser::ObjectParser &_properties)
	{
		std::string type;
		if (!_properties.Get("type", type))
			return false;
		type = Lower(type);
		return type == "hsgeneral" || type == "hs-general";
	}

	bool ValidateHSGeneralTraceSamplingSystems(const std::vector<SpinAPI::system_ptr> &_systems,
										 std::string &_error)
	{
		_error.clear();
		if (_systems.empty())
		{
			_error = "stochastic HSGeneral calculations require at least one spin system";
			return false;
		}

		for (const auto &system : _systems)
		{
			if (system == nullptr)
			{
				_error = "stochastic HSGeneral calculations cannot use a null spin system";
				return false;
			}

			const auto initialStates = system->InitialState();
			if (initialStates.size() != 1)
			{
				_error = "spin system \"" + system->Name() +
					"\" must define exactly one initial State object for stochastic trace sampling";
				return false;
			}
			if (initialStates.front() == nullptr)
			{
				_error = "spin system \"" + system->Name() +
					"\" uses a thermal initial state, which cannot be combined with stochastic trace sampling";
				return false;
			}
			if (!system->Operators().empty())
			{
				_error = "spin system \"" + system->Name() +
					"\" contains explicit relaxation operators, which are not yet supported with stochastic trace sampling";
				return false;
			}
			if (system->InitialStateCoherences() != SpinAPI::InitialStateCoherenceMode::Keep)
			{
				_error = "spin system \"" + system->Name() +
					"\" requests initial-state dephasing, which cannot be represented by pure-state trace samples";
				return false;
			}
			if (system->InitialStateFrame() == SpinAPI::StateFrame::Eigen)
			{
				_error = "spin system \"" + system->Name() +
					"\" uses frame=eigen, which cannot be represented by State-object trace samples";
				return false;
			}
		}

		return true;
	}

	void SeedHSGeneralRandomGenerator(const MSDParser::ObjectParser &_properties,
								  std::mt19937 &_generator,
								  std::ostream &_log)
	{
		bool autoseed = true;
		_properties.Get("autoseed", autoseed);
		if (autoseed)
		{
			_log << "Autoseed is on." << std::endl;
			return;
		}

		double seedNumber = 1.0;
		if (!_properties.Get("seed", seedNumber) || seedNumber == 0.0)
		{
			seedNumber = 1.0;
			_log << "No non-zero seed was specified. Using the deterministic default seed 1." << std::endl;
		}
		else
		{
			_log << "Seed number is " << seedNumber << "." << std::endl;
		}

		_generator.seed(static_cast<std::mt19937::result_type>(seedNumber));
	}

	bool BuildHSGeneralTraceSamples(const MSDParser::ObjectParser &_properties,
									const SpinAPI::system_ptr &_system,
									SpinAPI::SpinSpace &_space,
									arma::uword _sampleCount,
									std::mt19937 &_generator,
									SpinAPI::HilbertTraceSampleSet &_samples,
									std::ostream &_log,
									std::string &_error)
	{
		_error.clear();
		if (_system == nullptr)
		{
			_error = "cannot trace sample a null spin system";
			return false;
		}

		if (!ValidateHSGeneralTraceSamplingSystems({_system}, _error))
			return false;
		const auto initialStates = _system->InitialState();

		std::string samplingMethod;
		if (!ReadString(_properties, {"samplingmethod", "trace_sampling_method", "tracesamplingmethod"}, samplingMethod))
			samplingMethod = "suz";

		SpinAPI::TraceSamplingMethod method = SpinAPI::TraceSamplingMethod::SUZ;
		if (samplingMethod == "coherent" || samplingMethod == "spincoherent" || samplingMethod == "spin-coherent")
			method = SpinAPI::TraceSamplingMethod::SpinCoherent;
		else if (samplingMethod != "suz" && samplingMethod != "haar")
		{
			_error = "samplingmethod must be suz or coherent";
			return false;
		}

		if (!_space.BuildTraceSamples(initialStates.front(), _sampleCount, method, _generator, _samples, &_error))
			return false;

		_log << "HSGeneral trace sampling keeps state \"" << initialStates.front()->Name()
			 << "\" fixed and samples only omitted spins (subspace dimension "
			 << _samples.sampledSubspaceDimension << ")." << std::endl;
		_log << "Trace sampling method = "
			 << (method == SpinAPI::TraceSamplingMethod::SUZ ? "SU(Z)" : "spin coherent")
			 << "." << std::endl;
		return true;
	}

	bool BuildHSGeneralInitialDensityMatrix(const SpinAPI::system_ptr &_system,
										 SpinAPI::SpinSpace &_space,
										 arma::cx_mat &_density,
										 std::string &_error)
	{
		_error.clear();
		_density.reset();
		if (_system == nullptr)
		{
			_error = "cannot construct an initial state for a null spin system";
			return false;
		}

		const auto initialStates = _system->InitialState();
		if (initialStates.empty())
		{
			_error = "spin system \"" + _system->Name() + "\" does not define an initial state";
			return false;
		}

		std::vector<double> weights = _system->Weights();
		if (weights.empty())
			weights.assign(initialStates.size(), 1.0 / static_cast<double>(initialStates.size()));
		else if (weights.size() == 1 && initialStates.size() == 1)
			weights[0] = 1.0;
		else if (weights.size() != initialStates.size())
		{
			_error = "the number of initial-state weights does not match the number of initial states";
			return false;
		}

		for (double weight : weights)
		{
			if (!std::isfinite(weight) || weight < 0.0)
			{
				_error = "initial-state weights must be finite and non-negative";
				return false;
			}
		}
		const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
		if (!(weightSum > 0.0))
		{
			_error = "initial-state weights must have a positive sum";
			return false;
		}
		for (double &weight : weights)
			weight /= weightSum;

		_density.zeros(_space.HilbertSpaceDimensions(), _space.HilbertSpaceDimensions());
		for (size_t index = 0; index < initialStates.size(); ++index)
		{
			arma::cx_mat component;
			if (initialStates[index] == nullptr)
			{
				if (!_space.GetThermalState(
						_space, _system->Temperature(), _system->ThermalHamiltonianList(), component))
				{
					_error = "failed to construct the thermal initial state for spin system \"" +
						_system->Name() + "\"";
					return false;
				}
			}
			else if (!_space.GetState(initialStates[index], component))
			{
				_error = "failed to construct initial State \"" + initialStates[index]->Name() +
					"\" for spin system \"" + _system->Name() + "\"";
				return false;
			}

			const arma::cx_double componentTrace = arma::trace(component);
			if (!std::isfinite(std::real(componentTrace)) || std::abs(componentTrace) == 0.0)
			{
				_error = "an initial-state component has an invalid trace";
				return false;
			}
			_density += weights[index] * component / componentTrace;
		}
		return true;
	}

	bool BuildHSGeneralInitialStateFactors(const SpinAPI::system_ptr &_system,
										SpinAPI::SpinSpace &_space,
										arma::cx_mat &_factors,
										std::ostream &_log,
										std::string &_error)
	{
		_error.clear();
		_factors.reset();
		if (_system == nullptr)
		{
			_error = "cannot construct an initial state for a null spin system";
			return false;
		}
		if (_system->InitialStateCoherences() != SpinAPI::InitialStateCoherenceMode::Keep)
		{
			_error = "initial-state dephasing is not yet available for HSGeneral time evolution or yields";
			return false;
		}
		if (_system->InitialStateFrame() == SpinAPI::StateFrame::Eigen)
		{
			_error = "frame=eigen is not yet available for HSGeneral time evolution or yields";
			return false;
		}

		arma::cx_mat density;
		if (!BuildHSGeneralInitialDensityMatrix(_system, _space, density, _error))
			return false;

		if (!_space.FactorizeDensityMatrix(density, _factors, &_error))
			return false;

		_log << "HSGeneral constructed the normalized initial density from "
			 << _system->InitialState().size() << " state component(s) as " << _factors.n_cols
			 << " Hilbert-space factor(s)." << std::endl;
		return true;
	}

	const char *HSGeneralTargetName(HSGeneralTarget _target)
	{
		switch (_target)
		{
		case HSGeneralTarget::StaticDirectTimeEvolution: return "StaticHS-Direct-TimeEvo";
		case HSGeneralTarget::StaticDirectYields: return "StaticHS-Direct-Yields";
		case HSGeneralTarget::StaticStochasticTimeEvolution: return "StaticHS-Stoch-TimeEvo";
		case HSGeneralTarget::StaticStochasticYields: return "StaticHS-Stoch-Yields";
		case HSGeneralTarget::DynamicDirectTimeEvolution: return "DynamicHS-Direct-TimeEvo";
		case HSGeneralTarget::DynamicDirectYields: return "DynamicHS-Direct-Yields";
		case HSGeneralTarget::DynamicStochasticTimeEvolution: return "DynamicHS-Stoch-TimeEvo";
		case HSGeneralTarget::DynamicStochasticYields: return "DynamicHS-Stoch-Yields";
		case HSGeneralTarget::StaticDirectSpectra: return "StaticHS-Direct-Spectra";
		case HSGeneralTarget::StaticResonanceSpectra: return "StaticHS-Resonance-Spectra";
		}
		return "unknown";
	}
}
