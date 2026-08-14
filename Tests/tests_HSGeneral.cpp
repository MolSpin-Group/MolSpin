//////////////////////////////////////////////////////////////////////////////
// HSGeneral compatibility-layer tests.
//////////////////////////////////////////////////////////////////////////////
#include "HSGeneralConfiguration.h"
#include "TaskDynamicHSDirectTimeEvo.h"
#include "TaskDynamicHSDirectYields.h"
#include "TaskDynamicHSStochTimeEvo.h"
#include "TaskDynamicHSStochYields.h"
#include "TaskStaticHSDirectSpectra.h"
#include "TaskStaticHSDirectTimeEvo.h"
#include "TaskStaticHSDirectYields.h"
#include "TaskStaticHSResonanceSpectra.h"
#include "TaskStaticHSStochTimeEvo.h"
#include "TaskStaticHSStochYields.h"
#include "Interaction.h"
#include "ObjectParser.h"
#include "Pulse.h"
#include "PowderGrid.h"
#include "Spin.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include "RunSection.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace
{
	struct HSGeneralSpectraSystem
	{
		std::shared_ptr<SpinAPI::SpinSystem> spinSystem;
		std::shared_ptr<SpinAPI::State> initialState;
	};

	HSGeneralSpectraSystem BuildHSGeneralSpectraSystem()
	{
		auto spin = std::make_shared<SpinAPI::Spin>(
			"E", "type=electron;spin=1/2;tensor=matrix(\"2.0 0 0;0 2.0 0;0 0 2.4\");");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"zeeman", "type=zeeman;spins=E;field=0 0 -0.004;ignoretensors=false;"
			"commonprefactor=true;prefactor=1.0;");
		auto state = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto pulse = std::make_shared<SpinAPI::Pulse>(
			"cw", "type=LongPulseStaticField;field=0.0002 0 0;pulsetime=2.0;timestep=0.1;"
			"group=E;prefactorlist=1,1,1;commonprefactorlist=true;ignoretensorslist=true;");

		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		system->Add(spin);
		system->Add(zeeman);
		system->Add(state);
		system->Add(pulse);
		system->ValidateInteractions();
		system->ValidatePulses();
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties", "initialstate=Up;"));
		return {system, state};
	}

	HSGeneralSpectraSystem BuildHSGeneralTraceSpectraSystem()
	{
		auto spin = std::make_shared<SpinAPI::Spin>(
			"E", "type=electron;spin=1/2;tensor=matrix(\"2.0 0 0;0 2.0 0;0 0 2.4\");");
		auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1;");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"zeeman", "type=zeeman;spins=E;field=0 0 -0.004;ignoretensors=false;"
			"commonprefactor=true;prefactor=1.0;");
		auto state = std::make_shared<SpinAPI::State>("ElectronUp", "spin(E)=|1/2>;");

		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		system->Add(spin);
		system->Add(nucleus);
		system->Add(zeeman);
		system->Add(state);
		system->ValidateInteractions();
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=ElectronUp;frame=molecular;initialstatecoherences=keep;"));
		return {system, state};
	}

	HSGeneralSpectraSystem BuildHSGeneralPowderYieldSystem()
	{
		auto electron1 = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;");
		auto electron2 = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1/2;");
		auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1/2;");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"B0", "type=zeeman;spins=E1,E2;field=0 0 0.001;ignoretensors=true;commonprefactor=false;");
		auto singlet = std::make_shared<SpinAPI::State>(
			"Singlet", "spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto sink = std::make_shared<SpinAPI::Transition>(
			"sink", "type=sink;sourcestate=Singlet;rate=0.02;", system);
		system->Add(electron1);
		system->Add(electron2);
		system->Add(nucleus);
		system->Add(zeeman);
		system->Add(singlet);
		system->Add(sink);
		system->ValidateInteractions();
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=Singlet;frame=molecular;initialstatecoherences=keep;"));
		return {system, singlet};
	}

	HSGeneralSpectraSystem BuildHSGeneralDynamicPowderSystem()
	{
		auto electron1 = std::make_shared<SpinAPI::Spin>(
			"E1", "type=electron;spin=1/2;tensor=matrix(2.0 0 0;0 2.2 0;0 0 2.5);");
		auto electron2 = std::make_shared<SpinAPI::Spin>(
			"E2", "type=electron;spin=1/2;tensor=isotropic(2.0);");
		auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1/2;");
		auto static1 = std::make_shared<SpinAPI::Interaction>(
			"B01", "type=zeeman;spins=E1;field=0 0 0.02;ignoretensors=false;"
			"commonprefactor=false;prefactor=1;");
		auto static2 = std::make_shared<SpinAPI::Interaction>(
			"B02", "type=zeeman;spins=E2;field=0 0 0.02;ignoretensors=false;"
			"commonprefactor=false;prefactor=1;");
		auto drive = std::make_shared<SpinAPI::Interaction>(
			"B1", "type=zeeman;spins=E1,E2;field=0.003 0 0;ignoretensors=true;"
			"commonprefactor=false;prefactor=1;fieldtype=linearpolarized;"
			"frequency=0.7;phase=0.2;");
		auto singlet = std::make_shared<SpinAPI::State>(
			"Singlet", "spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
		auto electron1Up = std::make_shared<SpinAPI::State>("E1Up", "spin(E1)=|1/2>;");

		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto sink = std::make_shared<SpinAPI::Transition>(
			"sink", "type=sink;sourcestate=Singlet;rate=0.02;", system);
		system->Add(electron1);
		system->Add(electron2);
		system->Add(nucleus);
		system->Add(static1);
		system->Add(static2);
		system->Add(drive);
		system->Add(singlet);
		system->Add(electron1Up);
		system->Add(sink);
		system->ValidateInteractions();
		electron1Up->ParseFromSystem(*system);
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=Singlet;frame=molecular;initialstatecoherences=keep;"));
		return {system, singlet};
	}

	bool RunHSGeneralSpectraTask(const HSGeneralSpectraSystem &_fixture,
								 const std::string &_type,
								 const std::string &_properties,
								 std::string &_data,
								 std::string *_log = nullptr)
	{
		if (!_fixture.initialState->ParseFromSystem(*_fixture.spinSystem) ||
			!_fixture.spinSystem->ValidateTransitions({_fixture.spinSystem}).empty())
			return false;

		RunSection::RunSection runSection;
		runSection.Add(_fixture.spinSystem);
		MSDParser::ObjectParser parser("task", "type=" + _type + ";" + _properties);
		if (!runSection.Add(MSDParser::ObjectType::Task, parser))
			return false;

		auto task = runSection.GetTask("task");
		std::ostringstream log;
		std::ostringstream data;
		task->SetLogStream(log);
		task->SetDataStream(data);
		if (!runSection.Run(1))
			return false;
		_data = data.str();
		if (_log != nullptr)
			*_log = log.str();
		return true;
	}

	template <typename ExpectedTask>
	bool HSGeneralCreates(const std::string &_properties)
	{
		RunSection::RunSection runSection;
		MSDParser::ObjectParser parser("general", "type=HSGeneral;" + _properties);
		if (!runSection.Add(MSDParser::ObjectType::Task, parser))
			return false;
		return std::dynamic_pointer_cast<ExpectedTask>(runSection.GetTask("general")) != nullptr;
	}

	bool HSGeneralRejects(const std::string &_properties, const std::string &_expectedMessage)
	{
		MSDParser::ObjectParser parser("general", "type=HSGeneral;" + _properties);
		RunSection::HSGeneralConfiguration configuration;
		std::string error;
		return !RunSection::ResolveHSGeneralConfiguration(parser, configuration, error) &&
			error.find(_expectedMessage) != std::string::npos;
	}

	bool SpectraDataNumericallyEqual(const std::string &_left, const std::string &_right,
								  double _tolerance = 1.0e-12)
	{
		std::istringstream left(_left);
		std::istringstream right(_right);
		std::string leftHeader;
		std::string rightHeader;
		if (!std::getline(left, leftHeader) || !std::getline(right, rightHeader) ||
			leftHeader != rightHeader)
		{
			return false;
		}

		double leftValue = 0.0;
		double rightValue = 0.0;
		while (left >> leftValue)
		{
			if (!(right >> rightValue))
				return false;
			const double scale = std::max({1.0, std::abs(leftValue), std::abs(rightValue)});
			if (std::abs(leftValue - rightValue) > _tolerance * scale)
				return false;
		}
		return !(right >> rightValue);
	}

	bool ParseNumericData(const std::string &_data, std::string &_header,
						  std::vector<std::vector<double>> &_rows)
	{
		std::istringstream stream(_data);
		if (!std::getline(stream, _header))
			return false;
		_rows.clear();
		std::string line;
		while (std::getline(stream, line))
		{
			if (line.empty())
				continue;
			std::istringstream rowStream(line);
			std::vector<double> row;
			double value = 0.0;
			while (rowStream >> value)
				row.push_back(value);
			if (!row.empty())
				_rows.push_back(std::move(row));
		}
		return !_rows.empty();
	}

	bool DynamicPowderMatchesExplicitOrientations(const std::string &_calculation,
										 const std::string &_sampling,
										 std::string *_internalData = nullptr,
										 std::string *_internalLog = nullptr)
	{
		const int pointCount = 3;
		const std::string common =
			"dynamics=dynamic;calculation=" + _calculation + ";sampling=" + _sampling +
			";approximation=full;hamiltonianh0list=B01,B02;totaltime=0.3;timestep=0.1;"
			"propagationmethod=normal;" +
			(_sampling == "stochastic"
				? "montecarlosamples=8;samplingmethod=suz;autoseed=false;seed=31;"
				: "");

		auto internalFixture = BuildHSGeneralDynamicPowderSystem();
		std::string internal;
		std::string log;
		if (!RunHSGeneralSpectraTask(internalFixture, "HSGeneral",
				common + "powdersamplingpoints=" + std::to_string(pointCount) + ";",
				internal, &log))
		{
			return false;
		}

		std::string internalHeader;
		std::vector<std::vector<double>> internalRows;
		if (!ParseNumericData(internal, internalHeader, internalRows))
			return false;

		SpinAPI::PowderGrid grid;
		if (!SpinAPI::CreateUniformPowderGrid(
				pointCount, SpinAPI::PowderGridDomain::UpperHemisphere, grid))
		{
			return false;
		}

		std::vector<std::vector<double>> externalSum;
		for (const auto &orientation : grid)
		{
			auto fixture = BuildHSGeneralDynamicPowderSystem();
			std::ostringstream orientationProperty;
			orientationProperty << std::setprecision(17)
				<< "powderorientation=" << orientation.theta << " "
				<< orientation.phi << " " << orientation.weight << ";";
			std::string external;
			if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
					common + orientationProperty.str(), external))
			{
				return false;
			}

			std::string externalHeader;
			std::vector<std::vector<double>> externalRows;
			if (!ParseNumericData(external, externalHeader, externalRows) ||
				externalHeader != internalHeader || externalRows.size() != internalRows.size())
			{
				return false;
			}

			if (externalSum.empty())
				externalSum.assign(externalRows.size(), std::vector<double>(externalRows.front().size(), 0.0));
			for (size_t row = 0; row < externalRows.size(); ++row)
			{
				if (externalRows[row].size() != internalRows[row].size())
					return false;
				externalSum[row][0] = externalRows[row][0];
				externalSum[row][1] = externalRows[row][1];
				for (size_t column = 2; column < externalRows[row].size(); ++column)
					externalSum[row][column] += externalRows[row][column];
			}
		}

		for (size_t row = 0; row < internalRows.size(); ++row)
		{
			for (size_t column = 0; column < internalRows[row].size(); ++column)
			{
				const double scale = std::max({1.0, std::abs(internalRows[row][column]),
					std::abs(externalSum[row][column])});
				if (std::abs(internalRows[row][column] - externalSum[row][column]) > 2e-10 * scale)
					return false;
			}
		}

		if (_internalData != nullptr)
			*_internalData = internal;
		if (_internalLog != nullptr)
			*_internalLog = log;
		return true;
	}

	bool RunHSGeneralHigherSpinTask(const std::string &_calculation,
									 std::string &_data,
									 std::string &_log)
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1;");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"B0", "type=zeeman;spins=E;field=0 0 0.001;ignoretensors=true;commonprefactor=false;");
		auto initial = std::make_shared<SpinAPI::State>("T0", "spin(E)=|0>;");

		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto sink = std::make_shared<SpinAPI::Transition>(
			"sink", "type=sink;sourcestate=T0;rate=0.01;", system);
		system->Add(spin);
		system->Add(zeeman);
		system->Add(initial);
		system->Add(sink);
		system->ValidateInteractions();
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=T0;initialstatecoherences=keep;"));
		if (!initial->ParseFromSystem(*system) || !system->ValidateTransitions({system}).empty())
			return false;

		RunSection::RunSection runSection;
		runSection.Add(system);
		MSDParser::ObjectParser parser(
			"general", "type=HSGeneral;dynamics=static;calculation=" + _calculation +
			";sampling=direct;totaltime=0.2;timestep=0.1;propagationmethod=normal;"
			"transitionyields=true;yieldcorrections=false;");
		if (!runSection.Add(MSDParser::ObjectType::Task, parser))
			return false;

		auto task = runSection.GetTask("general");
		std::ostringstream log;
		std::ostringstream data;
		task->SetLogStream(log);
		task->SetDataStream(data);
		if (!task->Run())
			return false;
		_data = data.str();
		_log = log.str();
		return true;
	}
}

bool test_hsgeneral_dispatches_proven_modes()
{
	bool correct = true;
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectTimeEvo>(
		"dynamics=static;calculation=timeevolution;sampling=direct;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectYields>(
		"dynamics=static;calculation=yields;sampling=direct;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSStochTimeEvo>(
		"dynamics=static;calculation=timeevolution;sampling=stochastic;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSStochYields>(
		"dynamics=static;calculation=yields;sampling=stochastic;");
	correct &= HSGeneralCreates<RunSection::TaskDynamicHSDirectTimeEvo>(
		"dynamics=dynamic;calculation=timeevolution;sampling=direct;");
	correct &= HSGeneralCreates<RunSection::TaskDynamicHSDirectYields>(
		"dynamics=dynamic;calculation=yields;sampling=direct;");
	correct &= HSGeneralCreates<RunSection::TaskDynamicHSStochTimeEvo>(
		"dynamics=dynamic;calculation=timeevolution;sampling=stochastic;");
	correct &= HSGeneralCreates<RunSection::TaskDynamicHSStochYields>(
		"dynamics=dynamic;calculation=yields;sampling=stochastic;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectSpectra>(
		"calculation=spectra;secularization=true;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectSpectra>(
		"calculation=spectra;sampling=stochastic;secularization=true;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSResonanceSpectra>(
		"calculation=spectra;secularization=false;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectSpectra>(
		"calculation=timeevolution;powdersamplingpoints=5;approximation=full;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectSpectra>(
		"calculation=yields;powdersamplingpoints=5;approximation=secular;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectSpectra>(
		"dynamics=dynamic;calculation=timeevolution;sampling=direct;powdersamplingpoints=5;");
	correct &= HSGeneralCreates<RunSection::TaskStaticHSDirectSpectra>(
		"dynamics=dynamic;calculation=yields;sampling=stochastic;powdersamplingpoints=5;");
	return correct;
}

bool test_hsgeneral_defaults_are_explicit_and_compatible()
{
	MSDParser::ObjectParser parser("general", "type=hsgeneral;");
	RunSection::HSGeneralConfiguration configuration;
	std::string error;
	return RunSection::ResolveHSGeneralConfiguration(parser, configuration, error) &&
		configuration.target == RunSection::HSGeneralTarget::StaticDirectTimeEvolution &&
		configuration.dynamics == "static" &&
		configuration.calculation == "timeevolution" &&
		configuration.sampling == "direct" &&
		configuration.approximation == "full" &&
		!configuration.powderAveraging;
}

bool test_hsgeneral_spectra_default_is_secular()
{
	MSDParser::ObjectParser parser("general", "type=hsgeneral;calculation=spectra;");
	RunSection::HSGeneralConfiguration configuration;
	std::string error;
	return RunSection::ResolveHSGeneralConfiguration(parser, configuration, error) &&
		configuration.target == RunSection::HSGeneralTarget::StaticDirectSpectra &&
		configuration.approximation == "secular";
}

bool test_hsgeneral_secular_spectra_preserve_legacy_output()
{
	auto legacySystem = BuildHSGeneralSpectraSystem();
	auto generalSystem = BuildHSGeneralSpectraSystem();

	const std::string properties =
		"method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=5;"
		"hamiltonianh0list=zeeman;printtimeframe=pulse;integrationtimeframe=pulse;"
		"pulsesequence=[\"cw 0\"];totaltime=1;timestep=0.1;propagationmethod=normal;";

	std::string legacyData;
	std::string generalData;
	if (!RunHSGeneralSpectraTask(legacySystem, "StaticHS-Direct-Spectra", properties, legacyData) ||
		!RunHSGeneralSpectraTask(generalSystem, "HSGeneral",
			"calculation=spectra;secularization=true;" + properties, generalData))
	{
		return false;
	}

	// HSGeneral must be a dispatch-only addition for proven modes. Identical
	// physical inputs must therefore retain the exact legacy output contract.
	return legacyData == generalData;
}

bool test_hsgeneral_stochastic_uses_state_aware_sampling()
{
	auto electron = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1;");
	auto initial = std::make_shared<SpinAPI::State>("N0", "spin(N)=|0>;");

	auto system = std::make_shared<SpinAPI::SpinSystem>("System");
	system->Add(electron);
	system->Add(nucleus);
	system->Add(initial);
	auto transition = std::make_shared<SpinAPI::Transition>(
		"sink", "type=sink;sourcestate=N0;rate=0.001;", system);
	system->Add(transition);
	auto properties = std::make_shared<MSDParser::ObjectParser>(
		"properties", "initialstate=N0;");
	system->SetProperties(properties);
	if (!initial->ParseFromSystem(*system) || !system->ValidateTransitions({system}).empty())
		return false;

	RunSection::RunSection runSection;
	runSection.Add(system);
	MSDParser::ObjectParser parser(
		"general",
		"type=HSGeneral;dynamics=static;calculation=timeevolution;sampling=stochastic;"
		"montecarlosamples=64;samplingmethod=suz;autoseed=false;seed=9;"
		"totaltime=0.1;timestep=0.1;propagationmethod=normal;");
	if (!runSection.Add(MSDParser::ObjectType::Task, parser))
		return false;

	auto task = runSection.GetTask("general");
	std::ostringstream log;
	std::ostringstream data;
	task->SetLogStream(log);
	task->SetDataStream(data);
	if (!task->Run())
		return false;

	return log.str().find("samples only omitted spins (subspace dimension 2)") != std::string::npos &&
		data.str().find("System.N0") != std::string::npos;
}

bool test_hsgeneral_powder_stochastic_spectra_preserve_direct_result()
{
	auto directSystem = BuildHSGeneralTraceSpectraSystem();
	auto stochasticSystem = BuildHSGeneralTraceSpectraSystem();
	const std::string common =
		"calculation=spectra;secularization=true;method=timeevo;integration=false;"
		"cidsp=false;spinlist=E;powdersamplingpoints=7;hamiltonianh0list=zeeman;"
		"printtimeframe=full;totaltime=0.2;timestep=0.1;propagationmethod=normal;";

	std::string directData;
	std::string stochasticData;
	std::string stochasticLog;
	if (!RunHSGeneralSpectraTask(directSystem, "HSGeneral",
			common + "sampling=direct;", directData) ||
		!RunHSGeneralSpectraTask(stochasticSystem, "HSGeneral",
			common + "sampling=stochastic;montecarlosamples=32;samplingmethod=suz;"
			"autoseed=false;seed=17;", stochasticData, &stochasticLog))
	{
		return false;
	}

	// N is absent from the initial State and Hamiltonian. Sampling its identity
	// factor is therefore analytically exact for the electron observables, not
	// merely convergent in the large-sample limit.
	const bool equalData = SpectraDataNumericallyEqual(directData, stochasticData);
	const bool hasSamplingLog = stochasticLog.find("samples only omitted spins (subspace dimension 3)") != std::string::npos;
	const bool hasEngineLog = stochasticLog.find("normalized trace samples in the shared powder propagation engine") != std::string::npos;
	return equalData && hasSamplingLog && hasEngineLog;
}

bool test_hsgeneral_secular_spectra_require_explicit_h0()
{
	auto fixture = BuildHSGeneralSpectraSystem();
	if (!fixture.initialState->ParseFromSystem(*fixture.spinSystem))
		return false;

	RunSection::RunSection runSection;
	runSection.Add(fixture.spinSystem);
	MSDParser::ObjectParser parser(
		"general", "type=HSGeneral;calculation=spectra;secularization=true;"
		"method=timeevo;spinlist=E;powdersamplingpoints=3;totaltime=0.1;timestep=0.1;");
	if (!runSection.Add(MSDParser::ObjectType::Task, parser))
		return false;

	auto task = runSection.GetTask("general");
	std::ostringstream log;
	task->SetLogStream(log);
	return !task->IsValid() &&
		log.str().find("requires hamiltonianh0list") != std::string::npos;
}

bool test_hsgeneral_direct_modes_accept_higher_spin_without_nuclei()
{
	std::string timeData;
	std::string timeLog;
	std::string yieldData;
	std::string yieldLog;
	if (!RunHSGeneralHigherSpinTask("timeevolution", timeData, timeLog) ||
		!RunHSGeneralHigherSpinTask("yields", yieldData, yieldLog))
	{
		return false;
	}

	const std::string factorMessage =
		"constructed the normalized initial density from 1 state component(s) as 1 Hilbert-space factor(s)";
	return timeLog.find(factorMessage) != std::string::npos &&
		yieldLog.find(factorMessage) != std::string::npos &&
		timeData.find("System.T0") != std::string::npos &&
		yieldData.find("System.sink.yield") != std::string::npos;
}

bool test_hsgeneral_static_powder_timeevolution_and_yields()
{
	auto directTimeFixture = BuildHSGeneralPowderYieldSystem();
	auto stochasticTimeFixture = BuildHSGeneralPowderYieldSystem();
	auto directYieldFixture = BuildHSGeneralPowderYieldSystem();
	auto stochasticYieldFixture = BuildHSGeneralPowderYieldSystem();
	const std::string common =
		"dynamics=static;powdersamplingpoints=5;hamiltonianh0list=B0;"
		"totaltime=0.2;timestep=0.1;propagationmethod=normal;";
	const std::string stochastic =
		"sampling=stochastic;montecarlosamples=16;samplingmethod=suz;autoseed=false;seed=23;";

	std::string directTimeData;
	std::string stochasticTimeData;
	std::string timeLog;
	std::string directYieldData;
	std::string stochasticYieldData;
	std::string yieldLog;
	if (!RunHSGeneralSpectraTask(directTimeFixture, "HSGeneral",
			"calculation=timeevolution;approximation=full;sampling=direct;" + common,
			directTimeData) ||
		!RunHSGeneralSpectraTask(stochasticTimeFixture, "HSGeneral",
			"calculation=timeevolution;approximation=full;" + stochastic + common,
			stochasticTimeData, &timeLog) ||
		!RunHSGeneralSpectraTask(directYieldFixture, "HSGeneral",
			"calculation=yields;approximation=secular;sampling=direct;" + common,
			directYieldData) ||
		!RunHSGeneralSpectraTask(stochasticYieldFixture, "HSGeneral",
			"calculation=yields;approximation=secular;" + stochastic + common,
			stochasticYieldData, &yieldLog))
	{
		return false;
	}

	// N is dynamically disconnected and omitted from the initial State. Its
	// stochastic identity trace is exact for both observables, so any numerical
	// disagreement here indicates a normalization or powder-rotation regression.
	return SpectraDataNumericallyEqual(directTimeData, stochasticTimeData) &&
		SpectraDataNumericallyEqual(directYieldData, stochasticYieldData) &&
		stochasticTimeData.find("System.Singlet") != std::string::npos &&
		stochasticTimeData.find("System.sink.yield") == std::string::npos &&
		stochasticYieldData.find("System.sink.yield") != std::string::npos &&
		timeLog.find("calculation=timeevolution selects method=timeevo") != std::string::npos &&
		timeLog.find("powder Hamiltonian approximation = full") != std::string::npos &&
		yieldLog.find("calculation=yields selects method=timeinf") != std::string::npos &&
		yieldLog.find("powder Hamiltonian approximation = secular") != std::string::npos &&
		yieldLog.find("samples only omitted spins (subspace dimension 2)") != std::string::npos;
}

bool test_hsgeneral_dynamic_powder_internal_external_and_sampling()
{
	std::string directTime;
	std::string stochasticTime;
	std::string directYield;
	std::string stochasticYield;
	std::string directTimeLog;
	std::string yieldLog;
	if (!DynamicPowderMatchesExplicitOrientations("timeevolution", "direct", &directTime, &directTimeLog) ||
		!DynamicPowderMatchesExplicitOrientations("timeevolution", "stochastic", &stochasticTime) ||
		!DynamicPowderMatchesExplicitOrientations("yields", "direct", &directYield, &yieldLog) ||
		!DynamicPowderMatchesExplicitOrientations("yields", "stochastic", &stochasticYield))
	{
		return false;
	}

	std::string header;
	std::vector<std::vector<double>> timeRows;
	std::vector<std::vector<double>> yieldRows;
	if (!ParseNumericData(directTime, header, timeRows) ||
		!ParseNumericData(directYield, header, yieldRows))
	{
		return false;
	}

	// The nucleus is omitted from the initial State and dynamically decoupled,
	// so trace sampling its identity is exact. Dynamic yields deliberately emit
	// one finite-time integrated result rather than a timeinf solve.
	return SpectraDataNumericallyEqual(directTime, stochasticTime, 2e-10) &&
		SpectraDataNumericallyEqual(directYield, stochasticYield, 2e-10) &&
		timeRows.size() == 3 && yieldRows.size() == 1 &&
		directTime.find("System.Singlet") != std::string::npos &&
		directTime.find("System.E1Up") != std::string::npos &&
		directYield.find("System.sink.yield") != std::string::npos &&
		directTimeLog.find("Orientation-dependent molecular-frame state observables are rotated") != std::string::npos &&
		yieldLog.find("finite-time yield integration") != std::string::npos &&
		yieldLog.find("Writing finite-time integrated transition yields") != std::string::npos;
}

bool test_hsgeneral_powder_rejects_orientation_dependent_yield_state()
{
	auto fixture = BuildHSGeneralSpectraSystem();
	auto sink = std::make_shared<SpinAPI::Transition>(
		"sink", "type=sink;sourcestate=Up;rate=0.01;", fixture.spinSystem);
	fixture.spinSystem->Add(sink);
	if (!fixture.initialState->ParseFromSystem(*fixture.spinSystem) ||
		!fixture.spinSystem->ValidateTransitions({fixture.spinSystem}).empty())
	{
		return false;
	}

	RunSection::RunSection runSection;
	runSection.Add(fixture.spinSystem);
	MSDParser::ObjectParser parser(
		"general", "type=HSGeneral;dynamics=static;calculation=yields;sampling=direct;"
		"approximation=full;powdersamplingpoints=3;hamiltonianh0list=zeeman;");
	if (!runSection.Add(MSDParser::ObjectType::Task, parser))
		return false;

	auto task = runSection.GetTask("general");
	std::ostringstream log;
	task->SetLogStream(log);
	return !task->IsValid() &&
		log.str().find("requires rotationally invariant transition source states") != std::string::npos;
}

bool test_hsgeneral_rejects_unsupported_physics()
{
	return HSGeneralRejects(
			   "calculation=spectra;sampling=stochastic;approximation=full;", "full-Hamiltonian stochastic") &&
		HSGeneralRejects(
			"calculation=timeevolution;powderaveraging=true;", "requires powdersamplingpoints") &&
		HSGeneralRejects(
			"calculation=yields;powdersamplingpoints=3;method=timeevo;", "requires method=timeinf") &&
		HSGeneralRejects(
			"dynamics=dynamic;calculation=yields;powdersamplingpoints=3;method=timeinf;", "requires method=timeevo") &&
		HSGeneralRejects(
			"calculation=yields;relaxationmodel=nakajimazwanzig;", "Nakajima-Zwanzig") &&
		HSGeneralRejects(
			"calculation=yields;approximation=secular;", "applies only");
}

void AddHSGeneralTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("HSGeneral dispatches proven task modes", test_hsgeneral_dispatches_proven_modes));
	_cases.push_back(test_case("HSGeneral defaults are explicit and compatible", test_hsgeneral_defaults_are_explicit_and_compatible));
	_cases.push_back(test_case("HSGeneral spectra retain the secular default", test_hsgeneral_spectra_default_is_secular));
	_cases.push_back(test_case("HSGeneral secular spectra preserve legacy output", test_hsgeneral_secular_spectra_preserve_legacy_output));
	_cases.push_back(test_case("HSGeneral stochastic mode uses state-aware sampling", test_hsgeneral_stochastic_uses_state_aware_sampling));
	_cases.push_back(test_case("HSGeneral powder stochastic spectra preserve direct result", test_hsgeneral_powder_stochastic_spectra_preserve_direct_result));
	_cases.push_back(test_case("HSGeneral secular spectra require explicit H0", test_hsgeneral_secular_spectra_require_explicit_h0));
	_cases.push_back(test_case("HSGeneral direct modes accept higher spin without nuclei", test_hsgeneral_direct_modes_accept_higher_spin_without_nuclei));
	_cases.push_back(test_case("HSGeneral static powder supports time evolution and yields", test_hsgeneral_static_powder_timeevolution_and_yields));
	_cases.push_back(test_case("HSGeneral dynamic powder matches external decomposition and sampling", test_hsgeneral_dynamic_powder_internal_external_and_sampling));
	_cases.push_back(test_case("HSGeneral powder rejects orientation-dependent yield states", test_hsgeneral_powder_rejects_orientation_dependent_yield_state));
	_cases.push_back(test_case("HSGeneral rejects unsupported physics", test_hsgeneral_rejects_unsupported_physics));
}
