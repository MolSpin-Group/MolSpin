//////////////////////////////////////////////////////////////////////////////
// HSGeneral modular production-path and legacy-compatibility tests.
//////////////////////////////////////////////////////////////////////////////
#include "HSExecutionPlan.h"
#include "HSHamiltonianBuilder.h"
#include "HSObservableCollector.h"
#include "HSOrientationSampler.h"
#include "HSPropagator.h"
#include "HSReactionRelaxation.h"
#include "HSStatePreparation.h"
#include "TaskHSGeneral.h"
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
#include "Operator.h"
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

	HSGeneralSpectraSystem BuildHSGeneralPulseSystem()
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
		auto up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto flip = std::make_shared<SpinAPI::Pulse>(
			"flip", "type=InstantPulse;angle=180;rotationaxis=1 0 0;group=E;");
		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		system->Add(spin);
		system->Add(up);
		system->Add(flip);
		system->ValidatePulses();
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=Up;initialstatecoherences=keep;"));
		return {system, up};
	}

	HSGeneralSpectraSystem BuildHSGeneralPolarizationYieldSystem()
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
		auto up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto sink = std::make_shared<SpinAPI::Transition>(
			"sink", "type=sink;sourcestate=Up;rate=0.2;", system);
		system->Add(spin);
		system->Add(up);
		system->Add(sink);
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=Up;initialstatecoherences=keep;"));
		return {system, up};
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

	HSGeneralSpectraSystem BuildHSGeneralRelaxationSystem()
	{
		// The frozen direct-HS tasks require the historical radical-pair layout:
		// two spin-1/2 electrons plus at least one nuclear spin.  Keep the
		// nucleus uncoupled so this fixture tests relaxation parity rather than
		// an unrelated Hamiltonian difference.
		auto electron1 = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;");
		auto electron2 = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1/2;");
		auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1/2;");
		auto initial = std::make_shared<SpinAPI::State>(
			"Singlet", "spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
		auto relaxation = std::make_shared<SpinAPI::Operator>(
			"T1", "type=relaxationt1;spins=E1;rate=0.05;");
		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto sink = std::make_shared<SpinAPI::Transition>(
			"sink", "type=sink;sourcestate=Singlet;rate=0.02;", system);
		system->Add(electron1);
		system->Add(electron2);
		system->Add(nucleus);
		system->Add(initial);
		system->Add(relaxation);
		system->Add(sink);
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=Singlet;initialstatecoherences=keep;"));
		if (!initial->ParseFromSystem(*system)) return {nullptr, nullptr};
		const std::vector<SpinAPI::system_ptr> systems{system};
		if (!system->ValidateOperators(systems).empty()) return {nullptr, nullptr};
		return {system, initial};
	}

	HSGeneralSpectraSystem BuildHSGeneralDynamicRelaxationSystem()
	{
		auto electron1 = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;");
		auto electron2 = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1/2;");
		auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1/2;");
		auto staticField = std::make_shared<SpinAPI::Interaction>(
			"B0", "type=zeeman;spins=E1,E2;field=0 0 0.001;ignoretensors=true;commonprefactor=false;prefactor=1;");
		auto drive = std::make_shared<SpinAPI::Interaction>(
			"B1", "type=zeeman;spins=E1;field=0.0004 0 0;ignoretensors=true;commonprefactor=false;prefactor=1;"
			"fieldtype=linearpolarized;frequency=0.7;phase=0.2;");
		auto initial = std::make_shared<SpinAPI::State>(
			"Singlet", "spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
		auto relaxation = std::make_shared<SpinAPI::Operator>(
			"T1", "type=relaxationt1;spins=E1;rate=0.05;");
		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto sink = std::make_shared<SpinAPI::Transition>(
			"sink", "type=sink;sourcestate=Singlet;rate=0.02;", system);
		system->Add(electron1);
		system->Add(electron2);
		system->Add(nucleus);
		system->Add(staticField);
		system->Add(drive);
		system->Add(initial);
		system->Add(relaxation);
		system->Add(sink);
		if (!system->ValidateInteractions().empty()) return {nullptr, nullptr};
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"properties", "initialstate=Singlet;initialstatecoherences=keep;"));
		if (!initial->ParseFromSystem(*system)) return {nullptr, nullptr};
		const std::vector<SpinAPI::system_ptr> systems{system};
		if (!system->ValidateOperators(systems).empty()) return {nullptr, nullptr};
		return {system, initial};
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
			"properties", "initialstate=Singlet;frame=molecular;initialstatecoherences=keep;observablestateframe=molecular;"));
		return {system, singlet};
	}

	HSGeneralSpectraSystem BuildLuDMDirectSpectraRegressionSystem()
	{
		auto e1 = std::make_shared<SpinAPI::Spin>("e1", "spin=1;type=electron;tensor=isotropic(1.996);");
		auto D = std::make_shared<SpinAPI::Interaction>("D", "type=zfs;group1=e1;dvalue=-2340;evalue=304.2;prefactor=0.00628318530718;ignoretensors=true;commonprefactor=false;");
		auto Z = std::make_shared<SpinAPI::Interaction>("Z_e1", "type=zeeman;field=0 0 0.18;spins=e1;commonprefactor=true;ignoretensors=false;prefactor=1;");
		auto rot = std::make_shared<SpinAPI::Interaction>("pulse1_rot", "type=zeeman;field=0 0 -0.3499738;spins=e1;commonprefactor=true;ignoretensors=true;prefactor=2.002319304361;");
		auto mw = std::make_shared<SpinAPI::Interaction>("pulse1_mwamp", "type=zeeman;field=1e-7 0 0;spins=e1;commonprefactor=true;ignoretensors=false;prefactor=1;");
		auto Tz = std::make_shared<SpinAPI::State>("Tz", "spin(e1)=|0>;");
		auto Ty = std::make_shared<SpinAPI::State>("Ty", "spin(e1)=i|-1>+i|1>;");
		auto Tx = std::make_shared<SpinAPI::State>("Tx", "spin(e1)=|-1>-|1>;");
		auto T2 = std::make_shared<SpinAPI::Operator>("T2", "type=relaxationt2;rate=1.0;spins=e1;");
		auto T1 = std::make_shared<SpinAPI::Operator>("T1", "type=relaxationt1;rate=0.000133;spins=e1;");
		auto system = std::make_shared<SpinAPI::SpinSystem>("System");
		system->Add(e1); system->Add(D); system->Add(Z); system->Add(rot); system->Add(mw);
		system->Add(Tx); system->Add(Ty); system->Add(Tz); system->Add(T2); system->Add(T1);
		if (!system->ValidateInteractions().empty()) return {nullptr, nullptr};
		if (!system->ValidateStates().empty()) return {nullptr, nullptr};
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties", "initialstate=Tx,Ty,Tz;weights=0,0.47,0.53;frame=molecular;initialstatecoherences=dephase;"));
		const std::vector<SpinAPI::system_ptr> systems{system};
		if (!system->ValidateOperators(systems).empty()) return {nullptr, nullptr};
		return {system, Tx};
	}

	HSGeneralSpectraSystem BuildCISSDirectSpectraRegressionSystem()
	{
		auto FE1 = std::make_shared<SpinAPI::Spin>("FE1", "type=electron;spin=1/2;tensor=matrix(\"2.002804590178534 -0.000536758923585 0.000244421840035;-0.000536758923585 2.002604811480572 -0.000014943721405;0.000244421840035 -0.000014943721405 2.002490598340894\");");
		auto WE2 = std::make_shared<SpinAPI::Spin>("WE2", "type=electron;spin=1/2;tensor=matrix(\"2.002534889640355 0.000129163275332 -0.001006715399340;0.000129163275332 2.006438812220972 -0.000388279130307;-0.001006715399340 -0.000388279130307 2.005226298138673\");");
		auto z1 = std::make_shared<SpinAPI::Interaction>("zeeman1", "type=zeeman;field=0 0 3.380;group1=FE1;ignoretensors=false;commonprefactor=true;prefactor=1;");
		auto z2 = std::make_shared<SpinAPI::Interaction>("zeeman2", "type=zeeman;field=0 0 3.380;group1=WE2;ignoretensors=false;commonprefactor=true;prefactor=1;");
		auto mw = std::make_shared<SpinAPI::Interaction>("zeemanmw", "type=zeeman;field=0.00005 0 0;group1=FE1,WE2;ignoretensors=false;commonprefactor=true;prefactor=1;");
		auto rot = std::make_shared<SpinAPI::Interaction>("zeemanrotating", "type=zeeman;field=0 0 -3.38983638869464;group1=FE1,WE2;ignoretensors=true;commonprefactor=true;prefactor=2.002319304;");
		auto dip = std::make_shared<SpinAPI::Interaction>("dipolar", "type=doublespin;group1=FE1;group2=WE2;orientation=0,0,0;tensor=matrix(\"0.000082666666667 0 0;0 0.000082666666667 0;0 0 -0.000165333333333\");ignoretensors=true;commonprefactor=true;prefactor=2.002319304;");
		auto S = std::make_shared<SpinAPI::State>("Singlet", "spins(FE1,WE2)=|1/2,-1/2>-|-1/2,1/2>;");
		auto T0 = std::make_shared<SpinAPI::State>("T0", "spins(FE1,WE2)=|1/2,-1/2>+|-1/2,1/2>;");
		auto CISS = std::make_shared<SpinAPI::State>("CISS", "chi=1.318116071652818;spins(FE1,WE2)=cos(0.5chi)(|1/2,-1/2>-|-1/2,1/2>)+sin(0.5chi)(|1/2,-1/2>+|-1/2,1/2>);");
		auto identity = std::make_shared<SpinAPI::State>("Identity", "");
		auto system = std::make_shared<SpinAPI::SpinSystem>("system1");
		auto sink = std::make_shared<SpinAPI::Transition>("Product2", "type=sink;source=Identity;rate=0.025;", system);
		system->Add(FE1); system->Add(WE2); system->Add(z1); system->Add(z2); system->Add(mw); system->Add(rot); system->Add(dip);
		system->Add(S); system->Add(T0); system->Add(CISS); system->Add(identity); system->Add(sink);
		if (!system->ValidateInteractions().empty()) return {nullptr, nullptr};
		if (!system->ValidateStates().empty()) return {nullptr, nullptr};
		system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties", "initialstate=CISS;stateframe=molecular;initialstatecoherences=dephase;"));
		if (!system->ValidateTransitions({system}).empty()) return {nullptr, nullptr};
		return {system, CISS};
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

	template <typename ExpectedTask>
	bool LegacyTaskCreates(const std::string &_type)
	{
		RunSection::RunSection runSection;
		MSDParser::ObjectParser parser("legacy", "type=" + _type + ";");
		if (!runSection.Add(MSDParser::ObjectType::Task, parser))
			return false;
		return std::dynamic_pointer_cast<ExpectedTask>(runSection.GetTask("legacy")) != nullptr;
	}

	bool HSGeneralRejects(const std::string &_properties, const std::string &_expectedMessage)
	{
		MSDParser::ObjectParser parser("general", "type=HSGeneral;" + _properties);
		RunSection::General::HS::HSExecutionPlan plan;
		std::string error;
		return !RunSection::General::HS::ResolveExecutionPlan(parser, plan, error) &&
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

	bool SpectraDataEqualOnSharedTimeline(const std::string &_legacy,
		const std::string &_general, double _legacyObservableScale,
		double _tolerance)
	{
		std::string legacyHeader;
		std::string generalHeader;
		std::vector<std::vector<double>> legacyRows;
		std::vector<std::vector<double>> generalRows;
		if (!ParseNumericData(_legacy, legacyHeader, legacyRows) ||
			!ParseNumericData(_general, generalHeader, generalRows))
			return false;

		size_t legacyRow = 0;
		size_t generalRow = 0;
		size_t matchedRows = 0;
		while (legacyRow < legacyRows.size() && generalRow < generalRows.size())
		{
			if (legacyRows[legacyRow].size() != generalRows[generalRow].size() ||
				legacyRows[legacyRow].size() < 2)
				return false;

			const double timeDifference = legacyRows[legacyRow][1] - generalRows[generalRow][1];
			if (std::abs(timeDifference) <= 1.0e-12)
			{
				for (size_t column = 0; column < legacyRows[legacyRow].size(); ++column)
				{
					const double legacyValue = column < 2
						? legacyRows[legacyRow][column]
						: legacyRows[legacyRow][column] / _legacyObservableScale;
					const double generalValue = generalRows[generalRow][column];
					const double scale = std::max({1.0, std::abs(legacyValue), std::abs(generalValue)});
					if (std::abs(legacyValue - generalValue) > _tolerance * scale)
						return false;
				}
				++matchedRows;
				++legacyRow;
				++generalRow;
			}
			else if (timeDifference < 0.0)
			{
				++legacyRow;
			}
			else
			{
				++generalRow;
			}
		}

		// The spectroscopy task omits its initial boundary, while HSGeneral emits
		// both t=0 and the exact requested endpoint. Every legacy sample must
		// therefore occur on the General timeline; one General boundary row may be
		// task-specific.
		return matchedRows + 1 >= std::min(legacyRows.size(), generalRows.size());
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
				<< "powderorientation=0 " << orientation.theta << " "
				<< orientation.phi << " " << orientation.weight / (2.0 * arma::datum::pi) << ";";
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
			const size_t metadataColumns = (_calculation == "timeevolution") ? 2u : 1u;
			for (size_t row = 0; row < externalRows.size(); ++row)
			{
				if (externalRows[row].size() != internalRows[row].size())
					return false;
				for (size_t column = 0; column < metadataColumns; ++column)
					externalSum[row][column] = externalRows[row][column];
				for (size_t column = metadataColumns; column < externalRows[row].size(); ++column)
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

	bool DynamicPowderGammaMatchesExplicitOrientations()
	{
		const int pointCount = 3;
		const int gammaCount = 4;
		const std::string common =
			"dynamics=dynamic;calculation=yields;sampling=stochastic;approximation=full;"
			"hamiltonianh0list=B01,B02;totaltime=0.3;timestep=0.1;"
			"propagationmethod=normal;montecarlosamples=8;samplingmethod=suz;"
			"autoseed=false;seed=31;";

		auto internalFixture = BuildHSGeneralDynamicPowderSystem();
		std::string internal;
		std::string internalLog;
		if (!RunHSGeneralSpectraTask(internalFixture, "HSGeneral",
				common + "powdersamplingpoints=" + std::to_string(pointCount) +
				";powdergammapoints=" + std::to_string(gammaCount) + ";",
				internal, &internalLog))
		{
			return false;
		}

		std::string internalHeader;
		std::vector<std::vector<double>> internalRows;
		if (!ParseNumericData(internal, internalHeader, internalRows))
			return false;

		SpinAPI::PowderGrid grid;
		if (!SpinAPI::CreateUniformPowderGrid(
				pointCount, SpinAPI::PowderGridDomain::FullSphere, grid))
		{
			return false;
		}

		std::vector<std::vector<double>> externalSum;
		for (const auto &orientation : grid)
		{
			for (int gammaIndex = 0; gammaIndex < gammaCount; ++gammaIndex)
			{
				const double gamma = 2.0 * arma::datum::pi *
					(static_cast<double>(gammaIndex) + 0.5) / static_cast<double>(gammaCount);
				auto fixture = BuildHSGeneralDynamicPowderSystem();
				std::ostringstream orientationProperty;
				orientationProperty << std::setprecision(17)
					<< "powderorientation=" << gamma << " " << orientation.theta << " "
					<< orientation.phi << " "
					<< orientation.weight / (4.0 * arma::datum::pi * static_cast<double>(gammaCount)) << ";";
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
					externalSum[row][0] = externalRows[row][0];
					for (size_t column = 1; column < externalRows[row].size(); ++column)
						externalSum[row][column] += externalRows[row][column];
				}
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

		return internalLog.find("samples the third Euler angle with 4 points") != std::string::npos &&
			internalLog.find("12 total SO(3) orientations") != std::string::npos;
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


bool test_hsgeneral_matches_ludm_directspectra_relaxation_rwa_regression()
{
	auto legacyFixture = BuildLuDMDirectSpectraRegressionSystem();
	auto generalFixture = BuildLuDMDirectSpectraRegressionSystem();
	if (!legacyFixture.spinSystem || !generalFixture.spinSystem) return false;
	const std::string common = "totaltime=4;timestep=1;cidsp=false;spinlist=e1;hamiltonianh0list=D,Z_e1;hamiltonianh1list=pulse1_mwamp,pulse1_rot;powdersamplingpoints=12;";
	std::string legacy, general;
	if (!RunHSGeneralSpectraTask(legacyFixture, "StaticHS-Direct-Spectra", "method=timeevo;" + common, legacy) ||
		!RunHSGeneralSpectraTask(generalFixture, "HSGeneral", "dynamics=static;calculation=timeevolution;sampling=direct;approximation=secular;" + common, general)) return false;
	return SpectraDataEqualOnSharedTimeline(
		legacy, general, 2.0 * arma::datum::pi, 2.0e-12);
}

bool test_hsgeneral_matches_ciss_directspectra_timeinf_polarization_regression()
{
	auto legacyFixture = BuildCISSDirectSpectraRegressionSystem();
	auto generalFixture = BuildCISSDirectSpectraRegressionSystem();
	if (!legacyFixture.spinSystem || !generalFixture.spinSystem) return false;
	const std::string common = "cidsp=false;spinlist=FE1,WE2;hamiltonianh0list=zeeman1,zeeman2,dipolar;hamiltonianh1list=zeemanrotating,zeemanmw;initialstatehamiltonian=zeeman1,zeeman2,dipolar;powderfullsphere=true;powdersamplingpoints=12;reactionoperators=haberkorn;";
	std::string legacy, general;
	if (!RunHSGeneralSpectraTask(legacyFixture, "StaticHS-Direct-Spectra", "method=timeinf;" + common, legacy) ||
		!RunHSGeneralSpectraTask(generalFixture, "HSGeneral", "dynamics=static;calculation=yields;sampling=direct;approximation=secular;method=timeinf;" + common, general)) return false;
	std::string legacyNumeric = legacy;
	const std::string infToken = " inf ";
	const size_t infPos = legacyNumeric.find(infToken);
	if (infPos != std::string::npos) legacyNumeric.replace(infPos, infToken.size(), " 0 ");
	std::string lh, gh; std::vector<std::vector<double>> lr, gr;
	if (!ParseNumericData(legacyNumeric, lh, lr) || !ParseNumericData(general, gh, gr) || lr.size()!=1 || gr.size()!=1 || lr[0].size()!=8 || gr[0].size()!=7) return false;
	for (size_t c=0;c<6;++c) if (std::abs(lr[0][c+2]/(4.0*arma::datum::pi)-gr[0][c+1]) > 2.0e-12) return false;
	return true;
}

bool test_hsgeneral_dispatches_proven_modes()
{
	using RunSection::General::HS::TaskHSGeneral;
	bool correct = true;
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=static;calculation=timeevolution;sampling=direct;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=static;calculation=yields;sampling=direct;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=static;calculation=timeevolution;sampling=stochastic;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=static;calculation=yields;sampling=stochastic;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=dynamic;calculation=timeevolution;sampling=direct;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=dynamic;calculation=yields;sampling=direct;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=dynamic;calculation=timeevolution;sampling=stochastic;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=dynamic;calculation=yields;sampling=stochastic;");
	correct &= HSGeneralCreates<TaskHSGeneral>("calculation=timeevolution;powdersamplingpoints=5;hamiltonianh0list=H0;approximation=full;");
	correct &= HSGeneralCreates<TaskHSGeneral>("calculation=yields;powdersamplingpoints=5;hamiltonianh0list=H0;approximation=secular;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=dynamic;calculation=timeevolution;sampling=direct;powdersamplingpoints=5;hamiltonianh0list=H0;");
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=dynamic;calculation=yields;sampling=stochastic;powdersamplingpoints=5;hamiltonianh0list=H0;");

	// Spectroscopy is deliberately standalone and never an HSGeneral backend.
	correct &= LegacyTaskCreates<RunSection::TaskStaticHSDirectSpectra>("StaticHS-Direct-Spectra");
	correct &= LegacyTaskCreates<RunSection::TaskStaticHSResonanceSpectra>("StaticHS-Resonance-Spectra");

	// Legacy task names remain independently constructible; HSGeneral does not use them as backends.
	correct &= HSGeneralCreates<TaskHSGeneral>("dynamics=static;calculation=timeevolution;sampling=direct;");
	return correct;
}

bool test_hsgeneral_keeps_eight_core_legacy_tasks_independent()
{
	return LegacyTaskCreates<RunSection::TaskStaticHSDirectTimeEvo>("StaticHS-Direct-TimeEvo") &&
		LegacyTaskCreates<RunSection::TaskStaticHSDirectYields>("StaticHS-Direct-Yields") &&
		LegacyTaskCreates<RunSection::TaskStaticHSStochTimeEvo>("StaticHS-Stoch-TimeEvo") &&
		LegacyTaskCreates<RunSection::TaskStaticHSStochYields>("StaticHS-Stoch-Yields") &&
		LegacyTaskCreates<RunSection::TaskDynamicHSDirectTimeEvo>("DynamicHS-Direct-TimeEvo") &&
		LegacyTaskCreates<RunSection::TaskDynamicHSDirectYields>("DynamicHS-Direct-Yields") &&
		LegacyTaskCreates<RunSection::TaskDynamicHSStochTimeEvo>("DynamicHS-Stoch-TimeEvo") &&
		LegacyTaskCreates<RunSection::TaskDynamicHSStochYields>("DynamicHS-Stoch-Yields");
}

bool test_hsgeneral_static_direct_matches_frozen_legacy_reference()
{
	auto legacyTimeFixture = BuildHSGeneralPowderYieldSystem();
	auto generalTimeFixture = BuildHSGeneralPowderYieldSystem();
	auto legacyYieldFixture = BuildHSGeneralPowderYieldSystem();
	auto generalYieldFixture = BuildHSGeneralPowderYieldSystem();

	const std::string tail =
		"timestep=0.1;propagationmethod=normal;"
		"transitionyields=true;yieldcorrections=false;";
	std::string legacyTime, generalTime, legacyYield, generalYield;
	// The frozen tasks allocate ceil(t_total/dt) samples beginning at t=0,
	// so requesting T+dt is necessary to compare the same physical [0,T]
	// trajectory/integral with endpoint-correct HSGeneral.
	if (!RunHSGeneralSpectraTask(legacyTimeFixture, "StaticHS-Direct-TimeEvo",
			"totaltime=0.4;" + tail, legacyTime) ||
		!RunHSGeneralSpectraTask(generalTimeFixture, "HSGeneral",
			"dynamics=static;calculation=timeevolution;sampling=direct;totaltime=0.3;" + tail, generalTime) ||
		!RunHSGeneralSpectraTask(legacyYieldFixture, "StaticHS-Direct-Yields",
			"totaltime=0.4;" + tail, legacyYield) ||
		!RunHSGeneralSpectraTask(generalYieldFixture, "HSGeneral",
			"dynamics=static;calculation=yields;sampling=direct;totaltime=0.3;" + tail, generalYield))
	{
		return false;
	}

	// The frozen legacy writers use lower output precision, so compare the
	// numerical contract rather than byte-for-byte formatting.
	return SpectraDataNumericallyEqual(legacyTime, generalTime, 2.0e-6) &&
		SpectraDataNumericallyEqual(legacyYield, generalYield, 2.0e-6);
}

bool test_hsgeneral_defaults_are_explicit_and_compatible()
{
	MSDParser::ObjectParser parser("general", "type=hsgeneral;");
	RunSection::General::HS::HSExecutionPlan plan;
	std::string error;
	return RunSection::General::HS::ResolveExecutionPlan(parser, plan, error) &&
		plan.dynamics == RunSection::General::HS::Dynamics::Static &&
		plan.calculation == RunSection::General::HS::Calculation::TimeEvolution &&
		plan.sampling == RunSection::General::HS::Sampling::Direct &&
		plan.approximation == SpinAPI::HamiltonianApproximation::Full &&
		plan.orientation == RunSection::General::HS::OrientationMode::Identity &&
		plan.observableMode == RunSection::General::HS::ObservableMode::Auto &&
		plan.propagation == RunSection::General::HS::PropagationMethod::Exponential;
}

bool test_hsgeneral_explicit_observable_modes_are_unambiguous()
{
	auto statesFixture = BuildHSGeneralPolarizationYieldSystem();
	auto spinsFixture = BuildHSGeneralPolarizationYieldSystem();
	auto yieldsFixture = BuildHSGeneralPolarizationYieldSystem();
	auto cidspFixture = BuildHSGeneralPolarizationYieldSystem();
	std::string statesData, spinsData, yieldsData, cidspData, spinsLog;
	if (!RunHSGeneralSpectraTask(statesFixture, "HSGeneral",
			"calculation=timeevolution;observables=states;totaltime=0.1;timestep=0.1;",
			statesData) ||
		!RunHSGeneralSpectraTask(spinsFixture, "HSGeneral",
			"calculation=timeevolution;observables=spins;spinlist=E;totaltime=0.1;timestep=0.1;",
			spinsData, &spinsLog) ||
		!RunHSGeneralSpectraTask(yieldsFixture, "HSGeneral",
			"calculation=yields;observables=transitionyields;method=timeinf;",
			yieldsData) ||
		!RunHSGeneralSpectraTask(cidspFixture, "HSGeneral",
			"calculation=yields;observables=cidsp;spinlist=E;method=timeinf;",
			cidspData))
	{
		return false;
	}

	return statesData.find("System.Up") != std::string::npos &&
		statesData.find("System.E.Ix") == std::string::npos &&
		spinsData.find("System.E.Ix") != std::string::npos &&
		yieldsData.find("System.sink.yield") != std::string::npos &&
		cidspData.find("System.E.sink.yield.Ix") != std::string::npos &&
		spinsLog.find("observables=spins") != std::string::npos &&
		HSGeneralRejects(
			"calculation=timeevolution;observables=states;spinlist=E;",
			"cannot be combined with spinlist") &&
		HSGeneralRejects(
			"calculation=timeevolution;observables=spins;",
			"requires spinlist") &&
		HSGeneralRejects(
			"calculation=timeevolution;observables=transitionyields;",
			"requires calculation=yields") &&
		HSGeneralRejects(
			"calculation=yields;observables=cidsp;",
			"requires spinlist") &&
		HSGeneralRejects(
			"calculation=timeevolution;observables=unknown;",
			"observables must be");
}

bool test_hsgeneral_pulse_preparation_rotates_polarization()
{
	auto fixture = BuildHSGeneralPulseSystem();
	std::string data;
	std::string log;
	if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
		"dynamics=static;calculation=timeevolution;sampling=direct;spinlist=E;"
		"pulsesequence=[\"flip 0\"];totaltime=0.2;timestep=0.1;propagationmethod=normal;",
		data, &log)) return false;

	std::string header;
	std::vector<std::vector<double>> rows;
	if (!ParseNumericData(data, header, rows) || rows.empty() || rows.front().size() != 5)
		return false;
	// Columns: Step, Time, Ix, Iy, Iz. A 180-degree x pulse maps |up> to |down>.
	return std::abs(rows.front()[2]) < 1.0e-10 &&
		std::abs(rows.front()[3]) < 1.0e-10 &&
		std::abs(rows.front()[4] + 0.5) < 1.0e-10 &&
		log.find("Applied instant pulse \"flip\"") != std::string::npos;
}

bool test_hsgeneral_finite_pulse_includes_partial_final_interval()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	auto up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
	auto pulse = std::make_shared<SpinAPI::Pulse>("finite",
		"type=LongPulseStaticField;field=0.001 0 0;pulsetime=0.25;timestep=0.1;"
		"group=E;prefactorlist=1,1,1;commonprefactorlist=false;ignoretensorslist=true;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("System");
	system->Add(spin);
	system->Add(up);
	system->Add(pulse);
	if (!up->ParseFromSystem(*system) || !system->ValidatePulses().empty())
		return false;
	system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
		"properties", "initialstate=Up;initialstatecoherences=keep;"));

	HSGeneralSpectraSystem fixture{system, up};
	std::string data;
	std::string log;
	if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
		"dynamics=static;calculation=timeevolution;sampling=direct;spinlist=E;"
		"pulsesequence=[\"finite 0\"];printtimeframe=pulse;totaltime=0;"
		"timestep=0.1;propagationmethod=normal;", data, &log))
		return false;

	std::string header;
	std::vector<std::vector<double>> rows;
	if (!ParseNumericData(data, header, rows) || rows.size() != 4)
		return false;
	const std::vector<double> expectedTimes{0.0, 0.1, 0.2, 0.25};
	for (size_t i = 0; i < rows.size(); ++i)
		if (rows[i].size() < 2 || std::abs(rows[i][1] - expectedTimes[i]) > 1.0e-13)
			return false;
	return log.find("Applied finite pulse \"finite\" for 0.25 ns") != std::string::npos;
}

bool test_hsgeneral_dynamic_factor_rk4_uses_stage_generators()
{
	using namespace RunSection::General::HS;
	HSExecutionPlan plan;
	plan.propagation = PropagationMethod::RK4;
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	SpinAPI::SpinSpace space(std::vector<SpinAPI::spin_ptr>{spin});
	HSPropagator propagator(plan, space);

	const double h = 0.2;
	arma::sp_cx_mat zero(1, 1);
	arma::sp_cx_mat hStart(1, 1);
	arma::sp_cx_mat hMid(1, 1);
	arma::sp_cx_mat hEnd(1, 1);
	hMid(0, 0) = 0.25 * h * h;
	hEnd(0, 0) = h * h;
	arma::cx_mat factors(1, 1, arma::fill::ones);
	std::string error;
	if (!propagator.StepDynamicRK4(hStart, zero, hMid, zero, hEnd, zero,
		h, factors, error))
		return false;

	// H(t)=t^2 is scalar, so the exact factor is exp[-i integral_0^h t^2 dt].
	// A frozen midpoint generator would use h^3/4 and fails this check; the RK4
	// stages integrate the time dependence with Simpson weights to h^3/3.
	const arma::cx_double expected = std::exp(
		-arma::cx_double(0.0, 1.0) * (h * h * h / 3.0));
	return std::abs(factors(0, 0) - expected) < 1.0e-6;
}

bool test_hsgeneral_quantum_yield_cidnp_and_timeinf_polarization()
{
	auto quantumFixture = BuildHSGeneralPolarizationYieldSystem();
	auto cidnpFixture = BuildHSGeneralPolarizationYieldSystem();
	auto polarizationFixture = BuildHSGeneralPolarizationYieldSystem();
	std::string quantumData, cidnpData, polarizationData;
	if (!RunHSGeneralSpectraTask(quantumFixture, "HSGeneral",
		"dynamics=static;calculation=yields;sampling=direct;method=timeinf;transitionyields=true;",
		quantumData) ||
		!RunHSGeneralSpectraTask(cidnpFixture, "HSGeneral",
			"dynamics=static;calculation=yields;sampling=direct;method=timeinf;"
			"transitionyields=false;spinlist=E;cidnp=true;", cidnpData) ||
		!RunHSGeneralSpectraTask(polarizationFixture, "HSGeneral",
			"dynamics=static;calculation=yields;sampling=direct;method=timeinf;"
			"transitionyields=false;spinlist=E;cidsp=false;", polarizationData)) return false;

	std::string quantumHeader, cidnpHeader, polarizationHeader;
	std::vector<std::vector<double>> quantumRows, cidnpRows, polarizationRows;
	if (!ParseNumericData(quantumData, quantumHeader, quantumRows) ||
		!ParseNumericData(cidnpData, cidnpHeader, cidnpRows) ||
		!ParseNumericData(polarizationData, polarizationHeader, polarizationRows) ||
		quantumRows.front().size() != 2 || cidnpRows.front().size() != 4 ||
		polarizationRows.front().size() != 4) return false;

	// One |up> state with a single k=0.2 sink: total product yield = 1;
	// product Iz yield = +1/2; integrated surviving Iz = (1/2)/k = 2.5.
	return quantumHeader.find("System.sink.yield") != std::string::npos &&
		cidnpHeader.find("System.E.sink.yield.Iz") != std::string::npos &&
		polarizationHeader.find("System.E.Iz") != std::string::npos &&
		std::abs(quantumRows.front()[1] - 1.0) < 1.0e-10 &&
		std::abs(cidnpRows.front()[1]) < 1.0e-10 &&
		std::abs(cidnpRows.front()[2]) < 1.0e-10 &&
		std::abs(cidnpRows.front()[3] - 0.5) < 1.0e-10 &&
		std::abs(polarizationRows.front()[3] - 2.5) < 1.0e-10;
}

bool test_hsgeneral_secular_h0_with_explicit_dynamic_drive_supports_so3()
{
	auto fixture = BuildHSGeneralDynamicPowderSystem();
	std::string data;
	std::string log;
	if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
		"dynamics=dynamic;calculation=timeevolution;sampling=direct;approximation=secular;"
		"hamiltonianh0list=B01,B02;powdersamplingpoints=3;powdergammapoints=2;"
		"totaltime=0.3;timestep=0.1;propagationmethod=normal;", data, &log)) return false;

	std::string header;
	std::vector<std::vector<double>> rows;
	return ParseNumericData(data, header, rows) && rows.size() == 4 &&
		log.find("orientation=theta/phi/gamma") != std::string::npos &&
		log.find("Hamiltonian approximation: high-field secular") != std::string::npos;
}

bool test_hsgeneral_pulse_timeline_matches_directspectra()
{
	auto legacyFixture = BuildHSGeneralSpectraSystem();
	auto generalFixture = BuildHSGeneralSpectraSystem();
	const std::string common =
		"integration=false;cidsp=false;spinlist=E;powdersamplingpoints=5;"
		"hamiltonianh0list=zeeman;printtimeframe=full;integrationtimeframe=full;"
		"pulsesequence=[\"cw 0\"];totaltime=1;timestep=0.1;propagationmethod=normal;";
	std::string legacyData, generalData, generalLog;
	if (!RunHSGeneralSpectraTask(legacyFixture, "StaticHS-Direct-Spectra",
		"method=timeevo;sampling=direct;approximation=secular;" + common, legacyData) ||
		!RunHSGeneralSpectraTask(generalFixture, "HSGeneral",
			"dynamics=static;calculation=timeevolution;sampling=direct;approximation=secular;" + common,
			generalData, &generalLog)) return false;

	if (!SpectraDataEqualOnSharedTimeline(
		legacyData, generalData, 2.0 * arma::datum::pi, 2.0e-10)) return false;

	auto legacyIntegrated = BuildHSGeneralSpectraSystem();
	auto generalIntegrated = BuildHSGeneralSpectraSystem();
	std::string legacyIntegratedData, generalIntegratedData;
	const std::string integratedCommon =
		"integration=true;cidsp=false;spinlist=E;powdersamplingpoints=5;"
		"hamiltonianh0list=zeeman;printtimeframe=pulse;integrationtimeframe=pulse;"
		"pulsesequence=[\"cw 0\"];totaltime=1;timestep=0.1;propagationmethod=normal;";
	if (!RunHSGeneralSpectraTask(legacyIntegrated, "StaticHS-Direct-Spectra",
		"method=timeevo;sampling=direct;approximation=secular;" + integratedCommon, legacyIntegratedData) ||
		!RunHSGeneralSpectraTask(generalIntegrated, "HSGeneral",
			"dynamics=static;calculation=timeevolution;sampling=direct;approximation=secular;" + integratedCommon,
			generalIntegratedData)) return false;

	if (!SpectraDataEqualOnSharedTimeline(
		legacyIntegratedData, generalIntegratedData, 2.0 * arma::datum::pi, 2.0e-10)) return false;
	return generalLog.find("Running unified TaskHSGeneral") != std::string::npos;
}

bool test_hsgeneral_spectra_is_standalone()
{
	return HSGeneralRejects("calculation=spectra;", "standalone task") &&
		HSGeneralRejects("calculation=resonance;", "standalone task");
}

bool test_standalone_direct_spectra_retains_trace_sampling_and_approximation()
{
	auto directSystem = BuildHSGeneralSpectraSystem();
	auto stochasticSystem = BuildHSGeneralSpectraSystem();
	auto fullSystem = BuildHSGeneralSpectraSystem();
	const std::string common =
		"method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=5;powdergammapoints=3;"
		"hamiltonianh0list=zeeman;printtimeframe=pulse;integrationtimeframe=pulse;"
		"pulsesequence=[\"cw 0\"];totaltime=1;timestep=0.1;propagationmethod=normal;";

	std::string directData, stochasticData, fullData, stochasticLog, fullLog;
	if (!RunHSGeneralSpectraTask(directSystem, "StaticHS-Direct-Spectra",
		"sampling=direct;approximation=secular;" + common, directData) ||
		!RunHSGeneralSpectraTask(stochasticSystem, "StaticHS-Direct-Spectra",
			"sampling=stochastic;montecarlosamples=4;autoseed=false;seed=7;approximation=secular;" + common,
			stochasticData, &stochasticLog) ||
		!RunHSGeneralSpectraTask(fullSystem, "StaticHS-Direct-Spectra",
			"sampling=direct;approximation=full;" + common, fullData, &fullLog))
		return false;

	return SpectraDataNumericallyEqual(directData, stochasticData, 2.0e-10) &&
		stochasticLog.find("trace sampling keeps State") != std::string::npos &&
		stochasticLog.find("H0 approximation = secular") != std::string::npos &&
		stochasticLog.find("15 total SO(3) orientations") != std::string::npos &&
		fullLog.find("H0 approximation = full") != std::string::npos &&
		!fullData.empty();
}

bool test_hsgeneral_direct_relaxation_matches_frozen_legacy_reference()
{
	auto legacyTimeFixture = BuildHSGeneralRelaxationSystem();
	auto generalTimeFixture = BuildHSGeneralRelaxationSystem();
	auto legacyYieldFixture = BuildHSGeneralRelaxationSystem();
	auto generalYieldFixture = BuildHSGeneralRelaxationSystem();
	if (!legacyTimeFixture.spinSystem || !generalTimeFixture.spinSystem ||
		!legacyYieldFixture.spinSystem || !generalYieldFixture.spinSystem) return false;

	const std::string tail =
		"timestep=0.05;propagationmethod=normal;transitionyields=true;yieldcorrections=false;";
	std::string legacyTime, generalTime, legacyYield, generalYield, generalLog;
	if (!RunHSGeneralSpectraTask(legacyTimeFixture, "StaticHS-Direct-TimeEvo",
			"totaltime=0.45;" + tail, legacyTime) ||
		!RunHSGeneralSpectraTask(generalTimeFixture, "HSGeneral",
			"dynamics=static;calculation=timeevolution;sampling=direct;totaltime=0.4;" + tail, generalTime, &generalLog) ||
		!RunHSGeneralSpectraTask(legacyYieldFixture, "StaticHS-Direct-Yields",
			"totaltime=0.45;" + tail, legacyYield) ||
		!RunHSGeneralSpectraTask(generalYieldFixture, "HSGeneral",
			"dynamics=static;calculation=yields;sampling=direct;totaltime=0.4;" + tail, generalYield))
		return false;

	return SpectraDataNumericallyEqual(legacyTime, generalTime, 3.0e-6) &&
		SpectraDataNumericallyEqual(legacyYield, generalYield, 3.0e-6) &&
		generalLog.find("exponential Hamiltonian/reaction splitting") != std::string::npos;
}

bool test_hsgeneral_dynamic_direct_relaxation_matches_frozen_legacy_reference()
{
	auto legacyTimeFixture = BuildHSGeneralDynamicRelaxationSystem();
	auto generalTimeFixture = BuildHSGeneralDynamicRelaxationSystem();
	auto legacyYieldFixture = BuildHSGeneralDynamicRelaxationSystem();
	auto generalYieldFixture = BuildHSGeneralDynamicRelaxationSystem();
	if (!legacyTimeFixture.spinSystem || !generalTimeFixture.spinSystem ||
		!legacyYieldFixture.spinSystem || !generalYieldFixture.spinSystem) return false;

	const std::string tail =
		"timestep=0.05;propagationmethod=normal;transitionyields=true;yieldcorrections=false;";
	std::string legacyTime, generalTime, legacyYield, generalYield;
	if (!RunHSGeneralSpectraTask(legacyTimeFixture, "DynamicHS-Direct-TimeEvo",
			"totaltime=0.35;" + tail, legacyTime) ||
		!RunHSGeneralSpectraTask(generalTimeFixture, "HSGeneral",
			"dynamics=dynamic;calculation=timeevolution;sampling=direct;totaltime=0.3;" + tail, generalTime) ||
		!RunHSGeneralSpectraTask(legacyYieldFixture, "DynamicHS-Direct-Yields",
			"totaltime=0.35;" + tail, legacyYield) ||
		!RunHSGeneralSpectraTask(generalYieldFixture, "HSGeneral",
			"dynamics=dynamic;calculation=yields;sampling=direct;totaltime=0.3;" + tail, generalYield))
		return false;

	return SpectraDataNumericallyEqual(legacyTime, generalTime, 4.0e-6) &&
		SpectraDataNumericallyEqual(legacyYield, generalYield, 4.0e-6);
}

bool test_hsgeneral_yield_correction_matches_finite_decay_law()
{
	auto generalFixture = BuildHSGeneralPowderYieldSystem();
	const std::string common =
		"totaltime=0.5;timestep=0.05;propagationmethod=normal;transitionyields=true;yieldcorrections=true;";
	std::string generalData, generalLog;
	if (!RunHSGeneralSpectraTask(generalFixture, "HSGeneral",
			"dynamics=static;calculation=yields;sampling=direct;" + common, generalData, &generalLog))
		return false;
	std::string header;
	std::vector<std::vector<double>> rows;
	if (!ParseNumericData(generalData, header, rows) || rows.size() != 1 || rows.front().size() != 2)
		return false;
	const double rate = 0.02;
	const double total = 0.5;
	const double dt = 0.05;
	double trapezoid = 0.0;
	for (int step = 1; step <= 10; ++step)
	{
		const double left = static_cast<double>(step - 1) * dt;
		const double right = static_cast<double>(step) * dt;
		trapezoid += 0.5 * dt * rate *
			(std::exp(-rate * left) + std::exp(-rate * right));
	}
	const double expected = trapezoid / (1.0 - std::exp(-rate * total));
	return std::abs(rows.front()[1] - expected) < 1.0e-10 &&
		generalLog.find("Applied the legacy finite-time yield correction") != std::string::npos;
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

bool test_hsgeneral_stochastic_preparation_remains_factorized()
{
	auto e1 = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;");
	auto e2 = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1/2;");
	auto nucleus = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1;");
	auto singlet = std::make_shared<SpinAPI::State>("Singlet",
		"spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("System");
	system->Add(e1); system->Add(e2); system->Add(nucleus); system->Add(singlet);
	system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties", "initialstate=Singlet;"));
	if (!singlet->ParseFromSystem(*system))
		return false;

	RunSection::General::HS::HSExecutionPlan plan;
	plan.sampling = RunSection::General::HS::Sampling::Stochastic;
	plan.monteCarloSamples = 5;
	plan.samplingMethod = "suz";
	plan.autoSeed = false;
	plan.seed = 17;

	SpinAPI::SpinSpace space(system);
	space.UseSuperoperatorSpace(false);
	std::mt19937 generator(17);
	std::ostringstream log;
	std::string error;
	RunSection::General::HS::HSPreparedState prepared;
	if (!RunSection::General::HS::HSStatePreparation::Prepare(
		plan, system, space, prepared, generator, log, error))
		return false;

	return prepared.stochastic && prepared.density.is_empty() &&
		prepared.factors.n_rows == 12 && prepared.factors.n_cols == 5 &&
		prepared.traceSamples.sampledSubspaceDimension == 3;
}

bool test_hsgeneral_stochastic_powder_invariant_support_avoids_dense_rotation_cache()
{
	auto e1 = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;");
	auto e2 = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1/2;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("System");
	system->Add(e1); system->Add(e2);
	for (int i = 0; i < 6; ++i)
		system->Add(std::make_shared<SpinAPI::Spin>("N" + std::to_string(i), "type=nucleus;spin=1/2;"));

	auto singlet = std::make_shared<SpinAPI::State>("Singlet",
		"spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
	auto sink = std::make_shared<SpinAPI::Transition>(
		"Product", "type=sink;sourcestate=Singlet;rate=0.01;", system);
	system->Add(singlet); system->Add(sink);
	system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
		"properties", "initialstate=Singlet;frame=molecular;initialstatecoherences=keep;"));
	if (!singlet->ParseFromSystem(*system) || !system->ValidateTransitions({system}).empty())
		return false;

	RunSection::General::HS::HSExecutionPlan plan;
	plan.sampling = RunSection::General::HS::Sampling::Stochastic;
	plan.calculation = RunSection::General::HS::Calculation::Yields;
	plan.orientation = RunSection::General::HS::OrientationMode::Powder2D;
	plan.transitionYields = true;
	plan.monteCarloSamples = 4;
	plan.samplingMethod = "suz";
	plan.autoSeed = false;
	plan.seed = 23;

	SpinAPI::SpinSpace space(system);
	space.UseSuperoperatorSpace(false);
	std::mt19937 generator(23);
	std::ostringstream log;
	std::string error;
	RunSection::General::HS::HSPreparedState prepared;
	if (!RunSection::General::HS::HSStatePreparation::Prepare(
		plan, system, space, prepared, generator, log, error))
		return false;
	if (!prepared.stochastic || !prepared.rotationInvariant || prepared.hasRotationCache ||
		!prepared.rotationCache.Jx.is_empty() || !prepared.rotationCache.Jy.is_empty() ||
		!prepared.rotationCache.Jz.is_empty())
		return false;

	RunSection::General::HS::HSObservableCollector collector;
	if (!collector.Prepare(plan, system, space, log, error) || collector.Observables().size() != 1)
		return false;
	const auto &observable = collector.Observables().front();
	if (observable.rotateStateOrSource || !observable.stateRotationCache.Jx.is_empty() ||
		!observable.stateRotationCache.Jy.is_empty() || !observable.stateRotationCache.Jz.is_empty())
		return false;

	SpinAPI::HilbertReactionOperatorCache reactionCache;
	if (!space.CreateHilbertReactionOperatorCache(sink, reactionCache, true) ||
		!reactionCache.rotationInvariant || reactionCache.hasSourceRotation ||
		!reactionCache.sourceRotation.Jx.is_empty() || !reactionCache.sourceRotation.Jy.is_empty() ||
		!reactionCache.sourceRotation.Jz.is_empty())
		return false;

	// The invariant source must still produce the same sparse reaction operator
	// for a non-trivial crystallite rotation without constructing a dense cache.
	arma::mat rotation;
	if (!SpinAPI::CreateZYZRotationMatrix(0.4, 0.7, 1.1, rotation))
		return false;
	arma::sp_cx_mat rotatedReaction;
	return space.ReactionOperatorHilbertRotated(reactionCache, rotation, rotatedReaction) &&
		rotatedReaction.n_rows == 256 && rotatedReaction.n_cols == 256;
}

bool test_hsgeneral_orientation_sampler_builds_so3_grid()
{
	MSDParser::ObjectParser parser("general",
		"type=HSGeneral;dynamics=dynamic;calculation=yields;sampling=stochastic;"
		"powdersamplingpoints=7;powdergammapoints=5;hamiltonianh0list=H0;");
	RunSection::General::HS::HSExecutionPlan plan;
	std::string error;
	if (!RunSection::General::HS::ResolveExecutionPlan(parser, plan, error)) return false;
	std::vector<RunSection::General::HS::HSOrientation> orientations;
	std::ostringstream log;
	if (!RunSection::General::HS::HSOrientationSampler::Build(plan, orientations, log, error)) return false;
	if (orientations.size() != 35) return false;
	double weight = 0.0;
	for (const auto &orientation : orientations)
	{
		if (!orientation.frameToLab.is_finite() || std::abs(arma::det(orientation.frameToLab) - 1.0) > 1.0e-12) return false;
		weight += orientation.weight;
	}
	// Gamma-resolved uniform sampling defaults to a complete full-sphere SO(3)
	// integral. An explicitly requested upper hemisphere remains available as a
	// deliberate symmetry reduction.
	if (plan.powderDomain != SpinAPI::PowderGridDomain::FullSphere ||
		!plan.powderDomainAutoExpanded ||
		std::abs(weight - 1.0) >= 1.0e-12 ||
		log.str().find("35 total SO(3) orientations") == std::string::npos ||
		log.str().find("selected automatically") == std::string::npos)
	{
		return false;
	}

	MSDParser::ObjectParser reducedParser("general",
		"type=HSGeneral;calculation=yields;powdersamplingpoints=7;"
		"powdergammapoints=5;powderdomain=upper;hamiltonianh0list=H0;");
	RunSection::General::HS::HSExecutionPlan reducedPlan;
	if (!RunSection::General::HS::ResolveExecutionPlan(reducedParser, reducedPlan, error) ||
		reducedPlan.powderDomainAutoExpanded ||
		reducedPlan.powderDomain != SpinAPI::PowderGridDomain::UpperHemisphere)
	{
		return false;
	}
	std::vector<RunSection::General::HS::HSOrientation> reduced;
	std::ostringstream reducedLog;
	if (!RunSection::General::HS::HSOrientationSampler::Build(
		reducedPlan, reduced, reducedLog, error)) return false;
	double reducedWeight = 0.0;
	for (const auto &orientation : reduced) reducedWeight += orientation.weight;
	return std::abs(reducedWeight - 1.0) < 1.0e-12;
}

bool test_hsgeneral_powdergrid_api_selection()
{
	using namespace RunSection::General::HS;
	auto build = [](const std::string &properties, HSExecutionPlan &plan,
		std::vector<HSOrientation> &orientations) -> bool
	{
		MSDParser::ObjectParser parser("general", "type=HSGeneral;calculation=timeevolution;"
			"hamiltonianh0list=H0;" + properties);
		std::string error;
		if (!ResolveExecutionPlan(parser, plan, error)) return false;
		std::ostringstream log;
		return HSOrientationSampler::Build(plan, orientations, log, error);
	};

	HSExecutionPlan uniformPlan;
	std::vector<HSOrientation> uniform;
	if (!build("powdergrid=uniform;powdersamplingpoints=5;powderdomain=full;",
		uniformPlan, uniform)) return false;
	if (uniformPlan.powderGridType != SpinAPI::PowderGridType::Uniform ||
		uniformPlan.powderDomain != SpinAPI::PowderGridDomain::FullSphere ||
		uniform.size() != 5) return false;
	SpinAPI::PowderGrid uniformReference;
	if (!SpinAPI::CreateUniformPowderGrid(5, SpinAPI::PowderGridDomain::FullSphere,
		uniformReference)) return false;
	for (size_t i = 0; i < uniform.size(); ++i)
	{
		if (std::abs(uniform[i].beta - uniformReference[i].theta) > 1.0e-13 ||
			std::abs(uniform[i].gamma - uniformReference[i].phi) > 1.0e-13 ||
			std::abs(uniform[i].weight - uniformReference[i].weight / (4.0 * arma::datum::pi)) > 1.0e-13)
			return false;
	}

	HSExecutionPlan legacyPlan;
	std::vector<HSOrientation> legacyUniform;
	if (!build("powdergridtype=fibonacci;powdersamplingpoints=5;powderfullsphere=true;",
		legacyPlan, legacyUniform)) return false;
	if (legacyPlan.powderGridType != SpinAPI::PowderGridType::Uniform ||
		legacyUniform.size() != uniform.size()) return false;
	for (size_t i = 0; i < uniform.size(); ++i)
		if (arma::norm(uniform[i].frameToLab - legacyUniform[i].frameToLab, "fro") > 1.0e-13 ||
			std::abs(uniform[i].weight - legacyUniform[i].weight) > 1.0e-13)
			return false;

	HSExecutionPlan sophePlan;
	std::vector<HSOrientation> sophe;
	if (!build("powdergrid=sophe;powdergridsize=4;powdersymmetry=ci;powdergammapoints=3;",
		sophePlan, sophe)) return false;
	SpinAPI::PowderGrid sopheReference;
	if (!SpinAPI::CreateSophePowderGrid(4, "ci", sopheReference) ||
		sophe.size() != 3 * sopheReference.size()) return false;
	double sopheWeight = 0.0;
	for (const auto &orientation : sophe) sopheWeight += orientation.weight;
	double sopheReferenceWeight = 0.0;
	for (const auto &orientation : sopheReference) sopheReferenceWeight += orientation.weight;
	if (std::abs(sopheWeight - 1.0) > 1.0e-12 || !(sopheReferenceWeight > 0.0)) return false;

	HSExecutionPlan octantPlan;
	std::vector<HSOrientation> octant;
	if (!build("powdergrid=octant;powdersamplingpoints=3;", octantPlan, octant) ||
		octantPlan.powderGridType != SpinAPI::PowderGridType::Octant || octant.size() != 9)
		return false;
	double octantWeight = 0.0;
	for (const auto &orientation : octant) octantWeight += orientation.weight;
	if (std::abs(octantWeight - 1.0) > 1.0e-12) return false;

	return HSGeneralRejects(
		"calculation=timeevolution;hamiltonianh0list=H0;powdergrid=sophe;powdersamplingpoints=5;",
		"use powdergridsize") &&
		HSGeneralRejects(
			"calculation=timeevolution;hamiltonianh0list=H0;powdergrid=octant;powdersamplingpoints=3;powderdomain=full;",
			"fixed symmetry-reduced octant domain") &&
		HSGeneralRejects(
			"calculation=timeevolution;hamiltonianh0list=H0;powdergrid=unknown;powdersamplingpoints=3;",
			"powdergrid must be uniform, sophe, or octant");
}

bool test_hsgeneral_powder_requires_explicit_h0()
{
	return HSGeneralRejects(
		"dynamics=static;calculation=timeevolution;sampling=direct;powdersamplingpoints=3;",
		"requires hamiltonianh0list");
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

bool test_hsgeneral_static_timeinf_yield_has_correct_closed_form_limit()
{
	auto fixture = BuildHSGeneralPowderYieldSystem();
	std::string data;
	std::string log;
	if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
			"dynamics=static;calculation=yields;sampling=direct;method=timeinf;"
			"transitionyields=true;reactionoperators=haberkorn;", data, &log))
	{
		return false;
	}
	std::string header;
	std::vector<std::vector<double>> rows;
	if (!ParseNumericData(data, header, rows) || rows.size() != 1 || rows.front().size() != 2)
		return false;
	// The fixture starts in the singlet and has a single singlet sink. Its
	// infinite-time product yield is exactly one; the decoupled nucleus does
	// not alter that normalization.
	return header.find("Time") == std::string::npos &&
		std::abs(rows.front()[1] - 1.0) < 1.0e-10 &&
		log.find("yield-mode=timeinf") != std::string::npos;
}

bool test_hsgeneral_static_powder_timeevolution_and_yields()
{
	auto directTimeFixture = BuildHSGeneralPowderYieldSystem();
	auto stochasticTimeFixture = BuildHSGeneralPowderYieldSystem();
	auto directYieldFixture = BuildHSGeneralPowderYieldSystem();
	auto stochasticYieldFixture = BuildHSGeneralPowderYieldSystem();
	const std::string common =
		"dynamics=static;powdersamplingpoints=5;hamiltonianh0list=B0;"
		"totaltime=0.2;timestep=0.1;propagationmethod=normal;method=timeevo;";
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
		timeLog.find("orientation=theta/phi") != std::string::npos &&
		timeLog.find("Hamiltonian approximation: full") != std::string::npos &&
		yieldLog.find("yield-mode=finite") != std::string::npos &&
		yieldLog.find("Hamiltonian approximation: high-field secular") != std::string::npos &&
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
		timeRows.size() == 4 && yieldRows.size() == 1 &&
		directTime.find("System.Singlet") != std::string::npos &&
		directTime.find("System.E1Up") != std::string::npos &&
		directYield.find("System.sink.yield") != std::string::npos &&
		directTimeLog.find("Molecular-frame State observables follow each General orientation") != std::string::npos &&
		yieldLog.find("yield-mode=finite") != std::string::npos &&
		yieldLog.find("Writing finite-time integrated HS observables") != std::string::npos;
}

bool test_hsgeneral_dynamic_powder_gamma_matches_external_decomposition()
{
	return DynamicPowderGammaMatchesExplicitOrientations();
}

bool test_hsgeneral_powder_rotates_orientation_dependent_reaction_state()
{
	const int pointCount = 3;
	const std::string common =
		"dynamics=static;calculation=yields;sampling=direct;approximation=full;"
		"hamiltonianh0list=B01,B02;totaltime=0.3;timestep=0.1;propagationmethod=normal;";

	auto internalFixture = BuildHSGeneralDynamicPowderSystem();
	auto orientedSink = std::make_shared<SpinAPI::Transition>(
		"oriented_sink", "type=sink;sourcestate=E1Up;sourceframe=molecular;rate=0.01;", internalFixture.spinSystem);
	internalFixture.spinSystem->Add(orientedSink);
	std::string internal;
	if (!RunHSGeneralSpectraTask(internalFixture, "HSGeneral",
		common + "powdersamplingpoints=" + std::to_string(pointCount) + ";", internal)) return false;

	std::string internalHeader;
	std::vector<std::vector<double>> internalRows;
	if (!ParseNumericData(internal, internalHeader, internalRows) || internalRows.size() != 1) return false;

	SpinAPI::PowderGrid grid;
	if (!SpinAPI::CreateUniformPowderGrid(pointCount, SpinAPI::PowderGridDomain::UpperHemisphere, grid)) return false;
	std::vector<double> externalSum(internalRows.front().size(), 0.0);
	for (const auto &orientation : grid)
	{
		auto fixture = BuildHSGeneralDynamicPowderSystem();
		auto sink = std::make_shared<SpinAPI::Transition>(
			"oriented_sink", "type=sink;sourcestate=E1Up;sourceframe=molecular;rate=0.01;", fixture.spinSystem);
		fixture.spinSystem->Add(sink);
		std::ostringstream props;
		props << std::setprecision(17) << common
			<< "powderorientation=0 " << orientation.theta << " " << orientation.phi << " "
			<< orientation.weight / (2.0 * arma::datum::pi) << ";";
		std::string external, header;
		std::vector<std::vector<double>> rows;
		if (!RunHSGeneralSpectraTask(fixture, "HSGeneral", props.str(), external) ||
			!ParseNumericData(external, header, rows) || header != internalHeader || rows.size() != 1) return false;
		externalSum[0] = rows.front()[0];
		for (size_t col = 1; col < rows.front().size(); ++col) externalSum[col] += rows.front()[col];
	}

	for (size_t col = 0; col < internalRows.front().size(); ++col)
	{
		const double scale = std::max({1.0, std::abs(internalRows.front()[col]), std::abs(externalSum[col])});
		if (std::abs(internalRows.front()[col] - externalSum[col]) > 2.0e-10 * scale) return false;
	}
	return internalHeader.find("oriented_sink.yield") != std::string::npos;
}


bool test_hsgeneral_state_preparation_component_owns_orientation()
{
	auto fixture = BuildHSGeneralPowderYieldSystem();
	if (fixture.spinSystem == nullptr || fixture.initialState == nullptr ||
		!fixture.initialState->ParseFromSystem(*fixture.spinSystem))
		return false;

	MSDParser::ObjectParser parser("general",
		"type=HSGeneral;dynamics=static;calculation=timeevolution;sampling=direct;"
		"approximation=full;hamiltonianh0list=B0;");
	RunSection::General::HS::HSExecutionPlan plan;
	std::string error;
	if (!RunSection::General::HS::ResolveExecutionPlan(parser, plan, error))
		return false;

	SpinAPI::SpinSpace space(fixture.spinSystem);
	space.UseSuperoperatorSpace(false);
	std::mt19937 generator(7);
	std::ostringstream log;
	RunSection::General::HS::HSPreparedState reference;
	if (!RunSection::General::HS::HSStatePreparation::Prepare(
		plan, fixture.spinSystem, space, reference, generator, log, error))
		return false;

	RunSection::General::HS::HSOrientation orientation;
	orientation.alpha = 0.31;
	orientation.beta = 0.77;
	orientation.gamma = 1.13;
	if (!SpinAPI::CreateZYZRotationMatrix(orientation.alpha, orientation.beta,
		orientation.gamma, orientation.frameToLab))
		return false;

	RunSection::General::HS::HSOrientedState oriented;
	if (!RunSection::General::HS::HSStatePreparation::PrepareForOrientation(
		plan, space, reference, orientation, oriented, error))
		return false;

	return std::abs(std::real(arma::trace(oriented.density)) - 1.0) < 1.0e-12 &&
		arma::norm(oriented.factors * oriented.factors.t() - oriented.density, "fro") < 1.0e-10;
}

bool test_hsgeneral_eigen_thermal_uses_selected_oriented_hamiltonian()
{
	auto electron = std::make_shared<SpinAPI::Spin>(
		"E", "type=electron;spin=1/2;tensor=matrix(\"2.0 0 0;0 2.2 0;0 0 2.6\");");
	auto thermal = std::make_shared<SpinAPI::Interaction>(
		"thermal", "type=zeeman;spins=E;field=0 0 3.4;ignoretensors=false;"
		"commonprefactor=true;prefactor=1;");
	auto excluded = std::make_shared<SpinAPI::Interaction>(
		"excluded", "type=singlespin;spins=E;field=500 0 0;commonprefactor=true;prefactor=1;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("ThermalHS");
	system->Add(electron);
	system->Add(thermal);
	system->Add(excluded);
	if (!system->ValidateInteractions().empty()) return false;
	system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties",
		"initialstate=Thermal;initialstateframe=eigen;temperature=250;"
		"thermalhamiltonian=thermal;"));

	MSDParser::ObjectParser parser("general",
		"type=HSGeneral;dynamics=static;calculation=timeevolution;sampling=direct;"
		"approximation=secular;hamiltonianh0list=thermal,excluded;"
		"powderorientation=0.37 0.82 -0.21;totaltime=0;timestep=1;");
	RunSection::General::HS::HSExecutionPlan plan;
	std::string error;
	if (!RunSection::General::HS::ResolveExecutionPlan(parser, plan, error)) return false;

	SpinAPI::SpinSpace space(system);
	space.UseSuperoperatorSpace(false);
	std::mt19937 generator(11);
	std::ostringstream log;
	RunSection::General::HS::HSPreparedState reference;
	if (!RunSection::General::HS::HSStatePreparation::Prepare(
		plan, system, space, reference, generator, log, error) ||
		!reference.orientationSpecificThermal)
		return false;

	std::vector<RunSection::General::HS::HSOrientation> orientations;
	if (!RunSection::General::HS::HSOrientationSampler::Build(
		plan, orientations, log, error) || orientations.size() != 1)
		return false;
	RunSection::General::HS::HSOrientedState oriented;
	if (!RunSection::General::HS::HSStatePreparation::PrepareForOrientation(
		plan, space, reference, orientations.front(), oriented, error))
		return false;

	arma::sp_cx_mat selectedHamiltonian;
	arma::cx_mat expected;
	if (!space.BaseHamiltonianRotatedZYZ({"thermal"},
		orientations.front().frameToLab, selectedHamiltonian) ||
		!space.ThermalStateFromHamiltonian(
			arma::cx_mat(selectedHamiltonian), 250.0, expected))
		return false;
	return arma::norm(oriented.density - expected, "fro") < 1.0e-12 &&
		arma::norm(oriented.factors * oriented.factors.t() - expected, "fro") < 1.0e-10;
}

bool test_hsgeneral_spinapi_reaction_rotation_component()
{
	auto fixture = BuildHSGeneralDynamicPowderSystem();
	if (fixture.spinSystem == nullptr || fixture.initialState == nullptr ||
		!fixture.initialState->ParseFromSystem(*fixture.spinSystem))
		return false;

	auto sink = std::make_shared<SpinAPI::Transition>(
		"oriented_sink", "type=sink;sourcestate=E1Up;rate=0.04;", fixture.spinSystem);
	fixture.spinSystem->Add(sink);
	if (!fixture.spinSystem->ValidateTransitions({fixture.spinSystem}).empty())
		return false;

	SpinAPI::SpinSpace space(fixture.spinSystem);
	space.UseSuperoperatorSpace(false);
	SpinAPI::HilbertReactionOperatorCache cache;
	if (!space.CreateHilbertReactionOperatorCache(sink, cache))
		return false;

	arma::mat rotation;
	if (!SpinAPI::CreateZYZRotationMatrix(0.4, 0.8, 1.2, rotation))
		return false;
	arma::sp_cx_mat reaction;
	if (!space.ReactionOperatorHilbertRotated(cache, rotation, reaction))
		return false;

	arma::cx_mat rotatedProjector;
	if (!space.RotateState(arma::cx_mat(cache.sourceProjector), rotation, cache.sourceRotation, rotatedProjector))
		return false;
	return arma::norm(arma::cx_mat(reaction) - 0.5 * sink->Rate() * rotatedProjector, "fro") < 1.0e-12;
}

bool test_hsgeneral_propagator_owns_timeinf_solve()
{
	auto fixture = BuildHSGeneralSpectraSystem();
	if (fixture.spinSystem == nullptr)
		return false;

	RunSection::General::HS::HSExecutionPlan plan;
	SpinAPI::SpinSpace space(fixture.spinSystem);
	space.UseSuperoperatorSpace(false);
	RunSection::General::HS::HSReactionRelaxation reaction(plan, fixture.spinSystem, space);
	std::string error;
	if (!reaction.Validate(error))
		return false;
	RunSection::General::HS::HSPropagator propagator(plan, space);
	RunSection::General::HS::HSRelaxationContext context;

	const arma::uword dim = space.HilbertSpaceDimensions();
	arma::sp_cx_mat hamiltonian(dim, dim);
	arma::sp_cx_mat sink = 0.1 * arma::speye<arma::sp_cx_mat>(dim, dim);
	arma::cx_mat rho0(dim, dim, arma::fill::zeros);
	rho0(0, 0) = 1.0;
	arma::cx_mat integrated;
	if (!propagator.SolveTimeInfinity(hamiltonian, sink, rho0,
		reaction, context, integrated, error))
		return false;
	return arma::norm(integrated - 5.0 * rho0, "fro") < 1.0e-11;
}

bool test_hsgeneral_observable_component_owns_integration()
{
	RunSection::General::HS::HSObservableCollector collector;
	std::string error;
	arma::mat values(3, 1, arma::fill::ones);
	arma::rowvec finite;
	const std::vector<double> finiteTimes{0.0, 0.5, 1.25};
	if (!collector.IntegrateFiniteTime(values, finiteTimes, finite, error) ||
		finite.n_elem != 1 || std::abs(finite(0) - 1.25) > 1.0e-12)
		return false;

	const std::vector<double> times{0.0, 1.0, 2.0};
	arma::mat timeline;
	if (!collector.IntegrateTimeline(times, values, true, timeline, error))
		return false;
	return timeline.n_rows == 3 && timeline.n_cols == 1 &&
		std::abs(timeline(0, 0) - 1.0) < 1.0e-12 &&
		std::abs(timeline(1, 0) - 1.0) < 1.0e-12 &&
		std::abs(timeline(2, 0) - 2.0) < 1.0e-12;
}

bool test_hsgeneral_time_grid_includes_exact_endpoint()
{
	auto fixture = BuildHSGeneralSpectraSystem();
	std::string data;
	if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
		"dynamics=static;calculation=timeevolution;sampling=direct;"
		"observables=states;totaltime=0.25;timestep=0.2;", data))
		return false;
	std::string header;
	std::vector<std::vector<double>> rows;
	if (!ParseNumericData(data, header, rows) || rows.size() != 3)
		return false;
	return rows[0].size() >= 2 && rows[1].size() >= 2 && rows[2].size() >= 2 &&
		std::abs(rows[0][1]) < 1.0e-14 &&
		std::abs(rows[1][1] - 0.2) < 1.0e-14 &&
		std::abs(rows[2][1] - 0.25) < 1.0e-14;
}

bool test_hsgeneral_parser_uses_canonical_sampling_keywords_and_warns_krylovtol()
{
	if (!HSGeneralRejects(
		"calculation=timeevolution;sampling=stochastic;samplingmethod=unknown;",
		"samplingmethod must be suz or coherent"))
		return false;
	auto fixture = BuildHSGeneralSpectraSystem();
	std::string data;
	std::string log;
	if (!RunHSGeneralSpectraTask(fixture, "HSGeneral",
		"dynamics=static;calculation=timeevolution;sampling=direct;"
		"propagationmethod=krylov;krylovtol=1e-8;krylovsize=8;totaltime=0;timestep=1;",
		data, &log))
		return false;
	return log.find("parsed for compatibility but is not used") != std::string::npos;
}

bool test_hsgeneral_rejects_unsupported_physics()
{
	return HSGeneralRejects(
			   "calculation=spectra;", "standalone task") &&
		HSGeneralRejects(
			"calculation=timeevolution;approximation=rwa;",
			"not a static Hamiltonian secular approximation") &&
		HSGeneralRejects(
			"calculation=timeevolution;powderaveraging=true;", "requires a generated powder grid") &&
		HSGeneralRejects(
			"dynamics=dynamic;calculation=yields;method=timeinf;", "dynamic yields require finite-time") &&
		HSGeneralRejects(
			"calculation=yields;relaxationmodel=nakajimazwanzig;", "Nakajima-Zwanzig") &&
		HSGeneralRejects(
			"calculation=yields;approximation=secular;", "requires hamiltonianh0list") &&
		HSGeneralRejects(
			"calculation=yields;reactionoperators=lindblad;", "superspace reaction-superoperator") &&
		HSGeneralRejects(
			"calculation=yields;reactionoperators=unknown;", "must be haberkorn") &&
		HSGeneralRejects(
			"calculation=yields;powdergammapoints=4;", "requires a generated theta/phi powder grid");
}

void AddHSGeneralTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("HSGeneral matches LuDM DirectSpectra relaxation/RWA regression", test_hsgeneral_matches_ludm_directspectra_relaxation_rwa_regression));
	_cases.push_back(test_case("HSGeneral matches CISS DirectSpectra timeinf polarization regression", test_hsgeneral_matches_ciss_directspectra_timeinf_polarization_regression));
	_cases.push_back(test_case("HSGeneral dispatches proven task modes", test_hsgeneral_dispatches_proven_modes));
	_cases.push_back(test_case("HSGeneral keeps eight core legacy tasks independently constructible", test_hsgeneral_keeps_eight_core_legacy_tasks_independent));
	_cases.push_back(test_case("HSGeneral static direct modes match frozen legacy reference", test_hsgeneral_static_direct_matches_frozen_legacy_reference));
	_cases.push_back(test_case("HSGeneral defaults are explicit and compatible", test_hsgeneral_defaults_are_explicit_and_compatible));
	_cases.push_back(test_case("HSGeneral explicit observable modes are unambiguous", test_hsgeneral_explicit_observable_modes_are_unambiguous));
	_cases.push_back(test_case("HSGeneral pulse preparation rotates polarization", test_hsgeneral_pulse_preparation_rotates_polarization));
	_cases.push_back(test_case("HSGeneral finite pulse includes partial final interval", test_hsgeneral_finite_pulse_includes_partial_final_interval));
	_cases.push_back(test_case("HSGeneral dynamic factor RK4 uses time stages", test_hsgeneral_dynamic_factor_rk4_uses_stage_generators));
	_cases.push_back(test_case("HSGeneral pulse timeline matches DirectSpectra", test_hsgeneral_pulse_timeline_matches_directspectra));
	_cases.push_back(test_case("HSGeneral quantum yields, CIDNP and timeinf polarization are explicit", test_hsgeneral_quantum_yield_cidnp_and_timeinf_polarization));
	_cases.push_back(test_case("HSGeneral secular H0 with explicit dynamic drive supports SO3", test_hsgeneral_secular_h0_with_explicit_dynamic_drive_supports_so3));
	_cases.push_back(test_case("HSGeneral keeps spectroscopy standalone", test_hsgeneral_spectra_is_standalone));
	_cases.push_back(test_case("Standalone DirectSpectra retains trace sampling and approximation selection", test_standalone_direct_spectra_retains_trace_sampling_and_approximation));
	_cases.push_back(test_case("HSGeneral direct relaxation matches frozen legacy reference", test_hsgeneral_direct_relaxation_matches_frozen_legacy_reference));
	_cases.push_back(test_case("HSGeneral dynamic direct relaxation matches frozen legacy reference", test_hsgeneral_dynamic_direct_relaxation_matches_frozen_legacy_reference));
	_cases.push_back(test_case("HSGeneral yield correction matches finite decay law", test_hsgeneral_yield_correction_matches_finite_decay_law));
	_cases.push_back(test_case("HSGeneral stochastic mode uses state-aware sampling", test_hsgeneral_stochastic_uses_state_aware_sampling));
	_cases.push_back(test_case("HSGeneral stochastic state preparation remains factorized", test_hsgeneral_stochastic_preparation_remains_factorized));
	_cases.push_back(test_case("HSGeneral stochastic powder invariant support avoids dense rotation caches", test_hsgeneral_stochastic_powder_invariant_support_avoids_dense_rotation_cache));
	_cases.push_back(test_case("HSGeneral SO3 orientation sampler is explicit", test_hsgeneral_orientation_sampler_builds_so3_grid));
	_cases.push_back(test_case("HSGeneral powdergrid selector uses the shared SpinAPI grid API", test_hsgeneral_powdergrid_api_selection));
	_cases.push_back(test_case("HSGeneral powder propagation requires explicit H0", test_hsgeneral_powder_requires_explicit_h0));
	_cases.push_back(test_case("HSGeneral direct modes accept higher spin without nuclei", test_hsgeneral_direct_modes_accept_higher_spin_without_nuclei));
	_cases.push_back(test_case("HSGeneral static timeinf yield has the correct closed-form limit", test_hsgeneral_static_timeinf_yield_has_correct_closed_form_limit));
	_cases.push_back(test_case("HSGeneral static powder supports time evolution and yields", test_hsgeneral_static_powder_timeevolution_and_yields));
	_cases.push_back(test_case("HSGeneral dynamic powder matches external decomposition and sampling", test_hsgeneral_dynamic_powder_internal_external_and_sampling));
	_cases.push_back(test_case("HSGeneral dynamic powder gamma matches explicit SO(3) decomposition", test_hsgeneral_dynamic_powder_gamma_matches_external_decomposition));
	_cases.push_back(test_case("HSGeneral powder rotates orientation-dependent reaction states", test_hsgeneral_powder_rotates_orientation_dependent_reaction_state));
	_cases.push_back(test_case("HSGeneral state preparation owns orientation policy", test_hsgeneral_state_preparation_component_owns_orientation));
	_cases.push_back(test_case("HSGeneral eigen Thermal uses selected oriented Hamiltonian", test_hsgeneral_eigen_thermal_uses_selected_oriented_hamiltonian));
	_cases.push_back(test_case("SpinAPI owns rotated Hilbert reaction operators", test_hsgeneral_spinapi_reaction_rotation_component));
	_cases.push_back(test_case("HSGeneral propagator owns the timeinf solve", test_hsgeneral_propagator_owns_timeinf_solve));
	_cases.push_back(test_case("HSGeneral observable collector owns integration", test_hsgeneral_observable_component_owns_integration));
	_cases.push_back(test_case("HSGeneral time grid includes exact endpoint", test_hsgeneral_time_grid_includes_exact_endpoint));
	_cases.push_back(test_case("HSGeneral parser uses canonical sampling keywords and warns on krylovtol", test_hsgeneral_parser_uses_canonical_sampling_keywords_and_warns_krylovtol));
	_cases.push_back(test_case("HSGeneral rejects unsupported physics", test_hsgeneral_rejects_unsupported_physics));
}
