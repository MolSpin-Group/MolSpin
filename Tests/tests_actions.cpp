//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Unit test functions for the Action classes and ActionTargets.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////
#include "ActionTarget.h"
#include "ActionAddScalar.h"
#include "ActionMultiplyScalar.h"
#include "ActionAddVector.h"
#include "ActionScaleVector.h"
#include "ActionFibonacciSphere.h"
#include "ActionRotateVector.h"
#include "ActionLogSpace.h"
#include "Interaction.h"
#include "Pulse.h"
#include "PulseSequence.h"
#include "RunSection.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Transition.h"
#include "TransferChannel.h"
#include "HSExecutionPlan.h"
#include "HSOrientationSampler.h"
#include "HSHamiltonianBuilder.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
//////////////////////////////////////////////////////////////////////////////
// Tests the ActionTarget alias ActionScalar
// First we define a check-function
bool check_for_test_actiontargets_scalar(const double &_d) { return (_d < 100.0); }
// And then we define the test itself
bool test_actiontargets_scalar()
{
	// Setup objects for the test
	double d = 42.0;
	RunSection::ActionScalar as = RunSection::ActionScalar(d, &check_for_test_actiontargets_scalar);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_double(d, as.Get());
	isCorrect &= as.Set(10.01);
	isCorrect &= equal_double(d, 10.01);
	isCorrect &= equal_double(d, as.Get());
	isCorrect &= !as.Set(100.01); // Check function is false for values >100.0
	isCorrect &= equal_double(d, 10.01);
	isCorrect &= equal_double(d, as.Get());

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ActionTarget alias ActionVector
// First we define a check-function
bool check_for_test_actiontargets_vector(const arma::vec &_v) { return (!_v.has_nan() && !_v.has_inf() && _v.n_elem == 3); }
// And then we define the test itself
bool test_actiontargets_vector()
{
	// Setup objects for the test
	arma::vec v("1 0 0");
	RunSection::ActionVector av = RunSection::ActionVector(v, &check_for_test_actiontargets_vector);

	bool isCorrect = true;
	arma::vec testvec1("0 42 120");
	arma::vec testvec2("1 1");

	// Perform the test
	isCorrect &= equal_vec(v, av.Get());
	isCorrect &= av.Set(testvec1);
	isCorrect &= equal_vec(v, testvec1);
	isCorrect &= equal_vec(v, av.Get());
	isCorrect &= !av.Set(testvec2); // Check function is false for number of elements != 3
	isCorrect &= equal_vec(v, testvec1);
	isCorrect &= equal_vec(v, av.Get());

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the AddScalar Action
bool test_action_addscalar()
{
	// Setup objects for the test
	double value1 = 42.0;
	double value2 = 62.0;
	double value3 = 82.0;
	double value4 = 102.0;
	double d = value1;
	RunSection::ActionScalar as = RunSection::ActionScalar(d, nullptr); // The ActionScalar (without check function)
	std::map<std::string, RunSection::ActionScalar> asMap;				// ActionScalar map
	std::map<std::string, RunSection::ActionVector> avMap;				// ActionVector map
	asMap.insert(RunSection::NamedActionScalar("testscalar", as));		// Add ActionScalar to the map with the name "testscalar"
	std::string actionname = "test";
	std::string actioncontents = "scalar=testscalar;value=20;";
	MSDParser::ObjectParser parser(actionname, actioncontents);
	RunSection::ActionAddScalar action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_double(d, value1);
	action.Step(2);
	isCorrect &= equal_double(as.Get(), value2);
	action_ptr->Step(3);
	isCorrect &= equal_double(d, value3);
	action.Step(4);
	isCorrect &= equal_double(as.Get(), value4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the MultiplyScalar Action
bool test_action_multiplyscalar()
{
	// Setup objects for the test
	double value1 = 42.0;
	double value2 = 63.0;
	double value3 = 94.5;
	double value4 = 141.75;
	double d = value1;
	RunSection::ActionScalar as = RunSection::ActionScalar(d, nullptr); // The ActionScalar (without check function)
	std::map<std::string, RunSection::ActionScalar> asMap;				// ActionScalar map
	std::map<std::string, RunSection::ActionVector> avMap;				// ActionVector map
	asMap.insert(RunSection::NamedActionScalar("testscalar", as));		// Add ActionScalar to the map with the name "testscalar"
	std::string actionname = "test";
	std::string actioncontents = "scalar=testscalar;value=1.5;";
	MSDParser::ObjectParser parser(actionname, actioncontents);
	RunSection::ActionMultiplyScalar action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_double(d, value1);
	action.Step(2);
	isCorrect &= equal_double(as.Get(), value2);
	action_ptr->Step(3);
	isCorrect &= equal_double(d, value3);
	action.Step(4);
	isCorrect &= equal_double(as.Get(), value4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the AddVector Action
bool test_action_addvector()
{
	// Setup objects for the test
	arma::vec value1("1 0 0");
	arma::vec value2("1 0 2");
	arma::vec value3("1 0 4");
	arma::vec value4("1 0 6");
	arma::vec v = value1;
	RunSection::ActionVector av = RunSection::ActionVector(v, nullptr); // The ActionVector (without check function)
	std::map<std::string, RunSection::ActionScalar> asMap;				// ActionScalar map
	std::map<std::string, RunSection::ActionVector> avMap;				// ActionVector map
	avMap.insert(RunSection::NamedActionVector("testvec", av));			// Add ActionVector to the map with the name "testvec"
	std::string actionname = "test";
	std::string actioncontents = "vector=testvec;value=2;direction=0 0 100;"; // The "direction" vector should be normalized to 1
	MSDParser::ObjectParser parser(actionname, actioncontents);
	RunSection::ActionAddVector action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_vec(v, value1);
	action.Step(2);
	isCorrect &= equal_vec(av.Get(), value2);
	action_ptr->Step(3);
	isCorrect &= equal_vec(v, value3);
	action.Step(4);
	isCorrect &= equal_vec(av.Get(), value4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ScaleVector Action
bool test_action_scalevector()
{
	// Setup objects for the test
	arma::vec value1("1 0 0");
	arma::vec value2("2 0 0");
	arma::vec value3("4 0 0");
	arma::vec value4("8 0 0");
	arma::vec v = value1;
	RunSection::ActionVector av = RunSection::ActionVector(v, nullptr); // The ActionVector (without check function)
	std::map<std::string, RunSection::ActionScalar> asMap;				// ActionScalar map
	std::map<std::string, RunSection::ActionVector> avMap;				// ActionVector map
	avMap.insert(RunSection::NamedActionVector("testvec", av));			// Add ActionVector to the map with the name "testvec"
	std::string actionname = "test";
	std::string actioncontents = "vector=testvec;value=2;";
	MSDParser::ObjectParser parser(actionname, actioncontents);
	RunSection::ActionScaleVector action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_vec(v, value1);
	action.Step(2);
	isCorrect &= equal_vec(av.Get(), value2);
	action_ptr->Step(3);
	isCorrect &= equal_vec(v, value3);
	action.Step(4);
	isCorrect &= equal_vec(av.Get(), value4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the RotateVector Action
bool test_action_rotatevector()
{
	// Setup objects for the test
	arma::vec value1("1 0 0");
	arma::vec value2("0 1 0");
	arma::vec value3("-1 0 0");
	arma::vec value4("0 -1 0");
	arma::vec v = value1;
	RunSection::ActionVector av = RunSection::ActionVector(v, nullptr); // The ActionVector (without check function)
	std::map<std::string, RunSection::ActionScalar> asMap;				// ActionScalar map
	std::map<std::string, RunSection::ActionVector> avMap;				// ActionVector map
	avMap.insert(RunSection::NamedActionVector("testvec", av));			// Add ActionVector to the map with the name "testvec"
	std::string actionname = "test";
	std::string actioncontents = "vector=testvec;value=90;axis=0 0 1;";
	MSDParser::ObjectParser parser(actionname, actioncontents);
	RunSection::ActionRotateVector action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_vec(v, value1);
	action.Step(2);
	isCorrect &= equal_vec(av.Get(), value2);
	action_ptr->Step(3);
	isCorrect &= equal_vec(v, value3);
	action.Step(4);
	isCorrect &= equal_vec(av.Get(), value4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the FibonacciSphere Action
bool test_action_fibonaccisphere()
{
	arma::vec value1("0 1 0");
	arma::vec value2("0.707203326 0.4974874372 0.5023641165");		// i = 50;
	arma::vec value3("0.3292513656 -0.005025125628 0.9442289375");	// i = 100
	arma::vec value4("-0.09728003857 -0.9899497487 -0.1026454532"); // i = 198
	arma::vec v = value1;
	RunSection::ActionVector av = RunSection::ActionVector(v, nullptr);
	std::map<std::string, RunSection::ActionScalar> asMap;
	std::map<std::string, RunSection::ActionVector> avMap;
	avMap.insert(RunSection::NamedActionVector("testvec", av));
	std::string actioname = "test";
	std::string actioncontents = "vector=testvec;points=200;";
	MSDParser::ObjectParser parser(actioname, actioncontents);
	RunSection::ActionFibonacciSphere action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_vec(v, value1, 1e-4);
	for (int i = 1; i <= 50; i++)
	{
		action.Step(i);
	}
	isCorrect &= equal_vec(v, value2, 1e-4);
	for (int i = 51; i <= 100; i++)
	{
		action_ptr->Step(i);
	}
	isCorrect &= equal_vec(av.Get(), value3, 1e-4);
	for (int i = 101; i <= 198; i++)
	{
		action.Step(i);
	}
	isCorrect &= equal_vec(v, value4, 1e-4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the Logspace action
bool test_action_LogSpace()
{
	arma::rowvec logspace = arma::logspace<arma::rowvec>(0, 6, 50);
	double d = logspace[0];
	RunSection::ActionScalar as = RunSection::ActionScalar(d, nullptr); // The ActionScalar (without check function)
	std::map<std::string, RunSection::ActionScalar> asMap;				// ActionScalar map
	std::map<std::string, RunSection::ActionVector> avMap;				// ActionVector map
	asMap.insert(RunSection::NamedActionScalar("testscalar", as));
	std::string actionname = "test";
	std::string actioncontents = "scalar=testscalar;points=50;minvalue=0.0;maxvalue=6.0;";
	MSDParser::ObjectParser parser(actionname, actioncontents);
	RunSection::ActionLogSpace action(parser, asMap, avMap);
	RunSection::Action *action_ptr = &action;

	bool isCorrect = true;

	// perform the test
	isCorrect &= action_ptr->Validate();
	isCorrect &= equal_double(d, logspace[0]);
	for (int i = 1; i <= 10; i++)
	{
		action.Step(i);
	}
	isCorrect &= equal_double(d, logspace[10]);
	for (int i = 11; i <= 20; i++)
	{
		action_ptr->Step(i);
	}
	isCorrect &= equal_double(d, logspace[20]);
	for (int i = 21; i < 50; i++)
	{
		action.Step(i);
	}
	isCorrect &= equal_double(d, logspace[49]);

	// Return the result
	return isCorrect;
}

//////////////////////////////////////////////////////////////////////////////
// Invalid scheduling and geometry must be rejected during validation rather
// than failing later in the calculation loop.
bool test_action_validation_guards()
{
	double scalarValue = 1.0;
	arma::vec vectorValue = {1.0, 0.0, 0.0};
	arma::vec shortVectorValue = {1.0, 0.0};
	std::map<std::string, RunSection::ActionScalar> scalars{
		{"scalar", RunSection::ActionScalar(scalarValue, nullptr)}};
	std::map<std::string, RunSection::ActionVector> vectors{
		{"vector", RunSection::ActionVector(vectorValue, nullptr)},
		{"shortVector", RunSection::ActionVector(shortVectorValue, nullptr)}};

	MSDParser::ObjectParser zeroPeriodParser(
		"zeroPeriod", "scalar=scalar;value=1;period=0;");
	RunSection::ActionAddScalar zeroPeriod(zeroPeriodParser, scalars, vectors);

	MSDParser::ObjectParser zeroDirectionParser(
		"zeroDirection", "vector=vector;value=1;direction=0 0 0;");
	RunSection::ActionAddVector zeroDirection(zeroDirectionParser, scalars, vectors);

	MSDParser::ObjectParser zeroAxisParser(
		"zeroAxis", "vector=vector;value=10;axis=0 0 0;");
	RunSection::ActionRotateVector zeroAxis(zeroAxisParser, scalars, vectors);

	MSDParser::ObjectParser shortTargetParser(
		"shortTarget", "vector=shortVector;value=1;direction=1 0 0;");
	RunSection::ActionAddVector shortTarget(shortTargetParser, scalars, vectors);

	MSDParser::ObjectParser shortSphereParser(
		"shortSphere", "vector=vector;points=1;");
	RunSection::ActionFibonacciSphere shortSphere(shortSphereParser, scalars, vectors);

	MSDParser::ObjectParser emptyLogParser(
		"emptyLog", "scalar=scalar;points=0;min=0;max=1;");
	RunSection::ActionLogSpace emptyLog(emptyLogParser, scalars, vectors);

	return !zeroPeriod.Validate() &&
		   !zeroDirection.Validate() &&
		   !zeroAxis.Validate() &&
		   !shortTarget.Validate() &&
		   !shortSphere.Validate() &&
		   !emptyLog.Validate();
}

//////////////////////////////////////////////////////////////////////////////
// Arithmetic actions must not write infinities even when a custom target was
// registered without its own validation callback.
bool test_action_arithmetic_overflow_guards()
{
	double scalarValue = std::numeric_limits<double>::max();
	arma::vec vectorValue = {
		std::numeric_limits<double>::max(), 0.0, 0.0};
	std::map<std::string, RunSection::ActionScalar> scalars{
		{"scalar", RunSection::ActionScalar(scalarValue, nullptr)}};
	std::map<std::string, RunSection::ActionVector> vectors{
		{"vector", RunSection::ActionVector(vectorValue, nullptr)}};

	MSDParser::ObjectParser scalarParser(
		"scalarOverflow", "scalar=scalar;value=2;");
	RunSection::ActionMultiplyScalar scalarAction(scalarParser, scalars, vectors);
	MSDParser::ObjectParser vectorParser(
		"vectorOverflow", "vector=vector;value=2;");
	RunSection::ActionScaleVector vectorAction(vectorParser, scalars, vectors);
	if (!scalarAction.Validate() || !vectorAction.Validate())
		return false;

	scalarAction.Step(1);
	vectorAction.Step(1);
	return std::isfinite(scalarValue) &&
		   equal_double(scalarValue, std::numeric_limits<double>::max()) &&
		   vectorValue.is_finite() &&
		   equal_double(vectorValue(0), std::numeric_limits<double>::max());
}

//////////////////////////////////////////////////////////////////////////////
// Grid actions initialize the first calculation point during validation. Once
// the configured grid is exhausted, later RunSection::Step calls must be a
// no-op instead of indexing beyond the generated points.
bool test_action_grid_bounds()
{
	double scalarValue = 0.0;
	arma::vec vectorValue = {0.0, 2.0, 0.0};
	double delayedScalarValue = 7.0;
	arma::vec delayedVectorValue = {2.0, 0.0, 0.0};
	std::map<std::string, RunSection::ActionScalar> scalars{
		{"scalar", RunSection::ActionScalar(scalarValue, nullptr)},
		{"delayedScalar", RunSection::ActionScalar(delayedScalarValue, nullptr)}};
	std::map<std::string, RunSection::ActionVector> vectors{
		{"vector", RunSection::ActionVector(vectorValue, nullptr)},
		{"delayedVector", RunSection::ActionVector(delayedVectorValue, nullptr)}};

	MSDParser::ObjectParser logParser(
		"log", "scalar=scalar;points=3;min=0;max=2;");
	RunSection::ActionLogSpace logAction(logParser, scalars, vectors);
	if (!logAction.Validate() || !equal_double(scalarValue, 1.0))
		return false;
	logAction.Step(1);
	logAction.Step(2);
	if (!equal_double(scalarValue, 100.0))
		return false;
	logAction.Step(3);
	logAction.Step(4);
	if (!equal_double(scalarValue, 100.0))
		return false;

	MSDParser::ObjectParser sphereParser(
		"sphere", "vector=vector;points=2;");
	RunSection::ActionFibonacciSphere sphereAction(sphereParser, scalars, vectors);
	if (!sphereAction.Validate())
		return false;
	sphereAction.Step(1);
	const arma::vec finalPoint = vectorValue;
	sphereAction.Step(2);
	sphereAction.Step(3);

	if (!equal_vec(vectorValue, finalPoint, 1.0e-12) ||
		!equal_double(arma::norm(vectorValue), 2.0, 1.0e-12))
		return false;

	// A delayed grid must not consume or install point zero at validation.
	MSDParser::ObjectParser delayedLogParser(
		"delayedLog", "scalar=delayedScalar;points=2;min=0;max=1;first=3;");
	RunSection::ActionLogSpace delayedLog(delayedLogParser, scalars, vectors);
	if (!delayedLog.Validate() || !equal_double(delayedScalarValue, 7.0))
		return false;
	delayedLog.Step(2);
	if (!equal_double(delayedScalarValue, 7.0))
		return false;
	delayedLog.Step(3);
	if (!equal_double(delayedScalarValue, 1.0))
		return false;

	MSDParser::ObjectParser delayedSphereParser(
		"delayedSphere", "vector=delayedVector;points=2;first=3;");
	RunSection::ActionFibonacciSphere delayedSphere(delayedSphereParser, scalars, vectors);
	if (!delayedSphere.Validate() ||
		!equal_vec(delayedVectorValue, arma::vec({2.0, 0.0, 0.0}), 1.0e-12))
		return false;
	delayedSphere.Step(2);
	if (!equal_vec(delayedVectorValue, arma::vec({2.0, 0.0, 0.0}), 1.0e-12))
		return false;
	delayedSphere.Step(3);
	return equal_vec(delayedVectorValue, arma::vec({0.0, 2.0, 0.0}), 1.0e-12);
}

//////////////////////////////////////////////////////////////////////////////
// The reference basis used internally by RotateVector must remain orthogonal
// for axes with negative components. Compare against Rodrigues' formula.
bool test_action_rotatevector_arbitrary_axis()
{
	arma::vec value = {0.2, -0.4, 0.7};
	const arma::vec initial = value;
	arma::vec axis = {-1.0, 2.0, -3.0};
	axis /= arma::norm(axis);
	const double angleDegrees = 37.0;
	const double angle = angleDegrees * arma::datum::pi / 180.0;

	std::map<std::string, RunSection::ActionScalar> scalars;
	std::map<std::string, RunSection::ActionVector> vectors{
		{"vector", RunSection::ActionVector(value, nullptr)}};
	MSDParser::ObjectParser parser(
		"rotate", "vector=vector;value=37;axis=-1 2 -3;");
	RunSection::ActionRotateVector action(parser, scalars, vectors);
	if (!action.Validate())
		return false;

	action.Step(1);
	const arma::vec expected =
		initial * std::cos(angle) +
		arma::cross(axis, initial) * std::sin(angle) +
		axis * arma::dot(axis, initial) * (1.0 - std::cos(angle));
	return equal_vec(value, expected, 1.0e-12);
}

//////////////////////////////////////////////////////////////////////////////
// Each interaction owns a uniquely named field target. Two actions are
// applied between task runs, so both Zeeman fields and the next Hamiltonian
// must reflect the same sweep increment.
bool test_action_two_zeeman_fields_update_together()
{
	auto electron1 = std::make_shared<SpinAPI::Spin>(
		"E1", "type=electron;spin=1/2;tensor=2 2 2;");
	auto electron2 = std::make_shared<SpinAPI::Spin>(
		"E2", "type=electron;spin=1/2;tensor=2 2 2;");
	auto zeeman1 = std::make_shared<SpinAPI::Interaction>(
		"zeeman1",
		"type=zeeman;spins=E1;field=0 0 0.10;ignoretensors=true;"
		"commonprefactor=false;prefactor=1;");
	auto zeeman2 = std::make_shared<SpinAPI::Interaction>(
		"zeeman2",
		"type=zeeman;spins=E2;field=0 0 0.20;ignoretensors=true;"
		"commonprefactor=false;prefactor=1;");

	auto system = std::make_shared<SpinAPI::SpinSystem>("SweepSystem");
	system->Add(electron1);
	system->Add(electron2);
	system->Add(zeeman1);
	system->Add(zeeman2);
	if (!system->ValidateInteractions().empty())
		return false;

	RunSection::RunSection runSection;
	runSection.Add(system);
	const auto targets = runSection.GetActionVectors();
	if (targets.count("SweepSystem.zeeman1.field") != 1 ||
		targets.count("SweepSystem.zeeman2.field") != 1)
		return false;

	MSDParser::ObjectParser action1(
		"field1",
		"type=addvector;vector=SweepSystem.zeeman1.field;"
		"direction=0 0 1;value=0.05;");
	MSDParser::ObjectParser action2(
		"field2",
		"type=addvector;vector=SweepSystem.zeeman2.field;"
		"direction=0 0 1;value=0.05;");
	if (!runSection.Add(MSDParser::ObjectType::Action, action1) ||
		!runSection.Add(MSDParser::ObjectType::Action, action2))
		return false;

	SpinAPI::SpinSpace space(*system);
	arma::sp_cx_mat before;
	arma::sp_cx_mat after;
	if (!space.Hamiltonian(before) || !runSection.Step(2) || !space.Hamiltonian(after))
		return false;

	const arma::vec field1 = zeeman1->Field();
	const arma::vec field2 = zeeman2->Field();
	return equal_double(field1(2), 0.15, 1.0e-14) &&
		   equal_double(field2(2), 0.25, 1.0e-14) &&
		   arma::norm(after - before, "fro") > 1.0e-12;
}

//////////////////////////////////////////////////////////////////////////////
// A time-dependent interaction regenerates its instantaneous field in
// SetTime(). Its .field target is therefore readonly; actions must modify the
// persistent .basefield target that feeds the time-dependence function.
bool test_action_time_dependent_basefield()
{
	auto electron = std::make_shared<SpinAPI::Spin>(
		"E", "type=electron;spin=1/2;");
	auto drive = std::make_shared<SpinAPI::Interaction>(
		"drive",
		"type=singlespin;spins=E;fieldtype=linearpolarization;"
		"field=1 0 0;frequency=0;phase=0;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("DynamicSystem");
	system->Add(electron);
	system->Add(drive);
	if (!system->ValidateInteractions().empty())
		return false;

	RunSection::RunSection runSection;
	runSection.Add(system);
	const auto targets = runSection.GetActionVectors();
	auto field = targets.find("DynamicSystem.drive.field");
	auto basefield = targets.find("DynamicSystem.drive.basefield");
	if (field == targets.end() || basefield == targets.end() ||
		!field->second.IsReadonly() || basefield->second.IsReadonly())
		return false;

	MSDParser::ObjectParser invalidAction(
		"instantaneous",
		"type=addvector;vector=DynamicSystem.drive.field;"
		"direction=1 0 0;value=1;");
	if (runSection.Add(MSDParser::ObjectType::Action, invalidAction))
		return false;

	MSDParser::ObjectParser baseAction(
		"base",
		"type=addvector;vector=DynamicSystem.drive.basefield;"
		"direction=1 0 0;value=1;");
	if (!runSection.Add(MSDParser::ObjectType::Action, baseAction) ||
		!runSection.Step(2) ||
		!drive->SetTime(0.0))
		return false;

	return equal_vec(drive->Field(), arma::vec({2.0, 0.0, 0.0}), 1.0e-14);
}

//////////////////////////////////////////////////////////////////////////////
// ActionScalar aliases must refresh function-defined states, including a
// state containing several variables used by different functions.
bool test_action_state_variable_refresh()
{
	auto electron = std::make_shared<SpinAPI::Spin>(
		"E", "type=electron;spin=1/2;");
	auto state = std::make_shared<SpinAPI::State>(
		"Mix",
		"a=0;b=1.5707963267948966;"
		"spin(E)=cos(a)|1/2>+sin(b)|-1/2>;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("StateSystem");
	system->Add(electron);
	system->Add(state);
	if (!state->ParseFromSystem(*system))
		return false;

	RunSection::RunSection runSection;
	runSection.Add(system);
	MSDParser::ObjectParser action(
		"stateAngle",
		"type=addscalar;actionscalar=StateSystem.Mix.a;"
		"value=1.5707963267948966;");
	if (!runSection.Add(MSDParser::ObjectType::Action, action) ||
		!runSection.Step(2))
		return false;

	SpinAPI::SpinSpace space(electron);
	arma::cx_vec actual;
	arma::cx_vec expected(2, arma::fill::zeros);
	expected(1) = 1.0;
	return space.GetState(state, actual) && equal_vec(actual, expected, 1.0e-12);
}

//////////////////////////////////////////////////////////////////////////////
// PulseSequence exposes delay variables, but they are useful only if the
// enclosing SpinSystem forwards those targets to RunSection.
bool test_action_pulse_sequence_target_registration()
{
	auto electron = std::make_shared<SpinAPI::Spin>(
		"E", "type=electron;spin=1/2;");
	auto pulse = std::make_shared<SpinAPI::Pulse>(
		"CW",
		"type=longpulsestaticfield;group=E;field=0 0 1;pulsetime=1;"
		"prefactorlist=1;commonprefactorlist=true;ignoretensorslist=true;");
	std::vector<SpinAPI::spin_ptr> spins{electron};
	if (!pulse->ParseSpinGroups(spins))
		return false;

	auto sequence = std::make_shared<SpinAPI::PulseSequence>(
		"sequence", "tau=1;offset=0;sequence=CW,tau;");
	std::vector<SpinAPI::pulse_ptr> pulses{pulse};
	std::vector<SpinAPI::interaction_ptr> interactions;
	std::vector<SpinAPI::transition_ptr> transitions;
	if (!sequence->ParsePulseSequence(pulses, interactions, transitions))
		return false;

	auto system = std::make_shared<SpinAPI::SpinSystem>("PulseSystem");
	system->Add(electron);
	system->Add(pulse);
	system->Add(sequence);

	RunSection::RunSection runSection;
	runSection.Add(system);
	auto targets = runSection.GetActionScalars();
	auto tauTarget = targets.find("PulseSystem.sequence.tau");
	if (tauTarget == targets.end() || tauTarget->second.Set(-0.5))
		return false;

	MSDParser::ObjectParser action(
		"delay",
		"type=addscalar;scalar=PulseSystem.sequence.tau;value=0.5;");
	if (!runSection.Add(MSDParser::ObjectType::Action, action) ||
		!runSection.Step(2))
		return false;

	const auto &delays = sequence->Get_tau_list();
	auto delay = delays.find("tau");
	return delay != delays.end() && equal_double(delay->second, 1.5, 1.0e-14);
}

//////////////////////////////////////////////////////////////////////////////
// Scalar target names are not necessarily State variables. A missing system
// or object must not cause RunSection's state-refresh pass to index past its
// system collection.
bool test_action_nonstate_scalar_target_is_safe()
{
	double value = 0.0;
	RunSection::RunSection runSection;
	runSection.Add(
		"MissingSystem.MissingState.value",
		RunSection::ActionScalar(value, nullptr));
	MSDParser::ObjectParser action(
		"custom", "type=addscalar;scalar=MissingSystem.MissingState.value;value=2;");
	return runSection.Add(MSDParser::ObjectType::Action, action) &&
		   runSection.Step(2) &&
		   equal_double(value, 2.0);
}

//////////////////////////////////////////////////////////////////////////////
// Action targets must enforce the domains assumed by the downstream physics:
// normalized axes need a nonzero direction, durations and diffusion
// coefficients cannot be negative, and sampled broadband bounds are immutable.
bool test_action_physical_target_guards()
{
	auto electron = std::make_shared<SpinAPI::Spin>(
		"E", "type=electron;spin=1;");
	std::vector<SpinAPI::spin_ptr> spins{electron};

	auto instantPulse = std::make_shared<SpinAPI::Pulse>(
		"instant", "type=instantpulse;group=E;rotationaxis=1 0 0;angle=90;");
	auto longPulse = std::make_shared<SpinAPI::Pulse>(
		"long",
		"type=longpulsestaticfield;group=E;field=0 0 1;pulsetime=2;"
		"prefactorlist=1;commonprefactorlist=true;ignoretensorslist=true;");
	if (!instantPulse->ParseSpinGroups(spins) || !longPulse->ParseSpinGroups(spins))
		return false;

	std::vector<RunSection::NamedActionScalar> pulseScalars;
	std::vector<RunSection::NamedActionVector> pulseVectors;
	instantPulse->GetActionTargets(pulseScalars, pulseVectors, "GuardSystem");
	longPulse->GetActionTargets(pulseScalars, pulseVectors, "GuardSystem");

	bool axisRejected = false;
	for (auto &target : pulseVectors)
	{
		if (target.first == "GuardSystem.instant.rotationaxis")
			axisRejected = !target.second.Set(arma::zeros<arma::vec>(3));
	}
	bool durationRejected = false;
	for (auto &target : pulseScalars)
	{
		if (target.first == "GuardSystem.long.pulsetime")
			durationRejected = !target.second.Set(-1.0);
	}

	auto circular = std::make_shared<SpinAPI::Interaction>(
		"circular",
		"type=singlespin;spins=E;fieldtype=circularpolarization;"
		"field=1 0 0;axis=0 0 1;frequency=1;phase=0;");
	auto strain = std::make_shared<SpinAPI::Interaction>(
		"strain",
		"type=strain;group1=E;e=1 2 3 4 5 6;d=10 20 30 40 50 60;"
		"tensortype=broadband;minfreq=1;maxfreq=2;components=1;"
		"autoseed=false;seed=1;rwdcoeff=3;");
	auto system = std::make_shared<SpinAPI::SpinSystem>("GuardSystem");
	system->Add(electron);
	system->Add(circular);
	system->Add(strain);
	if (!system->ValidateInteractions().empty())
		return false;

	RunSection::RunSection runSection;
	runSection.Add(system);
	auto vectors = runSection.GetActionVectors();
	auto scalars = runSection.GetActionScalars();
	auto quantizationAxis = vectors.find("GuardSystem.E.quantizationaxis1");
	auto circularAxis = vectors.find("GuardSystem.circular.axis");
	auto minFrequency = scalars.find("GuardSystem.strain.minfreq");
	auto maxFrequency = scalars.find("GuardSystem.strain.maxfreq");
	auto diffusion = scalars.find("GuardSystem.strain.rwdcoeff");
	if (quantizationAxis == vectors.end() ||
		circularAxis == vectors.end() ||
		minFrequency == scalars.end() ||
		maxFrequency == scalars.end() ||
		diffusion == scalars.end())
		return false;

	return axisRejected &&
		   durationRejected &&
		   quantizationAxis->second.IsReadonly() &&
		   !circularAxis->second.Set(arma::zeros<arma::vec>(3)) &&
		   minFrequency->second.IsReadonly() &&
		   maxFrequency->second.IsReadonly() &&
		   equal_double(diffusion->second.Get(), 3.0, 1.0e-14) &&
		   !diffusion->second.Set(-1.0);
}

//////////////////////////////////////////////////////////////////////////////
// Helpers for end-to-end General task output across multiple RunSection steps.
bool action_general_output_column(const std::string &data,
                                  const std::string &label,
                                  std::size_t &column)
{
    std::istringstream lines(data);
    std::string line;
    while (std::getline(lines, line))
    {
        if (line.empty())
            continue;
        std::istringstream tokens(line);
        std::string token;
        std::size_t index = 0;
        while (tokens >> token)
        {
            if (token == label)
            {
                column = index;
                return true;
            }
            ++index;
        }
        return false;
    }
    return false;
}

bool action_general_last_numeric_row_for_step(const std::string &data,
                                              unsigned int step,
                                              std::vector<double> &row)
{
    row.clear();
    std::istringstream lines(data);
    std::string line;
    while (std::getline(lines, line))
    {
        std::istringstream values(line);
        std::vector<double> candidate;
        double value = 0.0;
        while (values >> value)
            candidate.push_back(value);
        if (!candidate.empty() &&
            std::abs(candidate.front() - static_cast<double>(step)) < 1.0e-12)
        {
            row = std::move(candidate);
        }
    }
    return !row.empty();
}

//////////////////////////////////////////////////////////////////////////////
// HSGeneral qualification: an ActionVector changes a Zeeman field after the
// first calculation step. Reusing the same General Hamiltonian builder must
// rebuild from the live Interaction object, including a non-trivial explicit
// molecular-to-lab orientation; no stale Hamiltonian is permitted.
bool test_action_hsgeneral_field_rebuild_after_step()
{
    auto electron = std::make_shared<SpinAPI::Spin>(
        "E", "type=electron;spin=1/2;tensor=isotropic(2.0);");
    auto field = std::make_shared<SpinAPI::Interaction>(
        "B0",
        "type=zeeman;spins=E;field=0 0 0;ignoretensors=true;"
        "commonprefactor=false;prefactor=1;");

    auto system = std::make_shared<SpinAPI::SpinSystem>("ActionHS");
    system->Add(electron);
    system->Add(field);
    if (!system->ValidateInteractions().empty())
        return false;

    RunSection::RunSection runSection;
    if (!runSection.Add(system))
        return false;
    const auto targets = runSection.GetActionVectors();
    if (targets.count("ActionHS.B0.field") != 1)
        return false;

    MSDParser::ObjectParser action(
        "fieldStep",
        "type=addvector;vector=ActionHS.B0.field;"
        "direction=1 2 3;value=0.01;");
    if (!runSection.Add(MSDParser::ObjectType::Action, action))
        return false;

    RunSection::General::HS::HSExecutionPlan plan;
    plan.approximation = SpinAPI::HamiltonianApproximation::Full;
    plan.orientation = RunSection::General::HS::OrientationMode::Explicit;
    plan.hasH0List = true;
    plan.h0List = {"B0"};

    RunSection::General::HS::HSOrientation orientation;
    orientation.alpha = 0.23;
    orientation.beta = 0.61;
    orientation.gamma = -0.17;
    orientation.weight = 1.0;
    if (!SpinAPI::CreateZYZRotationMatrix(
            orientation.alpha, orientation.beta, orientation.gamma,
            orientation.frameToLab))
        return false;

    SpinAPI::SpinSpace space(system);
    space.UseSuperoperatorSpace(false);
    RunSection::General::HS::HSHamiltonianBuilder builder(plan, space);

    arma::sp_cx_mat before, after;
    std::string error;
    if (!builder.BuildStatic(orientation, before, nullptr, error))
        return false;
    if (arma::norm(before, "fro") > 1.0e-14)
        return false;

    if (!runSection.Step(2))
        return false;
    const arma::vec updatedField = field->Field();
    if (updatedField.n_elem != 3 ||
        std::abs(arma::norm(updatedField, 2) - 0.01) > 1.0e-13)
        return false;

    if (!builder.BuildStatic(orientation, after, nullptr, error))
        return false;
    return arma::norm(after - before, "fro") > 1.0e-10;
}

//////////////////////////////////////////////////////////////////////////////
// SSGeneral qualification: a Transition rate Action is applied between two
// runs of the same TaskSSGeneral instance. The second RunLocal call must
// reconstruct the Liouvillian from the updated Transition rather than reuse a
// stale reaction generator.
bool test_action_ssgeneral_transition_rate_rebuilds_between_runs()
{
    auto electron = std::make_shared<SpinAPI::Spin>(
        "E", "type=electron;spin=1/2;");
    auto up = std::make_shared<SpinAPI::State>(
        "Up", "spin(E)=|1/2>;");
    auto all = std::make_shared<SpinAPI::State>("All", "");

    auto system = std::make_shared<SpinAPI::SpinSystem>("ActionSS");
    system->Add(electron);
    system->Add(up);
    system->Add(all);
    auto sink = std::make_shared<SpinAPI::Transition>(
        "sink", "type=sink;sourcestate=All;rate=0.1;", system);
    system->Add(sink);
    system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
        "properties", "initialstate=Up;initialstatecoherences=keep;"));

    if (!up->ParseFromSystem(*system) || !all->ParseFromSystem(*system) ||
        !system->ValidateTransitions({system}).empty())
        return false;

    RunSection::RunSection runSection;
    if (!runSection.Add(system))
        return false;
    const auto scalarTargets = runSection.GetActionScalars();
    if (scalarTargets.count("ActionSS.sink.rate") != 1)
        return false;

    MSDParser::ObjectParser action(
        "rateStep",
        "type=addscalar;scalar=ActionSS.sink.rate;value=0.1;");
    if (!runSection.Add(MSDParser::ObjectType::Action, action))
        return false;

    MSDParser::ObjectParser taskParser(
        "general",
        "type=SSGeneral;calculation=timeevolution;"
        "propagationmethod=exponential;observables=states;"
        "totaltime=1;timestep=1;");
    if (!runSection.Add(MSDParser::ObjectType::Task, taskParser))
        return false;
    auto task = runSection.GetTask("general");
    if (!task)
        return false;

    std::ostringstream log, data;
    task->SetLogStream(log);
    task->SetDataStream(data);

    runSection.Run(1);
    if (std::abs(sink->Rate() - 0.1) > 1.0e-14)
        return false;
    if (!runSection.Step(2) || std::abs(sink->Rate() - 0.2) > 1.0e-14)
        return false;
    runSection.Run(2);

    std::size_t upColumn = 0;
    std::vector<double> step1, step2;
    if (!action_general_output_column(
            data.str(), "ActionSS.Up.population", upColumn) ||
        !action_general_last_numeric_row_for_step(data.str(), 1, step1) ||
        !action_general_last_numeric_row_for_step(data.str(), 2, step2) ||
        upColumn >= step1.size() || upColumn >= step2.size())
        return false;

    const double ref1 = std::exp(-0.1);
    const double ref2 = std::exp(-0.2);
    return std::abs(step1[upColumn] - ref1) < 1.0e-11 &&
           std::abs(step2[upColumn] - ref2) < 1.0e-11 &&
           std::abs(step1[upColumn] - step2[upColumn]) > 1.0e-3;
}

//////////////////////////////////////////////////////////////////////////////
// MultiSSGeneral qualification: a source->target transfer-rate Action is
// applied between two runs of the same task. The direct-sum graph must be
// rebuilt, so source population at t=1 follows exp(-k t) for k=0.1 and 0.2.
bool test_action_multissgeneral_transfer_rate_rebuilds_between_runs()
{
    auto source = std::make_shared<SpinAPI::SpinSystem>("ActionA");
    auto target = std::make_shared<SpinAPI::SpinSystem>("ActionB");
    auto sourceLevel = std::make_shared<SpinAPI::Spin>("level", "spin=0;");
    auto targetLevel = std::make_shared<SpinAPI::Spin>("level", "spin=0;");
    auto sourceState = std::make_shared<SpinAPI::State>(
        "State", "spin(level)=|0>;");
    auto targetState = std::make_shared<SpinAPI::State>(
        "State", "spin(level)=|0>;");

    source->Add(sourceLevel);
    source->Add(sourceState);
    target->Add(targetLevel);
    target->Add(targetState);
    auto transfer = std::make_shared<SpinAPI::Transition>(
        "AtoB",
        "type=sink;sourcestate=State;target=ActionB;"
        "targetstate=State;rate=0.1;",
        source);
    source->Add(transfer);
    source->SetProperties(std::make_shared<MSDParser::ObjectParser>(
        "source_properties", "initialstate=State;"));
    target->SetProperties(std::make_shared<MSDParser::ObjectParser>(
        "target_properties", ""));

    const std::vector<SpinAPI::system_ptr> systems{source, target};
    if (!sourceState->ParseFromSystem(*source) ||
        !targetState->ParseFromSystem(*target) ||
        !source->ValidateTransitions(systems).empty() ||
        !target->ValidateTransitions(systems).empty())
        return false;

    RunSection::RunSection runSection;
    if (!runSection.Add(source) || !runSection.Add(target))
        return false;
    const auto scalarTargets = runSection.GetActionScalars();
    if (scalarTargets.count("ActionA.AtoB.rate") != 1)
        return false;

    MSDParser::ObjectParser action(
        "transferRateStep",
        "type=addscalar;scalar=ActionA.AtoB.rate;value=0.1;");
    if (!runSection.Add(MSDParser::ObjectType::Action, action))
        return false;

    MSDParser::ObjectParser taskParser(
        "general",
        "type=MultiSSGeneral;calculation=timeevolution;"
        "propagationmethod=exponential;observables=states;"
        "totaltime=1;timestep=1;");
    if (!runSection.Add(MSDParser::ObjectType::Task, taskParser))
        return false;
    auto task = runSection.GetTask("general");
    if (!task)
        return false;

    std::ostringstream log, data;
    task->SetLogStream(log);
    task->SetDataStream(data);

    runSection.Run(1);
    if (std::abs(transfer->Rate() - 0.1) > 1.0e-14)
        return false;
    if (!runSection.Step(2) || std::abs(transfer->Rate() - 0.2) > 1.0e-14)
        return false;
    runSection.Run(2);

    std::size_t sourceColumn = 0, targetColumn = 0;
    std::vector<double> step1, step2;
    if (!action_general_output_column(
            data.str(), "ActionA.State.population", sourceColumn) ||
        !action_general_output_column(
            data.str(), "ActionB.State.population", targetColumn) ||
        !action_general_last_numeric_row_for_step(data.str(), 1, step1) ||
        !action_general_last_numeric_row_for_step(data.str(), 2, step2) ||
        sourceColumn >= step1.size() || sourceColumn >= step2.size() ||
        targetColumn >= step1.size() || targetColumn >= step2.size())
        return false;

    const double pA1 = std::exp(-0.1);
    const double pA2 = std::exp(-0.2);
    return std::abs(step1[sourceColumn] - pA1) < 1.0e-11 &&
           std::abs(step2[sourceColumn] - pA2) < 1.0e-11 &&
           std::abs(step1[targetColumn] - (1.0 - pA1)) < 1.0e-11 &&
           std::abs(step2[targetColumn] - (1.0 - pA2)) < 1.0e-11;
}

//////////////////////////////////////////////////////////////////////////////
// Contract guard for profiled MultiSS transfers.
// A writable Action target must not silently become physically irrelevant.
bool test_action_multiss_profiled_rate_is_effective_or_rejected()
{
    auto source = std::make_shared<SpinAPI::SpinSystem>("ProfileA");
    auto target = std::make_shared<SpinAPI::SpinSystem>("ProfileB");
    auto sourceLevel = std::make_shared<SpinAPI::Spin>("level", "spin=0;");
    auto targetLevel = std::make_shared<SpinAPI::Spin>("level", "spin=0;");
    auto sourceState = std::make_shared<SpinAPI::State>(
        "State", "spin(level)=|0>;");
    auto targetState = std::make_shared<SpinAPI::State>(
        "State", "spin(level)=|0>;");
    source->Add(sourceLevel);
    source->Add(sourceState);
    target->Add(targetLevel);
    target->Add(targetState);

    auto transfer = std::make_shared<SpinAPI::Transition>(
        "pump",
        "type=sink;sourcestate=State;target=ProfileB;targetstate=State;"
        "rate=0.1;rateprofile=gaussian;pulsecenter=5;pulsefwhm=2;"
        "transferfraction=0.5;",
        source);
    source->Add(transfer);
    const std::vector<SpinAPI::system_ptr> systems{source, target};
    if (!sourceState->ParseFromSystem(*source) ||
        !targetState->ParseFromSystem(*target) ||
        !source->ValidateTransitions(systems).empty())
        return false;

    SpinAPI::TransferChannel before;
    std::string error;
    if (!SpinAPI::TransferChannel::Compile(transfer, before, error))
        return false;
    const double kBefore = before.Rate(5.0);

    RunSection::RunSection runSection;
    if (!runSection.Add(source) || !runSection.Add(target))
        return false;
    auto targets = runSection.GetActionScalars();
    auto targetIt = targets.find("ProfileA.pump.rate");

    // Safe semantics: do not expose an ineffective mutable target.
    if (targetIt == targets.end() || targetIt->second.IsReadonly())
        return true;

    MSDParser::ObjectParser action(
        "profileRateStep",
        "type=addscalar;scalar=ProfileA.pump.rate;value=0.4;");
    if (!runSection.Add(MSDParser::ObjectType::Action, action))
        return true; // Explicit rejection is also safe.

    if (!runSection.Step(2) || std::abs(transfer->Rate() - 0.5) > 1.0e-14)
        return false;

    SpinAPI::TransferChannel after;
    if (!SpinAPI::TransferChannel::Compile(transfer, after, error))
        return false;
    const double kAfter = after.Rate(5.0);

    // If writable, the actual compiled k(t) must change.
    return std::abs(kAfter - kBefore) >
           1.0e-12 * std::max({1.0, std::abs(kBefore), std::abs(kAfter)});
}

//////////////////////////////////////////////////////////////////////////////
// Profile ActionTarget permissions must mirror TransferChannel semantics.
// Writable means Transition::Rate() is physically used by k(t); read-only
// means another profile parameter owns the physical rate.
bool test_action_multiss_profile_rate_target_permissions()
{
    const auto check = [](const std::string &suffix, bool expectedReadonly)
    {
        auto source = std::make_shared<SpinAPI::SpinSystem>("ProfileSource");
        auto target = std::make_shared<SpinAPI::SpinSystem>("ProfileTarget");
        auto sourceLevel = std::make_shared<SpinAPI::Spin>("level", "spin=0;");
        auto targetLevel = std::make_shared<SpinAPI::Spin>("level", "spin=0;");
        auto sourceState = std::make_shared<SpinAPI::State>(
            "State", "spin(level)=|0>;");
        auto targetState = std::make_shared<SpinAPI::State>(
            "State", "spin(level)=|0>;");
        source->Add(sourceLevel);
        source->Add(sourceState);
        target->Add(targetLevel);
        target->Add(targetState);
        auto transition = std::make_shared<SpinAPI::Transition>(
            "transfer",
            "type=sink;sourcestate=State;target=ProfileTarget;"
            "targetstate=State;rate=0.1;" + suffix,
            source);
        source->Add(transition);

        const std::vector<SpinAPI::system_ptr> systems{source, target};
        if (!sourceState->ParseFromSystem(*source) ||
            !targetState->ParseFromSystem(*target) ||
            !source->ValidateTransitions(systems).empty())
            return false;

        RunSection::RunSection runSection;
        if (!runSection.Add(source) || !runSection.Add(target))
            return false;
        const auto scalars = runSection.GetActionScalars();
        const auto found = scalars.find("ProfileSource.transfer.rate");
        return found != scalars.end() &&
               found->second.IsReadonly() == expectedReadonly;
    };

    return
        check("rateprofile=constant;", false) &&
        check("rateprofile=gaussian;pulsecenter=5;pulsefwhm=2;", false) &&
        check("rateprofile=gaussian;pulsecenter=5;pulsefwhm=2;peakrate=0.4;", true) &&
        check("rateprofile=gaussian;pulsecenter=5;pulsefwhm=2;transferfraction=0.5;", true) &&
        check("rateprofile=rectangular;pulsestart=1;pulseend=2;", false) &&
        check("rateprofile=rectangular;pulsestart=1;pulseend=2;peakrate=0.4;", true) &&
        check("rateprofile=trajectory;profiletimes=0,1;profilerates=0.1,0.2;", true) &&
        check("rateprofile=instantaneous;eventtime=1;transferfraction=0.5;", true);
}

// Add all the Action classes test cases
void AddActionsTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("RunSection::ActionScalar test with check function", test_actiontargets_scalar));
	_cases.push_back(test_case("RunSection::ActionVector test with check function", test_actiontargets_vector));
	_cases.push_back(test_case("Action AddScalar", test_action_addscalar));
	_cases.push_back(test_case("Action MultiplyScalar", test_action_multiplyscalar));
	_cases.push_back(test_case("Action AddVector", test_action_addvector));
	_cases.push_back(test_case("Action ScaleVector", test_action_scalevector));
	_cases.push_back(test_case("Action RotateVector", test_action_rotatevector));
	_cases.push_back(test_case("Action FibonacciSphere", test_action_fibonaccisphere));
	_cases.push_back(test_case("Action Logspace", test_action_LogSpace));
	_cases.push_back(test_case("Action validation guards", test_action_validation_guards));
	_cases.push_back(test_case("Action arithmetic overflow guards", test_action_arithmetic_overflow_guards));
	_cases.push_back(test_case("Action grid bounds", test_action_grid_bounds));
	_cases.push_back(test_case("Action arbitrary-axis rotation", test_action_rotatevector_arbitrary_axis));
	_cases.push_back(test_case("Action two-Zeeman synchronized sweep", test_action_two_zeeman_fields_update_together));
	_cases.push_back(test_case("Action time-dependent base field", test_action_time_dependent_basefield));
	_cases.push_back(test_case("Action state-variable refresh", test_action_state_variable_refresh));
	_cases.push_back(test_case("Action PulseSequence target registration", test_action_pulse_sequence_target_registration));
	_cases.push_back(test_case("Action non-state scalar target safety", test_action_nonstate_scalar_target_is_safe));
	_cases.push_back(test_case("Action physical target guards", test_action_physical_target_guards));
	_cases.push_back(test_case("Action HSGeneral field rebuild after step", test_action_hsgeneral_field_rebuild_after_step));
	_cases.push_back(test_case("Action SSGeneral transition rate rebuild between runs", test_action_ssgeneral_transition_rate_rebuilds_between_runs));
	_cases.push_back(test_case("Action MultiSSGeneral transfer rate rebuild between runs", test_action_multissgeneral_transfer_rate_rebuilds_between_runs));
	_cases.push_back(test_case("Action profiled MultiSS rate is effective or rejected", test_action_multiss_profiled_rate_is_effective_or_rejected));
	_cases.push_back(test_case("Action profiled MultiSS rate target permissions", test_action_multiss_profile_rate_target_permissions));
}
//////////////////////////////////////////////////////////////////////////////
