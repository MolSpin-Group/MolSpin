//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Unit test functions for the SpinAPI module.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////
#include "SpinAPIDefines.h"
#include "Spin.h"
#include "Interaction.h"
#include "Operator.h"
#include "State.h"
#include "Transition.h"
#include "SpinSystem.h"
#include "SpinSpace.h"
#include "Function.h"
#include "Pulse.h"
#include "PulseSequence.h"
#include "PowderGrid.h"
#include <cmath>
#include <sstream>
//////////////////////////////////////////////////////////////////////////////
// Tests whether the spin quantum number is stored correctly.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_spinclass_spin_quantum_number()
{
	// Setup objects for the test
	std::string spin1_name = "testspin1";
	std::string spin1_contents = "spin=1/2;";
	SpinAPI::Spin spin1(spin1_name, spin1_contents);

	std::string spin2_name = "testspin2";
	std::string spin2_contents = "spin=1;";
	SpinAPI::Spin spin2(spin2_name, spin2_contents);

	std::string spin3_name = "testspin3";
	std::string spin3_contents = "spin=3/2;";
	SpinAPI::Spin spin3(spin3_name, spin3_contents);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= (spin1.S() == 1);
	isCorrect &= (spin2.S() == 2);
	isCorrect &= (spin3.S() == 3);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests whether the spin multiplicity is calculated correctly from a given
// spin quantum number. Tests three different spin quantum numbers.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_spinclass_multiplicity_from_s()
{
	// Setup objects for the test
	std::string spin1_name = "testspin1";
	std::string spin1_contents = "spin=1/2;";
	SpinAPI::Spin spin1(spin1_name, spin1_contents);

	std::string spin2_name = "testspin2";
	std::string spin2_contents = "spin=1;";
	SpinAPI::Spin spin2(spin2_name, spin2_contents);

	std::string spin3_name = "testspin3";
	std::string spin3_contents = "spin=3/2;";
	SpinAPI::Spin spin3(spin3_name, spin3_contents);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= (spin1.Multiplicity() == 2);
	isCorrect &= (spin2.Multiplicity() == 3);
	isCorrect &= (spin3.Multiplicity() == 4);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests whether the spin operators are calculated correctly a spin of 1/2.
bool test_spinapi_spinclass_spinmatrices_spinonehalf()
{
	// Setup objects for the test
	std::string spin_name = "testspin";
	std::string spin_contents = "spin=1/2;";
	SpinAPI::Spin spin(spin_name, spin_contents);

	arma::cx_mat sx = 0.5 * arma::cx_mat("0 1;1 0");
	arma::cx_mat sy = 0.5 * arma::cx_mat("0 (0,-1);(0,1) 0");
	arma::cx_mat sz = 0.5 * arma::cx_mat("1 0;0 -1");

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_matrices(spin.Sx(), sx);
	isCorrect &= equal_matrices(spin.Sy(), sy);
	isCorrect &= equal_matrices(spin.Sz(), sz);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests whether the spin operators are calculated correctly a spin of 1.
bool test_spinapi_spinclass_spinmatrices_spinone()
{
	// Setup objects for the test
	std::string spin_name = "testspin";
	std::string spin_contents = "spin=1;"; // 1 in units of hbar
	SpinAPI::Spin spin(spin_name, spin_contents);

	arma::cx_mat sx = sqrt(0.5) * arma::cx_mat("0 1 0;1 0 1;0 1 0");
	arma::cx_mat sy = sqrt(0.5) * arma::cx_mat("0 (0,-1) 0;(0,1) 0 (0,-1);0 (0,1) 0");
	arma::cx_mat sz = arma::cx_mat("1 0 0;0 0 0;0 0 -1");

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_matrices(spin.Sx(), sx);
	isCorrect &= equal_matrices(spin.Sy(), sy);
	isCorrect &= equal_matrices(spin.Sz(), sz);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests whether the spin operators are calculated correctly a spin of 3/2.
bool test_spinapi_spinclass_spinmatrices_spinthreehalf()
{
	// Setup objects for the test
	std::string spin_name = "testspin";
	std::string spin_contents = "spin=3/2;"; // 3/2 in units of hbar
	SpinAPI::Spin spin(spin_name, spin_contents);

	// Note: Automatic type deduction fails here
	arma::cx_mat sx = sqrt(3.0) * 0.5 * arma::cx_mat("0 1 0 0;1 0 0 0;0 0 0 1;0 0 1 0") + arma::cx_mat("0 0 0 0;0 0 1 0;0 1 0 0;0 0 0 0");
	arma::cx_mat sy = sqrt(3.0) * 0.5 * arma::cx_mat("0 (0,-1) 0 0;(0,1) 0 0 0;0 0 0 (0,-1);0 0 (0,1) 0") + arma::cx_mat("0 0 0 0;0 0 (0,-1) 0;0 (0,1) 0 0;0 0 0 0");
	arma::cx_mat sz = 0.5 * arma::cx_mat("3 0 0 0;0 1 0 0;0 0 -1 0;0 0 0 -3");

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_matrices(spin.Sx(), sx);
	isCorrect &= equal_matrices(spin.Sy(), sy);
	isCorrect &= equal_matrices(spin.Sz(), sz);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests an Interaction object with a static field, including the prefactor.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_fieldstatic()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=singlespin;prefactor=42.24;field=0 42 0;";
	SpinAPI::Interaction I(name, contents);

	auto field = arma::vec("0 42 0");
	double prefactor = 42.24;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_vec(I.Field(), field);
	isCorrect &= equal_double(I.Prefactor(), prefactor);
	isCorrect &= I.FieldType() == SpinAPI::InteractionFieldType::Static;
	isCorrect &= I.Type() == SpinAPI::InteractionType::SingleSpin;
	isCorrect &= !I.HasTimeDependence();
	isCorrect &= IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests an Interaction object with a linear polarized oscillating field.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_fieldlinearpolarization()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=singlespin;fieldtype=linearpolarization;field=1 0 0;frequency=0.159154943e+6;phase=24;";
	SpinAPI::Interaction I(name, contents);

	auto field = arma::vec("1 0 0");
	double frequency = 0.159154943e+6;
	double phase = 24;
	double testtime = 4.234e-5;
	auto testtimefield = field * cos(frequency * testtime + phase);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_vec(I.Field(), field * cos(phase));
	isCorrect &= I.SetTime(testtime);
	isCorrect &= equal_vec(I.Field(), testtimefield);
	isCorrect &= I.FieldType() == SpinAPI::InteractionFieldType::LinearPolarization;
	isCorrect &= I.Type() == SpinAPI::InteractionType::SingleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests an Interaction object with a circularly polarized oscillating field.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_fieldcircularpolarization_perpendicular()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=singlespin;fieldtype=circularpolarization;field=1 0 1;axis=0 0 1;frequency=0.159154943e+6;phase=24;perpendicularoscillations=true;";
	SpinAPI::Interaction I(name, contents);

	double frequency = 0.159154943e+6;
	double phase = 24;

	auto testfield1 = arma::vec("0 0 0");
	testfield1(0) = cos(phase);
	testfield1(1) = sin(phase);

	double testtime = 4.234e-5;
	auto testfield2 = arma::vec("0 0 0");
	testfield2(0) = cos(frequency * testtime + phase);
	testfield2(1) = sin(frequency * testtime + phase);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_vec(I.Field(), testfield1);
	isCorrect &= I.SetTime(testtime);
	isCorrect &= equal_vec(I.Field(), testfield2);
	isCorrect &= I.FieldType() == SpinAPI::InteractionFieldType::CircularPolarization;
	isCorrect &= I.Type() == SpinAPI::InteractionType::SingleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests an Interaction object with a circularly polarized oscillating field,
// that does not oscillate in the plane perpendicular to the axis..
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_fieldcircularpolarization_tilted()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=singlespin;fieldtype=circularpolarization;field=1 0 1;axis=0 0 1;frequency=0.159154943e+6;phase=24;perpendicularoscillations=false;";
	SpinAPI::Interaction I(name, contents);

	double frequency = 0.159154943e+6;
	double phase = 24;
	double outofplane_angle = 45.0 / 180.0 * M_PI;

	auto testfield1 = arma::vec("0 0 0");
	testfield1(0) = cos(phase) * cos(outofplane_angle) * sqrt(2);
	testfield1(1) = sin(phase) * cos(outofplane_angle) * sqrt(2);
	testfield1(2) = 1.0;

	double testtime = 4.234e-5;
	auto testfield2 = arma::vec("0 0 0");
	testfield2(0) = cos(frequency * testtime + phase) * cos(outofplane_angle) * sqrt(2);
	testfield2(1) = sin(frequency * testtime + phase) * cos(outofplane_angle) * sqrt(2);
	testfield2(2) = 1.0;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_vec(I.Field(), testfield1);
	isCorrect &= I.SetTime(testtime);
	isCorrect &= equal_vec(I.Field(), testfield2);
	isCorrect &= I.FieldType() == SpinAPI::InteractionFieldType::CircularPolarization;
	isCorrect &= I.Type() == SpinAPI::InteractionType::SingleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the state class
// DEPENDENCY NOTE: ObjectParser, SpinSystem, Spin
bool test_spinapi_state()
{
	// Setup objects for the test
	std::string spin1_name = "spin1";
	std::string spin1_contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(spin1_name, spin1_contents);

	std::string spin2_name = "spin2";
	std::string spin2_contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(spin2_name, spin2_contents);

	std::string spin3_name = "spin3";
	std::string spin3_contents = "spin=1/2;";
	auto spin3 = std::make_shared<SpinAPI::Spin>(spin3_name, spin3_contents);

	std::string spin4_name = "spin4";
	std::string spin4_contents = "spin=1/2;";
	auto spin4 = std::make_shared<SpinAPI::Spin>(spin4_name, spin4_contents);

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	std::string state_name = "state1";
	std::string state_contents = "spins(spin1,spin2,spin3)=|1/2,1/2,-1/2>-2i|1/2,-1/2,1/2>;";
	SpinAPI::State state(state_name, state_contents);

	SpinAPI::CompleteState cstate;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state.ParseFromSystem(spinsys);
	isCorrect &= state.IsCoupled(spin1);
	isCorrect &= state.IsCoupled(spin2);
	isCorrect &= state.IsCoupled(spin3);
	isCorrect &= !state.IsCoupled(spin4);
	isCorrect &= !state.GetCompleteState(spin4, cstate);
	isCorrect &= state.GetCompleteState(spin2, cstate);
	isCorrect &= cstate[0].second[0].first == 1;
	isCorrect &= cstate[1].second[1].first == -1;
	isCorrect &= cstate[2].second[0].first == -1;
	isCorrect &= cstate[2].second[1].first == 1;
	isCorrect &= equal_double(std::real(cstate[0].second[0].second), 1.0);
	isCorrect &= equal_double(std::imag(cstate[2].second[1].second), -2.0);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the Tensor class for basic functionality
bool test_spinapi_tensorclass_basics()
{
	// Setup objects for the test
	std::string t4str = "anisotropic(1 1 1)+isotropic(4)";
	arma::mat testmat1to4 = arma::mat("5 0 0;0 5 0;0 0 5");
	arma::mat testmat5 = arma::mat("1 2 3;2 5 6;3 6 9");

	SpinAPI::Tensor t1(5.0);
	SpinAPI::Tensor t2(0, 5, 5, 5);
	SpinAPI::Tensor t3(testmat1to4);
	SpinAPI::Tensor t4(t4str);
	SpinAPI::Tensor t5(testmat5);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= equal_matrices(t1.LabFrame(), testmat1to4);
	isCorrect &= equal_matrices(t2.LabFrame(), testmat1to4);
	isCorrect &= equal_matrices(t3.LabFrame(), testmat1to4);
	isCorrect &= equal_matrices(t4.LabFrame(), testmat1to4);
	isCorrect &= equal_matrices(t5.LabFrame(), testmat5);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: The union of all subspaces should give the total space.
bool test_spinapi_subspacefuncs_union()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");
	auto spin5 = std::make_shared<SpinAPI::Spin>("spin5", "spin=1/2;");
	auto spin6 = std::make_shared<SpinAPI::Spin>("spin6", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");					 // Subspace: spins 1, 3 and 4
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=doublespin;group1=spin5;group2=spin6;");						 // Subspace: spins 5 and 6
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=singlespin;group1=spin1,spin2,spin3;group2=spin4,spin5,spin6"); // Should not change anything as it is a single-spin interaction (group2 should be ignored)

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(spin5);
	spinsys.Add(spin6);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.Add(interaction3);
	spinsys.ValidateInteractions();

	auto spaces = SpinAPI::CompleteSubspaces(spinsys);
	std::vector<SpinAPI::spin_ptr> spaces_union;
	spaces_union.reserve(6);

	// Get the union of the subspaces
	for (auto i = spaces.begin(); i != spaces.end(); i++)
	{
		// Make sure that the capacity is large enough
		if (spaces_union.capacity() < spaces_union.size() + i->size())
			spaces_union.reserve(spaces_union.size() + i->size());

		// Insert new elements in the back
		spaces_union.insert(spaces_union.end(), i->begin(), i->end());
	}

	bool isCorrect = true;

	// Perform the test
	isCorrect &= (spaces.size() == 3);		 // There are 3 subspaces: [1,3,4], [2], [5,6]
	isCorrect &= (spaces_union.size() == 6); // We should have 6 spins in the union of the subspaces, [1-6]
	isCorrect &= (std::find(spaces_union.cbegin(), spaces_union.cend(), spin1) != spaces_union.cend());
	isCorrect &= (std::find(spaces_union.cbegin(), spaces_union.cend(), spin2) != spaces_union.cend());
	isCorrect &= (std::find(spaces_union.cbegin(), spaces_union.cend(), spin3) != spaces_union.cend());
	isCorrect &= (std::find(spaces_union.cbegin(), spaces_union.cend(), spin4) != spaces_union.cend());
	isCorrect &= (std::find(spaces_union.cbegin(), spaces_union.cend(), spin5) != spaces_union.cend());
	isCorrect &= (std::find(spaces_union.cbegin(), spaces_union.cend(), spin6) != spaces_union.cend());

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: The intersection of two disjoint subspaces should be the empty set/space.
bool test_spinapi_subspacefuncs_intersections()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");
	auto spin5 = std::make_shared<SpinAPI::Spin>("spin5", "spin=1/2;");
	auto spin6 = std::make_shared<SpinAPI::Spin>("spin6", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");					 // Subspace: spins 1, 3 and 4
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=doublespin;group1=spin5;group2=spin6;");						 // Subspace: spins 5 and 6
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=singlespin;group1=spin1,spin2,spin3;group2=spin4,spin5,spin6"); // Should not change anything as it is a single-spin interaction (group2 should be ignored)

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(spin5);
	spinsys.Add(spin6);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.Add(interaction3);
	spinsys.ValidateInteractions();

	auto spaces = SpinAPI::CompleteSubspaces(spinsys);

	// Make sure we know which subspace is which
	std::vector<SpinAPI::spin_ptr> *set1 = nullptr; // Will contain [1,3,4]
	std::vector<SpinAPI::spin_ptr> *set2 = nullptr; // Will contain [2]
	std::vector<SpinAPI::spin_ptr> *set3 = nullptr; // Will contain [5,6]
	for (auto i = spaces.begin(); i != spaces.end(); i++)
	{
		if (i->size() == 3)
		{
			set1 = &(*i);
		}
		else if (i->size() == 1)
		{
			set2 = &(*i);
		}
		else if (i->size() == 2)
		{
			set3 = &(*i);
		}
	}

	// Make sure that each subspace was found
	// Note that if all subspaces was found, we cannot have any extra subspaces due to the size check below
	if (set1 == nullptr || set2 == nullptr || set3 == nullptr)
		return false;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= (spaces.size() == 3); // There are 3 subspaces: [1,3,4], [2], [5,6]
	isCorrect &= (std::find((*set1).cbegin(), (*set1).cend(), spin1) != (*set1).cend());
	isCorrect &= (std::find((*set2).cbegin(), (*set2).cend(), spin2) != (*set2).cend());
	isCorrect &= (std::find((*set1).cbegin(), (*set1).cend(), spin3) != (*set1).cend());
	isCorrect &= (std::find((*set1).cbegin(), (*set1).cend(), spin4) != (*set1).cend());
	isCorrect &= (std::find((*set3).cbegin(), (*set3).cend(), spin5) != (*set3).cend());
	isCorrect &= (std::find((*set3).cbegin(), (*set3).cend(), spin6) != (*set3).cend());

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: The intersection of two disjoint subspaces should be the empty set/space.
bool test_spinapi_subspacefuncs_intersections2()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("nucleus1", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("nucleus2", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=hyperfine;group1=electron1;group2=nucleus1;tensor=isotropic(5e-4);");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=hyperfine;group1=electron1;group2=nucleus2;tensor=anisotropic(1e-4, 1e-4, 1e-3);");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.ValidateInteractions();

	auto spaces = SpinAPI::CompleteSubspaces(spinsys);

	// Make sure we know which subspace is which
	std::vector<SpinAPI::spin_ptr> *set1 = nullptr; // Will contain [1,3,4]
	std::vector<SpinAPI::spin_ptr> *set2 = nullptr; // Will contain [2]
	for (auto i = spaces.begin(); i != spaces.end(); i++)
	{
		if (i->size() == 3)
		{
			set1 = &(*i);
		}
		else if (i->size() == 1)
		{
			set2 = &(*i);
		}
	}

	// Make sure that each subspace was found
	// Note that if all subspaces was found, we cannot have any extra subspaces due to the size check below
	if (set1 == nullptr || set2 == nullptr)
		return false;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= (spaces.size() == 2); // There are 3 subspaces: [1,3,4], [2]
	isCorrect &= (std::find((*set1).cbegin(), (*set1).cend(), spin1) != (*set1).cend());
	isCorrect &= (std::find((*set2).cbegin(), (*set2).cend(), spin2) != (*set2).cend());
	isCorrect &= (std::find((*set1).cbegin(), (*set1).cend(), spin3) != (*set1).cend());
	isCorrect &= (std::find((*set1).cbegin(), (*set1).cend(), spin4) != (*set1).cend());

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: Interactions can extend a subspace.
bool test_spinapi_subspacefuncs_extendbyinteraction()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;"); // Subspace: spins 1, 3 and 4

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.ValidateInteractions();

	std::vector<SpinAPI::spin_ptr> space;
	space.push_back(spin1);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= (space.size() == 1);
	isCorrect &= interaction1->CompleteSet(space);
	isCorrect &= (space.size() == 3);
	isCorrect &= !interaction1->CompleteSet(space);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: Transitions can extend a subspace.
bool test_spinapi_subspacefuncs_extendbytransition()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	auto state = std::make_shared<SpinAPI::State>("state1", "spins(spin1,spin2,spin3)=|1/2,1/2,-1/2>-2i|1/2,-1/2,1/2>;");
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "sourcestate=state1;rate=1;", spinsys);

	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state);
	spinsys->Add(transition1);

	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	std::vector<SpinAPI::spin_ptr> space;
	space.push_back(spin1);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state->ParseFromSystem(*spinsys);
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0);
	isCorrect &= (space.size() == 1);
	isCorrect &= transition1->CompleteSet(space);
	isCorrect &= (space.size() == 3);
	isCorrect &= !transition1->CompleteSet(space);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: States can extend a subspace.
bool test_spinapi_subspacefuncs_extendbystate()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	SpinAPI::State state("state1", "spins(spin1,spin2,spin3)=|1/2,1/2,-1/2>-2i|1/2,-1/2,1/2>;");

	std::vector<SpinAPI::spin_ptr> space;
	space.push_back(spin1);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state.ParseFromSystem(spinsys);
	isCorrect &= (space.size() == 1);
	isCorrect &= state.CompleteSet(space);
	isCorrect &= (space.size() == 3);
	isCorrect &= !state.CompleteSet(space);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the subspace management functionality.
// Test: SpinSystems can extend a subspace.
bool test_spinapi_subspacefuncs_extendbyspinsys()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");
	auto spin5 = std::make_shared<SpinAPI::Spin>("spin5", "spin=1/2;");
	auto spin6 = std::make_shared<SpinAPI::Spin>("spin6", "spin=1/2;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	auto state = std::make_shared<SpinAPI::State>("state1", "spins(spin1,spin2,spin3)=|1/2,1/2,-1/2>-2i|1/2,-1/2,1/2>;"); // Couple spins: 1, 2, 3
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "sourcestate=state1;rate=1;", spinsys);
	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1;group2=spin5;"); // Couple spins: 1, 5

	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(spin5);
	spinsys->Add(spin6);
	spinsys->Add(state);
	spinsys->Add(transition1);
	spinsys->Add(interaction1);
	spinsys->ValidateInteractions();

	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	std::vector<SpinAPI::spin_ptr> space;
	space.push_back(spin1);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state->ParseFromSystem(*spinsys);
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0);
	isCorrect &= (space.size() == 1);
	isCorrect &= spinsys->CompleteSet(space);
	isCorrect &= (space.size() == 4); // Spins: 1, 2, 3, 5
	isCorrect &= !spinsys->CompleteSet(space);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the CreateOperator method.
bool test_spinapi_spinspace_sparsevsdense_createoperator()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=zeeman;group1=spin1,spin2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat denseM;
	arma::sp_cx_mat sparseM;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin1->Sx()), spin1, denseM);
	isCorrect &= space.CreateOperator(spin1->Sx(), spin1, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin2->Sy()), spin2, denseM);
	isCorrect &= space.CreateOperator(spin2->Sy(), spin2, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin3->Sz()), spin3, denseM);
	isCorrect &= space.CreateOperator(spin3->Sz(), spin3, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the Hamiltonian method.
bool test_spinapi_spinspace_sparsevsdense_hamiltonian()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=zeeman;group1=spin1,spin2;field=0.1 0.2 0.3;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat denseM;
	arma::sp_cx_mat sparseM;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.Hamiltonian(denseM);
	isCorrect &= space.Hamiltonian(sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	space.UseSuperoperatorSpace(true);
	isCorrect &= space.Hamiltonian(denseM);
	isCorrect &= space.Hamiltonian(sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the StaticHamiltonian method.
bool test_spinapi_spinspace_sparsevsdense_statichamiltonian()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=zeeman;group1=spin1,spin2;field=0.1 0.2 0.3;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=singlespin;group1=spin1,spin3;field=1 0 0;timedependence=oscillating;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=singlespin;group1=spin4,spin2;field=0 1 0;timedependence=circularpolarization;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.Add(interaction3);
	spinsys.Add(interaction4);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat denseM;
	arma::sp_cx_mat sparseM;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.StaticHamiltonian(denseM);
	isCorrect &= space.StaticHamiltonian(sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	space.UseSuperoperatorSpace(true);
	isCorrect &= space.StaticHamiltonian(denseM);
	isCorrect &= space.StaticHamiltonian(sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the DynamicHamiltonian method.
bool test_spinapi_spinspace_sparsevsdense_dynamichamiltonian()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=zeeman;group1=spin1,spin2;field=0.1 0.2 0.3;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=singlespin;group1=spin1,spin3;field=1 0 0;timedependence=oscillating;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=singlespin;group1=spin4,spin2;field=0 1 0;timedependence=circularpolarization;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.Add(interaction3);
	spinsys.Add(interaction4);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat denseM;
	arma::sp_cx_mat sparseM;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.DynamicHamiltonian(denseM);
	isCorrect &= space.DynamicHamiltonian(sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	space.UseSuperoperatorSpace(true);
	isCorrect &= space.DynamicHamiltonian(denseM);
	isCorrect &= space.DynamicHamiltonian(sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the InteractionOperator method.
bool test_spinapi_spinspace_sparsevsdense_interactionoperator()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=doublespin;group1=spin1,spin3;group2=spin4;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=zeeman;group1=spin1,spin2;field=0.1 0.2 0.3;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=singlespin;group1=spin1,spin3;field=1 0 0;timedependence=oscillating;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);
	spinsys.Add(interaction1);
	spinsys.Add(interaction2);
	spinsys.Add(interaction3);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat denseM;
	arma::sp_cx_mat sparseM;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.InteractionOperator(interaction1, denseM);
	isCorrect &= space.InteractionOperator(interaction1, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	isCorrect &= space.InteractionOperator(interaction2, denseM);
	isCorrect &= space.InteractionOperator(interaction2, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	isCorrect &= space.InteractionOperator(interaction3, denseM);
	isCorrect &= space.InteractionOperator(interaction3, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	space.UseSuperoperatorSpace(true);

	isCorrect &= space.InteractionOperator(interaction1, denseM);
	isCorrect &= space.InteractionOperator(interaction1, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	isCorrect &= space.InteractionOperator(interaction2, denseM);
	isCorrect &= space.InteractionOperator(interaction2, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);
	isCorrect &= space.InteractionOperator(interaction3, denseM);
	isCorrect &= space.InteractionOperator(interaction3, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the OperatorToSuperspace method.
bool test_spinapi_spinspace_sparsevsdense_operatortosuperspace()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	SpinAPI::SpinSpace space(spinsys);

	arma::sp_cx_mat tmp;
	arma::sp_cx_mat sparseM_HS;
	if (!space.CreateOperator(spin1->Sx(), spin1, sparseM_HS))
	{
		return false;
	}
	if (!space.CreateOperator(spin1->Sy(), spin2, tmp))
	{
		return false;
	}
	sparseM_HS += tmp;
	if (!space.CreateOperator(spin1->Sz(), spin3, tmp))
	{
		return false;
	}
	sparseM_HS += tmp;
	arma::cx_mat denseM_HS = arma::conv_to<arma::cx_mat>::from(sparseM_HS);

	arma::cx_vec sparseM_SSvec;
	arma::cx_vec denseM_SSvec;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.OperatorToSuperspace(sparseM_HS, sparseM_SSvec);
	isCorrect &= space.OperatorToSuperspace(denseM_HS, denseM_SSvec);
	isCorrect &= equal_vec(sparseM_SSvec, denseM_SSvec);
	isCorrect &= (denseM_SSvec.n_elem == denseM_HS.n_rows * denseM_HS.n_rows);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the OperatorFromSuperspace method.
bool test_spinapi_spinspace_sparsevsdense_operatorfromsuperspace()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	SpinAPI::SpinSpace space(spinsys);

	arma::sp_cx_mat tmp;
	arma::sp_cx_mat HSMat;
	if (!space.CreateOperator(spin1->Sx(), spin1, HSMat))
	{
		return false;
	}
	if (!space.CreateOperator(spin1->Sy(), spin2, tmp))
	{
		return false;
	}
	HSMat += tmp;
	if (!space.CreateOperator(spin1->Sz(), spin3, tmp))
	{
		return false;
	}
	HSMat += tmp;
	arma::cx_vec SSVec;
	space.OperatorToSuperspace(HSMat, SSVec);

	arma::sp_cx_mat sparseM_SSvec;
	arma::cx_mat denseM_SSvec;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.OperatorFromSuperspace(SSVec, sparseM_SSvec);
	isCorrect &= space.OperatorFromSuperspace(SSVec, denseM_SSvec);
	isCorrect &= equal_matrices(sparseM_SSvec, denseM_SSvec);
	isCorrect &= (SSVec.n_elem == denseM_SSvec.n_rows * denseM_SSvec.n_rows);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the SuperoperatorFromOperators method.
bool test_spinapi_spinspace_sparsevsdense_superoperatorfromoperators()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	SpinAPI::SpinSpace space(spinsys);

	arma::sp_cx_mat sparseM_HS1;
	arma::sp_cx_mat sparseM_HS2;
	if (!space.CreateOperator(spin1->Sx(), spin1, sparseM_HS1))
	{
		return false;
	}
	if (!space.CreateOperator(spin2->Sy(), spin2, sparseM_HS2))
	{
		return false;
	}
	arma::cx_mat denseM_HS1 = arma::conv_to<arma::cx_mat>::from(sparseM_HS1);
	arma::cx_mat denseM_HS2 = arma::conv_to<arma::cx_mat>::from(sparseM_HS2);

	arma::sp_cx_mat sparseM_SS;
	arma::cx_mat denseM_SS;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.SuperoperatorFromOperators(sparseM_HS1, sparseM_HS2, sparseM_SS);
	isCorrect &= space.SuperoperatorFromOperators(denseM_HS1, denseM_HS2, denseM_SS);
	isCorrect &= equal_matrices(sparseM_SS, denseM_SS);
	isCorrect &= (denseM_SS.n_rows == denseM_HS1.n_rows * denseM_HS2.n_rows);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the SuperoperatorFromLeftOperator method.
bool test_spinapi_spinspace_sparsevsdense_superoperatorfromleftoperator()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	SpinAPI::SpinSpace space(spinsys);

	arma::sp_cx_mat tmp;
	arma::sp_cx_mat sparseM_HS;
	if (!space.CreateOperator(spin1->Sx(), spin1, sparseM_HS))
	{
		return false;
	}
	if (!space.CreateOperator(spin1->Sy(), spin2, tmp))
	{
		return false;
	}
	sparseM_HS += tmp;
	if (!space.CreateOperator(spin1->Sz(), spin3, tmp))
	{
		return false;
	}
	sparseM_HS += tmp;
	arma::cx_mat denseM_HS = arma::conv_to<arma::cx_mat>::from(sparseM_HS);

	arma::sp_cx_mat sparseM_SS;
	arma::cx_mat denseM_SS;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.SuperoperatorFromLeftOperator(sparseM_HS, sparseM_SS);
	isCorrect &= space.SuperoperatorFromLeftOperator(denseM_HS, denseM_SS);
	isCorrect &= equal_matrices(sparseM_SS, denseM_SS);
	isCorrect &= (denseM_SS.n_rows == denseM_HS.n_rows * denseM_HS.n_rows);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the SuperoperatorFromRightOperator method.
bool test_spinapi_spinspace_sparsevsdense_superoperatorfromrightoperator()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);
	spinsys.Add(spin3);
	spinsys.Add(spin4);

	SpinAPI::SpinSpace space(spinsys);

	arma::sp_cx_mat tmp;
	arma::sp_cx_mat sparseM_HS;
	if (!space.CreateOperator(spin1->Sx(), spin1, sparseM_HS))
	{
		return false;
	}
	if (!space.CreateOperator(spin1->Sy(), spin2, tmp))
	{
		return false;
	}
	sparseM_HS += tmp;
	if (!space.CreateOperator(spin1->Sz(), spin3, tmp))
	{
		return false;
	}
	sparseM_HS += tmp;
	arma::cx_mat denseM_HS = arma::conv_to<arma::cx_mat>::from(sparseM_HS);

	arma::sp_cx_mat sparseM_SS;
	arma::cx_mat denseM_SS;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.SuperoperatorFromRightOperator(sparseM_HS, sparseM_SS);
	isCorrect &= space.SuperoperatorFromRightOperator(denseM_HS, denseM_SS);
	isCorrect &= equal_matrices(sparseM_SS, denseM_SS);
	isCorrect &= (denseM_SS.n_rows == denseM_HS.n_rows * denseM_HS.n_rows);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the sparse matrices generated by the SpinSpace class
// Test: Tests the ReactionOperator method.
bool test_spinapi_spinspace_sparsevsdense_reactionoperator()
{
	// Setup objects for the test
	auto spin1 = std::make_shared<SpinAPI::Spin>("spin1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("spin2", "spin=1/2;");
	auto spin3 = std::make_shared<SpinAPI::Spin>("spin3", "spin=1/2;");
	auto spin4 = std::make_shared<SpinAPI::Spin>("spin4", "spin=1/2;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	auto state = std::make_shared<SpinAPI::State>("state1", "spins(spin1,spin2,spin3)=|1/2,1/2,-1/2>-2i|1/2,-1/2,1/2>;");
	auto transition = std::make_shared<SpinAPI::Transition>("transition1", "sourcestate=state1;rate=1;", spinsys);

	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state);
	spinsys->Add(transition);

	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	SpinAPI::SpinSpace space(spinsys);

	arma::cx_mat denseM;
	arma::sp_cx_mat sparseM;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state->ParseFromSystem(*spinsys);
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0);
	isCorrect &= space.ReactionOperator(transition, denseM);
	isCorrect &= space.ReactionOperator(transition, sparseM);
	isCorrect &= equal_matrices(denseM, sparseM);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the SpinSpace reordering method for dense matrices
// DEPENDENCY NOTE: ObjectParser, Spin
bool test_spinapi_reorderbasis_densematrix()
{
	// Setup objects for the test
	std::string spin1_name = "spin1";
	std::string spin1_contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(spin1_name, spin1_contents);

	std::string spin2_name = "spin2";
	std::string spin2_contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(spin2_name, spin2_contents);

	std::string spin3_name = "spin3";
	std::string spin3_contents = "spin=1;";
	auto spin3 = std::make_shared<SpinAPI::Spin>(spin3_name, spin3_contents);

	std::vector<SpinAPI::spin_ptr> basis1;
	basis1.push_back(spin1);
	basis1.push_back(spin2);
	basis1.push_back(spin3);

	std::vector<SpinAPI::spin_ptr> basis2;
	basis2.push_back(spin3);
	basis2.push_back(spin1);
	basis2.push_back(spin2);

	SpinAPI::SpinSpace space1(basis1);
	SpinAPI::SpinSpace space2(basis2);
	arma::cx_mat A;
	arma::cx_mat B;
	arma::cx_mat O = arma::conv_to<arma::cx_mat>::from(spin1->Sx());

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space1.CreateOperator(O, spin1, A);
	isCorrect &= space2.CreateOperator(O, spin1, B);
	isCorrect &= !equal_matrices(A, B);
	isCorrect &= space1.ReorderBasis(B, basis2);
	isCorrect &= equal_matrices(A, B);
	isCorrect &= space2.CreateOperator(O, spin1, B);
	isCorrect &= space2.ReorderBasis(A, basis1, basis2);
	isCorrect &= equal_matrices(A, B);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the SpinSpace reordering method for sparse matrices
// DEPENDENCY NOTE: ObjectParser, Spin
bool test_spinapi_reorderbasis_sparsematrix()
{
	// Setup objects for the test
	std::string spin1_name = "spin1";
	std::string spin1_contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(spin1_name, spin1_contents);

	std::string spin2_name = "spin2";
	std::string spin2_contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(spin2_name, spin2_contents);

	std::string spin3_name = "spin3";
	std::string spin3_contents = "spin=1;";
	auto spin3 = std::make_shared<SpinAPI::Spin>(spin3_name, spin3_contents);

	std::vector<SpinAPI::spin_ptr> basis1;
	basis1.push_back(spin1);
	basis1.push_back(spin2);
	basis1.push_back(spin3);

	std::vector<SpinAPI::spin_ptr> basis2;
	basis2.push_back(spin3);
	basis2.push_back(spin1);
	basis2.push_back(spin2);

	SpinAPI::SpinSpace space1(basis1);
	SpinAPI::SpinSpace space2(basis2);
	arma::sp_cx_mat A;
	arma::sp_cx_mat B;
	arma::sp_cx_mat O = spin1->Sx();

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space1.CreateOperator(O, spin1, A);
	isCorrect &= space2.CreateOperator(O, spin1, B);
	isCorrect &= !equal_matrices(A, B);
	isCorrect &= space1.ReorderBasis(B, basis2);
	isCorrect &= equal_matrices(A, B);
	isCorrect &= space2.CreateOperator(O, spin1, B);
	isCorrect &= space2.ReorderBasis(A, basis1, basis2);
	isCorrect &= equal_matrices(A, B);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the SpinSpace spin management methods
// DEPENDENCY NOTE: ObjectParser, Spin
bool test_spinapi_spinspace_spinmanagement1()
{
	// Setup objects for the test
	std::string spin1_name = "spin1";
	std::string spin1_contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(spin1_name, spin1_contents);

	std::string spin2_name = "spin2";
	std::string spin2_contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(spin2_name, spin2_contents);

	std::string spin3_name = "spin3";
	std::string spin3_contents = "spin=1;";
	auto spin3 = std::make_shared<SpinAPI::Spin>(spin3_name, spin3_contents);

	SpinAPI::SpinSpace space;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.Add(spin1);
	isCorrect &= space.Add(spin2);
	isCorrect &= space.Contains(spin1);
	isCorrect &= space.Contains(spin2);
	isCorrect &= !space.Contains(spin3);
	isCorrect &= !space.Add(spin2);
	isCorrect &= space.Add(spin3);
	isCorrect &= space.Contains(spin3);
	isCorrect &= space.Remove(spin1);
	isCorrect &= !space.Contains(spin1);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the SpinSpace spin management methods
// DEPENDENCY NOTE: ObjectParser, Spin
bool test_spinapi_spinspace_spinmanagement2()
{
	// Setup objects for the test
	std::string spin1_name = "spin1";
	std::string spin1_contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(spin1_name, spin1_contents);

	std::string spin2_name = "spin2";
	std::string spin2_contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(spin2_name, spin2_contents);

	std::string spin3_name = "spin3";
	std::string spin3_contents = "spin=1;";
	auto spin3 = std::make_shared<SpinAPI::Spin>(spin3_name, spin3_contents);

	SpinAPI::SpinSpace space;

	std::vector<SpinAPI::spin_ptr> v1;
	v1.push_back(spin1);
	v1.push_back(spin2);
	v1.push_back(spin3);

	std::vector<SpinAPI::spin_ptr> v2;
	v2.push_back(spin1);
	v2.push_back(spin2);

	std::vector<SpinAPI::spin_ptr> v3;
	v3.push_back(spin3);
	v3.push_back(spin2);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= space.Add(v1);
	isCorrect &= space.Contains(v2);
	isCorrect &= space.Contains(spin1);
	space.ClearSpins();
	isCorrect &= !space.Contains(v2);
	isCorrect &= space.Add(v2);
	isCorrect &= space.Add(v3);
	isCorrect &= space.Contains(v1);
	isCorrect &= space.Remove(v2);
	isCorrect &= !space.Contains(v1);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
bool test_function_finding()
{
	// Setup objects for the test
	std::string sp1 = "spin1";
	std::string sp1Contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(sp1, sp1Contents);

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	//
	std::string state_name = "TestState";
	std::string state_contents = "x=3.14159265;spins(spin1)=cos(0.5x)|1/2>;";
	SpinAPI::State state(state_name, state_contents);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state.ParseFromSystem(spinsys);
	// auto func = state.GetFunctions()[0];
	// auto str = func->GetFunctionString();

	// if(str.compare("cos(0.5x)") == 0)
	//{
	//	isCorrect &= true;
	// }

	return isCorrect;
	// return true;
}
//////////////////////////////////////////////////////////////////////////////
bool test_state_function_grouped_superposition()
{
	std::string sp1 = "NDI";
	std::string sp1Contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(sp1, sp1Contents);

	std::string sp2 = "PXX";
	std::string sp2Contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(sp2, sp2Contents);

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);

	std::string state_name = "STmix";
	std::string state_contents =
		"a=0.7853981633974483;"
		"spins(NDI,PXX)="
		"   cos(a)( |1/2,-1/2> - |-1/2,1/2> )"
		" + sin(a)( |-1/2,-1/2> - |1/2,1/2> );";
	SpinAPI::State state(state_name, state_contents);

	SpinAPI::SpinSpace space;
	space.Add(spin1);
	space.Add(spin2);

	arma::cx_vec parsed;
	arma::cx_vec expected(4, arma::fill::zeros);
	const double a = 0.7853981633974483;
	const double invNorm = 1.0 / std::sqrt(2.0);
	expected(0) = -std::sin(a) * invNorm; // | 1/2,  1/2>
	expected(1) =  std::cos(a) * invNorm; // | 1/2, -1/2>
	expected(2) = -std::cos(a) * invNorm; // |-1/2,  1/2>
	expected(3) =  std::sin(a) * invNorm; // |-1/2, -1/2>

	bool isCorrect = true;
	bool parsedState = state.ParseFromSystem(spinsys);
	bool gotState = space.GetState(std::make_shared<SpinAPI::State>(state), parsed);
	isCorrect &= parsedState;
	isCorrect &= gotState;
	isCorrect &= equal_vec(parsed, expected, 1.0e-10);

	std::string exp_state_contents = "spins(NDI)=exp(0)|1/2>;";
	SpinAPI::State exp_state("ExpState", exp_state_contents);
	SpinAPI::SpinSpace oneSpinSpace;
	oneSpinSpace.Add(spin1);
	arma::cx_vec expParsed;
	arma::cx_vec expExpected(2, arma::fill::zeros);
	expExpected(0) = 1.0;
	isCorrect &= exp_state.ParseFromSystem(spinsys);
	isCorrect &= oneSpinSpace.GetState(std::make_shared<SpinAPI::State>(exp_state), expParsed);
	isCorrect &= equal_vec(expParsed, expExpected, 1.0e-10);

	std::string direct_phase_contents =
		"a=0.7853981633974483;"
		"spins(NDI,PXX)=cos(a)|1/2,-1/2>+I*sin(a)|-1/2,1/2>;";
	SpinAPI::State direct_phase_state("DirectPhaseState", direct_phase_contents);
	arma::cx_vec directPhaseParsed;
	arma::cx_vec directPhaseExpected(4, arma::fill::zeros);
	directPhaseExpected(1) = std::cos(a);
	directPhaseExpected(2) = arma::cx_double(0.0, std::sin(a));
	isCorrect &= direct_phase_state.ParseFromSystem(spinsys);
	isCorrect &= space.GetState(std::make_shared<SpinAPI::State>(direct_phase_state), directPhaseParsed);
	isCorrect &= equal_vec(directPhaseParsed, directPhaseExpected, 1.0e-10);

	std::string grouped_phase_contents =
		"a=0.7853981633974483;"
		"spins(NDI,PXX)=cos(a)(|1/2,-1/2>)+i*sin(a)(|-1/2,1/2>);";
	SpinAPI::State grouped_phase_state("GroupedPhaseState", grouped_phase_contents);
	arma::cx_vec groupedPhaseParsed;
	isCorrect &= grouped_phase_state.ParseFromSystem(spinsys);
	isCorrect &= space.GetState(std::make_shared<SpinAPI::State>(grouped_phase_state), groupedPhaseParsed);
	isCorrect &= equal_vec(groupedPhaseParsed, directPhaseExpected, 1.0e-10);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
bool test_state_easyspin_ciss_density_convention()
{
	std::string sp1 = "BDX";
	std::string sp1Contents = "spin=1/2;";
	auto spin1 = std::make_shared<SpinAPI::Spin>(sp1, sp1Contents);

	std::string sp2 = "NDI";
	std::string sp2Contents = "spin=1/2;";
	auto spin2 = std::make_shared<SpinAPI::Spin>(sp2, sp2Contents);

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);

	SpinAPI::SpinSpace space;
	space.Add(spin1);
	space.Add(spin2);

	const double chi = 1.5707963267948966;
	const double s = std::sin(0.5 * chi);
	const double c = std::cos(0.5 * chi);
	const double invsqrt2 = 1.0 / std::sqrt(2.0);

	// EasySpin's Sys.initState = {pc'*pc,'coupled'} with
	// pc = [0 1i*sin(chi/2) 0 cos(chi/2)] produces rho(T0,S) = -i*s*c.
	arma::cx_mat rhoCoupledExpected(4, 4, arma::fill::zeros);
	rhoCoupledExpected(1, 1) = s * s;
	rhoCoupledExpected(1, 3) = arma::cx_double(0.0, -s * c);
	rhoCoupledExpected(3, 1) = arma::cx_double(0.0, s * c);
	rhoCoupledExpected(3, 3) = c * c;

	arma::cx_mat uncoupledToCoupled(4, 4, arma::fill::zeros);
	uncoupledToCoupled(0, 0) = 1.0;		   // T+ = |alpha alpha>
	uncoupledToCoupled(1, 1) = invsqrt2;   // T0 = (|alpha beta> + |beta alpha>)/sqrt(2)
	uncoupledToCoupled(1, 2) = invsqrt2;
	uncoupledToCoupled(2, 3) = 1.0;		   // T- = |beta beta>
	uncoupledToCoupled(3, 1) = invsqrt2;   // S = (|alpha beta> - |beta alpha>)/sqrt(2)
	uncoupledToCoupled(3, 2) = -invsqrt2;

	auto molspinDensityInEasySpinCoupledBasis = [&](const std::string &contents, arma::cx_mat &rhoCoupled) {
		SpinAPI::State state("CISS", contents);
		arma::cx_mat rhoUncoupled;
		if (!state.ParseFromSystem(spinsys) ||
			!space.GetState(std::make_shared<SpinAPI::State>(state), rhoUncoupled))
		{
			return false;
		}
		rhoCoupled = uncoupledToCoupled * rhoUncoupled * uncoupledToCoupled.t();
		return true;
	};

	arma::cx_mat rhoCoupledSameOrder;
	std::string sameOrderContents =
		"chi=1.5707963267948966;"
		"spins(BDX,NDI)="
		"  cos(0.5*chi)(|1/2,-1/2> - |-1/2,1/2>)"
		" -I* sin(0.5*chi)(|1/2,-1/2> + |-1/2,1/2>);";

	arma::cx_mat rhoCoupledSwappedOrder;
	std::string swappedOrderContents =
		"chi=1.5707963267948966;"
		"spins(NDI,BDX)="
		"  cos(0.5*chi)(|1/2,-1/2> - |-1/2,1/2>)"
		" +I* sin(0.5*chi)(|1/2,-1/2> + |-1/2,1/2>);";

	bool isCorrect = true;
	isCorrect &= molspinDensityInEasySpinCoupledBasis(sameOrderContents, rhoCoupledSameOrder);
	isCorrect &= molspinDensityInEasySpinCoupledBasis(swappedOrderContents, rhoCoupledSwappedOrder);
	isCorrect &= equal_matrices(rhoCoupledSameOrder, rhoCoupledExpected, 1.0e-10);
	isCorrect &= equal_matrices(rhoCoupledSwappedOrder, rhoCoupledExpected, 1.0e-10);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
bool test_function_evaluation()
{
	// Setup objects for the test
	std::string function = "cos";
	std::string contents = "0.5x*x+y*y+c";
	auto TestFunc = SpinAPI::FunctionParser(function, contents);

	arma::cx_double val1 = {-0.1634667676, 0};
	arma::cx_double val2 = {0.1403316058, 0};
	arma::cx_double val3 = {0.3010526538, 0};
	double tolerance = 1e-5;

	bool isCorrect = true;

	double d1 = 0.5;
	double d2 = 1.1;
	double d3 = 0.4;
	void *v1 = (void *)&d1;
	void *v2 = (void *)&d2;
	void *v3 = (void *)&d3;

	arma::cx_double val = TestFunc->operator()({v1, v2, v3});
	// std::cout << val << std::endl;
	isCorrect &= (std::abs(val.real() - val1.real()) < tolerance);
	val = TestFunc->operator()({v3, v1, v2});
	isCorrect &= (std::abs(val.real() - val2.real()) < tolerance);
	val = TestFunc->operator()({v2, v3, v1});
	isCorrect &= (std::abs(val.real() - val3.real()) < tolerance);

	return isCorrect;
}

// Tests an Pulse object with an instantpulse type.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_instantpulse()
{
	// Setup objects for the test
	std::string name = "pulse1";
	std::string contents = "type=instantpulse;group=RPElectron1;rotationaxis=1 1 1;angle=42.24;";
	SpinAPI::Pulse P(name, contents);

	auto rotationaxis = arma::vec("1 1 1") / arma::norm(arma::vec("1 1 1"));
	double angle = 42.24;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= P.Type() == SpinAPI::PulseType::InstantPulse;
	isCorrect &= equal_vec(P.Rotationaxis(), rotationaxis);
	isCorrect &= equal_double(P.Angle(), angle);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests an Pulse object with an longpulsestaticfield type.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_longpulsestaticfield()
{
	// Setup objects for the test
	std::string name = "pulse2";
	std::string contents = "type=longpulsestaticfield;group=RPElectron1;field=0.0 7.1 14.2;pulsetime=42.24;prefactorlist=-176.085;commonprefactorlist=false;ignoretensorslist=true;timestep=0.42;";
	SpinAPI::Pulse P(name, contents);

	auto field = arma::vec("0.0 7.1 14.2");
	double pulsetime = 42.24;
	auto prefactorlist = arma::vec("-176.085");
	std::vector<bool> commonprefactorlist{0};
	std::vector<bool> ignortensorslist{1};
	double timestep = 0.42;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= P.Type() == SpinAPI::PulseType::LongPulseStaticField;
	isCorrect &= equal_vec(P.Field(), field);
	isCorrect &= equal_double(P.Pulsetime(), pulsetime);
	isCorrect &= equal_vec(P.PrefactorList(), prefactorlist);
	for (auto i = 0; i < (int)commonprefactorlist.size(); i++)
	{
		isCorrect &= commonprefactorlist[i] == P.AddCommonPrefactorList()[i];
	}
	for (auto i = 0; i < (int)ignortensorslist.size(); i++)
	{
		isCorrect &= ignortensorslist[i] == P.IgnoreTensorsList()[i];
	}
	isCorrect &= equal_double(P.Timestep(), timestep);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests an Pulse object with an longpulsed type.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_longpulse()
{
	// Setup objects for the test
	std::string name = "pulse3";
	std::string contents = "type=longpulse;group=RPElectron1;field=0.0 7.1 14.2;pulsetime=42.24;prefactorlist=-176.085;commonprefactorlist=false;ignoretensorslist=true;timestep=0.42;frequency=0.0004224;";
	SpinAPI::Pulse P(name, contents);

	auto field = arma::vec("0.0 7.1 14.2");
	double pulsetime = 42.24;
	auto prefactorlist = arma::vec("-176.085");
	std::vector<bool> commonprefactorlist{0};
	std::vector<bool> ignortensorslist{1};
	double timestep = 0.42;
	double frequency = 0.0004224;

	bool isCorrect = true;

	// Perform the test
	isCorrect &= P.Type() == SpinAPI::PulseType::LongPulse;
	isCorrect &= equal_vec(P.Field(), field);
	isCorrect &= equal_double(P.Pulsetime(), pulsetime);
	isCorrect &= equal_vec(P.PrefactorList(), prefactorlist);
	for (auto i = 0; i < (int)commonprefactorlist.size(); i++)
	{
		isCorrect &= commonprefactorlist[i] == P.AddCommonPrefactorList()[i];
	}
	for (auto i = 0; i < (int)ignortensorslist.size(); i++)
	{
		isCorrect &= ignortensorslist[i] == P.IgnoreTensorsList()[i];
	}
	isCorrect &= equal_double(P.Timestep(), timestep);
	isCorrect &= equal_double(P.Frequency(), frequency);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Named SpinAPI objects are case-sensitive. Pulse spin-group parsing must
// therefore preserve the spelling used by the corresponding Spin object.
bool test_spinapi_pulse_group_preserves_spin_name_case()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	SpinAPI::Pulse pulse(
		"pulse",
		"type=longpulsestaticfield;group= E ;field=0 0 1;pulsetime=1;"
		"prefactorlist=1;commonprefactorlist=true;ignoretensorslist=true;");

	std::vector<SpinAPI::spin_ptr> spins{spin};
	bool isCorrect = pulse.ParseSpinGroups(spins);
	const auto group = pulse.Group();
	isCorrect &= (group.size() == 1);
	if (group.size() == 1)
		isCorrect &= (group.front() == spin);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// PulseSequence resolves object names using the same case-sensitive naming
// convention as SpinSystem::*_find().
bool test_spinapi_pulse_sequence_preserves_object_name_case()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	auto pulse = std::make_shared<SpinAPI::Pulse>(
		"CW",
		"type=longpulsestaticfield;group=E;field=0 0 1;pulsetime=1;"
		"prefactorlist=1;commonprefactorlist=true;ignoretensorslist=true;");
	std::vector<SpinAPI::spin_ptr> spins{spin};
	if (!pulse->ParseSpinGroups(spins))
		return false;

	SpinAPI::PulseSequence sequence("sequence", "tau=1;offset=2;sequence=CW,tau;");
	std::vector<SpinAPI::pulse_ptr> pulses{pulse};
	std::vector<SpinAPI::interaction_ptr> interactions;
	std::vector<SpinAPI::transition_ptr> transitions;

	if (!sequence.ParsePulseSequence(pulses, interactions, transitions) ||
		!sequence.IsValid() ||
		sequence.size() != 1)
	{
		return false;
	}

	SpinAPI::PulseSequence copied(sequence);
	SpinAPI::PulseSequence assigned("assigned", "offset=0;sequence=CW,tau;tau=1;");
	assigned = sequence;
	return equal_double(copied.Get_offset(), 2.0) &&
		   equal_double(assigned.Get_offset(), 2.0);
}
//////////////////////////////////////////////////////////////////////////////

// In a multi-system propagation, inactive sequences must not shift the output
// slot used by a later active sequence.
bool test_spinapi_pulse_sequence_operator_preserves_system_index()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	auto pulse = std::make_shared<SpinAPI::Pulse>(
		"CW",
		"type=longpulsestaticfield;group=E;field=1 0 0;pulsetime=1;"
		"prefactorlist=1,1,1;commonprefactorlist=false;ignoretensorslist=true;");
	std::vector<SpinAPI::spin_ptr> spins{spin};
	if (!pulse->ParseSpinGroups(spins))
		return false;

	std::vector<SpinAPI::pulse_ptr> pulses{pulse};
	std::vector<SpinAPI::interaction_ptr> interactions;
	std::vector<SpinAPI::transition_ptr> transitions;
	auto inactive = std::make_shared<SpinAPI::PulseSequence>(
		"inactive", "tau=0;offset=5;sequence=CW,tau;");
	auto active = std::make_shared<SpinAPI::PulseSequence>(
		"active", "tau=0;offset=0;sequence=CW,tau;");
	if (!inactive->ParsePulseSequence(pulses, interactions, transitions) ||
		!active->ParsePulseSequence(pulses, interactions, transitions))
	{
		return false;
	}

	auto inactiveSpace = std::make_shared<SpinAPI::SpinSpace>(spin);
	auto activeSpace = std::make_shared<SpinAPI::SpinSpace>(spin);
	inactiveSpace->UseSuperoperatorSpace(true);
	activeSpace->UseSuperoperatorSpace(true);
	arma::cx_vec rho(activeSpace->SpaceDimensions(), arma::fill::zeros);

	auto operators = SpinAPI::GetPulseOperator(
		{{inactive, inactiveSpace}, {active, activeSpace}}, rho, 0.0);
	return operators.size() == 2 &&
		   arma::norm(operators[0], "fro") < 1e-14 &&
		   arma::norm(operators[1], "fro") > 1e-8;
}
//////////////////////////////////////////////////////////////////////////////

// Tests an Interaction object with a broadband time-dependent field.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_field_broadband()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=singlespin;fieldtype=broadband;autoseed=false;seed=1;field=0 0 0;minfreq=0.1e+6;maxfreq=0.2e+6;components=100;randomorientations=false;";
	SpinAPI::Interaction I(name, contents);

	auto testfield1 = arma::vec("0 0 0");
	double testtime = 1.0;

	bool isCorrect = true;

	// Perform the test - with zero standard deviation the field vector should not change with time
	isCorrect &= equal_vec(I.Field(), testfield1);
	isCorrect &= I.SetTime(testtime);
	isCorrect &= equal_vec(I.Field(), testfield1);
	isCorrect &= I.FieldType() == SpinAPI::InteractionFieldType::Broadband;
	isCorrect &= I.Type() == SpinAPI::InteractionType::SingleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Tests an Interaction object with an Ornstein-Uhlenbeck time-dependent field.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_field_ornsteinuhlenbeck()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=singlespin;fieldtype=ougeneral;autoseed=false;seed=1;field=0 0 0;correlationtime=10.0;timestep=1.0;randomorientations=true;";
	SpinAPI::Interaction I(name, contents);

	auto testfield1 = arma::vec("0 0 0");
	double testtime = 1.0;

	bool isCorrect = true;

	// Perform the test - with zero standard deviation the field vector should not change with time
	isCorrect &= equal_vec(I.Field(), testfield1);
	isCorrect &= I.SetTime(testtime);
	isCorrect &= equal_vec(I.Field(), testfield1);
	isCorrect &= I.FieldType() == SpinAPI::InteractionFieldType::OUGeneral;
	isCorrect &= I.Type() == SpinAPI::InteractionType::SingleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Tests an Interaction object with a monochromatic time-dependent tensor.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_tensor_monochromatic()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=doublespin;tensortype=monochromatic;tensor=matrix(0 0 0; 0 0 0; 0 0 0);frequency=1e+6;phase=1.5;";
	SpinAPI::Interaction I(name, contents);

	auto testmatrix = arma::mat("0 0 0; 0 0 0; 0 0 0");
	double testtime = 1.0;
	double frequency = 1e6;
	// pefrom the test
	bool isCorrect = true;
	isCorrect &= equal_double(I.GetTDFrequency(), frequency);
	SpinAPI::Tensor testTensor1 = *I.CouplingTensor();
	isCorrect &= equal_matrices(testTensor1.LabFrame(), testmatrix);
	isCorrect &= I.SetTime(testtime);
	SpinAPI::Tensor testTensor2 = *I.CouplingTensor();
	isCorrect &= equal_matrices(testTensor2.LabFrame(), testmatrix);
	isCorrect &= I.TensorType() == SpinAPI::InteractionTensorType::Monochromatic;
	isCorrect &= I.Type() == SpinAPI::InteractionType::DoubleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Tests an Interaction object with a broadband time-dependent tensor.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_tensor_broadband()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=doublespin;tensortype=broadband;autoseed=false;seed=1;tensor=matrix(0 0 0; 0 0 0; 0 0 0);minfreq=0.1e+6;maxfreq=0.2e+6;components=100;";
	SpinAPI::Interaction I(name, contents);

	auto testmatrix = arma::mat("0 0 0; 0 0 0; 0 0 0");
	double testtime = 1.0;

	// pefrom the test
	bool isCorrect = true;
	SpinAPI::Tensor testTensor1 = *I.CouplingTensor();
	isCorrect &= equal_matrices(testTensor1.LabFrame(), testmatrix);
	isCorrect &= I.SetTime(testtime);
	SpinAPI::Tensor testTensor2 = *I.CouplingTensor();
	isCorrect &= equal_matrices(testTensor2.LabFrame(), testmatrix);
	isCorrect &= I.TensorType() == SpinAPI::InteractionTensorType::Broadband;
	isCorrect &= I.Type() == SpinAPI::InteractionType::DoubleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Tests an Interaction object with an Ornstein-Uhlenbeck time-dependent tensor.
// DEPENDENCY NOTE: ObjectParser
bool test_spinapi_interaction_tensor_ornsteinuhlenbeck()
{
	// Setup objects for the test
	std::string name = "test1";
	std::string contents = "type=doublespin;tensortype=ougeneral;autoseed=false;seed=1;tensor=matrix(0 0 0; 0 0 0; 0 0 0);correlationtime=10.0;timestep=1.0;";
	SpinAPI::Interaction I(name, contents);

	auto testmatrix = arma::mat("0 0 0; 0 0 0; 0 0 0");
	double testtime = 1.0;

	// pefrom the test
	bool isCorrect = true;
	SpinAPI::Tensor testTensor1 = *I.CouplingTensor();
	isCorrect &= equal_matrices(testTensor1.LabFrame(), testmatrix);
	isCorrect &= I.SetTime(testtime);
	SpinAPI::Tensor testTensor2 = *I.CouplingTensor();
	isCorrect &= equal_matrices(testTensor2.LabFrame(), testmatrix);
	isCorrect &= I.TensorType() == SpinAPI::InteractionTensorType::OUGeneral;
	isCorrect &= I.Type() == SpinAPI::InteractionType::DoubleSpin;
	isCorrect &= I.HasTimeDependence();
	isCorrect &= !IsStatic(I);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_interaction_orientation_validation_and_exchange()
{
	std::ostringstream expectedError;
	std::streambuf *originalCerr = std::cerr.rdbuf(expectedError.rdbuf());
	SpinAPI::Interaction badZfs("badZfs", "type=zfs;D=1;E=0;orientation=1,2;");
	std::cerr.rdbuf(originalCerr);

	SpinAPI::Interaction exchange("exchange", "type=exchange;orientation=1,2,3;");

	bool isCorrect = true;
	isCorrect &= equal_vec(badZfs.Framelist(), arma::vec("0 0 0"));
	isCorrect &= equal_vec(exchange.Framelist(), arma::vec("0 0 0"));
	isCorrect &= (exchange.Type() == SpinAPI::InteractionType::Exchange);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Six-component strain input follows
// E={exx,exy,exz,eyy,eyz,ezz} and
// D={d43,d41,d26,d25,d16,d15}. Keep derived quantities and action
// targets tied to those documented components.
bool test_spinapi_strain_component_mapping()
{
	auto spin = std::make_shared<SpinAPI::Spin>("T", "type=electron;spin=1;");
	auto strain = std::make_shared<SpinAPI::Interaction>(
		"strain",
		"type=strain;group1=T;e=1 2 3 4 5 6;d=10 20 30 40 50 60;"
		"ignoretensors=true;commonprefactor=false;prefactor=1;");
	std::vector<SpinAPI::spin_ptr> spins{spin};
	if (!strain->ParseSpinGroups(spins) || !strain->IsValid())
		return false;

	std::vector<RunSection::NamedActionScalar> scalars;
	std::vector<RunSection::NamedActionVector> vectors;
	strain->GetActionTargets(scalars, vectors, "sys");

	std::map<std::string, double> values;
	for (const auto &entry : scalars)
		values[entry.first] = entry.second.Get();

	const double expectedEx1 = 3.0 - 0.5 * (60.0 / 50.0) * (1.0 - 4.0);
	const double expectedEx2 = 3.0 - 0.5 * (40.0 / 30.0) * (1.0 - 4.0);
	const double expectedEx = std::sqrt(
		(expectedEx1 * expectedEx1 + expectedEx2 * expectedEx2) / 2.0);
	const double expectedEy1 = 5.0 + (60.0 / 50.0) * 2.0;
	const double expectedEy2 = 5.0 + (40.0 / 30.0) * 2.0;
	const double expectedEy = std::sqrt(
		(expectedEy1 * expectedEy1 + expectedEy2 * expectedEy2) / 2.0);

	return equal_double(values["sys.strain.ex"], expectedEx) &&
		   equal_double(values["sys.strain.ey"], expectedEy) &&
		   equal_double(values["sys.strain.ez"], 16.0) &&
		   equal_double(values["sys.strain.exx"], 1.0) &&
		   equal_double(values["sys.strain.eyy"], 4.0) &&
		   equal_double(values["sys.strain.ezz"], 6.0) &&
		   equal_double(values["sys.strain.d43"], 10.0) &&
		   equal_double(values["sys.strain.d41"], 20.0) &&
		   equal_double(values["sys.strain.d26"], 30.0) &&
		   equal_double(values["sys.strain.d25"], 40.0) &&
		   equal_double(values["sys.strain.d16"], 50.0) &&
		   equal_double(values["sys.strain.d15"], 60.0);
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_zeeman_orientation_rotates_gtensor()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=matrix(1 0 0; 0 2 0; 0 0 3);");
	auto zeeman = std::make_shared<SpinAPI::Interaction>(
		"B0",
		"type=zeeman;spins=E;field=0 0 1;ignoretensors=false;commonprefactor=false;prefactor=1;"
		"orientation=0,1.5707963267948966,0;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);
	spinsys.Add(zeeman);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat H;
	arma::cx_mat Sz;

	bool isCorrect = true;
	isCorrect &= space.InteractionOperator(zeeman, H);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz);

	// A beta=pi/2 tensor-frame rotation moves the tensor x-axis onto lab z,
	// so Bz sees gxx=1 instead of the unrotated gzz=3.
	isCorrect &= equal_matrices(H, Sz, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_rotated_zeeman_hamiltonian_follows_powder_orientation()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=matrix(2 0 0; 0 3 0; 0 0 5);");
	auto microwave = std::make_shared<SpinAPI::Interaction>(
		"mw",
		"type=zeeman;spins=E;field=1 0 0;ignoretensors=false;commonprefactor=false;prefactor=1;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);
	spinsys.Add(microwave);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);
	space.UseFullTensorRotation(true);

	arma::mat identity = arma::eye<arma::mat>(3, 3);
	const double beta = M_PI / 2.0;
	arma::mat beta90 = {
		{std::cos(beta), 0.0, std::sin(beta)},
		{0.0, 1.0, 0.0},
		{-std::sin(beta), 0.0, std::cos(beta)}};

	arma::sp_cx_mat Hidentity;
	arma::sp_cx_mat Hrotated;
	arma::sp_cx_mat Sx;

	bool isCorrect = true;
	isCorrect &= space.BaseHamiltonianRotatedZYZ({"mw"}, identity, Hidentity);
	isCorrect &= space.BaseHamiltonianRotatedZYZ({"mw"}, beta90, Hrotated);
	isCorrect &= space.CreateOperator(spin->Sx(), spin, Sx);

	// A lab-fixed B1 along x sees gxx=2 for the identity crystallite. Rotating
	// the crystallite by beta=pi/2 moves molecular z onto lab x, so B1 sees gzz=5.
	isCorrect &= equal_matrices(Hidentity, 2.0 * Sx, 1e-12);
	isCorrect &= equal_matrices(Hrotated, 5.0 * Sx, 1e-12);
	isCorrect &= (arma::norm(arma::cx_mat(Hrotated - Hidentity), "fro") > 1.0);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_zfs_formalism_and_orientation()
{
	auto spin = std::make_shared<SpinAPI::Spin>("T", "type=electron;spin=1;");
	auto zfs = std::make_shared<SpinAPI::Interaction>(
		"ZFS",
		"type=zfs;group1=T;D=9;E=2;commonprefactor=false;prefactor=1;energyshift=true;");
	auto zfsRotated = std::make_shared<SpinAPI::Interaction>(
		"ZFSrot",
		"type=zfs;group1=T;D=9;E=0;commonprefactor=false;prefactor=1;energyshift=false;"
		"orientation=0,1.5707963267948966,0;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);
	spinsys.Add(zfs);
	spinsys.Add(zfsRotated);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat H;
	arma::cx_mat Hrot;
	arma::cx_mat Sx;
	arma::cx_mat Sy;
	arma::cx_mat Sz;

	bool isCorrect = true;
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz);
	isCorrect &= space.InteractionOperator(zfs, H);
	isCorrect &= space.InteractionOperator(zfsRotated, Hrot);

	const double s = 1.0;
	arma::cx_mat identity = arma::eye<arma::cx_mat>(H.n_rows, H.n_cols);
	arma::cx_mat expected = 9.0 * (Sz * Sz - (s * (s + 1.0) / 3.0) * identity) + 2.0 * (Sx * Sx - Sy * Sy);
	arma::cx_mat expectedRotated = 9.0 * Sx * Sx;

	isCorrect &= equal_matrices(H, expected, 1e-12);
	isCorrect &= equal_matrices(Hrot, expectedRotated, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_rotated_quadraticspin_matches_plain_for_identity_powder()
{
	auto spin = std::make_shared<SpinAPI::Spin>("T", "type=electron;spin=1;");
	auto quadratic = std::make_shared<SpinAPI::Interaction>(
		"Q",
		"type=quadraticspin;group1=T;tensor=matrix(\"1 0 0; 0 2 0; 0 0 4\");commonprefactor=false;prefactor=1;"
		"orientation=0.2,0.4,0.6;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);
	spinsys.Add(quadratic);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);
	space.UseFullTensorRotation(true);

	arma::sp_cx_mat plain;
	arma::sp_cx_mat rotated;
	arma::mat identityRotation = arma::eye<arma::mat>(3, 3);

	bool isCorrect = true;
	isCorrect &= space.InteractionOperator(quadratic, plain);
	isCorrect &= space.InteractionOperatorRotatedZYZ(quadratic, identityRotation, rotated);
	isCorrect &= equal_matrices(plain, rotated, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_phenomenological_relaxation_operator()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");
	auto relax = std::make_shared<SpinAPI::Operator>("R", "type=relaxationphenomenological;rate1=0.25;rate2=0.75;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems;
	systems.push_back(spinsys);

	bool isCorrect = true;
	isCorrect &= (spinsys->ValidateOperators(systems).size() == 0);

	SpinAPI::SpinSpace space(*spinsys);
	space.UseSuperoperatorSpace(true);

	arma::cx_mat R;
	arma::cx_mat expected = arma::zeros<arma::cx_mat>(4, 4);
	expected(0, 0) = -0.25;
	expected(0, 3) = 0.25;
	expected(1, 1) = -0.75;
	expected(2, 2) = -0.75;
	expected(3, 0) = 0.25;
	expected(3, 3) = -0.25;

	isCorrect &= space.RelaxationOperator(relax, R);
	isCorrect &= equal_matrices(R, expected, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_phenomenological_relaxation_framechange_is_basis_local()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");
	auto relax = std::make_shared<SpinAPI::Operator>("R", "type=relaxationphenomenological;rate1=0.25;rate2=0.75;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems;
	systems.push_back(spinsys);

	bool isCorrect = true;
	isCorrect &= (spinsys->ValidateOperators(systems).size() == 0);

	SpinAPI::SpinSpace space(*spinsys);
	space.UseSuperoperatorSpace(true);

	arma::cx_mat canonical;
	arma::cx_mat changed;
	const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
	arma::cx_mat x_basis(2, 2);
	x_basis(0, 0) = inv_sqrt2;
	x_basis(0, 1) = inv_sqrt2;
	x_basis(1, 0) = inv_sqrt2;
	x_basis(1, 1) = -inv_sqrt2;

	isCorrect &= space.RelaxationOperator(relax, canonical);
	isCorrect &= space.RelaxationOperatorFrameChange(relax, x_basis, changed);
	isCorrect &= equal_matrices(changed, canonical, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_phenomenological_rate2_preserves_populations_and_trace()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");
	auto relax = std::make_shared<SpinAPI::Operator>("R", "type=relaxationphenomenological;rate1=0.0;rate2=0.75;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems = {spinsys};

	bool isCorrect = (spinsys->ValidateOperators(systems).size() == 0);

	SpinAPI::SpinSpace space(*spinsys);
	space.UseSuperoperatorSpace(true);

	arma::cx_mat R;
	isCorrect &= space.RelaxationOperator(relax, R);

	arma::cx_mat rho = arma::zeros<arma::cx_mat>(2, 2);
	rho(0, 0) = 0.75;
	rho(1, 1) = 0.25;
	rho(0, 1) = arma::cx_double(0.2, 0.1);
	rho(1, 0) = std::conj(rho(0, 1));

	arma::cx_vec rhoVec;
	arma::cx_mat derivative;
	isCorrect &= space.OperatorToSuperspace(rho, rhoVec);
	isCorrect &= space.OperatorFromSuperspace(R * rhoVec, derivative);

	arma::cx_mat expected = arma::zeros<arma::cx_mat>(2, 2);
	expected(0, 1) = -0.75 * rho(0, 1);
	expected(1, 0) = -0.75 * rho(1, 0);

	isCorrect &= equal_matrices(derivative, expected, 1e-12);
	isCorrect &= (std::abs(arma::trace(derivative)) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_powder_phenomenological_rate2_uses_supplied_eigenbasis()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");
	auto relax = std::make_shared<SpinAPI::Operator>("R", "type=relaxationphenomenological;rate1=0.0;rate2=0.75;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems = {spinsys};

	bool isCorrect = (spinsys->ValidateOperators(systems).size() == 0);

	SpinAPI::SpinSpace space(*spinsys);
	space.UseSuperoperatorSpace(true);

	// Columns are the eigenvectors of Sx. A lab-frame |up_z><up_z| state
	// contains coherence in this basis. Pure eigenbasis rate2 damping therefore
	// changes its lab-frame diagonal elements while preserving trace.
	const double invSqrt2 = 1.0 / std::sqrt(2.0);
	arma::cx_mat xBasis = {
		{arma::cx_double(invSqrt2, 0.0), arma::cx_double(invSqrt2, 0.0)},
		{arma::cx_double(invSqrt2, 0.0), arma::cx_double(-invSqrt2, 0.0)}};
	arma::mat spatialRotation = arma::eye<arma::mat>(3, 3);

	arma::sp_cx_mat R;
	isCorrect &= space.PowderRelaxationOperator(relax, xBasis, spatialRotation, R);

	arma::cx_mat rho = arma::zeros<arma::cx_mat>(2, 2);
	rho(0, 0) = 1.0;

	arma::cx_vec rhoVec;
	arma::cx_mat derivative;
	isCorrect &= space.OperatorToSuperspace(rho, rhoVec);
	isCorrect &= space.OperatorFromSuperspace(R * rhoVec, derivative);

	arma::cx_mat expected = arma::zeros<arma::cx_mat>(2, 2);
	expected(0, 0) = -0.375;
	expected(1, 1) = 0.375;

	isCorrect &= equal_matrices(derivative, expected, 1e-12);
	isCorrect &= (std::abs(arma::trace(derivative)) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_hilbert_phenomenological_rate2_uses_supplied_eigenbasis()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	const double invSqrt2 = 1.0 / std::sqrt(2.0);
	arma::cx_mat xBasis = {
		{arma::cx_double(invSqrt2, 0.0), arma::cx_double(invSqrt2, 0.0)},
		{arma::cx_double(invSqrt2, 0.0), arma::cx_double(-invSqrt2, 0.0)}};
	std::vector<SpinAPI::HilbertRelaxationPhenomenologicalTerm> terms(1);
	terms[0].coherenceRate = 0.75;

	arma::cx_mat rho = arma::zeros<arma::cx_mat>(2, 2);
	rho(0, 0) = 1.0;

	arma::cx_mat expected = arma::zeros<arma::cx_mat>(2, 2);
	expected(0, 0) = -0.375;
	expected(1, 1) = 0.375;

	arma::cx_mat direct;
	arma::cx_mat superoperator;
	arma::cx_vec rhoVec;
	arma::cx_mat fromSuperoperator;
	bool isCorrect = true;
	isCorrect &= space.ApplyPhenomenologicalRelaxationHilbert(terms, xBasis, rho, direct);
	isCorrect &= space.PhenomenologicalRelaxationSuperoperatorHilbert(terms, xBasis, superoperator);
	isCorrect &= space.OperatorToSuperspace(rho, rhoVec);
	isCorrect &= space.OperatorFromSuperspace(superoperator * rhoVec, fromSuperoperator);
	isCorrect &= equal_matrices(direct, expected, 1e-12);
	isCorrect &= equal_matrices(fromSuperoperator, expected, 1e-12);
	isCorrect &= (std::abs(arma::trace(direct)) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_hilbert_phenomenological_finite_step_map_matches_superoperator()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	const double invSqrt2 = 1.0 / std::sqrt(2.0);
	arma::cx_mat xBasis = {
		{arma::cx_double(invSqrt2, 0.0), arma::cx_double(invSqrt2, 0.0)},
		{arma::cx_double(invSqrt2, 0.0), arma::cx_double(-invSqrt2, 0.0)}};
	std::vector<SpinAPI::HilbertRelaxationPhenomenologicalTerm> terms(2);
	terms[0].populationRate = 0.20;
	terms[0].coherenceRate = 0.35;
	terms[1].populationRate = 0.15;
	terms[1].coherenceRate = 0.40;
	const double timestep = 0.625;

	arma::cx_mat rho = {
		{arma::cx_double(0.70, 0.0), arma::cx_double(0.20, 0.10)},
		{arma::cx_double(0.20, -0.10), arma::cx_double(0.30, 0.0)}};
	arma::cx_mat mapped = rho;
	arma::cx_mat workspace;

	SpinAPI::HilbertPhenomenologicalRelaxationMap map;
	arma::cx_mat superoperator;
	arma::cx_vec rhoVec;
	arma::cx_mat fromSuperoperator;
	arma::cx_mat mappedInBasis = xBasis.t() * rho * xBasis;
	bool isCorrect = true;
	isCorrect &= space.CreatePhenomenologicalRelaxationMapHilbert(terms, xBasis, timestep, map);
	isCorrect &= space.ApplyPhenomenologicalRelaxationMapHilbert(map, mapped, workspace);
	isCorrect &= space.ApplyPhenomenologicalRelaxationMapInBasisHilbert(map, mappedInBasis);
	mappedInBasis = xBasis * mappedInBasis * xBasis.t();
	isCorrect &= space.PhenomenologicalRelaxationSuperoperatorHilbert(terms, xBasis, superoperator);
	isCorrect &= space.OperatorToSuperspace(rho, rhoVec);
	isCorrect &= space.OperatorFromSuperspace(arma::expmat(superoperator * timestep) * rhoVec, fromSuperoperator);
	isCorrect &= equal_matrices(mapped, fromSuperoperator, 1e-12);
	isCorrect &= equal_matrices(mappedInBasis, fromSuperoperator, 1e-12);
	isCorrect &= (std::abs(arma::trace(mapped) - arma::trace(rho)) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_rotate_state_maps_z_population_to_x_population()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat Sx;
	bool isCorrect = true;
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx);

	arma::cx_mat rhoZ = arma::zeros<arma::cx_mat>(2, 2);
	rhoZ(0, 0) = 1.0;

	arma::mat rotation = {
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 0.0},
		{-1.0, 0.0, 0.0}};

	arma::cx_mat rotated;
	isCorrect &= space.RotateState(rhoZ, rotation, rotated);

	arma::cx_mat expected = 0.5 * arma::eye<arma::cx_mat>(2, 2) + Sx;
	isCorrect &= equal_matrices(rotated, expected, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_rotate_state_singlet_invariant_triplet_t0_rotates()
{
	auto spin1 = std::make_shared<SpinAPI::Spin>("E1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("E2", "spin=1/2;");
	auto singlet = std::make_shared<SpinAPI::State>("S", "spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");
	auto tripletT0 = std::make_shared<SpinAPI::State>("T0", "spins(E1,E2)=|1/2,-1/2>+|-1/2,1/2>;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);

	bool isCorrect = true;
	isCorrect &= singlet->ParseFromSystem(spinsys);
	isCorrect &= tripletT0->ParseFromSystem(spinsys);

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat Psinglet;
	arma::cx_mat PT0;
	isCorrect &= space.GetState(singlet, Psinglet);
	isCorrect &= space.GetState(tripletT0, PT0);

	// Generic non-trivial powder rotation. The electron singlet is a scalar
	// under global spin rotation, while the triplet T0 component is not.
	const double beta = M_PI / 2.0;
	arma::mat rotation = {
		{std::cos(beta), 0.0, std::sin(beta)},
		{0.0, 1.0, 0.0},
		{-std::sin(beta), 0.0, std::cos(beta)}};

	arma::cx_mat rotatedSinglet;
	arma::cx_mat rotatedT0;
	SpinAPI::HilbertStateRotationCache singletCache;
	SpinAPI::HilbertStateRotationCache tripletCache;
	isCorrect &= space.CreateStateRotationCache(Psinglet, singletCache);
	isCorrect &= space.CreateStateRotationCache(PT0, tripletCache);
	isCorrect &= singletCache.rotationInvariant;
	isCorrect &= !tripletCache.rotationInvariant;
	isCorrect &= space.RotateState(Psinglet, rotation, rotatedSinglet);
	isCorrect &= space.RotateState(PT0, rotation, tripletCache, rotatedT0);

	isCorrect &= equal_matrices(rotatedSinglet, Psinglet, 1e-12);
	isCorrect &= (arma::norm(rotatedT0 - PT0, "fro") > 1e-3);
	isCorrect &= (std::abs(arma::trace(Psinglet * rotatedT0)) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_prepare_powder_initial_density_uses_rotation_invariance()
{
	auto spin1 = std::make_shared<SpinAPI::Spin>("E1", "spin=1/2;");
	auto spin2 = std::make_shared<SpinAPI::Spin>("E2", "spin=1/2;");
	auto singlet = std::make_shared<SpinAPI::State>("S", "spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin1);
	spinsys.Add(spin2);

	bool isCorrect = singlet->ParseFromSystem(spinsys);
	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(true);

	arma::cx_mat Psinglet;
	isCorrect &= space.GetState(singlet, Psinglet);

	SpinAPI::HilbertStateRotationCache cache;
	isCorrect &= space.CreateStateRotationCache(Psinglet, cache);
	isCorrect &= cache.rotationInvariant;

	const double beta = 0.83;
	arma::mat rotation = {
		{std::cos(beta), 0.0, std::sin(beta)},
		{0.0, 1.0, 0.0},
		{-std::sin(beta), 0.0, std::cos(beta)}};

	arma::cx_mat prepared;
	const std::vector<std::string> unusedHamiltonian;
	isCorrect &= space.PrepareInitialDensityForPowder(Psinglet, rotation, SpinAPI::StateFrame::Molecular, false, unusedHamiltonian, &cache, prepared);
	isCorrect &= equal_matrices(prepared, Psinglet, 1e-12);

	// PrepareInitialDensityForPowder temporarily switches to Hilbert operators
	// internally. Confirm that the caller's superspace setting is restored.
	isCorrect &= (space.SpaceDimensions() == space.SuperSpaceDimensions());
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_dephase_state_in_eigenbasis_removes_hamiltonian_coherences()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "spin=1/2;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat Sx;
	bool isCorrect = true;
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx);

	arma::cx_mat rhoZ = arma::zeros<arma::cx_mat>(2, 2);
	rhoZ(0, 0) = 1.0;

	arma::cx_mat dephased;
	isCorrect &= space.DephaseStateInEigenbasis(rhoZ, Sx, dephased);

	arma::cx_mat expected = 0.5 * arma::eye<arma::cx_mat>(2, 2);
	isCorrect &= equal_matrices(dephased, expected, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_triplet_keep_retains_free_induction_dephase_removes_it()
{
	auto spin = std::make_shared<SpinAPI::Spin>("T", "type=electron;spin=1;tensor=isotropic(2);");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::cx_mat Sx;
	arma::cx_mat Sy;
	arma::cx_mat Sz;
	bool isCorrect = true;
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy);
	isCorrect &= space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz);

	// Molecular Tz alignment rotated to a generic powder orientation. A secular
	// high-field Hamiltonian containing ZFS keeps the retained zero-field
	// coherences observable as free induction even without a microwave field.
	arma::cx_mat rhoMolecular = arma::zeros<arma::cx_mat>(3, 3);
	rhoMolecular(1, 1) = 1.0;
	const double beta = 0.7;
	arma::mat rotation = {
		{std::cos(beta), 0.0, std::sin(beta)},
		{0.0, 1.0, 0.0},
		{-std::sin(beta), 0.0, std::cos(beta)}};
	arma::cx_mat rhoOriented;
	isCorrect &= space.RotateState(rhoMolecular, rotation, rhoOriented);

	const arma::cx_mat H = 0.3 * Sz + 0.7 * Sz * Sz;
	arma::cx_mat rhoDephased;
	isCorrect &= space.DephaseStateInEigenbasis(rhoOriented, H, rhoDephased);

	const arma::cx_mat U = arma::expmat(arma::cx_double(0.0, -0.7) * H);
	const arma::cx_mat keepEvolved = U * rhoOriented * U.t();
	const arma::cx_mat dephasedEvolved = U * rhoDephased * U.t();
	const double keepTransverse = std::abs(arma::trace(Sx * keepEvolved)) + std::abs(arma::trace(Sy * keepEvolved));
	const double dephasedTransverse = std::abs(arma::trace(Sx * dephasedEvolved)) + std::abs(arma::trace(Sy * dephasedEvolved));

	isCorrect &= (keepTransverse > 1e-2);
	isCorrect &= (dephasedTransverse < 1e-12);
	isCorrect &= (std::abs(arma::trace(keepEvolved) - arma::trace(dephasedEvolved)) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_relaxation_t1_spin_one_population_exchange()
{
	auto spin = std::make_shared<SpinAPI::Spin>("T", "type=electron;spin=1;tensor=isotropic(2);");
	auto relax = std::make_shared<SpinAPI::Operator>("T1", "type=relaxationt1;spins=T;rate=1.0;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems = {spinsys};

	bool isCorrect = (spinsys->ValidateOperators(systems).size() == 0);

	SpinAPI::SpinSpace ss_space(*spinsys);
	ss_space.UseSuperoperatorSpace(true);

	arma::sp_cx_mat R;
	isCorrect &= ss_space.RelaxationOperator(relax, R);

	auto apply_superspace = [&](const arma::cx_mat &rho, arma::cx_mat &out) -> bool {
		arma::cx_vec rho_vec;
		if (!ss_space.OperatorToSuperspace(rho, rho_vec))
			return false;
		return ss_space.OperatorFromSuperspace(R * rho_vec, out);
	};

	arma::cx_mat rho_top = arma::zeros<arma::cx_mat>(3, 3);
	rho_top(0, 0) = 1.0;
	arma::cx_mat expected_top = arma::zeros<arma::cx_mat>(3, 3);
	expected_top(0, 0) = -1.0;
	expected_top(1, 1) = 1.0;

	arma::cx_mat rho_middle = arma::zeros<arma::cx_mat>(3, 3);
	rho_middle(1, 1) = 1.0;
	arma::cx_mat expected_middle = arma::zeros<arma::cx_mat>(3, 3);
	expected_middle(0, 0) = 1.0;
	expected_middle(1, 1) = -2.0;
	expected_middle(2, 2) = 1.0;

	arma::cx_mat rho_bottom = arma::zeros<arma::cx_mat>(3, 3);
	rho_bottom(2, 2) = 1.0;
	arma::cx_mat expected_bottom = arma::zeros<arma::cx_mat>(3, 3);
	expected_bottom(1, 1) = 1.0;
	expected_bottom(2, 2) = -1.0;

	arma::cx_mat out_top;
	arma::cx_mat out_middle;
	arma::cx_mat out_bottom;
	isCorrect &= apply_superspace(rho_top, out_top);
	isCorrect &= apply_superspace(rho_middle, out_middle);
	isCorrect &= apply_superspace(rho_bottom, out_bottom);
	isCorrect &= equal_matrices(out_top, expected_top, 1e-12);
	isCorrect &= equal_matrices(out_middle, expected_middle, 1e-12);
	isCorrect &= equal_matrices(out_bottom, expected_bottom, 1e-12);

	SpinAPI::SpinSpace hs_space(*spinsys);
	hs_space.UseSuperoperatorSpace(false);
	SpinAPI::HilbertRelaxationCache cache;
	arma::cx_mat hs_out;
	isCorrect &= hs_space.RelaxationOperator(relax, cache);
	isCorrect &= hs_space.ApplyRelaxationHilbert(cache, rho_middle, hs_out);
	isCorrect &= equal_matrices(hs_out, expected_middle, 1e-12);

	arma::cx_mat identity = arma::eye<arma::cx_mat>(3, 3);
	arma::cx_mat identity_out;
	isCorrect &= hs_space.ApplyRelaxationHilbert(cache, identity, identity_out);
	isCorrect &= equal_matrices(identity_out, arma::zeros<arma::cx_mat>(3, 3), 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_relaxation_t2_is_normalized_pure_dephasing()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);");
	auto relax = std::make_shared<SpinAPI::Operator>("T2", "type=relaxationt2;spins=E;rate=0.75;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems = {spinsys};

	bool isCorrect = (spinsys->ValidateOperators(systems).size() == 0);
	isCorrect &= (relax->Frame() == SpinAPI::RelaxationFrame::Lab);

	arma::cx_mat rho = arma::zeros<arma::cx_mat>(2, 2);
	rho(0, 0) = 0.70;
	rho(1, 1) = 0.30;
	rho(0, 1) = arma::cx_double(0.20, 0.10);
	rho(1, 0) = std::conj(rho(0, 1));

	// relaxationt2 means pure dephasing at the user-supplied rate 1/Tphi.
	// Populations and trace must remain untouched. T1 contributions to an
	// observed 1/T2 are supplied separately through relaxationt1.
	arma::cx_mat expected = arma::zeros<arma::cx_mat>(2, 2);
	expected(0, 1) = -0.75 * rho(0, 1);
	expected(1, 0) = -0.75 * rho(1, 0);

	SpinAPI::SpinSpace ss_space(*spinsys);
	ss_space.UseSuperoperatorSpace(true);
	arma::sp_cx_mat R;
	arma::cx_vec rho_vec;
	arma::cx_mat ss_out;
	isCorrect &= ss_space.RelaxationOperator(relax, R);
	isCorrect &= ss_space.OperatorToSuperspace(rho, rho_vec);
	isCorrect &= ss_space.OperatorFromSuperspace(R * rho_vec, ss_out);
	isCorrect &= equal_matrices(ss_out, expected, 1e-12);
	isCorrect &= (std::abs(arma::trace(ss_out)) < 1e-12);

	SpinAPI::SpinSpace hs_space(*spinsys);
	hs_space.UseSuperoperatorSpace(false);
	SpinAPI::HilbertRelaxationCache cache;
	arma::cx_mat hs_out;
	isCorrect &= hs_space.RelaxationOperator(relax, cache);
	isCorrect &= hs_space.ApplyRelaxationHilbert(cache, rho, hs_out);
	isCorrect &= equal_matrices(hs_out, expected, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_relaxation_random_fields_are_trace_preserving_and_powder_invariant()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);");
	auto relax = std::make_shared<SpinAPI::Operator>("RFR", "type=relaxationrandomfields;spins=E;rate=0.4;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems = {spinsys};

	bool isCorrect = (spinsys->ValidateOperators(systems).size() == 0);
	isCorrect &= (relax->Frame() == SpinAPI::RelaxationFrame::Molecular);

	arma::cx_mat rho = arma::zeros<arma::cx_mat>(2, 2);
	rho(0, 0) = 0.75;
	rho(1, 1) = 0.25;
	rho(0, 1) = arma::cx_double(0.20, 0.10);
	rho(1, 0) = std::conj(rho(0, 1));

	// For one spin-1/2 and equal Cartesian rates k, sum_j k D[S_j]
	// reduces to k * (trace(rho) I / 2 - rho). This is the one-spin
	// component of the random-field model used by Kattnig et al.
	const arma::cx_mat expected = 0.4 * (0.5 * arma::eye<arma::cx_mat>(2, 2) - rho);

	SpinAPI::SpinSpace ss_space(*spinsys);
	ss_space.UseSuperoperatorSpace(true);
	arma::sp_cx_mat R;
	arma::cx_vec rho_vec;
	arma::cx_mat ss_out;
	isCorrect &= ss_space.RelaxationOperator(relax, R);
	isCorrect &= ss_space.OperatorToSuperspace(rho, rho_vec);
	isCorrect &= ss_space.OperatorFromSuperspace(R * rho_vec, ss_out);
	isCorrect &= equal_matrices(ss_out, expected, 1e-12);
	isCorrect &= (std::abs(arma::trace(ss_out)) < 1e-12);

	SpinAPI::SpinSpace hs_space(*spinsys);
	hs_space.UseSuperoperatorSpace(false);
	SpinAPI::HilbertRelaxationCache cache;
	arma::cx_mat hs_out;
	isCorrect &= hs_space.RelaxationOperator(relax, cache);
	isCorrect &= hs_space.ApplyRelaxationHilbert(cache, rho, hs_out);
	isCorrect &= equal_matrices(hs_out, expected, 1e-12);

	// An isotropic random-field generator is invariant under powder rotation,
	// even though its default frame is molecular. This catches accidental
	// orientation-dependent loss terms in either propagation hierarchy.
	const double beta = M_PI / 2.0;
	arma::mat beta90 = {
		{std::cos(beta), 0.0, std::sin(beta)},
		{0.0, 1.0, 0.0},
		{-std::sin(beta), 0.0, std::cos(beta)}};
	arma::cx_mat basis = arma::eye<arma::cx_mat>(2, 2);
	arma::sp_cx_mat powder_R;
	isCorrect &= ss_space.PowderRelaxationOperatorEigenbasis(relax, basis, beta90, powder_R);
	isCorrect &= equal_matrices(arma::cx_mat(powder_R), arma::cx_mat(R), 1e-12);

	SpinAPI::HilbertRelaxationCache powder_cache;
	arma::cx_mat powder_hs_out;
	isCorrect &= hs_space.PowderRelaxationOperatorHilbert(relax, beta90, powder_cache);
	isCorrect &= hs_space.ApplyRelaxationHilbert(powder_cache, rho, powder_hs_out);
	isCorrect &= equal_matrices(powder_hs_out, expected, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_operator_copy_preserves_relaxation_configuration()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);");
	auto relax = std::make_shared<SpinAPI::Operator>("T2", "type=relaxationt2;spins=E;rate=0.75;frame=molecular;");

	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin);
	spinsys->Add(relax);
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> systems = {spinsys};

	bool isCorrect = (spinsys->ValidateOperators(systems).size() == 0);
	auto matches = [&relax](const SpinAPI::Operator &_operator)
	{
		return _operator.Name() == relax->Name() &&
			   _operator.Type() == SpinAPI::OperatorType::RelaxationT2 &&
			   _operator.SpinCount() == 1 &&
			   std::abs(_operator.Rate1() - 0.75) < 1e-12 &&
			   std::abs(_operator.Rate2() - 0.75) < 1e-12 &&
			   std::abs(_operator.Rate3() - 0.75) < 1e-12 &&
			   _operator.Frame() == SpinAPI::RelaxationFrame::Molecular &&
			   _operator.IsValid();
	};

	// Operators are copied by higher-level setup code. Every parsed property
	// and resolved spin pointer must survive both supported copy paths.
	SpinAPI::Operator copied(*relax);
	SpinAPI::Operator assigned("Placeholder", "type=unspecified;");
	assigned = *relax;
	isCorrect &= matches(copied);
	isCorrect &= matches(assigned);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_powder_grid_weights_and_rotation()
{
	SpinAPI::PowderGrid hemisphere;
	SpinAPI::PowderGrid fullSphere;
	bool isCorrect = true;
	isCorrect &= SpinAPI::CreateUniformPowderGrid(11, SpinAPI::PowderGridDomain::UpperHemisphere, hemisphere);
	isCorrect &= SpinAPI::CreateUniformPowderGrid(12, SpinAPI::PowderGridDomain::FullSphere, fullSphere);
	isCorrect &= (hemisphere.size() == 11);
	isCorrect &= (fullSphere.size() == 12);

	double hemisphereWeight = 0.0;
	for (const auto &orientation : hemisphere)
		hemisphereWeight += orientation.weight;
	double fullSphereWeight = 0.0;
	for (const auto &orientation : fullSphere)
		fullSphereWeight += orientation.weight;

	isCorrect &= (std::abs(hemisphereWeight - 2.0 * arma::datum::pi) < 1e-12);
	isCorrect &= (std::abs(fullSphereWeight - 4.0 * arma::datum::pi) < 1e-12);

	arma::mat R;
	isCorrect &= SpinAPI::CreateZYZRotationMatrix(0.0, arma::datum::pi / 2.0, 0.0, R);
	arma::vec z = {0.0, 0.0, 1.0};
	arma::vec rotated = R * z;
	isCorrect &= (arma::norm(rotated - arma::vec({1.0, 0.0, 0.0})) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_powder_grid_sophe_projection_helpers()
{
	bool isCorrect = true;

	SpinAPI::SopheGridParameters params;
	isCorrect &= SpinAPI::GetSopheGridParameters("Ci", params);
	isCorrect &= (std::abs(params.maxPhi - 2.0 * arma::datum::pi) < 1e-12);
	isCorrect &= (!params.closedPhi);
	isCorrect &= (params.nOctants == 4);

	SpinAPI::PowderGrid ciGrid;
	isCorrect &= SpinAPI::CreateSophePowderGrid(4, "ci", ciGrid);
	isCorrect &= (ciGrid.size() == static_cast<size_t>(SpinAPI::SopheGridPointCount(4, params.nOctants, params.closedPhi)));

	SpinAPI::PowderGrid axialGrid;
	isCorrect &= SpinAPI::CreateSophePowderGrid(5, "dinfh", axialGrid);
	double axialWeight = 0.0;
	for (const auto &orientation : axialGrid)
		axialWeight += orientation.weight;
	isCorrect &= (axialGrid.size() == 5);
	isCorrect &= (std::abs(axialWeight - 4.0 * arma::datum::pi) < 1e-12);

	SpinAPI::PowderGrid o3Grid;
	isCorrect &= SpinAPI::CreateSophePowderGrid(1, "o3", o3Grid);
	isCorrect &= (o3Grid.size() == 1);
	isCorrect &= (std::abs(o3Grid[0].weight - 4.0 * arma::datum::pi) < 1e-12);

	SpinAPI::PowderGrid octantGrid;
	isCorrect &= SpinAPI::CreateOctantPowderGrid(3, octantGrid);
	double octantWeight = 0.0;
	for (const auto &orientation : octantGrid)
		octantWeight += orientation.weight;
	isCorrect &= (octantGrid.size() == 9);
	isCorrect &= (std::abs(octantWeight - arma::datum::pi / 2.0) < 1e-12);

	SpinAPI::PowderGrid d2hGrid;
	SpinAPI::PowderProjectionMesh mesh;
	isCorrect &= SpinAPI::CreateSophePowderGrid(4, "d2h", d2hGrid);
	isCorrect &= SpinAPI::BuildSopheProjectionMesh(1, true, 4, d2hGrid, mesh);
	double meshWeight = 0.0;
	for (double weight : mesh.weights)
		meshWeight += weight;
	isCorrect &= (!mesh.axial);
	isCorrect &= (mesh.triangles.size() == 9);
	isCorrect &= (std::abs(meshWeight - 4.0 * arma::datum::pi) < 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_powder_hamiltonian_helper_matches_explicit_builders()
{
	auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=matrix(2 0 0; 0 3 0; 0 0 5);");
	auto staticField = std::make_shared<SpinAPI::Interaction>(
		"B0",
		"type=zeeman;spins=E;field=0 0 1;ignoretensors=false;commonprefactor=false;prefactor=1;");
	auto microwave = std::make_shared<SpinAPI::Interaction>(
		"mw",
		"type=zeeman;spins=E;field=1 0 0;ignoretensors=false;commonprefactor=false;prefactor=1;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(spin);
	spinsys.Add(staticField);
	spinsys.Add(microwave);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);
	space.UseFullTensorRotation(true);

	arma::mat rotation;
	SpinAPI::CreateZYZRotationMatrix(0.0, arma::datum::pi / 2.0, 0.0, rotation);

	arma::sp_cx_mat directH0;
	arma::sp_cx_mat directH1;
	arma::sp_cx_mat helperH0;
	arma::sp_cx_mat helperH1;
	arma::sp_cx_mat helperH;

	bool isCorrect = true;
	isCorrect &= space.BaseHamiltonianRotated_SA({"B0"}, rotation, directH0);
	isCorrect &= space.BaseHamiltonianRotatedZYZ({"mw"}, rotation, directH1);
	isCorrect &= space.PowderHamiltonianRotatedSA({"B0"}, {"mw"}, rotation, helperH0, helperH1, helperH);

	isCorrect &= equal_matrices(helperH0, directH0, 1e-12);
	isCorrect &= equal_matrices(helperH1, directH1, 1e-12);
	isCorrect &= equal_matrices(helperH, directH0 + directH1, 1e-12);

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////
// CHANGED 2026-07-07: Physics regression tests for the electron high-field
// secular powder Hamiltonian used by the spectroscopy tasks. These tests are
// deliberately small matrix tests: they preserve the required RWA/secular
// invariants without changing the public SpinAPI interface.
bool test_spinapi_secular_isotropic_hyperfine_is_sziz()
{
	auto e = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	auto n = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1/2;");
	auto hf = std::make_shared<SpinAPI::Interaction>("hf", "type=hyperfine;group1=E;group2=N;tensor=isotropic(1);commonprefactor=false;prefactor=1;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(e);
	spinsys.Add(n);
	spinsys.Add(hf);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::mat R = arma::eye<arma::mat>(3, 3);
	arma::sp_cx_mat H;
	arma::sp_cx_mat Sez;
	arma::sp_cx_mat Nsz;
	bool isCorrect = true;
	isCorrect &= space.InteractionOperatorRotated_SA(hf, R, H);
	isCorrect &= space.CreateOperator(e->Sz(), e, Sez);
	isCorrect &= space.CreateOperator(n->Sz(), n, Nsz);

	// In an electron rotating/high-field secular Hamiltonian isotropic A S.I
	// becomes A S_z I_z. The S_x I_x + S_y I_y flip-flop terms are nonsecular.
	isCorrect &= equal_matrices(H, Sez * Nsz, 1e-12);
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_secular_hyperfine_reversed_order_matches()
{
	auto e = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;");
	auto n = std::make_shared<SpinAPI::Spin>("N", "type=nucleus;spin=1;");
	const std::string tensor = "matrix(1 0.2 0.3;0.2 5 0.7;0.3 0.7 9)";
	auto hf_en = std::make_shared<SpinAPI::Interaction>("hf_en", "type=hyperfine;group1=E;group2=N;tensor=" + tensor + ";commonprefactor=false;prefactor=1;");
	auto hf_ne = std::make_shared<SpinAPI::Interaction>("hf_ne", "type=hyperfine;group1=N;group2=E;tensor=" + tensor + ";commonprefactor=false;prefactor=1;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(e);
	spinsys.Add(n);
	spinsys.Add(hf_en);
	spinsys.Add(hf_ne);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::mat R = arma::eye<arma::mat>(3, 3);
	arma::sp_cx_mat H_en;
	arma::sp_cx_mat H_ne;
	arma::sp_cx_mat Sez;
	arma::sp_cx_mat Nsx;
	arma::sp_cx_mat Nsy;
	arma::sp_cx_mat Nsz;
	bool isCorrect = true;
	isCorrect &= space.InteractionOperatorRotated_SA(hf_en, R, H_en);
	isCorrect &= space.InteractionOperatorRotated_SA(hf_ne, R, H_ne);
	isCorrect &= space.CreateOperator(e->Sz(), e, Sez);
	isCorrect &= space.CreateOperator(n->Sx(), n, Nsx);
	isCorrect &= space.CreateOperator(n->Sy(), n, Nsy);
	isCorrect &= space.CreateOperator(n->Sz(), n, Nsz);

	// group1=N/group2=E represents I^T A S. After reordering to electron first
	// inside the secular projection it must embed electron operators on the
	// electron Hilbert factor and nucleus operators on the nucleus factor.
	arma::sp_cx_mat expected = Sez * (0.3 * Nsx + 0.7 * Nsy + 9.0 * Nsz);
	isCorrect &= equal_matrices(H_en, expected, 1e-10);
	isCorrect &= equal_matrices(H_ne, expected, 1e-10);
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

bool test_spinapi_secular_electron_electron_historical_diagonal_projection()
{
	auto e1 = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;");
	auto e2 = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1/2;");
	auto ee = std::make_shared<SpinAPI::Interaction>("ee", "type=doublespin;group1=E1;group2=E2;tensor=matrix(1 4 0;4 1 0;0 0 3);commonprefactor=false;prefactor=1;");

	SpinAPI::SpinSystem spinsys("System");
	spinsys.Add(e1);
	spinsys.Add(e2);
	spinsys.Add(ee);
	spinsys.ValidateInteractions();

	SpinAPI::SpinSpace space(spinsys);
	space.UseSuperoperatorSpace(false);

	arma::mat R = arma::eye<arma::mat>(3, 3);
	arma::sp_cx_mat H;
	arma::sp_cx_mat Sx1;
	arma::sp_cx_mat Sy1;
	arma::sp_cx_mat Sz1;
	arma::sp_cx_mat Sx2;
	arma::sp_cx_mat Sy2;
	arma::sp_cx_mat Sz2;
	bool isCorrect = true;
	isCorrect &= space.InteractionOperatorRotated_SA(ee, R, H);
	isCorrect &= space.CreateOperator(e1->Sx(), e1, Sx1);
	isCorrect &= space.CreateOperator(e1->Sy(), e1, Sy1);
	isCorrect &= space.CreateOperator(e1->Sz(), e1, Sz1);
	isCorrect &= space.CreateOperator(e2->Sx(), e2, Sx2);
	isCorrect &= space.CreateOperator(e2->Sy(), e2, Sy2);
	isCorrect &= space.CreateOperator(e2->Sz(), e2, Sz2);

	// The historical _SA electron-electron projection keeps only the diagonal
	// tensor elements after frame/powder rotation. For an axial perpendicular
	// tensor this is also total-q=0 under common z rotation, and the off-diagonal
	// xy/yx terms in the input must not leak into the Hamiltonian.
	arma::sp_cx_mat expected = Sx1 * Sx2 + Sy1 * Sy2 + 3.0 * Sz1 * Sz2;
	isCorrect &= equal_matrices(H, expected, 1e-12);

	arma::sp_cx_mat Mz = Sz1 + Sz2;
	arma::sp_cx_mat comm = H * Mz - Mz * H;
	isCorrect &= (arma::norm(arma::conv_to<arma::cx_mat>::from(comm), "fro") < 1e-12);
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Fixed-dimension Krylov calls are used inside the pulse and free-evolution
// loops of several spectroscopy tasks. They must perform one Arnoldi build at
// the requested dimension; only TimeAdaptiveKrylov* may change that policy.
bool test_spinapi_krylov_general_respects_requested_dimension()
{
	SpinAPI::SpinSpace space;
	const int dimension = 4;
	const int krylovDimension = 2;
	const arma::cx_double dt(0.2, 0.0);

	arma::sp_cx_mat H(dimension, dimension);
	H.diag() = arma::cx_vec({0.0, 1.0, 2.0, 4.0});
	arma::cx_vec initial = arma::normalise(arma::cx_vec({1.0, 2.0, 3.0, 4.0}));

	arma::cx_mat basis(dimension, krylovDimension, arma::fill::zeros);
	arma::cx_mat hessenberg(krylovDimension, krylovDimension, arma::fill::zeros);
	basis.col(0) = initial;
	double residual = 0.0;
	space.ArnoldiProcess(H, initial, basis, hessenberg, krylovDimension, residual);

	arma::cx_vec e1(krylovDimension, arma::fill::zeros);
	e1(0) = 1.0;
	arma::cx_vec expected = basis * arma::expmat(hessenberg * dt) * e1;
	arma::cx_vec actual = space.KrylovExpmGeneral(H, initial, dt, krylovDimension, dimension);
	arma::cx_vec full = arma::expmat(arma::cx_mat(H) * dt) * initial;

	bool isCorrect = true;
	isCorrect &= (arma::norm(actual - expected, 2) < 1e-12);
	// This problem requires more than two Krylov vectors. If the fixed API
	// silently expands to the full space, actual would equal full.
	isCorrect &= (arma::norm(actual - full, 2) > 1e-6);
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// An invariant starting vector causes a happy Arnoldi/Lanczos breakdown after
// one basis vector. This is successful convergence, not a reason to repeatedly
// enlarge and rebuild the Krylov space.
bool test_spinapi_krylov_happy_breakdown_returns_exact_result()
{
	SpinAPI::SpinSpace space;
	const int dimension = 4;
	const arma::cx_double dt(0.2, 0.0);

	arma::sp_cx_mat H = 2.0 * arma::speye<arma::sp_cx_mat>(dimension, dimension);
	arma::cx_vec initial = arma::normalise(arma::cx_vec({1.0, 2.0, 3.0, 4.0}));
	arma::cx_vec expected = std::exp(2.0 * dt) * initial;

	arma::cx_vec general = space.KrylovExpmGeneral(H, initial, dt, dimension, dimension);
	arma::cx_vec symmetric = space.KrylovExpmSymm(H, initial, dt, dimension, dimension);

	return arma::norm(general - expected, 2) < 1e-12 &&
		   arma::norm(symmetric - expected, 2) < 1e-12;
}
//////////////////////////////////////////////////////////////////////////////

// The block overload follows the same fixed-dimension contract and must use
// the Hermitian inner product when orthogonalizing a complex Krylov basis.
bool test_spinapi_block_krylov_is_bounded_and_orthonormal()
{
	SpinAPI::SpinSpace space;
	const int dimension = 4;
	const arma::cx_double dt(0.1, 0.0);

	arma::sp_cx_mat H(dimension, dimension);
	H.diag() = arma::cx_vec({
		arma::cx_double(0.0, 0.2),
		arma::cx_double(1.0, -0.1),
		arma::cx_double(2.0, 0.3),
		arma::cx_double(4.0, -0.2)});
	arma::cx_mat initial(dimension, 1);
	initial.col(0) = arma::normalise(arma::cx_vec({
		arma::cx_double(1.0, 0.5),
		arma::cx_double(2.0, -1.0),
		arma::cx_double(-0.5, 2.0),
		arma::cx_double(1.5, 0.25)}));

	auto bounded = space.KrylovExpmGeneral(
		H, initial, dt, 2, dimension, nullptr);
	bool isCorrect = bounded.krybasis.n_cols == 2;
	isCorrect &= std::isfinite(bounded.error_estimate);
	isCorrect &= arma::norm(
		bounded.krybasis.t() * bounded.krybasis -
			arma::eye<arma::cx_mat>(2, 2),
		"fro") < 1e-12;

	auto full = space.KrylovExpmGeneral(
		H, initial, dt, dimension, dimension, nullptr);
	arma::cx_mat exactFactor = arma::expmat(arma::cx_mat(H) * dt) * initial;
	arma::cx_mat exactDensity = exactFactor * exactFactor.t();
	isCorrect &= arma::norm(full.result - exactDensity, "fro") < 1e-11;

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////

// Add all the SpinAPI test cases
void AddSpinAPITests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("SpinAPI::Spin::S()", test_spinapi_spinclass_spin_quantum_number));
	_cases.push_back(test_case("SpinAPI::Spin::Multiplicity()", test_spinapi_spinclass_multiplicity_from_s));
	_cases.push_back(test_case("SpinAPI::Spin::Sx, Sy, Sz for spin 1/2", test_spinapi_spinclass_spinmatrices_spinonehalf));
	_cases.push_back(test_case("SpinAPI::Spin::Sx, Sy, Sz for spin 1", test_spinapi_spinclass_spinmatrices_spinone));
	_cases.push_back(test_case("SpinAPI::Spin::Sx, Sy, Sz for spin 3/2", test_spinapi_spinclass_spinmatrices_spinthreehalf));
	_cases.push_back(test_case("SpinAPI::Interaction static field and prefactor", test_spinapi_interaction_fieldstatic));
	_cases.push_back(test_case("SpinAPI::Interaction dynamic field (linear polarization)", test_spinapi_interaction_fieldlinearpolarization));
	_cases.push_back(test_case("SpinAPI::Interaction dynamic field (circular perpendicular polarization)", test_spinapi_interaction_fieldcircularpolarization_perpendicular));
	_cases.push_back(test_case("SpinAPI::Interaction dynamic field (circular tilted polarization)", test_spinapi_interaction_fieldcircularpolarization_tilted));
	_cases.push_back(test_case("SpinAPI::State basic tests", test_spinapi_state));
	_cases.push_back(test_case("SpinAPI::Tensor basic tests", test_spinapi_tensorclass_basics));
	_cases.push_back(test_case("Spin subspace functions - union of all subspaces", test_spinapi_subspacefuncs_union));
	_cases.push_back(test_case("Spin subspace functions - intersections of subspaces", test_spinapi_subspacefuncs_intersections));
	_cases.push_back(test_case("Spin subspace functions - intersections of subspaces 2", test_spinapi_subspacefuncs_intersections2));
	_cases.push_back(test_case("Spin subspace functions - completion from interaction", test_spinapi_subspacefuncs_extendbyinteraction));
	_cases.push_back(test_case("Spin subspace functions - completion from transition", test_spinapi_subspacefuncs_extendbytransition));
	_cases.push_back(test_case("Spin subspace functions - completion from state", test_spinapi_subspacefuncs_extendbystate));
	_cases.push_back(test_case("Spin subspace functions - completion from spin system", test_spinapi_subspacefuncs_extendbyspinsys));
	_cases.push_back(test_case("SpinSpace::CreateOperator - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_createoperator));
	_cases.push_back(test_case("SpinSpace::Hamiltonian - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_hamiltonian));
	_cases.push_back(test_case("SpinSpace::StaticHamiltonian - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_statichamiltonian));
	_cases.push_back(test_case("SpinSpace::DynamicHamiltonian - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_dynamichamiltonian));
	_cases.push_back(test_case("SpinSpace::InteractionOperator - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_interactionoperator));
	_cases.push_back(test_case("SpinSpace::OperatorToSuperspace - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_operatortosuperspace));
	_cases.push_back(test_case("SpinSpace::OperatorFromSuperspace - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_operatorfromsuperspace));
	_cases.push_back(test_case("SpinSpace::SuperoperatorFromOperators - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_superoperatorfromoperators));
	_cases.push_back(test_case("SpinSpace::SuperoperatorFromLeftOperator - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_superoperatorfromleftoperator));
	_cases.push_back(test_case("SpinSpace::SuperoperatorFromRightOperator - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_superoperatorfromrightoperator));
	_cases.push_back(test_case("SpinSpace::ReactionOperator - comparing sparse and dense version", test_spinapi_spinspace_sparsevsdense_reactionoperator));
	_cases.push_back(test_case("SpinAPI::SpinSpace basis reordering methods (dense matrix)", test_spinapi_reorderbasis_densematrix));
	_cases.push_back(test_case("SpinAPI::SpinSpace basis reordering methods (sparse matrix)", test_spinapi_reorderbasis_sparsematrix));
	_cases.push_back(test_case("SpinAPI::SpinSpace spin management (Add, Contains, Remove)", test_spinapi_spinspace_spinmanagement1));
	_cases.push_back(test_case("SpinAPI::SpinSpace spin management (Vector Add,Vector Contains, Clear)", test_spinapi_spinspace_spinmanagement2));
	_cases.push_back(test_case("SpinAPI::StateFunctions validating function parsing", test_function_finding));
	_cases.push_back(test_case("SpinAPI::StateFunctions grouped superposition factors", test_state_function_grouped_superposition));
	_cases.push_back(test_case("SpinAPI::StateFunctions EasySpin CISS density convention", test_state_easyspin_ciss_density_convention));
	_cases.push_back(test_case("SpinAPI::Functions validating function evaluation", test_function_evaluation));
	_cases.push_back(test_case("SpinAPI::Pulse InstantPulse", test_spinapi_instantpulse));
	_cases.push_back(test_case("SpinAPI::Pulse LongPulseStaticField", test_spinapi_longpulsestaticfield));
	_cases.push_back(test_case("SpinAPI::Pulse LongPulse", test_spinapi_longpulse));
	_cases.push_back(test_case("SpinAPI::Pulse group preserves spin name case", test_spinapi_pulse_group_preserves_spin_name_case));
	_cases.push_back(test_case("SpinAPI::PulseSequence preserves object name case", test_spinapi_pulse_sequence_preserves_object_name_case));
	_cases.push_back(test_case("SpinAPI::PulseSequence preserves multi-system operator index", test_spinapi_pulse_sequence_operator_preserves_system_index));
	_cases.push_back(test_case("SpinAPI::Interaction BroadbandField", test_spinapi_interaction_field_broadband));
	_cases.push_back(test_case("SpinAPI::Interaction OUGeneralField", test_spinapi_interaction_field_ornsteinuhlenbeck));
	_cases.push_back(test_case("SpinAPI::Interaction MonochromaticTensor", test_spinapi_interaction_tensor_monochromatic));
	_cases.push_back(test_case("SpinAPI::Interaction BroadbandTensor", test_spinapi_interaction_tensor_broadband));
	_cases.push_back(test_case("SpinAPI::Interaction OUGeneralTensor", test_spinapi_interaction_tensor_ornsteinuhlenbeck));
	_cases.push_back(test_case("SpinAPI::Interaction orientation validation and exchange", test_spinapi_interaction_orientation_validation_and_exchange));
	_cases.push_back(test_case("SpinAPI::Interaction strain component mapping", test_spinapi_strain_component_mapping));
	_cases.push_back(test_case("SpinAPI::PowderGrid weights and ZYZ rotation", test_spinapi_powder_grid_weights_and_rotation));
	_cases.push_back(test_case("SpinAPI::PowderGrid SOPHE and projection helpers", test_spinapi_powder_grid_sophe_projection_helpers));
	_cases.push_back(test_case("SpinSpace::Powder Hamiltonian helper matches explicit builders", test_spinapi_powder_hamiltonian_helper_matches_explicit_builders));
	_cases.push_back(test_case("SpinSpace::Zeeman orientation rotates g-tensor", test_spinapi_zeeman_orientation_rotates_gtensor));
	_cases.push_back(test_case("SpinSpace::Rotated Zeeman Hamiltonian follows powder orientation", test_spinapi_rotated_zeeman_hamiltonian_follows_powder_orientation));
	_cases.push_back(test_case("SpinSpace::ZFS formalism and orientation", test_spinapi_zfs_formalism_and_orientation));
	_cases.push_back(test_case("SpinSpace::Rotated quadratic spin identity powder", test_spinapi_rotated_quadraticspin_matches_plain_for_identity_powder));
	_cases.push_back(test_case("SpinSpace::Phenomenological relaxation operator", test_spinapi_phenomenological_relaxation_operator));
	_cases.push_back(test_case("SpinSpace::Phenomenological relaxation frame change is basis-local", test_spinapi_phenomenological_relaxation_framechange_is_basis_local));
	_cases.push_back(test_case("SpinSpace::Phenomenological rate2 preserves populations and trace", test_spinapi_phenomenological_rate2_preserves_populations_and_trace));
	_cases.push_back(test_case("SpinSpace::Powder phenomenological rate2 uses supplied eigenbasis", test_spinapi_powder_phenomenological_rate2_uses_supplied_eigenbasis));
	_cases.push_back(test_case("SpinSpace::Hilbert phenomenological rate2 uses supplied eigenbasis", test_spinapi_hilbert_phenomenological_rate2_uses_supplied_eigenbasis));
	_cases.push_back(test_case("SpinSpace::Hilbert phenomenological finite-step map matches superoperator", test_spinapi_hilbert_phenomenological_finite_step_map_matches_superoperator));
	_cases.push_back(test_case("SpinSpace::RotateState maps z population to x population", test_spinapi_rotate_state_maps_z_population_to_x_population));
	_cases.push_back(test_case("SpinSpace::RotateState leaves singlet invariant and rotates T0", test_spinapi_rotate_state_singlet_invariant_triplet_t0_rotates));
	_cases.push_back(test_case("SpinSpace::PrepareInitialDensityForPowder skips invariant rotations", test_spinapi_prepare_powder_initial_density_uses_rotation_invariance));
	_cases.push_back(test_case("SpinSpace::DephaseStateInEigenbasis removes Hamiltonian coherences", test_spinapi_dephase_state_in_eigenbasis_removes_hamiltonian_coherences));
	_cases.push_back(test_case("SpinSpace::Triplet keep retains free induction while dephase removes it", test_spinapi_triplet_keep_retains_free_induction_dephase_removes_it));
	_cases.push_back(test_case("SpinSpace::T1 relaxation exchanges spin-1 populations", test_spinapi_relaxation_t1_spin_one_population_exchange));
	_cases.push_back(test_case("SpinSpace::T2 relaxation is normalized pure dephasing", test_spinapi_relaxation_t2_is_normalized_pure_dephasing));
	_cases.push_back(test_case("SpinSpace::Random fields preserve trace and isotropic powder invariance", test_spinapi_relaxation_random_fields_are_trace_preserving_and_powder_invariant));
	_cases.push_back(test_case("SpinSpace::Secular isotropic hyperfine is SzIz", test_spinapi_secular_isotropic_hyperfine_is_sziz));
	_cases.push_back(test_case("SpinSpace::Secular hyperfine reversed order matches", test_spinapi_secular_hyperfine_reversed_order_matches));
	_cases.push_back(test_case("SpinSpace::Secular electron-electron historical diagonal projection", test_spinapi_secular_electron_electron_historical_diagonal_projection));
	_cases.push_back(test_case("SpinSpace::Krylov fixed dimension is respected", test_spinapi_krylov_general_respects_requested_dimension));
	_cases.push_back(test_case("SpinSpace::Krylov happy breakdown converges", test_spinapi_krylov_happy_breakdown_returns_exact_result));
	_cases.push_back(test_case("SpinSpace::Block Krylov is bounded and orthonormal", test_spinapi_block_krylov_is_bounded_and_orthonormal));
	_cases.push_back(test_case("SpinAPI::Operator copy preserves relaxation configuration", test_spinapi_operator_copy_preserves_relaxation_configuration));
}
//////////////////////////////////////////////////////////////////////////////
