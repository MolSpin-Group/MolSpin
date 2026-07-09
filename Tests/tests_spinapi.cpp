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
#include "State.h"
#include "Transition.h"
#include "SpinSystem.h"
#include "SpinSpace.h"
#include "Function.h"
#include "Pulse.h"
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
	double outofplane_angle = 45.0 / 180.0 * arma::datum::pi;

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
	_cases.push_back(test_case("SpinAPI::Functions validating function evaluation", test_function_evaluation));
	_cases.push_back(test_case("SpinAPI::Pulse InstantPulse", test_spinapi_instantpulse));
	_cases.push_back(test_case("SpinAPI::Pulse LongPulseStaticField", test_spinapi_longpulsestaticfield));
	_cases.push_back(test_case("SpinAPI::Pulse LongPulse", test_spinapi_longpulse));
	_cases.push_back(test_case("SpinAPI::Interaction BroadbandField", test_spinapi_interaction_field_broadband));
	_cases.push_back(test_case("SpinAPI::Interaction OUGeneralField", test_spinapi_interaction_field_ornsteinuhlenbeck));
	_cases.push_back(test_case("SpinAPI::Interaction MonochromaticTensor", test_spinapi_interaction_tensor_monochromatic));
	_cases.push_back(test_case("SpinAPI::Interaction BroadbandTensor", test_spinapi_interaction_tensor_broadband));
	_cases.push_back(test_case("SpinAPI::Interaction OUGeneralTensor", test_spinapi_interaction_tensor_ornsteinuhlenbeck));
}
//////////////////////////////////////////////////////////////////////////////
