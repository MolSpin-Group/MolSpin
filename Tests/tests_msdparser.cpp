//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Unit test functions for the MSDParser module.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////
#include "ObjectParser.h"
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::Get method for strings
bool test_msdparser_objectparser_getstring()
{
	// Setup objects for the test
	std::string name1 = "test1";
	std::string contents1 = "testvalue=en test-string;";
	MSDParser::ObjectParser parser1(name1, contents1);

	std::string name2 = "test2";
	std::string contents2 = "key1=value1;testvalue=en test-string;key3=true;";
	MSDParser::ObjectParser parser2(name2, contents2);

	std::string name3 = "test3";
	std::string contents3 = "k=k;b=2;spins(electron1,electron2)=|1/2,-1/2>;testvalue=en test-string;";
	MSDParser::ObjectParser parser3(name3, contents3);

	bool isCorrect = true;
	std::string str;
	std::string testvalue = "en test-string";

	// Perform the test
	isCorrect &= parser1.Get("testvalue", str);
	isCorrect &= (str.compare(testvalue) == 0);
	isCorrect &= parser2.Get("testvalue", str);
	isCorrect &= (str.compare(testvalue) == 0);
	isCorrect &= parser3.Get("testvalue", str);
	isCorrect &= (str.compare(testvalue) == 0);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::Get method for ints
bool test_msdparser_objectparser_getint()
{
	// Setup objects for the test
	std::string name1 = "test1";
	std::string contents1 = "testvalue=42;";
	MSDParser::ObjectParser parser1(name1, contents1);

	std::string name2 = "test2";
	std::string contents2 = "key1=value1;testvalue=-42;key3=true;";
	MSDParser::ObjectParser parser2(name2, contents2);

	std::string name3 = "test3";
	std::string contents3 = "k=k;b=2;spins(electron1,electron2)=|1/2,-1/2>;testvalue=42;";
	MSDParser::ObjectParser parser3(name3, contents3);

	bool isCorrect = true;
	int val;
	int testvalue = 42;

	// Perform the test
	isCorrect &= parser1.Get("testvalue", val);
	isCorrect &= (val == testvalue);
	isCorrect &= parser2.Get("testvalue", val);
	isCorrect &= (val == -testvalue);
	isCorrect &= parser3.Get("testvalue", val);
	isCorrect &= (val == testvalue);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::Get method for floating point numbers
bool test_msdparser_objectparser_getdouble()
{
	// Setup objects for the test
	std::string name1 = "test1";
	std::string contents1 = "testvalue=42.42;";
	MSDParser::ObjectParser parser1(name1, contents1);

	std::string name2 = "test2";
	std::string contents2 = "key1=value1;testvalue=-42.42;key3=true;";
	MSDParser::ObjectParser parser2(name2, contents2);

	std::string name3 = "test3";
	std::string contents3 = "k=k;b=2;spins(electron1,electron2)=|1/2,-1/2>;testvalue=42.42;";
	MSDParser::ObjectParser parser3(name3, contents3);

	bool isCorrect = true;
	double val;
	double testvalue = 42.42;

	// Perform the test
	isCorrect &= parser1.Get("testvalue", val);
	isCorrect &= equal_double(val, testvalue);
	isCorrect &= parser2.Get("testvalue", val);
	isCorrect &= equal_double(val, -testvalue);
	isCorrect &= parser3.Get("testvalue", val);
	isCorrect &= equal_double(val, testvalue);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::Get method for booleans
bool test_msdparser_objectparser_getbool()
{
	// Setup objects for the test
	std::string name1 = "test1";
	std::string contents1 = "testvalue=true;";
	MSDParser::ObjectParser parser1(name1, contents1);

	std::string name2 = "test2";
	std::string contents2 = "key1=value1;testvalue=yes;key3=false;";
	MSDParser::ObjectParser parser2(name2, contents2);

	std::string name3 = "test3";
	std::string contents3 = "k=k;b=no;spins(electron1,electron2)=|1/2,-1/2>;testvalue=1;";
	MSDParser::ObjectParser parser3(name3, contents3);

	bool isCorrect = true;
	bool val;

	// Perform the test
	isCorrect &= parser1.Get("testvalue", val);
	isCorrect &= val;
	isCorrect &= parser2.Get("testvalue", val);
	isCorrect &= val;
	isCorrect &= parser3.Get("testvalue", val);
	isCorrect &= val;
	isCorrect &= parser2.Get("key3", val);
	isCorrect &= !val;
	isCorrect &= parser3.Get("b", val);
	isCorrect &= !val;

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::Get method for vectors
bool test_msdparser_objectparser_getvector()
{
	// Setup objects for the test
	std::string name1 = "test1";
	std::string contents1 = "testvalue=0 1 0;";
	MSDParser::ObjectParser parser1(name1, contents1);

	std::string name2 = "test2";
	std::string contents2 = "key1=value1;testvalue=0 1 0;key3=1 1 1 1;";
	MSDParser::ObjectParser parser2(name2, contents2);

	std::string name3 = "test3";
	std::string contents3 = "k=k;b=1 0;spins(electron1,electron2)=|1/2,-1/2>;testvalue=0 1 0;";
	MSDParser::ObjectParser parser3(name3, contents3);

	bool isCorrect = true;
	arma::vec v;
	arma::vec testvec = arma::vec("0 1 0");

	// Perform the test
	isCorrect &= parser1.Get("testvalue", v);
	isCorrect &= equal_vec(v, testvec);
	isCorrect &= parser2.Get("testvalue", v);
	isCorrect &= equal_vec(v, testvec);
	isCorrect &= parser3.Get("testvalue", v);
	isCorrect &= equal_vec(v, testvec);
	isCorrect &= !parser2.Get("key3", v);
	isCorrect &= !parser3.Get("b", v);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::Get method for tensors
// DEPENDENCY NOTE: SpinAPI::Tensor
bool test_msdparser_objectparser_gettensor()
{
	// Setup objects for the test
	std::string name1 = "test1";
	std::string contents1 = "testvalue=isotropic(5);";
	MSDParser::ObjectParser parser1(name1, contents1);

	std::string name2 = "test2";
	std::string contents2 = "key1=value1;testvalue=anisotropic(5 5 5);key3=1 1 1 1;";
	MSDParser::ObjectParser parser2(name2, contents2);

	std::string name3 = "test3";
	std::string contents3 = "k=k;b=1 0;spins(electron1,electron2)=|1/2,-1/2>;testvalue=matrix(5 0 0;0 5 0;0 0 5);";
	MSDParser::ObjectParser parser3(name3, contents3);

	std::string name4 = "test4";
	std::string contents4 = "testvalue=matrix(5 0 0;0 0 5;0 5 0)+axis1(1 0 0)+axis2(0 0 1)+axis3(0 1 0);";
	MSDParser::ObjectParser parser4(name4, contents4);

	std::string name5 = "test5";
	std::string contents5 = "testvalue=anisotropic(1 1 1)+isotropic(4);";
	MSDParser::ObjectParser parser5(name5, contents5);

	bool isCorrect = true;
	SpinAPI::Tensor t(0);
	arma::mat testmat = arma::mat("-5 0 0;0 5 0;0 0 5");
	SpinAPI::Tensor testtensor = SpinAPI::Tensor(5.0);

	// Perform the test
	isCorrect &= parser1.Get("testvalue", t);
	isCorrect &= equal_tensor(t, testtensor);
	isCorrect &= parser2.Get("testvalue", t);
	isCorrect &= equal_tensor(t, testtensor);
	isCorrect &= parser3.Get("testvalue", t);
	isCorrect &= equal_tensor(t, testtensor);
	isCorrect &= parser4.Get("testvalue", t);
	isCorrect &= equal_matrices(t.LabFrame(), testmat);
	isCorrect &= parser5.Get("testvalue", t);
	isCorrect &= equal_tensor(t, testtensor);

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests that comma-separated string lists ignore whitespace around entries.
bool test_msdparser_objectparser_getlist_string_trims()
{
	MSDParser::ObjectParser parser("test", "values=alpha, beta ,\tgamma ;");

	std::vector<std::string> values;
	bool isCorrect = true;
	isCorrect &= parser.GetList("values", values, ',');
	isCorrect &= (values.size() == 3);
	if (values.size() == 3)
	{
		isCorrect &= (values[0] == "alpha");
		isCorrect &= (values[1] == "beta");
		isCorrect &= (values[2] == "gamma");
	}

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests the ObjectParser::GetPulseSequence method with the syntax used by MSD inputs
bool test_msdparser_objectparser_getpulsesequence_quoted()
{
	MSDParser::ObjectParser parser("test", "pulsesequence=[\"pulse1 0.1\"], ['pulse2 2.5'];");

	std::vector<std::tuple<std::string, double>> sequence;
	bool isCorrect = true;
	isCorrect &= parser.GetPulseSequence("pulsesequence", sequence);
	isCorrect &= (sequence.size() == 2);
	if (sequence.size() == 2)
	{
		isCorrect &= (std::get<0>(sequence[0]) == "pulse1");
		isCorrect &= equal_double(std::get<1>(sequence[0]), 0.1);
		isCorrect &= (std::get<0>(sequence[1]) == "pulse2");
		isCorrect &= equal_double(std::get<1>(sequence[1]), 2.5);
	}

	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Add all the MSDParser test cases
void AddMSDParserTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("MSDParser::ObjectParser::Get to get string", test_msdparser_objectparser_getstring));
	_cases.push_back(test_case("MSDParser::ObjectParser::Get to get integer", test_msdparser_objectparser_getint));
	_cases.push_back(test_case("MSDParser::ObjectParser::Get to get double", test_msdparser_objectparser_getdouble));
	_cases.push_back(test_case("MSDParser::ObjectParser::Get to get boolean", test_msdparser_objectparser_getbool));
	_cases.push_back(test_case("MSDParser::ObjectParser::Get to get vector", test_msdparser_objectparser_getvector));
	_cases.push_back(test_case("MSDParser::ObjectParser::Get to get tensor", test_msdparser_objectparser_gettensor));
	_cases.push_back(test_case("MSDParser::ObjectParser::GetList trims string entries", test_msdparser_objectparser_getlist_string_trims));
	_cases.push_back(test_case("MSDParser::ObjectParser::Get quoted pulse sequence", test_msdparser_objectparser_getpulsesequence_quoted));
}
//////////////////////////////////////////////////////////////////////////////
