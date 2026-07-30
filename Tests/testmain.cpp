//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Note: Strictly speaking the tests here are not unit tests, since some test
// functions contain more than a single test, and since most classes/methods/
// functions are not tested in complete isolation. But the tests are still
// testing various crucial functionalities.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cstdio>
#include <vector>
#include <armadillo>
#include <unistd.h>

#include "Tensor.h"
#include "RunSection.h"
//////////////////////////////////////////////////////////////////////////////
using test_ptr = bool (*)();						// Function pointer alias
using test_case = std::pair<std::string, test_ptr>; // Function pointer and name
//////////////////////////////////////////////////////////////////////////////
// Include various testing functions
#include "assertfunctions.cpp"
//////////////////////////////////////////////////////////////////////////////
// Files with the other test functions
#include "tests_spinapi.cpp"
#include "tests_msdparser.cpp"
#include "tests_actions.cpp"
#include "tests_TaskStaticHSSymmetricDecay.cpp"
#include "tests_TaskStaticSS.cpp"
#include "tests_TaskMultiStaticSS.cpp"
#include "tests_TaskStaticRPOnlyHSSymDec.cpp"
#include "tests_TaskStaticSSSpectra.cpp"
#include "tests_TaskStaticHSTrEPRSpectra.cpp"
#include "tests_TaskStaticPowderSpectra.cpp"
#include "tests_utility.cpp"
//////////////////////////////////////////////////////////////////////////////
// CMake and the normal Makefile target run the complete suite. Developers can
// override these defaults when compiling a focused testmain executable.
#ifndef SPINAPI_TEST
#define SPINAPI_TEST 1
#endif
#ifndef MSDPARSER_TEST
#define MSDPARSER_TEST 1
#endif
#ifndef ACTION_TEST
#define ACTION_TEST 1
#endif
#ifndef STATICHSSDECAY_TEST
#define STATICHSSDECAY_TEST 1
#endif
#ifndef STATICSS_TEST
#define STATICSS_TEST 1
#endif
#ifndef MULTISTATICSS_TEST
#define MULTISTATICSS_TEST 1
#endif
#ifndef STATICRPONLY_TEST
#define STATICRPONLY_TEST 1
#endif
#ifndef STATICSSSPECTRA_TEST
#define STATICSSSPECTRA_TEST 1
#endif
#ifndef STATICHSTEPR_TEST
#define STATICHSTEPR_TEST 1
#endif
#ifndef STATICPOWDERSPECTRA_TEST
#define STATICPOWDERSPECTRA_TEST 1
#endif
#ifndef UTIL_TEST
#define UTIL_TEST 1
#endif
//////////////////////////////////////////////////////////////////////////////
std::string read_captured_stream(std::FILE *file)
{
	if (file == nullptr)
		return "";

	std::fflush(file);
	std::rewind(file);

	std::string output;
	char buffer[4096];
	size_t bytes_read = 0;
	while ((bytes_read = std::fread(buffer, 1, sizeof(buffer), file)) > 0)
		output.append(buffer, bytes_read);

	return output;
}
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	std::cout << "# -------------------------------------------------------" << std::endl;
	std::cout << "# Molecular Spin Dynamics" << std::endl;
	std::cout << "# " << std::endl;
	std::cout << "# Developed 2017-2019 by Claus Nielsen and 2021-2022 by Luca Gerhards." << std::endl;
	std::cout << "# (c) Quantum Biology and Computational Physics Group," << std::endl;
	std::cout << "# Carl von Ossietzky University of Oldenburg." << std::endl;
	std::cout << "# For more information see www.molspin.eu" << std::endl;
	std::cout << "# -------------------------------------------------------" << std::endl;
	std::cout << "# Unit Testing Module" << std::endl;
	std::cout << "# -------------------------------------------------------" << std::endl;

	// Collections of test cases
	std::vector<test_case> cases;
	std::vector<test_case> failed_cases;

	// Add test cases to the list


#if SPINAPI_TEST == 1
	AddSpinAPITests(cases);
#endif
#if MSDPARSER_TEST == 1 
	AddMSDParserTests(cases);
#endif
#if ACTION_TEST == 1 
	AddActionsTests(cases);
#endif
#if STATICHSSDECAY_TEST == 1 
	AddTaskStaticHSSymmetricDecayTests(cases);
#endif
#if STATICSS_TEST == 1
	AddTaskStaticSSTests(cases);
#endif
#if MULTISTATICSS_TEST == 1
	AddTaskMultiStaticSSTests(cases);
#endif
#if STATICRPONLY_TEST == 1 
	AddTaskStaticRPOnlyHSSymDecTests(cases);
#endif
#if STATICSSSPECTRA_TEST == 1 
	AddTaskStaticSSSpectraTests(cases);
#endif
#if STATICHSTEPR_TEST == 1 
	AddTaskStaticHSTrEPRSpectraTests(cases);
#endif
#if STATICPOWDERSPECTRA_TEST == 1 
	AddTaskStaticPowderSpectraTests(cases);
#endif
#if UTIL_TEST == 1 
	AddUtiltiyTests(cases);
#endif
	

	// Loop through all test cases and test them
	for (auto i = cases.cbegin(); i != cases.cend(); i++)
	{
		std::cout << "Running test \"" << i->first << "\" ......... " << std::flush;

		std::FILE *captured_stdout = std::tmpfile();
		std::FILE *captured_stderr = std::tmpfile();
		int original_stdout = -1;
		int original_stderr = -1;
		bool capture_active = false;
		if (captured_stdout != nullptr && captured_stderr != nullptr)
		{
			std::cout.flush();
			std::cerr.flush();
			std::fflush(stdout);
			std::fflush(stderr);
			original_stdout = dup(STDOUT_FILENO);
			original_stderr = dup(STDERR_FILENO);
			capture_active = (original_stdout >= 0 && original_stderr >= 0 &&
							  dup2(fileno(captured_stdout), STDOUT_FILENO) >= 0 &&
							  dup2(fileno(captured_stderr), STDERR_FILENO) >= 0);
		}

		bool passed_test = i->second();

		if (capture_active)
		{
			std::cout.flush();
			std::cerr.flush();
			std::fflush(stdout);
			std::fflush(stderr);
			dup2(original_stdout, STDOUT_FILENO);
			dup2(original_stderr, STDERR_FILENO);
		}
		if (original_stdout >= 0)
			close(original_stdout);
		if (original_stderr >= 0)
			close(original_stderr);

		if (passed_test)
		{
			std::cout << "PASSED!";
		}
		else
		{
			std::cout << "FAILED!";
			failed_cases.push_back(*i);
		}
		std::cout << std::endl;

		if (!passed_test)
		{
			std::string captured_stdout_content = read_captured_stream(captured_stdout);
			std::string captured_stderr_content = read_captured_stream(captured_stderr);
			if (!captured_stdout_content.empty())
				std::cout << captured_stdout_content;
			if (!captured_stderr_content.empty())
				std::cerr << captured_stderr_content;
		}

		if (captured_stdout != nullptr)
			std::fclose(captured_stdout);
		if (captured_stderr != nullptr)
			std::fclose(captured_stderr);
	}

	// Get number of passed and failed tests
	int failed = failed_cases.size();
	int passed = cases.size() - failed;

	// Get summary at the end
	std::cout << "# -------------------------------------------------------" << std::endl;
	std::cout << "Testing done!\nPassed:  " << passed << "\nFailed:  " << failed << "\nTotal:  " << cases.size() << std::endl;

	// Notify which cases has failed
	if (failed > 0)
	{
		std::cout << "\nThere were failed test cases:" << std::endl;
		for (auto i = failed_cases.cbegin(); i != failed_cases.cend(); i++)
			std::cout << " - " << i->first << std::endl;
	}

	return failed == 0 ? 0 : 1;
}
//////////////////////////////////////////////////////////////////////////////
