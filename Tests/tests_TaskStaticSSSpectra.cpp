//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Tests the Static Superspace method.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include "TaskStaticSSSpectra.h"

namespace
{
	std::vector<std::string> FilterSpinProjectionOutput(const std::vector<std::string> &_tokens, size_t _expected_size)
	{
		if (_tokens.size() <= _expected_size || _expected_size < 2)
			return _tokens;

		size_t expected_values = _expected_size - 2;
		size_t groups = expected_values / 3;
		if (groups == 0)
			return _tokens;

		std::vector<std::string> filtered;
		filtered.reserve(_expected_size);
		filtered.push_back(_tokens[0]);
		filtered.push_back(_tokens[1]);

		size_t idx = 2;
		for (size_t g = 0; g < groups; ++g)
		{
			if (idx + 2 >= _tokens.size())
				break;
			filtered.push_back(_tokens[idx]);
			filtered.push_back(_tokens[idx + 1]);
			filtered.push_back(_tokens[idx + 2]);
			idx += 5; // Ix, Iy, Iz, Ip, Im
		}

		return filtered;
	}
}

//////////////////////////////////////////////////////////////////////////////
// Tests a single calculation Method=timeinf CIDSP=true without any Actions
bool test_task_staticssspectra_method_timeinf_cidsp_true()
{
	// Setup objects for the test
	// Spins
	auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;tensor=isotropic(-2);");
	auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;tensor=isotropic(-2);");
	auto spin3 = std::make_shared<SpinAPI::Spin>("nucleus1", "spin=1/2;tensor=isotropic(1);");
	auto spin4 = std::make_shared<SpinAPI::Spin>("nucleus2", "spin=1/2;tensor=isotropic(1);");

	// Interactions
	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=hyperfine;group1=electron1;group2=nucleus1;tensor=isotropic(5e-4);prefactor=2.0023;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=hyperfine;group1=electron2;group2=nucleus2;tensor=anisotropic(1e-4, 1e-4, 1e-3);prefactor=2.0023;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=zeeman;spins=electron1,electron2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=-176.085;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=zeeman;spins=nucleus1,nucleus2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=0.267522;");

	// States
	auto state1 = std::make_shared<SpinAPI::State>("state1", "spins(electron1,electron2)=|1/2,-1/2>-|-1/2,1/2>;"); // Singlet
	auto state2 = std::make_shared<SpinAPI::State>("state2", "spin(electron1)=|1/2>;spin(electron2)=|1/2>;");	   // |T+>
	auto state3 = std::make_shared<SpinAPI::State>("state3", "");												   // Identity

	// SpinSystem
	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state1);
	spinsys->Add(state2);
	spinsys->Add(state3);
	spinsys->Add(interaction1);
	spinsys->Add(interaction2);
	spinsys->Add(interaction3);
	spinsys->Add(interaction4);
	spinsys->ValidateInteractions();
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	// Transition
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "type=sink;sourcestate=state3;rate=1e-4;", spinsys);
	spinsys->Add(transition1);

	// Add an ObjectParser with settings to the SpinSystem
	auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=state1;");
	spinsys->SetProperties(spinsysParser);

	// A RunSection to run the calculation
	RunSection::RunSection rs;
	rs.Add(spinsys);

	// Create a task and get a pointer to it
	std::string taskname = "testtask";
	MSDParser::ObjectParser taskParser(taskname, "type=staticss-spectra;method=timeinf;cidsp=true;spinlist=nucleus1,nucleus2;");
	rs.Add(MSDParser::ObjectType::Task, taskParser);
	auto task = rs.GetTask(taskname);

	// Set the Log and Data streams to something we can read off within this function
	std::ostringstream logstream;
	std::ostringstream datastream;
	task->SetLogStream(logstream);
	task->SetDataStream(datastream);

	// "Gold values", i.e. correct results to test against.
	std::string value1 = "1 inf 0 0 3.09231e-17 0 0 -3.76134e-17";

	// Convert value1 into a vector
	std::vector<std::string> result;
	std::istringstream stream(value1);
	for (std::string s; std::getline(stream, s, ' ');)
		result.push_back(s);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state1->ParseFromSystem(*spinsys);							// Get a valid state object; Singlet
	isCorrect &= state2->ParseFromSystem(*spinsys);							// Get a valid state object; |T+>
	isCorrect &= state3->ParseFromSystem(*spinsys);							// Get a valid state object; Identity
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0); // Put state object into transition
	isCorrect &= rs.Run(1);

	// Remove header from first run
	std::string result_string = datastream.str();
	auto lb = result_string.find("\n");
	if (lb != std::string::npos && lb < result_string.size() - 1)
	{
		result_string.erase(0, lb + 1);
	}

	// Convert result_string into a vector
	std::vector<std::string> result_vec_string;
	std::istringstream stream1(result_string);
	for (std::string s; std::getline(stream1, s, ' ');)
		result_vec_string.push_back(s);
	result_vec_string = FilterSpinProjectionOutput(result_vec_string, result.size());

	for (auto i = 0; i < (int)result.size(); i++)
	{
		if (i == 1)
		{
			isCorrect &= result[i] == result_vec_string[i];
		}
		else
		{
			isCorrect &= equal_doublesfromstring(result[i], result_vec_string[i]);
		}
	}

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests a single calculation Method=timeinf CIDSP=false without any Actions
bool test_task_staticssspectra_method_timeinf_cidsp_false()
{
	// Setup objects for the test
	// Spins
	auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;tensor=isotropic(-2);");
	auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;tensor=isotropic(-2);");
	auto spin3 = std::make_shared<SpinAPI::Spin>("nucleus1", "spin=1/2;tensor=isotropic(1);");
	auto spin4 = std::make_shared<SpinAPI::Spin>("nucleus2", "spin=1/2;tensor=isotropic(1);");

	// Interactions
	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=hyperfine;group1=electron1;group2=nucleus1;tensor=isotropic(5e-4);prefactor=2.0023;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=hyperfine;group1=electron2;group2=nucleus2;tensor=anisotropic(1e-4, 1e-4, 1e-3);prefactor=2.0023;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=zeeman;spins=electron1,electron2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=-176.085;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=zeeman;spins=nucleus1,nucleus2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=0.267522;");

	// States
	auto state1 = std::make_shared<SpinAPI::State>("state1", "spins(electron1,electron2)=|1/2,-1/2>-|-1/2,1/2>;"); // Singlet
	auto state2 = std::make_shared<SpinAPI::State>("state2", "spin(electron1)=|1/2>;spin(electron2)=|1/2>;");	   // |T+>
	auto state3 = std::make_shared<SpinAPI::State>("state3", "");												   // Identity

	// SpinSystem
	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state1);
	spinsys->Add(state2);
	spinsys->Add(state3);
	spinsys->Add(interaction1);
	spinsys->Add(interaction2);
	spinsys->Add(interaction3);
	spinsys->Add(interaction4);
	spinsys->ValidateInteractions();
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	// Transition
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "type=sink;sourcestate=state3;rate=1e-4;", spinsys);
	spinsys->Add(transition1);

	// Add an ObjectParser with settings to the SpinSystem
	auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=state1;");
	spinsys->SetProperties(spinsysParser);

	// A RunSection to run the calculation
	RunSection::RunSection rs;
	rs.Add(spinsys);

	// Create a task and get a pointer to it
	std::string taskname = "testtask";
	MSDParser::ObjectParser taskParser(taskname, "type=staticss-spectra;method=timeinf;cidsp=false;spinlist=nucleus1,nucleus2;");
	rs.Add(MSDParser::ObjectType::Task, taskParser);
	auto task = rs.GetTask(taskname);

	// Set the Log and Data streams to something we can read off within this function
	std::ostringstream logstream;
	std::ostringstream datastream;
	task->SetLogStream(logstream);
	task->SetDataStream(datastream);

	// "Gold values", i.e. correct results to test against.
	std::string value1 = "1 inf 0 0 4.45899e-13 0 0 -3.50155e-13";

	// Convert value1 into a vector
	std::vector<std::string> result;
	std::istringstream stream(value1);
	for (std::string s; std::getline(stream, s, ' ');)
		result.push_back(s);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state1->ParseFromSystem(*spinsys);							// Get a valid state object; Singlet
	isCorrect &= state2->ParseFromSystem(*spinsys);							// Get a valid state object; |T+>
	isCorrect &= state3->ParseFromSystem(*spinsys);							// Get a valid state object; Identity
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0); // Put state object into transition
	isCorrect &= rs.Run(1);													// Run a calculation

	// Remove header from first run
	std::string result_string = datastream.str();
	auto lb = result_string.find("\n");
	if (lb != std::string::npos && lb < result_string.size() - 1)
	{
		result_string.erase(0, lb + 1);
	}

	// Convert result_string into a vector
	std::vector<std::string> result_vec_string;
	std::istringstream stream1(result_string);
	for (std::string s; std::getline(stream1, s, ' ');)
		result_vec_string.push_back(s);
	result_vec_string = FilterSpinProjectionOutput(result_vec_string, result.size());

	for (auto i = 0; i < (int)result.size(); i++)
	{
		if (i == 1)
		{
			isCorrect &= result[i] == result_vec_string[i];
		}
		else
		{
			isCorrect &= equal_doublesfromstring(result[i], result_vec_string[i]);
		}
	}

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests a single calculation Method=timeinf CIDSP=false with Instantpulse without any Actions
bool test_task_staticssspectra_method_timeinf_cidsp_false_instant_pulse()
{
	// Setup objects for the test
	// Spins
	auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;tensor=isotropic(-2);");
	auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;tensor=isotropic(-2);");
	auto spin3 = std::make_shared<SpinAPI::Spin>("nucleus1", "spin=1/2;tensor=isotropic(1);");
	auto spin4 = std::make_shared<SpinAPI::Spin>("nucleus2", "spin=1/2;tensor=isotropic(1);");

	// Interactions
	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=hyperfine;group1=electron1;group2=nucleus1;tensor=isotropic(5e-4);prefactor=2.0023;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=hyperfine;group1=electron2;group2=nucleus2;tensor=anisotropic(1e-4, 1e-4, 1e-3);prefactor=2.0023;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=zeeman;spins=electron1,electron2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=-176.085;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=zeeman;spins=nucleus1,nucleus2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=0.267522;");

	// States
	auto state1 = std::make_shared<SpinAPI::State>("state1", "spins(electron1,electron2)=|1/2,-1/2>-|-1/2,1/2>;spin(nucleus1)=|1/2>;spin(nucleus2)=|1/2>;"); // Singlet
	auto state2 = std::make_shared<SpinAPI::State>("state2", "spin(electron1)=|1/2>;spin(electron2)=|1/2>;");												 // |T+>
	auto state3 = std::make_shared<SpinAPI::State>("state3", "");																							 // Identity

	// SpinSystem
	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state1);
	spinsys->Add(state2);
	spinsys->Add(state3);
	spinsys->Add(interaction1);
	spinsys->Add(interaction2);
	spinsys->Add(interaction3);
	spinsys->Add(interaction4);
	spinsys->ValidateInteractions();
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	// Transition
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "type=sink;sourcestate=state3;rate=1e-4;", spinsys);
	spinsys->Add(transition1);

	// Add an PulseObject with srttings to the SpinSystem
	auto pulse1 = std::make_shared<SpinAPI::Pulse>("pulse1", "type=instantpulse;group=nucleus1,nucleus2;rotationaxis=1 0 0;angle=90.0;");
	spinsys->Add(pulse1);
	spinsys->ValidatePulses();

	// Add an ObjectParser with settings to the SpinSystem
	auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=state1;");
	spinsys->SetProperties(spinsysParser);

	// A RunSection to run the calculation
	RunSection::RunSection rs;
	rs.Add(spinsys);

	// Create a task and get a pointer to it
	std::string taskname = "testtask";
	MSDParser::ObjectParser taskParser(taskname, "type=staticss-spectra;method=timeinf;cidsp=false;spinlist=nucleus1,nucleus2;pulsesequence=[pulse1 0.1];printtimeframe=freeevo;");
	rs.Add(MSDParser::ObjectType::Task, taskParser);
	auto task = rs.GetTask(taskname);

	// Set the Log and Data streams to something we can read off within this function
	std::ostringstream logstream;
	std::ostringstream datastream;
	task->SetLogStream(logstream);
	task->SetDataStream(datastream);

	// "Gold values", i.e. correct results to test against.
	std::string value1 = "1 inf 6.58402 0.0375752 1.29785e-12 -5.10086 0.0357764 1.21895e-12";

	// Convert value1 into a vector
	std::vector<std::string> result;
	std::istringstream stream(value1);
	for (std::string s; std::getline(stream, s, ' ');)
		result.push_back(s);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state1->ParseFromSystem(*spinsys);							// Get a valid state object; Singlet
	isCorrect &= state2->ParseFromSystem(*spinsys);							// Get a valid state object; |T+>
	isCorrect &= state3->ParseFromSystem(*spinsys);							// Get a valid state object; Identity
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0); // Put state object into transition
	isCorrect &= rs.Run(1);													// Run a calculation

	// Remove header from first run
	std::string result_string = datastream.str();
	auto lb = result_string.find("\n");
	if (lb != std::string::npos && lb < result_string.size() - 1)
	{
		result_string.erase(0, lb + 1);
	}

	// Convert result_string into a vector
	std::vector<std::string> result_vec_string;
	std::istringstream stream1(result_string);
	for (std::string s; std::getline(stream1, s, ' ');)
		result_vec_string.push_back(s);
	result_vec_string = FilterSpinProjectionOutput(result_vec_string, result.size());

	// for (const std::string& str : result_vec_string) {
    //     std::cout << str << " ";
    // }

	for (auto i = 0; i < (int)result.size(); i++)
	{
		if (i == 1)
		{
			isCorrect &= result[i] == result_vec_string[i];
		}
		else
		{
			isCorrect &= equal_doublesfromstring(result[i], result_vec_string[i]);
		}
	}

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests a single calculation Method=timeinf CIDSP=false with LongPulseStaticField without any Actions
bool test_task_staticssspectra_method_timeinf_cidsp_false_longpulsestaticfield_pulse()
{
	// Setup objects for the test
	// Spins
	auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;tensor=isotropic(-2);");
	auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;tensor=isotropic(-2);");
	auto spin3 = std::make_shared<SpinAPI::Spin>("nucleus1", "spin=1/2;tensor=isotropic(1);");
	auto spin4 = std::make_shared<SpinAPI::Spin>("nucleus2", "spin=1/2;tensor=isotropic(1);");

	// Interactions
	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=hyperfine;group1=electron1;group2=nucleus1;tensor=isotropic(5e-4);prefactor=2.0023;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=hyperfine;group1=electron2;group2=nucleus2;tensor=anisotropic(1e-4, 1e-4, 1e-3);prefactor=2.0023;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=zeeman;spins=electron1,electron2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=-176.085;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=zeeman;spins=nucleus1,nucleus2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=0.267522;");

	// States
	auto state1 = std::make_shared<SpinAPI::State>("state1", "spins(electron1,electron2)=|1/2,-1/2>-|-1/2,1/2>;spin(nucleus1)=|1/2>;spin(nucleus2)=|1/2>;"); // Singlet
	auto state2 = std::make_shared<SpinAPI::State>("state2", "spin(electron1)=|1/2>;spin(electron2)=|1/2>;");												 // |T+>
	auto state3 = std::make_shared<SpinAPI::State>("state3", "");																							 // Identity

	// SpinSystem
	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state1);
	spinsys->Add(state2);
	spinsys->Add(state3);
	spinsys->Add(interaction1);
	spinsys->Add(interaction2);
	spinsys->Add(interaction3);
	spinsys->Add(interaction4);
	spinsys->ValidateInteractions();
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	// Transition
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "type=sink;sourcestate=state3;rate=1e-4;", spinsys);
	spinsys->Add(transition1);

	// Add an PulseObject with srttings to the SpinSystem
	auto pulse1 = std::make_shared<SpinAPI::Pulse>("pulse2", "type=longpulsestaticfield;group=nucleus1,nucleus2;field=0.1 0.0 -0.1;pulsetime=58.7;prefactorlist=0.267522,0.267522,0.267522,0.267522,0.267522,0.267522;commonprefactorlist=false,false;ignoretensorslist=true,true;timestep=0.1;");
	spinsys->Add(pulse1);
	spinsys->ValidatePulses();

	// Add an ObjectParser with settings to the SpinSystem
	auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=state1;");
	spinsys->SetProperties(spinsysParser);

	// A RunSection to run the calculation
	RunSection::RunSection rs;
	rs.Add(spinsys);

	// Create a task and get a pointer to it
	std::string taskname = "testtask";
	MSDParser::ObjectParser taskParser(taskname, "type=staticss-spectra;method=timeinf;cidsp=false;spinlist=nucleus1,nucleus2;pulsesequence=[pulse2 0.1];printtimeframe=freeevo;");
	rs.Add(MSDParser::ObjectType::Task, taskParser);
	auto task = rs.GetTask(taskname);

	// Set the Log and Data streams to something we can read off within this function
	std::ostringstream logstream;
	std::ostringstream datastream;
	task->SetLogStream(logstream);
	task->SetDataStream(datastream);

	// "Gold values", i.e. correct results to test against.
	std::string value1 = "1 inf 4.0022 3.5372 2724.1 1.92041 2.79378 4496.78";

	// Convert value1 into a vector
	std::vector<std::string> result;
	std::istringstream stream(value1);
	for (std::string s; std::getline(stream, s, ' ');)
		result.push_back(s);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state1->ParseFromSystem(*spinsys);							// Get a valid state object; Singlet
	isCorrect &= state2->ParseFromSystem(*spinsys);							// Get a valid state object; |T+>
	isCorrect &= state3->ParseFromSystem(*spinsys);							// Get a valid state object; Identity
	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0); // Put state object into transition
	isCorrect &= rs.Run(1);													// Run a calculation

	// Remove header from first run
	std::string result_string = datastream.str();
	auto lb = result_string.find("\n");
	if (lb != std::string::npos && lb < result_string.size() - 1)
	{
		result_string.erase(0, lb + 1);
	}

	// Convert result_string into a vector
	std::vector<std::string> result_vec_string;
	std::istringstream stream1(result_string);
	for (std::string s; std::getline(stream1, s, ' ');)
		result_vec_string.push_back(s);
	result_vec_string = FilterSpinProjectionOutput(result_vec_string, result.size());

	// for (const std::string& str : result_vec_string) {
    //     std::cout << str << " ";
    // }

	for (auto i = 0; i < (int)result.size(); i++)
	{
		// First, check if the current value is "inf" and handle it separately
		if (result[i] == "inf" || result_vec_string[i] == "inf")
		{
			// If both values are "inf", they are considered equal
			isCorrect &= (result[i] == result_vec_string[i]);
		}
		else
		{
			// Convert strings to doubles for comparison
			try
			{
				double result_value = std::stod(result[i]);
				double result_vec_value = std::stod(result_vec_string[i]);

				// Only compare values greater than 1e-10
				if (std::abs(result_value) > 1e-10 || std::abs(result_vec_value) > 1e-10)
				{
					isCorrect &= equal_doublesfromstring(result[i], result_vec_string[i]);
				}
			}
			catch (const std::invalid_argument &)
			{
				// Handle the case where the string cannot be converted to a double
				isCorrect = false;
			}
		}
	}

	// Return the result
	return isCorrect;
}
//////////////////////////////////////////////////////////////////////////////
// Tests a single calculation Method=timeinf CIDSP=false with LongPulse without any Actions
#include <iostream> // Include this to print debug information

bool test_task_staticssspectra_method_timeinf_cidsp_false_longpulse_pulse()
{
	// Setup objects for the test
	// Spins
	auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;tensor=isotropic(-2);");
	auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;tensor=isotropic(-2);");
	auto spin3 = std::make_shared<SpinAPI::Spin>("nucleus1", "spin=1/2;tensor=isotropic(1);");
	auto spin4 = std::make_shared<SpinAPI::Spin>("nucleus2", "spin=1/2;tensor=isotropic(1);");

	// Interactions
	auto interaction1 = std::make_shared<SpinAPI::Interaction>("interaction1", "type=hyperfine;group1=electron1;group2=nucleus1;tensor=isotropic(5e-4);prefactor=2.0023;");
	auto interaction2 = std::make_shared<SpinAPI::Interaction>("interaction2", "type=hyperfine;group1=electron2;group2=nucleus2;tensor=anisotropic(1e-4, 1e-4, 1e-3);prefactor=2.0023;");
	auto interaction3 = std::make_shared<SpinAPI::Interaction>("interaction3", "type=zeeman;spins=electron1,electron2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=-176.085;");
	auto interaction4 = std::make_shared<SpinAPI::Interaction>("interaction4", "type=zeeman;spins=nucleus1,nucleus2;field=0 0 0.5;ignoretensors=true;commonprefactor=false;prefactor=0.267522;");

	// States
	auto state1 = std::make_shared<SpinAPI::State>("state1", "spins(electron1,electron2)=|1/2,-1/2>-|-1/2,1/2>;spin(nucleus1)=|1/2>;spin(nucleus2)=|1/2>;"); // Singlet
	auto state2 = std::make_shared<SpinAPI::State>("state2", "spin(electron1)=|1/2>;spin(electron2)=|1/2>;");												 // |T+>
	auto state3 = std::make_shared<SpinAPI::State>("state3", "");																							 // Identity

	// SpinSystem
	auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
	spinsys->Add(spin1);
	spinsys->Add(spin2);
	spinsys->Add(spin3);
	spinsys->Add(spin4);
	spinsys->Add(state1);
	spinsys->Add(state2);
	spinsys->Add(state3);
	spinsys->Add(interaction1);
	spinsys->Add(interaction2);
	spinsys->Add(interaction3);
	spinsys->Add(interaction4);
	spinsys->ValidateInteractions();
	std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
	spinsystems.push_back(spinsys);

	// Transition
	auto transition1 = std::make_shared<SpinAPI::Transition>("transition1", "type=sink;sourcestate=state3;rate=1e-4;", spinsys);
	spinsys->Add(transition1);

	// Add a PulseObject with settings to the SpinSystem
	auto pulse1 = std::make_shared<SpinAPI::Pulse>("pulse3", "type=longpulse;group=nucleus1,nucleus2;field=0.00005 0.00005 0.0;pulsetime=5.9;prefactorlist=0.267522,0.267522,0.267522,0.267522,0.267522,0.267522;commonprefactorlist=false,false;ignoretensorslist=true,true;timestep=0.1;frequency=0.0004224;");
	spinsys->Add(pulse1);
	spinsys->ValidatePulses();

	// Add an ObjectParser with settings to the SpinSystem
	auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=state1;");
	spinsys->SetProperties(spinsysParser);

	// A RunSection to run the calculation
	RunSection::RunSection rs;
	rs.Add(spinsys);

	// Create a task and get a pointer to it
	std::string taskname = "testtask";
	MSDParser::ObjectParser taskParser(taskname, "type=staticss-spectra;method=timeinf;cidsp=false;spinlist=nucleus1,nucleus2;pulsesequence=[pulse3 0.1];printtimeframe=freeevo;");
	rs.Add(MSDParser::ObjectType::Task, taskParser);
	auto task = rs.GetTask(taskname);

	// Set the Log and Data streams to something we can read off within this function
	std::ostringstream logstream;
	std::ostringstream datastream;
	task->SetLogStream(logstream);
	task->SetDataStream(datastream);

	// The long RF pulse should not destroy the z-polarized nuclear population.
	// With a 1e-4 sink rate, a spin-up nucleus gives an integrated Iz yield of
	// about 0.5 / 1e-4 = 5000, while transverse components remain tiny.
	std::string value1 = "1 inf 0 0 5000 0 0 5000";

	// Convert value1 into a vector
	std::vector<std::string> result;
	std::istringstream stream(value1);
	for (std::string s; std::getline(stream, s, ' ');)
		result.push_back(s);

	bool isCorrect = true;

	// Perform the test
	isCorrect &= state1->ParseFromSystem(*spinsys);
	// std::cout << "State1 Parse: " << (isCorrect ? "Pass" : "Fail") << std::endl;

	isCorrect &= state2->ParseFromSystem(*spinsys);
	// std::cout << "State2 Parse: " << (isCorrect ? "Pass" : "Fail") << std::endl;

	isCorrect &= state3->ParseFromSystem(*spinsys);
	// std::cout << "State3 Parse: " << (isCorrect ? "Pass" : "Fail") << std::endl;

	isCorrect &= ((spinsys->ValidateTransitions(spinsystems)).size() == 0);
	// std::cout << "Transitions Validate: " << (isCorrect ? "Pass" : "Fail") << std::endl;

	isCorrect &= rs.Run(1);
	// std::cout << "Run Section: " << (isCorrect ? "Pass" : "Fail") << std::endl;

	// Remove header from first run
	std::string result_string = datastream.str();
	auto lb = result_string.find("\n");
	if (lb != std::string::npos && lb < result_string.size() - 1)
	{
		result_string.erase(0, lb + 1);
	}

	// Convert result_string into a vector
	std::vector<std::string> result_vec_string;
	std::istringstream stream1(result_string);
	for (std::string s; std::getline(stream1, s, ' ');)
		result_vec_string.push_back(s);
	result_vec_string = FilterSpinProjectionOutput(result_vec_string, result.size());

	if (result_vec_string.size() != result.size())
	{
		return false;
	}

	isCorrect &= equal_doublesfromstring(result[0], result_vec_string[0]);
	isCorrect &= (result_vec_string[1] == "inf");

	const double n1x = std::stod(result_vec_string[2]);
	const double n1y = std::stod(result_vec_string[3]);
	const double n1z = std::stod(result_vec_string[4]);
	const double n2x = std::stod(result_vec_string[5]);
	const double n2y = std::stod(result_vec_string[6]);
	const double n2z = std::stod(result_vec_string[7]);

	isCorrect &= (std::abs(n1x) < 1e-2);
	isCorrect &= (std::abs(n1y) < 1e-2);
	isCorrect &= (std::abs(n2x) < 1e-2);
	isCorrect &= (std::abs(n2y) < 1e-2);
	isCorrect &= equal_double(n1z, 5000.0, 5.0);
	isCorrect &= equal_double(n2z, 5000.0, 5.0);
	isCorrect &= (n1z > 0.0);
	isCorrect &= (n2z > 0.0);

	return isCorrect;
}

//////////////////////////////////////////////////////////////////////////////
// Add all the test cases
void AddTaskStaticSSSpectraTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("Task StaticSSSpectra test 1", test_task_staticssspectra_method_timeinf_cidsp_true));
	_cases.push_back(test_case("Task StaticSSSpectra test 2", test_task_staticssspectra_method_timeinf_cidsp_false));
	_cases.push_back(test_case("Task StaticSSSpectra test 3", test_task_staticssspectra_method_timeinf_cidsp_false_instant_pulse));
	_cases.push_back(test_case("Task StaticSSSpectra test 4", test_task_staticssspectra_method_timeinf_cidsp_false_longpulsestaticfield_pulse));
	_cases.push_back(test_case("Task StaticSSSpectra test 5", test_task_staticssspectra_method_timeinf_cidsp_false_longpulse_pulse));
}
//////////////////////////////////////////////////////////////////////////////
