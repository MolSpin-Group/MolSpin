//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Physics and numerical regression tests for TaskMultiStaticSS.
//////////////////////////////////////////////////////////////////////////////
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "RunSection.h"
#include "TaskMultiStaticSS.h"

namespace
{
	bool ParseMultiStaticData(const std::string &_data, std::vector<double> &_row)
	{
		std::istringstream stream(_data);
		std::string line;
		if (!std::getline(stream, line) || !std::getline(stream, line))
			return false;

		std::istringstream values(line);
		double value = 0.0;
		while (values >> value)
			_row.push_back(value);
		return !_row.empty();
	}

	bool RunMultiStaticTask(const std::vector<SpinAPI::system_ptr> &_systems,
							const std::string &_properties,
							std::vector<double> &_row,
							std::string *_log = nullptr)
	{
		RunSection::RunSection runSection;
		for (const auto &system : _systems)
			runSection.Add(system);

		MSDParser::ObjectParser parser(
			"testtask", "type=multistaticss;diagnostics=true;" + _properties);
		if (!runSection.Add(MSDParser::ObjectType::Task, parser))
		{
			std::cout << "Failed to add MultiStaticSS task." << std::endl;
			return false;
		}

		auto task = runSection.GetTask("testtask");
		if (task == nullptr)
		{
			std::cout << "Failed to retrieve MultiStaticSS task." << std::endl;
			return false;
		}

		std::ostringstream logstream;
		std::ostringstream datastream;
		task->SetLogStream(logstream);
		task->SetDataStream(datastream);
		if (!runSection.Run(1))
		{
			std::cout << logstream.str() << datastream.str();
			return false;
		}

		if (_log != nullptr)
			*_log = logstream.str();
		const bool parsed = ParseMultiStaticData(datastream.str(), _row);
		if (!parsed)
			std::cout << logstream.str() << datastream.str();
		return parsed;
	}

	struct TransferNetwork
	{
		std::vector<SpinAPI::system_ptr> systems;
		std::vector<SpinAPI::state_ptr> states;
	};

	TransferNetwork BuildTransferNetwork()
	{
		TransferNetwork network;
		auto source = std::make_shared<SpinAPI::SpinSystem>("Source");
		auto target = std::make_shared<SpinAPI::SpinSystem>("Target");
		auto sourceSpin = std::make_shared<SpinAPI::Spin>("E1", "type=electron;spin=1/2;tensor=isotropic(2);");
		// A different target dimension verifies the rectangular creation map
		// from a 2x2 source density to a 3x3 target density.
		auto targetSpin = std::make_shared<SpinAPI::Spin>("E2", "type=electron;spin=1;tensor=isotropic(2);");
		auto sourceUp = std::make_shared<SpinAPI::State>("Up", "spin(E1)=|1/2>;");
		auto sourceIdentity = std::make_shared<SpinAPI::State>("Identity", "");
		auto targetDown = std::make_shared<SpinAPI::State>("Down", "spin(E2)=|0>;");
		auto targetIdentity = std::make_shared<SpinAPI::State>("Identity", "");

		source->Add(sourceSpin);
		source->Add(sourceUp);
		source->Add(sourceIdentity);
		target->Add(targetSpin);
		target->Add(targetDown);
		target->Add(targetIdentity);

		auto transfer = std::make_shared<SpinAPI::Transition>(
			"transfer", "type=sink;sourcestate=Up;target=Target;targetstate=Down;rate=2;", source);
		auto sourceLoss = std::make_shared<SpinAPI::Transition>(
			"source_loss", "type=sink;sourcestate=Identity;rate=1;", source);
		auto targetLoss = std::make_shared<SpinAPI::Transition>(
			"target_loss", "type=sink;sourcestate=Identity;rate=1;", target);
		source->Add(transfer);
		source->Add(sourceLoss);
		target->Add(targetLoss);

		source->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"source_settings", "initialstate=Up;"));
		target->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"target_settings", ""));

		network.systems = {source, target};
		network.states = {sourceUp, sourceIdentity, targetDown, targetIdentity};
		return network;
	}

	struct PowderSystem
	{
		SpinAPI::system_ptr system;
		std::vector<SpinAPI::state_ptr> states;
	};

	PowderSystem BuildAnalyticPowderSystem()
	{
		PowderSystem result;
		result.system = std::make_shared<SpinAPI::SpinSystem>("System");
		auto spin = std::make_shared<SpinAPI::Spin>(
			"E", "type=electron;spin=1/2;tensor=isotropic(2);");
		auto field = std::make_shared<SpinAPI::Interaction>(
			"B", "type=zeeman;spins=E;field=0 0 0.01;ignoretensors=true;"
				 "commonprefactor=false;prefactor=100;");
		auto up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto down = std::make_shared<SpinAPI::State>("Down", "spin(E)=|-1/2>;");
		result.system->Add(spin);
		result.system->Add(field);
		result.system->Add(up);
		result.system->Add(down);
		result.system->Add(std::make_shared<SpinAPI::Transition>(
			"up_loss", "type=sink;sourcestate=Up;rate=1;", result.system));
		result.system->Add(std::make_shared<SpinAPI::Transition>(
			"down_loss", "type=sink;sourcestate=Down;rate=1;", result.system));
		result.system->SetProperties(std::make_shared<MSDParser::ObjectParser>(
			"settings",
			"initialstate=Up;initialstateframe=molecular;"
			"transitionstateframe=molecular;observablestateframe=molecular;"));
		result.system->ValidateInteractions();
		result.states = {up, down};
		return result;
	}
}

// The represented transfer and external-loss yields are analytic. This test
// catches the historical missing minus sign, -i(H+K) block error, overwritten
// transfer blocks, and subsystem slicing overrun.
bool test_task_multistaticss_trace_consistent_transfer_network()
{
	TransferNetwork network = BuildTransferNetwork();
	bool ok = true;
	for (const auto &state : network.states)
	{
		auto owner = state->Name() == "Down" ? network.systems[1] :
					 (state == network.states[3] ? network.systems[1] : network.systems[0]);
		ok &= state->ParseFromSystem(*owner);
	}
	for (const auto &system : network.systems)
		ok &= system->ValidateTransitions(network.systems).empty();

	std::vector<double> row;
	std::vector<double> legacyDefault;
	std::string log;
	ok &= RunMultiStaticTask(
		network.systems,
		"transitionyields=true;hamiltonianmode=full;linearsolver=dense;",
		row,
		&log);
	ok &= RunMultiStaticTask(
		network.systems,
		"transitionyields=true;linearsolver=dense;",
		legacyDefault);
	ok &= row.size() == legacyDefault.size();
	if (row.size() == legacyDefault.size())
		for (size_t index = 0; index < row.size(); ++index)
			ok &= std::abs(row[index] - legacyDefault[index]) < 1.0e-12;

	// Columns: step, transfer, source loss, target loss, sum of all channels.
	ok &= row.size() == 5;
	if (row.size() == 5)
	{
		// The default task data precision is six significant digits.
		ok &= std::abs(row[1] - 2.0 / 3.0) < 1.0e-5;
		ok &= std::abs(row[2] - 1.0 / 3.0) < 1.0e-5;
		ok &= std::abs(row[3] - 2.0 / 3.0) < 1.0e-5;
		ok &= std::abs(row[4] - 5.0 / 3.0) < 1.0e-5;
	}
	ok &= log.find("relative residual=") != std::string::npos;
	if (!ok)
	{
		std::cout << log;
		for (double value : row)
			std::cout << value << " ";
		std::cout << std::endl;
	}
	return ok;
}

// H=Sz gives a unit precession frequency, while the two projective losses add
// to unit total decay. For a molecular z state at cos(theta)=+/-1/2,
// Phi_up = 3/4 + cos(theta)^2/4 = 0.8125 and Phi_down = 0.1875.
bool test_task_multistaticss_powder_rotates_initial_and_reaction_states()
{
	PowderSystem powder = BuildAnalyticPowderSystem();
	bool ok = true;
	for (const auto &state : powder.states)
		ok &= state->ParseFromSystem(*powder.system);
	std::vector<SpinAPI::system_ptr> systems = {powder.system};
	ok &= powder.system->ValidateTransitions(systems).empty();

	std::vector<double> averaged;
	ok &= RunMultiStaticTask(
		systems,
		"transitionyields=true;hamiltonianmode=rotated_zyz;"
		"powdersamplingpoints=2;powderdomain=full;linearsolver=dense;",
		averaged);
	ok &= averaged.size() == 4;
	if (averaged.size() == 4)
	{
		ok &= std::abs(averaged[1] - 0.8125) < 1.0e-6;
		ok &= std::abs(averaged[2] - 0.1875) < 1.0e-6;
		ok &= std::abs(averaged[3] - 1.0) < 1.0e-6;
	}

	SpinAPI::PowderGrid grid;
	ok &= SpinAPI::CreateUniformPowderGrid(
		2, SpinAPI::PowderGridDomain::FullSphere, grid);
	std::vector<double> explicitAverage(3, 0.0);
	for (const auto &orientation : grid)
	{
		std::ostringstream properties;
		properties << std::setprecision(17)
				   << "transitionyields=true;hamiltonianmode=rotated_zyz;"
				   << "powderorientation=" << orientation.theta << " "
				   << orientation.phi << " 1;linearsolver=dense;";
		std::vector<double> row;
		ok &= RunMultiStaticTask(systems, properties.str(), row);
		ok &= row.size() == 4;
		if (row.size() == 4)
			for (size_t column = 0; column < explicitAverage.size(); ++column)
				explicitAverage[column] += 0.5 * row[column + 1];
	}
	if (averaged.size() == 4)
		for (size_t column = 0; column < explicitAverage.size(); ++column)
			ok &= std::abs(explicitAverage[column] - averaged[column + 1]) < 1.0e-8;

	// At an aligned orientation this model contains only an Sz Zeeman term,
	// so exact and high-field secular builders must be identical. This checks
	// that rotated_sa is wired through the task without making it the default.
	std::vector<double> alignedExact;
	std::vector<double> alignedSecular;
	ok &= RunMultiStaticTask(
		systems,
		"transitionyields=true;hamiltonianmode=rotated_zyz;"
		"powderorientation=0 0 1;linearsolver=dense;",
		alignedExact);
	ok &= RunMultiStaticTask(
		systems,
		"transitionyields=true;hamiltonianmode=rotated_sa;"
		"powderorientation=0 0 1;linearsolver=dense;",
		alignedSecular);
	ok &= alignedExact.size() == alignedSecular.size();
	if (alignedExact.size() == alignedSecular.size())
		for (size_t column = 0; column < alignedExact.size(); ++column)
			ok &= std::abs(alignedExact[column] - alignedSecular[column]) < 1.0e-10;

	if (!ok)
	{
		std::cout << "Averaged:";
		for (double value : averaged)
			std::cout << " " << value;
		std::cout << "\nExplicit:";
		for (double value : explicitAverage)
			std::cout << " " << value;
		std::cout << std::endl;
	}
	return ok;
}

void AddTaskMultiStaticSSTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case(
		"Task MultiStaticSS trace-consistent transfer network",
		test_task_multistaticss_trace_consistent_transfer_network));
	_cases.push_back(test_case(
		"Task MultiStaticSS rotates Hamiltonian, initial state, and reactions",
		test_task_multistaticss_powder_rotates_initial_and_reaction_states));
}
//////////////////////////////////////////////////////////////////////////////
