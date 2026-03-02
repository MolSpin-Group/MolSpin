//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Tests the StaticHSTrEPRSpectra task with multiple Zeeman interactions.
//
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "RunSection.h"
#include "TaskStaticHSTrEPRSpectra.h"

namespace
{
	std::shared_ptr<SpinAPI::SpinSystem> BuildTwoZeemanSystem(double B1, double B2)
	{
		auto spin1 = std::make_shared<SpinAPI::Spin>(
			"FE1",
			"type=electron;spin=1/2;tensor=matrix(\"2.0033 0 0; 0 2.0025 0; 0 0 2.0021\");");
		auto spin2 = std::make_shared<SpinAPI::Spin>(
			"WE2",
			"type=electron;spin=1/2;tensor=matrix(\"2.0066 0 0; 0 2.0054 0; 0 0 2.0022\");");

		std::ostringstream zeeman1_props;
		zeeman1_props << "type=zeeman;spins=FE1;field=0 0 " << B1
						  << ";orientation=-0.3194,-2.0822,-0.4014;ignoretensors=false;commonprefactor=true;prefactor=1.0;";
		auto zeeman1 = std::make_shared<SpinAPI::Interaction>("zeeman1", zeeman1_props.str());

		std::ostringstream zeeman2_props;
		zeeman2_props << "type=zeeman;spins=WE2;field=0 0 " << B2
						  << ";ignoretensors=false;commonprefactor=true;prefactor=1.0;";
		auto zeeman2 = std::make_shared<SpinAPI::Interaction>("zeeman2", zeeman2_props.str());

		auto state_init = std::make_shared<SpinAPI::State>("Init", "spin(FE1)=|1/2>;spin(WE2)=|1/2>;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin1);
		spinsys->Add(spin2);
		spinsys->Add(zeeman1);
		spinsys->Add(zeeman2);
		spinsys->Add(state_init);
		spinsys->ValidateInteractions();

		auto props = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Init;");
		spinsys->SetProperties(props);
		return spinsys;
	}

	bool ExtractColumn(const std::string &data, const std::string &colName, std::vector<double> &out)
	{
		std::istringstream stream(data);
		std::string line;
		if (!std::getline(stream, line))
			return false;

		std::vector<std::string> headers;
		{
			std::istringstream h(line);
			for (std::string tok; h >> tok;)
				headers.push_back(tok);
		}
		auto it = std::find(headers.begin(), headers.end(), colName);
		if (it == headers.end())
			return false;
		const size_t idx = static_cast<size_t>(std::distance(headers.begin(), it));

		out.clear();
		while (std::getline(stream, line))
		{
			if (line.empty())
				continue;
			std::istringstream ls(line);
			std::vector<double> values;
			for (std::string tok; ls >> tok;)
			{
				try
				{
					values.push_back(std::stod(tok));
				}
				catch (const std::exception &)
				{
					return false;
				}
			}
			if (values.size() <= idx)
				return false;
			out.push_back(values[idx]);
		}

		return !out.empty();
	}

	bool RunTrEPRTask(const std::shared_ptr<SpinAPI::SpinSystem> &spinsys, bool enforceSync, std::vector<double> &out)
	{
		RunSection::RunSection rs;
		rs.Add(spinsys);

		MSDParser::ObjectParser taskParser(
			"testtask",
			std::string("type=statichs-trepr-spectra;mwfrequency=95.0;linewidth=0.1;lineshape=gaussian;") +
			"electron1=FE1;electron2=WE2;fieldinteraction=zeeman1;initialstate=Init;" +
			"HamiltonianH0list=zeeman1,zeeman2;powdersamplingpoints=1;powdergridtype=fibonacci;" +
			"powdergammapoints=1;powderfullsphere=true;fulltensorrotation=true;" +
			"sweepcache=false;" +
			(std::string("enforce_zeeman_sync=") + (enforceSync ? "true" : "false") + ";"));

		rs.Add(MSDParser::ObjectType::Task, taskParser);
		auto task = rs.GetTask("testtask");
		if (task == nullptr)
			return false;

		std::ostringstream logstream;
		std::ostringstream datastream;
		task->SetLogStream(logstream);
		task->SetDataStream(datastream);

		if (!rs.Run(1))
			return false;

		return ExtractColumn(datastream.str(), "System.Total_perp", out);
	}
}

void AddTaskStaticHSTrEPRSpectraTests(std::vector<test_case> &cases)
{
	cases.push_back(test_case("TrEPR multi-zeeman sync", []() {
		std::vector<double> synced;
		std::vector<double> equal;

		auto sys_mismatch = BuildTwoZeemanSystem(3.380, 3.381);
		if (!RunTrEPRTask(sys_mismatch, true, synced))
			return false;

		auto sys_equal = BuildTwoZeemanSystem(3.380, 3.380);
		if (!RunTrEPRTask(sys_equal, false, equal))
			return false;

		if (synced.size() != equal.size())
			return false;
		if (synced.empty())
			return false;

		const double tol = 1e-6;
		for (size_t i = 0; i < synced.size(); ++i)
		{
			if (std::abs(synced[i] - equal[i]) > tol * std::max(1.0, std::abs(equal[i])))
				return false;
		}

		return true;
	}));
}
