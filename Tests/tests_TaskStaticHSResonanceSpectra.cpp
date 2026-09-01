//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Tests the StaticHSResonanceSpectra task with multiple Zeeman interactions.
//
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "RunSection.h"
#include "NuclearZeeman.h"
#include "TaskStaticHSDirectSpectra.h"
#include "TaskStaticHSResonanceSpectra.h"

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

	std::shared_ptr<SpinAPI::SpinSystem> BuildComplexTripletSystem(double eValue = 304.2)
	{
		auto spin = std::make_shared<SpinAPI::Spin>(
			"E", "type=electron;spin=1;tensor=isotropic(1.996);");
		std::ostringstream zfsProperties;
		zfsProperties << "type=zfs;group1=E;dvalue=-2340;evalue=" << eValue
					  << ";prefactor=0.00628318530718;ignoretensors=true;commonprefactor=false;";
		auto zfs = std::make_shared<SpinAPI::Interaction>("D", zfsProperties.str());
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"Z", "type=zeeman;spins=E;field=0 0 0.25;"
				 "ignoretensors=false;commonprefactor=true;prefactor=1.0;");

		auto tx = std::make_shared<SpinAPI::State>("Tx", "spin(E)=|-1>-|1>;");
		auto ty = std::make_shared<SpinAPI::State>("Ty", "spin(E)=i|-1>+i|1>;");
		auto tz = std::make_shared<SpinAPI::State>("Tz", "spin(E)=|0>;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("Triplet");
		spinsys->Add(spin);
		spinsys->Add(zfs);
		spinsys->Add(zeeman);
		spinsys->Add(tx);
		spinsys->Add(ty);
		spinsys->Add(tz);
		spinsys->ValidateInteractions();
		tx->ParseFromSystem(*spinsys);
		ty->ParseFromSystem(*spinsys);
		tz->ParseFromSystem(*spinsys);

		auto props = std::make_shared<MSDParser::ObjectParser>(
			"spinsyssettings", "initialstate=Tx,Ty,Tz;weights=0,0.47,0.53;frame=molecular;");
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
				char *end = nullptr;
				const double value = std::strtod(tok.c_str(), &end);
				if (end == tok.c_str() || end == nullptr || *end != '\0')
					return false;
				values.push_back(value);
			}

			if (values.size() <= idx)
				return false;
			out.push_back(values[idx]);
		}

		return !out.empty();
	}

	bool RunResonanceTaskWithProperties(const std::shared_ptr<SpinAPI::SpinSystem> &spinsys, bool enforceSync,
									const std::string &spectrumProperties, std::vector<double> &out)
	{
		RunSection::RunSection rs;
		rs.Add(spinsys);

		MSDParser::ObjectParser taskParser(
			"testtask",
			std::string("type=statichs-resonance-spectra;mwfrequency=95.0;lineshape=gaussian;") +
			spectrumProperties +
			"fieldinteraction=zeeman1;initialstate=Init;" +
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

	bool RunResonanceTask(const std::shared_ptr<SpinAPI::SpinSystem> &spinsys, bool enforceSync, std::vector<double> &out)
	{
		return RunResonanceTaskWithProperties(spinsys, enforceSync, "linewidth=0.1;detectspins=FE1,WE2;", out);
	}

	bool RunComplexTripletSweep(bool useCache, const std::string &cacheMode, bool useMzBlocks,
								std::vector<double> &out, double eValue = 304.2, int powderPoints = 32)
	{
		RunSection::RunSection rs;
		rs.Add(BuildComplexTripletSystem(eValue));

		MSDParser::ObjectParser settingsParser("general", "steps=181;");
		rs.Add(MSDParser::ObjectType::Settings, settingsParser);

		MSDParser::ObjectParser taskParser(
			"testtask",
			"type=statichs-resonance-spectra;mwfrequency=9.75;linewidth=2.0;lineshape=gaussian;"
			"detectspins=E;fieldinteraction=Z;hamiltonianh0list=D,Z;"
			"powdersamplingpoints=" + std::to_string(powderPoints) + ";powdergridtype=fibonacci;powdergammapoints=1;"
			"powderfullsphere=true;fulltensorrotation=true;sweepcache=" +
			std::string(useCache ? "true" : "false") + ";sweepcachemode=" + cacheMode +
			";mzblocks=" + std::string(useMzBlocks ? "true" : "false") + ";");
		rs.Add(MSDParser::ObjectType::Task, taskParser);
		auto task = rs.GetTask("testtask");
		if (task == nullptr)
			return false;

		MSDParser::ObjectParser actionParser(
			"field", "type=addvector;vector=Triplet.Z.field;direction=0 0 1;value=0.001;");
		if (!rs.Add(MSDParser::ObjectType::Action, actionParser))
			return false;

		std::ostringstream logstream;
		std::ostringstream datastream;
		task->SetLogStream(logstream);
		task->SetDataStream(datastream);

		for (unsigned int step = 1; step <= 181; ++step)
		{
			if (!rs.Run(step))
				return false;
			if (step < 181 && !rs.Step(step + 1))
				return false;
		}

		return ExtractColumn(datastream.str(), "Triplet.Total_perp", out);
	}

	bool RunAnisotropicMomentOwnershipSweep(
		bool useCache,
		std::vector<double> &out)
	{
		RunSection::RunSection rs;
		auto spinsys = BuildTwoZeemanSystem(3.370, 3.370);
		if (spinsys == nullptr)
			return false;

		auto init = spinsys->states_find("Init");
		if (init == nullptr || !init->ParseFromSystem(*spinsys))
			return false;

		rs.Add(spinsys);

		MSDParser::ObjectParser settingsParser(
			"general", "steps=41;");
		rs.Add(MSDParser::ObjectType::Settings, settingsParser);

		MSDParser::ObjectParser taskParser(
			"testtask",
			"type=statichs-resonance-spectra;"
			"mwfrequency=95.0;linewidth=0.5;lineshape=gaussian;"
			"detectspins=FE1,WE2;fieldinteraction=zeeman1;"
			"initialstate=Init;"
			"hamiltonianh0list=zeeman1,zeeman2;"
			"powdersamplingpoints=12;powdergridtype=fibonacci;"
			"powdergammapoints=3;powderfullsphere=true;"
			"fulltensorrotation=true;"
			"sweepcache=" + std::string(useCache ? "true" : "false") +
			";sweepcachemode=exact;mzblocks=true;"
			"enforce_zeeman_sync=true;");
		if (!rs.Add(MSDParser::ObjectType::Task, taskParser))
			return false;

		MSDParser::ObjectParser action1(
			"field1",
			"type=addvector;vector=System.zeeman1.field;"
			"direction=0 0 1;value=0.0005;");
		MSDParser::ObjectParser action2(
			"field2",
			"type=addvector;vector=System.zeeman2.field;"
			"direction=0 0 1;value=0.0005;");
		if (!rs.Add(MSDParser::ObjectType::Action, action1) ||
			!rs.Add(MSDParser::ObjectType::Action, action2))
			return false;

		auto task = rs.GetTask("testtask");
		if (task == nullptr)
			return false;

		std::ostringstream logstream;
		std::ostringstream datastream;
		task->SetLogStream(logstream);
		task->SetDataStream(datastream);

		for (unsigned int step = 1; step <= 41; ++step)
		{
			if (!rs.Run(step))
				return false;
			if (step < 41 && !rs.Step(step + 1))
				return false;
		}

		const std::vector<std::string> columns = {
			"System.Total_x",
			"System.Total_y",
			"System.Total_perp",
			"System.Cross_x",
			"System.Cross_y",
			"System.FE1_x",
			"System.FE1_y",
			"System.FE1_perp",
			"System.FE1_p",
			"System.FE1_m",
			"System.WE2_x",
			"System.WE2_y",
			"System.WE2_perp",
			"System.WE2_p",
			"System.WE2_m",
		};

		out.clear();
		for (const auto &column : columns)
		{
			std::vector<double> values;
			if (!ExtractColumn(datastream.str(), column, values))
				return false;
			out.insert(out.end(), values.begin(), values.end());
		}

		return !out.empty();
	}


	struct R2KBTaskResult
	{
		bool runOk = false;
		std::string log;
		std::string data;
	};

	std::shared_ptr<SpinAPI::SpinSystem> BuildR2KBOneVSystem(
		double fieldT, bool molecularFrame)
	{
		auto electron = std::make_shared<SpinAPI::Spin>(
			"E", "type=electron;spin=1/2;tensor=isotropic(2.0023);");
		auto nucleus = std::make_shared<SpinAPI::Spin>(
			"V", "type=nucleus;spin=7/2;isotope=51V;tensor=isotropic(1.0);");

		std::ostringstream bprops;
		bprops << std::setprecision(17)
			   << "type=zeeman;spins=E;field=0 0 " << fieldT << ";"
			   << "ignoretensors=false;commonprefactor=true;prefactor=1.0;";
		auto b0 = std::make_shared<SpinAPI::Interaction>("B0", bprops.str());

		auto hfc = std::make_shared<SpinAPI::Interaction>(
			"A",
			"type=hyperfine;group1=E;group2=V;"
			"tensor=isotropic(0.0047);commonprefactor=false;prefactor=1.0;");

		arma::vec field(3, arma::fill::zeros);
		field(2) = fieldT;
		SpinAPI::interaction_ptr nz;
		std::string error;
		if (!SpinAPI::NuclearZeeman::CreateInteraction(
				"NZ", nucleus, field, nz, error))
			return nullptr;

		auto up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");

		auto system = std::make_shared<SpinAPI::SpinSystem>("HybridSystem");
		system->Add(electron);
		system->Add(nucleus);
		system->Add(b0);
		system->Add(hfc);
		system->Add(nz);
		system->Add(up);
		if (!system->ValidateInteractions().empty() ||
			!up->ParseFromSystem(*system))
			return nullptr;

		auto props = std::make_shared<MSDParser::ObjectParser>(
			"spinsyssettings",
			std::string("initialstate=Up;frame=") +
			(molecularFrame ? "molecular;" : "fixed;"));
		if (!system->SetProperties(props))
			return nullptr;
		return system;
	}

	bool RunR2KBTask(
		const std::shared_ptr<SpinAPI::SpinSystem> &system,
		const std::string &extraProperties,
		R2KBTaskResult &result)
	{
		result = R2KBTaskResult{};
		if (system == nullptr)
			return false;

		RunSection::RunSection rs;
		rs.Add(system);

		MSDParser::ObjectParser taskParser(
			"testtask",
			"type=statichs-resonance-spectra;"
			"mwfrequency=9.5;linewidth=0.2;lineshape=gaussian;"
			"detectspins=E;fieldinteraction=B0;initialstate=Up;"
			"powdersamplingpoints=1;powdergridtype=fibonacci;"
			"powdergammapoints=1;powderfullsphere=true;"
			"fulltensorrotation=true;mzblocks=true;sweepcache=false;" +
			extraProperties);
		if (!rs.Add(MSDParser::ObjectType::Task, taskParser))
			return false;

		auto task = rs.GetTask("testtask");
		if (task == nullptr)
			return false;

		std::ostringstream logstream;
		std::ostringstream datastream;
		task->SetLogStream(logstream);
		task->SetDataStream(datastream);

		result.runOk = rs.Run(1);
		result.log = logstream.str();
		result.data = datastream.str();
		return true;
	}

	bool R2KBHasDataRow(const std::string &data)
	{
		std::istringstream stream(data);
		std::string line;
		int nonempty = 0;
		while (std::getline(stream, line))
		{
			if (!line.empty())
				++nonempty;
		}
		return nonempty >= 2;
	}

	double NormalizedRms(const std::vector<double> &lhs, const std::vector<double> &rhs)
	{
		if (lhs.size() != rhs.size() || lhs.empty())
			return std::numeric_limits<double>::infinity();

		double lhsScale = 0.0;
		double rhsScale = 0.0;
		for (size_t i = 0; i < lhs.size(); ++i)
		{
			lhsScale = std::max(lhsScale, std::abs(lhs[i]));
			rhsScale = std::max(rhsScale, std::abs(rhs[i]));
		}
		if (lhsScale <= 0.0 || rhsScale <= 0.0)
			return std::numeric_limits<double>::infinity();

		double squaredError = 0.0;
		for (size_t i = 0; i < lhs.size(); ++i)
		{
			const double delta = lhs[i] / lhsScale - rhs[i] / rhsScale;
			squaredError += delta * delta;
		}
		return std::sqrt(squaredError / static_cast<double>(lhs.size()));
	}
}

void AddTaskStaticHSResonanceSpectraTests(std::vector<test_case> &cases)
{
	cases.push_back(test_case("Resonance spectra R2K-B explicit hybrid task route", []() {
		const double fieldT = 0.3389868917139098;
		R2KBTaskResult result;
		if (!RunR2KBTask(
				BuildR2KBOneVSystem(fieldT, false),
				"solver=hybrid;perturbativenuclei=V;"
				"hamiltonianh0list=B0,A,NZ;hybridfieldstep=0.0001;",
				result) || !result.runOk)
			return false;

		std::vector<double> signal;
		if (!ExtractColumn(result.data, "HybridSystem.Total_perp", signal) ||
			signal.size() != 1 || !std::isfinite(signal.front()) ||
			std::abs(signal.front()) <= 1.0e-16)
			return false;

		return result.log.find(
			"Hybrid resonance explicit partition: 1 perturbative nuclei; product nuclear dimension = 8; largest nuclear diagonalization = 8") !=
			std::string::npos;
	}));

	cases.push_back(test_case("Resonance spectra R2K-B incomplete hybrid H0 fails closed", []() {
		const double fieldT = 0.3389868917139098;
		R2KBTaskResult result;
		if (!RunR2KBTask(
				BuildR2KBOneVSystem(fieldT, false),
				"solver=hybrid;perturbativenuclei=V;"
				"hamiltonianh0list=B0,A;hybridfieldstep=0.0001;",
				result) || !result.runOk)
			return false;

		return !R2KBHasDataRow(result.data) &&
			result.log.find(
				"Hybrid resonance requires HamiltonianH0list to cover the complete static SpinSystem interaction set exactly once.") !=
			std::string::npos;
	}));

	cases.push_back(test_case("Resonance spectra R2K-B molecular hybrid state fails closed", []() {
		const double fieldT = 0.3389868917139098;
		R2KBTaskResult result;
		if (!RunR2KBTask(
				BuildR2KBOneVSystem(fieldT, true),
				"solver=hybrid;perturbativenuclei=V;"
				"hamiltonianh0list=B0,A,NZ;hybridfieldstep=0.0001;",
				result) || !result.runOk)
			return false;

		return !R2KBHasDataRow(result.data) &&
			result.log.find(
				"Hybrid resonance requires initialstateframe=fixed in R2K-B") !=
			std::string::npos;
	}));

	cases.push_back(test_case("Resonance spectra R2K-B exact default solver parity", []() {
		const double fieldT = 0.3389868917139098;
		R2KBTaskResult defaultResult, explicitResult;
		if (!RunR2KBTask(
				BuildR2KBOneVSystem(fieldT, false),
				"hamiltonianh0list=B0,A,NZ;",
				defaultResult) ||
			!RunR2KBTask(
				BuildR2KBOneVSystem(fieldT, false),
				"solver=exact;hamiltonianh0list=B0,A,NZ;",
				explicitResult) ||
			!defaultResult.runOk || !explicitResult.runOk)
			return false;

		return R2KBHasDataRow(defaultResult.data) &&
			defaultResult.data == explicitResult.data;
	}));

	cases.push_back(test_case("Spectroscopy task registry aliases", []() {
		const std::vector<std::string> resonanceNames = {
			"statichs-resonance-spectra",
			"StaticHS-Resonance-Spectra",
			"statichs-trepr-spectra",
			"StaticHS-TrEPR-Spectra",
		};
		for (const auto &taskType : resonanceNames)
		{
			RunSection::RunSection rs;
			MSDParser::ObjectParser parser("resonance", "type=" + taskType + ";");
			if (!rs.Add(MSDParser::ObjectType::Task, parser))
				return false;
			if (std::dynamic_pointer_cast<RunSection::TaskStaticHSResonanceSpectra>(rs.GetTask("resonance")) == nullptr)
				return false;
		}

		RunSection::RunSection directSection;
		MSDParser::ObjectParser directParser("direct", "type=StaticHS-Direct-Spectra;");
		if (!directSection.Add(MSDParser::ObjectType::Task, directParser))
			return false;
		return std::dynamic_pointer_cast<RunSection::TaskStaticHSDirectSpectra>(directSection.GetTask("direct")) != nullptr;
	}));

	cases.push_back(test_case("Resonance spectra multi-zeeman sync", []() {
		std::vector<double> synced;
		std::vector<double> equal;

		auto sys_mismatch = BuildTwoZeemanSystem(3.380, 3.381);
		if (!RunResonanceTask(sys_mismatch, true, synced))
			return false;

		auto sys_equal = BuildTwoZeemanSystem(3.380, 3.380);
		if (!RunResonanceTask(sys_equal, false, equal))
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

	cases.push_back(test_case("Resonance spectra detectspins list trimming", []() {
		std::vector<double> compactInput;
		std::vector<double> spacedInput;

		auto sys_compact = BuildTwoZeemanSystem(3.380, 3.380);
		if (!RunResonanceTaskWithProperties(sys_compact, false, "linewidth=0.2;detectspins=FE1,WE2;", compactInput))
			return false;

		auto sys_spaced = BuildTwoZeemanSystem(3.380, 3.380);
		if (!RunResonanceTaskWithProperties(sys_spaced, false, "linewidth=0.2;detectspins=FE1, WE2;", spacedInput))
			return false;

		if (compactInput.size() != spacedInput.size() || compactInput.empty())
			return false;

		const double tol = 1e-6;
		for (size_t i = 0; i < compactInput.size(); ++i)
		{
			if (std::abs(compactInput[i] - spacedInput[i]) > tol * std::max(1.0, std::abs(compactInput[i])))
				return false;
		}

		return true;
	}));

	cases.push_back(test_case("Resonance spectra complex eigenbasis transform", []() {
		std::vector<double> exact;
		std::vector<double> crossing;
		const bool exactOk = RunComplexTripletSweep(true, "exact", true, exact);
		const bool crossingOk = RunComplexTripletSweep(true, "approx", true, crossing);
		if (!exactOk || !crossingOk || exact.size() != crossing.size() || exact.empty())
		{
			std::cerr << "Failed to construct comparable Resonance spectra sweeps: exact=" << exactOk
					  << ", crossing=" << crossingOk << ", sizes=" << exact.size()
					  << "/" << crossing.size() << std::endl;
			return false;
		}

		const double normalizedRms = NormalizedRms(exact, crossing);
		if (normalizedRms >= 0.02)
			std::cerr << "Exact/crossing normalized RMS mismatch: " << normalizedRms << std::endl;
		return normalizedRms < 0.02;
	}));

	cases.push_back(test_case("Resonance spectra canonical magnetic-moment old-vs-new exact parity", []() {
		std::vector<double> cachedManual;
		std::vector<double> uncachedCanonical;
		if (!RunAnisotropicMomentOwnershipSweep(true, cachedManual) ||
			!RunAnisotropicMomentOwnershipSweep(false, uncachedCanonical))
			return false;

		if (cachedManual.empty() ||
			cachedManual.size() != uncachedCanonical.size())
			return false;

		double cachedMaxAbs = 0.0;
		double uncachedMaxAbs = 0.0;
		for (size_t i = 0; i < cachedManual.size(); ++i)
		{
			if (!std::isfinite(cachedManual[i]) ||
				!std::isfinite(uncachedCanonical[i]))
				return false;
			cachedMaxAbs =
				std::max(cachedMaxAbs, std::abs(cachedManual[i]));
			uncachedMaxAbs =
				std::max(uncachedMaxAbs, std::abs(uncachedCanonical[i]));
		}

		const double commonScale =
			std::max(cachedMaxAbs, uncachedMaxAbs);
		if (!(commonScale > 0.0) || !std::isfinite(commonScale))
			return false;

		long double diff2 = 0.0L;
		long double cached2 = 0.0L;
		long double uncached2 = 0.0L;
		for (size_t i = 0; i < cachedManual.size(); ++i)
		{
			const long double a =
				static_cast<long double>(cachedManual[i] / commonScale);
			const long double b =
				static_cast<long double>(uncachedCanonical[i] / commonScale);
			const long double d = a - b;
			diff2 += d * d;
			cached2 += a * a;
			uncached2 += b * b;
		}

		const long double reference2 =
			std::max(cached2, uncached2);
		if (!(reference2 > 0.0L))
			return false;

		const double relativeL2 =
			static_cast<double>(std::sqrt(diff2 / reference2));
		const double peakRelative =
			std::abs(cachedMaxAbs - uncachedMaxAbs) / commonScale;
		const double shapeNormalizedRms =
			NormalizedRms(cachedManual, uncachedCanonical);

		if (relativeL2 >= 1e-10 ||
			peakRelative >= 1e-10 ||
			shapeNormalizedRms >= 1e-10)
		{
			std::cerr
				<< std::setprecision(17)
				<< "Magnetic-moment parity mismatch:"
				<< " relative_l2=" << relativeL2
				<< " peak_relative=" << peakRelative
				<< " shape_normalized_rms=" << shapeNormalizedRms
				<< " cached_maxabs=" << cachedMaxAbs
				<< " uncached_maxabs=" << uncachedMaxAbs
				<< std::endl;
			return false;
		}

		return true;
	}));

	cases.push_back(test_case("Resonance spectra exact cache equivalence", []() {
		std::vector<double> cached;
		std::vector<double> uncached;
		if (!RunComplexTripletSweep(true, "exact", true, cached) ||
			!RunComplexTripletSweep(false, "exact", true, uncached))
			return false;
		const double normalizedRms = NormalizedRms(cached, uncached);
		if (normalizedRms >= 1e-10)
			std::cerr << "Exact cached/uncached normalized RMS mismatch: " << normalizedRms << std::endl;
		return normalizedRms < 1e-10;
	}));

	cases.push_back(test_case("Resonance spectra canonical uncached Mz block equivalence", []() {
		std::vector<double> blocked;
		std::vector<double> full;
		if (!RunComplexTripletSweep(false, "exact", true, blocked, 0.0, 1) ||
			!RunComplexTripletSweep(false, "exact", false, full, 0.0, 1))
			return false;
		const double normalizedRms = NormalizedRms(blocked, full);
		if (normalizedRms >= 1e-12)
			std::cerr << "Canonical uncached Mz-block/full normalized RMS mismatch: "
					  << normalizedRms << std::endl;
		return normalizedRms < 1e-12;
	}));

	cases.push_back(test_case("Resonance spectra Mz block equivalence", []() {
		std::vector<double> blocked;
		std::vector<double> full;
		// An axial ZFS tensor at the identity orientation conserves total Mz,
		// ensuring the first run exercises block diagonalization rather than
		// merely passing through its validity guard.
		if (!RunComplexTripletSweep(true, "exact", true, blocked, 0.0, 1) ||
			!RunComplexTripletSweep(true, "exact", false, full, 0.0, 1))
			return false;
		const double normalizedRms = NormalizedRms(blocked, full);
		if (normalizedRms >= 1e-12)
			std::cerr << "Mz-block/full normalized RMS mismatch: " << normalizedRms << std::endl;
		return normalizedRms < 1e-12;
	}));
}
