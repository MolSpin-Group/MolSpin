//////////////////////////////////////////////////////////////////////////////
// MolSpin Unit Testing Module
//
// Tests the StaticSSPowderSpectra and StaticHSDirectSpectra tasks.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "Operator.h"
#include "RunSection.h"
#include "SpinSpace.h"
#include "TaskStaticHSDirectSpectra.h"
#include "TaskStaticSSPowderSpectra.h"

namespace
{
	struct TwoElectronSystem
	{
		std::shared_ptr<SpinAPI::SpinSystem> spinsys;
		std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
		std::shared_ptr<SpinAPI::State> state_tplus;
		std::shared_ptr<SpinAPI::State> state_identity;
	};

	struct OneElectronSystem
	{
		std::shared_ptr<SpinAPI::SpinSystem> spinsys;
		std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems;
		std::shared_ptr<SpinAPI::State> state_up;
	};

	TwoElectronSystem BuildTwoElectronSystem(double _bx, double _by, double _bz, bool _add_transition, double _rate)
	{
		TwoElectronSystem system;

		auto spin1 = std::make_shared<SpinAPI::Spin>("electron1", "spin=1/2;tensor=isotropic(2);");
		auto spin2 = std::make_shared<SpinAPI::Spin>("electron2", "spin=1/2;tensor=isotropic(2);");

		std::ostringstream zeeman_props;
		zeeman_props << "type=zeeman;spins=electron1,electron2;field=" << _bx << " " << _by << " " << _bz
					 << ";ignoretensors=true;commonprefactor=true;prefactor=1.0;";
		auto zeeman = std::make_shared<SpinAPI::Interaction>("zeeman", zeeman_props.str());

		auto state_tplus = std::make_shared<SpinAPI::State>("Tplus", "spin(electron1)=|1/2>;spin(electron2)=|1/2>;");
		auto state_identity = std::make_shared<SpinAPI::State>("Identity", "");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin1);
		spinsys->Add(spin2);
		spinsys->Add(zeeman);
		spinsys->Add(state_tplus);
		spinsys->Add(state_identity);
		spinsys->ValidateInteractions();

		if (_add_transition)
		{
			auto transition = std::make_shared<SpinAPI::Transition>("sink", "type=sink;sourcestate=Identity;rate=" + std::to_string(_rate) + ";", spinsys);
			spinsys->Add(transition);
		}

		auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Tplus;");
		spinsys->SetProperties(spinsysParser);

		system.spinsys = spinsys;
		system.spinsystems.push_back(spinsys);
		system.state_tplus = state_tplus;
		system.state_identity = state_identity;

		return system;
	}

	bool PrepareSystem(const TwoElectronSystem &_system)
	{
		bool ok = true;
		ok &= _system.state_tplus->ParseFromSystem(*_system.spinsys);
		ok &= _system.state_identity->ParseFromSystem(*_system.spinsys);
		ok &= (_system.spinsys->ValidateTransitions(_system.spinsystems).size() == 0);
		return ok;
	}

	OneElectronSystem BuildOneElectronSystem(
		const std::string &_spin_system_properties = "initialstate=Up;",
		const std::string &_pulse_name = "cw",
		const std::string &_pulse_properties =
			"type=LongPulseStaticField;field=0.0002 0 0;pulsetime=2.0;timestep=0.1;group=E;"
			"prefactorlist=1,1,1;commonprefactorlist=true;ignoretensorslist=true;")
	{
		OneElectronSystem system;

		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=matrix(\"2.0 0 0; 0 2.0 0; 0 0 2.4\");");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"zeeman",
			"type=zeeman;spins=E;field=0 0 -0.004;ignoretensors=false;commonprefactor=true;prefactor=1.0;");
		auto state_up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto pulse = std::make_shared<SpinAPI::Pulse>(_pulse_name, _pulse_properties);

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);
		spinsys->Add(zeeman);
		spinsys->Add(state_up);
		spinsys->Add(pulse);
		spinsys->ValidateInteractions();
		spinsys->ValidatePulses();

		auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", _spin_system_properties);
		spinsys->SetProperties(spinsysParser);

		system.spinsys = spinsys;
		system.spinsystems.push_back(spinsys);
		system.state_up = state_up;
		return system;
	}

	OneElectronSystem BuildOneElectronThermalSystem()
	{
		return BuildOneElectronSystem("initialstate=thermal;temperature=280.0;thermalhamiltonian=zeeman;");
	}

	OneElectronSystem BuildOneElectronThermalSystemWithPulse(const std::string &_pulse_name, const std::string &_pulse_properties)
	{
		return BuildOneElectronSystem("initialstate=thermal;temperature=280.0;thermalhamiltonian=zeeman;", _pulse_name, _pulse_properties);
	}

	bool PrepareSystem(const OneElectronSystem &_system)
	{
		return _system.state_up->ParseFromSystem(*_system.spinsys);
	}

	bool RunPowderTask(const std::shared_ptr<SpinAPI::SpinSystem> &_spinsys, const std::string &_task_type, const std::string &_props, std::string &_data)
	{
		RunSection::RunSection rs;
		rs.Add(_spinsys);

		std::string taskname = "testtask";
		MSDParser::ObjectParser taskParser(taskname, "type=" + _task_type + ";" + _props);
		rs.Add(MSDParser::ObjectType::Task, taskParser);
		auto task = rs.GetTask(taskname);

		std::ostringstream logstream;
		std::ostringstream datastream;
		task->SetLogStream(logstream);
		task->SetDataStream(datastream);

		if (!rs.Run(1))
			return false;

		_data = datastream.str();
		std::string _log = logstream.str();
		std::cout << _data << "\n" << _log << std::endl; 
		return true;
	}

	bool ParseDataRows(const std::string &_data, std::vector<std::vector<double>> &_rows, std::vector<double> *_times = nullptr)
	{
		std::istringstream stream(_data);
		std::string line;
		bool header_skipped = false;
		while (std::getline(stream, line))
		{
			if (line.empty())
				continue;
			if (!header_skipped)
			{
				header_skipped = true;
				continue;
			}

			std::istringstream line_stream(line);
			std::string token;
			int token_index = 0;
			double row_time = 0.0;
			std::vector<double> row;
			while (line_stream >> token)
			{
				if (token_index < 2)
				{
					if (token_index == 1 && _times != nullptr)
					{
						try
						{
							row_time = std::stod(token);
						}
						catch (const std::exception &)
						{
							return false;
						}
					}
					token_index++;
					continue;
				}

				try
				{
					row.push_back(std::stod(token));
				}
				catch (const std::exception &)
				{
					return false;
				}

				token_index++;
			}

			if (!row.empty())
			{
				if (row.size() % 5 == 0)
				{
					std::vector<double> filtered;
					filtered.reserve((row.size() / 5) * 3);
					for (size_t idx = 0; idx < row.size(); idx += 5)
					{
						filtered.push_back(row[idx]);
						filtered.push_back(row[idx + 1]);
						filtered.push_back(row[idx + 2]);
					}
					_rows.push_back(filtered);
				}
				else
				{
					_rows.push_back(row);
				}
				if (_times != nullptr)
					_times->push_back(row_time);
			}
		}

		return !_rows.empty();
	}

	size_t CountDataRows(const std::string &_data)
	{
		std::istringstream stream(_data);
		std::string line;
		bool header_skipped = false;
		size_t rows = 0;
		while (std::getline(stream, line))
		{
			if (line.empty())
				continue;
			if (!header_skipped)
			{
				header_skipped = true;
				continue;
			}
			++rows;
		}
		return rows;
	}

	bool CheckTripletStructure(const std::vector<double> &_row, double _tol_zero, double _tol_equal)
	{
		if (_row.size() < 6)
			return false;

		bool ok = true;
		ok &= std::abs(_row[0]) < _tol_zero;
		ok &= std::abs(_row[1]) < _tol_zero;
		ok &= std::abs(_row[3]) < _tol_zero;
		ok &= std::abs(_row[4]) < _tol_zero;

		ok &= (_row[2] > 0.0);
		ok &= (_row[5] > 0.0);
		ok &= equal_double(_row[2], _row[5], _tol_equal);
		ok &= (std::abs(_row[2]) > 1e-6);

		return ok;
	}

	bool RowsConstant(const std::vector<std::vector<double>> &_rows, double _tol)
	{
		if (_rows.size() < 2)
			return false;

		for (size_t r = 1; r < _rows.size(); ++r)
		{
			if (_rows[r].size() != _rows[0].size())
				return false;
			for (size_t i = 0; i < _rows[r].size(); ++i)
			{
				if (!equal_double(_rows[r][i], _rows[0][i], _tol))
					return false;
			}
		}

		return true;
	}

	bool RowsLinearIncrease(const std::vector<std::vector<double>> &_rows, size_t _col, double _tol)
	{
		if (_rows.size() < 4 || _rows[0].size() <= _col)
			return false;

		double delta_ref = _rows[2][_col] - _rows[1][_col];
		if (delta_ref <= 0.0)
			return false;

		for (size_t r = 3; r < _rows.size(); ++r)
		{
			double delta = _rows[r][_col] - _rows[r - 1][_col];
			if (delta < 0.0)
				return false;
			if (!equal_double(delta, delta_ref, _tol))
				return false;
		}

		return true;
	}

	bool RowsClose(const std::vector<std::vector<double>> &_a, const std::vector<std::vector<double>> &_b, double _tol)
	{
		if (_a.size() != _b.size() || _a.empty())
			return false;

		for (size_t r = 0; r < _a.size(); ++r)
		{
			if (_a[r].size() != _b[r].size())
				return false;
			for (size_t i = 0; i < _a[r].size(); ++i)
			{
				if (!equal_double(_a[r][i], _b[r][i], _tol))
					return false;
			}
		}

		return true;
	}

	bool RowsCloseOnSharedTimeline(const std::vector<double> &_a_times,
								  const std::vector<std::vector<double>> &_a_rows,
								  const std::vector<double> &_b_times,
								  const std::vector<std::vector<double>> &_b_rows,
								  double _time_tol,
								  double _value_tol)
	{
		if (_a_times.size() != _a_rows.size() || _b_times.size() != _b_rows.size())
			return false;

		auto row_close = [_value_tol](const std::vector<double> &_a, const std::vector<double> &_b) {
			if (_a.size() != _b.size())
				return false;
			for (size_t i = 0; i < _a.size(); ++i)
			{
				if (!equal_double(_a[i], _b[i], _value_tol))
					return false;
			}
			return true;
		};

		auto unique_indices = [&](const std::vector<double> &_times,
								  const std::vector<std::vector<double>> &_rows,
								  std::vector<size_t> &_indices) {
			for (size_t i = 0; i < _times.size(); ++i)
			{
				if (!_indices.empty() && std::abs(_times[i] - _times[_indices.back()]) <= _time_tol)
				{
					if (!row_close(_rows[i], _rows[_indices.back()]))
						return false;
					continue;
				}
				if (!_indices.empty() && _times[i] < _times[_indices.back()])
					return false;
				_indices.push_back(i);
			}
			return true;
		};

		std::vector<size_t> a_indices;
		std::vector<size_t> b_indices;
		if (!unique_indices(_a_times, _a_rows, a_indices) ||
			!unique_indices(_b_times, _b_rows, b_indices))
			return false;

		size_t a = 0;
		size_t b = 0;
		size_t matching_times = 0;
		while (a < a_indices.size() && b < b_indices.size())
		{
			double delta = _a_times[a_indices[a]] - _b_times[b_indices[b]];
			if (std::abs(delta) <= _time_tol)
			{
				if (!row_close(_a_rows[a_indices[a]], _b_rows[b_indices[b]]))
					return false;
				++matching_times;
				++a;
				++b;
			}
			else if (delta < 0.0)
			{
				++a;
			}
			else
			{
				++b;
			}
		}

		// Every point on the shorter unique timeline must have a physical-time
		// match. One task may additionally report a terminal endpoint.
		return matching_times > 0 && matching_times == std::min(a_indices.size(), b_indices.size());
	}

	std::shared_ptr<SpinAPI::SpinSystem> BuildXBandRwaSystem(double _fieldT, double _rotatingFrameFieldT, double _h1FieldT, double _transverseH0FieldT)
	{
		const double ge = 2.00231930436256;

		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(1);");

		std::ostringstream b0Props;
		b0Props << "type=zeeman;spins=E;field=0 0 " << std::setprecision(16) << _fieldT
				<< ";ignoretensors=true;commonprefactor=true;prefactor=" << ge << ";";
		auto b0 = std::make_shared<SpinAPI::Interaction>("B0", b0Props.str());

		std::ostringstream rotProps;
		rotProps << "type=zeeman;spins=E;field=0 0 " << std::setprecision(16) << _rotatingFrameFieldT
				 << ";ignoretensors=true;commonprefactor=true;prefactor=" << ge << ";";
		auto rotatingOffset = std::make_shared<SpinAPI::Interaction>("RotOffset", rotProps.str());

		std::ostringstream transverseProps;
		transverseProps << "type=zeeman;spins=E;field=" << std::setprecision(16) << _transverseH0FieldT
						<< " 0 0;ignoretensors=true;commonprefactor=true;prefactor=" << ge << ";";
		auto transverseH0 = std::make_shared<SpinAPI::Interaction>("TransverseH0", transverseProps.str());

		std::ostringstream h1Props;
		h1Props << "type=zeeman;spins=E;field=" << std::setprecision(16) << _h1FieldT
				<< " 0 0;ignoretensors=true;commonprefactor=true;prefactor=" << ge << ";";
		auto h1 = std::make_shared<SpinAPI::Interaction>("MW", h1Props.str());

		auto stateUp = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);
		spinsys->Add(b0);
		spinsys->Add(rotatingOffset);
		spinsys->Add(transverseH0);
		spinsys->Add(h1);
		spinsys->Add(stateUp);
		spinsys->ValidateInteractions();

		auto props = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Up;");
		spinsys->SetProperties(props);
		stateUp->ParseFromSystem(*spinsys);
		return spinsys;
	}

	bool FinalSzFromRwaTask(const std::string &_taskType,
							double _fieldT,
							double _rotatingFrameFieldT,
							double _h1FieldT,
							double _transverseH0FieldT,
							double _totalTime,
							double _timeStep,
							double &_sz)
	{
		auto spinsys = BuildXBandRwaSystem(_fieldT, _rotatingFrameFieldT, _h1FieldT, _transverseH0FieldT);

		std::ostringstream props;
		props << "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=1;"
			  << "hamiltonianh0list=B0,RotOffset,TransverseH0;hamiltonianh1list=MW;"
			  << "totaltime=" << std::setprecision(16) << _totalTime << ";"
			  << "timestep=" << std::setprecision(16) << _timeStep << ";";
		if (_taskType == "statichs-direct-spectra")
			props << "propagationmethod=normal;";

		std::string data;
		std::vector<std::vector<double>> rows;
		if (!RunPowderTask(spinsys, _taskType, props.str(), data) ||
			!ParseDataRows(data, rows) ||
			rows.empty() ||
			rows.back().size() < 3)
		{
			return false;
		}

		_sz = rows.back()[2];
		return true;
	}
}

//////////////////////////////////////////////////////////////////////////////
// Time-inf: with decay and zero Hamiltonian, integrated values equal initial.
bool test_task_staticpowder_timeinf_triplet_expected()
{
	auto system = BuildTwoElectronSystem(0.0, 0.0, 0.0, true, 1.0);
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeinf;cidsp=false;spinlist=electron1,electron2;powdersamplingpoints=1;"
						"hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;totaltime=1.0;timestep=0.1;";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);
	ok &= (ss_rows.size() == 1);
	ok &= (hs_rows.size() == 1);
	ok &= CheckTripletStructure(ss_rows[0], 1e-8, 1e-8);
	ok &= CheckTripletStructure(hs_rows[0], 1e-8, 1e-8);
	ok &= RowsClose(ss_rows, hs_rows, 1e-6);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Time-evo: with zero Hamiltonian, values should be constant (no drift).
bool test_task_staticpowder_timeevo_constant_no_drift()
{
	auto system = BuildTwoElectronSystem(0.0, 0.0, 0.0, false, 0.0);
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=electron1,electron2;powdersamplingpoints=1;"
						"hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;totaltime=0.95;timestep=0.1;";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	ok &= (ss_rows.size() >= 2);
	ok &= (hs_rows.size() >= 2);
	ok &= CheckTripletStructure(ss_rows.front(), 1e-8, 1e-8);
	ok &= CheckTripletStructure(hs_rows.front(), 1e-8, 1e-8);
	ok &= RowsConstant(ss_rows, 1e-8);
	ok &= RowsConstant(hs_rows, 1e-8);
	ok &= RowsClose(ss_rows, hs_rows, 1e-6);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Time-evo output represents completed propagation intervals, not the
// unpropagated t = 0 boundary.
bool test_task_staticpowder_timeevo_endpoint_timestamps()
{
	auto system = BuildTwoElectronSystem(0.0, 0.0, 0.0, false, 0.0);
	bool ok = PrepareSystem(system);

	for (const std::string &task_type : {std::string("staticss-powderspectra"), std::string("statichs-direct-spectra")})
	{
		const std::string hs_method = task_type == "statichs-direct-spectra" ? "propagationmethod=normal;" : "";
		const std::string common = "method=timeevo;integration=false;cidsp=false;spinlist=electron1,electron2;"
							   "powdersamplingpoints=1;hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;"
							   "timestep=0.2;" + hs_method;

		std::string zero_time_data;
		ok &= RunPowderTask(system.spinsys, task_type, common + "totaltime=0.0;", zero_time_data);
		ok &= (CountDataRows(zero_time_data) == 0);

		std::string one_step_data;
		std::vector<std::vector<double>> rows;
		std::vector<double> times;
		ok &= RunPowderTask(system.spinsys, task_type, common + "totaltime=0.2;", one_step_data);
		ok &= ParseDataRows(one_step_data, rows, &times);
		ok &= (rows.size() == 1 && times.size() == 1);
		if (times.size() == 1)
			ok &= (std::abs(times.front() - 0.2) < 1.0e-12);

		std::string decimal_grid_data;
		rows.clear();
		times.clear();
		const std::string decimal_grid = common.substr(0, common.find("timestep=0.2;")) +
			"timestep=0.1;" + hs_method + "totaltime=0.3;";
		ok &= RunPowderTask(system.spinsys, task_type, decimal_grid, decimal_grid_data);
		ok &= ParseDataRows(decimal_grid_data, rows, &times);
		ok &= (rows.size() == 3 && times.size() == 3);
		if (times.size() == 3)
			ok &= (std::abs(times.back() - 0.3) < 1.0e-12);
	}

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Time-evo integration: with constant values, integral should be linear in time.
bool test_task_staticpowder_timeevo_integration_linear()
{
	auto system = BuildTwoElectronSystem(0.0, 0.0, 0.0, false, 0.0);
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=true;cidsp=false;spinlist=electron1,electron2;powdersamplingpoints=1;"
						"hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;totaltime=0.95;timestep=0.1;";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	if (ss_rows.empty() || hs_rows.empty())
		return false;

	double dt = 0.1;
	double expected_time = (static_cast<double>(ss_rows.size()) - 1.0) * dt;
	ok &= (expected_time > 0.0);
	ok &= RowsLinearIncrease(ss_rows, 2, 1e-8);
	ok &= RowsLinearIncrease(hs_rows, 2, 1e-8);
	ok &= CheckTripletStructure(ss_rows.back(), 1e-8, 1e-8);
	ok &= CheckTripletStructure(hs_rows.back(), 1e-8, 1e-8);
	ok &= RowsClose(ss_rows, hs_rows, 1e-6);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Time-evo: HS and SS should agree for non-trivial dynamics.
bool test_task_staticpowder_timeevo_ss_hs_agree()
{
	auto system = BuildTwoElectronSystem(1.0, 0.2, 0.0, false, 0.0);
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=electron1,electron2;powdersamplingpoints=3;"
						"hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;totaltime=0.95;timestep=0.1;";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	if (ss_rows.empty() || hs_rows.empty())
		return false;
	ok &= (ss_rows.size() == hs_rows.size());

	const auto &ss_last = ss_rows.back();
	const auto &hs_last = hs_rows.back();
	if (ss_last.size() != hs_last.size())
		return false;

	double max_abs = 0.0;
	for (size_t i = 0; i < ss_last.size(); ++i)
	{
		max_abs = std::max(max_abs, std::abs(ss_last[i]));
		ok &= equal_double(ss_last[i], hs_last[i], 1e-5);
	}

	ok &= (max_abs > 1e-3);
	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Time-evo: HS and SS should agree when relaxation operators are present.
	bool test_task_staticpowder_timeevo_relaxation_hs_ss_agree()
	{
		auto system = BuildTwoElectronSystem(0.0, 0.0, 0.0, false, 0.0);
		bool ok = PrepareSystem(system);

	auto relax = std::make_shared<SpinAPI::Operator>(
		"relax",
		"type=relaxationrandomfields;spins=electron1,electron2;rate=0.2;");
	system.spinsys->Add(relax);
	ok &= (system.spinsys->ValidateOperators(system.spinsystems).size() == 0);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=electron1,electron2;powdersamplingpoints=1;"
						"hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;totaltime=0.95;timestep=0.1;";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	if (ss_rows.empty() || hs_rows.empty())
		return false;
	ok &= (ss_rows.size() == hs_rows.size());
	ok &= (ss_rows.front().size() >= 6);
	ok &= (hs_rows.front().size() >= 6);

	double ss_decay = ss_rows.front()[2] - ss_rows.back()[2];
	double hs_decay = hs_rows.front()[2] - hs_rows.back()[2];
	ok &= (ss_decay > 1e-3);
	ok &= (hs_decay > 1e-3);
	ok &= RowsClose(ss_rows, hs_rows, 5e-4);

		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Phenomenological relaxation is defined in the orientation-specific H0
	// eigenbasis. Rotate the molecular-frame |up_z><up_z| state into a
	// lab-frame Sx coherence while H0 remains Sz. The non-commuting H1 term
	// exercises the NZ transform from the H0 relaxation basis into the
	// propagation eigenbasis.
	bool test_task_staticpowder_phenomenological_relaxation_eigenbasis_three_tasks()
	{
		OneElectronSystem system;

		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"zeeman",
			"type=zeeman;spins=E;field=0 0 1;ignoretensors=true;commonprefactor=false;prefactor=1.0;");
		auto drive = std::make_shared<SpinAPI::Interaction>(
			"drive",
			"type=zeeman;spins=E;field=0.25 0 0;ignoretensors=true;commonprefactor=false;prefactor=1.0;");
		auto state_up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto relax = std::make_shared<SpinAPI::Operator>(
			"Rphen",
			"type=relaxationphenomenological;rate1=0.0;rate2=1.0;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);
		spinsys->Add(zeeman);
		spinsys->Add(drive);
		spinsys->Add(state_up);
		spinsys->Add(relax);
		spinsys->ValidateInteractions();
		auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Up;frame=molecular;");
		spinsys->SetProperties(spinsysParser);

		system.spinsys = spinsys;
		system.spinsystems.push_back(spinsys);
		system.state_up = state_up;

		bool ok = PrepareSystem(system);
		ok &= (system.spinsys->ValidateOperators(system.spinsystems).size() == 0);

		std::string ss_data;
		std::string nz_data;
		std::string hs_data;
		std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=1;"
							"powderorientation=1.5707963267948966 0 1;"
							"hamiltonianh0list=zeeman;hamiltonianh1list=drive;totaltime=0.55;timestep=0.1;";

		ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
		ok &= RunPowderTask(system.spinsys, "staticss-powderspectra-nakajimazwanzig", props, nz_data);
		ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

		std::vector<std::vector<double>> ss_rows;
		std::vector<std::vector<double>> nz_rows;
		std::vector<std::vector<double>> hs_rows;
		ok &= ParseDataRows(ss_data, ss_rows);
		ok &= ParseDataRows(nz_data, nz_rows);
		ok &= ParseDataRows(hs_data, hs_rows);

		if (ss_rows.empty() || nz_rows.empty() || hs_rows.empty())
			return false;

		ok &= (ss_rows.size() == nz_rows.size());
		ok &= (ss_rows.size() == hs_rows.size());
		ok &= RowsClose(ss_rows, nz_rows, 1e-10);
		ok &= RowsClose(ss_rows, hs_rows, 5e-5);

		ok &= (std::abs(ss_rows.front()[0] - ss_rows.back()[0]) > 1e-2);

		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Molecule-fixed powder relaxation axes rotate before the basis transform,
	// while dedicated Bloch-style T2 defaults to a lab-fixed B0 axis. A
	// molecule-fixed z channel at beta=pi/2 becomes an unrotated x channel.
	bool test_task_staticpowder_lindblad_relaxation_axes_follow_orientation()
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);");
		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);

		auto relax_z = std::make_shared<SpinAPI::Operator>(
			"Rz",
			"type=relaxationlindblad;spins=E;rate1=0.0;rate2=0.0;rate3=1.0;");
		auto relax_x = std::make_shared<SpinAPI::Operator>(
			"Rx",
			"type=relaxationlindblad;spins=E;rate1=1.0;rate2=0.0;rate3=0.0;");
		auto t2_z_lab = std::make_shared<SpinAPI::Operator>(
			"T2zLab",
			"type=relaxationt2;spins=E;rate=0.5;");
		auto t2_z_molecular = std::make_shared<SpinAPI::Operator>(
			"T2zMolecular",
			"type=relaxationt2;spins=E;rate=0.5;frame=molecular;");
		spinsys->Add(relax_z);
		spinsys->Add(relax_x);
		spinsys->Add(t2_z_lab);
		spinsys->Add(t2_z_molecular);

		std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems = {spinsys};
		bool ok = (spinsys->ValidateOperators(spinsystems).size() == 0);
		ok &= (relax_z->Frame() == SpinAPI::RelaxationFrame::Molecular);
		ok &= (t2_z_lab->Frame() == SpinAPI::RelaxationFrame::Lab);
		ok &= (t2_z_molecular->Frame() == SpinAPI::RelaxationFrame::Molecular);

		SpinAPI::SpinSpace space(*spinsys);
		space.UseSuperoperatorSpace(true);

		arma::cx_mat basis;
		basis.eye(space.HilbertSpaceDimensions(), space.HilbertSpaceDimensions());

		arma::mat identity;
		identity.eye(3, 3);

		arma::mat beta90;
		const double beta = M_PI / 2.0;
		beta90 = {
			{std::cos(beta), 0.0, std::sin(beta)},
			{0.0, 1.0, 0.0},
			{-std::sin(beta), 0.0, std::cos(beta)}};

		arma::sp_cx_mat rotated_z;
		arma::sp_cx_mat unrotated_z;
		arma::sp_cx_mat expected_x;
		arma::sp_cx_mat rotated_t2_z_lab;
		arma::sp_cx_mat rotated_t2_z_molecular;
		arma::sp_cx_mat powder_t2_eigenbasis;
		arma::sp_cx_mat powder_t2_propagationbasis;
		ok &= space.RelaxationOperatorFrameChangeRotated(relax_z, basis, beta90, rotated_z);
		ok &= space.RelaxationOperatorFrameChangeRotated(relax_z, basis, identity, unrotated_z);
		ok &= space.RelaxationOperatorFrameChange(relax_x, basis, expected_x);
		ok &= space.RelaxationOperatorFrameChangeRotated(t2_z_lab, basis, beta90, rotated_t2_z_lab);
		ok &= space.RelaxationOperatorFrameChangeRotated(t2_z_molecular, basis, beta90, rotated_t2_z_molecular);
		ok &= space.PowderRelaxationOperatorEigenbasis(t2_z_molecular, basis, beta90, powder_t2_eigenbasis);
		ok &= space.PowderRelaxationOperator(t2_z_molecular, basis, beta90, powder_t2_propagationbasis);

		arma::cx_mat rotated_z_dense(rotated_z);
		arma::cx_mat unrotated_z_dense(unrotated_z);
		arma::cx_mat expected_x_dense(expected_x);
		arma::cx_mat rotated_t2_z_lab_dense(rotated_t2_z_lab);
		arma::cx_mat rotated_t2_z_molecular_dense(rotated_t2_z_molecular);
		arma::cx_mat powder_t2_eigenbasis_dense(powder_t2_eigenbasis);
		arma::cx_mat powder_t2_propagationbasis_dense(powder_t2_propagationbasis);

		ok &= (arma::norm(rotated_z_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(rotated_t2_z_lab_dense - unrotated_z_dense, "fro") < 1e-10);
		ok &= (arma::norm(rotated_t2_z_molecular_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(powder_t2_eigenbasis_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(powder_t2_propagationbasis_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(rotated_z_dense - unrotated_z_dense, "fro") > 1e-3);

		SpinAPI::SpinSpace hs_space(*spinsys);
		hs_space.UseSuperoperatorSpace(false);
		SpinAPI::HilbertRelaxationCache rotated_z_cache;
		SpinAPI::HilbertRelaxationCache unrotated_z_cache;
		SpinAPI::HilbertRelaxationCache expected_x_cache;
		ok &= hs_space.PowderRelaxationOperatorHilbert(relax_z, beta90, rotated_z_cache);
		ok &= hs_space.RelaxationOperator(relax_z, unrotated_z_cache);
		ok &= hs_space.RelaxationOperator(relax_x, expected_x_cache);

		arma::cx_mat rho = arma::zeros<arma::cx_mat>(2, 2);
		rho(0, 0) = 1.0;
		arma::cx_mat rotated_z_hs;
		arma::cx_mat unrotated_z_hs;
		arma::cx_mat expected_x_hs;
		ok &= hs_space.ApplyRelaxationHilbert(rotated_z_cache, rho, rotated_z_hs);
		ok &= hs_space.ApplyRelaxationHilbert(unrotated_z_cache, rho, unrotated_z_hs);
		ok &= hs_space.ApplyRelaxationHilbert(expected_x_cache, rho, expected_x_hs);
		ok &= (arma::norm(rotated_z_hs - expected_x_hs, "fro") < 1e-10);
		ok &= (arma::norm(rotated_z_hs - unrotated_z_hs, "fro") > 1e-3);

		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// End-to-end Hilbert powder propagation must use the orientation-specific
	// relaxation cache. At beta=pi/2, molecular Sz dephasing becomes lab Sx
	// dephasing and relaxes an initial |up_z><up_z| state.
	bool test_task_staticpowder_hilbert_lindblad_relaxation_axes_follow_orientation()
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=isotropic(2);");
		auto zeeman = std::make_shared<SpinAPI::Interaction>(
			"zeeman",
			"type=zeeman;spins=E;field=0 0 0;ignoretensors=true;commonprefactor=false;prefactor=1.0;");
		auto state_up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");
		auto relax_z = std::make_shared<SpinAPI::Operator>(
			"Rz",
			"type=relaxationlindblad;spins=E;rate1=0.0;rate2=0.0;rate3=1.0;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);
		spinsys->Add(zeeman);
		spinsys->Add(state_up);
		spinsys->Add(relax_z);
		spinsys->ValidateInteractions();
		auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Up;");
		spinsys->SetProperties(spinsysParser);
		std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems = {spinsys};

		bool ok = state_up->ParseFromSystem(*spinsys);
		ok &= (spinsys->ValidateOperators(spinsystems).size() == 0);

		std::string ss_data;
		std::string hs_data;
		std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=1;"
							"powderorientation=1.5707963267948966 0 1;"
							"hamiltonianh0list=zeeman;hamiltonianh1list=zeeman;totaltime=0.55;timestep=0.1;";

		ok &= RunPowderTask(spinsys, "staticss-powderspectra", props, ss_data);
		ok &= RunPowderTask(spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

		std::vector<std::vector<double>> ss_rows;
		std::vector<std::vector<double>> hs_rows;
		ok &= ParseDataRows(ss_data, ss_rows);
		ok &= ParseDataRows(hs_data, hs_rows);
		if (ss_rows.empty() || hs_rows.empty())
			return false;

		ok &= RowsClose(ss_rows, hs_rows, 1e-6);
		ok &= (ss_rows.front().size() == 3);
		ok &= (ss_rows.front()[2] - ss_rows.back()[2] > 1e-2);

		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// A lab-fixed B1 along x must see the crystallite-rotated g tensor. For this
	// one-spin system H0 is zero and H1 is the only Hamiltonian, so <Sz(t)> has
	// the analytic form 0.5*cos(g_eff*t). Identity orientation gives g_eff=gxx=2;
	// beta=pi/2 gives g_eff=gzz=5.
	bool test_task_staticpowder_h1_zeeman_follows_powder_orientation()
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=matrix(2 0 0; 0 3 0; 0 0 5);");
		auto h0 = std::make_shared<SpinAPI::Interaction>(
			"B0",
			"type=zeeman;spins=E;field=0 0 0;ignoretensors=true;commonprefactor=false;prefactor=1;");
		auto h1 = std::make_shared<SpinAPI::Interaction>(
			"mw",
			"type=zeeman;spins=E;field=1 0 0;ignoretensors=false;commonprefactor=false;prefactor=1;");
		auto state_up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);
		spinsys->Add(h0);
		spinsys->Add(h1);
		spinsys->Add(state_up);
		spinsys->ValidateInteractions();
		auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Up;");
		spinsys->SetProperties(spinsysParser);

		const double final_time = 0.4;
		auto finalSzFor = [&](const std::string &_task_type, double _theta, double &_time, double &_sz) {
			std::ostringstream props;
			props << "method=timeevo;integration=false;cidsp=false;spinlist=E;"
				  << "powderorientation=" << _theta << " 0 1;"
				  << "hamiltonianh0list=B0;hamiltonianh1list=mw;"
				  << "totaltime=" << final_time << ";timestep=0.1;";
			if (_task_type == "statichs-direct-spectra")
				props << "propagationmethod=normal;";

			std::string data;
			std::vector<std::vector<double>> rows;
			if (!RunPowderTask(spinsys, _task_type, props.str(), data) ||
				!ParseDataRows(data, rows) ||
				rows.empty() ||
				rows.back().size() < 3)
			{
				return false;
			}

			std::istringstream stream(data);
			std::string line;
			if (!std::getline(stream, line))
				return false;
			while (std::getline(stream, line))
			{
				if (line.empty())
					continue;
				std::istringstream line_stream(line);
				std::string step_token;
				std::string time_token;
				if (!(line_stream >> step_token >> time_token))
					return false;
				try
				{
					_time = std::stod(time_token);
				}
				catch (const std::exception &)
				{
					return false;
				}
			}

			_sz = rows.back()[2];
			return true;
		};

		bool ok = state_up->ParseFromSystem(*spinsys);
		const double beta90 = M_PI / 2.0;

		for (const std::string task_type : {"staticss-powderspectra", "statichs-direct-spectra"})
		{
			double identity_time = 0.0;
			double rotated_time = 0.0;
			double identity_sz = 0.0;
			double rotated_sz = 0.0;
			bool task_ok = true;
			task_ok &= finalSzFor(task_type, 0.0, identity_time, identity_sz);
			task_ok &= finalSzFor(task_type, beta90, rotated_time, rotated_sz);

			const double expected_identity = 0.5 * std::cos(2.0 * identity_time);
			const double expected_rotated = 0.5 * std::cos(5.0 * rotated_time);
			task_ok &= equal_double(identity_sz, expected_identity, 1e-8);
			task_ok &= equal_double(rotated_sz, expected_rotated, 1e-8);
			task_ok &= (std::abs(identity_sz - rotated_sz) > 0.3);
			if (!task_ok)
			{
				std::cerr << "H1 powder orientation check failed for " << task_type
						  << ": identity time=" << identity_time
						  << ", beta90 time=" << rotated_time
						  << ": identity Sz=" << identity_sz
						  << ", beta90 Sz=" << rotated_sz
						  << ", expected identity=" << expected_identity
						  << ", expected beta90=" << expected_rotated << std::endl;
			}
			ok &= task_ok;
		}

		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Real-field rotating-frame sanity check:
	// X-band EPR at 9.5 GHz gives B_res = omega/(gamma_e g_e) ~= 0.339 T for a
	// free electron. The time-propagation powder tasks do not read mwfrequency;
	// the rotating-frame offset is supplied explicitly as a Zeeman H0 term, as
	// in the Cry input files. At B_res the H0 detuning cancels and the transverse
	// H1 term performs a pi rotation. A 50 mT detuning suppresses that transfer.
	// A transverse H0 field is also added and must be removed by the secular H0
	// builder, so it should not change the resonant result.
	bool test_task_staticpowder_xband_rwa_and_secular_real_field()
	{
		const double gammaElectronRadPerNsT = 8.79410005e+1;
		const double ge = 2.00231930436256;
		const double xBandGHz = 9.5;
		const double omegaXBand = 2.0 * arma::datum::pi * xBandGHz;
		const double resonanceFieldT = omegaXBand / (gammaElectronRadPerNsT * ge);

		const double h1FieldT = 0.002;
		const double rabiAngularFrequency = gammaElectronRadPerNsT * ge * h1FieldT;
		const double piPulseTimeNs = arma::datum::pi / rabiAngularFrequency;
		const double timestepNs = piPulseTimeNs / 40.0;
		const double detunedFieldT = resonanceFieldT + 0.050;

		bool ok = true;
		for (const std::string taskType : {"staticss-powderspectra", "statichs-direct-spectra"})
		{
			double resonantSz = 0.0;
			double resonantWithTransverseH0Sz = 0.0;
			double detunedSz = 0.0;

			ok &= FinalSzFromRwaTask(taskType, resonanceFieldT, -resonanceFieldT, h1FieldT, 0.0, piPulseTimeNs, timestepNs, resonantSz);
			ok &= FinalSzFromRwaTask(taskType, resonanceFieldT, -resonanceFieldT, h1FieldT, 0.050, piPulseTimeNs, timestepNs, resonantWithTransverseH0Sz);
			ok &= FinalSzFromRwaTask(taskType, detunedFieldT, -resonanceFieldT, h1FieldT, 0.0, piPulseTimeNs, timestepNs, detunedSz);

			ok &= (resonantSz < -0.45);
			ok &= (detunedSz > 0.45);
			ok &= equal_double(resonantSz, resonantWithTransverseH0Sz, 2e-4);
		}

		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// The HS direct task historically sampled the upper hemisphere. For comparison
	// with EasySpin GridSymmetry='Ci', users need a true full-sphere grid. This
	// regression uses an anisotropic one-electron microwave Hamiltonian for which
	// the two grids give different averaged dynamics; it verifies that the
	// powderfullsphere keyword is wired into the generated grid.
	bool test_task_hs_direct_powderfullsphere_changes_grid()
	{
		auto spin = std::make_shared<SpinAPI::Spin>("E", "type=electron;spin=1/2;tensor=matrix(2 0 0; 0 3 0; 0 0 5);");
		auto h0 = std::make_shared<SpinAPI::Interaction>(
			"B0",
			"type=zeeman;spins=E;field=0 0 0;ignoretensors=true;commonprefactor=false;prefactor=1;");
		auto h1 = std::make_shared<SpinAPI::Interaction>(
			"mw",
			"type=zeeman;spins=E;field=1 0 0;ignoretensors=false;commonprefactor=false;prefactor=1;");
		auto state_up = std::make_shared<SpinAPI::State>("Up", "spin(E)=|1/2>;");

		auto spinsys = std::make_shared<SpinAPI::SpinSystem>("System");
		spinsys->Add(spin);
		spinsys->Add(h0);
		spinsys->Add(h1);
		spinsys->Add(state_up);
		spinsys->ValidateInteractions();
		auto spinsysParser = std::make_shared<MSDParser::ObjectParser>("spinsyssettings", "initialstate=Up;");
		spinsys->SetProperties(spinsysParser);

		auto finalSzFor = [&](bool fullSphere, double &sz) {
			std::ostringstream props;
			props << "method=timeevo;integration=false;cidsp=false;spinlist=E;"
				  << "powdersamplingpoints=2;powderfullsphere=" << (fullSphere ? "true" : "false") << ";"
				  << "hamiltonianh0list=B0;hamiltonianh1list=mw;"
				  << "totaltime=0.4;timestep=0.1;propagationmethod=normal;";

			std::string data;
			std::vector<std::vector<double>> rows;
			if (!RunPowderTask(spinsys, "statichs-direct-spectra", props.str(), data) ||
				!ParseDataRows(data, rows) ||
				rows.empty() ||
				rows.back().size() < 3)
			{
				return false;
			}

			sz = rows.back()[2];
			return true;
		};

		bool ok = state_up->ParseFromSystem(*spinsys);
		double hemisphereSz = 0.0;
		double fullSphereSz = 0.0;
		ok &= finalSzFor(false, hemisphereSz);
		ok &= finalSzFor(true, fullSphereSz);

		ok &= std::isfinite(hemisphereSz);
		ok &= std::isfinite(fullSphereSz);
		ok &= (std::abs(hemisphereSz - fullSphereSz) > 1e-3);
		return ok;
	}

	//////////////////////////////////////////////////////////////////////////////
	// Time-evo powder CW: HS direct spectra should also work for a non-RP one-electron system.
	bool test_task_staticpowder_timeevo_oneelectron_hs_ss_agree()
	{
		auto system = BuildOneElectronSystem();
		bool ok = PrepareSystem(system);

		std::string ss_data;
		std::string hs_data;
		std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=5;"
							"hamiltonianh0list=zeeman;printtimeframe=pulse;integrationtimeframe=pulse;"
							"pulsesequence=[\"cw 0\"];totaltime=1;timestep=0.1;";

		ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
		ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

		std::vector<std::vector<double>> ss_rows;
		std::vector<std::vector<double>> hs_rows;
		ok &= ParseDataRows(ss_data, ss_rows);
		ok &= ParseDataRows(hs_data, hs_rows);

		if (ss_rows.empty() || hs_rows.empty())
			return false;

		ok &= (ss_rows.size() == hs_rows.size());
		ok &= (ss_rows.front().size() == 3);
		ok &= (hs_rows.front().size() == 3);
		ok &= RowsClose(ss_rows, hs_rows, 1e-5);

		return ok;
	}

//////////////////////////////////////////////////////////////////////////////
// Time-evo: HS and SS should agree for a one-electron thermal powder spectrum.
bool test_task_staticpowder_timeevo_oneelectron_thermal_hs_ss_agree()
{
	auto system = BuildOneElectronThermalSystem();
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=7;"
						"hamiltonianh0list=zeeman;totaltime=0.95;timestep=0.1;";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	if (ss_rows.empty() || hs_rows.empty())
		return false;

	ok &= (ss_rows.size() == hs_rows.size());
	ok &= (ss_rows.front().size() == 3);
	ok &= (hs_rows.front().size() == 3);
	ok &= RowsClose(ss_rows, hs_rows, 1e-10);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Pulse time-evo: HS and SS should agree for a one-electron thermal powder spectrum.
bool test_task_staticpowder_pulse_oneelectron_thermal_hs_ss_agree()
{
	auto system = BuildOneElectronThermalSystem();
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=7;"
						"hamiltonianh0list=zeeman;totaltime=1;timestep=0.1;printtimeframe=pulse;"
						"integrationtimeframe=pulse;pulsesequence=[\"cw 0\"];";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	if (ss_rows.empty() || hs_rows.empty())
		return false;

	ok &= (ss_rows.size() == hs_rows.size());
	ok &= (ss_rows.front().size() == 3);
	ok &= (hs_rows.front().size() == 3);
	ok &= RowsClose(ss_rows, hs_rows, 1e-10);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Instant pulse + free evolution: HS and SS should agree for a one-electron thermal powder spectrum.
bool test_task_staticpowder_instantpulse_oneelectron_thermal_hs_ss_agree()
{
	auto system = BuildOneElectronThermalSystemWithPulse(
		"inst",
		"type=InstantPulse;angle=90;rotationaxis=1 0 0;group=E;");
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=7;"
						"hamiltonianh0list=zeeman;totaltime=1;timestep=0.1;printtimeframe=full;"
						"integrationtimeframe=full;pulsesequence=[\"inst 1.0\"];";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	std::vector<double> ss_times;
	std::vector<double> hs_times;
	ok &= ParseDataRows(ss_data, ss_rows, &ss_times);
	ok &= ParseDataRows(hs_data, hs_rows, &hs_times);

	if (ss_rows.empty() || hs_rows.empty())
		return false;

	ok &= (ss_rows.front().size() == 3);
	ok &= (hs_rows.front().size() == 3);
	ok &= RowsCloseOnSharedTimeline(ss_times, ss_rows, hs_times, hs_rows, 1e-12, 1e-10);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Long pulse: HS and SS should agree for a one-electron thermal powder spectrum.
bool test_task_staticpowder_longpulse_oneelectron_thermal_hs_ss_agree()
{
	auto system = BuildOneElectronThermalSystemWithPulse(
		"lp",
		"type=LongPulse;field=0.0002 0 0;frequency=2.5;pulsetime=2.0;timestep=0.1;group=E;"
		"prefactorlist=1,1,1;commonprefactorlist=true;ignoretensorslist=true;");
	bool ok = PrepareSystem(system);

	std::string ss_data;
	std::string hs_data;
	std::string props = "method=timeevo;integration=false;cidsp=false;spinlist=E;powdersamplingpoints=7;"
						"hamiltonianh0list=zeeman;totaltime=1;timestep=0.1;printtimeframe=pulse;"
						"integrationtimeframe=pulse;pulsesequence=[\"lp 0\"];";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	if (ss_rows.empty() || hs_rows.empty())
		return false;

	ok &= (ss_rows.size() == hs_rows.size());
	ok &= (ss_rows.front().size() == 3);
	ok &= (hs_rows.front().size() == 3);
	ok &= RowsClose(ss_rows, hs_rows, 1e-10);

	return ok;
}

//////////////////////////////////////////////////////////////////////////////
// Add all the test cases
void AddTaskStaticPowderSpectraTests(std::vector<test_case> &_cases)
{
	_cases.push_back(test_case("Task StaticPowderSpectra timeinf triplet", test_task_staticpowder_timeinf_triplet_expected));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo constancy", test_task_staticpowder_timeevo_constant_no_drift));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo endpoint timestamps", test_task_staticpowder_timeevo_endpoint_timestamps));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo integration", test_task_staticpowder_timeevo_integration_linear));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo SS/HS agree", test_task_staticpowder_timeevo_ss_hs_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo relaxation HS/SS agree", test_task_staticpowder_timeevo_relaxation_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra phenomenological relaxation eigenbasis across tasks", test_task_staticpowder_phenomenological_relaxation_eigenbasis_three_tasks));
	_cases.push_back(test_case("Task StaticPowderSpectra Lindblad relaxation axes follow orientation", test_task_staticpowder_lindblad_relaxation_axes_follow_orientation));
	_cases.push_back(test_case("Task StaticPowderSpectra Hilbert Lindblad axes follow orientation", test_task_staticpowder_hilbert_lindblad_relaxation_axes_follow_orientation));
	_cases.push_back(test_case("Task StaticPowderSpectra H1 Zeeman follows powder orientation", test_task_staticpowder_h1_zeeman_follows_powder_orientation));
	_cases.push_back(test_case("Task StaticPowderSpectra X-band RWA and secular H0", test_task_staticpowder_xband_rwa_and_secular_real_field));
	_cases.push_back(test_case("Task StaticHSDirectSpectra powderfullsphere changes grid", test_task_hs_direct_powderfullsphere_changes_grid));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo one-electron HS/SS agree", test_task_staticpowder_timeevo_oneelectron_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo one-electron thermal HS/SS agree", test_task_staticpowder_timeevo_oneelectron_thermal_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra pulse one-electron thermal HS/SS agree", test_task_staticpowder_pulse_oneelectron_thermal_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra instant pulse one-electron thermal HS/SS agree", test_task_staticpowder_instantpulse_oneelectron_thermal_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra long pulse one-electron thermal HS/SS agree", test_task_staticpowder_longpulse_oneelectron_thermal_hs_ss_agree));
}
