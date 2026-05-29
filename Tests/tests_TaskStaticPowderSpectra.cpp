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
		return true;
	}

	bool ParseDataRows(const std::string &_data, std::vector<std::vector<double>> &_rows)
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
			std::vector<double> row;
			while (line_stream >> token)
			{
				if (token_index < 2)
				{
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
			}
		}

		return !_rows.empty();
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

	void TrimDuplicateTailRow(std::vector<std::vector<double>> &_rows, double _tol)
	{
		if (_rows.size() < 2)
			return;
		if (_rows[_rows.size() - 1].size() != _rows[_rows.size() - 2].size())
			return;
		for (size_t i = 0; i < _rows.back().size(); ++i)
		{
			if (!equal_double(_rows[_rows.size() - 1][i], _rows[_rows.size() - 2][i], _tol))
				return;
		}
		_rows.pop_back();
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
	// Powder rotations must rotate anisotropic relaxation axes before the basis
	// transform. A z-axis Lindblad term at beta=pi/2 should become the same
	// superoperator as an unrotated x-axis Lindblad term.
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
		auto t2_z = std::make_shared<SpinAPI::Operator>(
			"T2z",
			"type=relaxationt2;spins=E;rate=1.0;");
		spinsys->Add(relax_z);
		spinsys->Add(relax_x);
		spinsys->Add(t2_z);

		std::vector<std::shared_ptr<SpinAPI::SpinSystem>> spinsystems = {spinsys};
		bool ok = (spinsys->ValidateOperators(spinsystems).size() == 0);

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
		arma::sp_cx_mat rotated_t2_z;
		arma::sp_cx_mat powder_t2_eigenbasis;
		arma::sp_cx_mat powder_t2_propagationbasis;
		ok &= space.RelaxationOperatorFrameChangeRotated(relax_z, basis, beta90, rotated_z);
		ok &= space.RelaxationOperatorFrameChangeRotated(relax_z, basis, identity, unrotated_z);
		ok &= space.RelaxationOperatorFrameChange(relax_x, basis, expected_x);
		ok &= space.RelaxationOperatorFrameChangeRotated(t2_z, basis, beta90, rotated_t2_z);
		ok &= space.PowderRelaxationOperatorEigenbasis(t2_z, basis, beta90, powder_t2_eigenbasis);
		ok &= space.PowderRelaxationOperator(t2_z, basis, beta90, powder_t2_propagationbasis);

		arma::cx_mat rotated_z_dense(rotated_z);
		arma::cx_mat unrotated_z_dense(unrotated_z);
		arma::cx_mat expected_x_dense(expected_x);
		arma::cx_mat rotated_t2_z_dense(rotated_t2_z);
		arma::cx_mat powder_t2_eigenbasis_dense(powder_t2_eigenbasis);
		arma::cx_mat powder_t2_propagationbasis_dense(powder_t2_propagationbasis);

		ok &= (arma::norm(rotated_z_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(rotated_t2_z_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(powder_t2_eigenbasis_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(powder_t2_propagationbasis_dense - expected_x_dense, "fro") < 1e-10);
		ok &= (arma::norm(rotated_z_dense - unrotated_z_dense, "fro") > 1e-3);

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
						"pulsesequence=[\"cw 0\"];totaltime=0;timestep=0.1;";

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
						"hamiltonianh0list=zeeman;totaltime=0;timestep=0.1;printtimeframe=pulse;"
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
						"hamiltonianh0list=zeeman;totaltime=0;timestep=0.1;printtimeframe=full;"
						"integrationtimeframe=full;pulsesequence=[\"inst 1.0\"];";

	ok &= RunPowderTask(system.spinsys, "staticss-powderspectra", props, ss_data);
	ok &= RunPowderTask(system.spinsys, "statichs-direct-spectra", props + "propagationmethod=normal;", hs_data);

	std::vector<std::vector<double>> ss_rows;
	std::vector<std::vector<double>> hs_rows;
	ok &= ParseDataRows(ss_data, ss_rows);
	ok &= ParseDataRows(hs_data, hs_rows);

	TrimDuplicateTailRow(hs_rows, 1e-12);

	if (ss_rows.empty() || hs_rows.empty())
		return false;

	ok &= (ss_rows.size() == hs_rows.size());
	ok &= (ss_rows.front().size() == 3);
	ok &= (hs_rows.front().size() == 3);
	ok &= RowsClose(ss_rows, hs_rows, 1e-10);

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
						"hamiltonianh0list=zeeman;totaltime=0;timestep=0.1;printtimeframe=pulse;"
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
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo integration", test_task_staticpowder_timeevo_integration_linear));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo SS/HS agree", test_task_staticpowder_timeevo_ss_hs_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo relaxation HS/SS agree", test_task_staticpowder_timeevo_relaxation_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra phenomenological relaxation eigenbasis across tasks", test_task_staticpowder_phenomenological_relaxation_eigenbasis_three_tasks));
	_cases.push_back(test_case("Task StaticPowderSpectra Lindblad relaxation axes follow orientation", test_task_staticpowder_lindblad_relaxation_axes_follow_orientation));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo one-electron HS/SS agree", test_task_staticpowder_timeevo_oneelectron_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra timeevo one-electron thermal HS/SS agree", test_task_staticpowder_timeevo_oneelectron_thermal_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra pulse one-electron thermal HS/SS agree", test_task_staticpowder_pulse_oneelectron_thermal_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra instant pulse one-electron thermal HS/SS agree", test_task_staticpowder_instantpulse_oneelectron_thermal_hs_ss_agree));
	_cases.push_back(test_case("Task StaticPowderSpectra long pulse one-electron thermal HS/SS agree", test_task_staticpowder_longpulse_oneelectron_thermal_hs_ss_agree));
}
