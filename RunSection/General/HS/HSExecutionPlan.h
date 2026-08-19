/////////////////////////////////////////////////////////////////////////
// HSExecutionPlan (RunSection::General::HS)
// ------------------
// Physics-only execution description for TaskHSGeneral. This layer parses and
// validates orthogonal choices; it must never name/select legacy BasicTask
// implementations or perform propagation/state construction itself.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_HS_HSExecutionPlan
#define MOD_RunSection_General_HS_HSExecutionPlan

#include <string>
#include <vector>
#include <tuple>
#include "MSDParserfwd.h"
#include "SpinSpace.h"
#include "PowderGrid.h"

namespace RunSection::General::HS
{
	enum class Dynamics { Static, Dynamic };
	enum class Calculation { TimeEvolution, Yields };
	enum class Sampling { Direct, Stochastic };
	enum class OrientationMode { Identity, Powder2D, PowderSO3, Explicit };
	enum class PropagationMethod { Exponential, AutoExpm, Krylov, RK4 };
	enum class YieldMode { FiniteTime, TimeInfinity };
	enum class TimelineWindow { Pulse, FreeEvolution, Full };

	struct HSExecutionPlan
	{
		Dynamics dynamics = Dynamics::Static;
		Calculation calculation = Calculation::TimeEvolution;
		Sampling sampling = Sampling::Direct;
		SpinAPI::HamiltonianApproximation approximation = SpinAPI::HamiltonianApproximation::Full;
		OrientationMode orientation = OrientationMode::Identity;
		PropagationMethod propagation = PropagationMethod::Exponential;
		YieldMode yieldMode = YieldMode::FiniteTime;

		// Powder-grid selection follows the shared SpinAPI PowderGrid vocabulary.
		// Uniform is the historical golden-angle grid. SOPHE symmetry defines its
		// own spatial domain; Octant is the historical symmetry-reduced octant grid.
		SpinAPI::PowderGridType powderGridType = SpinAPI::PowderGridType::Uniform;
		SpinAPI::PowderGridDomain powderDomain = SpinAPI::PowderGridDomain::UpperHemisphere;
		int powderPoints = 0;
		int powderGridSize = 4;
		std::string powderSymmetry = "c1";
		int powderGammaPoints = 1;
		double powderGammaOffset = 0.0;
		double explicitTheta = 0.0;
		double explicitPhi = 0.0;
		double explicitWeight = 1.0;

		double totalTime = 10000.0;
		double timeStep = 1.0;
		std::string precision = "single";
		int krylovSize = 16;
		// Parsed only as an explicit compatibility warning: the current SpinAPI
		// Krylov propagator is controlled by krylovsize and exposes no tolerance.
		bool krylovToleranceSpecified = false;
		double requestedKrylovTolerance = 0.0;

		arma::uword monteCarloSamples = 1;
		std::string samplingMethod = "suz";
		bool autoSeed = true;
		double seed = 1.0;

		std::vector<std::string> h0List;
		std::vector<std::string> h1List;
		bool hasH0List = false;
		bool hasH1List = false;
		bool transitionYields = true;
		bool yieldCorrections = false;

		// Observable selection. With an empty spinList, time-evolution output
		// defaults to configured State populations. A spinList selects Ix/Iy/Iz
		// polarization observables; cidsp=true conditions them on each reaction
		// channel (rate * I_alpha * P_source), which is the historical CIDSP/CIDNP
		// convention used by the spectroscopy tasks.
		std::vector<std::string> spinList;
		bool cidsp = false;

		// Optional task-level pulse timeline using the established syntax
		// pulsesequence=["pulse delay"],... . HSPropagator emits pulse/delay
		// samples when requested and returns the prepared state for subsequent
		// free propagation or a time-infinity solve.
		std::vector<std::tuple<std::string, double>> pulseSequence;
		bool hasPulseSequence = false;

		// Segment-aware time-evolution output. These compatibility keywords are
		// deliberately represented as execution-plan policy rather than being
		// hard-coded into a spectroscopy task. Pulse includes the complete
		// task-level preparation sequence (finite pulses plus their delays).
		TimelineWindow printWindow = TimelineWindow::FreeEvolution;
		TimelineWindow integrationWindow = TimelineWindow::FreeEvolution;
		bool integrateTimeEvolution = false;

		// Initial-state eigenbasis dephasing uses this named Hamiltonian list.
		// If absent, hamiltonianh0list is used when available.
		std::vector<std::string> initialStateHamiltonian;
		bool hasInitialStateHamiltonian = false;

		bool IsPowder() const { return orientation != OrientationMode::Identity; }
		bool IsDynamic() const { return dynamics == Dynamics::Dynamic; }
		bool IsStochastic() const { return sampling == Sampling::Stochastic; }
		bool UsesTimeInfinityYields() const { return calculation == Calculation::Yields && yieldMode == YieldMode::TimeInfinity; }
	};

	bool ResolveExecutionPlan(const MSDParser::ObjectParser &_properties,
		HSExecutionPlan &_plan, std::string &_error);

	const char *ToString(Dynamics _value);
	const char *ToString(Calculation _value);
	const char *ToString(Sampling _value);
	const char *ToString(OrientationMode _value);
	const char *ToString(PropagationMethod _value);
	const char *ToString(YieldMode _value);
	const char *ToString(TimelineWindow _value);
}

#endif
