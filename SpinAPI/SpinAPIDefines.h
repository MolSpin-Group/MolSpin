/////////////////////////////////////////////////////////////////////////
// Defines (SpinAPI Module)
// ------------------
// Definitions used by the SpinAPI Module.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Defines
#define MOD_SpinAPI_Defines

namespace SpinAPI
{
	// Used by the Spin class
	enum class SpinType
	{
		Electron,
		Nucleus,
		NotSpecified,
	};

	// Used by the Trajectory class
	enum class InterpolationType
	{
		Stepwise, // Stepwise constant function
		Linear,	  // Linear interpolation
	};

	// Used by the Interaction class
	enum class InteractionType
	{
		Undefined,
		SingleSpin,
		DoubleSpin,
		QuadraticSpin,
		Exchange,
		Zfs,
		SemiClassicalField,
		Strain
	};

	// Used by the Interaction class to determine the time-dependence of the field for SingleSpin interactions
	enum class InteractionFieldType
	{
		Static,
		LinearPolarization,	  // Monochromatic linearly polarized radiation, parameters: "frequency", "phase"
		CircularPolarization, // Monochromatic circularly polarized radiation, parameters: "frequency", "phase", "axis"
		Broadband,			  // Broadband noise, parameters: "minfreq", "maxfreq", "stdev", "components", "randomorientations"
		OUGeneral,			  // Ornstein-Uhlenbeck noise, parameters: "correlationtime", "stdev", "timestep", "randomorientations"
		Trajectory,			  // Time-dependence is given by a trajectory
	};

	// Used by the Interaction class to determine the time-dependence of the tensor for DoubleSpin interactions
	enum class InteractionTensorType
	{
		Static,
		Monochromatic, // Monochromatic noise, parameters: "frequency", "phase", "amplitude"
		Broadband,	   // Broaband noise, parameters: "minfreq", "maxfreq", "stdev", "components"
		OUGeneral,	   // Ornstein-Uhlenbeck noise, parameters: "correlationtime", "stdev", "timestep"
		Trajectory,	   // Time-dependence is given by a trajectory
	};

	// Used by the Transition class
	enum class TransitionType
	{
		Source,
		Sink,
	};

	// Frame convention for initial density matrices in powder calculations.
	// Fixed:      the density matrix is already in the lab frame.
	// Molecular: the density matrix follows the molecular frame and is rotated
	//            for each powder orientation before propagation.
	// Eigen:     reserved for states constructed directly from a Hamiltonian
	//            eigenbasis, e.g. thermal states.
	enum class StateFrame
	{
		Fixed,
		Molecular,
		Eigen,
	};

	// Optional treatment of coherences in a prepared initial density matrix.
	// DephaseEigenbasis keeps populations in the selected Hamiltonian
	// eigenbasis and removes coherences before propagation starts.
	enum class InitialStateCoherenceMode
	{
		Keep,
		DephaseEigenbasis,
	};

	// Spatial frame used by spin-operator relaxation channels in powder tasks.
	// Lab:       axes remain fixed relative to the external magnetic field.
	// Molecular: axes follow the molecular orientation used for the current
	//            powder point before any Hamiltonian-basis transformation.
	enum class RelaxationFrame
	{
		Lab,
		Molecular,
	};

	// The types of supported reaction operators
	enum class ReactionOperatorType
	{
		Unspecified, // Used by the transition class - SpinAPI::SpinSpace will use the reaction operator type assigned to it if a transition has an unspecified reaction operator type
		Haberkorn,
		Lindblad,
	};

	// The types of special operators defined in SpinAPI::Operator objects
	enum class OperatorType
	{
		Unspecified,
		RelaxationLindblad, // Raw Cartesian single-spin Lindblad channels D[Sx], D[Sy], D[Sz]
		RelaxationLindbladDoubleSpin,
		RelaxationDephasing,
		RelaxationRandomFields, // Cartesian random-field channels sum_j rate_j D[Sj]
		RelaxationT1,		   // Symmetric high-temperature population transfer: rate/2 * (D[S+] + D[S-])
		RelaxationT2,		   // Pure dephasing: 2 * rate * D[Sz], where rate is 1/Tphi for spin 1/2
		RelaxationPhenomenological, // Global population/coherence damping in the working eigenbasis
	};

	// The types of special operators defined in SpinAPI::Operator objects
	enum class PulseType
	{
		Unspecified,
		InstantPulse,
		LongPulse,
		LongPulseStaticField,
		MWPulse,
		ShapedPulse,
	};
	//typedef std::vector<std::tuple<std::string, double>> PulseSequence;

	// Types of standard outputs based on ActionTargets, to be used to by the StandardOutput class
	enum class StandardOutputType
	{
		VectorXYZ,
		VectorAngle,
		VectorLength,
		VectorDot,
		Scalar,
		Undefined,
	};
}

#endif
