//#include runsettings
// -------------------------------------------------------------
SpinSystem RPSystem
{
	// ---------------------------------------------------------
	// Spins
	// ---------------------------------------------------------
	Spin RPElectron1
	{
		type = electron;
		tensor = isotropic(2.0023);
		spin = 1/2;
	}
	
	Spin RPElectron2
	{
		type = electron;
		tensor = isotropic(2.0023);
		spin = 1/2;
	}
	Spin FN5
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
	Spin FN10
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}

	// ---------------------------------------------------------
	// Interactions
	// ---------------------------------------------------------
	Interaction zeeman1
	{
		type = zeeman;
		field = "0 0 5e-05";
		spins = RPElectron1,RPElectron2;
	}

	Interaction FADHYP1
	{
		type = hyperfine;
		group1 = RPElectron1;
		group2 = FN5;
		tensor = matrix("-0.099 -0.003 0.000; -0.003 -0.087 0.000; 0.000 0.000 1.757");
		prefactor = 1.0e-3;
	}
 	Interaction FADHYP2
	{
		type = hyperfine;
		group1 = RPElectron1;
		group2 = FN10;
		tensor = matrix("-0.015 -0.002 0.000; -0.002 -0.024 0.000; 0.000 0.000 0.605");
		prefactor = 1.0e-3;
	}

	Interaction radical1SemiClassical
	{
		type = semiclassicalfield;
		group1 = RPElectron1;
		HyperfineField = "(matrix(-0.201 0.033 0.000; 0.033 -0.527 0.000; 0.000 0.000 -0.434),1,0.5),
						  (matrix(0.407 0.0 0.0; 0.0 0.407 0.0; 0.0 0.0 0.407),1,0.5),
						  (matrix(0.440 0.000 0.000; 0.000 0.440 0.000; 0.000 0.000 0.440),1,0.5),
						  (matrix(-0.142 0.0 0.0; 0.0 -0.142 0.0; 0.0 0.0 -0.142),1,0.5),
						  (matrix(0.067 -0.025 0.0; -0.025 0.108 0.0; 0.0 0.0 -0.005),1,0.5)";
		prefactor = 1.0e-3;
		orientations = 250;
	}
	
	Interaction radical2SemiClassical
	{
		type = semiclassicalfield;
		group1 = RPElectron2;
		HyperfineField = "(matrix(-0.053 0.059 -0.046; 0.059 0.564 -0.565; -0.046 -0.565 0.453),1,0.5),
						  (matrix(-1.001 0.206 0.193; 0.206 -0.442 0.307; 0.193 0.307 -0.352),1,0.5),
						  (matrix(-0.571 0.161 0.196; 0.161 -0.484 0.084; 0.196 0.084 -0.408),1,0.5),
						  (matrix(-0.443 0.127 0.149; 0.127 -0.354 0.095; 0.149 0.095 -0.294),1,0.5),
						  (matrix(-0.043 -0.074 -0.068; -0.074 -0.279 -0.032; -0.068 -0.032 -0.303),1,0.5),
						  (matrix(-0.275 -0.157 -0.175; -0.157 -0.273 0.092; -0.175 0.092 -0.285),1,0.5),
					      (matrix(1.572 0.016 0.047; 0.016 1.516 0.063; 0.047 0.063 1.726),1,0.5)";
		prefactor = 1.0e-3;
		orientations = 250;
	}

	// ---------------------------------------------------------

	// ---------------------------------------------------------
	Interaction dipolar
	{
		type = dipole;
		group1=RPElectron1;
		group2=RPElectron2;
		IgnoreTensors=true;
		Prefactor=2.0023;
		tensor = matrix("8.623266749825693e-05 -0.0002195846233716766 5.037162196142737e-05;-0.00021958462337167664 -0.00025606501555984707 0.00010312541375461049;5.0371621961427384e-05 0.00010312541375461049 0.00016983234806159007");
	}

 	// ---------------------------------------------------------
	// Spin States
	// ---------------------------------------------------------
	State Singlet	// |S>
	{
		spins(RPElectron1,RPElectron2) = |1/2,-1/2> - |-1/2,1/2>;
	}
	
	State T0	// |T0>
	{
		spins(RPElectron1,RPElectron2) = |1/2,-1/2> + |-1/2,1/2>;
	}
	
	State Tp	// |T+>
	{
		spin(RPElectron2) = |1/2>;
		spin(RPElectron1) = |1/2>;
	}
	
	State Tm	// |T->
	{
		spin(RPElectron2) = |-1/2>;
		spin(RPElectron1) = |-1/2>;
	}
	
	State Identity	// Identity projection
	{
	}
	
	// ---------------------------------------------------------
	// Transitions
	// ---------------------------------------------------------
	Transition Product1
	{
		type = sink;
		source = Singlet;	// spin-independent reaction
		rate = 0.00;

	}
		Transition Product2
	{
		type = sink;
		source = T0;	// spin-independent reaction
		rate = 0.000;

	}
	Transition Product3
	{
		type = sink;
		source = Tp;	// spin-independent reaction
		rate = 0.000;

	}
	Transition Product4
	{
		type = sink;
		source = Tm;	// spin-independent reaction
		rate = 0.000;

	}
	Transition Product_identity
	{
		type = sink;
		source = Identity;	// spin-independent reaction
		rate = 0.00;

	}

	Properties Properties
	{
		initialstate = Singlet;
	}
}
Settings
{
	// ---------------------------------------------------------
	// General settings
	// ---------------------------------------------------------
	Settings general
	{
		steps = 1;
	}
	// ---------------------------------------------------------
	// Actions
	// ---------------------------------------------------------
	//Action scan1
	//{
	//	type = rotatevector; 
	//	vector = RPsystem.zeeman1.field; 
	//	axis = "0 1 0";
	//	value = 9;
	//}
	// ---------------------------------------------------------
	// Outputs objects
	// ---------------------------------------------------------
	//Output orientation
	//{
	//	type = vectorxyz;
	//	vector = RPSystem.zeeman1.field;
	//}
}
Run
{
	Task main
	{
		type = StaticSS-timeevolution;
		logfile = "SW_log2.txt";
		datafile = "SW_result2.dat";
		transitionyields = false;
		totaltime = 1000;
		timestep = 1;
	}
}
