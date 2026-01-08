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
		HyperfineField = "(isotropic(-0.201),1,0.5),
						  (isotropic(0.407),1,0.5),
						  (isotropic(0.440),1,0.5),
						  (isotropic(-0.142),1,0.5),
						  (isotropic(0.067),1,0.5)";
		prefactor = 1.0e-3;
		orientations = 300;
	}
	
	Interaction radical2SemiClassical
	{
		type = semiclassicalfield;
		group1 = RPElectron2;
		HyperfineField = "(isotropic(-0.053),1,0.5),
						  (isotropic(-1.001),1,0.5),
						  (isotropic(-0.571),1,0.5),
						  (isotropic(-0.443),1,0.5),
						  (isotropic(-0.043),1,0.5),
						  (isotropic(-0.275),1,0.5),
					      (isotropic(1.572),1,0.5)";
		prefactor = 1.0e-3;
		orientations = 300;
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
		rate = 0.000;

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
		rate = 0.000;

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
		logfile = "SW_log3.txt";
		datafile = "SW_result3.dat";
		transitionyields = false;
		totaltime = 1000;
		timestep = 1;
	}
}
