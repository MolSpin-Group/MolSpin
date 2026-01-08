//#include runsettings
// -------------------------------------------------------------
SpinSystem RPSystem
{
	// ---------------------------------------------------------
	// Spins
	// ---------------------------------------------------------
	Spin E1
	{
		type = electron;
		tensor = isotropic(2.0023);
		spin = 1/2;
	}	
	Spin E2
	{
		type = electron;
		tensor = isotropic(2.0023);
		spin = 1/2;
	}
	Spin FADN5
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin FADN10
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin FADH6
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin FADHBeta1
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin FADH8123
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin FADH7123
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin FADH9
	{
		tensor = isotropic("1.0");
		spin = 1/2;
	}
    Spin TRPNE1
    {
        tensor = isotropic("1.0");
		spin = 1/2;
    }
    Spin TRPHE1
    {
        tensor = isotropic("1.0");
		spin = 1/2;
    }
    Spin TRPHE3
    {
        tensor = isotropic("1.0");
		spin = 1/2;
    }
    Spin TRPHZ2
    {
        tensor = isotropic("1.0");
		spin = 1/2;
    }
    Spin TRPHH2
    {
        tensor = isotropic("1.0");
		spin = 1/2;
    }
    Spin TRPHD1
    {
        tensor = isotropic("1.0");
		spin = 1/2;
    }
    Spin TRPHbeta1
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
		spins = E1,E2;
	}

	Interaction FADHYP1 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADN5; tensor = matrix("-0.099 -0.003 0.000; -0.003 -0.087 0.000; 0.000 0.000 1.757"); }
    Interaction FADHYP2 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADN10; tensor = matrix("-0.015 -0.002 0.000; -0.002 -0.024 0.000; 0.000 0.000 0.605"); }
    Interaction FADHYP3 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADH6; tensor = matrix("-0.201 0.033 0.000; 0.033 -0.527 0.000; 0.000 0.000 -0.434"); }
    Interaction FADHYP4 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADHBeta1; tensor = matrix("0.407 0.0 0.0; 0.0 0.407 0.0; 0.0 0.0 0.407"); }
    Interaction FADHYP5 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADH8123; tensor = matrix("0.440 0.000 0.000; 0.000 0.440 0.000; 0.000 0.000 0.440"); }
    Interaction FADHYP6 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADH7123; tensor = matrix("-0.142 0.0 0.0; 0.0 -0.142 0.0; 0.0 0.0 -0.142"); }
    Interaction FADHYP7 { prefactor = 1.0e-3; type = hyperfine; group1 = E1; group2 = FADH9; tensor = matrix("0.067 -0.025 0.0; -0.025 0.108 0.0; 0.0 0.0 -0.005"); }
    Interaction TRPHYP1 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPNE1; tensor = matrix("-0.053 0.059 -0.046; 0.059 0.564 -0.565; -0.046 -0.565 0.453"); }
    Interaction TRPHYP2 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPHE1; tensor = matrix("-1.001 0.206 0.193; 0.206 -0.442 0.307; 0.193 0.307 -0.352"); }
    Interaction TRPHYP3 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPHE3; tensor = matrix("-0.571 0.161 0.196; 0.161 -0.484 0.084; 0.196 0.084 -0.408"); }
    Interaction TRPHYP4 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPHZ2; tensor = matrix("-0.443 0.127 0.149; 0.127 -0.354 0.095; 0.149 0.095 -0.294"); }
    Interaction TRPHYP5 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPHH2; tensor = matrix("-0.043 -0.074 -0.068; -0.074 -0.279 -0.032; -0.068 -0.032 -0.303"); }
    Interaction TRPHYP6 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPHD1; tensor = matrix("-0.275 -0.157 -0.175; -0.157 -0.273 0.092; -0.175 0.092 -0.285"); }
    Interaction TRPHYP7 { prefactor = 1.0e-3; type = hyperfine; group1 = E2; group2 = TRPHbeta1; tensor = matrix("1.572 0.016 0.047; 0.016 1.516 0.063; 0.047 0.063 1.726"); }

	// ---------------------------------------------------------

	// ---------------------------------------------------------
	Interaction dipolar
	{
		type = dipole;
		group1=E1;
		group2=E2;
		IgnoreTensors=true;
		Prefactor=2.0023;
		tensor = matrix("8.623266749825693e-05 -0.0002195846233716766 5.037162196142737e-05;-0.00021958462337167664 -0.00025606501555984707 0.00010312541375461049;5.0371621961427384e-05 0.00010312541375461049 0.00016983234806159007");
	}

 	// ---------------------------------------------------------
	// Spin States
	// ---------------------------------------------------------
	State Singlet	// |S>
	{
		spins(E1,E2) = |1/2,-1/2> - |-1/2,1/2>;
	}
	
	State T0	// |T0>
	{
		spins(E1,E2) = |1/2,-1/2> + |-1/2,1/2>;
	}
	
	State Tp	// |T+>
	{
		spin(E2) = |1/2>;
		spin(E1) = |1/2>;
	}
	
	State Tm	// |T->
	{
		spin(E2) = |-1/2>;
		spin(E1) = |-1/2>;
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
		rate = 0.02;

	}
		Transition Product2
	{
		type = sink;
		source = T0;	// spin-independent reaction
		rate = 0.00;

	}
	Transition Product3
	{
		type = sink;
		source = Tp;	// spin-independent reaction
		rate = 0.00;

	}
	Transition Product4
	{
		type = sink;
		source = Tm;	// spin-independent reaction
		rate = 0.00;

	}
	Transition Product_identity
	{
		type = sink;
		source = Identity;	// spin-independent reaction
		rate = 0.01;

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
		type = statichs-stoch-yields;
		logfile = "SSE_logfile2.txt";
		datafile = "SSE_result2.dat";
		//transitionyields = false;
		transitionyields = true;
		initialstate = Singlet;
		sampligmethod = "SUZ";
		propagationmethod = "autoexpm";
		precision = "double";
		yieldcorrections = false;
		montecarlosamples = 3;
		totaltime = 2000;
		timestep = 1;
		autoseed = false;
		seed = 66904056;

	}
}
