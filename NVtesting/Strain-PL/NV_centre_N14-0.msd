SpinSystem GS
{
    Spin e1
    {
        spin = 1;
        type = electron;
        tensor = isotropic(2);
    }

    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic(1.0);
    }

//Interactions

    Interaction zfs
    {
        type = zfs;
        group1 = e1;
        dvalue = 0.1023696699;
        E = 0;
        //prefactor = 0.0178412132e-3; //half of the normal conversion because the electron is included twice
        prefactor = 0.5;
        energyshift = true;
    }

    Interaction E1N14
    {
        type = hyperfine;
        group1 = e1;
        group2 = N14;
        tensor = matrix("-2.70 0 0; 0 -2.70 0; 0 0 -2.14"); //Mhz
        prefactor = 0.035682426404996e-3;
    }

    Interaction N14nqp
    {
        type = zfs;
        group1 = N14;
        dvalue = -5.01;
        //dvalue = -4.96;
        E = 0.0;
        prefactor = 6.283185306e-3; 
        commonprefactor = false;
        energyshift = true;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1;
        field = "0.0 0.0 0.09";
    }
    
    Interaction nuclearzeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.09";
        prefactor = -0.019327078; //g_n = 3.076Mhz/T -> 19.327078Mrad/sT -> 0.019327078rad/(ns)T
        commonprefactor = false;
    }

    Interaction strain
    {
        type = strain;
        group1 = e1;
        E = "1e-5 1e-5 1e-5 0.707e-5 0.707e-5 0.707e-5";
        D = "2.3 -6.42 -2.83 -2.6 19.66 5.7";
        prefactor = 0.035682426404996;
        //prefactor = 0.0;
    }

    Interaction strain_modulated
    {
        type = strain;
        group1 = e1;
        E = "5e-6 5e-6 5e-6 3.535e-6 3.535e-6 3.535e-6";
        D = "2.3 -6.42 -2.83 -2.6 19.66 5.7";
        prefactor = 0.035682426404996;
        //prefactor = 0.0;
        tensortype = broadband;
        minfreq = 0;
        maxfreq = 50;
        components = 50000;
        autoseed = "true";
        rwdcoeff = 0.5;
    }

//States

    //consider all possible nuclear spin configurations

    State T0      {spin(e1) = |0>;}
    State TP      {spin(e1) = |1>;}
    State TD      {spin(e1) = |-1>;}

    State T0_U    {spin(e1) = |0>; spin(N14) = |1>;}
    State T0_Z    {spin(e1) = |0>; spin(N14) = |0>;}
    State T0_D    {spin(e1) = |0>; spin(N14) = |-1>;}

    State TP_U    {spin(e1) = |1>; spin(N14) = |1>;}
    State TP_Z    {spin(e1) = |1>; spin(N14) = |0>;}
    State TP_D    {spin(e1) = |1>; spin(N14) = |-1>;}

    State TD_U    {spin(e1) = |-1>; spin(N14) = |1>;}
    State TD_Z    {spin(e1) = |-1>; spin(N14) = |0>;}
    State TD_D    {spin(e1) = |-1>; spin(N14) = |-1>;}

    State Identity
    {
    }

//Transitions

    Transition T0UP{source = T0_U; targetsystem = ES; targetstate = T0_U; rate = 0.01;}
    Transition T0ZP{source = T0_Z; targetsystem = ES; targetstate = T0_Z; rate = 0.01;}
    Transition T0DP{source = T0_D; targetsystem = ES; targetstate = T0_D; rate = 0.01;}
    Transition TPUP{source = TP_U; targetsystem = ES; targetstate = TP_U; rate = 0.01;}
    Transition TPZP{source = TP_Z; targetsystem = ES; targetstate = TP_Z; rate = 0.01;}
    Transition TPDP{source = TP_D; targetsystem = ES; targetstate = TP_D; rate = 0.01;}
    Transition TDUP{source = TD_U; targetsystem = ES; targetstate = TD_U; rate = 0.01;}
    Transition TDZP{source = TD_Z; targetsystem = ES; targetstate = TD_Z; rate = 0.01;}
    Transition TDDP{source = TD_D; targetsystem = ES; targetstate = TD_D; rate = 0.01;}

    Properties prop
	{
		initialstate = T0;
	}
}

SpinSystem ES
{
    Spin e1
    {
        spin = 1;
        type = electron;
        tensor = isotropic(2);
    }

    //nitrogen-14
    Spin N14
    { 
        spin = 1;
        type = nucleus;
        tensor = isotropic(1.0);
    }

    //Interactions

    Interaction zfs
    {
        type = zfs;
        group1 = e1;
        dvalue = 1420;
        prefactor = 0.0178412132e-3; //half of the normal conversion because the electron is included twice
        energyshift = true;
    }

    Interaction E1N14
    {
        type = hyperfine;
        group1 = e1;
        group2 = N14;
        tensor = matrix("40 0 0; 0 40 0; 0 0 -23"); //Mhz
        prefactor = 0.035682426404996e-3;
    }

    Interaction N14nqp
    {
        type = zfs;
        group1 = N14;
        dvalue = -5.01;
        prefactor = 6.283185306e-3; 
        commonprefactor = false;
        energyshift = true;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1;
        field = "0.0 0.0 0.09";
    }
    
    Interaction nuclearzeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.09";
        prefactor = -0.019327078; //g_n = 3.076Mhz/T -> 19.327078Mrad/sT -> 0.019327078rad/(ns)T
        commonprefactor = false;
    }

    Interaction strain
    {
        type = strain;
        group1 = e1;
        E = "1e-5 1e-5 1e-5 0.707e-5 0.707e-5 0.707e-5";
        D = "2.3 -6.42 -2.83 -2.6 19.66 5.7";
        prefactor = 0.48171275646745; //1/28.024 * ~13.5 (the strain succeptibilty is around 13.5+-0.5 times stronger in the ES than the GS)
        //prefactor = 0.0;
    }

    Interaction strain_modulated
    {
        type = strain;
        group1 = e1;
        E = "5e-6 5e-6 5e-6 3.535e-6 3.535e-6 3.535e-6";
        D = "2.3 -6.42 -2.83 -2.6 19.66 5.7";
        prefactor = 0.48171275646745;
        //prefactor = 0.0;
        tensortype = broadband;
        minfreq = 0;
        maxfreq = 50;
        components = 50000;
        autoseed = "true";
        rwdcoeff = 0.5;
    }

    //States

    //consider all possible nuclear spin configurations

    State T0      {spin(e1) = |0>;}

    State T0_U    {spin(e1) = |0>; spin(N14) = |1>;}
    State T0_Z    {spin(e1) = |0>; spin(N14) = |0>;}
    State T0_D    {spin(e1) = |0>; spin(N14) = |-1>;}

    State TP_U    {spin(e1) = |1>; spin(N14) = |1>;}
    State TP_Z    {spin(e1) = |1>; spin(N14) = |0>;}
    State TP_D    {spin(e1) = |1>; spin(N14) = |-1>;}

    State TD_U    {spin(e1) = |-1>; spin(N14) = |1>;}
    State TD_Z    {spin(e1) = |-1>; spin(N14) = |0>;}
    State TD_D    {spin(e1) = |-1>; spin(N14) = |-1>;}

    //Transitions
    //RadiativeDecay

    Transition T0URD{source = T0_U; targetsystem = GS; targetstate = T0_U; rate = 0.077;}
    Transition T0ZRD{source = T0_Z; targetsystem = GS; targetstate = T0_Z; rate = 0.077;}
    Transition T0DRD{source = T0_D; targetsystem = GS; targetstate = T0_D; rate = 0.077;}
    Transition TPURD{source = TP_U; targetsystem = GS; targetstate = TP_U; rate = 0.077;}
    Transition TPZRD{source = TP_Z; targetsystem = GS; targetstate = TP_Z; rate = 0.077;}
    Transition TPDRD{source = TP_D; targetsystem = GS; targetstate = TP_D; rate = 0.077;}
    Transition TDURD{source = TD_U; targetsystem = GS; targetstate = TD_U; rate = 0.077;}
    Transition TDZRD{source = TD_Z; targetsystem = GS; targetstate = TD_Z; rate = 0.077;}
    Transition TDDRD{source = TD_D; targetsystem = GS; targetstate = TD_D; rate = 0.077;}
    Transition T0UISC{source = T0_U; targetsystem = MS; targetstate = I; rate = 0.001;}
    Transition T0ZISC{source = T0_Z; targetsystem = MS; targetstate = I; rate = 0.001;}
    Transition T0DISC{source = T0_D; targetsystem = MS; targetstate = I; rate = 0.001;}

    Transition TPUISC{source = TP_U; targetsystem = MS; targetstate = I; rate = 0.083;}
    Transition TPZISC{source = TP_Z; targetsystem = MS; targetstate = I; rate = 0.083;}
    Transition TPDISC{source = TP_D; targetsystem = MS; targetstate = I; rate = 0.083;}

    Transition TDUISC{source = TD_U; targetsystem = MS; targetstate = I; rate = 0.083;}
    Transition TDZISC{source = TD_Z; targetsystem = MS; targetstate = I; rate = 0.083;}
    Transition TDDISC{source = TD_D; targetsystem = MS; targetstate = I; rate = 0.083;}


    State Identity
    {
    }

    Properties prop
	{
	}
}

SpinSystem MS
{
    State I
    {

    }

    Transition MSGST0U{source = I; targetsystem = GS; targetstate = T0_U; rate = 0.0033;}
    Transition MSGST0Z{source = I; targetsystem = GS; targetstate = T0_Z; rate = 0.0033;}
    Transition MSGST0D{source = I; targetsystem = GS; targetstate = T0_D; rate = 0.0033;}
    Transition MSGSTPU{source = I; targetsystem = GS; targetstate = TP_U; rate = 0.001;}
    Transition MSGSTPZ{source = I; targetsystem = GS; targetstate = TP_Z; rate = 0.001;}
    Transition MSGSTPD{source = I; targetsystem = GS; targetstate = TP_D; rate = 0.001;}
    Transition MSGSTDP{source = I; targetsystem = GS; targetstate = TD_U; rate = 0.001;}
    Transition MSGSTDZ{source = I; targetsystem = GS; targetstate = TD_Z; rate = 0.001;}
    Transition MSGSTDD{source = I; targetsystem = GS; targetstate = TD_D; rate = 0.001;}
}

Run
{
    Task TE
    {
        type       = multistaticss-timeevolution;
        //type = MultiDynamicHS-timeevolution;
        logfile    = "NVtesting/Strain-PL/1.log";
        datafile   = "NVtesting/Strain-PL/1.dat";
        ReactionOperator = Haberkorn;
        transitionyields = false;
        //propagator = RK45;

        timestep   = 0.1;        // ns
        totaltime  = 10000;     // ns
        //totaltime = 1000;
        minimumtimestep = 0.01;
        maximumtimestep = 20;
        //atol = 1e-6;
        //rtol = 1e-5;
        atol = 1e-8;
        rtol = 1e-10;
    }
}

Settings
{
    Settings general
	{
        steps = 501;
	}

    Output fieldstrength
    {
        type = length;
        vector = GS.zeeman.field;
    }

    Output E
    {
        type = scalar;
        scalar = GS.strain.ex;
    }

    Output delta_E
    {
        type = scalar;
        scalar = GS.strain_modulated.ex;
    }

    Output D
    {
        type = scalar;
        scalar = GS.strain_modulated.rwdcoeff;
    }

    Action increasefieldstrength1
	{
		type = addvector;
		vector = GS.zeeman.field;
		direction = "0 0 1";
        value = 0.0001;
        period = 1;
        
	}

    Action increasefieldstrength2
	{
		type = addvector;
		vector = GS.nuclearzeeman.field;
		direction = "0 0 1"; 
        value = 0.0001;
        period = 1;
	}

    Action increasefieldstrength3
	{
		type = addvector;
		vector = ES.zeeman.field;
		direction = "0 0 1";
        value = 0.0001;
        period = 1;
	}

    Action increasefieldstrength4
	{
		type = addvector;
		vector = ES.nuclearzeeman.field;
		direction = "0 0 1"; 
        value = 0.0001;
        period = 1;
	}

}
