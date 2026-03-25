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
        //dvalue = 2868.91;
        dvalue = 0.1023696699;
        //D = 0;
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
        tensortype = "monochromatic";
        frequency = 0.0;
    }

    Interaction N14nqp
    {
        type = zfs;
        group1 = N14;
        dvalue = -5.01;
        //dvalue = -4.96;
        E = 0.0;
        //prefactor = 1.591549508e-6;
        prefactor = 6.283185306e-3; 
        commonprefactor = false;
        energyshift = true;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1;
        //field = "0.0 0.0 0.0101";
        field = "0.0 0.0 0.5";
    }
    
    Interaction nuclearzeeman
    {
        type = singlespin;
        spins = N14;
        //field = "0.0 0.0 0.101";
        //field = "0.0 0.0 0.095";
        field = "0.0 0.0 0.5";
        prefactor = -0.019327078; //g_n = 3.076Mhz/T -> 19.327078Mrad/sT -> 0.019327078rad/(ns)T
        commonprefactor = false;
    }

//States

    //consider all possible nuclear spin configurations

    State T0      {spin(e1) = |0>;}
    //State TP      {spin(e1) = |1>:}
    //State TD      {spin(e1) = |-1>:}

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
        tensortype = "monochromatic";
        frequency = 0.0;
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
        //field = "0.0 0.0 0.02";
        //field = "0.0 0.0 0.095";
        field = "0.0 0.0 0.5";
    }
    
    Interaction nuclearzeeman
    {
        type = singlespin;
        spins = N14;
        //field = "0.0 0.0 0.02";
        //field = "0.0 0.0 0.095";
        field = "0.0 0.0 0.5";
        prefactor = -0.019327078; //g_n = 3.076Mhz/T -> 19.327078Mrad/sT -> 0.019327078rad/(ns)T
        commonprefactor = false;
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

    //NonRadiativeDecay
    Transition T0UT0UD{source = T0_U; targetsystem = GS; targetstate = T0_U; rate = 0.00425;}
    Transition T0ZT0ZD{source = T0_Z; targetsystem = GS; targetstate = T0_Z; rate = 0.00425;}
    Transition T0DT0DD{source = T0_D; targetsystem = GS; targetstate = T0_D; rate = 0.00425;}
    Transition T0UTPUD{source = T0_U; targetsystem = GS; targetstate = TP_U; rate = 0.000375;}
    Transition T0ZTPZD{source = T0_Z; targetsystem = GS; targetstate = TP_Z; rate = 0.000375;}
    Transition T0DTPDD{source = T0_D; targetsystem = GS; targetstate = TP_D; rate = 0.000375;}
    Transition T0UTDUD{source = T0_U; targetsystem = GS; targetstate = TD_U; rate = 0.000375;}
    Transition T0ZTDZD{source = T0_Z; targetsystem = GS; targetstate = TD_Z; rate = 0.000375;}
    Transition T0DTDDD{source = T0_D; targetsystem = GS; targetstate = TD_D; rate = 0.000375;}

    Transition TPUT0UD{source = TP_U; targetsystem = GS; targetstate = T0_U; rate = 0.051;}
    Transition TPZT0ZD{source = TP_Z; targetsystem = GS; targetstate = T0_Z; rate = 0.051;}
    Transition TPDT0DD{source = TP_D; targetsystem = GS; targetstate = T0_D; rate = 0.051;}
    Transition TPUTPUD{source = TP_U; targetsystem = GS; targetstate = TP_U; rate = 0.0045;}
    Transition TPZTPZD{source = TP_Z; targetsystem = GS; targetstate = TP_Z; rate = 0.0045;}
    Transition TPDTPDD{source = TP_D; targetsystem = GS; targetstate = TP_D; rate = 0.0045;}
    Transition TPUTDUD{source = TP_U; targetsystem = GS; targetstate = TD_U; rate = 0.0045;}
    Transition TPZTDZD{source = TP_Z; targetsystem = GS; targetstate = TD_Z; rate = 0.0045;}
    Transition TPDTDDD{source = TP_D; targetsystem = GS; targetstate = TD_D; rate = 0.0045;}

    Transition TDUT0UD{source = TD_U; targetsystem = GS; targetstate = T0_U; rate = 0.051;}
    Transition TDZT0ZD{source = TD_Z; targetsystem = GS; targetstate = T0_Z; rate = 0.051;}
    Transition TDDT0DD{source = TD_D; targetsystem = GS; targetstate = T0_D; rate = 0.051;}
    Transition TDUTPUD{source = TD_U; targetsystem = GS; targetstate = TP_U; rate = 0.0045;}
    Transition TDZTPZD{source = TD_Z; targetsystem = GS; targetstate = TP_Z; rate = 0.0045;}
    Transition TDDTPDD{source = TD_D; targetsystem = GS; targetstate = TP_D; rate = 0.0045;}
    Transition TDUTDUD{source = TD_U; targetsystem = GS; targetstate = TD_U; rate = 0.0045;}
    Transition TDZTDZD{source = TD_Z; targetsystem = GS; targetstate = TD_Z; rate = 0.0045;}
    Transition TDDTDDD{source = TD_D; targetsystem = GS; targetstate = TD_D; rate = 0.0045;}

    State Identity
    {
    }

    Properties prop
	{
	}
}

Run
{
    Task TE
    {
        type       = MultiStaticSS-timeevolution;
        //type = MultiDynamicHS-timeevolution;
        logfile    = "NV_centre_N14_GSLAC-SS.log";
        datafile   = "NV_centre_N14_GSLAC-SS.dat";
        ReactionOperator = Haberkorn;
        transitionyields = false;
        //propagator = RK45;

        TimeStep   = 1;        // ns
        TotalTime  = 5000;     // ns
        minimumtimestep = 0.001;
        maximumtimestep = 15;
        //atol = 1e-6;
        //rtol = 1e-5;
        atol = 1e-12;
        rtol = 1e-12;
    }
}

Settings
{
    Settings general
	{
		//steps = 20541;
        //steps = 1;
        steps = 720;
	}

    Output fieldstrength
    {
        type = length;
        vector = GS.zeeman.field;
    }

    Output hyperfinefrequency
    {
        type = scalar;
        scalar = GS.E1N14.frequency;
    }

    Action increasefreq1
    {
        type = addscalar;
        scalar = GS.E1N14.frequency;
        value = 0.02;
        first = 1;
        //last = 600;
        last = 60;
        //last = 6;
        loop = true;
    }

    Action increasefreq2
    {
        type = addscalar;
        scalar = ES.E1N14.frequency;
        value = 0.02;
        first = 1;
        //last = 600;
        last = 60;
        loop = true;
    }

    Action increasefieldstrength1
	{
		type = addvector;
		vector = GS.zeeman.field;
		direction = "0 0 1";
        //value = 50e-9;
        //value = 50e-6;
        //value = 0.005;
        value = 0.1;
        //period = 501;
        //period = 6;
        //period = 600;
        period = 60;
        
	}

    Action increasefieldstrength2
	{
		type = addvector;
		vector = GS.nuclearzeeman.field;
		direction = "0 0 1"; 
        //value = 50e-9;
        //value = 50e-6;
        //value = 0.005;
        value = 0.1;
        //period = 501;
        period = 60;
	}

    Action increasefieldstrength3
	{
		type = addvector;
		vector = ES.zeeman.field;
		direction = "0 0 1";
        //value = 10e-7;
        //value = 0.005;
        value = 0.1;
        //period = 501;
        period = 60;
	}

    Action increasefieldstrength4
	{
		type = addvector;
		vector = ES.nuclearzeeman.field;
		direction = "0 0 1"; 
        //value = 10e-7;
        //value = 0.005;
        value = 0.1;
        //period = 501;
        period = 60;
	}
}
