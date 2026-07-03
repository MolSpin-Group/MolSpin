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
        //dvalue = -5.01;
        dvalue = -4.95;
        E = 0.0;
        prefactor = 6.283185306e-3; 
        commonprefactor = false;
        energyshift = true;
    }

    Interaction zeeman
    {
        type = singlespin;
        spins = e1;
        field = "0.0 0.0 0.101";
    }
    
    Interaction nuclearzeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.101";
        prefactor = -0.019327078; //g_n = 3.076Mhz/T -> 19.327078Mrad/sT -> 0.019327078rad/(ns)T
        commonprefactor = false;
    }

    Interaction strain
    {
        type = strain;
        group1 = e1;
        E = "-4.0e-5 0.0 0.0 -4.0e-5 0.0 8.3e-4";
        D = "2.3 -6.42 -2.83 -2.6 19.66 5.7";
        prefactor = 0.035682426404996;
        //prefactor = 0.0;
    }
//
    //Interaction strain_modulated
    //{
    //    type = strain;
    //    group1 = e1;
    //    E = "5e-6 5e-6 5e-6 3.535e-6 3.535e-6 3.535e-6";
    //    D = "2.3 -6.42 -2.83 -2.6 19.66 5.7";
    //    //prefactor = 0.035682426404996;
    //    prefactor = 0.0;
    //    //tensortype = broadband;
    //    //minfreq = 0;
    //    //maxfreq = 50;
    //    //components = 50000;
    //    //autoseed = "true";
    //    //rwdcoeff = 0.5;
    //}

    Pulse pulsed_ODMR
    {
        type = LongPulse;
        //field = "1.263e-4 0.0 0.0";
        field = "0.6315e-4 0.0 0.0";
        frequency = 0.062832; //10Mhz
        //pulsetime = 100.0;
        pulsetime = 200.0;
        group = e1,N14;
        prefactorlist = 1.0, 1.0, 1.0, -0.019327078, -0.019327078, -0.019327078;
        commonprefactorlist = true,false;
        ignoretensorslist = false,false;
        timestep = 0.5;
    }

    PulseSequence ODMR_sequence
    {
        tau1 = 1000.0;
        sequence = pulsed_ODMR, tau1;
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

    //Transition T0UP{source = T0_U; targetsystem = ES; targetstate = T0_U; rate = 0.01;}
    //Transition T0ZP{source = T0_Z; targetsystem = ES; targetstate = T0_Z; rate = 0.01;}
    //Transition T0DP{source = T0_D; targetsystem = ES; targetstate = T0_D; rate = 0.01;}
    //Transition TPUP{source = TP_U; targetsystem = ES; targetstate = TP_U; rate = 0.01;}
    //Transition TPZP{source = TP_Z; targetsystem = ES; targetstate = TP_Z; rate = 0.01;}
    //Transition TPDP{source = TP_D; targetsystem = ES; targetstate = TP_D; rate = 0.01;}
    //Transition TDUP{source = TD_U; targetsystem = ES; targetstate = TD_U; rate = 0.01;}
    //Transition TDZP{source = TD_Z; targetsystem = ES; targetstate = TD_Z; rate = 0.01;}
    //Transition TDDP{source = TD_D; targetsystem = ES; targetstate = TD_D; rate = 0.01;}

    Properties prop
	{
		initialstate = T0_U;
	}
}

Run
{
    Task TE
    {
        type       = multistaticss-timeevolution;
        logfile    = "NVtesting/ODMR/10-3.log";
        datafile   = "NVtesting/ODMR/10-3.dat";
        ReactionOperator = Haberkorn;
        transitionyields = false;

        timestep   = 0.1;        // ns
        totaltime  = 200;     // ns
        minimumtimestep = 0.01;
        maximumtimestep = 5;
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
        steps = 401;
	}

    Output fieldstrength
    {
        type = length;
        vector = GS.zeeman.field;
    }

    //Output E
    //{
    //    type = scalar;
    //    scalar = GS.strain.ex;
    //}
//
    //Output delta_E
    //{
    //    type = scalar;
    //    scalar = GS.strain_modulated.ex;
    //}
//
    //Output D
    //{
    //    type = scalar;
    //    scalar = GS.strain_modulated.rwdcoeff;
    //}

    Action increasefieldstrength1
	{
		type = addvector;
		vector = GS.zeeman.field;
		direction = "0 0 1";
        value = 5e-6;
        period = 1;
        
	}

    Action increasefieldstrength2
	{
		type = addvector;
		vector = GS.nuclearzeeman.field;
		direction = "0 0 1"; 
        value = 5e-6;
        period = 1;
	}

    //Action increasefieldstrength3
	//{
	//	type = addvector;
	//	vector = ES.zeeman.field;
	//	direction = "0 0 1";
    //    value = 0.0001;
    //    period = 1;
	//}
//
    //Action increasefieldstrength4
	//{
	//	type = addvector;
	//	vector = ES.nuclearzeeman.field;
	//	direction = "0 0 1"; 
    //    value = 0.0001;
    //    period = 1;
	//}

}
