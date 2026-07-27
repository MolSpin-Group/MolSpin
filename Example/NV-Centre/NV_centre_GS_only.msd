SpinSystem GroundState
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
        field = "0.0 0.0 0.102";
        //field = "0.0 0.0 0.0";
    }
    
    Interaction nuclearzeeman
    {
        type = singlespin;
        spins = N14;
        field = "0.0 0.0 0.102";
        //field = "0.0 0.0 0.0";
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

    State Identity
    {

    }

    Properties prop
	{
		initialstate = T0;
	}
}

Run
{
    Task EigenValue
    {
        type = eigenvalues;
        Hamiltonian = true;
        refstates = T0_U, T0_Z, T0_D, TP_U, TP_Z, TP_D, TD_U, TD_Z, TD_D;
        logfile = "eigenvalues_gs_Z.log";
        //datafile = "../../results/eigenvalues_gs_Z.dat";
        datafile = "eigenvalues_gs_Z.dat";
    }
}

Settings
{
    Settings general
	{
		steps = 20000;
	}

    Output fieldstrength
    {
        type = length;
        vector = GroundState.zeeman.field;
    }

    Action increasefieldstrength1
	{
		type = addvector;
		vector = GroundState.zeeman.field;
		direction = "0 0 1";
        value = 50e-9;
        //value = 50e-6;
	}

    Action increasefieldstrength2
	{
		type = addvector;
		vector = GroundState.nuclearzeeman.field;
		direction = "0 0 1"; 
        value = 50e-9;
        //value = 50e-6;
	}
}