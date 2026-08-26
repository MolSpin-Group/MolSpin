// General/MultiSS smoke test: finite optical pumping + local historical NZ.
// Numerical parameters are synthetic regression values, NOT Tobias Groß data.
SpinSystem S0
{
    Spin label { spin=0; }
    State G { spin(label)=|0>; }
    Transition pump
    {
        type=sink; source=G; targetsystem=S1; targetstate=E;
        rate=0; rateprofile=gaussian; pulsecenter=2.0; pulsefwhm=0.5;
        transferfraction=0.4;
    }
    Properties properties { initialstate=G; initialpopulation=1.0; }
}
SpinSystem S1
{
    Spin label { spin=0; }
    State E { spin(label)=|0>; }
    Transition cs { type=sink; source=E; targetsystem=CSS; targetstate=Singlet; rate=0.08; }
    Transition fluorescence { type=sink; source=E; rate=0.02; }
    Properties properties { }
}
SpinSystem CSS
{
    Spin e1 { type=electron; spin=1/2; tensor=isotropic(2.0023); }
    Spin e2 { type=electron; spin=1/2; tensor=isotropic(2.0030); }
    Interaction B1
    {
        type=zeeman; field="0 0 0.02"; spins=e1;
        // Synthetic historical-NZ correlation source.  This exercises the
        // established Interaction g/tau_c syntax; it is not a fitted dyad value.
        ops=1; terms=1; g=0.002; tau_c=0.6;
    }
    Interaction B2 { type=zeeman; field="0 0 0.02"; spins=e2; }
    State Singlet { spins(e1,e2)=|1/2,-1/2> - |-1/2,1/2>; }
    State T0      { spins(e1,e2)=|1/2,-1/2> + |-1/2,1/2>; }
    State Tp      { spin(e1)=|1/2>; spin(e2)=|1/2>; }
    State Tm      { spin(e1)=|-1/2>; spin(e2)=|-1/2>; }
    Transition back { type=sink; source=Singlet; targetsystem=S1; targetstate=E; rate=0.01; }
    Transition ks   { type=sink; source=Singlet; rate=0.015; }
    Transition kt0  { type=sink; source=T0; rate=0.004; }
    Transition ktp  { type=sink; source=Tp; rate=0.004; }
    Transition ktm  { type=sink; source=Tm; rate=0.004; }
    Properties properties { }
}
Settings { Settings general { steps=1; notifications=details; } }
Run
{
    Task combined
    {
        type=MultiSSGeneral; calculation=timeevolution;
        relaxationmodel=historical_nz;
        propagationmethod=rk4; timestep=0.02; totaltime=10;
        observables=both; transitionfluxes=true;
        datafile="Tobias_Gross_time_profile_NZ_smoketest.dat";
        logfile="Tobias_Gross_time_profile_NZ_smoketest.log";
    }
}
