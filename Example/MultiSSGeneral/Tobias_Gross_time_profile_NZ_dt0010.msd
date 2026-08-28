// General/MultiSS smoke test: finite optical pumping plus local NZ relaxation.
// Numerical parameters are synthetic regression values, NOT Tobias Groß data.
SpinSystem S0
{
    Spin label { spin=0; }
    Spin N1 { type=nucleus; spin=1; }
    State G { spin(label)=|0>; }
    Transition pump
    {
        type=sink; source=G; targetsystem=S1; targetstate=E;
        preservespins=N1;
        rate=0; rateprofile=gaussian; pulsecenter=2.0; pulsefwhm=0.5;
        transferfraction=0.4;
    }
    Properties properties { initialstate=G; initialpopulation=1.0; }
}
SpinSystem S1
{
    Spin label { spin=0; }
    Spin N1 { type=nucleus; spin=1; }
    State E { spin(label)=|0>; }
    Transition cs { type=sink; source=E; targetsystem=CSS; targetstate=Singlet; preservespins=N1; rate=0.08; }
    Transition fluorescence { type=sink; source=E; rate=0.02; }
    Properties properties { }
}
SpinSystem CSS
{
    // Synthetic rotational-diffusion regression values, NOT Tobias data.
    // g principal values used to derive G_GTA: (2.0020,2.0030,2.0060).
    Spin e1 { type=electron; spin=1/2; tensor=isotropic(2.0036666667); }
    Spin e2 { type=electron; spin=1/2; tensor=isotropic(2.0030); }
    Spin N1 { type=nucleus; spin=1; }
    Interaction B1
    {
        type=zeeman; field="0 0 0.02"; spins=e1;
        // Isotropic rotational modulation: T00 does not fluctuate.
        // With def_g=1 the five equal values are operator factors (J amplitude=G^2).
        ops=0; terms=1; def_g=1; coeff=0;
        g=0,0.115779707143,0.115779707143,0.115779707143,0.115779707143,0.115779707143;
        tau_c=0.6;
    }
    Interaction B2 { type=zeeman; field="0 0 0.02"; spins=e2; }
    Interaction A1
    {
        type=hyperfine; group1=e1; group2=N1; ignoretensors=true;
        // Synthetic principal HFC=(0.0002,0.0003,0.0008) T; static Aiso only.
        tensor=isotropic(0.000433333333); prefactor=2.0036666667;
        ops=0; terms=1; def_g=1; coeff=0;
        g=0,0.035823423785,0.035823423785,0.035823423785,0.035823423785,0.035823423785;
        tau_c=0.6;
    }
    State Singlet { spins(e1,e2)=|1/2,-1/2> - |-1/2,1/2>; }
    State T0      { spins(e1,e2)=|1/2,-1/2> + |-1/2,1/2>; }
    State Tp      { spin(e1)=|1/2>; spin(e2)=|1/2>; }
    State Tm      { spin(e1)=|-1/2>; spin(e2)=|-1/2>; }
    Transition back { type=sink; source=Singlet; targetsystem=S1; targetstate=E; preservespins=N1; rate=0.01; }
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
        relaxationmodel=nakajima_zwanzig;
        propagationmethod=krylov; krylovdimension=30; krylovtolerance=1e-10;
        timestep=0.01; totaltime=10;
        observables=both; transitionfluxes=true;
        datafile="Tobias_Gross_time_profile_NZ_dt0010.dat";
        logfile="Tobias_Gross_time_profile_NZ_dt0010.log";
    }
}
