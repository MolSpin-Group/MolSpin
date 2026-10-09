// Small noncommuting H + ST-dephasing benchmark. Times in ns, rates in ns^-1.
// Z is a synthetic angular-frequency interaction, with no magnetic prefactor.
SpinSystem pair
{
    Spin E1 { type=electron; spin=1/2; }
    Spin E2 { type=electron; spin=1/2; }
    Spin N  { type=nucleus; spin=1/2; }
    Interaction Z
    {
        type=zeeman; spins=E1; field="0 0 0.8";
        ignoretensors=true; commonprefactor=false; prefactor=1;
    }
    State S  { spins(E1,E2)=|1/2,-1/2>-|-1/2,1/2>; }
    State T0 { spins(E1,E2)=|1/2,-1/2>+|-1/2,1/2>; }
    State Tp { spins(E1,E2)=|1/2,1/2>; }
    State Tm { spins(E1,E2)=|-1/2,-1/2>; }
    Transition sinkS  { type=sink; sourcestate=S;  rate=0.2; }
    Transition sinkT0 { type=sink; sourcestate=T0; rate=0.2; }
    Transition sinkTp { type=sink; sourcestate=Tp; rate=0.2; }
    Transition sinkTm { type=sink; sourcestate=Tm; rate=0.2; }
    Operator STDephasing
    {
        type=relaxationdephasing;
        spins=E1,E2;
        rate=0.7;
    }
    Properties initial { initialstate=S; initialstatecoherences=keep; }
}
Settings { Settings general { steps=1; } }
Run
{
    Task direct
    {
        type=HSGeneral; dynamics=static; calculation=timeevolution;
        sampling=direct; propagationmethod=rk4;
        timestep=0.025; totaltime=2;
        logfile="st-direct.log"; datafile="st-direct.dat";
    }
    Task stochastic
    {
        type=HSGeneral; dynamics=static; calculation=timeevolution;
        sampling=stochastic; propagationmethod=autoexpm; precision=double;
        samplingmethod=suz; montecarlosamples=8192; autoseed=false; seed=431;
        timestep=0.025; totaltime=2;
        logfile="st-stochastic.log"; datafile="st-stochastic.dat";
    }
}
