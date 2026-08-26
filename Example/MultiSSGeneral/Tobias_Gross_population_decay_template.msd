// ============================================================================
// Tobias Groß population-decay template for MultiSSGeneral
// ============================================================================
// PURPOSE
//   Demonstrate a directly runnable multi-electronic-manifold calculation with
//   S0 -> S1 optical pumping, reversible S1 <-> singlet-CSS charge separation,
//   CSS spin-state resolved output, and terminal population decay.
//
// IMPORTANT
//   All numerical rates/fields below are DEMONSTRATION VALUES ONLY. Replace them
//   with the measured/fitted values and Hamiltonian parameters of the actual
//   experiment before scientific use.
//
// PHYSICS / LITERATURE
//   Reversible S1/CSS spin-dynamical hierarchy: DOI 10.1039/D6CP00916F
//   Related delayed-fluorescence experiment/model: DOI 10.1039/D6SC02081J
//   Finite Gaussian optical rate precedent: DOI 10.1038/ncomms14000
//   Instantaneous pump/push limit (alternative to the Gaussian below):
//       DOI 10.1126/science.abl4254
//
// `preservespins = N1` DOES NOT mean that the laser drives N1. It means the
// transfer map is the identity on N1, so its populations and coherences are
// transported unchanged between manifolds.
// ============================================================================

SpinSystem S0
{
    // A spin-0 bookkeeping degree of freedom labels this electronic manifold.
    // N1 is included because this example asks the nuclear state to survive
    // optical excitation and later charge separation.
    Spin electronic { spin = 0; }
    Spin N1 { spin = 1; type = nucleus; }

    State S0State
    {
        spin(electronic) = |0>;
        // N1 is intentionally omitted: the State support is identity on N1.
    }

    Transition OpticalPump
    {
        type = sink;
        source = S0State;
        targetsystem = S1;
        targetstate = S1State;

        // N1 is unchanged by this transfer. This is NOT a driven-spin list.
        preservespins = N1;

        // Incoherent finite laser pulse represented as a Gaussian rate k(t).
        // transferfraction=0.65 fixes the integrated one-way action so an
        // isolated S0 -> S1 channel would transfer 65% asymptotically.
        rate = 0;
        rateprofile = gaussian;
        pulsecenter = 20.0;
        pulsefwhm = 5.0;
        transferfraction = 0.65;
    }

    Properties properties
    {
        initialstate = S0State;
        initialpopulation = 1.0;
    }
}

SpinSystem S1
{
    Spin electronic { spin = 0; }
    Spin N1 { spin = 1; type = nucleus; }

    State S1State
    {
        spin(electronic) = |0>;
        // Again N1 is omitted because it is a preserved memory degree of freedom.
    }

    Transition ChargeSeparation
    {
        type = sink;
        source = S1State;
        targetsystem = CSS;
        targetstate = Singlet;
        preservespins = N1;
        rate = 0.020;          // DEMO VALUE, inverse MolSpin time unit
    }

    Transition S1Decay
    {
        type = sink;
        source = S1State;
        rate = 0.005;          // DEMO terminal decay; product not explicitly represented
    }

    // No initialstate: this manifold starts with exactly zero population.
    Properties properties { }
}

SpinSystem CSS
{
    Spin e1
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic("2.0023");
    }
    Spin e2
    {
        spin = 1/2;
        type = electron;
        tensor = isotropic("2.0023");
    }
    Spin N1
    {
        spin = 1;
        type = nucleus;
    }

    // Minimal demonstration Hamiltonian. Replace/extend this block with the
    // experimentally justified g tensors, hyperfine tensors, exchange,
    // dipolar interactions, and relaxation model for the real dyad/triad.
    Interaction B0
    {
        type = zeeman;
        field = "0 0 0.10";
        spins = e1,e2;
    }

    // Singlet/triplet are SUBSPACES of this one quantum CSS SpinSystem. They
    // are not separate electronic SpinSystems in the full quantum model.
    State Singlet
    {
        spins(e1,e2) = |1/2,-1/2> - |-1/2,1/2>;
        // N1 omitted -> identity support on the nuclear subspace.
    }
    State T0
    {
        spins(e1,e2) = |1/2,-1/2> + |-1/2,1/2>;
    }
    State Tplus
    {
        spin(e1) = |1/2>;
        spin(e2) = |1/2>;
    }
    State Tminus
    {
        spin(e1) = |-1/2>;
        spin(e2) = |-1/2>;
    }
    State Identity { }

    // Reversible return from the *singlet CSS subspace* to S1 while keeping the
    // complete N1 density operator. Quantum S/T mixing inside CSS should come
    // from the CSS Hamiltonian/relaxation, not an extra classical k_ST unless a
    // deliberately reduced kinetic model is being fitted.
    Transition ReverseChargeSeparation
    {
        type = sink;
        source = Singlet;
        targetsystem = S1;
        targetstate = S1State;
        preservespins = N1;
        rate = 0.0015;         // DEMO VALUE
    }

    Transition SingletProduct
    {
        type = sink;
        source = Singlet;
        rate = 0.0020;         // DEMO terminal singlet-product loss
    }
    Transition T0Product
    {
        type = sink;
        source = T0;
        rate = 0.0005;         // DEMO terminal triplet loss
    }
    Transition TplusProduct
    {
        type = sink;
        source = Tplus;
        rate = 0.0005;
    }
    Transition TminusProduct
    {
        type = sink;
        source = Tminus;
        rate = 0.0005;
    }

    // No initialstate: CSS starts empty and is populated through S1.
    Properties properties { }
}

Settings
{
    Settings general
    {
        steps = 1;
        notifications = details;
    }
}

Run
{
    Task PopulationDecay
    {
        type = MultiSSGeneral;
        calculation = timeevolution;

        // RK4 is required for finite time-dependent Gaussian rates/events.
        propagationmethod = rk4;
        timestep = 0.10;
        totaltime = 500.0;

        // `both` prints total manifold populations and all named State/subspace
        // populations. transitionfluxes adds k(t) Tr(G rho_source) columns.
        observables = both;
        transitionfluxes = true;

        // `full` uses the fixed full Hamiltonian. For powder calculations use
        // rotated_zyz plus a common powder grid; all manifolds then receive the
        // same molecular orientation sample.
        hamiltonianmode = full;

        logfile = "Tobias_Gross_population_decay.log";
        datafile = "Tobias_Gross_population_decay.dat";
    }
}
