# HSGeneral and standalone HS spectroscopy

`HSGeneral` is MolSpin's production interface for generic, single-SpinSystem
Hilbert-space propagation, yields and polarization observables. Its
implementation lives entirely in `RunSection/General/HS/` and does not dispatch
to historical tasks.

Field-swept spectroscopy is intentionally separate. Use
`StaticHS-Direct-Spectra` or `StaticHS-Resonance-Spectra` for the corresponding
spectroscopy algorithms. `StaticHS-Direct-Spectra` is standalone again and
retains its pulse, relaxation, trace-sampling, full/secular and SO(3) powder
capabilities without any `HSGeneralConfiguration` helper.

## HSGeneral selectors

```text
Task main
{
    type = HSGeneral;

    dynamics = static;           # static | dynamic
    calculation = timeevolution; # timeevolution | yields
    sampling = direct;           # direct | stochastic
    approximation = full;        # full | secular
}
```

`calculation=spectra` is rejected; use a standalone spectrum task.

When `approximation=secular`, `hamiltonianh0list` is required. H0 is
secularized while `hamiltonianh1list` remains full, both for a single
orientation and for powder calculations.

## State-aware trace sampling

The SpinSystem's actual initial `State` defines what remains fixed. Omitted
spins form the trace-sampled subspace. A singlet electron pair with unpolarized
nuclei can use

```text
sampling = stochastic;
samplingmethod = suz;
montecarlosamples = 128;
autoseed = false;
seed = 12345;
```

with the nuclear spins omitted from the singlet State definition.

A general relaxation dissipator cannot be combined with pure-state trace
sampling and is rejected. Use `sampling=direct` for density-matrix relaxation.

## Powder and SO(3)

```text
powdergrid = uniform;
powdersamplingpoints = 200;
powderdomain = full;
powdergammapoints = 16;
hamiltonianh0list = B0,HFC1,HFC2,HFC3;
```

`powderfullsphere=true` remains compatible with `powderdomain=full`. Full SO(3) sampling is required when a
second lab-frame direction, such as a linearly polarized B1, removes axial
symmetry about B0.

## RWA/high-field versus explicit time-dependent driving

Rotating-frame/high-field construction:

```text
approximation = secular;
hamiltonianh0list = B0,HFCs;
hamiltonianh1list = MW,Offset;
```

H0 is secularized; H1 is retained in full.

For explicit RF propagation, define the B1 field as a time-dependent
`Interaction` and use `dynamics=dynamic`. With `approximation=full`, the full
Hamiltonian is propagated. With `approximation=secular`, only H0 is
secularized; the explicit dynamic drive remains full. Powder calculations
rotate the dynamic interaction at every crystallite, including gamma in SO(3)
mode.

## Dynamic powder RYDMR

```text
Task RYDMR
{
    type = HSGeneral;
    dynamics = dynamic;
    calculation = yields;
    sampling = stochastic;
    approximation = full;

    samplingmethod = suz;
    montecarlosamples = 128;
    autoseed = false;
    seed = 12345;

    powdergrid = uniform;
    powdersamplingpoints = 200;
    powderdomain = full;
    powdergammapoints = 16;
    hamiltonianh0list = B0,HFC1,HFC2,HFC3;

    totaltime = 5000;
    timestep = 0.5;
    propagationmethod = krylov;
    transitionyields = true;
    reactionoperators = haberkorn;
}
```

## Pulse preparation

The historical task-level pulse preparation syntax is supported:

```text
pulsesequence = ["pulse1 10.0"], ["pulse2 0"];
```

Instant, `LongPulseStaticField` and `LongPulse` Pulse objects are supported in
static HSGeneral. The sequence prepares the state for subsequent free evolution
or a static `timeinf` solve, and time-evolution calculations may also emit or
integrate the pulse/delay timeline via `printtimeframe` and
`integrationtimeframe`. For continuous excitation and for a physical powder B1
field, prefer a time-dependent Interaction object.

The separate SpinAPI `PulseSequence` timeline object used by multi-system SS
workflows is not automatically activated by single-system HSGeneral.

## Polarization, CIDNP/CIDSP and quantum yields

State populations are the default time-evolution observables. To report spin
polarization:

```text
spinlist = N1,N2;
cidsp = false;
```

which outputs lab-frame Ix/Iy/Iz.

For transition-conditioned product polarization:

```text
spinlist = N1;
cidnp = true;   # cidsp=true is equivalent
```

which outputs `k * I_alpha * P_source`.

For ordinary product/quantum yields:

```text
calculation = yields;
transitionyields = true;
```

which outputs `k * P_source`.

## Time infinity

Static yields/polarization can request

```text
method = timeinf;
```

HSGeneral solves the time-integrated Liouville equation `L x = -rho0` using
`arma::solve`. It does not explicitly construct `inv(L)`. This is the same
inverse-response physics as the standalone DirectSpectra timeinf route while
avoiding an explicit matrix inverse. Dynamic Hamiltonians use finite-time
integration instead.

## Standalone DirectSpectra

```text
Task spectrum
{
    type = StaticHS-Direct-Spectra;

    sampling = stochastic;       # direct | stochastic
    samplingmethod = suz;
    montecarlosamples = 128;
    autoseed = false;
    seed = 12345;

    approximation = secular;     # secular | full
    hamiltonianh0list = B0;
    hamiltonianh1list = MW;
    powdersamplingpoints = 1200;
    powdergammapoints = 8;       # optional full SO(3)

    method = timeevo;
    spinlist = E1,E2;
}
```

See `docs/HS_GENERAL.md` for the maintained HSGeneral physics/input contract
and `docs/GENERAL_TASK_ARCHITECTURE.md` for the repository-wide task boundary.


## LuDM / spin-1 DirectSpectra migration

For the historical rotating-wave `StaticHS-Direct-Spectra` pattern with
`HamiltonianH0list`, `HamiltonianH1list`, T1/T2 operators, powder averaging and
`spinlist`, use:

```text
Task main
{
    type = HSGeneral;
    dynamics = static;
    calculation = timeevolution;
    sampling = direct;
    approximation = secular;
    propagationmethod = normal;
    totaltime = 200;
    timestep = 1.0;
    spinlist = e1;
    hamiltonianh0list = D,Z_e1;
    hamiltonianh1list = pulse1_mwamp,pulse1_rot;
    powdersamplingpoints = 1200;
}
```

The Interaction objects in H1 are not Pulse objects; they are the full H1 part
of the secular/RWA H0+H1 construction.

## Time-infinity CISS polarization migration

For historical `method=timeinf` DirectSpectra calculations that used
`spinlist`, use `calculation=yields; method=timeinf`. A supplied `spinlist`
selects integrated polarization unless `transitionyields=true` is explicitly
requested.
