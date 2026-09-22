# Stochastic Hilbert relaxation

Implementation and validation record, 2026-09-21. Base: `big_merge-1`,
`5e4da3ec7bb9970c142131959f77110a23f5d707`.
The 2026-09-22 integration also retains upstream Makefile commit `810a643`.
The vanadium studies, papers, data, and `MolSpin_Manual_20260916` are excluded.

## Input and quick test

Define the physical process inside the SpinSystem using the existing syntax:

```text
Operator STDephasing
{
    type = relaxationdephasing;
    spins = E1,E2;
    rate = 0.7;
}
```

Both selected spins must be distinct physical spin-1/2 particles. Their positions
in the system need not be adjacent. `rate` is in ns^-1, and task times are in ns.
Stochastic propagation is selected through the existing task options:

```text
type = HSGeneral;
sampling = stochastic;
montecarlosamples = 8192;
autoseed = false;
seed = 431;
propagationmethod = autoexpm;
precision = double;
```

There is no task-level dephasing switch. Change only `sampling=direct` to use the
Hilbert density representation. Other relaxation types currently have no
stochastic unraveling and give an error when their physically used rates are
nonzero. Valid zero-rate Operators are no-ops on the stochastic path.

From the repository root, build and run the complete regression suite:

```bash
cmake -S . -B /tmp/molspin-st-relaxation-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/molspin-st-relaxation-build -j 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ctest --test-dir /tmp/molspin-st-relaxation-build --output-on-failure
```

Run the complete parser-verified example and its checker in a temporary directory:

```bash
molspin_repo="$PWD"
mkdir -p /tmp/molspin-st-relaxation-example
cd /tmp/molspin-st-relaxation-example
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/molspin-st-relaxation-build/molspin \
  "$molspin_repo/Tests/StochasticRelaxation/st_dephasing.msd"
python3 "$molspin_repo/Tests/StochasticRelaxation/check_output.py"
```

This produces `st-direct.dat`, `st-stochastic.dat`, and their logs. The example
uses a synthetic noncommuting angular-frequency Hamiltonian, an initially
singlet electron pair, an unpolarized spectator nucleus, ST dephasing at 0.7
ns^-1, and uniform reaction loss at 0.2 ns^-1. It is a numerical validation model,
not a vanadium parameterization. The stochastic log must say **no density
propagation**. The checker tests populations, time points, and survival.
Repeat with the same seed to reproduce the data; increase `montecarlosamples`
and halve `timestep` separately to distinguish sampling noise from integration
error. A single Monte Carlo error need not decrease monotonically with sample
count. Seeded reproducibility is tested within the same build/standard library and
with the same State objects and spin basis. The shared-system SpinSpace
constructor sorts spin pointers. Different allocation histories can change the
basis order, the finite SU(Z) sample realization, or which sampling path applies.
The SU(Z) distribution and ensemble physics remain basis independent, but equal
seeds do not promise identical finite-sample values across different basis
orders. Roundoff can also differ. The byte-equality regressions control State
construction separately from RNG behavior; this work does not change the
existing SpinSpace ordering policy.

## Mathematical convention and order

The audited density and superspace implementations both define

\[
 P_S=\tfrac14 I-\mathbf S_1\cdot\mathbf S_2,\qquad P_T=I-P_S,
 \qquad \mathcal R(\rho)=-k(P_S\rho P_T+P_T\rho P_S).
\]

The two-spin-1/2 restriction follows from this projector formula;
`Spin::S()` stores twice the physical spin and must equal 1. Define

\[
 U=P_S-P_T=2P_S-I,\qquad U^2=U^\dagger U=I.
\]

Since `U rho U† - rho` equals `-2(PS rho PT + PT rho PS)`, the exact mapping is

\[
 \mathcal R(\rho)=\frac{k}{2}(U\rho U^\dagger-\rho),\qquad
 \lambda=\frac{k}{2},\qquad
 p_{\rm flip}(h)=\frac{1-e^{-kh}}{2}.
\]

The implementation evaluates the probability using `expm1` for small intervals.
A flip applies U to one trajectory and never renormalizes it. Thus relaxation
alone preserves each trajectory's norm, while Haberkorn loss remains in
`G=-iH-K`, with `K=sum_c rate_c P_c/2`. ST coherence decays as `exp(-k t)`;
singlet and triplet populations are constant under relaxation alone.

One-channel relaxation over any finite interval is exact. For several channels,
a Poisson count with mean `h sum_j lambda_j` and independent rate-weighted channel
marks samples the exact relaxation semigroup, including noncommuting unitaries.
It does not apply independent channel parities in a fixed order.

Finite propagation uses relaxation half-step, deterministic full-step,
relaxation half-step. This is **weak second order** for combined noncommuting
H/K/R, subject to accuracy of the deterministic propagator. Dynamic exponential
steps use midpoint H/K. Dynamic RK4 retains start/mid/end stage generators, but
the outer split still limits total order to two. Finite pulse intervals and
subsequent delays use the same split; instantaneous pulses add no relaxation
time. Nonintegral final pulse/delay intervals are included on the new route.
For sinusoidal finite pulses the new stochastic route samples at the midpoint.

## Architecture

- `Operator` owns input validation and physical rates. Invalid ST spin lists
  now fail, including unresolved names, extra spins, and duplicate spins.
  `MSDParser::Load()` returns false if any Operator fails validation; it cannot
  silently discard the requested relaxation and run a different model.
- `SpinAPI` owns the shared sparse ST projector construction used by superspace,
  density, and stochastic representations. It exposes capability queries,
  cache preparation, and a finite stochastic map. Tasks contain no ST algebra.
- `HSStatePreparation` validates the initial state and builds trace samples;
  it no longer decides relaxation policy. Existing thermal-state, coherence,
  and eigenframe restrictions remain.
- `HSReactionRelaxation` explicitly chooses no relaxation, Hilbert density,
  or stochastic trajectories. Operator presence alone no longer forces density
  propagation. Cache preparation follows the current action-updated rates.
- `HSPropagator` wraps the existing factor methods with stochastic half-steps,
  including dynamic RK4 and the pulse timeline. Direct-density equations are
  unchanged.
- Legacy compatible tasks consume the same SpinAPI cache and application
  routine. DirectSpectra uses HSPropagator on its new relaxing factor path.
- A separate RNG is derived from a copy of the configured trace-sampling RNG.
  Empty/zero channels and zero elapsed time consume no relaxation random draws.
  DirectSpectra has an orientation-specific relaxation stream, independent of
  OpenMP scheduling. Floating-point powder reductions may differ at roundoff
  across thread counts.

A common global spin rotation leaves PS, PT and U invariant. Molecular initial
factors still rotate when their represented state is not invariant. The new
rotation operation applies sparse embedded single-spin rotations, avoiding a
dense full-space rotation cache for stochastic initial-state preparation.

## Support matrix

| Task/representation | Nonzero ST Operator | Limits |
|---|---|---|
| SS / SSGeneral / MultiSSGeneral | Existing superspace generator | Existing task restrictions remain |
| HSGeneral, direct | Existing Hilbert density propagation | Existing supported relaxation types retained |
| HSGeneral, stochastic | Random-unitary trajectories | Static/dynamic, normal, autoexpm, fixed-step Krylov, RK4; finite-time calculations |
| StaticHS-Stoch-TimeEvo / Yields | Random-unitary trajectories | Normal and autoexpm; existing radical-pair layout |
| DynamicHS-Stoch-TimeEvo / Yields | Random-unitary trajectories | Autoexpm fixed steps; existing dynamic radical-pair requirements |
| StaticHS-Direct-Spectra, stochastic | Random-unitary trajectories | Normal, autoexpm, fixed-step Krylov; powder and supported pulse types |
| Four legacy direct time/yield tasks | Existing Hilbert density propagation | Unsupported nonzero Operators now fail explicitly |
| Four `StaticHS-*-Symm-Uncoupled` tasks | Rejected | Joint ST operation cannot act separately on the two radical factors |
| Other guarded legacy HS propagation families | Rejected | Their existing algorithms implement no Operator relaxation contribution |

The other guarded families are DynamicHSTimeEvo, PeriodicHSTimeEvo,
MultiDynamicHSTimeEvo, StaticHSSymmetricDecay, StaticRPOnlyHSSymDec, and the
unregistered DynamicHSDirectSpectra implementation. Static resonance/linewidth
calculations are outside this time-propagation change.

Explicitly unsupported:

- Nonzero T1, T2, phenomenological and other non-ST stochastic relaxation.
- Active stochastic relaxation with `method=timeinf`: that algorithm constructs
  a density/superspace solve. Use direct timeinf or finite stochastic integration.
- Legacy stochastic adaptive Krylov plus relaxation: its accepted/retried
  intervals are not exposed safely for noise insertion. Fixed-step HSGeneral
  Krylov is available.
- Joint relaxation in separated-radical representations.
- Anisotropic finite-field legacy Pulse objects under powder rotation. HSGeneral
  already rejects this case; the new relaxing DirectSpectra route does too.
  Supported isotropic finite pulses and instantaneous rotations remain available.
- HSGeneral's existing combination of dynamic Interactions and a task-level
  pulse sequence remains unsupported.

## Audit, regression and numerical evidence

Before edits, a fresh GCC 13.3.0 C++17 Release build with system Armadillo and
OpenMP passed **365/365** internal tests (one CTest executable), 1.80 s. There
were no baseline test failures. Environment warnings about the Conda-provided
libcurl/libtinfo were present before the changes.

The audit confirmed the requested factor of two, units, and overall architecture.
It also found invalid Operators were removed by validation without necessarily
failing file loading, legacy adaptive Krylov needed an explicit restriction,
and legacy stochastic observable contractions could allocate Nsamples² Gram
matrices. The new relaxing paths use columnwise contractions. Tests also exposed
uninitialized legacy `yieldcorrections` flags; all four affected direct/stochastic
yield tasks now default to false. Explicitly configured behavior is preserved.
Other newly initialized legacy RNG/sample defaults replace undefined behavior;
explicit valid seeds retain their original sampling stream.

The additional tests in `Tests/tests_StochasticRelaxation.cpp` cover:

1. ST spin-domain validation and parser failure on invalid objects.
2. Projector algebra, unitary involution, and equality of all three generator
   representations on a complex non-Hermitian test matrix.
3. Exact coherence law, constant S/T populations, and individual norm conservation.
4. Two noncommuting channels versus the exact superspace exponential.
5. Independent RNG derivation, zero/no-op behavior, and repeatability.
6. All HSGeneral factor propagators versus density propagation, uniform reaction
   survival, genuinely dynamic H, molecular powder states, and rotation oracles.
7. All four legacy stochastic tasks versus density references.
8. DirectSpectra's three factor propagators versus density, including powder.
9. Finite pulses, partial final intervals, delays, instantaneous pulses,
   unequal reaction rates, and a nonzero microwave drive.
10. Unsupported Operator, adaptive, timeinf, and separated-radical errors.
11. Six task families with no Operator versus a zero Operator and repeated seeds.
12. Noise-free split order, Monte Carlo RMS convergence across independent seeds,
    and a large sparse factor/memory check.

Initial implementation Release result: **385/385 internal tests passed**, including 20 new
regressions; **1/1 CTest executable passed**, 13.24 s. All 365 pre-existing tests
are unchanged and pass, including direct-HS/SS relaxation and frozen stochastic
references. `git diff --check` passes. No new compiler warning was reported by
the final Release build.

With 8,192 trajectories, `dt=0.025 ns`, and a 2-ns simulation, maximum absolute
observable differences from the corresponding deterministic density results were:

| Comparison | Maximum absolute difference |
|---|---:|
| HSGeneral static, each of normal / autoexpm / Krylov / RK4 | 0.00127664 |
| HSGeneral dynamic, normal | 0.00105645 |
| HSGeneral dynamic, RK4 | 0.00106303 |
| Legacy static time evolution | 0.001255 |
| Legacy static yields | 0.0001065 |
| Legacy dynamic time evolution | 0.001057 |
| Legacy dynamic yields | 0.0000931 |
| DirectSpectra, each of its three supported stochastic propagators | 0.00669778 |
| HSGeneral normalized molecular powder average | 0.0026436 |
| DirectSpectra molecular powder angular integral | 0.0166103 |
| Unequal reactions plus finite microwave pulse, both frameworks | 0.00459072 |

The integration tests use fixed seeds and a six-standard-error bound for a
bounded population/polarization estimator, `6/(2 sqrt(8192)) = 0.0331456`.
Survival is checked separately to 2e-6 or better. The complete input-file example
measured population error **0.0012766377** and survival error **8.129053e-13**.
The noncommuting static fixture explicitly checks that omitting relaxation
changes the result by more than 0.07, so an ignored Operator cannot pass.

Relaxation-only Monte Carlo RMS over 64 independent seeds:

| Trajectories | Observed RMS | Binomial prediction |
|---:|---:|---:|
| 256 | 0.0273405 | 0.0271246 |
| 1,024 | 0.0147419 | 0.0135623 |
| 4,096 | 0.00687418 | 0.00678116 |

For noncommuting H and U, RMS over 32 seeds was 0.0139995, 0.00607961, and
0.00268727 at these same ensemble sizes. This test compares with the exact
ensemble expectation of the chosen split, isolating sampling error.
Independently, noise-free propagation against the full Liouvillian exponential
produced the following Frobenius errors at 1 ns:

| Timestep (ns) | Splitting error |
|---:|---:|
| 0.2 | 0.00531137 |
| 0.1 | 0.00132064 |
| 0.05 | 0.000329716 |
| 0.025 | 0.0000824014 |

Successive ratios are approximately four, confirming second order. The same
seeded powder calculation is also compared at one and two OpenMP threads with
a 1e-12 numerical tolerance.

The 20 focused tests also passed with AddressSanitizer and UndefinedBehaviorSanitizer
(`-O1 -fsanitize=address,undefined -fno-omit-frame-pointer`), with no reported
memory or undefined-behavior error. Instrumentation covered SpinSpace, Operator,
HSPropagator, HSReactionRelaxation, HSStatePreparation, TaskHSGeneral,
StaticHSDirectSpectra, and the four legacy stochastic tasks, plus the test code.
Other baseline library objects were not instrumented. Leak detection was
disabled; this is not a full-library leak or thread-sanitizer certification.
The final run log is `/tmp/molspin-st-relaxation-sanitize/final-results.log`.

Build/test logs for this checkout are in
`/tmp/molspin-st-relaxation-{release-build,final-tests}.log`; the initial audit,
focused numeric outputs, and isolated memory run are under
`/tmp/molspin-st-relaxation-audit/`. These temporary artifacts are excluded from
any commit. The commands above reproduce the maintained Release tests.

## Memory and scope

Finite stochastic relaxation stores D-by-Nsamples factors and sparse cached
Hilbert operators. Neither its kernel nor the new factor propagation branches
form `B B†` or a D²-by-D² superoperator. Density products in the new tests are
small-system **oracles**, not production propagation. The large-D test prepares
and rotates D=8192, Nsamples=4 factors, checks that the prepared density is empty,
and checks sparsity of the ST cache. An isolated Release execution of this
test peaked at **17,692 KiB RSS** (`/usr/bin/time -v`, exit 0). A dense complex
density alone at that D
would occupy 1 GiB.

This is a state/relaxation memory claim, not a claim that every existing task
allocation is linear: `propagationmethod=normal` constructs a dense Hamiltonian
exponential; some existing observable/reaction powder caches can also be dense.
Higham and Krylov avoid that dense exponential. DirectSpectra still has its
historical angular integral normalization, while HSGeneral reports normalized
powder averages; compare each against its corresponding density backend.

Remaining technical debt: general quantum-jump unravelings, safely accepted
adaptive stochastic steps, a shared legacy pulse timeline, and broader legacy
input-validation cleanup. No new relaxation user keywords or Hamiltonian-noise
workarounds were introduced. No existing test expectation was weakened.


## SU(Z) follow-up audit

The user's follow-up prompted an independent check of initial-state sampling,
normalization, and the simultaneous trace/relaxation averages. The sampler
still uses the normalized complex-Gaussian SU(Z) construction described in
[Fay, Lindoy and Manolopoulos, section II.3.2](https://arxiv.org/html/2102.13430#S2.SS3.SSS2):

\[
 |\chi\rangle=\frac{\sum_{n=1}^{Z}(x_n+i y_n)|n\rangle}
 {\sqrt{\sum_{n=1}^{Z}(x_n^2+y_n^2)}},\qquad
 x_n,y_n\sim\mathcal N(0,1),\qquad
 \mathbb E[|\chi\rangle\langle\chi|]=I_Z/Z.
\]

Here Z is the dimension of the **entire omitted spin subspace**, not a separate
classical direction sampled for each nucleus. The initial electronic State is
kept fixed. HSGeneral stores each initial column as
`(|psi_fixed> tensor |chi>)/sqrt(M)`, so its columnwise expectation contraction
already includes the `1/M` sample average and the nuclear identity's `1/Z`
normalization. No extra factor of Z belongs in those normalized observables.
Samples are generated at preparation and propagated through the timeline;
there is no timestep-by-timestep resampling of the nuclear initial condition.
`samplingmethod=suz` remains the default. `coherent` is a separate optional
spin-coherent-state sampling method.

Each of the M trace-sampled vectors receives its own stochastic relaxation
history during that one propagation. Thus `montecarlosamples=1000` means 1,000
vectors per orientation, not 1,000 vectors each propagated through another
ensemble of histories. The same M-column ensemble averages both the nuclear
trace and ST dephasing. There is no separate relaxation-trajectory keyword.

The audit found and fixed a **pre-existing basis-ordering edge case** in
BuildTraceSamples' optimized leading-state path. GetStateSubSpace concatenates
entangled groups without reordering them. For spins ordered A,B,C,N, a singlet
on A,C with B explicitly up produced samples outside the intended initial-state
support. Before the fix, `norm(P*B-B,"fro")` was 4.89898 for 32 samples.
The optimized route is now used only when each entangled group is contiguous;
interleaved groups use the existing sparse projected-Gaussian route. That route
is also uniform SU(Z) sampling on the specified support. The usual leading E1,E2
singlet and its legacy RNG stream are unchanged.

A second pre-existing ordering error affected the shared State construction for
entangled groups of three or more spins. The local ket was built in SpinSpace
order, but the subsequent permutation assumed State-declaration order for the
remaining group members. Dense/sparse projectors and complete vectors now track
the actual ket order. An independent N,A,C,B three-spin probe gave a projector
Frobenius error of 2 before this correction. This matters for both the SU(Z)
fallback and its direct-density reference, so agreement between those two
backends alone was insufficient to expose it.

The first full-suite attempt at the interleaving regression failed because the
test assumed that the shared-system SpinSpace constructor preserves insertion
order; it sorts spin pointers. The test now explicitly specifies A,B,C,N, so its
layout is independent of allocation history. Production spin ordering was not
changed.

Three additional regressions establish:

- Interleaved entangled-state preparation agrees with an independently written
  ket oracle. With 8,192 samples, support error is **1.90e-14**, and the sampled
  density's Frobenius error from the normalized exact support is **0.00288137**.
- A three-spin entangled State is checked in **all 24 permutations** of A,B,C,N.
  Explicit bit-indexed kets independently validate dense and sparse projectors,
  full state vectors, SU(Z) sample support and the ensemble density.
- Static and genuinely dynamic HSGeneral propagation with two hyperfine-coupled
  nuclei (I=1/2 and I=1, hence Z=6), ST dephasing and Haberkorn loss agrees with
  direct density propagation. In the initial standalone run, maximum population
  differences at 8,192 samples were **0.00181227** and **0.00071835**, respectively.
  The final sanitizer run gave **0.00119448** and **0.00127506**; spin ordering
  can change the finite sample realization as explained above. Survival is checked too.

The original ST integration fixtures used an uncoupled spectator nucleus; these
additional cases explicitly exercise trace sampling during electron-nuclear
entanglement. The original frozen SU(Z)-versus-legacy stream test remains in the
full suite. Follow-up build/output logs are in
`/tmp/molspin-st-relaxation-suz-build.log` and
`/tmp/molspin-st-relaxation-audit/suz-tests-results.log`.

Follow-up Release result: **388/388 passed**, zero failures (24.92 s). This
includes all 365 baseline cases and 23 added relaxation/SU(Z) regressions.
The full output is `/tmp/molspin-st-relaxation-suz-tests.log`; per-case results
are in `/tmp/molspin-st-relaxation-build/Testing/Temporary/LastTest.log`.

The final focused **23/23 ASan/UBSan cases passed** after rebuilding the changed
SpinSpace object and relinking the driver. No address/undefined-behavior
diagnostics were reported. As in the initial audit, 11 relevant production
translation units and the focused test driver were instrumented; remaining
objects came from the Release library, and leak detection was disabled. Output:
`/tmp/molspin-st-relaxation-sanitize/suz-all-results.log`.

The documented MSD example was rerun with the final Release binary: maximum
population difference **0.0012766377** and survival error **8.13e-13**, both
passing its checker. The 1,546 excluded study/manual files still match their
preservation hashes. At that validation checkpoint, no files had been staged,
committed or pushed.

There are two statistical averages. For trajectory observable X, nuclear seed
chi and relaxation history eta, the law of total variance gives

\[
 \operatorname{Var}(\bar X)=\frac1M\left(
 \mathbb E_\eta[\operatorname{Var}_\chi(X\mid\eta)]
 +\operatorname{Var}_\eta[\mathbb E_\chi(X\mid\eta)]\right).
\]

SU(Z) self-averaging reduces the trace component. It does not guarantee that
the relaxation-history component vanishes as the nuclear space grows. Thus a
large Z does not by itself justify one or very few ST trajectories. Increase
`montecarlosamples`, examine independent seeds, and converge `timestep`
separately. The exact relaxation-only map does not eliminate the second-order
H/K/R splitting error. This implementation is validated for the supported
Operator/model combinations; it is not a guarantee for arbitrary inputs or
all possible relaxation mechanisms.

## Changed-file manifest

```text
Example/spectroscopy/README_HSGeneral.md
MSDParser/MSDParser.cpp
README.md
RunSection/General/HS/HSPropagator.cpp
RunSection/General/HS/HSPropagator.h
RunSection/General/HS/HSReactionRelaxation.cpp
RunSection/General/HS/HSReactionRelaxation.h
RunSection/General/HS/HSRelaxationValidation.h
RunSection/General/HS/HSStatePreparation.cpp
RunSection/General/HS/HSStatePreparation.h
RunSection/General/HS/STOCHASTIC_RELAXATION.md
RunSection/General/HS/TaskHSGeneral.cpp
RunSection/Tasks/TaskDynamicHSDirectSpectra.cpp
RunSection/Tasks/TaskDynamicHSDirectTimeEvo.cpp
RunSection/Tasks/TaskDynamicHSDirectYields.cpp
RunSection/Tasks/TaskDynamicHSStochTimeEvo.cpp
RunSection/Tasks/TaskDynamicHSStochYields.cpp
RunSection/Tasks/TaskDynamicHSTimeEvo.cpp
RunSection/Tasks/TaskMultiDynamicHSTimeEvo.cpp
RunSection/Tasks/TaskPeriodicHSTimeEvo.cpp
RunSection/Tasks/TaskStaticHSDirectSpectra.cpp
RunSection/Tasks/TaskStaticHSDirectTimeEvo.cpp
RunSection/Tasks/TaskStaticHSDirectTimeEvoSymmUncoupled.cpp
RunSection/Tasks/TaskStaticHSDirectYields.cpp
RunSection/Tasks/TaskStaticHSDirectYieldsSymmUncoupled.cpp
RunSection/Tasks/TaskStaticHSStochTimeEvo.cpp
RunSection/Tasks/TaskStaticHSStochTimeEvoSymmUncoupled.cpp
RunSection/Tasks/TaskStaticHSStochYields.cpp
RunSection/Tasks/TaskStaticHSStochYieldsSymmUncoupled.cpp
RunSection/Tasks/TaskStaticHSSymmetricDecay.cpp
RunSection/Tasks/TaskStaticRPOnlyHSSymDec.cpp
SpinAPI/Operator.cpp
SpinAPI/SpinSpace.h
SpinAPI/SpinSpace/SpinSpace_relaxation.cpp
SpinAPI/SpinSpace/SpinSpace_states.cpp
Tests/StochasticRelaxation/check_output.py
Tests/StochasticRelaxation/st_dephasing.msd
Tests/testmain.cpp
Tests/tests_StochasticRelaxation.cpp
makefile
```

At the initial validation checkpoint, no commits or staged files had been
created. The 1,546 study/manual files in the pre-implementation preservation
manifest were checked by SHA-256 and were unchanged. The 2026-09-22 integration
uses only the 40 paths listed above; study data and the manual are excluded.
