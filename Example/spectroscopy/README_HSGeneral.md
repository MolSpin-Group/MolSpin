# HSGeneral migration interface

`HSGeneral` is an additive task interface that maps independent physical
choices to existing, tested Hilbert-space task implementations. Existing task
names and their defaults remain supported. This interface is intended to let
callers migrate incrementally without duplicating MolSpin's compatibility
matrix.

## Keywords

| Keyword | Values | Default |
| --- | --- | --- |
| `dynamics` | `static`, `dynamic` | `static` |
| `calculation` | `timeevolution`, `yields`, `spectra` | `timeevolution` |
| `sampling` | `direct`, `stochastic` | `direct` |
| `approximation` | `full`, `secular` | `full`, except `spectra` defaults to `secular` |
| `secularization` | `true`, `false` | Equivalent override for `approximation` |

All task-specific propagation, pulse, output, and powder-grid keywords are
passed unchanged to the selected implementation.

For example, this preserves the established high-field/RWA spectra path:

```text
Task main
{
    type = HSGeneral;
    calculation = spectra;
    secularization = true;
    method = timeevo;
    spinlist = E1,E2;
    hamiltonianh0list = B0;
    hamiltonianh1list = MW,RotatingFrameOffset;
    powdersamplingpoints = 1200;
}
```

Setting `secularization = false` selects the full-Hamiltonian resonance
spectra implementation. No implicit fallback between these two physical
models is performed.

`HSGeneral` secular spectra require `hamiltonianh0list`. This prevents the
legacy direct-spectra fallback from replacing a requested rotated high-field
Hamiltonian with an unrotated full Hamiltonian when H0 was not classified.

## Current compatibility boundary

The migration supports direct or stochastic static/dynamic non-powder time
evolution and yields, direct static spectra with either Hamiltonian
approximation, and stochastic secular/RWA spectra. The shared powder engine
also supports static or explicitly time-dependent Hamiltonians, direct or
stochastic sampling, time evolution, static exact integrated yields, and
dynamic finite-time yields. Stochastic sampling uses the configured initial
`State`: spins specified by that state remain fixed, while omitted spins are
trace sampled. Powder stochastic calculations rotate every sample factor with
the same orientation used for H0 and H1.

For static powder dynamics, `calculation = timeevolution` selects
`method = timeevo` and writes the populations of the configured State objects.
`calculation = yields` selects `method = timeinf` and writes exact integrated
`rate * P_source` transition yields. Dynamic powder yields instead use
`method = timeevo` and integrate to `totaltime`, because a time-dependent
Hamiltonian has no single time-independent inverse. Supplying a contradictory
`method` is an error. Molecular-frame state-population observables, including
orientation-dependent states such as individual triplet sublevels, rotate with
each crystallite. Transition source states must currently be rotationally
invariant because the reaction operator and yield projector have to be rotated
together.

The following combinations are rejected with an explicit error until their
physics has been migrated and tested:

- full-Hamiltonian stochastic spectra;
- powder propagation with time-dependent transition rates or Pulse objects;
- powder propagation with orientation-dependent transition source states;
- Nakajima-Zwanzig relaxation through `HSGeneral`;
- phenomenological operators with stochastic pure-state sampling;
- thermal or initial-state-dephased stochastic sampling;
- eigen-frame stochastic sampling;
- `secular` non-powder time evolution or yields outside the powder/spectra
  implementation.

Use the existing specialized task names for these calculations during the
migration. Rejecting unsupported combinations is intentional: `HSGeneral`
must never substitute a different approximation silently.

## Shared powder-physics API

The migration uses SpinAPI primitives rather than reimplementing rotations in
individual tasks:

- `SpinSpace::PowderHamiltonianRotated(...)` builds the orientation-specific
  `H0`, `H1`, and total Hamiltonian. `HamiltonianApproximation::Full` retains
  every rotated H0 term; `HamiltonianApproximation::Secular` applies the
  high-field projection to H0 only. H1 is fully rotated in both cases.
- `SpinSpace::PrepareInitialDensityForPowder(...)` has an explicit
  full/secular overload for eigenbasis dephasing. The original overload remains
  a secular compatibility wrapper, so existing powder tasks are unchanged.
- `SpinSpace::RotateStateFactors(...)` applies the same molecular-to-lab spin
  rotation to pure-state and Monte Carlo factors. This is the prerequisite for
  powder trace sampling without constructing a dense density matrix per
  sample.
- State-population projectors use the same cached spin-space rotation, so
  observables and molecular-frame initial conditions remain in the same frame.

The powder direct and stochastic paths use these APIs in the shared
direct-spectra engine. Remaining combinations stay rejected until their
propagation and output contracts have equivalent coverage.
