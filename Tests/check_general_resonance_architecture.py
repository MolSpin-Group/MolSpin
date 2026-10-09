#!/usr/bin/env python3
"""Architecture guard for the reusable General resonance capability."""
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
errors = []

def text(rel):
    p = root / rel
    if not p.is_file():
        errors.append(f"missing {rel}")
        return ""
    return p.read_text(encoding="utf-8", errors="replace")

base = Path("RunSection/General/Resonance")
required = [
    "ResonanceTypes.h",
    "ResonanceLineshape.h", "ResonanceLineshape.cpp",
    "ResonanceFieldJacobian.h", "ResonanceFieldJacobian.cpp",
    "ResonanceTransitionDetector.h", "ResonanceTransitionDetector.cpp",
    "ResonanceTransitionMoments.h", "ResonanceTransitionMoments.cpp",
    "GeneralResonanceHamiltonian.h", "GeneralResonanceHamiltonian.cpp",
    "ResonanceSpectrumEvaluator.h", "ResonanceSpectrumEvaluator.cpp",
]
bodies = {}
for name in required:
    rel = str(base / name)
    bodies[name] = text(rel)
    body = bodies[name]
    if "Molecular Spin Dynamics Software" not in body or "See LICENSE.txt" not in body:
        errors.append(f"{rel} lacks MolSpin source/license banner")
    for forbidden in ["BasicTask", "TaskStaticHSResonanceSpectra", "RunSection/Tasks/"]:
        if forbidden in body:
            errors.append(f"{rel} depends on historical Task ownership ({forbidden})")

adapter = bodies.get("GeneralResonanceHamiltonian.cpp", "")
if "HSHamiltonianBuilder" not in adapter or "BuildStatic" not in adapter:
    errors.append("GeneralResonanceHamiltonian is not a thin adapter over the General HS Hamiltonian builder")
for forbidden in ["BaseHamiltonianRotatedZYZ", "InteractionOperatorRotatedZYZ", "CreateZYZRotationMatrix"]:
    if forbidden in adapter:
        errors.append(f"GeneralResonanceHamiltonian reimplements lower-level orientation physics ({forbidden})")

for name, token in [
    ("ResonanceLineshape.cpp", "ResonanceLineshape::Evaluate"),
    ("ResonanceFieldJacobian.cpp", "DiagonalEnergyDerivatives"),
    ("ResonanceTransitionDetector.cpp", "ResonanceTransitionDetector::Detect"),
    ("ResonanceTransitionMoments.cpp", "ResonanceTransitionMoments::Evaluate"),
    ("ResonanceSpectrumEvaluator.cpp", "ResonanceSpectrumEvaluator::Evaluate"),
]:
    if token not in bodies.get(name, ""):
        errors.append(f"{name} does not expose its intended reusable resonance responsibility")

cmake = text("CMakeLists.txt")
makefile = text("makefile")
for src in [
    "ResonanceLineshape.cpp", "ResonanceFieldJacobian.cpp", "ResonanceTransitionDetector.cpp",
    "ResonanceTransitionMoments.cpp", "GeneralResonanceHamiltonian.cpp", "ResonanceSpectrumEvaluator.cpp",
]:
    if f"General/Resonance/{src}" not in cmake:
        errors.append(f"CMake omits General/Resonance/{src}")
    obj = src.replace(".cpp", ".o")
    if obj not in makefile:
        errors.append(f"legacy makefile omits {obj}")
if "PATH_RUNSECTION_GENERAL_RESONANCE" not in makefile:
    errors.append("legacy makefile has no dedicated General resonance object path")

registration = text("Tests/testmain.cpp")
tests = text("Tests/tests_GeneralResonanceCore.cpp")
if "AddGeneralResonanceCoreTests" not in registration:
    errors.append("General resonance tests are not registered")
for token in [
    "frozen legacy isotropic-g sweep parity",
    "frozen legacy axial-g sweep parity",
    "frozen legacy hyperfine sweep parity",
    "preserves full/secular distinction",
]:
    if token not in tests:
        errors.append(f"General resonance qualification is missing test contract: {token}")

# The General task owns lifecycle only; all numerical services are task-free.
services = [
    "ResonanceExecutionPlan", "ResonanceDiagnostics", "ResonanceSystemPreparation",
    "ResonanceOrientationSampler", "ResonanceFieldSweep", "ResonanceDiagonalization",
    "ResonanceSpectrumProcessing",
]
service_files = [stem + ext for stem in services for ext in (".h", ".cpp")]
service_files += ["ResonanceCalculations.h", "ResonanceExactCalculation.cpp",
    "ResonanceExactSweep.cpp", "ResonanceHybridCalculation.cpp", "ResonanceMeshCalculation.cpp"]
for name in service_files:
    body = text(str(base / name))
    for forbidden in ("BasicTask", "TaskStaticHSResonanceSpectra", "RunSection/Tasks/",
                      "RunSettings()", "this->Properties()", "this->Data()"):
        if forbidden in body:
            errors.append(f"General resonance service {name} has task ownership: {forbidden}")
    if name.endswith(".cpp"):
        if f"General/Resonance/{name}" not in cmake or name.replace(".cpp", ".o") not in makefile:
            errors.append(f"General resonance service {name} is missing from a build system")

task_header = text(str(base / "TaskResonanceGeneral.h"))
task = text(str(base / "TaskResonanceGeneral.cpp"))
if "public BasicTask" not in task_header:
    errors.append("TaskResonanceGeneral must own a BasicTask lifecycle directly")
for forbidden in ("TaskStaticHSResonanceSpectra", "eig_sym", "RotateState", "BaseHamiltonian",
                  "ThermalStateFromHamiltonian", "CreateZYZRotationMatrix", "std::exp("):
    if forbidden in task or forbidden in task_header:
        errors.append(f"TaskResonanceGeneral dispatches legacy code or owns numerical physics: {forbidden}")
for token in ("ResolveResonanceExecutionPlan", "FieldSweep::Prepare", "ResonanceExactCalculation::Point",
              "ResonanceExactCalculation::Sweep", "ResonanceHybridCalculation::Point",
              "ResonanceMeshCalculation::Sweep", "ResonanceSpectrumProcessing::WriteSample"):
    if token not in task:
        errors.append(f"TaskResonanceGeneral is missing its service seam: {token}")
preparation = text(str(base / "ResonanceSystemPreparation.cpp"))
for token in ("GeneralResonanceHamiltonian", "HSStatePreparation::BuildInitialDensity",
              "HSStatePreparation::PrepareDensityForOrientation"):
    if token not in preparation:
        errors.append(f"General resonance no longer shares General preparation: {token}")
if "ResonanceLineshape::Evaluate" not in text(str(base / "ResonanceSpectrumProcessing.cpp")):
    errors.append("General resonance processing duplicates the lineshape kernel")
for name in ("ResonanceExactCalculation.cpp", "ResonanceExactSweep.cpp", "ResonanceMeshCalculation.cpp"):
    body = text(str(base / name))
    if "ExactResonanceSolver::Generate" not in body or "SystemPreparation::BuildHamiltonian" not in body:
        errors.append(f"{name} bypasses shared General Hamiltonian/line solvers")
factory = text("RunSection/RunSection_CreateTask.cpp")
for token in ('"ResonanceGeneral"', '"statichs-resonance-spectra"', "std::make_shared<General::Resonance::TaskResonanceGeneral>"):
    if token not in factory:
        errors.append(f"General or legacy resonance factory registration is missing: {token}")
if "General/Resonance/TaskResonanceGeneral.cpp" not in cmake or "TaskResonanceGeneral.o" not in makefile:
    errors.append("TaskResonanceGeneral is missing from a build system")
if "AddResonanceGeneralTests(cases)" not in registration or "tests_ResonanceGeneral.cpp" not in makefile:
    errors.append("General task qualification is not registered in both build systems")
qualification = text("Tests/tests_ResonanceGeneral.cpp")
for token in ("legacy parity", "analytic g=2", "full Gibbs", "independent systems", "restart rebuilds"):
    if token not in qualification:
        errors.append(f"General task qualification is missing: {token}")

if errors:
    print("General resonance architecture guard: FAIL")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)
print("General resonance architecture guard: PASS")
