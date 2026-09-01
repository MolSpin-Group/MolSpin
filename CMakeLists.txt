# Set cmake version requirement
cmake_minimum_required(VERSION 3.14)

project(MolSpin LANGUAGES CXX)

option(ENABLE_NATIVE_ARCH "Compile with -march=native when supported." ON)
option(ENABLE_LOOP_UNROLLING "Compile with -funroll-loops when supported." ON)

# MolSpin uses the Armadillo wrapper API, so the BLAS/LAPACK implementation is
# selected by the Armadillo installation. Do not link a particular backend
# here: that would break valid MKL, BLIS, Accelerate, and Conda installations.
find_package(Armadillo 8.500 REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# FindArmadillo in older CMake versions only exports variables. Normalize that
# interface to a target so include paths and libraries propagate together.
if (NOT TARGET Armadillo::Armadillo)
    add_library(Armadillo::Armadillo INTERFACE IMPORTED)
    set_target_properties(Armadillo::Armadillo PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ARMADILLO_LIBRARIES}"
    )
endif()

# Compiler options shared by all MolSpin targets.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" MOLSPIN_HAS_MARCH_NATIVE)
check_cxx_compiler_flag("-funroll-loops" MOLSPIN_HAS_LOOP_UNROLLING)

function(molspin_configure_target target)
    target_compile_definitions(${target} PRIVATE
        ARMA_DONT_PRINT_FAST_MATH_WARNING
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall)
        if (ENABLE_NATIVE_ARCH AND MOLSPIN_HAS_MARCH_NATIVE)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
        if (ENABLE_LOOP_UNROLLING AND MOLSPIN_HAS_LOOP_UNROLLING)
            target_compile_options(${target} PRIVATE -funroll-loops)
        endif()
    endif()
endfunction()

# Paths
set(PATH_SOURCE_SPINAPI "${CMAKE_CURRENT_SOURCE_DIR}/SpinAPI")
set(PATH_SOURCE_MSDPARSER "${CMAKE_CURRENT_SOURCE_DIR}/MSDParser")
set(PATH_SOURCE_RUNSECTION "${CMAKE_CURRENT_SOURCE_DIR}/RunSection")
set(PATH_SOURCE_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/Tests")

# --------------------------------------------------------------------------
# SpinAPI Module
# --------------------------------------------------------------------------
set(SOURCES_SPINAPI
        ${PATH_SOURCE_SPINAPI}/SpinSystem.cpp
        ${PATH_SOURCE_SPINAPI}/Spin.cpp
        ${PATH_SOURCE_SPINAPI}/Interaction.cpp
        ${PATH_SOURCE_SPINAPI}/Transition.cpp
        ${PATH_SOURCE_SPINAPI}/Operator.cpp
        ${PATH_SOURCE_SPINAPI}/Pulse.cpp
        ${PATH_SOURCE_SPINAPI}/State.cpp
        ${PATH_SOURCE_SPINAPI}/SpinSpace.cpp
        ${PATH_SOURCE_SPINAPI}/StandardOutput.cpp
        ${PATH_SOURCE_SPINAPI}/Tensor.cpp
        ${PATH_SOURCE_SPINAPI}/Trajectory.cpp
        ${PATH_SOURCE_SPINAPI}/SubSystem.cpp
        ${PATH_SOURCE_SPINAPI}/Function.cpp
        ${PATH_SOURCE_SPINAPI}/PulseSequence.cpp
        ${PATH_SOURCE_SPINAPI}/PowderGrid.cpp
        ${PATH_SOURCE_SPINAPI}/TimeProfile.cpp
        ${PATH_SOURCE_SPINAPI}/TransferChannel.cpp
        ${PATH_SOURCE_SPINAPI}/QuantumMap.cpp
        ${PATH_SOURCE_SPINAPI}/Relaxation.cpp
        ${PATH_SOURCE_SPINAPI}/Redfield.cpp
        ${PATH_SOURCE_SPINAPI}/NakajimaZwanzig.cpp
)

# --------------------------------------------------------------------------
# MSD Parser Module
# --------------------------------------------------------------------------
set(SOURCES_MSDPARSER
        ${PATH_SOURCE_MSDPARSER}/MSDParser.cpp
        ${PATH_SOURCE_MSDPARSER}/FileReader.cpp
        ${PATH_SOURCE_MSDPARSER}/ObjectParser.cpp
)

# --------------------------------------------------------------------------
# RunSection Module
# --------------------------------------------------------------------------
set(SOURCES_RUNSECTION
        ${PATH_SOURCE_RUNSECTION}/RunSection.cpp
        ${PATH_SOURCE_RUNSECTION}/BasicTask.cpp
        ${PATH_SOURCE_RUNSECTION}/Action.cpp
        ${PATH_SOURCE_RUNSECTION}/Settings.cpp
        ${PATH_SOURCE_RUNSECTION}/OutputHandler.cpp
        ${PATH_SOURCE_RUNSECTION}/Utility.cpp
        ${PATH_SOURCE_RUNSECTION}/CubicSpline.cpp
        ${PATH_SOURCE_RUNSECTION}/General/GeneralOrientationSampler.cpp
        ${PATH_SOURCE_RUNSECTION}/General/Resonance/ResonanceLineshape.cpp
        ${PATH_SOURCE_RUNSECTION}/General/Resonance/ResonanceFieldJacobian.cpp
        ${PATH_SOURCE_RUNSECTION}/General/Resonance/ResonanceTransitionDetector.cpp
        ${PATH_SOURCE_RUNSECTION}/General/Resonance/ResonanceTransitionMoments.cpp
        ${PATH_SOURCE_RUNSECTION}/General/Resonance/GeneralResonanceHamiltonian.cpp
        ${PATH_SOURCE_RUNSECTION}/General/Resonance/ResonanceSpectrumEvaluator.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSExecutionPlan.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSStatePreparation.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSOrientationSampler.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSHamiltonianBuilder.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSReactionRelaxation.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSPropagator.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/HSObservableCollector.cpp
        ${PATH_SOURCE_RUNSECTION}/General/HS/TaskHSGeneral.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSLiouvillianBuilder.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSInteractionRelaxation.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSNakajimaZwanzigBuilder.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSRedfieldBuilder.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSExecutionPlan.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSOrientationSampler.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSSystemPreparation.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSPropagator.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/SSObservableCollector.cpp
        ${PATH_SOURCE_RUNSECTION}/General/SS/TaskSSGeneral.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSExecutionPlan.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSOrientationSampler.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSSystemPreparation.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSNetworkBuilder.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSEventController.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSPropagator.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/MultiSSObservableCollector.cpp
        ${PATH_SOURCE_RUNSECTION}/General/MultiSS/TaskMultiSSGeneral.cpp
)

# --------------------------------------------------------------------------
# RunSection Tasks
# --------------------------------------------------------------------------
set(SOURCES_RUNSECTION_TASKS
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSS.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSSymmetricDecay.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskHamiltonianEigenvalues.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticRPOnlyHSSymDec.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskPeriodicSSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskPeriodicHSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskGammaCompute.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiStaticSSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiStaticSS.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiDynamicHSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiStaticSSTimeEvoSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfield.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfieldSparse.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfieldTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfieldTimeEvoSparse.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiStaticSSRedfieldTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSNakajimaZwanzig.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSNakajimaZwanzigTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiStaticSSNakajimaZwanzigTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSSpectraNakajimaZwanzig.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSCIDNP.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticRPOnlyHSSymDecRedfield.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSStochYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSStochTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectYieldsSymmUncoupled.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectTimeEvoSymmUncoupled.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSStochYieldsSymmUncoupled.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSStochTimeEvoSymmUncoupled.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSDirectYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSDirectTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSStochYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSStochTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSResonanceSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSPowderSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSPowderSpectraNakajimaZwanzig.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskActionSpectrumHistogram.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskActionSpectrumHistogramRPOnlyDec.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSPump.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiRadicalPairSSTimeEvo.cpp
)

# --------------------------------------------------------------------------
# RunSection Actions
# --------------------------------------------------------------------------
set(SOURCES_RUNSECTION_ACTIONS
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionRotateVector.cpp
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionScaleVector.cpp
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionAddVector.cpp
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionAddScalar.cpp
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionMultiplyScalar.cpp
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionFibonacciSphere.cpp
        ${PATH_SOURCE_RUNSECTION}/Actions/ActionLogSpace.cpp
)

# --------------------------------------------------------------------------
# Unit Tests
# --------------------------------------------------------------------------
set(SOURCES_TESTS
        ${PATH_SOURCE_TESTS}/testmain.cpp
)

# --------------------------------------------------------------------------
# Define Targets
# --------------------------------------------------------------------------
# Core functionality libraries
add_library(molspin_core STATIC ${SOURCES_SPINAPI} ${SOURCES_MSDPARSER} ${SOURCES_RUNSECTION} ${SOURCES_RUNSECTION_TASKS} ${SOURCES_RUNSECTION_ACTIONS})
target_include_directories(molspin_core PUBLIC
    ${PATH_SOURCE_SPINAPI}
    ${PATH_SOURCE_MSDPARSER}
    ${PATH_SOURCE_RUNSECTION}
    ${PATH_SOURCE_RUNSECTION}/Tasks
    ${PATH_SOURCE_RUNSECTION}/General/Resonance
    ${PATH_SOURCE_RUNSECTION}/General/HS
    ${PATH_SOURCE_RUNSECTION}/General/SS
    ${PATH_SOURCE_RUNSECTION}/General/MultiSS
    ${PATH_SOURCE_RUNSECTION}/Tasks/Custom
    ${PATH_SOURCE_RUNSECTION}/Actions
)
target_link_libraries(molspin_core PUBLIC
    Armadillo::Armadillo
    OpenMP::OpenMP_CXX
)
molspin_configure_target(molspin_core)

# MolSpin application
add_executable(molspin ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(molspin PRIVATE molspin_core ${CMAKE_DL_LIBS})
molspin_configure_target(molspin)

# Unit tests
add_executable(molspintest ${SOURCES_TESTS})
target_include_directories(molspintest PRIVATE ${PATH_SOURCE_TESTS})
target_link_libraries(molspintest PRIVATE molspin_core)
molspin_configure_target(molspintest)

# Slow physical resonance validation.
#
# This target retains the expensive R2K-D exact-vs-hybrid X/W powder oracles,
# but is deliberately EXCLUDE_FROM_ALL and is not registered with CTest.
# Build and run it explicitly when a physical resonance certificate is needed:
#
#   cmake --build build --target molspinresonanceslowtest
#   ./build/molspinresonanceslowtest
#
# Successful-test output is replayed for this target so the R2K-D normalized
# RMS certificate values remain visible.
add_executable(molspinresonanceslowtest EXCLUDE_FROM_ALL ${SOURCES_TESTS})
target_include_directories(molspinresonanceslowtest PRIVATE ${PATH_SOURCE_TESTS})
target_link_libraries(molspinresonanceslowtest PRIVATE molspin_core)
target_compile_definitions(molspinresonanceslowtest PRIVATE
    SPINAPI_TEST=0
    MSDPARSER_TEST=0
    ACTION_TEST=0
    STATICHSSDECAY_TEST=0
    STATICSS_TEST=0
    MULTISTATICSS_TEST=0
    STATICRPONLY_TEST=0
    STATICSSSPECTRA_TEST=0
    STATICHSRESONANCE_TEST=1
    STATICPOWDERSPECTRA_TEST=0
    UTIL_TEST=0
    HSGENERAL_TEST=0
    SSGENERAL_TEST=0
    MULTISSGENERAL_TEST=0
    GENERALLOG_TEST=0
    GENERALORIENTATION_TEST=0
    GENERALPHYSICS_TEST=0
    GENERALSPECTROSCOPY_TEST=0
    MOLSPIN_SLOW_RESONANCE_ORACLES=1
    MOLSPIN_REPLAY_SUCCESS_OUTPUT=1
)
molspin_configure_target(molspinresonanceslowtest)

enable_testing()
add_test(NAME molspintest COMMAND molspintest)
