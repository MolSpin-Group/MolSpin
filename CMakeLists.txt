# Set cmake version requirement
cmake_minimum_required(VERSION 3.14)

project(MolSpin)

option(LINK_OPENBLAS "Whether OpenBLAS should be linked." ON)
option(LINK_LAPACK "Whether LAPACK should be linked." ON)
option(ENABLE_OPENMP "Whether OpenMP should be available from the MolSpin code." ON)
option(ENABLE_NATIVE_ARCH "Compile with -march=native when supported." ON)
option(ENABLE_LOOP_UNROLLING "Compile with -funroll-loops when supported." ON)
option(ENABLE_GNU_CONCEPTS_TS "Compile with GCC -fconcepts for C++17 abbreviated-template compatibility." ON)

# Search for OpenMP
if (ENABLE_OPENMP)
find_package(OpenMP)
if (OpenMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
endif()

find_package(Armadillo)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Compiler options
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -g -Wall -DARMA_DONT_PRINT_FAST_MATH_WARNING)
if (ENABLE_NATIVE_ARCH)
    add_compile_options(-march=native)
endif()
if (ENABLE_LOOP_UNROLLING)
    add_compile_options(-funroll-loops)
endif()
if (ENABLE_GNU_CONCEPTS_TS AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fconcepts)
endif()

# Paths
set(PATH_SOURCE_SPINAPI "${CMAKE_CURRENT_SOURCE_DIR}/SpinAPI")
set(PATH_SOURCE_MSDPARSER "${CMAKE_CURRENT_SOURCE_DIR}/MSDParser")
set(PATH_SOURCE_RUNSECTION "${CMAKE_CURRENT_SOURCE_DIR}/RunSection")
set(PATH_SOURCE_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/Tests")

# Include directories
include_directories(${PATH_SOURCE_SPINAPI} ${PATH_SOURCE_MSDPARSER} ${PATH_SOURCE_RUNSECTION} ${PATH_SOURCE_TESTS})
include_directories("${PATH_SOURCE_RUNSECTION}/Tasks" "${PATH_SOURCE_RUNSECTION}/Tasks/Custom" "${PATH_SOURCE_RUNSECTION}/Actions")

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
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSTrEPRSpectra.cpp
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

# MolSpin application
add_executable(molspin ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(molspin molspin_core ${ARMADILLO_LIBRARIES} -lopenblas -llapack -fopenmp)

# Unit tests
add_executable(molspintest ${SOURCES_TESTS})
target_link_libraries(molspintest molspin_core ${ARMADILLO_LIBRARIES} -lopenblas -llapack -fopenmp)

enable_testing()
add_test(NAME molspintest COMMAND molspintest)
