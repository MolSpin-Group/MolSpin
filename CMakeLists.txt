# Set cmake version requirement
cmake_minimum_required(VERSION 3.14)

project(MolSpin)

# Search for OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Armadillo)
include_directories(${ARMADILLO_INCLUDE_DIRS})

option(LINK_OPENBLAS "Whether OpenBLAS should be linked." ON)
option(LINK_LAPACK "Whether LAPACK should be linked." ON)
option(ENABLE_OPENMP "Whether OpenMP should be available from the MolSpin code." ON)

# Compiler options
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3 -g -Wall -march=native -funroll-loops -fopenmp -DARMA_DONT_PRINT_FAST_MATH_WARNING)

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
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiDynamicHSTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfield.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfieldSparse.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfieldTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSRedfieldTimeEvoSparse.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskMultiStaticSSRedfieldTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSCIDNP.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticRPOnlyHSSymDecRedfield.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSStochYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSStochTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticHSDirectTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSDirectYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSDirectTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSStochYields.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskDynamicHSStochTimeEvo.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSPowderSpectra.cpp
        ${PATH_SOURCE_RUNSECTION}/Tasks/TaskStaticSSPowderSpectraNakajimaZwanzig.cpp
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
