# Set cmake version requirement
cmake_minimum_required(VERSION 3.16)

project(MolSpin LANGUAGES CXX)

include(CTest)

option(MOLSPIN_ENABLE_NATIVE_OPTIMIZATIONS "Enable non-portable CPU-specific optimizations." OFF)
option(MOLSPIN_ENABLE_UNROLL_LOOPS "Enable loop unrolling compiler hints when supported." ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Armadillo REQUIRED)
find_package(OpenMP REQUIRED)

find_library(MOLSPIN_OPENBLAS_LIBRARY
    NAMES openblas libopenblas
)

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

target_include_directories(molspin_core
    PUBLIC
        ${PATH_SOURCE_SPINAPI}
        ${PATH_SOURCE_MSDPARSER}
        ${PATH_SOURCE_RUNSECTION}
        ${PATH_SOURCE_TESTS}
        "${PATH_SOURCE_RUNSECTION}/Tasks"
        "${PATH_SOURCE_RUNSECTION}/Tasks/Custom"
        "${PATH_SOURCE_RUNSECTION}/Actions"
)

if (MSVC)
    target_compile_options(molspin_core PRIVATE /O2 /W4 /EHsc /bigobj)
    target_compile_definitions(molspin_core PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX _USE_MATH_DEFINES ARMA_DONT_PRINT_FAST_MATH_WARNING)
else()
    target_compile_options(molspin_core PRIVATE -O3 -Wall)
    target_compile_definitions(molspin_core PRIVATE ARMA_DONT_PRINT_FAST_MATH_WARNING)

    if (MOLSPIN_ENABLE_NATIVE_OPTIMIZATIONS)
        target_compile_options(molspin_core PRIVATE -march=native)
    endif()

    if (MOLSPIN_ENABLE_UNROLL_LOOPS)
        target_compile_options(molspin_core PRIVATE -funroll-loops)
    endif()
endif()

if (TARGET Armadillo::Armadillo)
    target_link_libraries(molspin_core PUBLIC Armadillo::Armadillo)
elseif (TARGET Armadillo::armadillo)
    target_link_libraries(molspin_core PUBLIC Armadillo::armadillo)
elseif (TARGET armadillo)
    target_link_libraries(molspin_core PUBLIC armadillo)
else()
    target_include_directories(molspin_core PUBLIC ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(molspin_core PUBLIC ${ARMADILLO_LIBRARIES})
endif()

target_link_libraries(molspin_core PUBLIC OpenMP::OpenMP_CXX)

if (MOLSPIN_OPENBLAS_LIBRARY)
    target_link_libraries(molspin_core PUBLIC ${MOLSPIN_OPENBLAS_LIBRARY})
    target_compile_definitions(molspin_core PUBLIC MOLSPIN_HAS_OPENBLAS=1)
endif()

# MolSpin application
add_executable(molspin ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_link_libraries(molspin PRIVATE molspin_core)

# Keep lookup of bundled runtime libraries relative to the executable.
if (APPLE)
    set_target_properties(molspin PROPERTIES
        BUILD_RPATH "@loader_path/lib"
        INSTALL_RPATH "@loader_path/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
elseif (UNIX)
    set_target_properties(molspin PROPERTIES
        BUILD_RPATH "\$ORIGIN/lib"
        INSTALL_RPATH "\$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Unit tests
add_executable(molspintest ${SOURCES_TESTS})
target_link_libraries(molspintest PRIVATE molspin_core)
add_test(NAME molspintest COMMAND molspintest)
