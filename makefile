# --------------------------------------------------------------------------
# Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
# (c) 2019 Quantum Biology and Computational Physics Group.
# See LICENSE.txt for license information.
# ----
# Molspin requires the Armadillo C++ library version 8.5 or newer to be
# installed. You should install OpenBLAS, Intel MKL, or other math libraries
# before installing Armadillo, please see documentation for Armadillo.
#
# MolSpin was developed using gcc 5.4.0.
# --------------------------------------------------------------------------
# Armadillo's wrapper selects and links its configured BLAS/LAPACK backend.
# pkg-config also supplies non-standard include/library paths, as used by
# Conda. Both variables can be overridden for a custom installation.
PKG_CONFIG ?= pkg-config
ARMADILLO_CFLAGS ?= $(shell $(PKG_CONFIG) --cflags armadillo 2>/dev/null)
ARMADILLO_LIBS ?= $(shell $(PKG_CONFIG) --libs armadillo 2>/dev/null || echo -larmadillo)
# Example:
# make ARMADILLO_CFLAGS="-I/path/include" ARMADILLO_LIBS="-L/path/lib -larmadillo"
# --------------------------------------------------------------------------
# If you have different versions of gcc or the C++ stdlib installed,
# adding the following to LFLAGS may help:
#LSTATICLIBS = -static-libstdc++ -static-libgcc
# --------------------------------------------------------------------------
# SpinAPI module
PATH_SPINAPI = ./SpinAPI

OBJS_SPINAPI = $(PATH_SPINAPI)/SpinSystem.o $(PATH_SPINAPI)/Spin.o $(PATH_SPINAPI)/Interaction.o $(PATH_SPINAPI)/Transition.o $(PATH_SPINAPI)/Operator.o $(PATH_SPINAPI)/Pulse.o $(PATH_SPINAPI)/State.o $(PATH_SPINAPI)/SpinSpace.o $(PATH_SPINAPI)/StandardOutput.o $(PATH_SPINAPI)/Tensor.o $(PATH_SPINAPI)/Trajectory.o $(PATH_SPINAPI)/SubSystem.o $(PATH_SPINAPI)/Function.o $(PATH_SPINAPI)/PulseSequence.o $(PATH_SPINAPI)/PowderGrid.o
DEP_SPINAPI = 

# --------------------------------------------------------------------------
# MSD-Parser module
PATH_MSDPARSER = ./MSDParser
OBJS_MSDPARSER = $(PATH_MSDPARSER)/MSDParser.o $(PATH_MSDPARSER)/FileReader.o $(PATH_MSDPARSER)/ObjectParser.o
DEP_MSDPARSER = $(PATH_MSDPARSER)/MSDParser.h
# --------------------------------------------------------------------------
# RunSection module
PATH_RUNSECTION = ./RunSection
OBJS_RUNSECTION = $(PATH_RUNSECTION)/RunSection.o $(PATH_RUNSECTION)/BasicTask.o $(PATH_RUNSECTION)/Action.o $(PATH_RUNSECTION)/Settings.o $(PATH_RUNSECTION)/OutputHandler.o $(PATH_RUNSECTION)/Utility.o $(PATH_RUNSECTION)/CubicSpline.o
DEP_RUNSECTION = $(PATH_RUNSECTION)/RunSection.h

# ---
# Unified Hilbert-space production architecture
PATH_RUNSECTION_GENERAL_HS = ./RunSection/General/HS
OBJS_RUNSECTION_GENERAL_HS = $(PATH_RUNSECTION_GENERAL_HS)/HSExecutionPlan.o $(PATH_RUNSECTION_GENERAL_HS)/HSStatePreparation.o $(PATH_RUNSECTION_GENERAL_HS)/HSOrientationSampler.o $(PATH_RUNSECTION_GENERAL_HS)/HSHamiltonianBuilder.o $(PATH_RUNSECTION_GENERAL_HS)/HSReactionRelaxation.o $(PATH_RUNSECTION_GENERAL_HS)/HSPropagator.o $(PATH_RUNSECTION_GENERAL_HS)/HSObservableCollector.o $(PATH_RUNSECTION_GENERAL_HS)/TaskHSGeneral.o
# ---
# RunSection custom tasks
PATH_RUNSECTION_CUSTOMTASKS = ./RunSection/Tasks/Custom
OBJS_RUNSECTION_CUSTOMTASKS =
DEP_RUNSECTION_CUSTOMTASKS =
# ---
# RunSection tasks
PATH_RUNSECTION_TASKS = ./RunSection/Tasks
OBJS_RUNSECTION_TASKS = $(PATH_RUNSECTION_TASKS)/TaskStaticSS.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSSymmetricDecay.o $(PATH_RUNSECTION_TASKS)/TaskHamiltonianEigenvalues.o $(PATH_RUNSECTION_TASKS)/TaskStaticRPOnlyHSSymDec.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskDynamicHSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskPeriodicSSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskPeriodicHSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskGammaCompute.o $(PATH_RUNSECTION_TASKS)/TaskMultiStaticSSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskMultiDynamicHSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSRedfield.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSRedfieldSparse.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSRedfieldTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSRedfieldTimeEvoSparse.o $(PATH_RUNSECTION_TASKS)/TaskMultiStaticSSRedfieldTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSSpectra.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSCIDNP.o $(PATH_RUNSECTION_TASKS)/TaskStaticRPOnlyHSSymDecRedfield.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSStochYields.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSStochTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSDirectYields.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSDirectTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskDynamicHSDirectYields.o $(PATH_RUNSECTION_TASKS)/TaskDynamicHSDirectTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskDynamicHSStochYields.o $(PATH_RUNSECTION_TASKS)/TaskDynamicHSStochTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSDirectYieldsSymmUncoupled.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSDirectTimeEvoSymmUncoupled.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSStochYieldsSymmUncoupled.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSStochTimeEvoSymmUncoupled.o $(PATH_RUNSECTION_TASKS)/TaskActionSpectrumHistogram.o $(PATH_RUNSECTION_TASKS)/TaskActionSpectrumHistogramRPOnlyDec.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSPump.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSNakajimaZwanzigTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSNakajimaZwanzig.o $(PATH_RUNSECTION_TASKS)/TaskMultiStaticSSTimeEvoSpectra.o $(PATH_RUNSECTION_TASKS)/TaskMultiStaticSSNakajimaZwanzigTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskMultiRadicalPairSSTimeEvo.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSSpectraNakajimaZwanzig.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSDirectSpectra.o $(PATH_RUNSECTION_TASKS)/TaskStaticHSResonanceSpectra.o $(PATH_RUNSECTION_TASKS)/TaskMultiStaticSS.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSPowderSpectra.o $(PATH_RUNSECTION_TASKS)/TaskStaticSSPowderSpectraNakajimaZwanzig.o $(OBJS_RUNSECTION_CUSTOMTASKS)
DEP_RUNSECTION_TASKS = $(DEP_RUNSECTION_CUSTOMTASKS)
# ---
# RunSection actions
PATH_RUNSECTION_ACTIONS = ./RunSection/Actions
OBJS_RUNSECTION_ACTIONS = $(PATH_RUNSECTION_ACTIONS)/ActionRotateVector.o $(PATH_RUNSECTION_ACTIONS)/ActionScaleVector.o $(PATH_RUNSECTION_ACTIONS)/ActionAddVector.o $(PATH_RUNSECTION_ACTIONS)/ActionAddScalar.o $(PATH_RUNSECTION_ACTIONS)/ActionMultiplyScalar.o $(PATH_RUNSECTION_ACTIONS)/ActionFibonacciSphere.o $(PATH_RUNSECTION_ACTIONS)/ActionLogSpace.o
DEP_RUNSECTION_ACTIONS =
# --------------------------------------------------------------------------
# Unit testing module
PATH_TESTS = ./Tests
OBJS_TEST_COMMON = $(OBJS_SPINAPI) $(OBJS_MSDPARSER) $(OBJS_RUNSECTION) $(OBJS_RUNSECTION_GENERAL_HS) $(OBJS_RUNSECTION_TASKS) $(OBJS_RUNSECTION_ACTIONS)
OBJS_TESTS = $(PATH_TESTS)/testmain.o $(OBJS_TEST_COMMON)
OBJS_TESTS_DEBUG = $(PATH_TESTS)/testmain_debug.o $(OBJS_TEST_COMMON)
DEP_TESTS =
# --------------------------------------------------------------------------
#LinearAlgebra Vendor code
PATH_LINALG_VENDOR = ./Vendor/
#---------------------------------------------------------------------------
# General Compilation Options
OBJECTS = main.o $(OBJS_SPINAPI) $(OBJS_MSDPARSER) $(OBJS_RUNSECTION) $(OBJS_RUNSECTION_GENERAL_HS) $(OBJS_RUNSECTION_TASKS) $(OBJS_RUNSECTION_ACTIONS)
CXX ?= g++
CXXSTD ?= -std=c++17
OPTFLAGS ?= -O3
ARCHFLAGS ?= -march=native
LOOPFLAGS ?= -funroll-loops
OPENMPFLAGS ?= -fopenmp
DLFLAGS ?= -ldl
COMPATFLAGS ?=
WARNFLAGS ?= -Wall
DEBUGFLAGS ?= -g
DEFINES ?= -DARMA_DONT_PRINT_FAST_MATH_WARNING -DARMA_NO_DEBUG -DASSERT=1 -DNEGATIVERATES=0
TESTDEFINES ?= -DSPINAPI_TEST=0 -DMSDPARSER_TEST=0 -DACTION_TEST=0 -DSTATICHSSDECAY_TEST=0 -DSTATICSS_TEST=0 -DSTATICRPONLY_TEST=0 -DSTATICSSSPECTRA_TEST=0 -DSTATICHSRESONANCE_TEST=0 -DSTATICPOWDERSPECTRA_TEST=1 -DHSGENERAL_TEST=1 -DUTIL_TEST=0
CC = $(CXX) $(CXXSTD)		# Compiler to use
LFLAGS = $(WARNFLAGS) $(DEBUGFLAGS) -DARMA_DONT_PRINT_FAST_MATH_WARNING $(OPTFLAGS)	# Linker Flags
CFLAGS = $(WARNFLAGS) -c $(ARCHFLAGS) $(LOOPFLAGS) $(COMPATFLAGS) $(DEBUGFLAGS) $(OPENMPFLAGS) $(DEFINES) $(OPTFLAGS) # Compile flags to .o
TESTCFLAGS = $(CFLAGS) $(TESTDEFINES)
LDLIBS = $(ARMADILLO_LIBS) $(OPENMPFLAGS) $(DLFLAGS)
# Example portability override: make ARCHFLAGS= LOOPFLAGS=

#DEBUGLFLAGS = -Wall -g -DARMA_DONT_PRINT_FAST_MATH_WARNING
#DEBUGCFLAGS = -Wall -c -march=native -funroll-loops -g -fopenmp -DARMA_DONT_PRINT_FAST_MATH_WARNING -Werror -Wextra

# --------------------------------------------------------------------------
# Compilation of the main program
# --------------------------------------------------------------------------
SEARCHDIR_MOLSPIN = -I$(PATH_SPINAPI) -I$(PATH_MSDPARSER) -I$(PATH_RUNSECTION) -I$(PATH_RUNSECTION_GENERAL_HS) -I$(PATH_RUNSECTION_TASKS) -I$(PATH_RUNSECTION_CUSTOMTASKS) -I$(PATH_RUNSECTION_ACTIONS) -I$(PATH_LINALG_VENDOR) $(ARMADILLO_CFLAGS)
molspin: $(OBJECTS)
	$(CC) $(LFLAGS) $^ $(LDLIBS) -o $@

SEARCHDIR_MAIN = -I$(PATH_SPINAPI) -I$(PATH_MSDPARSER) -I$(PATH_RUNSECTION) -I$(PATH_RUNSECTION_GENERAL_HS) -I$(PATH_RUNSECTION_TASKS) -I$(PATH_RUNSECTION_CUSTOMTASKS) -I$(PATH_RUNSECTION_ACTIONS) -I$(PATH_LINALG_VENDOR) $(ARMADILLO_CFLAGS)
main.o: main.cpp $(DEP_MSDPARSER) $(DEP_SPINAPI)
	$(CC) $(CFLAGS) $(SEARCHDIR_MAIN) main.cpp -o main.o
#---------------------------------------------------------------------------
# Debug Complimation of the main program
#---------------------------------------------------------------------------
#DEBUGEXE = debug/molspin
#DEBUGOBJECTS = $(addprefix debug/,$(OBJECTS))
#$(DEBUGEXE): $(OBJECTS)
#	$(DEBUGCC) $(DEBUGLFLAGS) $^ $(SEARCHDIR_MOLSPIN) -o $@
#
#
#debug/main.o: main.cpp $(DEP_MSDPARSER) $(DEP_SPINAPI)
#	$(DEBUGCC) $(DEBUGCFLAGS) $(SEARCHDIR_MAIN) main.cpp -o debug/main.o
# --------------------------------------------------------------------------
# Specific compilation rules
# --------------------------------------------------------------------------
# Make sure that changes to RunSection_CreateTask.cpp triggers recompilation of the RunSection class
$(PATH_RUNSECTION)/RunSection.o: $(PATH_RUNSECTION)/RunSection.cpp $(PATH_RUNSECTION)/RunSection_CreateTask.cpp $(PATH_RUNSECTION)/RunSection.h
	$(CC) $(CFLAGS) $(SEARCHDIR_MOLSPIN) $(PATH_RUNSECTION)/RunSection.cpp -o $(PATH_RUNSECTION)/RunSection.o

# The SpinSpace class has been split into multiple source files due to its complexity
$(PATH_SPINAPI)/SpinSpace.o: $(PATH_SPINAPI)/SpinSpace.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_management.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_states.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_operators.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_hamiltonians.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_pulses.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_transitions.cpp $(PATH_SPINAPI)/SpinSpace/SpinSpace_relaxation.cpp $(PATH_SPINAPI)/SpinSpace.h
	$(CC) $(CFLAGS) $(SEARCHDIR_MOLSPIN) $(PATH_SPINAPI)/SpinSpace.cpp -o $(PATH_SPINAPI)/SpinSpace.o

# --------------------------------------------------------------------------
# General compilation rule
# --------------------------------------------------------------------------
%.o: %.cpp %.h
	$(CC) $(CFLAGS) $(SEARCHDIR_MOLSPIN) $< -o $@
# --------------------------------------------------------------------------
# Unit testing module
# --------------------------------------------------------------------------
SEARCHDIR_TESTS = $(SEARCHDIR_MAIN) -I$(PATH_TESTS)
TESTMAIN_DEPS = $(PATH_TESTS)/testmain.cpp $(PATH_TESTS)/assertfunctions.cpp $(PATH_TESTS)/tests_spinapi.cpp $(PATH_TESTS)/tests_msdparser.cpp $(PATH_TESTS)/tests_actions.cpp $(PATH_TESTS)/tests_TaskStaticHSSymmetricDecay.cpp $(PATH_TESTS)/tests_TaskStaticSS.cpp $(PATH_TESTS)/tests_TaskMultiStaticSS.cpp $(PATH_TESTS)/tests_TaskStaticRPOnlyHSSymDec.cpp $(PATH_TESTS)/tests_TaskStaticSSSpectra.cpp $(PATH_TESTS)/tests_TaskStaticHSResonanceSpectra.cpp $(PATH_TESTS)/tests_TaskStaticPowderSpectra.cpp $(PATH_TESTS)/tests_HSGeneral.cpp $(PATH_TESTS)/tests_utility.cpp

.PHONY: test test_debug
# The normal target always runs the complete suite.
test: $(OBJS_TESTS)
	$(CC) $(LFLAGS) $(OBJS_TESTS) $(LDLIBS) -o $(PATH_TESTS)/molspintest
	$(PATH_TESTS)/molspintest

$(PATH_TESTS)/testmain.o: $(TESTMAIN_DEPS)
	$(CC) $(CFLAGS) $(SEARCHDIR_TESTS) $(PATH_TESTS)/testmain.cpp -o $(PATH_TESTS)/testmain.o

# The focused target uses TESTDEFINES and a separate object/executable so its
# selection cannot leak into a subsequent full-suite build.
test_debug: $(OBJS_TESTS_DEBUG)
	$(CC) $(LFLAGS) $(OBJS_TESTS_DEBUG) $(LDLIBS) -o $(PATH_TESTS)/molspintest-debug
	$(PATH_TESTS)/molspintest-debug

$(PATH_TESTS)/testmain_debug.o: $(TESTMAIN_DEPS)
	$(CC) $(TESTCFLAGS) $(SEARCHDIR_TESTS) $(PATH_TESTS)/testmain.cpp -o $(PATH_TESTS)/testmain_debug.o

# --------------------------------------------------------------------------
# Misc tasks
# --------------------------------------------------------------------------
# Clean-up binaries for clean recompilation
.PHONY: clean
clean:
	rm -f *.o $(PATH_MSDPARSER)/*.o $(PATH_SPINAPI)/*.o $(PATH_RUNSECTION)/*.o $(PATH_RUNSECTION_ACTIONS)/*.o $(PATH_RUNSECTION_TASKS)/*.o $(PATH_RUNSECTION_CUSTOMTASKS)/*.o molspin $(PATH_TESTS)/*.o $(PATH_TESTS)/molspintest $(PATH_TESTS)/molspintest-debug
#	rm debug/*.o

# Clean-up testing binaries and run the test again
.PHONY: cleantest
cleantest:
	rm -f $(PATH_TESTS)/testmain.o $(PATH_TESTS)/testmain_debug.o $(PATH_TESTS)/molspintest $(PATH_TESTS)/molspintest-debug
