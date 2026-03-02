/////////////////////////////////////////////////////////////////////////
// CreateTask method of the RunSection class (RunSection module)
//
// This is where you add new task classes, so that they can be created
// when specified in the input file. Note that this is also where you
// choose the value for the "type" keyword in the input file that will
// be identified with a certain task class.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Include task classes here
#include "TaskStaticSS.h"
#include "TaskStaticHSSymmetricDecay.h"
#include "TaskHamiltonianEigenvalues.h"
#include "TaskStaticRPOnlyHSSymDec.h"
#include "TaskStaticSSTimeEvo.h"
#include "TaskDynamicHSTimeEvo.h"
#include "TaskPeriodicSSTimeEvo.h"
#include "TaskPeriodicHSTimeEvo.h"
#include "TaskGammaCompute.h"
#include "TaskMultiStaticSSTimeEvo.h"
#include "TaskMultiStaticSS.h" // added by Benji 
#include "TaskMultiDynamicHSTimeEvo.h"

#include "TaskMultiStaticSSTimeEvoSpectra.h"

#include "TaskStaticSSRedfield.h"
#include "TaskStaticSSRedfieldSparse.h"
#include "TaskStaticSSRedfieldTimeEvo.h"
#include "TaskStaticSSRedfieldTimeEvoSparse.h"
#include "TaskMultiStaticSSRedfieldTimeEvo.h"
// #include "TaskStaticRPOnlyHSSymDecRedfield.h"
#include "TaskStaticSSNakajimaZwanzig.h"
#include "TaskStaticSSNakajimaZwanzigTimeEvo.h"
#include "TaskMultiStaticSSNakajimaZwanzigTimeEvo.h"

#include "TaskStaticSSSpectra.h"
#include "TaskStaticSSPowderSpectra.h"
#include "TaskStaticSSPowderSpectraNakajimaZwanzig.h"
#include "TaskStaticSSSpectraNakajimaZwanzig.h"
#include "TaskStaticSSCIDNP.h"

#include "TaskStaticHSStochYields.h"
#include "TaskStaticHSStochTimeEvo.h"
#include "TaskStaticHSDirectYields.h"
#include "TaskStaticHSDirectTimeEvo.h"
#include "TaskDynamicHSDirectYields.h"
#include "TaskDynamicHSDirectTimeEvo.h"
#include "TaskDynamicHSStochYields.h"
#include "TaskDynamicHSStochTimeEvo.h"
#include "TaskStaticHSDirectYieldsSymmUncoupled.h"
#include "TaskStaticHSStochYieldsSymmUncoupled.h"
#include "TaskStaticHSDirectTimeEvoSymmUncoupled.h"
#include "TaskStaticHSStochTimeEvoSymmUncoupled.h"
#include "TaskStaticHSDirectSpectra.h"
#include "TaskStaticHSTrEPRSpectra.h"
// #include "TaskDynamicHSDirectSpectra.h"

#include "TaskActionSpectrumHistogram.h"
#include "TaskActionSpectrumHistogramRPOnlyDec.h"

#include "TaskStaticSSPump.h"

#include "TaskMultiRadicalPairSSTimeEvo.h" //added by Benji

/////////////////////////////////////////////////////////////////////////
namespace RunSection
{
	std::shared_ptr<BasicTask> RunSection::CreateTask(const std::string &_tasktype, const MSDParser::ObjectParser &_obj)
	{
		// The task pointer to be assigned and returned
		std::shared_ptr<BasicTask> task = nullptr;

		// Create a task of the proper type
		if (_tasktype.compare("staticss") == 0 || _tasktype.compare("staticivp") == 0)
		{
			task = std::make_shared<TaskStaticSS>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-symmetricdecay") == 0 || _tasktype.compare("statichs") == 0)
		{
			task = std::make_shared<TaskStaticHSSymmetricDecay>(_obj, *this);
		}
		else if (_tasktype.compare("eigenvalues") == 0 || _tasktype.compare("hamiltonianeigenvalues") == 0)
		{
			task = std::make_shared<TaskHamiltonianEigenvalues>(_obj, *this);
		}
		else if (_tasktype.compare("rp-symmetricuncoupled") == 0 || _tasktype.compare("rp-uncoupled") == 0)
		{
			task = std::make_shared<TaskStaticRPOnlyHSSymDec>(_obj, *this);
		}
		else if (_tasktype.compare("staticss-timeevolution") == 0)
		{
			task = std::make_shared<TaskStaticSSTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("dynamichs-timeevolution") == 0)
		{
			task = std::make_shared<TaskDynamicHSTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("periodicss-timeevolution") == 0)
		{
			task = std::make_shared<TaskPeriodicSSTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("periodichs-timeevolution") == 0)
		{
			task = std::make_shared<TaskPeriodicHSTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("gamma-compute") == 0)
		{
			task = std::make_shared<TaskGammaCompute>(_obj, *this);
		}
		else if (_tasktype.compare("multistaticss-timeevolution") == 0 || _tasktype.compare("staticss-multisystem") == 0)
		{
			task = std::make_shared<TaskMultiStaticSSTimeEvo>(_obj, *this);
		}
		// NEW (Added by Benji Tigg): is a direct calculation of the yield via the laplacian rather than time-evolution 
		else if (_tasktype.compare("multistaticss") == 0 || _tasktype.compare("MultiStaticSS") == 0 )
		{
			task = std::make_shared<TaskMultiStaticSS>(_obj, *this);
		}
		else if (_tasktype.compare("multidynamichs-timeevolution") == 0 || _tasktype.compare("dynamichs-multisystem") == 0)
		{
			task = std::make_shared<TaskMultiDynamicHSTimeEvo>(_obj, *this);
		}

		// NEW (Added by Luca Gerhards): Including BWR-Relaxation Theory Task Class [TaskStaticSSRedfield.cpp] as CreateTask member.

		else if (_tasktype.compare("redfield-relaxation") == 0 || _tasktype.compare("Redfield-Relaxation") == 0)
		{
			task = std::make_shared<TaskStaticSSRedfield>(_obj, *this);
		}
		else if (_tasktype.compare("redfield-relaxation-sparse") == 0 || _tasktype.compare("Redfield-Relaxation-Sparse") == 0)
		{
			task = std::make_shared<TaskStaticSSRedfieldSparse>(_obj, *this);
		}
		else if (_tasktype.compare("redfield-relaxation-timeevolution") == 0 || _tasktype.compare("Redfield-Relaxation-Timeevolution") == 0)
		{
			task = std::make_shared<TaskStaticSSRedfieldTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("redfield-relaxation-timeevolution-sparse") == 0 || _tasktype.compare("Redfield-Relaxation-Timeevolution-Sparse") == 0)
		{
			task = std::make_shared<TaskStaticSSRedfieldTimeEvoSparse>(_obj, *this);
		}
		else if (_tasktype.compare("multistaticss-redfield-timeevolution") == 0 || _tasktype.compare("redfield-multisystem") == 0)
		{
			task = std::make_shared<TaskMultiStaticSSRedfieldTimeEvo>(_obj, *this);
		}
		// else if(_tasktype.compare("RP-Symmetricuncoupled-Redfield") ==0 || _tasktype.compare("rp-symmetricuncoupled-redfield") == 0) {task = std::make_shared<TaskStaticRPOnlyHSSymDecRedfield>(_obj, *this);}

		// NEW (Added by Luca Gerhards): Including NZ-Relaxation Theory Task Class [TaskStaticSSNakajimaZwanzig.cpp] as CreateTask member.
		else if (_tasktype.compare("nakajimazwanzig-relaxation") == 0 || _tasktype.compare("NakajimaZwanzig-Relaxation") == 0)
		{
			task = std::make_shared<TaskStaticSSNakajimaZwanzig>(_obj, *this);
		}
		else if (_tasktype.compare("nakajimazwanzig-relaxation-timeevolution") == 0 || _tasktype.compare("NakajimaZwanzig-Relaxation-Timeevolution") == 0)
		{
			task = std::make_shared<TaskStaticSSNakajimaZwanzigTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("multistaticss-nakajimazwanzig-timeevolution") == 0 || _tasktype.compare("nakajimazwanzig-multisystem") == 0)
		{
			task = std::make_shared<TaskMultiStaticSSNakajimaZwanzigTimeEvo>(_obj, *this);
		}

		// NEW (ADDED by Luca Gerhards): Including spectroscopy task such as CIDNP
		else if (_tasktype.compare("staticss-cidnp") == 0 || _tasktype.compare("StaticSS-CIDNP") == 0)
		{
			task = std::make_shared<TaskStaticSSCIDNP>(_obj, *this);
		}

		// NEW (Added by Irina Anisimova and Luca Gehards):Spectroscopy module
		else if (_tasktype.compare("staticss-spectra") == 0 || _tasktype.compare("StaticSS-Spectra") == 0)
		{
			task = std::make_shared<TaskStaticSSSpectra>(_obj, *this);
		}

		// NEW (Added by Irina Anisimova and Luca Gehards):Spectroscopy module
		else if (_tasktype.compare("staticss-powderspectra") == 0 || _tasktype.compare("StaticSS-Powderspectra") == 0)
		{
			task = std::make_shared<TaskStaticSSPowderSpectra>(_obj, *this);
		}
		else if (_tasktype.compare("staticss-powderspectra-nakajimazwanzig") == 0 || _tasktype.compare("StaticSS-Powderspectra-Nakajimazwanzig") == 0)
		{
			task = std::make_shared<TaskStaticSSPowderSpectraNakajimaZwanzig>(_obj, *this);
		}


		// NEW (Added by Luca Gerhards):Spectroscopy module with spin relaxation through Nakahima-Zwanzig equation
		else if (_tasktype.compare("staticss-spectra-nakajimazwanzig") == 0 || _tasktype.compare("StaticSS-Spectra-Nakajimazwanzig") == 0)
		{
			task = std::make_shared<TaskStaticSSSpectraNakajimaZwanzig>(_obj, *this);
		}

		// NEW (Added by Luca Gerhards): Including spectroscopy multi-spin system task
		else if (_tasktype.compare("multistaticss-timeevolution-spectra") == 0 || _tasktype.compare("staticss-multisystem-spectra") == 0)
		{
			task = std::make_shared<TaskMultiStaticSSTimeEvoSpectra>(_obj, *this);
		}

		// NEW (Added by Gediminas Pazera and Luca Gerhards): Including SSE theory as CreateTask member.
		else if (_tasktype.compare("statichs-stoch-yields") == 0 || _tasktype.compare("StaticHS-Stoch-Yields") == 0)
		{
			task = std::make_shared<TaskStaticHSStochYields>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-stoch-timeevo") == 0 || _tasktype.compare("StaticHS-Stoch-TimeEvo") == 0)
		{
			task = std::make_shared<TaskStaticHSStochTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-direct-yields") == 0 || _tasktype.compare("StaticHS-Direct-Yields") == 0)
		{
			task = std::make_shared<TaskStaticHSDirectYields>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-direct-timeevo") == 0 || _tasktype.compare("StaticHS-Direct-TimeEvo") == 0)
		{
			task = std::make_shared<TaskStaticHSDirectTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("dynamichs-direct-yields") == 0 || _tasktype.compare("DynamicHS-Direct-Yields") == 0)
		{
			task = std::make_shared<TaskDynamicHSDirectYields>(_obj, *this);
		}
		else if (_tasktype.compare("dynamichs-direct-timeevo") == 0 || _tasktype.compare("DynamicHS-Direct-TimeEvo") == 0)
		{
			task = std::make_shared<TaskDynamicHSDirectTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("dynamichs-stoch-yields") == 0 || _tasktype.compare("DynamicHS-Stoch-Yields") == 0)
		{
			task = std::make_shared<TaskDynamicHSStochYields>(_obj, *this);
		}
		else if (_tasktype.compare("dynamichs-stoch-timeevo") == 0 || _tasktype.compare("DynamicHS-Stoch-TimeEvo") == 0)
		{
			task = std::make_shared<TaskDynamicHSStochTimeEvo>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-direct-yields-symm-uncoupled") == 0 || _tasktype.compare("StaticHS-Direct-Yields-Symm-Uncoupled") == 0)
		{
			task = std::make_shared<TaskStaticHSDirectYieldsSymmUncoupled>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-direct-timeevo-symm-uncoupled") == 0 || _tasktype.compare("StaticHS-Direct-TimeEvo-Symm-Uncoupled") == 0)
		{
			task = std::make_shared<TaskStaticHSDirectTimeEvoSymmUncoupled>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-stoch-yields-symm-uncoupled") == 0 || _tasktype.compare("StaticHS-Stoch-Yields-Symm-Uncoupled") == 0)
		{
			task = std::make_shared<TaskStaticHSStochYieldsSymmUncoupled>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-stoch-timeevo-symm-uncoupled") == 0 || _tasktype.compare("StaticHS-Stoch-TimeEvo-Symm-Uncoupled") == 0)
		{
			task = std::make_shared<TaskStaticHSStochTimeEvoSymmUncoupled>(_obj, *this);
		}

		// NEW (ADDED by Luca Gerhards): Spectroscopy task in Hilbert space
		else if (_tasktype.compare("statichs-direct-spectra") == 0 || _tasktype.compare("StaticHS-Direct-Yields") == 0)
		{
			task = std::make_shared<TaskStaticHSDirectSpectra>(_obj, *this);
		}
		else if (_tasktype.compare("statichs-trepr-spectra") == 0 || _tasktype.compare("StaticHS-TrEPR-Spectra") == 0)
		{
			task = std::make_shared<TaskStaticHSTrEPRSpectra>(_obj, *this);
		}

		// NEW (ADDED by Luca Gerhards): Including Action Histrograms in the context of Hamish Hiscock
		else if (_tasktype.compare("actionspectrumhistogram") == 0 || _tasktype.compare("ActionSpectrumHistogram") == 0)
		{
			task = std::make_shared<TaskActionSpectrumHistogram>(_obj, *this);
		}
		// Still buggy - do not use yet.
		else if (_tasktype.compare("actionspectrumhistogramrponlydec") == 0 || _tasktype.compare("ActionSpectrumHistogramRPOnlyDec") == 0)
		{
			task = std::make_shared<TaskActionSpectrumHistogramRPOnlyDec>(_obj, *this);
		}

		// NEW (ADDED by Oliver Russell): Including Pumpprobe time evolution
		else if (_tasktype.compare("staticss-pump") == 0)
		{
			task = std::make_shared<TaskStaticSSPump>(_obj, *this);
		}

		//Being removed due to task cleanup - Benji
		// NEW (ADDED by Benji Tigg)
		else if (_tasktype.compare("multiradicalpairss-timeevolution") == 0)
		{
			task = std::make_shared<TaskMultiRadicalPairSSTimeEvo>(_obj, *this);
		}

		// NOTE: To add a new task class, just add another "else if" here...
		// The string used in the "compare" method is the "type" to be specified in the MolSpin input file

		// Return the task instance pointer - or nullptr if the type was not recognized
		return task;
	}
}
/////////////////////////////////////////////////////////////////////////
