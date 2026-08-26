/////////////////////////////////////////////////////////////////////////
// ResonanceLineshape (RunSection::General::Resonance)
// ------------------
// Normalized field-domain Gaussian/Lorentzian line profiles.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_Resonance_ResonanceLineshape
#define MOD_RunSection_General_Resonance_ResonanceLineshape

#include "ResonanceTypes.h"

namespace RunSection::General::Resonance
{
    class ResonanceLineshape
    {
    public:
        static double Evaluate(Lineshape _kind, double _delta_mT, double _fwhm_mT);
    };
}

#endif
