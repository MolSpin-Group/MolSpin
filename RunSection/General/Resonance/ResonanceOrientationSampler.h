/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Shared resonance orientation ensemble and SOPHE projection metadata.
#ifndef MOLSPIN_RESONANCEORIENTATIONSAMPLER_H
#define MOLSPIN_RESONANCEORIENTATIONSAMPLER_H
#include "ResonanceExecutionPlan.h"
#include "SpinAPIfwd.h"
#include "PowderGrid.h"

namespace RunSection::General::Resonance
{

    struct ResonanceOrientationGrid
    {
        SpinAPI::PowderGrid grid;
        SpinAPI::SopheGridParameters sophe;
        bool usesSophe = false, hasSopheParameters = false;
        int gridSize = 0, gammaPoints = 1;
        double gammaWeight = 1.;
    };
    namespace OrientationSampling
    {
        struct SymmetryFlags
        {
            bool allIsotropic = true;
            bool allDiag = true;
            bool allAxialZ = true;
            bool anyTensor = false;
        };
        void UpdateSymmetryFlags(const arma::mat &M, SymmetryFlags &flags, bool fullTensorRotation,
                                 double relTol);
        std::string AutoDetectSopheSymmetry(const SpinAPI::system_ptr &system,
                                            const SpinAPI::interaction_ptr &fieldInteraction,
                                            const std::vector<std::string> &h0list, bool fullTensorRotation);
        arma::mat PassiveZYZRotation(const arma::vec &angles);
        bool Rotation(double alpha, double beta, double gamma, arma::mat &rotation);
        bool Build(const ResonanceExecutionPlan &plan, const SpinAPI::system_ptr &system,
                   const SpinAPI::interaction_ptr &field, const std::vector<std::string> &hamiltonian,
                   ResonanceOrientationGrid &out, std::ostream &log, std::string &error);
    } // namespace OrientationSampling

} // namespace RunSection::General::Resonance
#endif
