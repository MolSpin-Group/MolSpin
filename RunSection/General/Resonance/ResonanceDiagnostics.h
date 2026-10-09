/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
// Optional orientation diagnostics, independent of task/output lifecycle.
#ifndef MOLSPIN_RESONANCEDIAGNOSTICS_H
#define MOLSPIN_RESONANCEDIAGNOSTICS_H
#include <limits>
#include <string>
#include <vector>
#include "MSDParserfwd.h"

namespace RunSection::General::Resonance
{
    struct OrientationDiagnostics
    {
        bool enabled = false;
        double fieldMinT = -std::numeric_limits<double>::infinity();
        double fieldMaxT = std::numeric_limits<double>::infinity();
        int maxOrientations = 0;
        std::string file;

        static OrientationDiagnostics FromProperties(const MSDParser::ObjectParser &_props);
        void InitialiseFile(const std::string &_systemName, const std::vector<std::string> &_spinNames) const;
        bool ShouldRecord(size_t _gridIndex, double _resonanceFieldT) const;
    };
} // namespace RunSection::General::Resonance
#endif
