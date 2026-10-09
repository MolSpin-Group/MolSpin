/////////////////////////////////////////////////////////////////////////
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "ResonanceDiagnostics.h"
#include "ObjectParser.h"
#include <algorithm>
#include <fstream>

namespace RunSection::General::Resonance
{
    OrientationDiagnostics OrientationDiagnostics::FromProperties(const MSDParser::ObjectParser &_props)
    {
        OrientationDiagnostics diagnostics;
        (void)_props.Get("debugpowder", diagnostics.enabled);
        (void)_props.Get("debugresonance", diagnostics.enabled);
        (void)_props.Get("debugtrepr", diagnostics.enabled);
        (void)_props.Get("debugorientationdump", diagnostics.enabled);
        if (!diagnostics.enabled)
            return diagnostics;

        (void)_props.Get("debugfieldmin", diagnostics.fieldMinT);
        (void)_props.Get("debugfieldmax", diagnostics.fieldMaxT);
        (void)_props.Get("debugmaxorientations", diagnostics.maxOrientations);

        std::string datafile;
        if (_props.Get("datafile", datafile) && !datafile.empty())
            diagnostics.file = datafile + ".orientation_debug.tsv";
        else
            diagnostics.file = "statichs_resonance.orientation_debug.tsv";
        (void)_props.Get("debugfile", diagnostics.file);

        if (diagnostics.fieldMinT > diagnostics.fieldMaxT)
            std::swap(diagnostics.fieldMinT, diagnostics.fieldMaxT);
        return diagnostics;
    }
    void OrientationDiagnostics::InitialiseFile(const std::string &_systemName,
                                                const std::vector<std::string> &_spinNames) const
    {
        if (!enabled)
            return;

        std::ofstream out(file, std::ios::out | std::ios::trunc);
        out << "system\tgrid_index\tgamma_index\ttheta\tphi\tgamma\tweight\ttransition_m\ttransition_"
               "n\tresonance_T\tresonance_mT\ttotal_x\ttotal_y\ttotal_perp\tcross_x\tcross_y";
        for (const auto &spinName : _spinNames)
        {
            out << "\t" << _systemName << "." << spinName << "_x";
            out << "\t" << _systemName << "." << spinName << "_y";
            out << "\t" << _systemName << "." << spinName << "_perp";
            out << "\t" << _systemName << "." << spinName << "_p";
            out << "\t" << _systemName << "." << spinName << "_m";
        }
        out << "\n";
    }
    bool OrientationDiagnostics::ShouldRecord(size_t _gridIndex, double _resonanceFieldT) const
    {
        return enabled && (maxOrientations <= 0 || _gridIndex < static_cast<size_t>(maxOrientations)) &&
               _resonanceFieldT >= fieldMinT && _resonanceFieldT <= fieldMaxT;
    }
} // namespace RunSection::General::Resonance
