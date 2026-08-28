/////////////////////////////////////////////////////////////////////////
// SSLiouvillianBuilder (RunSection::General::SS)
// ------------------
// Reusable *single-SpinSystem* superspace preparation layer.
//
// HIERARCHY CONTRACT
//   SpinAPI                       : reusable operators / physical primitives
//       ^
//   General/SS                   : one-system Liouville preparation (this file)
//       ^
//   General/MultiSS              : direct-sum graph/network assembly
//       ^
//   TaskMultiSSGeneral           : lifecycle, logging and output only
//
// This class intentionally never follows a Transition into another SpinSystem.
// Intersystem kinetics belongs to General/MultiSS through SpinAPI::TransferChannel.
// It also never calls legacy RunSection/Tasks implementations. Those remain
// executable, independent numerical references.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_SSLiouvillianBuilder
#define MOD_RunSection_General_SS_SSLiouvillianBuilder

#include <armadillo>
#include <string>
#include "SpinAPIfwd.h"
#include "SpinSpace.h"

namespace RunSection::General::SS
{
    enum class SSHamiltonianMode
    {
        FixedFull,
        RotatedFull,
        RotatedSecular
    };

    enum class SSRelaxationModel
    {
        Operators,
        NakajimaZwanzig,
        Redfield
    };

    struct SSPreparedSystem
    {
        SpinAPI::system_ptr system;
        std::shared_ptr<SpinAPI::SpinSpace> space;
        arma::sp_cx_mat hamiltonian;
        arma::sp_cx_mat internalLiouvillian;
        arma::cx_mat initialDensity;
    };

    class SSLiouvillianBuilder
    {
    public:
        static bool Prepare(const SpinAPI::system_ptr &_system,
            SSHamiltonianMode _mode, const arma::mat &_rotation,
            SSPreparedSystem &_prepared, std::string &_error,
            SSRelaxationModel _relaxationModel = SSRelaxationModel::Operators);

        static bool BuildHamiltonian(const SpinAPI::system_ptr &_system,
            SpinAPI::SpinSpace &_space, SSHamiltonianMode _mode,
            const arma::mat &_rotation, arma::sp_cx_mat &_hamiltonian,
            std::string &_error);

        static bool BuildInitialDensity(const SpinAPI::system_ptr &_system,
            SpinAPI::SpinSpace &_space, const arma::sp_cx_mat &_hamiltonian,
            const arma::mat &_rotation,
            arma::cx_mat &_density, std::string &_error);

        static bool BuildInternalLiouvillian(const SpinAPI::system_ptr &_system,
            SpinAPI::SpinSpace &_space, const arma::sp_cx_mat &_hamiltonian,
            const arma::mat &_rotation, arma::sp_cx_mat &_liouvillian,
            std::string &_error,
            SSRelaxationModel _relaxationModel = SSRelaxationModel::Operators);
    };
}

#endif
