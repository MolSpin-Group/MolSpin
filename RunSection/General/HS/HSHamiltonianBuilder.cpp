/////////////////////////////////////////////////////////////////////////
// DEVELOPER WORKFLOW / OWNERSHIP MAP
// ----------------------------------------------------------------------
// HSGeneral Hamiltonian policy layer.
//
// What is done here:
//   - Selects static/dynamic, full/secular and H0/H1 Hamiltonian construction paths.
//   - Applies the current crystallite rotation but leaves interaction physics to SpinAPI::SpinSpace.
//
// Connections to the General framework / SpinAPI:
//   - Called by TaskHSGeneral and pulse/dynamic propagation preparation.
//   - SpinAPI owns Interaction -> Hilbert Hamiltonian conversion and tensor rotation.
//   - SSGeneral uses SSLiouvillianBuilder instead; MultiSS obtains each local Hamiltonian through SSGeneral.
//
// Why this ownership is used:
//   - The builder chooses a SpinAPI path rather than reimplementing interaction algebra in RunSection.
//   - H0 is approximated while H1 remains explicit so a static high-field approximation is not confused with a driven rotating-wave model.
//
// TODO:
//   - Introduce a shared realization-aware Hamiltonian provider before enabling semiclassical-field / strain ensembles in HSGeneral; one realization must be a square Hamiltonian, not the current composite semiclassical matrix.
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
// HSHamiltonianBuilder implementation (RunSection::General::HS)
// ------------------
// Policy layer over SpinSpace Hamiltonian construction. No interaction physics
// is implemented here; the builder only selects static/dynamic, full/secular,
// H0/H1, and powder-aware SpinAPI paths.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2026 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "HSHamiltonianBuilder.h"

namespace RunSection::General::HS
{
	bool HSHamiltonianBuilder::BuildStatic(const HSOrientation &orientation,
		arma::sp_cx_mat &hamiltonian, arma::sp_cx_mat *h0, std::string &error) const
	{
		error.clear();
		// An explicit H0/H1 split is meaningful both with and without powder
		// averaging.  Using the identity orientation for a single crystal keeps
		// the RWA/secular contract identical to the powder path: H0 is selected
		// with the requested approximation, H1 is retained in full.
		if (plan.hasH0List)
		{
			SpinAPI::HilbertPowderHamiltonian parts;
			if (!space.PowderHamiltonianRotated(plan.h0List,
				plan.hasH1List ? plan.h1List : std::vector<std::string>(),
				orientation.frameToLab, plan.approximation, parts))
			{
				error = plan.IsPowder()
					? "failed to construct the orientation-specific H0/H1 Hamiltonian"
					: "failed to construct the explicit H0/H1 Hamiltonian";
				return false;
			}
			hamiltonian = parts.total;
			if (h0 != nullptr) *h0 = parts.H0;
			return true;
		}

		if (plan.IsPowder()) { error = "powder Hamiltonian construction requires hamiltonianh0list"; return false; }
		bool ok = plan.IsDynamic() ? space.StaticHamiltonian(hamiltonian) : space.Hamiltonian(hamiltonian);
		if (!ok) { error = "failed to construct the Hilbert-space Hamiltonian"; return false; }
		if (h0 != nullptr) *h0 = hamiltonian;
		return true;
	}

	bool HSHamiltonianBuilder::BuildAtTime(const HSOrientation &orientation, double time,
		const arma::sp_cx_mat &staticHamiltonian, arma::sp_cx_mat &hamiltonian, std::string &error)
	{
		error.clear();
		if (!plan.IsDynamic() || !space.HasTimedependentInteractions()) { hamiltonian = staticHamiltonian; return true; }
		space.SetTime(time);
		arma::sp_cx_mat dynamic;
		const bool ok = plan.IsPowder()
			? space.DynamicHamiltonianRotatedZYZ(orientation.frameToLab, dynamic)
			: space.DynamicHamiltonian(dynamic);
		if (!ok) { error = "failed to construct the time-dependent Hilbert-space Hamiltonian"; return false; }
		hamiltonian = staticHamiltonian + dynamic;
		return true;
	}
}
