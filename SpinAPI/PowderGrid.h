/////////////////////////////////////////////////////////////////////////
// PowderGrid class (SpinAPI Module)
// ------------------
// Special grid for creating powder averages.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_PowderGrid
#define MOD_SpinAPI_PowderGrid

#include <armadillo>
#include <array>
#include <string>
#include <vector>

namespace SpinAPI
{
	struct PowderOrientation
	{
		double theta = 0.0;
		double phi = 0.0;
		double weight = 1.0;
	};

	using PowderGrid = std::vector<PowderOrientation>;

	struct SopheGridParameters
	{
		double maxPhi = 0.0;
		bool closedPhi = false;
		int nOctants = 0;
	};

	struct PowderProjectionMesh
	{
		bool axial = false;
		std::vector<std::array<unsigned int, 3>> triangles;
		std::vector<double> weights;
	};

	enum class PowderGridDomain
	{
		UpperHemisphere,
		FullSphere
	};

	// MolSpin's powder spectra use this active ZYZ matrix as frame_to_lab.
	// Tensor-aware Hamiltonian builders then apply R * T * R.t(), i.e. the
	// whole crystallite is reoriented relative to the static field and
	// microwave axes while internal molecular tensor frames remain fixed.
	bool CreateZYZRotationMatrix(double alpha, double beta, double gamma, arma::mat &R);

	// Golden-angle grid, uniform in cos(theta). The hemisphere grid is the
	// historical MolSpin convention; the full-sphere variant is needed for
	// symmetry-free powder integrations and external comparisons.
	bool CreateUniformPowderGrid(int npoints, PowderGridDomain domain, PowderGrid &grid);

	// SOPHE-style grids and projection meshes are shared by pepper-like
	// resonance projection tasks and can also be reused by future powder tasks.
	bool GetSopheGridParameters(const std::string &symmetry, SopheGridParameters &params);
	int SopheGridPointCount(int gridSize, int nOctants, bool closedPhi);
	bool CreateSophePowderGrid(int gridSize, const std::string &symmetry, PowderGrid &grid);

	// Historical quarter-sphere grid used by StaticSS powder workflows. It is
	// kept as a named API helper so task code no longer owns grid formulas.
	bool CreateOctantPowderGrid(int npoints, PowderGrid &grid);

	bool BuildSopheProjectionMesh(int nOctants, bool closedPhi, int gridSize, const PowderGrid &grid, PowderProjectionMesh &mesh);
	void ProjectPowderZones(const std::vector<double> &position, const std::vector<double> &amplitude, const std::vector<double> &segmentWeights, const std::vector<double> &xAxis, std::vector<double> &spectrum);
	void ProjectPowderTriangles(const std::vector<std::array<unsigned int, 3>> &triangles, const std::vector<double> &areas, const std::vector<double> &position, const std::vector<double> &amplitude, const std::vector<double> &xAxis, std::vector<double> &spectrum);

	PowderOrientation IdentityPowderOrientation();
}

#endif
