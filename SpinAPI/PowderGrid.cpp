/////////////////////////////////////////////////////////////////////////
// PowderGrid class (SpinAPI Module)
// ------------------
// Special grid for creating powder averages.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////

#include "PowderGrid.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>

namespace SpinAPI
{
	static std::string LowerCase(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
					   { return static_cast<char>(std::tolower(c)); });
		return value;
	}

	static double ClampUnit(double value)
	{
		return std::max(-1.0, std::min(1.0, value));
	}

	static std::array<double, 3> AngToVec(double theta, double phi)
	{
		const double st = std::sin(theta);
		return {st * std::cos(phi), st * std::sin(phi), std::cos(theta)};
	}

	static double Dot(const std::array<double, 3> &a, const std::array<double, 3> &b)
	{
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}

	static double SphericalTriangleArea(const std::array<double, 3> &x1,
										const std::array<double, 3> &x2,
										const std::array<double, 3> &x3)
	{
		const double a1 = std::acos(ClampUnit(Dot(x2, x3)));
		const double a2 = std::acos(ClampUnit(Dot(x3, x1)));
		const double a3 = std::acos(ClampUnit(Dot(x1, x2)));
		const double s = 0.5 * (a1 + a2 + a3);
		const double t1 = std::tan(0.5 * s);
		const double t2 = std::tan(0.5 * (s - a1));
		const double t3 = std::tan(0.5 * (s - a2));
		const double t4 = std::tan(0.5 * (s - a3));
		const double prod = std::max(0.0, t1 * t2 * t3 * t4);
		return 4.0 * std::atan(std::sqrt(prod));
	}

	static bool BuildClosedOctantProjectionMesh(int gridSize,
												const PowderGrid &grid,
												PowderProjectionMesh &mesh)
	{
		if (gridSize < 2)
			return false;

		const size_t expectedPoints = static_cast<size_t>(gridSize) * static_cast<size_t>(gridSize + 1) / 2;
		if (grid.size() != expectedPoints)
			return false;

		std::vector<std::array<double, 3>> vecs(grid.size());
		for (size_t i = 0; i < grid.size(); ++i)
		{
			const double theta = grid[i].theta;
			const double phi = grid[i].phi;
			vecs[i] = AngToVec(theta, phi);
		}

		std::vector<size_t> rowOffset(static_cast<size_t>(gridSize + 1), 0);
		for (int row = 0; row < gridSize; ++row)
			rowOffset[static_cast<size_t>(row + 1)] = rowOffset[static_cast<size_t>(row)] + static_cast<size_t>(row + 1);

		mesh.axial = false;
		mesh.triangles.clear();
		mesh.weights.clear();
		mesh.triangles.reserve(static_cast<size_t>(gridSize - 1) * static_cast<size_t>(gridSize - 1));
		mesh.weights.reserve(static_cast<size_t>(gridSize - 1) * static_cast<size_t>(gridSize - 1));

		for (int row = 0; row < gridSize - 1; ++row)
		{
			const size_t upper = rowOffset[static_cast<size_t>(row)];
			const size_t lower = rowOffset[static_cast<size_t>(row + 1)];
			const int upperLen = row + 1;

			for (int j = 0; j < upperLen; ++j)
			{
				const unsigned int a = static_cast<unsigned int>(upper + static_cast<size_t>(j));
				const unsigned int b = static_cast<unsigned int>(lower + static_cast<size_t>(j));
				const unsigned int c = static_cast<unsigned int>(lower + static_cast<size_t>(j + 1));
				mesh.triangles.push_back({a, b, c});
				mesh.weights.push_back(SphericalTriangleArea(vecs[a], vecs[b], vecs[c]));
			}

			for (int j = 0; j < upperLen - 1; ++j)
			{
				const unsigned int a = static_cast<unsigned int>(upper + static_cast<size_t>(j));
				const unsigned int b = static_cast<unsigned int>(lower + static_cast<size_t>(j + 1));
				const unsigned int c = static_cast<unsigned int>(upper + static_cast<size_t>(j + 1));
				mesh.triangles.push_back({a, b, c});
				mesh.weights.push_back(SphericalTriangleArea(vecs[a], vecs[b], vecs[c]));
			}
		}

		const double sum = std::accumulate(mesh.weights.begin(), mesh.weights.end(), 0.0);
		if (!(sum > 0.0))
			return false;

		const double scale = 4.0 * arma::datum::pi / sum;
		for (double &w : mesh.weights)
			w *= scale;

		return true;
	}

	bool CreateZYZRotationMatrix(double alpha, double beta, double gamma, arma::mat &R)
	{
		arma::mat R1 = {
			{std::cos(alpha), -std::sin(alpha), 0.0},
			{std::sin(alpha), std::cos(alpha), 0.0},
			{0.0, 0.0, 1.0}};

		arma::mat R2 = {
			{std::cos(beta), 0.0, std::sin(beta)},
			{0.0, 1.0, 0.0},
			{-std::sin(beta), 0.0, std::cos(beta)}};

		arma::mat R3 = {
			{std::cos(gamma), -std::sin(gamma), 0.0},
			{std::sin(gamma), std::cos(gamma), 0.0},
			{0.0, 0.0, 1.0}};

		R = R1 * R2 * R3;
		return true;
	}

	bool CreateUniformPowderGrid(int npoints, PowderGridDomain domain, PowderGrid &grid)
	{
		if (npoints < 1)
			return false;

		grid.resize(static_cast<size_t>(npoints));

		const bool fullSphere = (domain == PowderGridDomain::FullSphere);
		const double golden = arma::datum::pi * (1.0 + std::sqrt(5.0));
		const double solidAngle = fullSphere ? 4.0 : 2.0;

		for (int i = 0; i < npoints; ++i)
		{
			const double index = static_cast<double>(i) + 0.5;
			const double z = fullSphere
								 ? (1.0 - 2.0 * index / static_cast<double>(npoints))
								 : (1.0 - index / static_cast<double>(npoints));
			grid[static_cast<size_t>(i)] = {
				std::acos(z),
				golden * index,
				solidAngle * arma::datum::pi / static_cast<double>(npoints)};
		}

		return true;
	}

	bool GetSopheGridParameters(const std::string &symmetry, SopheGridParameters &params)
	{
		const std::string sym = LowerCase(symmetry);
		params = SopheGridParameters{};

		if (sym == "c1")
		{
			params.maxPhi = 2.0 * arma::datum::pi;
			params.closedPhi = false;
			params.nOctants = 8;
		}
		else if (sym == "ci")
		{
			params.maxPhi = 2.0 * arma::datum::pi;
			params.closedPhi = false;
			params.nOctants = 4;
		}
		else if (sym == "c2h")
		{
			params.maxPhi = arma::datum::pi;
			params.closedPhi = false;
			params.nOctants = 2;
		}
		else if (sym == "s6")
		{
			params.maxPhi = 2.0 * arma::datum::pi / 3.0;
			params.closedPhi = false;
			params.nOctants = 2;
		}
		else if (sym == "c4h")
		{
			params.maxPhi = arma::datum::pi / 2.0;
			params.closedPhi = false;
			params.nOctants = 1;
		}
		else if (sym == "c6h")
		{
			params.maxPhi = arma::datum::pi / 3.0;
			params.closedPhi = false;
			params.nOctants = 1;
		}
		else if (sym == "d2h" || sym == "th")
		{
			params.maxPhi = arma::datum::pi / 2.0;
			params.closedPhi = true;
			params.nOctants = 1;
		}
		else if (sym == "d3d")
		{
			params.maxPhi = arma::datum::pi / 3.0;
			params.closedPhi = true;
			params.nOctants = 1;
		}
		else if (sym == "d4h" || sym == "oh")
		{
			params.maxPhi = arma::datum::pi / 4.0;
			params.closedPhi = true;
			params.nOctants = 1;
		}
		else if (sym == "d6h")
		{
			params.maxPhi = arma::datum::pi / 6.0;
			params.closedPhi = true;
			params.nOctants = 1;
		}
		else if (sym == "dinfh")
		{
			params.maxPhi = 0.0;
			params.closedPhi = true;
			params.nOctants = 0;
		}
		else if (sym == "o3")
		{
			params.maxPhi = 0.0;
			params.closedPhi = true;
			params.nOctants = -1;
		}
		else
		{
			return false;
		}

		return true;
	}

	int SopheGridPointCount(int gridSize, int nOctants, bool closedPhi)
	{
		if (gridSize < 1)
			return 0;
		if (nOctants == -1)
			return 1;
		if (nOctants == 0)
			return gridSize;

		int nOct = (nOctants == 8) ? 4 : nOctants;
		int nOrient = gridSize + nOct * gridSize * (gridSize - 1) / 2;
		if (!closedPhi)
			nOrient -= (gridSize - 1);

		if (nOctants == 8)
		{
			int nPhi = nOct * (gridSize - 1) + 1;
			nOrient += (nOrient - nPhi + 1);
		}

		return nOrient;
	}

	bool CreateSophePowderGrid(int gridSize, const std::string &symmetry, PowderGrid &grid)
	{
		grid.clear();
		if (gridSize < 1)
			return false;

		SopheGridParameters params;
		if (!GetSopheGridParameters(symmetry, params))
			return false;

		if (params.nOctants == -1)
		{
			grid.push_back({0.0, 0.0, 4.0 * arma::datum::pi});
			return true;
		}

		if (params.nOctants == 0)
		{
			if (gridSize < 2)
				return false;
			const double dtheta = (arma::datum::pi / 2.0) / static_cast<double>(gridSize - 1);
			std::vector<double> boundaries;
			boundaries.reserve(static_cast<size_t>(gridSize + 1));
			boundaries.push_back(0.0);
			for (int i = 0; i < gridSize - 1; ++i)
				boundaries.push_back(dtheta * (0.5 + static_cast<double>(i)));
			boundaries.push_back(arma::datum::pi / 2.0);

			grid.reserve(static_cast<size_t>(gridSize));
			for (int i = 0; i < gridSize; ++i)
			{
				const double theta = dtheta * static_cast<double>(i);
				const double w = -2.0 * (2.0 * arma::datum::pi) * (std::cos(boundaries[i + 1]) - std::cos(boundaries[i]));
				grid.push_back({theta, 0.0, w});
			}

			return true;
		}

		if (gridSize < 2)
			return false;

		const int nOct = (params.nOctants == 8) ? 4 : params.nOctants;
		const double dtheta = (arma::datum::pi / 2.0) / static_cast<double>(gridSize - 1);
		const double sindth2 = std::sin(dtheta / 2.0);
		const double w0 = params.closedPhi ? 0.5 : 1.0;

		const int nOrientations = gridSize + nOct * gridSize * (gridSize - 1) / 2;
		std::vector<double> phi(static_cast<size_t>(nOrientations), 0.0);
		std::vector<double> theta(static_cast<size_t>(nOrientations), 0.0);
		std::vector<double> weights(static_cast<size_t>(nOrientations), 0.0);

		phi[0] = 0.0;
		theta[0] = 0.0;
		weights[0] = params.maxPhi * (1.0 - std::cos(dtheta / 2.0));

		int start = 1;
		for (int iSlice = 2; iSlice <= gridSize - 1; ++iSlice)
		{
			const int nPhi = nOct * (iSlice - 1) + 1;
			const double dPhi = params.maxPhi / static_cast<double>(nPhi - 1);
			for (int j = 0; j < nPhi; ++j)
			{
				const int idx = start + j;
				double w = 2.0 * std::sin((iSlice - 1) * dtheta) * sindth2 * dPhi;
				if (j == 0)
					w *= w0;
				else if (j == nPhi - 1)
					w *= 0.5;
				weights[idx] = w;
				phi[idx] = dPhi * static_cast<double>(j);
				theta[idx] = dtheta * static_cast<double>(iSlice - 1);
			}
			start += nPhi;
		}

		const int nPhiEq = nOct * (gridSize - 1) + 1;
		const double dPhiEq = params.maxPhi / static_cast<double>(nPhiEq - 1);
		for (int j = 0; j < nPhiEq; ++j)
		{
			const int idx = start + j;
			double w = sindth2 * dPhiEq;
			if (j == 0)
				w *= w0;
			else if (j == nPhiEq - 1)
				w *= 0.5;
			weights[idx] = w;
			phi[idx] = dPhiEq * static_cast<double>(j);
			theta[idx] = arma::datum::pi / 2.0;
		}

		if (!params.closedPhi)
		{
			std::vector<int> rmv;
			rmv.reserve(static_cast<size_t>(gridSize - 1));
			int csum = 0;
			for (int i = 1; i <= gridSize - 1; ++i)
			{
				csum += nOct * i + 1;
				rmv.push_back(csum);
			}

			std::vector<double> phi2;
			std::vector<double> theta2;
			std::vector<double> weights2;
			phi2.reserve(phi.size() - rmv.size());
			theta2.reserve(theta.size() - rmv.size());
			weights2.reserve(weights.size() - rmv.size());

			size_t rmv_pos = 0;
			for (size_t idx = 0; idx < phi.size(); ++idx)
			{
				if (rmv_pos < rmv.size() && static_cast<int>(idx) == rmv[rmv_pos])
				{
					++rmv_pos;
					continue;
				}
				phi2.push_back(phi[idx]);
				theta2.push_back(theta[idx]);
				weights2.push_back(weights[idx]);
			}

			phi.swap(phi2);
			theta.swap(theta2);
			weights.swap(weights2);
		}

		if (params.nOctants == 8)
		{
			const int nPhi = nPhiEq;
			const int N = static_cast<int>(theta.size());
			int start_idx = N - nPhi;
			if (start_idx < 0)
				start_idx = 0;

			std::vector<double> phi_add;
			std::vector<double> theta_add;
			std::vector<double> weights_add;
			phi_add.reserve(static_cast<size_t>(start_idx + 1));
			theta_add.reserve(static_cast<size_t>(start_idx + 1));
			weights_add.reserve(static_cast<size_t>(start_idx + 1));

			for (int i = start_idx; i >= 0; --i)
			{
				weights[i] *= 0.5;
				phi_add.push_back(phi[i]);
				theta_add.push_back(arma::datum::pi - theta[i]);
				weights_add.push_back(weights[i]);
			}

			phi.insert(phi.end(), phi_add.begin(), phi_add.end());
			theta.insert(theta.end(), theta_add.begin(), theta_add.end());
			weights.insert(weights.end(), weights_add.begin(), weights_add.end());
		}

		const double scale = 2.0 * (2.0 * arma::datum::pi / params.maxPhi);
		for (auto &w : weights)
			w *= scale;

		grid.reserve(phi.size());
		for (size_t i = 0; i < phi.size(); ++i)
			grid.push_back({theta[i], phi[i], weights[i]});

		return true;
	}

	bool CreateOctantPowderGrid(int npoints, PowderGrid &grid)
	{
		if (npoints < 1)
			return false;

		grid.resize(static_cast<size_t>(npoints * npoints));

		int idx = 0;
		for (int k = 0; k < npoints; ++k)
		{
			const double u = (static_cast<double>(k) + 0.5) / static_cast<double>(npoints);
			const double theta = std::acos(u);

			for (int j = 0; j < npoints; ++j)
			{
				const double phi = (static_cast<double>(j) + 0.5) * (arma::datum::pi / 2.0) / static_cast<double>(npoints);
				const double weight = (arma::datum::pi / 2.0 / static_cast<double>(npoints)) * (1.0 / static_cast<double>(npoints));
				grid[static_cast<size_t>(idx)] = {theta, phi, weight};
				++idx;
			}
		}

		return true;
	}

	bool BuildSopheProjectionMesh(int nOctants,
								  bool closedPhi,
								  int gridSize,
								  const PowderGrid &grid,
								  PowderProjectionMesh &mesh)
	{
		mesh = PowderProjectionMesh{};

		if (nOctants == 0)
		{
			if (grid.size() < 2)
				return false;

			mesh.axial = true;
			mesh.weights.resize(grid.size() - 1);
			for (size_t i = 0; i + 1 < grid.size(); ++i)
			{
				const double theta1 = grid[i].theta;
				const double theta2 = grid[i + 1].theta;
				mesh.weights[i] = (std::cos(theta1) - std::cos(theta2)) * 4.0 * arma::datum::pi;
			}
			return true;
		}

		if (nOctants == 1 && closedPhi)
			return BuildClosedOctantProjectionMesh(gridSize, grid, mesh);

		return false;
	}

	void ProjectPowderZones(const std::vector<double> &position,
							const std::vector<double> &amplitude,
							const std::vector<double> &segmentWeights,
							const std::vector<double> &xAxis,
							std::vector<double> &spectrum)
	{
		if (position.size() < 2 || position.size() != amplitude.size() || segmentWeights.size() + 1 != position.size() || xAxis.size() < 2)
			return;

		const double delta = xAxis[1] - xAxis[0];
		if (delta == 0.0)
			return;

		const long nPoints = static_cast<long>(xAxis.size());
		for (size_t iSeg = 0; iSeg < segmentWeights.size(); ++iSeg)
		{
			double left = position[iSeg];
			double right = position[iSeg + 1];
			if (!std::isfinite(left) || !std::isfinite(right))
				continue;

			if (left > right)
				std::swap(left, right);

			left = (left - xAxis[0]) / delta;
			right = (right - xAxis[0]) / delta;
			long first = static_cast<long>(left);
			long last = static_cast<long>(right);
			if (first >= nPoints || last < 0)
				continue;

			const double meanAmp = 0.5 * (amplitude[iSeg] + amplitude[iSeg + 1]) / delta;
			if (first == last)
			{
				if (first >= 0 && first < nPoints)
					spectrum[static_cast<size_t>(first)] += meanAmp * segmentWeights[iSeg];
			}
			else
			{
				const double height = meanAmp * segmentWeights[iSeg] / (right - left);
				if (first >= 0)
					spectrum[static_cast<size_t>(first)] += height * (static_cast<double>(first) + 1.0 - left);
				else
					first = -1;

				if (last < nPoints)
					spectrum[static_cast<size_t>(last)] += height * (right - static_cast<double>(last));
				else
					last = nPoints;

				for (long idx = first + 1; idx < last; ++idx)
					spectrum[static_cast<size_t>(idx)] += height;
			}
		}
	}

	void ProjectPowderTriangles(const std::vector<std::array<unsigned int, 3>> &triangles,
								const std::vector<double> &areas,
								const std::vector<double> &position,
								const std::vector<double> &amplitude,
								const std::vector<double> &xAxis,
								std::vector<double> &spectrum)
	{
		if (triangles.empty() || triangles.size() != areas.size() || position.empty() || position.size() != amplitude.size() || xAxis.size() < 2)
			return;

		const double delta = xAxis[1] - xAxis[0];
		if (delta == 0.0)
			return;

		const int nPoints = static_cast<int>(xAxis.size());
		for (size_t iTri = 0; iTri < triangles.size(); ++iTri)
		{
			const auto tri = triangles[iTri];
			double left = position[tri[0]];
			double middle = position[tri[1]];
			double right = position[tri[2]];
			if (!std::isfinite(left) || !std::isfinite(middle) || !std::isfinite(right))
				continue;

			if (right < left)
				std::swap(left, right);
			if (middle < left)
			{
				std::swap(left, middle);
			}
			else if (middle > right)
			{
				std::swap(right, middle);
			}

			left = (left - xAxis[0]) / delta;
			middle = (middle - xAxis[0]) / delta;
			right = (right - xAxis[0]) / delta;

			const double width = right - left;
			const double width1 = middle - left;
			const double width2 = right - middle;
			const double meanAmp = (amplitude[tri[0]] + amplitude[tri[1]] + amplitude[tri[2]]) / 3.0;
			const double area = areas[iTri];

			int first1 = static_cast<int>(left);
			int last1 = static_cast<int>(middle);
			int first2 = last1;
			int last2 = static_cast<int>(right);

			if (width1 > 0.0 && first1 < nPoints && last1 >= 0)
			{
				const double f0 = (2.0 * meanAmp * area / width) / width1 / delta;
				if (first1 == last1)
				{
					if (first1 >= 0 && first1 < nPoints)
						spectrum[static_cast<size_t>(first1)] += f0 * width1 * width1 / 2.0;
				}
				else
				{
					if (first1 >= 0)
						spectrum[static_cast<size_t>(first1)] += f0 * (static_cast<double>(first1) + 1.0 - left) * (static_cast<double>(first1) + 1.0 - left) / 2.0;
					else
						first1 = -1;

					if (last1 < nPoints)
						spectrum[static_cast<size_t>(last1)] += f0 * (((static_cast<double>(last1) + middle) / 2.0) - left) * (middle - static_cast<double>(last1));
					else
						last1 = nPoints;

					const double leftShift = left - 0.5;
					for (int idx = first1 + 1; idx < last1; ++idx)
						spectrum[static_cast<size_t>(idx)] += f0 * (static_cast<double>(idx) - leftShift);
				}
			}

			if (width2 > 0.0 && first2 < nPoints && last2 >= 0)
			{
				const double f0 = (2.0 * meanAmp * area / width) / width2 / delta;
				if (first2 == last2)
				{
					if (first2 >= 0 && first2 < nPoints)
						spectrum[static_cast<size_t>(first2)] += f0 * width2 * width2 / 2.0;
				}
				else
				{
					if (first2 >= 0)
						spectrum[static_cast<size_t>(first2)] += f0 * (static_cast<double>(first2) + 1.0 - middle) * (right - (middle + static_cast<double>(first2) + 1.0) / 2.0);
					else
						first2 = -1;

					if (last2 < nPoints)
						spectrum[static_cast<size_t>(last2)] += f0 * (right - static_cast<double>(last2)) * (right - static_cast<double>(last2)) / 2.0;
					else
						last2 = nPoints;

					const double rightShift = right - 0.5;
					for (int idx = first2 + 1; idx < last2; ++idx)
						spectrum[static_cast<size_t>(idx)] += f0 * (rightShift - static_cast<double>(idx));
				}
			}

			if (width1 == 0.0 && width2 == 0.0)
			{
				const int first = static_cast<int>(left);
				if (first >= 0 && first < nPoints)
					spectrum[static_cast<size_t>(first)] += meanAmp * area / delta;
			}
		}
	}

	PowderOrientation IdentityPowderOrientation()
	{
		return PowderOrientation{};
	}
}
