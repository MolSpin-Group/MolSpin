/////////////////////////////////////////////////////////////////////////
// TaskStaticHSResonanceSpectra implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>

#include "ActionAddVector.h"
#include "TaskStaticHSResonanceSpectra.h"
#include "ExactResonanceSolver.h"
#include "HybridNuclearResonancePartitionBuilder.h"
#include "HybridNuclearResonancePreparation.h"
#include "HybridNuclearResonanceSolver.h"
#include "ResonanceMagneticMomentBuilder.h"
#include "ResonanceSpectrumEvaluator.h"
#include "ResonanceTypes.h"
#include "ObjectParser.h"
#include "Settings.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Interaction.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RunSection
{
	// -----------------------------------------------------
	// TaskStaticHSResonanceSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSResonanceSpectra::TaskStaticHSResonanceSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection)
		: BasicTask(_parser, _runsection),
		  mwFrequencyGHz(0.0),
		  linewidth_mT(0.0),
		  lineshape("gaussian"),

		  detectionHarmonic(0),
		  modulationAmplitude_mT(0.0),

		  powderGridType("sophe"),
		  powderGridSymmetry("auto"),
		  powderGridSize(0),
		  powdersamplingpoints(0),
		  powderGammaPoints(1),
		  powderFullSphere(true),
		  fullTensorRotation(true),
		  useMzBlocks(true),
		  useSweepCache(true),

		  sweepCacheExact(true),
		  sweepCacheResfields(false),

		  sweepCacheResfieldPoints(0),
		  detectSpinNames(),
		  fieldInteractionName(""),
		  enforceZeemanSync(false),
		  initialStateName(""),
		  hamiltonianH0list(),
		  resonanceSolverMode("exact"),
		  hybridPerturbativeNucleusNames(),
		  hybridFieldStepT(1.0e-4),
		  hybridMinimumCoreStateOverlap(0.90),
		  hybridMinimumNuclearStateOverlap(0.90),
		  hybridJacobianRelativeTolerance(1.0e-4),
		  hybridJacobianAbsoluteTolerance(1.0e-5),
		  hybridOverlapThreshold(1.0e-14),
		  hybridMinimumCumulativeOverlapWeight(0.0),
		  hybridMaximumComponentsPerCoreTransition(0)
	{
	}

	TaskStaticHSResonanceSpectra::~TaskStaticHSResonanceSpectra()
	{
	}

	// -----------------------------------------------------
	// TaskStaticHSResonanceSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticHSResonanceSpectra::RunLocal()
	{
		this->Log() << "Running task StaticHS-Resonance-Spectra." << std::endl;

		// Workflow:
		// 1. Resolve the static Hamiltonian list, lab-field Zeeman interaction,
		//    and spins used for microwave detection.
		// 2. Prefer the sweep cache when the field sweep is linear: the cached
		//    path computes the same transition moments but avoids repeating
		//    setup work for every output field.
		// 3. Build the requested powder grid and gamma sampling.
		// 4. For each orientation, rotate the whole crystallite into the lab
		//    frame, diagonalize H0, evaluate resonant transitions, and add their
		//    weighted contributions to the spectrum.

		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->Log() << "Sweep cache " << (this->useSweepCache ? "enabled" : "disabled");
			if (this->useSweepCache)
			{
				const char *mode = this->sweepCacheExact ? "exact" : (this->sweepCacheResfields ? "resonanceprojection" : "approx");
				this->Log() << " (mode: " << mode << ")";
			}
			this->Log() << "." << std::endl;
		}

		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto sysIt = systems.cbegin(); sysIt != systems.cend(); sysIt++)
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;

			// Hybrid nuclear resonance must branch before the historical exact
			// route constructs a full-system SpinSpace. Otherwise perturbative
			// nuclei would still pay the full product-Hilbert-space cost.
			if (this->resonanceSolverMode == "hybrid")
			{
				this->RunHybridSystem(*sysIt);
				continue;
			}

			SpinAPI::SpinSpace space(*(*sysIt));
			space.UseSuperoperatorSpace(false);
			space.UseFullTensorRotation(this->fullTensorRotation);
			const arma::uword spaceDim = space.HilbertSpaceDimensions();
			const auto allSpins = (*sysIt)->Spins();
			const MzBlocks mzBlocks = BuildMzBlocks(allSpins);
			const bool hasMzBlocks = (this->useMzBlocks && !mzBlocks.mz2.empty() && mzBlocks.mz2.size() == static_cast<size_t>(spaceDim) && mzBlocks.blocks.size() > 1);
			if (this->fullTensorRotation)
			{
				this->Log() << "Full tensor rotation enabled (off-diagonal terms retained)." << std::endl;
			}

			// Build list of interactions to include in H0
			std::vector<std::string> h0list = this->hamiltonianH0list;
			if (h0list.empty())
			{
				for (const auto &interaction : (*sysIt)->Interactions())
				{
					if (!SpinAPI::IsStatic(*interaction))
						continue;
					h0list.push_back(interaction->Name());
				}
			}

			if (h0list.empty())
			{
				this->Log() << "No interactions specified for Hamiltonian H0 in SpinSystem \"" << (*sysIt)->Name() << "\". Skipping." << std::endl;
				continue;
			}

			// The Zeeman interaction defines the laboratory field direction and the
			// field derivative dH/dB needed to convert energy detuning into field
			// detuning through the resonance Jacobian.
			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			if (!this->ResolveFieldInteraction((*sysIt), fieldInteraction))
			{
				this->Log() << "No Zeeman interaction found in SpinSystem \"" << (*sysIt)->Name() << "\". Need a Zeeman interaction for field->frequency mapping." << std::endl;
				continue;
			}

			std::vector<SpinAPI::spin_ptr> detectSpins;
			std::vector<std::string> detectSpinNames;
			if (!this->ResolveDetectionSpins((*sysIt), fieldInteraction, detectSpins, detectSpinNames))
			{
				this->Log() << "Failed to resolve detection spins in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}
			if (detectSpins.empty())
			{
				this->Log() << "No detection spins available in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}
			this->Log() << "Using " << detectSpins.size() << " detection spins in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;

			std::vector<SpinAPI::interaction_ptr> zeemanInteractions = CollectZeemanInteractions((*sysIt), h0list);
			if (zeemanInteractions.empty() && fieldInteraction != nullptr)
				zeemanInteractions.push_back(fieldInteraction);

			FieldSyncGuard zeemanSync;
			if (this->enforceZeemanSync && fieldInteraction != nullptr && !zeemanInteractions.empty())
			{
				const arma::vec fieldRef = fieldInteraction->Field();
				if (fieldRef.n_elem == 3 && fieldRef.is_finite())
				{
					const double tol = 1e-10;
					for (const auto &inter : zeemanInteractions)
					{
						if (inter == nullptr)
							continue;
						const arma::vec f = inter->Field();
						if (f.n_elem != 3 || !f.is_finite())
							continue;
						if (arma::norm(f - fieldRef) > tol)
						{
							this->Log() << "Enforcing Zeeman field synchronization to match \"" << fieldInteraction->Name() << "\"." << std::endl;
							break;
						}
					}
					zeemanSync.Apply(zeemanInteractions, fieldRef);
				}
			}

			arma::vec Bvec = fieldInteraction->Field();
			if (Bvec.n_elem != 3)
			{
				this->Log() << "Zeeman interaction \"" << fieldInteraction->Name() << "\" does not provide a 3-vector field." << std::endl;
				continue;
			}
			if (!this->enforceZeemanSync && zeemanInteractions.size() > 1)
			{
				const double tol = 1e-10;
				for (const auto &inter : zeemanInteractions)
				{
					if (inter == nullptr || inter == fieldInteraction)
						continue;
					const arma::vec f = inter->Field();
					if (f.n_elem != 3 || !f.is_finite())
						continue;
					if (arma::norm(f - Bvec) > tol)
					{
						this->Log() << "Warning: Zeeman interactions have mismatched fields. Consider enforce_zeeman_sync=true for EasySpin-compatible behavior." << std::endl;
						break;
					}
				}
			}
			const double Bmag = arma::norm(Bvec);
			if (!std::isfinite(Bmag) || Bmag <= 0.0)
			{
				this->Log() << "Zeeman field magnitude is invalid (" << Bmag << ")." << std::endl;
				continue;
			}
			const double field_mT = 1.0e3 * Bmag;

			auto cacheIt = this->spectrumCache.find((*sysIt)->Name());
			if (this->useSweepCache && this->RunSettings()->CurrentStep() == 1 && cacheIt == this->spectrumCache.end())
			{
				arma::vec field0;
				arma::vec fieldStep;
				bool cacheOk = this->GetLinearFieldSweep((*sysIt), fieldInteraction, field0, fieldStep);
				if (cacheOk && !this->enforceZeemanSync && zeemanInteractions.size() > 1)
				{
					std::map<std::string, arma::vec> stepsByTarget;
					std::string error;
					if (!CollectAddVectorSteps(this->Actions(), this->RunSettings()->Steps(), stepsByTarget, error))
					{
						cacheOk = false;
						this->Log() << "Sweep cache disabled: " << error << std::endl;
					}
					else
					{
						const arma::vec refField = fieldInteraction->Field();
						const double tol = 1e-8;
						for (const auto &inter : zeemanInteractions)
						{
							if (inter == nullptr || inter == fieldInteraction)
								continue;
							const arma::vec f = inter->Field();
							if (f.n_elem != 3 || !f.is_finite() ||
								!IsParallel(f, refField, 1e-6) ||
								std::abs(arma::norm(f) - arma::norm(refField)) > tol)
							{
								cacheOk = false;
								this->Log() << "Sweep cache disabled: Zeeman fields are not synchronized." << std::endl;
								break;
							}
							const std::string target = (*sysIt)->Name() + "." + inter->Name() + ".field";
							auto it = stepsByTarget.find(target);
							if (it == stepsByTarget.end() || arma::norm(it->second - fieldStep) > tol)
							{
								cacheOk = false;
								this->Log() << "Sweep cache disabled: Zeeman sweep steps are not synchronized." << std::endl;
								break;
							}
						}
					}
				}

				if (cacheOk)
				{
					SpectrumCache cache;
					if (this->BuildCachedSpectrum((*sysIt), fieldInteraction, field0, fieldStep, cache))
					{
						this->spectrumCache.emplace((*sysIt)->Name(), std::move(cache));
						cacheIt = this->spectrumCache.find((*sysIt)->Name());
					}
				}
			}
			if (this->useSweepCache && cacheIt != this->spectrumCache.end())
			{
				const auto &cache = cacheIt->second;
				const auto idx = static_cast<size_t>(this->RunSettings()->CurrentStep() - 1);
				if (idx < cache.field_mT.size())
				{
					this->Data() << this->RunSettings()->CurrentStep() << " ";
					this->Data() << this->RunSettings()->Time() << " ";
					this->WriteStandardOutput(this->Data());
					this->Data() << cache.field_mT[idx] << " "
								 << cache.total_x[idx] << " " << cache.total_y[idx] << " " << cache.total_perp[idx] << " "
								 << cache.cross_x[idx] << " " << cache.cross_y[idx] << " ";
					for (size_t i = 0; i < cache.spin_names.size(); ++i)
					{
						this->Data() << cache.spin_x[i][idx] << " " << cache.spin_y[i][idx] << " " << cache.spin_perp[i][idx] << " "
									 << cache.spin_p[i][idx] << " " << cache.spin_m[i][idx] << " ";
					}
					this->Data() << std::endl;
					continue;
				}
			}

			// Build the initial density matrix. The task supports either explicit
			// projector states or an orientation-dependent thermal state in the
			// eigenframe. Since the observables are linear in rho, multiple initial
			// states are summed before normalization.
			const SpinAPI::StateFrame initialStateFrame = (*sysIt)->InitialStateFrame();
			if (initialStateFrame == SpinAPI::StateFrame::Molecular)
				this->Log() << "Initial state frame = molecular." << std::endl;

			arma::cx_mat rho0;
			bool hasInitialState = false;
			bool useOrientationThermal = false;
			std::vector<std::string> thermalhamiltonian_list;
			double thermalTemperature = 0.0;
			if (initialStateFrame == SpinAPI::StateFrame::Eigen)
			{
				if (!this->initialStateName.empty())
				{
					this->Log() << "frame = eigen currently requires Thermal to be specified via the SpinSystem initialstate property in StaticHS-Resonance-Spectra." << std::endl;
					continue;
				}

				auto initial_states = (*sysIt)->InitialState();
				if (initial_states.size() != 1 || initial_states.front() != nullptr)
				{
					this->Log() << "frame = eigen currently requires a single Thermal initial state in StaticHS-Resonance-Spectra." << std::endl;
					continue;
				}

				useOrientationThermal = true;
				thermalhamiltonian_list = (*sysIt)->ThermalHamiltonianList();
				thermalTemperature = (*sysIt)->Temperature();
				this->Log() << "Initial state = thermal (orientation-dependent eigen frame)" << std::endl;
			}
			else if (!this->initialStateName.empty())
			{
				auto state = (*sysIt)->states_find(this->initialStateName);
				if (state == nullptr)
				{
					this->Log() << "Initial state \"" << this->initialStateName << "\" not found in SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				}
				else
				{
					if (space.GetState(state, rho0))
						hasInitialState = true;
				}
			}
			if (!hasInitialState && !useOrientationThermal)
			{
				auto initial_states = (*sysIt)->InitialState();
				if (initial_states.empty())
				{
					this->Log() << "Skipping SpinSystem \"" << (*sysIt)->Name() << "\" as no initial state was specified." << std::endl;
					continue;
				}

				std::vector<double> initial_weights = (*sysIt)->Weights();
				const bool useInitialWeights = (initial_weights.size() == initial_states.size());
				if (useInitialWeights)
				{
					double sum_weights = std::accumulate(initial_weights.begin(), initial_weights.end(), 0.0);
					if (sum_weights > 0.0)
					{
						for (double &weight : initial_weights)
							weight /= sum_weights;
					}
				}
				else if (!initial_weights.empty())
				{
					this->Log() << "Initial-state weights count does not match initialstate count. Ignoring weights." << std::endl;
				}

				for (size_t stateIndex = 0; stateIndex < initial_states.size(); ++stateIndex)
				{
					auto state = initial_states.cbegin() + static_cast<std::ptrdiff_t>(stateIndex);
					if ((*state) == nullptr)
					{
						this->Log() << "Thermal initial states are not supported in StaticHS-Resonance-Spectra." << std::endl;
						continue;
					}

					arma::cx_mat tmp;
					if (!space.GetState(*state, tmp))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*state)->Name() << "\" of SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
						continue;
					}

					if (useInitialWeights)
						tmp *= initial_weights[stateIndex];

					if (!hasInitialState)
					{
						rho0 = tmp;
						hasInitialState = true;
					}
					else
					{
						rho0 += tmp;
					}
				}
			}

			if (!hasInitialState)
			{
				if (!useOrientationThermal)
				{
					this->Log() << "Failed to construct initial state for SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
					continue;
				}
			}
			if (!useOrientationThermal)
				rho0 /= arma::trace(rho0);

			// Build powder grid (theta,phi) and optional gamma sampling.
			int numPoints = this->powdersamplingpoints;
			SpinAPI::PowderGrid grid;
			const bool useSopheGrid = (this->powderGridType == "sophe");
			std::string gridSymmetry = this->powderGridSymmetry;
			if (useSopheGrid)
			{
				std::string symLower = ToLower(gridSymmetry);
				if (symLower.empty() || symLower == "auto" || symLower == "automatic")
				{
					gridSymmetry = AutoDetectSopheSymmetry((*sysIt), fieldInteraction, h0list, this->fullTensorRotation);
					this->Log() << "Auto-detected SOPHE grid symmetry: " << gridSymmetry << "." << std::endl;
				}
			}
			if (useSopheGrid)
			{
				int gridSize = this->powderGridSize;
				if (gridSize < 2)
				{
					SpinAPI::SopheGridParameters sopheParams;
					if (numPoints > 1 && SpinAPI::GetSopheGridParameters(gridSymmetry, sopheParams))
					{
						int bestSize = 0;
						int bestDiff = std::numeric_limits<int>::max();
						for (int candidate = 2; candidate <= 200; ++candidate)
						{
							int count = SpinAPI::SopheGridPointCount(candidate, sopheParams.nOctants, sopheParams.closedPhi);
							int diff = std::abs(count - numPoints);
							if (diff < bestDiff)
							{
								bestDiff = diff;
								bestSize = candidate;
								if (diff == 0)
									break;
							}
						}
						if (bestSize > 0)
							gridSize = bestSize;
					}
					if (gridSize < 2)
						gridSize = 19;
				}

				if (!SpinAPI::CreateSophePowderGrid(gridSize, gridSymmetry, grid))
				{
					this->Log() << "Failed to obtain SOPHE grid for powder averaging." << std::endl;
					continue;
				}
				numPoints = static_cast<int>(grid.size());
				this->Log() << "Using SOPHE grid (" << gridSymmetry << ", GridSize=" << gridSize << ") with " << numPoints << " orientations." << std::endl;
			}
			else if (numPoints > 1)
			{
				if (!this->CreateUniformGrid(numPoints, grid))
				{
					this->Log() << "Failed to obtain a uniform grid for powder averaging." << std::endl;
					continue;
				}
				this->Log() << "Using powder averaging with " << numPoints << " orientations." << std::endl;
				if (this->powderFullSphere)
					this->Log() << "Using full-sphere powder grid." << std::endl;
			}
			else
			{
				grid.clear();
				grid.push_back({0.0, 0.0, 1.0});
				numPoints = 1;
			}
			const int gamma_points = (numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
			const double gamma_weight = useSopheGrid ? (2.0 * arma::datum::pi / static_cast<double>(gamma_points))
													 : (1.0 / static_cast<double>(gamma_points));
			if (this->powderGammaPoints > 1)
				this->Log() << "Sampling gamma with " << this->powderGammaPoints << " points per orientation." << std::endl;

			// Field-domain linewidth (FWHM, mT). The spectrum is constructed as a
			// field-swept experiment, so broadening is applied directly in field units.
			const double lwB_mT = std::abs(this->linewidth_mT);

			// Detection-spin / Zeeman selection remains task-owned. Tensor,
			// frame, Hilbert embedding and prefactor interpretation are canonical.
			std::vector<General::Resonance::ResonanceMagneticMomentTerm>
				magneticMomentTerms;
			magneticMomentTerms.reserve(detectSpins.size());
			bool magneticMomentTermsOk = true;
			for (size_t i = 0; i < detectSpins.size(); ++i)
			{
				auto zeeman =
					FindZeemanForSpin(detectSpins[i], zeemanInteractions);

				// Preserve the historical fallback only when the designated field
				// interaction actually owns this detection spin.
				if (zeeman == nullptr && fieldInteraction != nullptr)
				{
					const auto group = fieldInteraction->Group1();
					if (std::find(
							group.begin(), group.end(),
							detectSpins[i]) != group.end())
					{
						zeeman = fieldInteraction;
					}
				}

				if (zeeman == nullptr)
				{
					this->Log()
						<< "No Zeeman interaction owns detection spin \""
						<< detectSpins[i]->Name()
						<< "\"; exact resonance magnetic moment is undefined."
						<< std::endl;
					magneticMomentTermsOk = false;
					break;
				}

				General::Resonance::ResonanceMagneticMomentTerm term;
				term.spin = detectSpins[i];
				term.zeeman = zeeman;
				magneticMomentTerms.push_back(std::move(term));
			}
			if (!magneticMomentTermsOk)
				continue;

			// Accumulators
			double total_x = 0.0;
			double total_y = 0.0;
			double total_perp = 0.0;
			double cross_x = 0.0;
			double cross_y = 0.0;
			std::vector<double> spin_x(detectSpins.size(), 0.0);
			std::vector<double> spin_y(detectSpins.size(), 0.0);
			std::vector<double> spin_perp(detectSpins.size(), 0.0);
			std::vector<double> spin_p(detectSpins.size(), 0.0);
			std::vector<double> spin_m(detectSpins.size(), 0.0);

			// For dH/dB we need the Zeeman Hamiltonian only (rotated per orientation)
			std::vector<std::string> zeelist;
			zeelist.reserve(zeemanInteractions.size());
			for (const auto &inter : zeemanInteractions)
				zeelist.push_back(inter->Name());

			const size_t spin_count = detectSpins.size();

			General::Resonance::SpectrumRequest resonanceRequest;
			resonanceRequest.microwaveFrequencyGHz = this->mwFrequencyGHz;
			resonanceRequest.linewidth_mT = lwB_mT;
			resonanceRequest.lineshape =
				(this->lineshape == "lorentzian")
				? General::Resonance::Lineshape::Lorentzian
				: General::Resonance::Lineshape::Gaussian;
			resonanceRequest.populationThreshold = 1.0e-15;
			resonanceRequest.minimumSlope = 1.0e-15;
			resonanceRequest.maximumDBdOmega = 1.0e5;

			// For one powder orientation we:
			// 1. rotate the static Hamiltonian and the initial state,
			// 2. diagonalize H,
			// 3. evaluate transition populations and magnetic dipole matrix elements,
			// 4. convert energy detuning to field detuning through dB/dE,
			// 5. accumulate absorptive/emissive intensities.
			auto accumulate_grid = [&](int grid_num, SpinAPI::SpinSpace &space_local,
									   double &acc_total_x, double &acc_total_y, double &acc_total_perp,
									   double &acc_cross_x, double &acc_cross_y,
									   std::vector<double> &acc_spin_x, std::vector<double> &acc_spin_y,
									   std::vector<double> &acc_spin_perp, std::vector<double> &acc_spin_p,
									   std::vector<double> &acc_spin_m)
			{
				auto [theta, phi, w_solid] = grid[grid_num];
				const double base_weight = w_solid;

				for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
				{
					double gamma = 0.0;
					if (gamma_points > 1)
						gamma = 2.0 * arma::datum::pi * (static_cast<double>(gamma_idx) + 0.5) / static_cast<double>(gamma_points);

					const double w = base_weight * gamma_weight;

					arma::mat Rot;
					if (!this->CreatePassiveZYZRotationMatrix(phi, theta, gamma, Rot))
						continue;

					// Rot is the same molecular-to-lab orientation used by the
					// Hamiltonian and canonical magnetic-moment builder.
					arma::cx_mat rho_oriented;
					if (useOrientationThermal)
					{
						arma::sp_cx_mat Hthermal_sp;
						if (!space_local.BaseHamiltonianRotatedZYZ(thermalhamiltonian_list, Rot, Hthermal_sp) ||
							!space_local.ThermalStateFromHamiltonian(arma::cx_mat(Hthermal_sp), thermalTemperature, rho_oriented))
							continue;
					}
					else
					{
						rho_oriented = rho0;
						// Projector and explicit density states may be defined
						// in the molecular frame. Rotate them with the same
						// powder orientation used for the Hamiltonian.
						if (initialStateFrame == SpinAPI::StateFrame::Molecular && !space_local.RotateState(rho0, Rot, rho_oriented))
							continue;
					}

						// Pepper-like resonance search path: Resonance spectra diagonalizes
						// the full Hilbert Hamiltonian at each field/orientation.
						// This deliberately does not call
						// PowderHamiltonianRotatedSA; resonance positions are
						// obtained from full transition energies, not from the
						// rotating-frame time-propagation Liouvillian.
						arma::sp_cx_mat H0_sp;
					if (!space_local.BaseHamiltonianRotatedZYZ(h0list, Rot, H0_sp))
						continue;

					arma::vec eigval;
					arma::cx_mat eigvec;
					bool have_eig = false;
					if (hasMzBlocks && IsBlockDiagonalMz(H0_sp, mzBlocks.mz2, 1e-12))
					{
						have_eig = EigSymBlockMz(H0_sp, mzBlocks.blocks, eigval, eigvec);
					}
					else
					{
						arma::cx_mat H0 = arma::cx_mat(H0_sp);
						have_eig = arma::eig_sym(eigval, eigvec, H0);
					}
					if (!have_eig)
						continue;

					// Task retains Hamiltonian construction and optional Mz-block
					// diagonalization. General Resonance owns subsequent Jacobian and
					// transition physics.
					arma::sp_cx_mat Hz_sp;
					if (!space_local.BaseHamiltonianRotatedZYZ(zeelist, Rot, Hz_sp))
						continue;
					arma::sp_cx_mat dHdB_sp = Hz_sp / Bmag; // rad/ns/T

					std::vector<General::Resonance::ResonanceDetectionOperator>
						detectionChannels;
					std::string magneticMomentError;
					if (!General::Resonance::ResonanceMagneticMomentBuilder::
							BuildTransverseChannels(
								space_local,
								magneticMomentTerms,
								Rot,
								this->fullTensorRotation,
								detectionChannels,
								magneticMomentError))
					{
						continue;
					}
					if (detectionChannels.size() != spin_count)
						continue;

					arma::cx_mat muxT =
						arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					arma::cx_mat muyT =
						arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					for (const auto &channel : detectionChannels)
					{
						muxT += channel.x;
						muyT += channel.y;
					}

					General::Resonance::ResonanceLineSet resonanceLines;
					std::string resonanceError;
					if (!General::Resonance::ExactResonanceSolver::Generate(
							eigval, eigvec, rho_oriented, dHdB_sp,
							muxT, muyT, resonanceRequest,
							resonanceLines, resonanceError, detectionChannels))
						continue;

					General::Resonance::SpectrumPoint resonancePoint;
					if (!General::Resonance::ResonanceSpectrumEvaluator::Evaluate(
							resonanceLines, resonanceRequest, resonancePoint, resonanceError))
						continue;
					if (resonancePoint.channels.size() != spin_count)
						continue;

					const double loc_total_x = resonancePoint.totalX;
					const double loc_total_y = resonancePoint.totalY;
					const double loc_total_perp = resonancePoint.totalPerpendicular;
					const double loc_cross_x = resonancePoint.crossX;
					const double loc_cross_y = resonancePoint.crossY;
					std::vector<double> loc_spin_x(spin_count, 0.0);
					std::vector<double> loc_spin_y(spin_count, 0.0);
					std::vector<double> loc_spin_perp(spin_count, 0.0);
					std::vector<double> loc_spin_p(spin_count, 0.0);
					std::vector<double> loc_spin_m(spin_count, 0.0);
					for (size_t i = 0; i < spin_count; ++i)
					{
						const auto &channel = resonancePoint.channels[i];
						loc_spin_x[i] = channel.x;
						loc_spin_y[i] = channel.y;
						loc_spin_perp[i] = channel.perpendicular;
						loc_spin_p[i] = channel.plus;
						loc_spin_m[i] = channel.minus;
					}

					acc_total_x += w * loc_total_x;
					acc_total_y += w * loc_total_y;
					acc_total_perp += w * loc_total_perp;
					acc_cross_x += w * loc_cross_x;
					acc_cross_y += w * loc_cross_y;

					for (size_t i = 0; i < spin_count; ++i)
					{
						acc_spin_x[i] += w * loc_spin_x[i];
						acc_spin_y[i] += w * loc_spin_y[i];
						acc_spin_perp[i] += w * loc_spin_perp[i];
						acc_spin_p[i] += w * loc_spin_p[i];
						acc_spin_m[i] += w * loc_spin_m[i];
					}
				}
			};

#ifdef _OPENMP
#pragma omp parallel
			{
				SpinAPI::SpinSpace space_local(*(*sysIt));
				space_local.UseSuperoperatorSpace(false);
				space_local.UseFullTensorRotation(this->fullTensorRotation);

				double local_total_x = 0.0;
				double local_total_y = 0.0;
				double local_total_perp = 0.0;
				double local_cross_x = 0.0;
				double local_cross_y = 0.0;
				std::vector<double> local_spin_x(spin_count, 0.0);
				std::vector<double> local_spin_y(spin_count, 0.0);
				std::vector<double> local_spin_perp(spin_count, 0.0);
				std::vector<double> local_spin_p(spin_count, 0.0);
				std::vector<double> local_spin_m(spin_count, 0.0);

#pragma omp for
				for (int grid_num = 0; grid_num < numPoints; ++grid_num)
				{
					accumulate_grid(grid_num, space_local, local_total_x, local_total_y, local_total_perp, local_cross_x, local_cross_y,
									local_spin_x, local_spin_y, local_spin_perp, local_spin_p, local_spin_m);
				}

#pragma omp critical
				{
					total_x += local_total_x;
					total_y += local_total_y;
					total_perp += local_total_perp;
					cross_x += local_cross_x;
					cross_y += local_cross_y;
					for (size_t i = 0; i < spin_count; ++i)
					{
						spin_x[i] += local_spin_x[i];
						spin_y[i] += local_spin_y[i];
						spin_perp[i] += local_spin_perp[i];
						spin_p[i] += local_spin_p[i];
						spin_m[i] += local_spin_m[i];
					}
				}
			}
#else
			for (int grid_num = 0; grid_num < numPoints; ++grid_num)
			{
				accumulate_grid(grid_num, space, total_x, total_y, total_perp, cross_x, cross_y,
								spin_x, spin_y, spin_perp, spin_p, spin_m);
			}
#endif

			// Output
			this->Data() << this->RunSettings()->CurrentStep() << " ";
			this->Data() << this->RunSettings()->Time() << " ";
			this->WriteStandardOutput(this->Data());
			this->Data() << field_mT << " "
						 << total_x << " " << total_y << " " << total_perp << " "
						 << cross_x << " " << cross_y << " ";
			for (size_t i = 0; i < detectSpinNames.size(); ++i)
			{
				this->Data() << spin_x[i] << " " << spin_y[i] << " " << spin_perp[i] << " "
							 << spin_p[i] << " " << spin_m[i] << " ";
			}
			this->Data() << std::endl;
		}

		return true;
	}


	bool TaskStaticHSResonanceSpectra::RunHybridSystem(const SpinAPI::system_ptr &_system)
	{
		using namespace General::Resonance;

		if (_system == nullptr)
			return false;

		// The canonical hybrid partition owns every physical interaction exactly
		// once. Therefore a task-level H0 subset is valid only when it is the
		// complete static SpinSystem interaction set. We never silently add an
		// interaction omitted by the user and never silently drop physical terms.
		std::vector<std::string> h0list = this->hamiltonianH0list;
		if (h0list.empty())
		{
			for (const auto &interaction : _system->Interactions())
			{
				if (interaction != nullptr && SpinAPI::IsStatic(*interaction))
					h0list.push_back(interaction->Name());
			}
		}

		const auto systemInteractions = _system->Interactions();
		if (h0list.size() != systemInteractions.size())
		{
			this->Log() << "Hybrid resonance requires HamiltonianH0list to cover the complete static SpinSystem interaction set exactly once." << std::endl;
			return false;
		}

		for (const auto &interaction : systemInteractions)
		{
			if (interaction == nullptr || interaction->HasTimeDependence())
			{
				this->Log() << "Hybrid resonance currently requires a completely static SpinSystem." << std::endl;
				return false;
			}

			const auto count = static_cast<std::size_t>(
				std::count(h0list.begin(), h0list.end(), interaction->Name()));
			if (count != 1)
			{
				this->Log() << "Hybrid resonance requires HamiltonianH0list to cover the complete static SpinSystem interaction set exactly once." << std::endl;
				return false;
			}
		}
		for (const auto &name : h0list)
		{
			if (_system->interactions_find(name) == nullptr ||
				std::count(h0list.begin(), h0list.end(), name) != 1)
			{
				this->Log() << "Hybrid resonance HamiltonianH0list contains an unknown or duplicate interaction." << std::endl;
				return false;
			}
		}

		SpinAPI::interaction_ptr fieldInteraction = nullptr;
		if (!this->ResolveFieldInteraction(_system, fieldInteraction) ||
			fieldInteraction == nullptr)
		{
			this->Log() << "Hybrid resonance requires a designated Zeeman field interaction." << std::endl;
			return false;
		}

		std::vector<SpinAPI::spin_ptr> detectSpins;
		std::vector<std::string> detectSpinNames;
		if (!this->ResolveDetectionSpins(
				_system, fieldInteraction, detectSpins, detectSpinNames) ||
			detectSpins.empty())
		{
			this->Log() << "Hybrid resonance failed to resolve exact-core EPR detection spins." << std::endl;
			return false;
		}

		std::vector<SpinAPI::interaction_ptr> zeemanInteractions =
			CollectZeemanInteractions(_system, h0list);
		if (zeemanInteractions.empty() ||
			std::find(zeemanInteractions.begin(), zeemanInteractions.end(),
				fieldInteraction) == zeemanInteractions.end())
		{
			this->Log() << "Hybrid resonance requires the designated field interaction and every physical Zeeman term in HamiltonianH0list." << std::endl;
			return false;
		}

		arma::vec Bvec = fieldInteraction->Field();
		if (Bvec.n_elem != 3 || !Bvec.is_finite())
		{
			this->Log() << "Hybrid resonance field interaction does not provide a finite 3-vector." << std::endl;
			return false;
		}

		FieldSyncGuard centerSync;
		if (this->enforceZeemanSync)
		{
			centerSync.Apply(zeemanInteractions, Bvec);
		}
		else
		{
			const double tol = 1.0e-10;
			for (const auto &interaction : zeemanInteractions)
			{
				if (interaction == nullptr)
					return false;
				const arma::vec field = interaction->Field();
				if (field.n_elem != 3 || !field.is_finite() ||
					arma::norm(field - Bvec) > tol)
				{
					this->Log() << "Hybrid resonance requires synchronized Zeeman field vectors; use enforce_zeeman_sync=true or supply identical fields." << std::endl;
					return false;
				}
			}
		}

		// Re-read after optional synchronization.
		Bvec = fieldInteraction->Field();
		const double Bmag = arma::norm(Bvec);
		if (!std::isfinite(Bmag) || Bmag <= 0.0 ||
			this->hybridFieldStepT <= 0.0 ||
			this->hybridFieldStepT >= Bmag)
		{
			this->Log() << "Hybrid resonance field magnitude/finite-difference step is invalid." << std::endl;
			return false;
		}
		const arma::vec fieldDirection = Bvec / Bmag;
		const double field_mT = 1.0e3 * Bmag;

		// R2K-C qualifies two distinct exact-core state policies:
		//   1. a fixed explicit state, preserving R2K-B semantics;
		//   2. an orientation/field-dependent Thermal state generated from an
		//      explicitly exact-core-owned thermal Hamiltonian.
		// Perturbative nuclear factors remain maximally mixed in the first-order
		// independent-factor solver. Molecular-frame explicit-state rotation is
		// still a separate unqualified transformation and therefore fails closed.
		const SpinAPI::StateFrame initialStateFrame =
			_system->InitialStateFrame();
		HybridNuclearResonanceCoreStateMode coreStateMode =
			HybridNuclearResonanceCoreStateMode::ExplicitFixed;
		SpinAPI::state_ptr exactCoreState = nullptr;
		std::vector<SpinAPI::interaction_ptr> exactCoreThermalInteractions;
		double thermalTemperatureK = 300.0;

		if (initialStateFrame == SpinAPI::StateFrame::Fixed)
		{
			if (!this->initialStateName.empty())
			{
				exactCoreState =
					_system->states_find(this->initialStateName);
			}
			else
			{
				const auto initialStates = _system->InitialState();
				if (initialStates.size() == 1 &&
					initialStates.front() != nullptr)
					exactCoreState = initialStates.front();
			}
			if (exactCoreState == nullptr)
			{
				this->Log() << "Hybrid resonance fixed-state mode requires exactly one explicit non-Thermal initial state." << std::endl;
				return false;
			}
		}
		else if (initialStateFrame == SpinAPI::StateFrame::Eigen)
		{
			if (!this->initialStateName.empty())
			{
				this->Log() << "Hybrid resonance frame=eigen requires Thermal to be specified through the SpinSystem initialstate property." << std::endl;
				return false;
			}

			const auto initialStates = _system->InitialState();
			if (initialStates.size() != 1 ||
				initialStates.front() != nullptr)
			{
				this->Log() << "Hybrid resonance frame=eigen requires a single Thermal SpinSystem initial state." << std::endl;
				return false;
			}

			const auto thermalNames =
				_system->ThermalHamiltonianList();
			if (thermalNames.empty())
			{
				this->Log() << "Hybrid resonance R2K-C requires an explicit non-empty thermalhamiltonian list for frame=eigen." << std::endl;
				return false;
			}
			for (const auto &name : thermalNames)
			{
				auto interaction = _system->interactions_find(name);
				if (interaction == nullptr)
				{
					this->Log() << "Hybrid resonance thermal Hamiltonian interaction "" << name << "" was not found in the SpinSystem." << std::endl;
					return false;
				}
				exactCoreThermalInteractions.push_back(interaction);
			}

			thermalTemperatureK = _system->Temperature();
			if (!std::isfinite(thermalTemperatureK) ||
				thermalTemperatureK <= 0.0)
			{
				this->Log() << "Hybrid resonance thermal temperature must be finite and positive." << std::endl;
				return false;
			}

			coreStateMode =
				HybridNuclearResonanceCoreStateMode::ThermalEigen;
			this->Log() << "Hybrid resonance initial state = thermal exact core at "
						<< thermalTemperatureK
						<< " K; perturbative nuclear reference = maximally mixed."
						<< std::endl;
		}
		else
		{
			this->Log() << "Hybrid resonance molecular-frame explicit-state rotation is not yet qualified; use frame=fixed or frame=eigen with Thermal." << std::endl;
			return false;
		}

		std::vector<HybridNuclearResonanceExplicitNucleus> perturbative;
		perturbative.reserve(this->hybridPerturbativeNucleusNames.size());
		for (const auto &name : this->hybridPerturbativeNucleusNames)
		{
			auto nucleus = _system->spins_find(name);
			if (nucleus == nullptr || nucleus->Type() != SpinAPI::SpinType::Nucleus)
			{
				this->Log() << "Hybrid resonance perturbative nucleus \"" << name
							<< "\" is missing or is not SpinType::Nucleus." << std::endl;
				return false;
			}
			HybridNuclearResonanceExplicitNucleus spec;
			spec.nucleus = nucleus;
			spec.overlapThreshold = this->hybridOverlapThreshold;
			spec.fieldIndependentProjection = false;
			perturbative.push_back(std::move(spec));
		}

		std::vector<ResonanceMagneticMomentTerm> detectionTerms;
		detectionTerms.reserve(detectSpins.size());
		for (const auto &spin : detectSpins)
		{
			auto zeeman = FindZeemanForSpin(spin, zeemanInteractions);
			if (zeeman == nullptr && fieldInteraction != nullptr)
			{
				const auto group = fieldInteraction->Group1();
				if (std::find(group.begin(), group.end(), spin) != group.end())
					zeeman = fieldInteraction;
			}
			if (zeeman == nullptr)
			{
				this->Log() << "Hybrid resonance detection spin \"" << spin->Name()
							<< "\" has no owned Zeeman interaction." << std::endl;
				return false;
			}
			detectionTerms.push_back({spin, zeeman});
		}

		HybridNuclearResonanceExplicitPartitionRequest partitionRequest;
		partitionRequest.system = _system;
		partitionRequest.perturbativeNuclei = perturbative;
		partitionRequest.fieldInteractions = zeemanInteractions;
		partitionRequest.detectionTerms = detectionTerms;
		partitionRequest.coreStateMode = coreStateMode;
		partitionRequest.exactCoreState = exactCoreState;
		partitionRequest.exactCoreThermalInteractions =
			exactCoreThermalInteractions;
		partitionRequest.thermalTemperatureK = thermalTemperatureK;
		partitionRequest.fullTensorRotation = this->fullTensorRotation;
		partitionRequest.minimumCumulativeOverlapWeight =
			this->hybridMinimumCumulativeOverlapWeight;
		partitionRequest.maximumComponentsPerCoreTransition =
			this->hybridMaximumComponentsPerCoreTransition;
		// Finite-difference branch tracking is currently qualified only for
		// unmerged center components. Merging remains disabled in this task gate.
		partitionRequest.mergeFrequencyToleranceRadNs = 0.0;

		HybridNuclearResonancePartition partition;
		std::string hybridError;
		if (!HybridNuclearResonancePartitionBuilder::Build(
				partitionRequest, partition, hybridError))
		{
			this->Log() << "Hybrid resonance partition rejected: " << hybridError << std::endl;
			return false;
		}

		// Keep the established task powder/grid semantics. Only line generation
		// changes; the common General spectrum evaluator still owns detuning,
		// field Jacobian use, lineshape, and resolved-channel aggregation.
		int numPoints = this->powdersamplingpoints;
		SpinAPI::PowderGrid grid;
		const bool useSopheGrid = (this->powderGridType == "sophe");
		std::string gridSymmetry = this->powderGridSymmetry;
		if (useSopheGrid)
		{
			std::string symLower = ToLower(gridSymmetry);
			if (symLower.empty() || symLower == "auto" || symLower == "automatic")
			{
				gridSymmetry = AutoDetectSopheSymmetry(
					_system, fieldInteraction, h0list, this->fullTensorRotation);
				this->Log() << "Auto-detected SOPHE grid symmetry: " << gridSymmetry << "." << std::endl;
			}
		}
		if (useSopheGrid)
		{
			int gridSize = this->powderGridSize;
			if (gridSize < 2)
			{
				SpinAPI::SopheGridParameters sopheParams;
				if (numPoints > 1 && SpinAPI::GetSopheGridParameters(gridSymmetry, sopheParams))
				{
					int bestSize = 0;
					int bestDiff = std::numeric_limits<int>::max();
					for (int candidate = 2; candidate <= 200; ++candidate)
					{
						const int count = SpinAPI::SopheGridPointCount(
							candidate, sopheParams.nOctants, sopheParams.closedPhi);
						const int diff = std::abs(count - numPoints);
						if (diff < bestDiff)
						{
							bestDiff = diff;
							bestSize = candidate;
							if (diff == 0)
								break;
						}
					}
					if (bestSize > 0)
						gridSize = bestSize;
				}
				if (gridSize < 2)
					gridSize = 19;
			}
			if (!SpinAPI::CreateSophePowderGrid(gridSize, gridSymmetry, grid))
			{
				this->Log() << "Hybrid resonance failed to obtain SOPHE powder grid." << std::endl;
				return false;
			}
			numPoints = static_cast<int>(grid.size());
		}
		else if (numPoints > 1)
		{
			if (!this->CreateUniformGrid(numPoints, grid))
			{
				this->Log() << "Hybrid resonance failed to obtain uniform powder grid." << std::endl;
				return false;
			}
		}
		else
		{
			grid.clear();
			grid.push_back({0.0, 0.0, 1.0});
			numPoints = 1;
		}

		const int gammaPoints =
			(numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
		const double gammaWeight = useSopheGrid
			? (2.0 * arma::datum::pi / static_cast<double>(gammaPoints))
			: (1.0 / static_cast<double>(gammaPoints));

		SpectrumRequest resonanceRequest;
		resonanceRequest.microwaveFrequencyGHz = this->mwFrequencyGHz;
		resonanceRequest.linewidth_mT = std::abs(this->linewidth_mT);
		resonanceRequest.lineshape =
			(this->lineshape == "lorentzian")
			? Lineshape::Lorentzian
			: Lineshape::Gaussian;
		resonanceRequest.populationThreshold = 1.0e-15;
		resonanceRequest.minimumSlope = 1.0e-15;
		resonanceRequest.maximumDBdOmega = 1.0e5;

		HybridNuclearResonanceFieldResponseRequest fieldResponse;
		fieldResponse.fieldT = Bmag;
		fieldResponse.fieldStepT = this->hybridFieldStepT;
		fieldResponse.minimumCoreStateOverlap =
			this->hybridMinimumCoreStateOverlap;
		fieldResponse.minimumNuclearStateOverlap =
			this->hybridMinimumNuclearStateOverlap;
		fieldResponse.jacobianRelativeTolerance =
			this->hybridJacobianRelativeTolerance;
		fieldResponse.jacobianAbsoluteTolerance =
			this->hybridJacobianAbsoluteTolerance;

		const std::size_t spinCount = detectSpins.size();
		double total_x = 0.0;
		double total_y = 0.0;
		double total_perp = 0.0;
		double cross_x = 0.0;
		double cross_y = 0.0;
		std::vector<double> spin_x(spinCount, 0.0);
		std::vector<double> spin_y(spinCount, 0.0);
		std::vector<double> spin_perp(spinCount, 0.0);
		std::vector<double> spin_p(spinCount, 0.0);
		std::vector<double> spin_m(spinCount, 0.0);

		std::size_t maxProductNuclearDimension = 1;
		std::size_t maxLargestDiagonalizedNuclearDimension = 0;
		double maxDiscardedWeight = 0.0;
		bool anyPruning = false;

		// Sequential on purpose in R2K-B: the finite-difference provider
		// temporarily mutates and restores shared physical Zeeman fields. A later
		// gate may parallelize over cloned immutable field realizations.
		for (int gridIndex = 0; gridIndex < numPoints; ++gridIndex)
		{
			auto [theta, phi, solidWeight] = grid[gridIndex];
			for (int gammaIndex = 0; gammaIndex < gammaPoints; ++gammaIndex)
			{
				const double gamma = (gammaPoints > 1)
					? 2.0 * arma::datum::pi *
						(static_cast<double>(gammaIndex) + 0.5) /
						static_cast<double>(gammaPoints)
					: 0.0;
				const double weight = solidWeight * gammaWeight;

				arma::mat rotation;
				double alphaForRotation = phi;
				double betaForRotation = theta;
				double gammaForRotation = gamma;
				if (!this->CreatePassiveZYZRotationMatrix(
						alphaForRotation, betaForRotation,
						gammaForRotation, rotation))
					return false;

				General::HS::HSOrientation orientation;
				orientation.alpha = phi;
				orientation.beta = theta;
				orientation.gamma = gamma;
				orientation.weight = weight;
				orientation.frameToLab = rotation;

				HybridNuclearResonancePointProvider provider =
					[&](double fieldT,
						HybridNuclearResonancePoint &point,
						std::string &error)
					{
						const arma::vec displacedField = fieldDirection * fieldT;
						FieldSyncGuard displacedSync;
						displacedSync.Apply(zeemanInteractions, displacedField);
						return HybridNuclearResonancePreparation::BuildPoint(
							partition, orientation, fieldT, point, error);
					};

				ResonanceLineSet resonanceLines;
				HybridNuclearResonanceReport report;
				if (!HybridNuclearResonanceSolver::GenerateFirstOrderFiniteDifference(
						provider, fieldResponse, resonanceRequest,
						resonanceLines, report, hybridError))
				{
					this->Log() << "Hybrid resonance finite-difference solver rejected orientation: "
							<< hybridError << std::endl;
					return false;
				}

				SpectrumPoint resonancePoint;
				if (!ResonanceSpectrumEvaluator::Evaluate(
						resonanceLines, resonanceRequest,
						resonancePoint, hybridError))
				{
					this->Log() << "Hybrid resonance spectrum evaluation failed: "
							<< hybridError << std::endl;
					return false;
				}
				if (resonancePoint.channels.size() != spinCount)
				{
					this->Log() << "Hybrid resonance resolved detection-channel cardinality changed." << std::endl;
					return false;
				}

				total_x += weight * resonancePoint.totalX;
				total_y += weight * resonancePoint.totalY;
				total_perp += weight * resonancePoint.totalPerpendicular;
				cross_x += weight * resonancePoint.crossX;
				cross_y += weight * resonancePoint.crossY;
				for (std::size_t i = 0; i < spinCount; ++i)
				{
					spin_x[i] += weight * resonancePoint.channels[i].x;
					spin_y[i] += weight * resonancePoint.channels[i].y;
					spin_perp[i] += weight * resonancePoint.channels[i].perpendicular;
					spin_p[i] += weight * resonancePoint.channels[i].plus;
					spin_m[i] += weight * resonancePoint.channels[i].minus;
				}

				maxProductNuclearDimension = std::max(
					maxProductNuclearDimension, report.productNuclearDimension);
				maxLargestDiagonalizedNuclearDimension = std::max(
					maxLargestDiagonalizedNuclearDimension,
					report.largestDiagonalizedNuclearDimension);
				maxDiscardedWeight = std::max(
					maxDiscardedWeight,
					report.maximumDiscardedNuclearWeightFraction);
				anyPruning = anyPruning || report.pruningApplied;
			}
		}

		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->Log() << "Hybrid resonance explicit partition: "
						<< perturbative.size() << " perturbative nuclei; product nuclear dimension = "
						<< maxProductNuclearDimension
						<< "; largest nuclear diagonalization = "
						<< maxLargestDiagonalizedNuclearDimension
						<< "; max discarded nuclear weight = "
						<< maxDiscardedWeight
						<< "; pruning = " << (anyPruning ? "yes" : "no") << "."
						<< std::endl;
		}

		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->Data() << this->RunSettings()->Time() << " ";
		this->WriteStandardOutput(this->Data());
		this->Data() << field_mT << " "
					 << total_x << " " << total_y << " " << total_perp << " "
					 << cross_x << " " << cross_y << " ";
		for (std::size_t i = 0; i < detectSpinNames.size(); ++i)
		{
			this->Data() << spin_x[i] << " " << spin_y[i] << " "
						 << spin_perp[i] << " " << spin_p[i] << " "
						 << spin_m[i] << " ";
		}
		this->Data() << std::endl;
		return true;
	}

	void TaskStaticHSResonanceSpectra::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		_stream << "Time ";
		this->WriteStandardOutputHeader(_stream);

		auto systems = this->SpinSystems();
		for (auto i = systems.cbegin(); i != systems.cend(); i++)
		{
			SpinAPI::interaction_ptr fieldInteraction = nullptr;
			std::vector<SpinAPI::spin_ptr> detectSpins;
			std::vector<std::string> detectSpinNames;
			this->ResolveFieldInteraction((*i), fieldInteraction);
			if (!this->ResolveDetectionSpins((*i), fieldInteraction, detectSpins, detectSpinNames))
			{
				detectSpinNames.clear();
			}

			_stream << (*i)->Name() << ".Field_mT ";
			_stream << (*i)->Name() << ".Total_x ";
			_stream << (*i)->Name() << ".Total_y ";
			_stream << (*i)->Name() << ".Total_perp ";
			_stream << (*i)->Name() << ".Cross_x ";
			_stream << (*i)->Name() << ".Cross_y ";

			for (const auto &spinName : detectSpinNames)
			{
				_stream << (*i)->Name() << "." << spinName << "_x ";
				_stream << (*i)->Name() << "." << spinName << "_y ";
				_stream << (*i)->Name() << "." << spinName << "_perp ";
				_stream << (*i)->Name() << "." << spinName << "_p ";
				_stream << (*i)->Name() << "." << spinName << "_m ";
			}
		}

		_stream << std::endl;
	}

	bool TaskStaticHSResonanceSpectra::Validate()
	{
		bool hasFrequency = this->Properties()->Get("mwfrequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			hasFrequency = this->Properties()->Get("frequency", this->mwFrequencyGHz);
		if (!hasFrequency)
		{
			this->Log() << "Failed to obtain mwfrequency/frequency. Using frequency = 0 by default." << std::endl;
		}

		if (!this->Properties()->Get("linewidth", this->linewidth_mT))
		{
			this->Log() << "Failed to obtain linewidth. Using linewidth = 0 by default." << std::endl;
			this->linewidth_mT = 0.0;
		}

		if (this->Properties()->Get("lineshape", this->lineshape))
		{
			this->lineshape = ToLower(this->lineshape);
		}
		else
		{
			this->lineshape = "gaussian";
		}


		// Harmonic post-processing models field-modulated detection after the
		// absorption spectrum has been assembled on the sweep cache.
		if (!this->Properties()->Get("harmonic", this->detectionHarmonic) &&
			!this->Properties()->Get("detectionharmonic", this->detectionHarmonic) &&
			!this->Properties()->Get("detection_harmonic", this->detectionHarmonic))
		{
			this->detectionHarmonic = 0;
		}
		if (this->detectionHarmonic < 0)
		{
			this->Log() << "Negative harmonic values are not supported here. Using harmonic = 0." << std::endl;
			this->detectionHarmonic = 0;
		}
		if (this->detectionHarmonic > 2)
		{
			this->Log() << "Only harmonic = 0, 1, or 2 is supported. Using harmonic = 2." << std::endl;
			this->detectionHarmonic = 2;
		}

		if (!this->Properties()->Get("modamp", this->modulationAmplitude_mT) &&
			!this->Properties()->Get("modulationamplitude", this->modulationAmplitude_mT) &&
			!this->Properties()->Get("modulation_amplitude", this->modulationAmplitude_mT) &&
			!this->Properties()->Get("fieldmodulation", this->modulationAmplitude_mT))
		{
			this->modulationAmplitude_mT = 0.0;
		}
		if (this->modulationAmplitude_mT < 0.0)
		{
			this->Log() << "Negative modulation amplitudes are not supported. Using modulation amplitude = 0 mT." << std::endl;
			this->modulationAmplitude_mT = 0.0;
		}


		if (!this->Properties()->Get("powdersamplingpoints", this->powdersamplingpoints))
		{
			this->powdersamplingpoints = 0;
		}

		if (!this->Properties()->Get("sweepcache", this->useSweepCache) &&
			!this->Properties()->Get("cache_sweep", this->useSweepCache) &&
			!this->Properties()->Get("sweep_cache", this->useSweepCache))
		{
			this->useSweepCache = true;
		}


		if (this->detectionHarmonic > 0 && !this->useSweepCache)
		{
			this->Log() << "Detection harmonic post-processing requires the sweep cache. Enabling sweepcache=true." << std::endl;
			this->useSweepCache = true;
		}

		std::string sweepCacheMode;
		if (this->Properties()->Get("sweepcachemode", sweepCacheMode) ||
			this->Properties()->Get("sweep_cache_mode", sweepCacheMode) ||
			this->Properties()->Get("cache_sweep_mode", sweepCacheMode))
		{
			sweepCacheMode = ToLower(sweepCacheMode);
			if (sweepCacheMode == "exact" || sweepCacheMode == "direct" || sweepCacheMode == "matrix")
			{
				this->sweepCacheExact = true;
				this->sweepCacheResfields = false;
			}
			else if (sweepCacheMode == "resonanceprojection" || sweepCacheMode == "projection" ||
					 sweepCacheMode == "projectedresfields" || sweepCacheMode == "resfields" ||
					 sweepCacheMode == "resfield")
			{
				this->sweepCacheExact = false;
				this->sweepCacheResfields = true;
			}
			else if (sweepCacheMode == "approx" || sweepCacheMode == "approximate" || sweepCacheMode == "crossing" ||
					 sweepCacheMode == "resonance")
			{
				this->sweepCacheExact = false;
				this->sweepCacheResfields = false;
			}
			else
			{
				this->Log() << "Unknown sweepcachemode \"" << sweepCacheMode << "\". Using "
							<< (this->sweepCacheExact ? "exact" : (this->sweepCacheResfields ? "resonanceprojection" : "approx")) << "." << std::endl;
			}
		}

		int resfieldPoints = 0;
		if (this->Properties()->Get("resfieldspoints", resfieldPoints) ||
			this->Properties()->Get("resfields_points", resfieldPoints) ||
			this->Properties()->Get("sweepcachepoints", resfieldPoints) ||
			this->Properties()->Get("sweep_cache_points", resfieldPoints))
		{
			if (resfieldPoints >= 2)
				this->sweepCacheResfieldPoints = resfieldPoints;
			else
				this->sweepCacheResfieldPoints = 0;
		}

		if (this->Properties()->Get("powdergridtype", this->powderGridType))
		{
			this->powderGridType = ToLower(this->powderGridType);
		}
		else
		{
			this->powderGridType = "sophe";
		}
		this->Properties()->Get("powdergridsymmetry", this->powderGridSymmetry);
		if (!this->Properties()->Get("powdergridsize", this->powderGridSize))
		{
			this->powderGridSize = 0;
		}

		if (!this->Properties()->Get("powdergammapoints", this->powderGammaPoints))
		{
			this->Properties()->Get("powdergammastps", this->powderGammaPoints);
		}
		if (this->powderGammaPoints < 1)
		{
			this->powderGammaPoints = 1;
		}

		this->Properties()->Get("powderfullsphere", this->powderFullSphere);
		this->Properties()->Get("fulltensorrotation", this->fullTensorRotation);
		this->Properties()->Get("mzblocks", this->useMzBlocks);

		this->resonanceSolverMode = "exact";
		std::string resonanceSolver;
		if (this->Properties()->Get("solver", resonanceSolver) ||
			this->Properties()->Get("resonancesolver", resonanceSolver) ||
			this->Properties()->Get("resonance_solver", resonanceSolver))
		{
			resonanceSolver = ToLower(resonanceSolver);
			if (resonanceSolver == "exact")
				this->resonanceSolverMode = "exact";
			else if (resonanceSolver == "hybrid")
				this->resonanceSolverMode = "hybrid";
			else if (resonanceSolver == "auto")
			{
				this->Log() << "solver=auto is not yet qualified for StaticHS-Resonance-Spectra; select exact or explicit hybrid." << std::endl;
				return false;
			}
			else
			{
				this->Log() << "Unknown resonance solver \"" << resonanceSolver << "\"." << std::endl;
				return false;
			}
		}

		this->hybridPerturbativeNucleusNames.clear();
		if (!this->Properties()->GetList("perturbativenuclei", this->hybridPerturbativeNucleusNames, ',') &&
			!this->Properties()->GetList("hybridperturbativenuclei", this->hybridPerturbativeNucleusNames, ','))
		{
			this->Properties()->GetList("hybrid_perturbative_nuclei", this->hybridPerturbativeNucleusNames, ',');
		}

		if (!this->Properties()->Get("hybridfieldstep", this->hybridFieldStepT))
			this->Properties()->Get("hybrid_field_step", this->hybridFieldStepT);
		if (!this->Properties()->Get("hybridminimumcorestateoverlap", this->hybridMinimumCoreStateOverlap))
			this->Properties()->Get("hybrid_minimum_core_state_overlap", this->hybridMinimumCoreStateOverlap);
		if (!this->Properties()->Get("hybridminimumnuclearstateoverlap", this->hybridMinimumNuclearStateOverlap))
			this->Properties()->Get("hybrid_minimum_nuclear_state_overlap", this->hybridMinimumNuclearStateOverlap);
		if (!this->Properties()->Get("hybridjacobianreltol", this->hybridJacobianRelativeTolerance))
			this->Properties()->Get("hybrid_jacobian_relative_tolerance", this->hybridJacobianRelativeTolerance);
		if (!this->Properties()->Get("hybridjacobianabstol", this->hybridJacobianAbsoluteTolerance))
			this->Properties()->Get("hybrid_jacobian_absolute_tolerance", this->hybridJacobianAbsoluteTolerance);
		if (!this->Properties()->Get("hybridoverlapthreshold", this->hybridOverlapThreshold))
			this->Properties()->Get("hybrid_overlap_threshold", this->hybridOverlapThreshold);
		if (!this->Properties()->Get("hybridminimumcumulativeoverlapweight", this->hybridMinimumCumulativeOverlapWeight))
			this->Properties()->Get("hybrid_minimum_cumulative_overlap_weight", this->hybridMinimumCumulativeOverlapWeight);

		int hybridMaximumComponents = 0;
		if (this->Properties()->Get("hybridmaximumcomponentspercoretransition", hybridMaximumComponents) ||
			this->Properties()->Get("hybrid_maximum_components_per_core_transition", hybridMaximumComponents))
		{
			if (hybridMaximumComponents < 0)
			{
				this->Log() << "Hybrid maximum component count must be non-negative." << std::endl;
				return false;
			}
			this->hybridMaximumComponentsPerCoreTransition =
				static_cast<std::size_t>(hybridMaximumComponents);
		}

		if (this->resonanceSolverMode == "hybrid")
		{
			if (this->hybridPerturbativeNucleusNames.empty())
			{
				this->Log() << "solver=hybrid requires an explicit perturbativenuclei list." << std::endl;
				return false;
			}
			if (this->useSweepCache)
			{
				this->Log() << "solver=hybrid R2K-B requires sweepcache=false; hybrid cache semantics are not yet qualified." << std::endl;
				return false;
			}
			if (this->detectionHarmonic != 0)
			{
				this->Log() << "solver=hybrid R2K-B supports harmonic=0 only." << std::endl;
				return false;
			}
			if (!std::isfinite(this->hybridFieldStepT) || this->hybridFieldStepT <= 0.0 ||
				!std::isfinite(this->hybridMinimumCoreStateOverlap) ||
				this->hybridMinimumCoreStateOverlap <= 0.0 || this->hybridMinimumCoreStateOverlap > 1.0 ||
				!std::isfinite(this->hybridMinimumNuclearStateOverlap) ||
				this->hybridMinimumNuclearStateOverlap <= 0.0 || this->hybridMinimumNuclearStateOverlap > 1.0 ||
				!std::isfinite(this->hybridJacobianRelativeTolerance) || this->hybridJacobianRelativeTolerance < 0.0 ||
				!std::isfinite(this->hybridJacobianAbsoluteTolerance) || this->hybridJacobianAbsoluteTolerance < 0.0 ||
				!std::isfinite(this->hybridOverlapThreshold) || this->hybridOverlapThreshold < 0.0 || this->hybridOverlapThreshold > 1.0 ||
				!std::isfinite(this->hybridMinimumCumulativeOverlapWeight) ||
				this->hybridMinimumCumulativeOverlapWeight < 0.0 || this->hybridMinimumCumulativeOverlapWeight > 1.0)
			{
				this->Log() << "Invalid explicit hybrid resonance numerical controls." << std::endl;
				return false;
			}
			this->Log() << "General Resonance solver = explicit hybrid nuclear treatment." << std::endl;
		}

		this->Log() << "Full-Hamiltonian resonance detection model: mwfrequency = " << this->mwFrequencyGHz
					<< " GHz, linewidth = " << this->linewidth_mT
					<< " mT, lineshape = " << this->lineshape
					<< ", harmonic = " << this->detectionHarmonic
					<< ", modulation amplitude = " << this->modulationAmplitude_mT << " mT." << std::endl;
		this->Log() << "Full-Hamiltonian resonance powder grid request: type = " << this->powderGridType
					<< ", symmetry = " << this->powderGridSymmetry
					<< ", sampling points = " << this->powdersamplingpoints
					<< ", gamma points = " << this->powderGammaPoints
					<< ", full sphere = " << (this->powderFullSphere ? "true" : "false") << "." << std::endl;

		this->detectSpinNames.clear();
		this->Properties()->GetList("detectspins", this->detectSpinNames, ',');

		this->Properties()->Get("fieldinteraction", this->fieldInteractionName);
		this->Properties()->Get("enforce_zeeman_sync", this->enforceZeemanSync);
		this->Properties()->Get("enforcezeemansync", this->enforceZeemanSync);
		this->Properties()->Get("initialstate", this->initialStateName);

		if (this->Properties()->GetList("hamiltonianh0list", this->hamiltonianH0list, ','))
		{
			this->Log() << "HamiltonianH0list = [";
			for (size_t j = 0; j < this->hamiltonianH0list.size(); j++)
			{
				this->Log() << this->hamiltonianH0list[j];
				if (j < this->hamiltonianH0list.size() - 1)
					this->Log() << ", ";
			}
			this->Log() << "]" << std::endl;
		}

		return true;
	}


	// -----------------------------------------------------
	// Task-specific helper methods
	// -----------------------------------------------------
	double TaskStaticHSResonanceSpectra::LineshapeValue(double _delta, double _fwhm) const
	{
		if (!std::isfinite(_delta) || !std::isfinite(_fwhm))
			return 0.0;

		if (_fwhm <= 0.0)
		{
			return (std::abs(_delta) < 1e-12) ? 1.0 : 0.0;
		}

		const double x = _delta / _fwhm;
		if (this->lineshape == "lorentzian")
		{
			const double gamma = 0.5 * _fwhm;
			return (1.0 / arma::datum::pi) * (gamma / (_delta * _delta + gamma * gamma));
		}

		const double pref = std::sqrt(4.0 * std::log(2.0) / arma::datum::pi) / _fwhm;
		return pref * std::exp(-4.0 * std::log(2.0) * x * x);
	}


	std::vector<double> TaskStaticHSResonanceSpectra::ApplyFieldHarmonic(const std::vector<double> &_field_mT, const std::vector<double> &_channel) const
	{
		if (this->detectionHarmonic <= 0 || _field_mT.size() != _channel.size() || _field_mT.size() < 3)
			return _channel;

		double meanStep = 0.0;
		size_t stepCount = 0;
		for (size_t i = 1; i < _field_mT.size(); ++i)
		{
			const double dx = std::abs(_field_mT[i] - _field_mT[i - 1]);
			if (std::isfinite(dx) && dx > 0.0)
			{
				meanStep += dx;
				++stepCount;
			}
		}
		if (stepCount == 0)
			return _channel;
		meanStep /= static_cast<double>(stepCount);

		const double span = (this->modulationAmplitude_mT > 0.0) ? this->modulationAmplitude_mT : meanStep;
		const double h = (this->modulationAmplitude_mT > 0.0) ? (0.5 * span) : span;
		if (!std::isfinite(h) || h <= 0.0)
			return _channel;

		const bool ascending = (_field_mT.back() >= _field_mT.front());
		auto interp = [&](double x) -> double
		{
			if (ascending)
			{
				if (x <= _field_mT.front())
					return _channel.front();
				if (x >= _field_mT.back())
					return _channel.back();
				auto it = std::lower_bound(_field_mT.begin(), _field_mT.end(), x);
				const size_t hi = static_cast<size_t>(std::distance(_field_mT.begin(), it));
				const size_t lo = hi - 1;
				const double denom = _field_mT[hi] - _field_mT[lo];
				if (std::abs(denom) <= 0.0)
					return _channel[lo];
				const double t = (x - _field_mT[lo]) / denom;
				return (1.0 - t) * _channel[lo] + t * _channel[hi];
			}

			if (x >= _field_mT.front())
				return _channel.front();
			if (x <= _field_mT.back())
				return _channel.back();
			auto it = std::lower_bound(_field_mT.begin(), _field_mT.end(), x, std::greater<double>());
			const size_t hi = static_cast<size_t>(std::distance(_field_mT.begin(), it));
			const size_t lo = hi - 1;
			const double denom = _field_mT[hi] - _field_mT[lo];
			if (std::abs(denom) <= 0.0)
				return _channel[lo];
			const double t = (x - _field_mT[lo]) / denom;
			return (1.0 - t) * _channel[lo] + t * _channel[hi];
		};

		std::vector<double> out(_channel.size(), 0.0);
		for (size_t i = 0; i < _channel.size(); ++i)
		{
			const double x = _field_mT[i];
			const double ym = interp(x - h);
			const double y0 = _channel[i];
			const double yp = interp(x + h);

			if (this->detectionHarmonic == 1)
				out[i] = (yp - ym) / (2.0 * h);
			else
				out[i] = (yp - 2.0 * y0 + ym) / (h * h);
		}

		return out;
	}

	void TaskStaticHSResonanceSpectra::ApplyDetectionHarmonic(SpectrumCache &_cache) const
	{
		if (this->detectionHarmonic <= 0)
			return;

		auto apply = [&](std::vector<double> &channel)
		{
			channel = this->ApplyFieldHarmonic(_cache.field_mT, channel);
		};

		apply(_cache.total_x);
		apply(_cache.total_y);
		apply(_cache.total_perp);
		apply(_cache.cross_x);
		apply(_cache.cross_y);
		for (size_t i = 0; i < _cache.spin_names.size(); ++i)
		{
			apply(_cache.spin_x[i]);
			apply(_cache.spin_y[i]);
			apply(_cache.spin_perp[i]);
			apply(_cache.spin_p[i]);
			apply(_cache.spin_m[i]);
		}
	}


	bool TaskStaticHSResonanceSpectra::CreatePassiveZYZRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		// EasySpin convention: passive ZYZ Euler rotation (molecular -> lab).
		const double ca = std::cos(_alpha), sa = std::sin(_alpha);
		const double cb = std::cos(_beta), sb = std::sin(_beta);
		const double cg = std::cos(_gamma), sg = std::sin(_gamma);

		arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
		arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
		arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};

		_R = Rg * Rb * Ra;
		return true;
	}

	bool TaskStaticHSResonanceSpectra::CreateUniformGrid(int &_Npoints, SpinAPI::PowderGrid &_uniformGrid) const
	{
		const auto domain = this->powderFullSphere ? SpinAPI::PowderGridDomain::FullSphere
												   : SpinAPI::PowderGridDomain::UpperHemisphere;
		return SpinAPI::CreateUniformPowderGrid(_Npoints, domain, _uniformGrid);
	}

	bool TaskStaticHSResonanceSpectra::ResolveFieldInteraction(const SpinAPI::system_ptr &_system, SpinAPI::interaction_ptr &_fieldInteraction) const
	{
		_fieldInteraction = nullptr;
		if (_system == nullptr)
			return false;

		if (!this->fieldInteractionName.empty())
			_fieldInteraction = _system->interactions_find(this->fieldInteractionName);

		if (_fieldInteraction == nullptr)
		{
			for (auto inter = _system->interactions_cbegin(); inter != _system->interactions_cend(); inter++)
			{
				std::string type;
				if ((*inter)->Properties()->Get("type", type))
				{
					type = ToLower(type);
					if (type == "zeeman")
					{
						_fieldInteraction = (*inter);
						break;
					}
				}
			}
		}

		return (_fieldInteraction != nullptr);
	}

	bool TaskStaticHSResonanceSpectra::ResolveDetectionSpins(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction,
														 std::vector<SpinAPI::spin_ptr> &_spins, std::vector<std::string> &_spinNames) const
	{
		_spins.clear();
		_spinNames.clear();
		if (_system == nullptr)
			return false;

		auto add_spin = [&](const SpinAPI::spin_ptr &spin) -> bool
		{
			if (spin == nullptr)
				return false;
			for (const auto &existing : _spins)
			{
				if (existing == spin)
					return true;
			}
			_spins.push_back(spin);
			return true;
		};

		// Detection spins define which magnetic dipole operators contribute to the
		// reported per-spin channels. If the user does not specify them explicitly,
		// we fall back to the Zeeman interaction and finally to all spins.
		if (!this->detectSpinNames.empty())
		{
			for (const auto &name : this->detectSpinNames)
			{
				auto spin = _system->spins_find(name);
				if (spin == nullptr)
					return false;
				add_spin(spin);
			}
		}
		else if (_fieldInteraction != nullptr)
		{
			auto group = _fieldInteraction->Group1();
			for (const auto &spin : group)
			{
				add_spin(spin);
			}
		}

		if (_spins.empty())
		{
			auto allSpins = _system->Spins();
			for (const auto &spin : allSpins)
				add_spin(spin);
		}

		if (_spins.empty())
			return false;

		for (const auto &spin : _spins)
			_spinNames.push_back(spin->Name());

		return true;
	}

	bool TaskStaticHSResonanceSpectra::GetLinearFieldSweep(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0, arma::vec &_fieldStep) const
	{
		if (_system == nullptr || _fieldInteraction == nullptr)
			return false;

		_field0 = _fieldInteraction->Field();
		if (_field0.n_elem != 3 || !_field0.is_finite())
			return false;

		const std::string target = _system->Name() + "." + _fieldInteraction->Name() + ".field";

		std::map<std::string, arma::vec> stepsByTarget;
		std::string error;
		if (!CollectAddVectorSteps(this->Actions(), this->RunSettings()->Steps(), stepsByTarget, error))
			return false;

		auto it = stepsByTarget.find(target);
		if (it == stepsByTarget.end())
			return false;

		_fieldStep = it->second;
		if (arma::norm(_field0) > 0.0 && arma::norm(_fieldStep) > 0.0)
		{
			arma::vec dir0 = arma::normalise(_field0);
			arma::vec dirStep = arma::normalise(_fieldStep);
			if (arma::norm(arma::cross(dir0, dirStep)) > 1e-6)
				return false;
		}

		return true;
	}

	bool TaskStaticHSResonanceSpectra::BuildCachedSpectrum(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, const arma::vec &_field0, const arma::vec &_fieldStep, SpectrumCache &_cache)
	{
		// Cached sweep workflow:
		// - build the field axis once from the AddVector sweep,
		// - separate field-dependent Zeeman terms from the static Hamiltonian,
		// - evaluate transition moments for each powder orientation,
		// - deposit the resonances either directly or through the projection mesh.
		if (_system == nullptr || _fieldInteraction == nullptr)
			return false;

		const unsigned int steps = this->RunSettings()->Steps();
		if (steps < 2)
			return false;

		if (_field0.n_elem != 3 || !_field0.is_finite())
			return false;

		std::vector<double> field_T(steps, 0.0);
		_cache.field_mT.assign(steps, 0.0);
		_cache.total_x.assign(steps, 0.0);
		_cache.total_y.assign(steps, 0.0);
		_cache.total_perp.assign(steps, 0.0);
		_cache.cross_x.assign(steps, 0.0);
		_cache.cross_y.assign(steps, 0.0);
		_cache.steps = steps;

		arma::vec Bvec = _field0;
		for (unsigned int i = 0; i < steps; ++i)
		{
			const double Bmag = arma::norm(Bvec);
			if (!std::isfinite(Bmag) || Bmag <= 0.0)
				return false;
			field_T[i] = Bmag;
			_cache.field_mT[i] = 1.0e3 * Bmag;
			Bvec += _fieldStep;
		}

		const bool useResfieldsCache = this->sweepCacheResfields;
		const bool useApproxCache = (!this->sweepCacheExact && !useResfieldsCache);
		// Cache modes:
		// - exact: diagonalize at every output field point.
		// - approx: locate resonances on a coarser scan and broaden them directly.
		// - resonanceprojection: locate resonance fields first and, when possible,
		//   project the orientation mesh continuously onto the output field axis.
		const double dBstep = (steps > 1) ? (field_T[1] - field_T[0]) : 0.0;
		const double dBabs = std::abs(dBstep);

		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		SpinAPI::SpinSpace space(*_system);
		space.UseSuperoperatorSpace(false);
		space.UseFullTensorRotation(this->fullTensorRotation);
		const arma::uword spaceDim = space.HilbertSpaceDimensions();
		const auto allSpins = _system->Spins();
		const MzBlocks mzBlocks = BuildMzBlocks(allSpins);
		const bool hasMzBlocks = (this->useMzBlocks && !mzBlocks.mz2.empty() && mzBlocks.mz2.size() == static_cast<size_t>(spaceDim) && mzBlocks.blocks.size() > 1);

		std::vector<std::string> h0list = this->hamiltonianH0list;
		if (h0list.empty())
		{
			for (const auto &interaction : _system->Interactions())
			{
				if (!SpinAPI::IsStatic(*interaction))
					continue;
				h0list.push_back(interaction->Name());
			}
		}

		if (h0list.empty())
			return false;

		std::vector<SpinAPI::interaction_ptr> zeemanInteractions = CollectZeemanInteractions(_system, h0list);
		if (zeemanInteractions.empty() && _fieldInteraction != nullptr)
			zeemanInteractions.push_back(_fieldInteraction);

		auto isZeemanName = [&](const std::string &name) -> bool {
			for (const auto &inter : zeemanInteractions)
			{
				if (inter != nullptr && inter->Name() == name)
					return true;
			}
			return false;
		};

		std::vector<std::string> h0list_noB;
		h0list_noB.reserve(h0list.size());
		for (const auto &name : h0list)
		{
			if (!isZeemanName(name))
				h0list_noB.push_back(name);
		}

		std::vector<SpinAPI::spin_ptr> detectSpins;
		std::vector<std::string> detectSpinNames;
		if (!this->ResolveDetectionSpins(_system, _fieldInteraction, detectSpins, detectSpinNames))
			return false;
		if (detectSpins.empty())
			return false;
		const OrientationDiagnostics orientationDebug = OrientationDiagnostics::FromProperties(*this->Properties());
		if (orientationDebug.enabled)
			orientationDebug.InitialiseFile(_system->Name(), detectSpinNames);
		std::ofstream debugOut;
		if (orientationDebug.enabled)
			debugOut.open(orientationDebug.file, std::ios::out | std::ios::app);
		_cache.spin_names = detectSpinNames;
		_cache.spin_x.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_y.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_perp.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_p.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_m.assign(detectSpins.size(), std::vector<double>(steps, 0.0));

		// Prepare the same initial-state machinery used in the direct path. The
		// cache only changes how the field sweep is sampled, not the underlying
		// quantum-mechanical intensities.
		const SpinAPI::StateFrame initialStateFrame = _system->InitialStateFrame();
		arma::cx_mat rho0;
		bool hasInitialState = false;
		bool useOrientationThermal = false;
		std::vector<std::string> thermalhamiltonian_list;
		double thermalTemperature = 0.0;
		if (initialStateFrame == SpinAPI::StateFrame::Eigen)
		{
			if (!this->initialStateName.empty())
				return false;

			auto initial_states = _system->InitialState();
			if (initial_states.size() != 1 || initial_states.front() != nullptr)
				return false;

			useOrientationThermal = true;
			thermalhamiltonian_list = _system->ThermalHamiltonianList();
			thermalTemperature = _system->Temperature();
		}
		else if (!this->initialStateName.empty())
		{
			auto state = _system->states_find(this->initialStateName);
			if (state != nullptr && space.GetState(state, rho0))
				hasInitialState = true;
		}
		if (!hasInitialState && !useOrientationThermal)
		{
			auto initial_states = _system->InitialState();
			if (initial_states.empty())
				return false;

			std::vector<double> initial_weights = _system->Weights();
			const bool useInitialWeights = (initial_weights.size() == initial_states.size());
			if (useInitialWeights)
			{
				double sum_weights = std::accumulate(initial_weights.begin(), initial_weights.end(), 0.0);
				if (sum_weights > 0.0)
				{
					for (double &weight : initial_weights)
						weight /= sum_weights;
				}
			}

			for (size_t stateIndex = 0; stateIndex < initial_states.size(); ++stateIndex)
			{
				auto state = initial_states.cbegin() + static_cast<std::ptrdiff_t>(stateIndex);
				if ((*state) == nullptr)
					continue;

				arma::cx_mat tmp;
				if (!space.GetState(*state, tmp))
					continue;

				if (useInitialWeights)
					tmp *= initial_weights[stateIndex];

				if (!hasInitialState)
				{
					rho0 = tmp;
					hasInitialState = true;
				}
				else
				{
					rho0 += tmp;
				}
			}
		}

		if (!hasInitialState && !useOrientationThermal)
			return false;
		if (!useOrientationThermal)
			rho0 /= arma::trace(rho0);

		std::vector<arma::cx_mat> Sx_list(detectSpins.size());
		std::vector<arma::cx_mat> Sy_list(detectSpins.size());
		std::vector<arma::cx_mat> Sz_list(detectSpins.size());
		for (size_t i = 0; i < detectSpins.size(); ++i)
		{
			auto spin = detectSpins[i];
			if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx_list[i]) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy_list[i]) ||
				!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz_list[i]))
			{
				return false;
			}
		}

		int numPoints = this->powdersamplingpoints;
		SpinAPI::PowderGrid grid;
		const bool useSopheGrid = (this->powderGridType == "sophe");
		std::string gridSymmetry = this->powderGridSymmetry;
		SpinAPI::SopheGridParameters sopheParams;
		bool haveSopheParams = false;
		int sopheGridSize = 0;
		if (useSopheGrid)
		{
			std::string symLower = ToLower(gridSymmetry);
			if (symLower.empty() || symLower == "auto" || symLower == "automatic")
			{
				gridSymmetry = AutoDetectSopheSymmetry(_system, _fieldInteraction, h0list, this->fullTensorRotation);
				this->Log() << "Auto-detected SOPHE grid symmetry: " << gridSymmetry << "." << std::endl;
			}
			haveSopheParams = SpinAPI::GetSopheGridParameters(gridSymmetry, sopheParams);
		}
		if (useSopheGrid)
		{
			int gridSize = this->powderGridSize;
			if (gridSize < 2)
			{
				if (numPoints > 1 && haveSopheParams)
				{
					int bestSize = 0;
					int bestDiff = std::numeric_limits<int>::max();
					for (int candidate = 2; candidate <= 200; ++candidate)
					{
						int count = SpinAPI::SopheGridPointCount(candidate, sopheParams.nOctants, sopheParams.closedPhi);
						int diff = std::abs(count - numPoints);
						if (diff < bestDiff)
						{
							bestDiff = diff;
							bestSize = candidate;
							if (diff == 0)
								break;
						}
					}
					if (bestSize > 0)
						gridSize = bestSize;
				}
				if (gridSize < 2)
					gridSize = 19;
			}

			if (!SpinAPI::CreateSophePowderGrid(gridSize, gridSymmetry, grid))
				return false;
			numPoints = static_cast<int>(grid.size());
			sopheGridSize = gridSize;
		}
		else if (numPoints > 1)
		{
			if (!this->CreateUniformGrid(numPoints, grid))
				return false;
		}
		else
		{
			grid.clear();
			grid.push_back({0.0, 0.0, 1.0});
			numPoints = 1;
		}
		const int gamma_points = (numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
		const double gamma_weight = useSopheGrid ? (2.0 * arma::datum::pi / static_cast<double>(gamma_points))
												 : (1.0 / static_cast<double>(gamma_points));

		const double lwB_mT = std::abs(this->linewidth_mT);
		General::Resonance::SpectrumRequest resonanceRequest;
		resonanceRequest.microwaveFrequencyGHz = this->mwFrequencyGHz;
		resonanceRequest.linewidth_mT = lwB_mT;
		resonanceRequest.lineshape =
			(this->lineshape == "lorentzian")
			? General::Resonance::Lineshape::Lorentzian
			: General::Resonance::Lineshape::Gaussian;
		resonanceRequest.populationThreshold = 1.0e-15;
		resonanceRequest.minimumSlope = 1.0e-15;
		resonanceRequest.maximumDBdOmega = 1.0e5;

		double lineWindow = 0.0;
		if (useApproxCache || useResfieldsCache)
		{
			const double lwB_T = lwB_mT * 1.0e-3;
			lineWindow = (lwB_T > 0.0 && dBabs > 0.0) ? 6.0 * lwB_T : 0.0;
		}

		// Determine Zeeman interaction for each detection spin (frame + prefactor).
		std::vector<SpinAPI::interaction_ptr> spinZeeman(detectSpins.size(), nullptr);
		for (size_t i = 0; i < detectSpins.size(); ++i)
		{
			spinZeeman[i] = FindZeemanForSpin(detectSpins[i], zeemanInteractions);
			if (spinZeeman[i] == nullptr)
				spinZeeman[i] = _fieldInteraction;
		}

		// Base tensors (as specified on spins), rotated from tensor frame to molecular frame.
		std::vector<arma::mat> g_frame_base(detectSpins.size());
		std::vector<double> mu_prefactors(detectSpins.size(), 1.0);
		for (size_t i = 0; i < detectSpins.size(); ++i)
		{
			arma::mat g_base = arma::conv_to<arma::mat>::from(detectSpins[i]->GetTensor().LabFrame());
			if (spinZeeman[i] != nullptr && spinZeeman[i]->IgnoreTensors())
				g_base = arma::eye<arma::mat>(3, 3);

			arma::mat RFrame = arma::eye<arma::mat>(3, 3);
			if (spinZeeman[i] != nullptr)
				RFrame = PassiveZYZRotation(spinZeeman[i]->Framelist());
			const arma::mat RFrame_T2M = RFrame.t();
			g_frame_base[i] = RFrame_T2M * g_base * RFrame_T2M.t();

			if (spinZeeman[i] != nullptr)
			{
				double mu = spinZeeman[i]->Prefactor();
				if (spinZeeman[i]->AddCommonPrefactor())
					mu *= 8.79410005e+1;
				mu_prefactors[i] = mu;
			}
		}

		std::vector<std::string> zeelist;
		zeelist.reserve(zeemanInteractions.size());
		for (const auto &inter : zeemanInteractions)
			zeelist.push_back(inter->Name());

		const arma::uword dim = space.HilbertSpaceDimensions();
		std::vector<std::pair<arma::uword, arma::uword>> transitions;
		transitions.reserve(static_cast<size_t>(dim) * static_cast<size_t>(dim - 1) / 2);
		for (arma::uword m = 0; m < dim; ++m)
		{
			for (arma::uword n = m + 1; n < dim; ++n)
				transitions.emplace_back(m, n);
		}

		const arma::cx_double I(0.0, 1.0);
		const size_t spin_count = detectSpins.size();
		const bool projectionCandidate = useSopheGrid && haveSopheParams && numPoints > 1 && lwB_mT <= 0.0 && (useApproxCache || useResfieldsCache);
		SpinAPI::PowderProjectionMesh projectionMesh;
		bool projectionPossible = false;
		size_t projectionSamples = 0;
		std::vector<unsigned int> projectionCounts;
		std::vector<double> projectionPositions;
		std::vector<double> projectionTotalX;
		std::vector<double> projectionTotalY;
		std::vector<double> projectionTotalPerp;
		std::vector<double> projectionCrossX;
		std::vector<double> projectionCrossY;
		std::vector<std::vector<double>> projectionSpinX;
		std::vector<std::vector<double>> projectionSpinY;
		std::vector<std::vector<double>> projectionSpinPerp;
		std::vector<std::vector<double>> projectionSpinP;
		std::vector<std::vector<double>> projectionSpinM;
		if (projectionCandidate)
		{
			projectionPossible = SpinAPI::BuildSopheProjectionMesh(sopheParams.nOctants, sopheParams.closedPhi, sopheGridSize, grid, projectionMesh);
			if (projectionPossible)
			{
				projectionSamples = static_cast<size_t>(numPoints) * static_cast<size_t>(gamma_points);
				const size_t projectionSize = transitions.size() * projectionSamples;
				const double nan = std::numeric_limits<double>::quiet_NaN();
				projectionCounts.assign(projectionSize, 0U);
				projectionPositions.assign(projectionSize, nan);
				projectionTotalX.assign(projectionSize, 0.0);
				projectionTotalY.assign(projectionSize, 0.0);
				projectionTotalPerp.assign(projectionSize, 0.0);
				projectionCrossX.assign(projectionSize, 0.0);
				projectionCrossY.assign(projectionSize, 0.0);
				projectionSpinX.assign(spin_count, std::vector<double>(projectionSize, 0.0));
				projectionSpinY.assign(spin_count, std::vector<double>(projectionSize, 0.0));
				projectionSpinPerp.assign(spin_count, std::vector<double>(projectionSize, 0.0));
				projectionSpinP.assign(spin_count, std::vector<double>(projectionSize, 0.0));
				projectionSpinM.assign(spin_count, std::vector<double>(projectionSize, 0.0));
			}
		}

		const std::vector<double> *field_scan = &field_T;
		std::vector<double> field_scan_storage;
		unsigned int scan_steps = steps;
		if (useResfieldsCache)
		{
			// The resonance scan only needs to resolve sign changes in the transition
			// detuning, so it can use a coarser field mesh than the final output axis.
			unsigned int points = (this->sweepCacheResfieldPoints > 1) ? static_cast<unsigned int>(this->sweepCacheResfieldPoints) : 0U;
			if (points < 2)
			{
				points = std::min<unsigned int>(steps, 80U);
			}
			if (points < 2)
				points = 2;

			field_scan_storage.resize(points);
			const double Bmin = field_T.front();
			const double Bmax = field_T.back();
			const double step = (points > 1) ? ((Bmax - Bmin) / static_cast<double>(points - 1)) : 0.0;
			for (unsigned int i = 0; i < points; ++i)
			{
				field_scan_storage[i] = Bmin + step * static_cast<double>(i);
			}
			field_scan = &field_scan_storage;
			scan_steps = points;
		}

		for (int grid_num = 0; grid_num < numPoints; ++grid_num)
		{
			auto [theta, phi, w_solid] = grid[grid_num];
			const double base_weight = w_solid;

			for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
			{
				double gamma = 0.0;
				if (gamma_points > 1)
					gamma = 2.0 * arma::datum::pi * (static_cast<double>(gamma_idx) + 0.5) / static_cast<double>(gamma_points);

				const double w = base_weight * gamma_weight;

				arma::mat Rot;
				if (!this->CreatePassiveZYZRotationMatrix(phi, theta, gamma, Rot))
					continue;

				const arma::mat Rpowder = Rot;
				arma::cx_mat rho_oriented;
				if (useOrientationThermal)
				{
					arma::sp_cx_mat Hthermal_sp;
					if (!space.BaseHamiltonianRotatedZYZ(thermalhamiltonian_list, Rot, Hthermal_sp) ||
						!space.ThermalStateFromHamiltonian(arma::cx_mat(Hthermal_sp), thermalTemperature, rho_oriented))
						continue;
				}
				else
				{
					rho_oriented = rho0;
					// Keep molecular-frame initial conditions aligned with
					// the resonance orientation selected for this grid point.
					if (initialStateFrame == SpinAPI::StateFrame::Molecular && !space.RotateState(rho0, Rot, rho_oriented))
						continue;
					}

					// Cached Resonance spectra still follows the pepper-style Hilbert
					// formulation. Splitting Hstatic and Hz only accelerates
					// the field scan; it must remain the full rotated ZYZ
					// Hamiltonian, not the secular H0/H1 propagation helper.
					arma::sp_cx_mat Hstatic_sp;
					if (h0list_noB.empty())
				{
					Hstatic_sp = arma::zeros<arma::sp_cx_mat>(dim, dim);
				}
				else if (!space.BaseHamiltonianRotatedZYZ(h0list_noB, Rot, Hstatic_sp))
				{
					continue;
				}

				arma::sp_cx_mat Hz_sp;
				if (!space.BaseHamiltonianRotatedZYZ(zeelist, Rot, Hz_sp))
					continue;
				const bool can_block = hasMzBlocks && IsBlockDiagonalMz(Hstatic_sp, mzBlocks.mz2, 1e-12) && IsBlockDiagonalMz(Hz_sp, mzBlocks.mz2, 1e-12);
				arma::cx_mat Hstatic = arma::cx_mat(Hstatic_sp);
				arma::cx_mat Hz = arma::cx_mat(Hz_sp);
				const double invField0 = 1.0 / field_T[0];
				arma::cx_mat dHdB = Hz * invField0;
				const arma::sp_cx_mat dHdB_sp = Hz_sp * invField0;

				std::vector<arma::cx_mat> mux_list(spin_count);
				std::vector<arma::cx_mat> muy_list(spin_count);
				bool tensor_dim_ok = true;
				for (size_t i = 0; i < spin_count; ++i)
				{
					arma::mat g = Rpowder * g_frame_base[i] * Rpowder.t();
					if (!this->fullTensorRotation)
						g = g % arma::eye<arma::mat>(3, 3);

					arma::cx_mat mux = g(0, 0) * Sx_list[i] + g(1, 0) * Sy_list[i] + g(2, 0) * Sz_list[i];
					arma::cx_mat muy = g(0, 1) * Sx_list[i] + g(1, 1) * Sy_list[i] + g(2, 1) * Sz_list[i];

					const double mu_prefactor = mu_prefactors[i];
					if (mu_prefactor != 1.0)
					{
						mux *= mu_prefactor;
						muy *= mu_prefactor;
					}

					if (mux.n_rows != spaceDim || mux.n_cols != spaceDim || muy.n_rows != spaceDim || muy.n_cols != spaceDim)
						tensor_dim_ok = false;
					mux_list[i] = mux;
					muy_list[i] = muy;
				}
				if (!tensor_dim_ok)
					continue;

				if (useApproxCache || useResfieldsCache)
				{
					// Track each transition detuning along the scan and detect zero
					// crossings. Each zero crossing corresponds to a resonance field.
					std::vector<double> prev_delta(transitions.size(), 0.0);
					arma::cx_mat prev_eigvec;
					bool have_prev = false;

					for (unsigned int step = 0; step < scan_steps; ++step)
					{
						arma::vec eigval;
						arma::cx_mat eigvec;
						bool have_eig = false;
						const double Bscan = (*field_scan)[step];
						if (can_block)
						{
							const double scale = Bscan * invField0;
							arma::sp_cx_mat H_sp = Hstatic_sp + scale * Hz_sp;
							have_eig = EigSymBlockMz(H_sp, mzBlocks.blocks, eigval, eigvec);
						}
						else
						{
							arma::cx_mat H = Hstatic + Bscan * dHdB;
							have_eig = arma::eig_sym(eigval, eigvec, H);
						}
						if (!have_eig)
							continue;

						arma::cx_mat dHdB_ev = dHdB * eigvec;
						arma::vec dHdB_diag(dim);
						for (arma::uword i = 0; i < dim; ++i)
						{
							dHdB_diag(i) = std::real(arma::cdot(eigvec.col(i), dHdB_ev.col(i)));
						}

						std::vector<double> curr_delta(transitions.size(), 0.0);
						for (size_t t = 0; t < transitions.size(); ++t)
						{
							const auto [m, n] = transitions[t];
							curr_delta[t] = (eigval(n) - eigval(m)) - omega_mw;
						}

						if (!have_prev)
						{
							prev_delta = curr_delta;
							prev_eigvec = eigvec;
							have_prev = true;
							continue;
						}

						for (size_t t = 0; t < transitions.size(); ++t)
						{
							const double d1 = prev_delta[t];
							const double d2 = curr_delta[t];
							if (d1 == 0.0 && d2 == 0.0)
								continue;
							if (d1 == 0.0 || d2 == 0.0 || (d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
							{
								const double denom = (d1 - d2);
								if (std::abs(denom) < 1e-15)
									continue;
								const double tfrac = d1 / denom;
								const double Bres = (*field_scan)[step - 1] + tfrac * ((*field_scan)[step] - (*field_scan)[step - 1]);

								const bool use_prev = std::abs(d1) <= std::abs(d2);
								const arma::cx_mat &eigvec_use = use_prev ? prev_eigvec : eigvec;

								const auto [m, n] = transitions[t];
								const arma::cx_vec Um = eigvec_use.col(m);
								const arma::cx_vec Vn = eigvec_use.col(n);


								const double population = std::real(arma::cdot(Um, rho_oriented * Um)) - std::real(arma::cdot(Vn, rho_oriented * Vn));

								if (std::abs(population) < 1e-15)
									continue;

								const double abs_domega_dB = std::abs(dHdB_diag(n) - dHdB_diag(m));
								if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
									continue;
								const double dBdE = 1.0 / abs_domega_dB;
								if (dBdE > 1e5)
									continue;

								std::vector<double> amp_spin_x(spin_count, 0.0);
								std::vector<double> amp_spin_y(spin_count, 0.0);
								std::vector<double> amp_spin_perp(spin_count, 0.0);
								std::vector<double> amp_spin_p(spin_count, 0.0);
								std::vector<double> amp_spin_m(spin_count, 0.0);

								arma::cx_double muTx(0.0, 0.0);
								arma::cx_double muTy(0.0, 0.0);
								double I_sum_x = 0.0;
								double I_sum_y = 0.0;

								for (size_t i = 0; i < spin_count; ++i)
								{
									const arma::cx_double muix = arma::cdot(Um, mux_list[i] * Vn);
									const arma::cx_double muiy = arma::cdot(Um, muy_list[i] * Vn);
									muTx += muix;
									muTy += muiy;

									const double Iix = std::norm(muix);
									const double Iiy = std::norm(muiy);
									I_sum_x += Iix;
									I_sum_y += Iiy;

									amp_spin_x[i] = population * Iix * dBdE;
									amp_spin_y[i] = population * Iiy * dBdE;
									amp_spin_perp[i] = population * 0.5 * (Iix + Iiy) * dBdE;

									const arma::cx_double mup = muix + I * muiy;
									const arma::cx_double mum = muix - I * muiy;
									amp_spin_p[i] = population * std::norm(mup) * dBdE;
									amp_spin_m[i] = population * std::norm(mum) * dBdE;
								}

								const double ITx = std::norm(muTx);
								const double ITy = std::norm(muTy);
								const double ICx = ITx - I_sum_x;
								const double ICy = ITy - I_sum_y;

								const double amp_total_x = population * ITx * dBdE;
								const double amp_total_y = population * ITy * dBdE;
								const double amp_total_perp = population * 0.5 * (ITx + ITy) * dBdE;
								const double amp_crossx = population * ICx * dBdE;
								const double amp_crossy = population * ICy * dBdE;

								if (orientationDebug.ShouldRecord(grid_num, Bres) && debugOut.good())
								{
									debugOut << _system->Name() << "\t"
											 << grid_num << "\t"
											 << gamma_idx << "\t"
											 << std::setprecision(12) << theta << "\t"
											 << phi << "\t"
											 << gamma << "\t"
											 << w << "\t"
											 << m << "\t"
											 << n << "\t"
											 << Bres << "\t"
											 << (1.0e3 * Bres) << "\t"
											 << amp_total_x << "\t"
											 << amp_total_y << "\t"
											 << amp_total_perp << "\t"
											 << amp_crossx << "\t"
											 << amp_crossy;
									for (size_t i = 0; i < spin_count; ++i)
									{
										debugOut << "\t" << amp_spin_x[i]
												 << "\t" << amp_spin_y[i]
												 << "\t" << amp_spin_perp[i]
												 << "\t" << amp_spin_p[i]
												 << "\t" << amp_spin_m[i];
									}
									debugOut << "\n";
								}

								if (projectionPossible)
								{
									// Store one resonance field per transition and
									// orientation sample. These values are projected
									// onto the final sweep axis after the powder loop.
									const size_t sampleIndex = static_cast<size_t>(gamma_idx) * static_cast<size_t>(numPoints) + static_cast<size_t>(grid_num);
									const size_t projectionIndex = t * projectionSamples + sampleIndex;
									if (projectionCounts[projectionIndex] == 0U)
									{
										projectionPositions[projectionIndex] = 1.0e3 * Bres;
										projectionTotalX[projectionIndex] = amp_total_x;
										projectionTotalY[projectionIndex] = amp_total_y;
										projectionTotalPerp[projectionIndex] = amp_total_perp;
										projectionCrossX[projectionIndex] = amp_crossx;
										projectionCrossY[projectionIndex] = amp_crossy;
										for (size_t i = 0; i < spin_count; ++i)
										{
											projectionSpinX[i][projectionIndex] = amp_spin_x[i];
											projectionSpinY[i][projectionIndex] = amp_spin_y[i];
											projectionSpinPerp[i][projectionIndex] = amp_spin_perp[i];
											projectionSpinP[i][projectionIndex] = amp_spin_p[i];
											projectionSpinM[i][projectionIndex] = amp_spin_m[i];
										}
									}
									else
									{
										projectionPossible = false;
									}
									projectionCounts[projectionIndex] += 1U;
								}

								if (lwB_mT <= 0.0 || dBabs == 0.0)
								{
									size_t idx = 0;
									if (dBstep != 0.0)
									{
										const double pos = (Bres - field_T[0]) / dBstep;
										idx = static_cast<size_t>(std::llround(pos));
									}

									if (idx < steps)
									{
										_cache.total_x[idx] += w * amp_total_x;
										_cache.total_y[idx] += w * amp_total_y;
										_cache.total_perp[idx] += w * amp_total_perp;
										_cache.cross_x[idx] += w * amp_crossx;
										_cache.cross_y[idx] += w * amp_crossy;

										for (size_t i = 0; i < spin_count; ++i)
										{
											_cache.spin_x[i][idx] += w * amp_spin_x[i];
											_cache.spin_y[i][idx] += w * amp_spin_y[i];
											_cache.spin_perp[i][idx] += w * amp_spin_perp[i];
											_cache.spin_p[i][idx] += w * amp_spin_p[i];
											_cache.spin_m[i][idx] += w * amp_spin_m[i];
										}
									}
								}
								else
								{
									int start = 0;
									int end = static_cast<int>(steps) - 1;
									if (lineWindow > 0.0 && dBabs > 0.0)
									{
										const int half = static_cast<int>(std::ceil(lineWindow / dBabs));
										const double pos = (Bres - field_T[0]) / dBstep;
										const int center = static_cast<int>(std::llround(pos));
										start = center - half;
										end = center + half;

										if (end < 0 || start >= static_cast<int>(steps))
											continue;

										start = std::max(0, start);
										end = std::min(static_cast<int>(steps) - 1, end);
									}

									for (int j = start; j <= end; ++j)
									{
										const double deltaB_mT = (field_T[static_cast<size_t>(j)] - Bres) * 1.0e3;
										const double L = this->LineshapeValue(deltaB_mT, lwB_mT);
										if (L == 0.0)
											continue;

										const double weight = w * L;
										_cache.total_x[static_cast<size_t>(j)] += weight * amp_total_x;
										_cache.total_y[static_cast<size_t>(j)] += weight * amp_total_y;
										_cache.total_perp[static_cast<size_t>(j)] += weight * amp_total_perp;
										_cache.cross_x[static_cast<size_t>(j)] += weight * amp_crossx;
										_cache.cross_y[static_cast<size_t>(j)] += weight * amp_crossy;

										for (size_t i = 0; i < spin_count; ++i)
										{
											_cache.spin_x[i][static_cast<size_t>(j)] += weight * amp_spin_x[i];
											_cache.spin_y[i][static_cast<size_t>(j)] += weight * amp_spin_y[i];
											_cache.spin_perp[i][static_cast<size_t>(j)] += weight * amp_spin_perp[i];
											_cache.spin_p[i][static_cast<size_t>(j)] += weight * amp_spin_p[i];
											_cache.spin_m[i][static_cast<size_t>(j)] += weight * amp_spin_m[i];
										}
									}
								}
							}
						}

						prev_delta.swap(curr_delta);
						prev_eigvec = eigvec;
					}
				}
				else
				{
					// Cached exact resonance now uses the same canonical magnetic-
					// moment ownership as the uncached exact route. Detection-spin
					// and Zeeman-interaction selection remain task-owned.
					std::vector<General::Resonance::ResonanceMagneticMomentTerm>
						magneticMomentTerms;
					magneticMomentTerms.reserve(spin_count);
					bool magneticMomentTermsOk = true;
					for (size_t i = 0; i < spin_count; ++i)
					{
						auto zeeman =
							FindZeemanForSpin(detectSpins[i], zeemanInteractions);

						// Preserve the historical designated-field fallback only
						// when that interaction actually owns the detection spin.
						if (zeeman == nullptr && _fieldInteraction != nullptr)
						{
							const auto group = _fieldInteraction->Group1();
							if (std::find(
									group.begin(), group.end(),
									detectSpins[i]) != group.end())
							{
								zeeman = _fieldInteraction;
							}
						}

						if (zeeman == nullptr)
						{
							magneticMomentTermsOk = false;
							break;
						}

						General::Resonance::ResonanceMagneticMomentTerm term;
						term.spin = detectSpins[i];
						term.zeeman = zeeman;
						magneticMomentTerms.push_back(std::move(term));
					}
					if (!magneticMomentTermsOk)
						continue;

					std::vector<General::Resonance::ResonanceDetectionOperator>
						detectionChannels;
					std::string magneticMomentError;
					if (!General::Resonance::ResonanceMagneticMomentBuilder::
							BuildTransverseChannels(
								space,
								magneticMomentTerms,
								Rot,
								this->fullTensorRotation,
								detectionChannels,
								magneticMomentError))
					{
						continue;
					}
					if (detectionChannels.size() != spin_count)
						continue;

					arma::cx_mat muxT =
						arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					arma::cx_mat muyT =
						arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					for (const auto &channel : detectionChannels)
					{
						muxT += channel.x;
						muyT += channel.y;
					}

					for (unsigned int step = 0; step < steps; ++step)
					{
						arma::vec eigval;
						arma::cx_mat eigvec;
						bool have_eig = false;
						if (can_block)
						{
							const double scale =
								field_T[step] * invField0;
							arma::sp_cx_mat H_sp =
								Hstatic_sp + scale * Hz_sp;
							have_eig = EigSymBlockMz(
								H_sp, mzBlocks.blocks,
								eigval, eigvec);
						}
						else
						{
							arma::cx_mat H =
								Hstatic + field_T[step] * dHdB;
							have_eig =
								arma::eig_sym(eigval, eigvec, H);
						}
						if (!have_eig)
							continue;

						General::Resonance::ResonanceLineSet resonanceLines;
						std::string resonanceError;
						if (!General::Resonance::ExactResonanceSolver::Generate(
								eigval, eigvec, rho_oriented, dHdB_sp,
								muxT, muyT, resonanceRequest,
								resonanceLines, resonanceError,
								detectionChannels))
						{
							continue;
						}

						General::Resonance::SpectrumPoint resonancePoint;
						if (!General::Resonance::ResonanceSpectrumEvaluator::Evaluate(
								resonanceLines, resonanceRequest,
								resonancePoint, resonanceError))
						{
							continue;
						}
						if (resonancePoint.channels.size() != spin_count)
							continue;

						_cache.total_x[step] +=
							w * resonancePoint.totalX;
						_cache.total_y[step] +=
							w * resonancePoint.totalY;
						_cache.total_perp[step] +=
							w * resonancePoint.totalPerpendicular;
						_cache.cross_x[step] +=
							w * resonancePoint.crossX;
						_cache.cross_y[step] +=
							w * resonancePoint.crossY;

						for (size_t i = 0; i < spin_count; ++i)
						{
							const auto &channel =
								resonancePoint.channels[i];
							_cache.spin_x[i][step] +=
								w * channel.x;
							_cache.spin_y[i][step] +=
								w * channel.y;
							_cache.spin_perp[i][step] +=
								w * channel.perpendicular;
							_cache.spin_p[i][step] +=
								w * channel.plus;
							_cache.spin_m[i][step] +=
								w * channel.minus;
						}
					}
				}
			}
		}

		if (projectionPossible)
		{
			// Replace the provisional binned cache by the continuously projected
			// spectrum assembled from resonance fields and the powder-orientation mesh.
			auto projectChannel = [&](const std::vector<double> &amplitudeData, std::vector<double> &target)
			{
				std::fill(target.begin(), target.end(), 0.0);
				std::vector<double> positions(static_cast<size_t>(numPoints), std::numeric_limits<double>::quiet_NaN());
				std::vector<double> amplitudes(static_cast<size_t>(numPoints), 0.0);
				std::vector<double> local(target.size(), 0.0);

				for (int gamma_idx = 0; gamma_idx < gamma_points; ++gamma_idx)
				{
					for (size_t t = 0; t < transitions.size(); ++t)
					{
						bool hasResonance = false;
						for (int grid_num = 0; grid_num < numPoints; ++grid_num)
						{
							const size_t sampleIndex = static_cast<size_t>(gamma_idx) * static_cast<size_t>(numPoints) + static_cast<size_t>(grid_num);
							const size_t projectionIndex = t * projectionSamples + sampleIndex;
							positions[static_cast<size_t>(grid_num)] = projectionPositions[projectionIndex];
							amplitudes[static_cast<size_t>(grid_num)] = amplitudeData[projectionIndex];
							hasResonance = hasResonance || std::isfinite(positions[static_cast<size_t>(grid_num)]);
						}

						if (!hasResonance)
							continue;

						std::fill(local.begin(), local.end(), 0.0);
						if (projectionMesh.axial)
							SpinAPI::ProjectPowderZones(positions, amplitudes, projectionMesh.weights, _cache.field_mT, local);
						else
							SpinAPI::ProjectPowderTriangles(projectionMesh.triangles, projectionMesh.weights, positions, amplitudes, _cache.field_mT, local);

						for (size_t j = 0; j < target.size(); ++j)
							target[j] += gamma_weight * local[j];
					}
				}
			};

			projectChannel(projectionTotalX, _cache.total_x);
			projectChannel(projectionTotalY, _cache.total_y);
			projectChannel(projectionTotalPerp, _cache.total_perp);
			projectChannel(projectionCrossX, _cache.cross_x);
			projectChannel(projectionCrossY, _cache.cross_y);
			for (size_t i = 0; i < spin_count; ++i)
			{
				projectChannel(projectionSpinX[i], _cache.spin_x[i]);
				projectChannel(projectionSpinY[i], _cache.spin_y[i]);
				projectChannel(projectionSpinPerp[i], _cache.spin_perp[i]);
				projectChannel(projectionSpinP[i], _cache.spin_p[i]);
				projectChannel(projectionSpinM[i], _cache.spin_m[i]);
			}
		}


		this->ApplyDetectionHarmonic(_cache);


		return true;
	}

	// -----------------------------------------------------
	// TaskStaticHSResonanceSpectra private helper methods
	// -----------------------------------------------------

	TaskStaticHSResonanceSpectra::OrientationDiagnostics TaskStaticHSResonanceSpectra::OrientationDiagnostics::FromProperties(const MSDParser::ObjectParser &_props)
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

	void TaskStaticHSResonanceSpectra::OrientationDiagnostics::InitialiseFile(const std::string &_systemName, const std::vector<std::string> &_spinNames) const
	{
		if (!enabled)
			return;

		std::ofstream out(file, std::ios::out | std::ios::trunc);
		out << "system\tgrid_index\tgamma_index\ttheta\tphi\tgamma\tweight\ttransition_m\ttransition_n\tresonance_T\tresonance_mT\ttotal_x\ttotal_y\ttotal_perp\tcross_x\tcross_y";
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

	bool TaskStaticHSResonanceSpectra::OrientationDiagnostics::ShouldRecord(size_t _gridIndex, double _resonanceFieldT) const
	{
		return enabled &&
			   (maxOrientations <= 0 || _gridIndex < static_cast<size_t>(maxOrientations)) &&
			   _resonanceFieldT >= fieldMinT &&
			   _resonanceFieldT <= fieldMaxT;
	}

	void TaskStaticHSResonanceSpectra::FieldSyncGuard::Apply(const std::vector<SpinAPI::interaction_ptr> &_interactions, const arma::vec &_field)
	{
		saved.clear();
		saved.reserve(_interactions.size());
		for (const auto &inter : _interactions)
		{
			if (inter == nullptr)
				continue;
			const arma::vec current = inter->Field();
			if (current.n_elem != 3 || !current.is_finite())
				continue;
			saved.emplace_back(inter, current);
			arma::vec tmp = _field;
			inter->SetField(tmp);
		}
	}

	TaskStaticHSResonanceSpectra::FieldSyncGuard::~FieldSyncGuard()
	{
		for (auto &entry : saved)
		{
			arma::vec tmp = entry.second;
			entry.first->SetField(tmp);
		}
	}

	std::string TaskStaticHSResonanceSpectra::ToLower(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
					   { return static_cast<char>(std::tolower(c)); });
		return value;
	}

	// If the Hamiltonian conserves total Mz, the Hilbert space splits into
	// independent sectors. Diagonalizing these sectors is only an algebraic
	// acceleration; it does not modify the physical model.
	TaskStaticHSResonanceSpectra::MzBlocks TaskStaticHSResonanceSpectra::BuildMzBlocks(const std::vector<SpinAPI::spin_ptr> &spins)
	{
		MzBlocks result;
		if (spins.empty())
			return result;

		const size_t nspins = spins.size();
		std::vector<size_t> mult(nspins);
		std::vector<int> svals(nspins);
		size_t dim = 1;
		for (size_t i = 0; i < nspins; ++i)
		{
			mult[i] = static_cast<size_t>(spins[i]->Multiplicity());
			svals[i] = spins[i]->S();
			dim *= mult[i];
		}

		std::vector<size_t> stride(nspins, 1);
		for (size_t i = nspins; i-- > 0;)
		{
			if (i + 1 < nspins)
				stride[i] = stride[i + 1] * mult[i + 1];
		}

		result.mz2.resize(dim);
		for (size_t idx = 0; idx < dim; ++idx)
		{
			int total = 0;
			for (size_t i = 0; i < nspins; ++i)
			{
				const size_t local = (idx / stride[i]) % mult[i];
				const int m = svals[i] - 2 * static_cast<int>(local);
				total += m;
			}
			result.mz2[idx] = total;
		}

		std::map<int, std::vector<arma::uword>> groups;
		for (arma::uword i = 0; i < result.mz2.size(); ++i)
			groups[result.mz2[i]].push_back(i);

		result.blocks.reserve(groups.size());
		for (auto &kv : groups)
		{
			arma::uvec idx(static_cast<arma::uword>(kv.second.size()));
			for (size_t i = 0; i < kv.second.size(); ++i)
				idx(static_cast<arma::uword>(i)) = kv.second[i];
			result.blocks.push_back(std::move(idx));
		}

		return result;
	}

	arma::mat TaskStaticHSResonanceSpectra::PassiveZYZRotation(const arma::vec &fr)
	{
		double a = (fr.n_elem >= 1) ? fr(0) : 0.0;
		double b = (fr.n_elem >= 2) ? fr(1) : 0.0;
		double g = (fr.n_elem >= 3) ? fr(2) : 0.0;

		const double ca = std::cos(a), sa = std::sin(a);
		const double cb = std::cos(b), sb = std::sin(b);
		const double cg = std::cos(g), sg = std::sin(g);

		arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
		arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
		arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};
		return Rg * Rb * Ra;
	}

	bool TaskStaticHSResonanceSpectra::IsZeemanInteraction(const SpinAPI::interaction_ptr &inter)
	{
		if (inter == nullptr)
			return false;
		if (!SpinAPI::IsStatic(*inter))
			return false;
		if (inter->Type() != SpinAPI::InteractionType::SingleSpin)
			return false;
		const arma::vec field = inter->Field();
		return (field.n_elem == 3 && field.is_finite());
	}

	std::vector<SpinAPI::interaction_ptr> TaskStaticHSResonanceSpectra::CollectZeemanInteractions(const SpinAPI::system_ptr &system, const std::vector<std::string> &h0list)
	{
		std::vector<SpinAPI::interaction_ptr> out;
		if (system == nullptr)
			return out;

		out.reserve(h0list.size());
		for (const auto &name : h0list)
		{
			auto inter = system->interactions_find(name);
			if (!IsZeemanInteraction(inter))
				continue;
			out.push_back(inter);
		}

		std::sort(out.begin(), out.end(), [](const auto &a, const auto &b) {
			return a.get() < b.get();
		});
		out.erase(std::unique(out.begin(), out.end()), out.end());
		return out;
	}

	SpinAPI::interaction_ptr TaskStaticHSResonanceSpectra::FindZeemanForSpin(const SpinAPI::spin_ptr &spin, const std::vector<SpinAPI::interaction_ptr> &zeemanList)
	{
		if (spin == nullptr)
			return nullptr;
		for (const auto &inter : zeemanList)
		{
			if (inter == nullptr)
				continue;
			const auto group = inter->Group1();
			if (std::find(group.begin(), group.end(), spin) != group.end())
				return inter;
		}
		return nullptr;
	}

	bool TaskStaticHSResonanceSpectra::IsParallel(const arma::vec &a, const arma::vec &b, double tol)
	{
		if (a.n_elem != 3 || b.n_elem != 3)
			return false;
		const double na = arma::norm(a);
		const double nb = arma::norm(b);
		if (!std::isfinite(na) || !std::isfinite(nb) || na == 0.0 || nb == 0.0)
			return false;
		return (arma::norm(arma::cross(a / na, b / nb)) <= tol);
	}

	bool TaskStaticHSResonanceSpectra::CollectAddVectorSteps(const std::vector<std::shared_ptr<Action>> &actions,
							   unsigned int steps,
							   std::map<std::string, arma::vec> &stepsOut,
							   std::string &error)
	{
		stepsOut.clear();
		error.clear();

		for (const auto &action : actions)
		{
			auto add = std::dynamic_pointer_cast<ActionAddVector>(action);
			if (!add)
			{
				error = "Non-AddVector action present.";
				return false;
			}

			std::string targetName;
			if (!add->GetProperties()->Get("vector", targetName))
				add->GetProperties()->Get("actionvector", targetName);
			if (targetName.empty())
			{
				error = "AddVector action missing target vector.";
				return false;
			}

			arma::vec direction;
			if (!add->GetProperties()->Get("direction", direction) || direction.n_elem != 3 || !direction.is_finite())
			{
				error = "AddVector action has invalid direction.";
				return false;
			}
			direction = arma::normalise(direction);

			if (add->Period() != 1 || add->First() != 1)
			{
				error = "AddVector action has non-unit period or does not start at step 1.";
				return false;
			}
			if (add->Last() != 0 && add->Last() < steps)
			{
				error = "AddVector action terminates before the end of the run.";
				return false;
			}

			const arma::vec step = add->Value() * direction;
			auto it = stepsOut.find(targetName);
			if (it != stepsOut.end())
			{
				// RunSection applies every action. Multiple increments for one
				// target therefore compose into one net sweep step.
				it->second += step;
			}
			else
			{
				stepsOut.emplace(targetName, step);
			}
		}

		return true;
	}

	void TaskStaticHSResonanceSpectra::UpdateSymmetryFlags(const arma::mat &M, SymmetryFlags &flags, bool fullTensorRotation, double relTol)
	{
		arma::mat A = M;
		if (!fullTensorRotation)
			A = A % arma::eye<arma::mat>(3, 3);

		double maxAbs = 0.0;
		double maxOff = 0.0;
		for (arma::uword r = 0; r < 3; ++r)
		{
			for (arma::uword c = 0; c < 3; ++c)
			{
				const double v = std::abs(A(r, c));
				maxAbs = std::max(maxAbs, v);
				if (r != c)
					maxOff = std::max(maxOff, v);
			}
		}

		if (!std::isfinite(maxAbs) || maxAbs == 0.0)
			return;

		flags.anyTensor = true;
		if (maxOff > relTol * maxAbs)
		{
			flags.allDiag = false;
			flags.allAxialZ = false;
			flags.allIsotropic = false;
			return;
		}

		const double a = A(0, 0);
		const double b = A(1, 1);
		const double c = A(2, 2);
		const double mean = (a + b + c) / 3.0;
		const double maxDev = std::max({std::abs(a - mean), std::abs(b - mean), std::abs(c - mean)});
		if (maxDev > relTol * maxAbs)
			flags.allIsotropic = false;

		const bool xy_eq = (std::abs(a - b) <= relTol * maxAbs);
		if (!xy_eq && maxDev > relTol * maxAbs)
			flags.allAxialZ = false;
	}

	// The SOPHE grid can exploit point-group symmetry. We inspect the static
	// tensors entering the Hamiltonian and choose the largest symmetry that
	// leaves the tensor set invariant, which reduces the number of powder
	// orientations needed for the same orientational integral.
	std::string TaskStaticHSResonanceSpectra::AutoDetectSopheSymmetry(const SpinAPI::system_ptr &system,
										 const SpinAPI::interaction_ptr &fieldInteraction,
										 const std::vector<std::string> &h0list,
										 bool fullTensorRotation)
	{
		if (system == nullptr)
			return "c1";

		const double relTol = 1e-8;
		SymmetryFlags flags;

		for (const auto &name : h0list)
		{
			auto inter = system->interactions_find(name);
			if (inter == nullptr)
				continue;
			if (!SpinAPI::IsStatic(*inter))
				continue;

			if (inter->Type() == SpinAPI::InteractionType::SingleSpin)
			{
				arma::mat R = PassiveZYZRotation(inter->Framelist());
				arma::mat Rt = R.t();
				for (const auto &spin : inter->Group1())
				{
					arma::mat G = arma::conv_to<arma::mat>::from(spin->GetTensor().LabFrame());
					if (inter->IgnoreTensors())
						G = arma::eye<arma::mat>(3, 3);
					G = Rt * G * Rt.t();
					UpdateSymmetryFlags(G, flags, fullTensorRotation, relTol);
				}
			}
			else if (SpinAPI::HasTensor(*inter))
			{
				arma::mat A = arma::conv_to<arma::mat>::from(inter->CouplingTensor()->LabFrame());
				arma::mat R = PassiveZYZRotation(inter->Framelist());
				arma::mat Rt = R.t();
				A = Rt * A * Rt.t();
				UpdateSymmetryFlags(A, flags, fullTensorRotation, relTol);
			}
			else if (inter->Type() == SpinAPI::InteractionType::Zfs)
			{
				const double D = inter->Dvalue();
				const double E = inter->Evalue();
				const double maxAbs = std::max(std::abs(D), std::abs(E));
				if (maxAbs > relTol)
				{
					flags.anyTensor = true;
					flags.allIsotropic = false;
					if (std::abs(E) > relTol)
					{
						flags.allAxialZ = false;
					}
				}
			}
			else if (inter->Type() == SpinAPI::InteractionType::SemiClassicalField)
			{
				return "c1";
			}
		}

		if (fieldInteraction != nullptr)
		{
			arma::mat R = PassiveZYZRotation(fieldInteraction->Framelist());
			arma::mat Rt = R.t();
			for (const auto &spin : fieldInteraction->Group1())
			{
				arma::mat G = arma::conv_to<arma::mat>::from(spin->GetTensor().LabFrame());
				if (fieldInteraction->IgnoreTensors())
					G = arma::eye<arma::mat>(3, 3);
				G = Rt * G * Rt.t();
				UpdateSymmetryFlags(G, flags, fullTensorRotation, relTol);
			}
		}

		if (!flags.anyTensor || flags.allIsotropic)
			return "o3";
		if (flags.allAxialZ)
			return "dinfh";
		if (flags.allDiag)
			return "d2h";
		return "c1";
	}

	bool TaskStaticHSResonanceSpectra::IsBlockDiagonalMz(const arma::sp_cx_mat &H, const std::vector<int> &mz2, double relTol)
	{
		if (H.n_nonzero == 0)
			return true;
		double maxAbs = 0.0;
		for (auto it = H.begin(); it != H.end(); ++it)
			maxAbs = std::max(maxAbs, std::abs(*it));
		if (maxAbs == 0.0)
			return true;
		const double thresh = maxAbs * relTol;
		for (auto it = H.begin(); it != H.end(); ++it)
		{
			if (std::abs(*it) <= thresh)
				continue;
			if (mz2[it.row()] != mz2[it.col()])
				return false;
		}
		return true;
	}

	bool TaskStaticHSResonanceSpectra::EigSymBlockMz(const arma::sp_cx_mat &H, const std::vector<arma::uvec> &blocks, arma::vec &eigval, arma::cx_mat &eigvec)
	{
		const arma::uword dim = H.n_rows;
		eigval.set_size(dim);
		eigvec.zeros(dim, dim);

		struct Entry
		{
			double val;
			size_t block;
			arma::uword local;
		};

		std::vector<Entry> entries;
		entries.reserve(dim);
		std::vector<arma::cx_mat> block_vecs(blocks.size());

		std::vector<int> block_id(static_cast<size_t>(dim), -1);
		std::vector<arma::uword> local_pos(static_cast<size_t>(dim), 0);
		for (size_t b = 0; b < blocks.size(); ++b)
		{
			const arma::uvec &idx = blocks[b];
			for (arma::uword i = 0; i < idx.n_elem; ++i)
			{
				block_id[idx(i)] = static_cast<int>(b);
				local_pos[idx(i)] = i;
			}
		}

		std::vector<arma::cx_mat> block_mats(blocks.size());
		for (size_t b = 0; b < blocks.size(); ++b)
		{
			const arma::uvec &idx = blocks[b];
			block_mats[b].zeros(idx.n_elem, idx.n_elem);
		}

		for (auto it = H.begin(); it != H.end(); ++it)
		{
			const int b = block_id[it.row()];
			if (b < 0)
				continue;
			if (block_id[it.col()] != b)
				continue;
			block_mats[static_cast<size_t>(b)](local_pos[it.row()], local_pos[it.col()]) = *it;
		}

		for (size_t b = 0; b < blocks.size(); ++b)
		{
			const arma::uvec &idx = blocks[b];
			if (idx.n_elem == 0)
				continue;
			arma::vec evals;
			arma::cx_mat evecs;
			if (!arma::eig_sym(evals, evecs, block_mats[b]))
				return false;
			block_vecs[b] = std::move(evecs);
			for (arma::uword k = 0; k < evals.n_elem; ++k)
				entries.push_back({evals(k), b, k});
		}

		if (entries.size() != static_cast<size_t>(dim))
			return false;

		std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b)
				  { return a.val < b.val; });

		for (arma::uword col = 0; col < dim; ++col)
		{
			const auto &e = entries[col];
			eigval(col) = e.val;
			const arma::uvec &idx = blocks[e.block];
			for (arma::uword i = 0; i < idx.n_elem; ++i)
			{
				eigvec(idx(i), col) = block_vecs[e.block](i, e.local);
			}
		}

		return true;
	}

}
