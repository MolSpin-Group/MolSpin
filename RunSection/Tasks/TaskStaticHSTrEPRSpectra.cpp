/////////////////////////////////////////////////////////////////////////
// TaskStaticHSTrEPRSpectra implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

#include "ActionAddVector.h"
#include "TaskStaticHSTrEPRSpectra.h"
#include "ObjectParser.h"
#include "Settings.h"
#include "Spin.h"
#include "SpinSpace.h"
#include "SpinSystem.h"
#include "State.h"
#include "Interaction.h"
#include <omp.h>

namespace RunSection
{
	namespace
	{
		std::string ToLower(std::string value)
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
						   { return static_cast<char>(std::tolower(c)); });
			return value;
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

		struct MzBlocks
		{
			std::vector<int> mz2;		  // total Mz in units of 1/2
			std::vector<arma::uvec> blocks; // basis indices grouped by Mz
		};

		MzBlocks BuildMzBlocks(const std::vector<SpinAPI::spin_ptr> &spins)
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

		arma::mat PassiveZXZRotation(const arma::vec &fr)
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

		bool IsZeemanInteraction(const SpinAPI::interaction_ptr &inter)
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

		std::vector<SpinAPI::interaction_ptr> CollectZeemanInteractions(const SpinAPI::system_ptr &system, const std::vector<std::string> &h0list)
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

		SpinAPI::interaction_ptr FindZeemanForSpin(const SpinAPI::spin_ptr &spin, const std::vector<SpinAPI::interaction_ptr> &zeemanList)
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

		bool IsParallel(const arma::vec &a, const arma::vec &b, double tol)
		{
			if (a.n_elem != 3 || b.n_elem != 3)
				return false;
			const double na = arma::norm(a);
			const double nb = arma::norm(b);
			if (!std::isfinite(na) || !std::isfinite(nb) || na == 0.0 || nb == 0.0)
				return false;
			return (arma::norm(arma::cross(a / na, b / nb)) <= tol);
		}

		bool CollectAddVectorSteps(const std::vector<std::shared_ptr<Action>> &actions,
								   unsigned int steps,
								   std::map<std::string, arma::vec> &stepsOut,
								   std::string &error)
		{
			stepsOut.clear();
			error.clear();

			const double tol = 1e-12;
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
					error = "AddVector action has non-unit period or nonzero start.";
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
					if (arma::norm(it->second - step) > tol)
					{
						error = "Conflicting AddVector actions for the same target.";
						return false;
					}
				}
				else
				{
					stepsOut.emplace(targetName, step);
				}
			}

			return true;
		}

		struct FieldSyncGuard
		{
			std::vector<std::pair<SpinAPI::interaction_ptr, arma::vec>> saved;

			void Apply(const std::vector<SpinAPI::interaction_ptr> &interactions, const arma::vec &field)
			{
				saved.clear();
				saved.reserve(interactions.size());
				for (const auto &inter : interactions)
				{
					if (inter == nullptr)
						continue;
					const arma::vec current = inter->Field();
					if (current.n_elem != 3 || !current.is_finite())
						continue;
					saved.emplace_back(inter, current);
					arma::vec tmp = field;
					inter->SetField(tmp);
				}
			}

			~FieldSyncGuard()
			{
				for (auto &entry : saved)
				{
					arma::vec tmp = entry.second;
					entry.first->SetField(tmp);
				}
			}
		};

		struct SymmetryFlags
		{
			bool allIsotropic = true;
			bool allDiag = true;
			bool allAxialZ = true;
			bool anyTensor = false;
		};

		void UpdateSymmetryFlags(const arma::mat &M, SymmetryFlags &flags, bool fullTensorRotation, double relTol)
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

		std::string AutoDetectSopheSymmetry(const SpinAPI::system_ptr &system,
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
					arma::mat R = PassiveZXZRotation(inter->Framelist());
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
					arma::mat R = PassiveZXZRotation(inter->Framelist());
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
				arma::mat R = PassiveZXZRotation(fieldInteraction->Framelist());
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

		bool IsBlockDiagonalMz(const arma::sp_cx_mat &H, const std::vector<int> &mz2, double relTol)
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

		bool EigSymBlockMz(const arma::sp_cx_mat &H, const std::vector<arma::uvec> &blocks, arma::vec &eigval, arma::cx_mat &eigvec)
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

	// -----------------------------------------------------
	// TaskStaticHSTrEPRSpectra Constructors and Destructor
	// -----------------------------------------------------
	TaskStaticHSTrEPRSpectra::TaskStaticHSTrEPRSpectra(const MSDParser::ObjectParser &_parser, const RunSection &_runsection)
		: BasicTask(_parser, _runsection),
		  mwFrequencyGHz(0.0),
		  linewidth_mT(0.0),
		  linewidthFad_mT(0.0),
		  linewidthDonor_mT(0.0),
		  lineshape("gaussian"),
		  powderGridType("fibonacci"),
		  powderGridSymmetry("D2h"),
		  powderGridSize(0),
		  powdersamplingpoints(0),
		  powderGammaPoints(1),
		  powderFullSphere(true),
		  fullTensorRotation(true),
		  useSweepCache(true),
		  sweepCacheExact(false),
		  sweepCacheResfields(false),
		  sweepCacheResfieldPoints(0),
		  detectSpinNames(),
		  electron1Name(""),
		  electron2Name(""),
		  fieldInteractionName(""),
		  enforceZeemanSync(false),
		  initialStateName(""),
		  hamiltonianH0list()
	{
	}

	TaskStaticHSTrEPRSpectra::~TaskStaticHSTrEPRSpectra()
	{
	}

	// -----------------------------------------------------
	// TaskStaticHSTrEPRSpectra protected methods
	// -----------------------------------------------------
	bool TaskStaticHSTrEPRSpectra::RunLocal()
	{
		this->Log() << "Running task StaticHS-TrEPR-Spectra." << std::endl;
		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->Log() << "Sweep cache " << (this->useSweepCache ? "enabled" : "disabled");
			if (this->useSweepCache)
			{
				const char *mode = this->sweepCacheExact ? "exact" : (this->sweepCacheResfields ? "resfields" : "approx");
				this->Log() << " (mode: " << mode << ")";
			}
			this->Log() << "." << std::endl;
		}

		if (this->RunSettings()->CurrentStep() == 1)
		{
			this->WriteHeader(this->Data());
		}

		// Excitation angular frequency (rad/ns)
		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		// Loop through all SpinSystems
		auto systems = this->SpinSystems();
		for (auto sysIt = systems.cbegin(); sysIt != systems.cend(); sysIt++)
		{
			this->Log() << "\nStarting with SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;

			SpinAPI::SpinSpace space(*(*sysIt));
			space.UseSuperoperatorSpace(false);
			space.UseFullTensorRotation(this->fullTensorRotation);
			const arma::uword spaceDim = space.HilbertSpaceDimensions();
			const auto allSpins = (*sysIt)->Spins();
			const MzBlocks mzBlocks = BuildMzBlocks(allSpins);
			const bool hasMzBlocks = (!mzBlocks.mz2.empty() && mzBlocks.mz2.size() == static_cast<size_t>(spaceDim) && mzBlocks.blocks.size() > 1);
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

			// Determine Zeeman interaction used for B and for dH/dB (Jacobian)
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

			// Build initial density matrix
			arma::cx_mat rho0;
			bool hasInitialState = false;
			if (!this->initialStateName.empty())
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
			if (!hasInitialState)
			{
				auto initial_states = (*sysIt)->InitialState();
				if (initial_states.empty())
				{
					this->Log() << "Skipping SpinSystem \"" << (*sysIt)->Name() << "\" as no initial state was specified." << std::endl;
					continue;
				}

				for (auto state = initial_states.cbegin(); state != initial_states.cend(); state++)
				{
					arma::cx_mat tmp;
					if (!space.GetState(*state, tmp))
					{
						this->Log() << "Failed to obtain projection matrix onto state \"" << (*state)->Name() << "\" of SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
						continue;
					}

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
				this->Log() << "Failed to construct initial state for SpinSystem \"" << (*sysIt)->Name() << "\"." << std::endl;
				continue;
			}
			rho0 /= arma::trace(rho0);

			// Embed *bare* spin operators (Sx,Sy,Sz) into the full Hilbert space.
			// Build magnetic dipole operators from orientation-rotated tensors to match H0's frame.
			std::vector<arma::cx_mat> Sx_list(detectSpins.size());
			std::vector<arma::cx_mat> Sy_list(detectSpins.size());
			std::vector<arma::cx_mat> Sz_list(detectSpins.size());
			bool operators_ok = true;
			for (size_t i = 0; i < detectSpins.size(); ++i)
			{
				auto spin = detectSpins[i];
				if (!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sx()), spin, Sx_list[i]) ||
					!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sy()), spin, Sy_list[i]) ||
					!space.CreateOperator(arma::conv_to<arma::cx_mat>::from(spin->Sz()), spin, Sz_list[i]))
				{
					this->Log() << "Failed to build bare spin operators for spin \"" << spin->Name() << "\"." << std::endl;
					operators_ok = false;
					break;
				}
			}
			if (!operators_ok)
				continue;

			// Build powder grid (theta,phi) and optional gamma sampling.
			int numPoints = this->powdersamplingpoints;
			std::vector<std::tuple<double, double, double>> grid;
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
					double maxPhi = 0.0;
					bool closedPhi = false;
					int nOctants = 0;
					if (numPoints > 1 && this->SopheGridParams(gridSymmetry, maxPhi, closedPhi, nOctants))
					{
						int bestSize = 0;
						int bestDiff = std::numeric_limits<int>::max();
						for (int candidate = 2; candidate <= 200; ++candidate)
						{
							int count = SopheGridPointCount(candidate, nOctants, closedPhi);
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

				if (!this->CreateSopheGrid(gridSize, gridSymmetry, grid))
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
				grid.emplace_back(0.0, 0.0, 1.0);
				numPoints = 1;
			}
			const int gamma_points = (numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
			const double gamma_weight = useSopheGrid ? (2.0 * arma::datum::pi / static_cast<double>(gamma_points))
													 : (1.0 / static_cast<double>(gamma_points));
			if (this->powderGammaPoints > 1)
				this->Log() << "Sampling gamma with " << this->powderGammaPoints << " points per orientation." << std::endl;

			// Field-domain linewidth (FWHM, mT). Apply the lineshape in field units.
			const double lwB_mT = std::abs(this->linewidth_mT);

			// Determine Zeeman interaction for each detection spin (frame + prefactor).
			std::vector<SpinAPI::interaction_ptr> spinZeeman(detectSpins.size(), nullptr);
			for (size_t i = 0; i < detectSpins.size(); ++i)
			{
				spinZeeman[i] = FindZeemanForSpin(detectSpins[i], zeemanInteractions);
				if (spinZeeman[i] == nullptr)
					spinZeeman[i] = fieldInteraction;
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
					RFrame = PassiveZXZRotation(spinZeeman[i]->Framelist());
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

			const arma::cx_double I(0.0, 1.0);
			const size_t spin_count = detectSpins.size();

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
					if (!this->CreateRotationMatrix(phi, theta, gamma, Rot))
						continue;

					// Rot is a PASSIVE ZXZ rotation (EasySpin convention) from molecular to lab frame.
					// Use it directly for tensor rotation and magnetic dipole operators.
					const arma::mat Rpowder = Rot;

					// Build rotated base Hamiltonian
					arma::sp_cx_mat H0_sp;
					if (!space_local.BaseHamiltonianRotatedZXZ(h0list, Rot, H0_sp))
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

					const arma::cx_mat Udag = arma::trans(arma::conj(eigvec));
					const arma::cx_mat rho_eig = Udag * rho0 * eigvec;
					const arma::vec rho_diag = arma::real(rho_eig.diag());

					// Zeeman-only rotated Hamiltonian -> dH/dB magnitude for field-to-energy Jacobian.
					arma::sp_cx_mat Hz_sp;
					if (!space_local.BaseHamiltonianRotatedZXZ(zeelist, Rot, Hz_sp))
						continue;
					arma::sp_cx_mat dHdB_sp = Hz_sp / Bmag; // rad/ns/T
					// For Hermitian dHdB, Re(<n|dHdB|n> - <m|dHdB|m>) gives d(En-Em)/dB.
					// Compute diagonal expectations once per orientation to avoid O(N^4) mat-vecs in the transition loop.
					arma::cx_mat dHdB_ev = dHdB_sp * eigvec;
					arma::vec dHdB_diag(spaceDim);
					for (arma::uword i = 0; i < spaceDim; ++i)
					{
						dHdB_diag(i) = std::real(arma::cdot(eigvec.col(i), dHdB_ev.col(i)));
					}

					std::vector<arma::cx_mat> mux_list(spin_count);
					std::vector<arma::cx_mat> muy_list(spin_count);
					arma::cx_mat muxT = arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
					arma::cx_mat muyT = arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
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
						muxT += mux;
						muyT += muy;
					}
					if (!tensor_dim_ok)
						continue;

					// Transform mu operators into eigenbasis
					arma::cx_mat muxT_eig = Udag * muxT * eigvec;
					arma::cx_mat muyT_eig = Udag * muyT * eigvec;
					std::vector<arma::cx_mat> mux_eig(spin_count);
					std::vector<arma::cx_mat> muy_eig(spin_count);
					for (size_t i = 0; i < spin_count; ++i)
					{
						mux_eig[i] = Udag * mux_list[i] * eigvec;
						muy_eig[i] = Udag * muy_list[i] * eigvec;
					}

					const arma::uword dim = eigval.n_elem;
					if (dim != spaceDim)
						continue;

					double loc_total_x = 0.0;
					double loc_total_y = 0.0;
					double loc_total_perp = 0.0;
					double loc_cross_x = 0.0;
					double loc_cross_y = 0.0;
					std::vector<double> loc_spin_x(spin_count, 0.0);
					std::vector<double> loc_spin_y(spin_count, 0.0);
					std::vector<double> loc_spin_perp(spin_count, 0.0);
					std::vector<double> loc_spin_p(spin_count, 0.0);
					std::vector<double> loc_spin_m(spin_count, 0.0);

					for (arma::uword m = 0; m < dim; ++m)
					{
						const double rho_mm = rho_diag(m);
						for (arma::uword n = m + 1; n < dim; ++n)
						{
							const double population = rho_mm - rho_diag(n);
							if (std::abs(population) < 1e-15)
								continue;

							const double deltaOmega = (eigval(n) - eigval(m)) - omega_mw; // rad/ns
							const double abs_domega_dB = std::abs(dHdB_diag(n) - dHdB_diag(m));
							if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
								continue;

							const double dBdE = 1.0 / abs_domega_dB; // T / (rad/ns)
							// Jacobian safeguard: the 1/g factor (dB/dE) can diverge when
							// d(E_n-E_m)/dB \approx 0 (near avoided crossings / degeneracies).
							// Skip these transitions to avoid unphysical spikes.
							if (dBdE > 1e5)
								continue;

							const double deltaB = deltaOmega * dBdE; // T
							const double deltaB_mT = 1.0e3 * deltaB;

							const double L = this->LineshapeValue(deltaB_mT, lwB_mT);
							if (L == 0.0)
								continue;

							const double wField = dBdE * L;

							const double ITx = std::norm(muxT_eig(m, n));
							const double ITy = std::norm(muyT_eig(m, n));
							double I_sum_x = 0.0;
							double I_sum_y = 0.0;

							for (size_t i = 0; i < spin_count; ++i)
							{
								const arma::cx_double muix = mux_eig[i](m, n);
								const arma::cx_double muiy = muy_eig[i](m, n);
								const double Iix = std::norm(muix);
								const double Iiy = std::norm(muiy);

								I_sum_x += Iix;
								I_sum_y += Iiy;

								loc_spin_x[i] += population * Iix * wField;
								loc_spin_y[i] += population * Iiy * wField;
								loc_spin_perp[i] += population * 0.5 * (Iix + Iiy) * wField;

								const arma::cx_double mup = muix + I * muiy;
								const arma::cx_double mum = muix - I * muiy;
								loc_spin_p[i] += population * std::norm(mup) * wField;
								loc_spin_m[i] += population * std::norm(mum) * wField;
							}

							const double ICx = ITx - I_sum_x;
							const double ICy = ITy - I_sum_y;

							loc_total_x += population * ITx * wField;
							loc_total_y += population * ITy * wField;
							loc_total_perp += population * 0.5 * (ITx + ITy) * wField;
							loc_cross_x += population * ICx * wField;
							loc_cross_y += population * ICy * wField;
						}
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

	double TaskStaticHSTrEPRSpectra::LineshapeValue(double _delta, double _fwhm) const
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

	double TaskStaticHSTrEPRSpectra::LinewidthToOmega(double _fwhm_mT, double _giso) const
	{
		const double muB_over_hbar = 8.79410005e+1; // rad / ns / T
		return std::abs(_fwhm_mT) * 1.0e-3 * muB_over_hbar * std::abs(_giso);
	}

	bool TaskStaticHSTrEPRSpectra::CreateRotationMatrix(double &_alpha, double &_beta, double &_gamma, arma::mat &_R) const
	{
		// EasySpin convention: passive ZXZ Euler rotation (molecular -> lab).
		const double ca = std::cos(_alpha), sa = std::sin(_alpha);
		const double cb = std::cos(_beta), sb = std::sin(_beta);
		const double cg = std::cos(_gamma), sg = std::sin(_gamma);

		arma::mat Ra = {{ca, sa, 0.0}, {-sa, ca, 0.0}, {0.0, 0.0, 1.0}};
		arma::mat Rb = {{cb, 0.0, -sb}, {0.0, 1.0, 0.0}, {sb, 0.0, cb}};
		arma::mat Rg = {{cg, sg, 0.0}, {-sg, cg, 0.0}, {0.0, 0.0, 1.0}};

		_R = Rg * Rb * Ra;
		return true;
	}

	bool TaskStaticHSTrEPRSpectra::SopheGridParams(const std::string &_symmetry, double &_maxPhi, bool &_closedPhi, int &_nOctants) const
	{
		std::string sym = ToLower(_symmetry);
		if (sym == "c1")
		{
			_maxPhi = 2.0 * arma::datum::pi;
			_closedPhi = false;
			_nOctants = 8;
		}
		else if (sym == "ci")
		{
			_maxPhi = 2.0 * arma::datum::pi;
			_closedPhi = false;
			_nOctants = 4;
		}
		else if (sym == "c2h")
		{
			_maxPhi = arma::datum::pi;
			_closedPhi = false;
			_nOctants = 2;
		}
		else if (sym == "s6")
		{
			_maxPhi = 2.0 * arma::datum::pi / 3.0;
			_closedPhi = false;
			_nOctants = 2;
		}
		else if (sym == "c4h")
		{
			_maxPhi = arma::datum::pi / 2.0;
			_closedPhi = false;
			_nOctants = 1;
		}
		else if (sym == "c6h")
		{
			_maxPhi = arma::datum::pi / 3.0;
			_closedPhi = false;
			_nOctants = 1;
		}
		else if (sym == "d2h")
		{
			_maxPhi = arma::datum::pi / 2.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "th")
		{
			_maxPhi = arma::datum::pi / 2.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "d3d")
		{
			_maxPhi = arma::datum::pi / 3.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "d4h")
		{
			_maxPhi = arma::datum::pi / 4.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "oh")
		{
			_maxPhi = arma::datum::pi / 4.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "d6h")
		{
			_maxPhi = arma::datum::pi / 6.0;
			_closedPhi = true;
			_nOctants = 1;
		}
		else if (sym == "dinfh")
		{
			_maxPhi = 0.0;
			_closedPhi = true;
			_nOctants = 0;
		}
		else if (sym == "o3")
		{
			_maxPhi = 0.0;
			_closedPhi = true;
			_nOctants = -1;
		}
		else
		{
			return false;
		}

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::CreateSopheGrid(int _gridSize, const std::string &_symmetry, std::vector<std::tuple<double, double, double>> &_grid) const
	{
		_grid.clear();
		if (_gridSize < 1)
			return false;

		double maxPhi = 0.0;
		bool closedPhi = false;
		int nOctants = 0;
		if (!this->SopheGridParams(_symmetry, maxPhi, closedPhi, nOctants))
			return false;

		if (nOctants == -1)
		{
			_grid.emplace_back(0.0, 0.0, 4.0 * arma::datum::pi);
			return true;
		}

		if (nOctants == 0)
		{
			if (_gridSize < 2)
				return false;
			const double dtheta = (arma::datum::pi / 2.0) / static_cast<double>(_gridSize - 1);
			std::vector<double> boundaries;
			boundaries.reserve(static_cast<size_t>(_gridSize + 1));
			boundaries.push_back(0.0);
			for (int i = 0; i < _gridSize - 1; ++i)
				boundaries.push_back(dtheta * (0.5 + static_cast<double>(i)));
			boundaries.push_back(arma::datum::pi / 2.0);

			_grid.reserve(static_cast<size_t>(_gridSize));
			for (int i = 0; i < _gridSize; ++i)
			{
				const double theta = dtheta * static_cast<double>(i);
				const double w = -2.0 * (2.0 * arma::datum::pi) * (std::cos(boundaries[i + 1]) - std::cos(boundaries[i]));
				_grid.emplace_back(theta, 0.0, w);
			}

			return true;
		}

		if (_gridSize < 2)
			return false;

		const int nOct = (nOctants == 8) ? 4 : nOctants;
		const double dtheta = (arma::datum::pi / 2.0) / static_cast<double>(_gridSize - 1);
		const double sindth2 = std::sin(dtheta / 2.0);
		const double w0 = closedPhi ? 0.5 : 1.0;

		const int nOrientations = _gridSize + nOct * _gridSize * (_gridSize - 1) / 2;
		std::vector<double> phi(static_cast<size_t>(nOrientations), 0.0);
		std::vector<double> theta(static_cast<size_t>(nOrientations), 0.0);
		std::vector<double> weights(static_cast<size_t>(nOrientations), 0.0);

		phi[0] = 0.0;
		theta[0] = 0.0;
		weights[0] = maxPhi * (1.0 - std::cos(dtheta / 2.0));

		int start = 1;
		for (int iSlice = 2; iSlice <= _gridSize - 1; ++iSlice)
		{
			const int nPhi = nOct * (iSlice - 1) + 1;
			const double dPhi = maxPhi / static_cast<double>(nPhi - 1);
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

		const int nPhiEq = nOct * (_gridSize - 1) + 1;
		const double dPhiEq = maxPhi / static_cast<double>(nPhiEq - 1);
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

		if (!closedPhi)
		{
			std::vector<int> rmv;
			rmv.reserve(static_cast<size_t>(_gridSize - 1));
			int csum = 0;
			for (int i = 1; i <= _gridSize - 1; ++i)
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

		if (nOctants == 8)
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

		const double scale = 2.0 * (2.0 * arma::datum::pi / maxPhi);
		for (auto &w : weights)
			w *= scale;

		_grid.reserve(phi.size());
		for (size_t i = 0; i < phi.size(); ++i)
			_grid.emplace_back(theta[i], phi[i], weights[i]);

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::CreateUniformGrid(int &_Npoints, std::vector<std::tuple<double, double, double>> &_uniformGrid) const
	{
		std::vector<double> theta(_Npoints);
		std::vector<double> phi(_Npoints);
		std::vector<double> weight(_Npoints);

		_uniformGrid.resize(_Npoints);

		const double golden = arma::datum::pi * (1.0 + std::sqrt(5.0));

		for (int i = 0; i < _Npoints; ++i)
		{
			double index = static_cast<double>(i) + 0.5;

			if (this->powderFullSphere)
			{
				theta[i] = std::acos(1.0 - 2.0 * index / _Npoints);
				phi[i] = golden * index;
				weight[i] = 4 * arma::datum::pi / _Npoints;
			}
			else
			{
				theta[i] = std::acos(1.0 - index / _Npoints);
				phi[i] = golden * index;
				weight[i] = 2 * arma::datum::pi / _Npoints;
			}
			_uniformGrid[i] = {theta[i], phi[i], weight[i]};
		}

		return true;
	}

	bool TaskStaticHSTrEPRSpectra::ResolveFieldInteraction(const SpinAPI::system_ptr &_system, SpinAPI::interaction_ptr &_fieldInteraction) const
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

	bool TaskStaticHSTrEPRSpectra::ResolveDetectionSpins(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction,
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
		else if (!this->electron1Name.empty() || !this->electron2Name.empty())
		{
			if (!this->electron1Name.empty())
			{
				auto spin = _system->spins_find(this->electron1Name);
				if (spin == nullptr)
					return false;
				add_spin(spin);
			}
			if (!this->electron2Name.empty())
			{
				auto spin = _system->spins_find(this->electron2Name);
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

	bool TaskStaticHSTrEPRSpectra::GetLinearFieldSweep(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, arma::vec &_field0, arma::vec &_fieldStep) const
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

	bool TaskStaticHSTrEPRSpectra::BuildCachedSpectrum(const SpinAPI::system_ptr &_system, const SpinAPI::interaction_ptr &_fieldInteraction, const arma::vec &_field0, const arma::vec &_fieldStep, SpectrumCache &_cache)
	{
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
		const double dBstep = (steps > 1) ? (field_T[1] - field_T[0]) : 0.0;
		const double dBabs = std::abs(dBstep);

		const double omega_mw = 2.0 * arma::datum::pi * this->mwFrequencyGHz;

		SpinAPI::SpinSpace space(*_system);
		space.UseSuperoperatorSpace(false);
		space.UseFullTensorRotation(this->fullTensorRotation);
		const arma::uword spaceDim = space.HilbertSpaceDimensions();
		const auto allSpins = _system->Spins();
		const MzBlocks mzBlocks = BuildMzBlocks(allSpins);
		const bool hasMzBlocks = (!mzBlocks.mz2.empty() && mzBlocks.mz2.size() == static_cast<size_t>(spaceDim) && mzBlocks.blocks.size() > 1);

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
		_cache.spin_names = detectSpinNames;
		_cache.spin_x.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_y.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_perp.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_p.assign(detectSpins.size(), std::vector<double>(steps, 0.0));
		_cache.spin_m.assign(detectSpins.size(), std::vector<double>(steps, 0.0));

		arma::cx_mat rho0;
		bool hasInitialState = false;
		if (!this->initialStateName.empty())
		{
			auto state = _system->states_find(this->initialStateName);
			if (state != nullptr && space.GetState(state, rho0))
				hasInitialState = true;
		}
		if (!hasInitialState)
		{
			auto initial_states = _system->InitialState();
			if (initial_states.empty())
				return false;

			for (auto state = initial_states.cbegin(); state != initial_states.cend(); state++)
			{
				arma::cx_mat tmp;
				if (!space.GetState(*state, tmp))
					continue;

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
			return false;
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
		std::vector<std::tuple<double, double, double>> grid;
		const bool useSopheGrid = (this->powderGridType == "sophe");
		std::string gridSymmetry = this->powderGridSymmetry;
		if (useSopheGrid)
		{
			std::string symLower = ToLower(gridSymmetry);
			if (symLower.empty() || symLower == "auto" || symLower == "automatic")
			{
				gridSymmetry = AutoDetectSopheSymmetry(_system, _fieldInteraction, h0list, this->fullTensorRotation);
				this->Log() << "Auto-detected SOPHE grid symmetry: " << gridSymmetry << "." << std::endl;
			}
		}
		if (useSopheGrid)
		{
			int gridSize = this->powderGridSize;
			if (gridSize < 2)
			{
				double maxPhi = 0.0;
				bool closedPhi = false;
				int nOctants = 0;
				if (numPoints > 1 && this->SopheGridParams(gridSymmetry, maxPhi, closedPhi, nOctants))
				{
					int bestSize = 0;
					int bestDiff = std::numeric_limits<int>::max();
					for (int candidate = 2; candidate <= 200; ++candidate)
					{
						int count = SopheGridPointCount(candidate, nOctants, closedPhi);
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

			if (!this->CreateSopheGrid(gridSize, gridSymmetry, grid))
				return false;
			numPoints = static_cast<int>(grid.size());
		}
		else if (numPoints > 1)
		{
			if (!this->CreateUniformGrid(numPoints, grid))
				return false;
		}
		else
		{
			grid.clear();
			grid.emplace_back(0.0, 0.0, 1.0);
			numPoints = 1;
		}
		const int gamma_points = (numPoints > 1) ? std::max(1, this->powderGammaPoints) : 1;
		const double gamma_weight = useSopheGrid ? (2.0 * arma::datum::pi / static_cast<double>(gamma_points))
												 : (1.0 / static_cast<double>(gamma_points));

		const double lwB_mT = std::abs(this->linewidth_mT);
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
				RFrame = PassiveZXZRotation(spinZeeman[i]->Framelist());
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

		const std::vector<double> *field_scan = &field_T;
		std::vector<double> field_scan_storage;
		unsigned int scan_steps = steps;
		if (useResfieldsCache)
		{
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
				if (!this->CreateRotationMatrix(phi, theta, gamma, Rot))
					continue;

				const arma::mat Rpowder = Rot;

				arma::sp_cx_mat Hstatic_sp;
				if (h0list_noB.empty())
				{
					Hstatic_sp = arma::zeros<arma::sp_cx_mat>(dim, dim);
				}
				else if (!space.BaseHamiltonianRotatedZXZ(h0list_noB, Rot, Hstatic_sp))
				{
					continue;
				}

				arma::sp_cx_mat Hz_sp;
				if (!space.BaseHamiltonianRotatedZXZ(zeelist, Rot, Hz_sp))
					continue;
				const bool can_block = hasMzBlocks && IsBlockDiagonalMz(Hstatic_sp, mzBlocks.mz2, 1e-12) && IsBlockDiagonalMz(Hz_sp, mzBlocks.mz2, 1e-12);
				arma::cx_mat Hstatic = arma::cx_mat(Hstatic_sp);
				arma::cx_mat Hz = arma::cx_mat(Hz_sp);
				const double invField0 = 1.0 / field_T[0];
				arma::cx_mat dHdB = Hz * invField0;

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

								const double population = std::real(arma::cdot(Um, rho0 * Um)) - std::real(arma::cdot(Vn, rho0 * Vn));
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
					for (unsigned int step = 0; step < steps; ++step)
					{
						arma::vec eigval;
						arma::cx_mat eigvec;
						bool have_eig = false;
						if (can_block)
						{
							const double scale = field_T[step] * invField0;
							arma::sp_cx_mat H_sp = Hstatic_sp + scale * Hz_sp;
							have_eig = EigSymBlockMz(H_sp, mzBlocks.blocks, eigval, eigvec);
						}
						else
						{
							arma::cx_mat H = Hstatic + field_T[step] * dHdB;
							have_eig = arma::eig_sym(eigval, eigvec, H);
						}
						if (!have_eig)
							continue;

						const arma::cx_mat Udag = arma::trans(arma::conj(eigvec));
						const arma::cx_mat rho_eig = Udag * rho0 * eigvec;
						const arma::vec rho_diag = arma::real(rho_eig.diag());

						arma::cx_mat dHdB_ev = dHdB * eigvec;
						arma::vec dHdB_diag(dim);
						for (arma::uword i = 0; i < dim; ++i)
						{
							dHdB_diag(i) = std::real(arma::cdot(eigvec.col(i), dHdB_ev.col(i)));
						}

						arma::cx_mat muxT = arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
						arma::cx_mat muyT = arma::zeros<arma::cx_mat>(spaceDim, spaceDim);
						for (size_t i = 0; i < spin_count; ++i)
						{
							muxT += mux_list[i];
							muyT += muy_list[i];
						}

						arma::cx_mat muxT_eig = Udag * muxT * eigvec;
						arma::cx_mat muyT_eig = Udag * muyT * eigvec;
						std::vector<arma::cx_mat> mux_eig(spin_count);
						std::vector<arma::cx_mat> muy_eig(spin_count);
						for (size_t i = 0; i < spin_count; ++i)
						{
							mux_eig[i] = Udag * mux_list[i] * eigvec;
							muy_eig[i] = Udag * muy_list[i] * eigvec;
						}

						double loc_total_x = 0.0;
						double loc_total_y = 0.0;
						double loc_total_perp = 0.0;
						double loc_cross_x = 0.0;
						double loc_cross_y = 0.0;
						std::vector<double> loc_spin_x(spin_count, 0.0);
						std::vector<double> loc_spin_y(spin_count, 0.0);
						std::vector<double> loc_spin_perp(spin_count, 0.0);
						std::vector<double> loc_spin_p(spin_count, 0.0);
						std::vector<double> loc_spin_m(spin_count, 0.0);

						for (const auto &trans : transitions)
						{
							const auto m = trans.first;
							const auto n = trans.second;
							const double population = rho_diag(m) - rho_diag(n);
							if (std::abs(population) < 1e-15)
								continue;

							const double deltaOmega = (eigval(n) - eigval(m)) - omega_mw;
							const double abs_domega_dB = std::abs(dHdB_diag(n) - dHdB_diag(m));
							if (!std::isfinite(abs_domega_dB) || abs_domega_dB < 1e-15)
								continue;

							const double dBdE = 1.0 / abs_domega_dB;
							if (dBdE > 1e5)
								continue;

							const double deltaB_mT = 1.0e3 * (deltaOmega * dBdE);
							const double L = this->LineshapeValue(deltaB_mT, lwB_mT);
							if (L == 0.0)
								continue;

							const double wField = dBdE * L;

							const double ITx = std::norm(muxT_eig(m, n));
							const double ITy = std::norm(muyT_eig(m, n));
							double I_sum_x = 0.0;
							double I_sum_y = 0.0;

							for (size_t i = 0; i < spin_count; ++i)
							{
								const arma::cx_double muix = mux_eig[i](m, n);
								const arma::cx_double muiy = muy_eig[i](m, n);
								const double Iix = std::norm(muix);
								const double Iiy = std::norm(muiy);

								I_sum_x += Iix;
								I_sum_y += Iiy;

								loc_spin_x[i] += population * Iix * wField;
								loc_spin_y[i] += population * Iiy * wField;
								loc_spin_perp[i] += population * 0.5 * (Iix + Iiy) * wField;

								const arma::cx_double mup = muix + I * muiy;
								const arma::cx_double mum = muix - I * muiy;
								loc_spin_p[i] += population * std::norm(mup) * wField;
								loc_spin_m[i] += population * std::norm(mum) * wField;
							}

							const double ICx = ITx - I_sum_x;
							const double ICy = ITy - I_sum_y;

							loc_total_x += population * ITx * wField;
							loc_total_y += population * ITy * wField;
							loc_total_perp += population * 0.5 * (ITx + ITy) * wField;
							loc_cross_x += population * ICx * wField;
							loc_cross_y += population * ICy * wField;
						}

						_cache.total_x[step] += w * loc_total_x;
						_cache.total_y[step] += w * loc_total_y;
						_cache.total_perp[step] += w * loc_total_perp;
						_cache.cross_x[step] += w * loc_cross_x;
						_cache.cross_y[step] += w * loc_cross_y;

						for (size_t i = 0; i < spin_count; ++i)
						{
							_cache.spin_x[i][step] += w * loc_spin_x[i];
							_cache.spin_y[i][step] += w * loc_spin_y[i];
							_cache.spin_perp[i][step] += w * loc_spin_perp[i];
							_cache.spin_p[i][step] += w * loc_spin_p[i];
							_cache.spin_m[i][step] += w * loc_spin_m[i];
						}
					}
				}
			}
		}

		return true;
	}

	void TaskStaticHSTrEPRSpectra::WriteHeader(std::ostream &_stream)
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

	bool TaskStaticHSTrEPRSpectra::Validate()
	{
		bool hasFrequency = this->Properties()->Get("mwfrequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			hasFrequency = this->Properties()->Get("frequency", this->mwFrequencyGHz);
		if (!hasFrequency)
			hasFrequency = this->Properties()->Get("rffrequency", this->mwFrequencyGHz);
		if (!hasFrequency)
		{
			this->Log() << "Failed to obtain mwfrequency/frequency. Using frequency = 0 by default." << std::endl;
		}

		const bool hasLinewidth = this->Properties()->Get("linewidth", this->linewidth_mT);
		const bool hasLegacyFad = this->Properties()->Get("linewidth_fad", this->linewidthFad_mT);
		const bool hasLegacyDonor = this->Properties()->Get("linewidth_donor", this->linewidthDonor_mT);
		if (!hasLinewidth)
		{
			if (!hasLegacyFad && !hasLegacyDonor)
			{
				this->Log() << "Failed to obtain linewidth. Using linewidth = 0 by default." << std::endl;
			}
			this->linewidth_mT = 0.5 * (std::abs(this->linewidthFad_mT) + std::abs(this->linewidthDonor_mT));
		}

		if (this->Properties()->Get("lineshape", this->lineshape))
		{
			this->lineshape = ToLower(this->lineshape);
		}
		else
		{
			this->lineshape = "gaussian";
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
		else if (sweepCacheMode == "resfields" || sweepCacheMode == "resfield")
		{
			this->sweepCacheExact = false;
			this->sweepCacheResfields = true;
		}
		else if (sweepCacheMode == "approx" || sweepCacheMode == "approximate" || sweepCacheMode == "crossing" ||
				 sweepCacheMode == "resonance" || sweepCacheMode == "pepper")
		{
			this->sweepCacheExact = false;
			this->sweepCacheResfields = false;
		}
		else
		{
			this->Log() << "Unknown sweepcachemode \"" << sweepCacheMode << "\". Using "
						<< (this->sweepCacheExact ? "exact" : (this->sweepCacheResfields ? "resfields" : "approx")) << "." << std::endl;
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
			this->powderGridType = "fibonacci";
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

		this->detectSpinNames.clear();
		this->Properties()->GetList("detectspins", this->detectSpinNames, ',');
		this->Properties()->Get("electron1", this->electron1Name);
		this->Properties()->Get("electron2", this->electron2Name);

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
}
