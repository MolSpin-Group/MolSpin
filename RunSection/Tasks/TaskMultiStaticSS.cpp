/////////////////////////////////////////////////////////////////////////
// TaskMultiStaticSS implementation (RunSection module)
//
// Integrated steady-state calculation for a network of SpinSystems.
// Optional powder averaging rotates every molecular tensor and every state
// explicitly declared to live in the molecular frame.
//
// Molecular Spin Dynamics Software.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include "TaskMultiStaticSS.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "Interaction.h"
#include "ObjectParser.h"
#include "Operator.h"
#include "Settings.h"
#include "State.h"
#include "Transition.h"

namespace RunSection
{
	TaskMultiStaticSS::TaskMultiStaticSS(const MSDParser::ObjectParser &_parser, const RunSection &_runsection)
		: BasicTask(_parser, _runsection),
		  productYieldsOnly(false),
		  reactionOperators(SpinAPI::ReactionOperatorType::Haberkorn),
		  hamiltonianMode(HamiltonianMode::FullFixed),
		  linearSolver(LinearSolver::Automatic),
		  powderDomain(SpinAPI::PowderGridDomain::FullSphere),
		  powderGridType("uniform"),
		  powderSymmetry("c1"),
		  powderSamplingPoints(1),
		  powderGridSize(4),
		  normalizePowderWeights(true),
		  diagnostics(true),
		  solverResidualTolerance(1.0e-8),
		  denseSolverThreshold(192)
	{
	}

	TaskMultiStaticSS::~TaskMultiStaticSS()
	{
	}

	// ---------------------------------------------------------------------
	// Main workflow
	// ---------------------------------------------------------------------
	bool TaskMultiStaticSS::RunLocal()
	{
		this->Log() << "Running method StaticSS-MultiSystem." << std::endl;

		if (this->RunSettings()->CurrentStep() == 1)
			this->WriteHeader(this->Data());

		// 1. Build one common crystallite grid. Every SpinSystem in the reaction
		// network represents the same molecule/encounter orientation, so using
		// independent grids for individual blocks would destroy frame covariance.
		SpinAPI::PowderGrid grid;
		bool explicitGrid = false;
		if (!this->CreatePowderGrid(grid, explicitGrid))
			return false;

		// 2. Cache subsystem dimensions, initial-state definitions, and the
		// contiguous offsets used by the direct-sum Liouville space.
		std::vector<SystemContext> contexts;
		arma::uword totalDimension = 0;
		if (!this->PrepareSystemContexts(contexts, totalDimension))
			return false;

		if (this->diagnostics)
		{
			this->Log() << "MultiStaticSS configuration: Hamiltonian=" << this->HamiltonianModeName()
						<< ", orientations=" << grid.size()
						<< (explicitGrid ? " (explicit)" : " (internal)")
						<< ", direct-sum Liouville dimension=" << totalDimension << "." << std::endl;
		}

		// 3. For each crystallite, assemble L, solve
		//
		//        L * integral_0^infinity rho(t) dt = -rho(0),
		//
		// and project the integrated density onto the requested observables.
		std::vector<double> powderAverage;
		for (size_t orientationIndex = 0; orientationIndex < grid.size(); ++orientationIndex)
		{
			const SpinAPI::PowderOrientation &orientation = grid[orientationIndex];
			arma::mat rotation;
			if (!SpinAPI::CreateZYZRotationMatrix(0.0, orientation.theta, orientation.phi, rotation))
			{
				this->Log() << "Failed to create powder rotation matrix for orientation "
							<< orientationIndex << "." << std::endl;
				return false;
			}

			arma::sp_cx_mat liouvillian;
			arma::cx_vec initialDensity;
			if (!this->BuildLiouvillian(contexts, rotation, liouvillian, initialDensity))
				return false;

			arma::cx_vec integratedDensity;
			if (!this->SolveIntegratedDensity(liouvillian, initialDensity, integratedDensity))
				return false;

			std::vector<double> projected;
			if (!this->ProjectOrientation(contexts, rotation, integratedDensity, projected))
				return false;

			if (powderAverage.empty())
				powderAverage.assign(projected.size(), 0.0);
			if (powderAverage.size() != projected.size())
			{
				this->Log() << "Internal error: orientation-dependent output column count changed." << std::endl;
				return false;
			}

			for (size_t column = 0; column < projected.size(); ++column)
				powderAverage[column] += orientation.weight * projected[column];

			if (this->diagnostics &&
				(grid.size() <= 10 || orientationIndex == 0 || orientationIndex + 1 == grid.size()))
			{
				this->Log() << "Completed orientation " << orientationIndex + 1 << "/" << grid.size()
							<< " (theta=" << orientation.theta << ", phi=" << orientation.phi
							<< ", normalized weight=" << orientation.weight << ")." << std::endl;
			}
		}

		// 4. A task step produces one row. Actions may change fields, rates, or
		// tensors before the next step, at which point all matrices are rebuilt.
		this->Data() << this->RunSettings()->CurrentStep() << " ";
		this->WriteStandardOutput(this->Data());
		for (double value : powderAverage)
			this->Data() << value << " ";
		this->Data() << std::endl;

		return true;
	}

	// ---------------------------------------------------------------------
	// Input validation and output layout
	// ---------------------------------------------------------------------
	bool TaskMultiStaticSS::Validate()
	{
		this->productYieldsOnly = false;
		this->Properties()->Get("transitionyields", this->productYieldsOnly);

		std::string value;
		if (this->Properties()->Get("reactionoperators", value))
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
						   { return static_cast<char>(std::tolower(c)); });
			if (value == "haberkorn")
				this->reactionOperators = SpinAPI::ReactionOperatorType::Haberkorn;
			else if (value == "lindblad")
				this->reactionOperators = SpinAPI::ReactionOperatorType::Lindblad;
			else
			{
				this->Log() << "Unknown reactionoperators value \"" << value << "\"." << std::endl;
				return false;
			}
		}

		bool powderRequested = false;
		this->Properties()->Get("powderaveraging", powderRequested);
		bool explicitHamiltonianMode = this->Properties()->Get("hamiltonianmode", value) ||
									 this->Properties()->Get("hamiltonian_mode", value);
		if (explicitHamiltonianMode)
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
						   { return static_cast<char>(std::tolower(c)); });
			if (value == "full" || value == "fixed" || value == "legacy")
				this->hamiltonianMode = HamiltonianMode::FullFixed;
			else if (value == "rotated_zyz" || value == "rotatedzyz" || value == "rotated_full" ||
					 value == "exact" || value == "nonsecular")
				this->hamiltonianMode = HamiltonianMode::RotatedFull;
			else if (value == "rotated_sa" || value == "rotatedsa" || value == "sa" ||
					 value == "secular" || value == "highfield")
				this->hamiltonianMode = HamiltonianMode::RotatedSecular;
			else
			{
				this->Log() << "Unknown hamiltonianmode \"" << value
							<< "\". Use full, rotated_zyz, or rotated_sa." << std::endl;
				return false;
			}
		}
		else if (powderRequested)
		{
			// Exact rotation is the physically general default when a user asks
			// for powder averaging without explicitly requesting high field.
			this->hamiltonianMode = HamiltonianMode::RotatedFull;
		}

		this->Properties()->Get("powdersamplingpoints", this->powderSamplingPoints);
		this->Properties()->Get("powdergridsize", this->powderGridSize);
		this->Properties()->Get("normalizepowderweights", this->normalizePowderWeights);
		this->Properties()->Get("diagnostics", this->diagnostics);
		this->Properties()->Get("solverresidualtolerance", this->solverResidualTolerance);
		this->Properties()->Get("densesolverthreshold", this->denseSolverThreshold);

		if (powderRequested && this->powderSamplingPoints == 1)
			this->powderSamplingPoints = 100;
		if (this->powderSamplingPoints < 1 || this->powderGridSize < 1)
		{
			this->Log() << "Powder grid sizes must be positive." << std::endl;
			return false;
		}
		if (!(this->solverResidualTolerance > 0.0) || !std::isfinite(this->solverResidualTolerance))
		{
			this->Log() << "solverresidualtolerance must be finite and positive." << std::endl;
			return false;
		}

		if (this->Properties()->Get("powdergrid", this->powderGridType) ||
			this->Properties()->Get("powder_grid", this->powderGridType))
		{
			std::transform(this->powderGridType.begin(), this->powderGridType.end(), this->powderGridType.begin(), [](unsigned char c)
						   { return static_cast<char>(std::tolower(c)); });
		}
		if (this->powderGridType != "uniform" && this->powderGridType != "golden" &&
			this->powderGridType != "octant" && this->powderGridType != "sophe")
		{
			this->Log() << "Unknown powdergrid \"" << this->powderGridType
						<< "\". Use uniform, octant, or sophe." << std::endl;
			return false;
		}
		this->Properties()->Get("powdersymmetry", this->powderSymmetry);

		if (this->Properties()->Get("powderdomain", value))
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
						   { return static_cast<char>(std::tolower(c)); });
			if (value == "full" || value == "fullsphere" || value == "sphere")
				this->powderDomain = SpinAPI::PowderGridDomain::FullSphere;
			else if (value == "upper" || value == "hemisphere" || value == "upperhemisphere")
				this->powderDomain = SpinAPI::PowderGridDomain::UpperHemisphere;
			else
			{
				this->Log() << "Unknown powderdomain \"" << value << "\"." << std::endl;
				return false;
			}
		}

		if (this->Properties()->Get("linearsolver", value) || this->Properties()->Get("solver", value))
		{
			std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
						   { return static_cast<char>(std::tolower(c)); });
			if (value == "auto" || value == "automatic")
				this->linearSolver = LinearSolver::Automatic;
			else if (value == "sparse" || value == "superlu")
				this->linearSolver = LinearSolver::Sparse;
			else if (value == "dense" || value == "lapack")
				this->linearSolver = LinearSolver::Dense;
			else
			{
				this->Log() << "Unknown linear solver \"" << value << "\"." << std::endl;
				return false;
			}
		}

		for (const auto &system : this->SpinSystems())
		{
			SpinAPI::SpinSpace space(system);
			if (space.HasTimedependentInteractions() || space.HasTimedependentTransitions())
			{
				this->Log() << "StaticSS-MultiSystem cannot use time-dependent interactions or transitions in SpinSystem \""
							<< system->Name() << "\"." << std::endl;
				return false;
			}
		}

		if (this->hamiltonianMode == HamiltonianMode::RotatedSecular)
		{
			this->Log() << "Using the high-field secular Hamiltonian. This approximation is not validated automatically; "
						<< "use rotated_zyz when Zeeman energies do not dominate ZFS, hyperfine, and dipolar couplings."
						<< std::endl;
		}

		return true;
	}

	void TaskMultiStaticSS::WriteHeader(std::ostream &_stream)
	{
		_stream << "Step ";
		this->WriteStandardOutputHeader(_stream);

		for (const auto &system : this->SpinSystems())
		{
			if (this->productYieldsOnly)
			{
				for (const auto &transition : system->Transitions())
				if (transition->SourceState() != nullptr)
					_stream << system->Name() << "." << transition->Name() << ".yield ";
			}
			else
			{
				for (const auto &state : system->States())
					_stream << system->Name() << "." << state->Name() << " ";
			}
		}
		if (this->productYieldsOnly)
			_stream << "SumYield ";
		_stream << std::endl;
	}

	// ---------------------------------------------------------------------
	// Powder grid and subsystem preparation
	// ---------------------------------------------------------------------
	bool TaskMultiStaticSS::CreatePowderGrid(SpinAPI::PowderGrid &_grid, bool &_explicitGrid)
	{
		_explicitGrid = this->ReadExplicitPowderGrid(_grid);
		if (!_explicitGrid)
		{
			if (this->hamiltonianMode == HamiltonianMode::FullFixed || this->powderSamplingPoints == 1)
			{
				_grid = {SpinAPI::IdentityPowderOrientation()};
			}
			else if (this->powderGridType == "uniform" || this->powderGridType == "golden")
			{
				if (!SpinAPI::CreateUniformPowderGrid(this->powderSamplingPoints, this->powderDomain, _grid))
					return false;
			}
			else if (this->powderGridType == "octant")
			{
				if (!SpinAPI::CreateOctantPowderGrid(this->powderSamplingPoints, _grid))
					return false;
			}
			else if (this->powderGridType == "sophe")
			{
				if (!SpinAPI::CreateSophePowderGrid(this->powderGridSize, this->powderSymmetry, _grid))
					return false;
			}
		}

		if (_grid.empty())
		{
			this->Log() << "Powder grid contains no orientations." << std::endl;
			return false;
		}

		double weightSum = 0.0;
		for (const auto &orientation : _grid)
		{
			if (!std::isfinite(orientation.theta) || !std::isfinite(orientation.phi) ||
				!std::isfinite(orientation.weight) || orientation.weight < 0.0)
			{
				this->Log() << "Powder orientation angles and weights must be finite; weights must be nonnegative."
							<< std::endl;
				return false;
			}
			weightSum += orientation.weight;
		}

		if (!(weightSum > 0.0))
		{
			this->Log() << "Powder orientation weights sum to zero." << std::endl;
			return false;
		}
		if (this->normalizePowderWeights)
			for (auto &orientation : _grid)
				orientation.weight /= weightSum;

		if (this->hamiltonianMode == HamiltonianMode::FullFixed && _explicitGrid)
		{
			this->Log() << "Note: hamiltonianmode=full preserves the legacy lab-frame Hamiltonian, so the explicit "
						<< "orientation does not rotate tensors. Use rotated_zyz or rotated_sa for orientation dependence."
						<< std::endl;
		}

		return true;
	}

	bool TaskMultiStaticSS::PrepareSystemContexts(std::vector<SystemContext> &_contexts, arma::uword &_totalDimension)
	{
		_contexts.clear();
		_totalDimension = 0;
		double totalPreparedPopulation = 0.0;

		for (const auto &system : this->SpinSystems())
		{
			SystemContext context;
			context.system = system;
			context.space = std::make_shared<SpinAPI::SpinSpace>(system);
			context.space->SetReactionOperatorType(this->reactionOperators);
			context.space->UseSuperoperatorSpace(true);
			context.offset = _totalDimension;
			context.superDimension = context.space->SpaceDimensions();
			context.initialStates = system->InitialState();
			context.initialFrame = system->InitialStateFrame();
			context.dephaseInitialState =
				(system->InitialStateCoherences() == SpinAPI::InitialStateCoherenceMode::DephaseEigenbasis);
			if (context.initialFrame == SpinAPI::StateFrame::Eigen &&
				(context.initialStates.size() != 1 || context.initialStates.front() != nullptr))
			{
				this->Log() << "Initial-state frame=eigen for SpinSystem \"" << system->Name()
							<< "\" requires exactly one Thermal initial state." << std::endl;
				return false;
			}
			if (system->GetProperties() != nullptr)
				system->GetProperties()->Get("initialpopulation", context.initialPopulation);
			if (!std::isfinite(context.initialPopulation) || context.initialPopulation < 0.0)
			{
				this->Log() << "initialpopulation for SpinSystem \"" << system->Name()
							<< "\" must be finite and nonnegative." << std::endl;
				return false;
			}

			context.initialWeights = system->Weights();
			if (!context.initialStates.empty())
			{
				if (context.initialWeights.size() != context.initialStates.size())
					context.initialWeights.assign(context.initialStates.size(), 1.0);

				double sum = 0.0;
				for (double weight : context.initialWeights)
				{
					if (!std::isfinite(weight) || weight < 0.0)
					{
						this->Log() << "Initial-state weights for SpinSystem \"" << system->Name()
									<< "\" must be finite and nonnegative." << std::endl;
						return false;
					}
					sum += weight;
				}
				if (!(sum > 0.0))
				{
					this->Log() << "Initial-state weights for SpinSystem \"" << system->Name()
								<< "\" sum to zero." << std::endl;
					return false;
				}
				for (double &weight : context.initialWeights)
					weight /= sum;
			}

			if (this->diagnostics)
			{
				const char *frame = context.initialFrame == SpinAPI::StateFrame::Molecular
										? "molecular"
										: (context.initialFrame == SpinAPI::StateFrame::Eigen ? "eigen" : "fixed");
				this->Log() << "SpinSystem \"" << system->Name() << "\": Hilbert dimension="
							<< context.space->HilbertSpaceDimensions() << ", Liouville dimension="
							<< context.superDimension << ", initial-state frame=" << frame;
				if (context.initialStates.empty())
					this->Log() << ", no initial state (zero population)";
				else
					this->Log() << ", initial population=" << context.initialPopulation;
				this->Log()
							<< (context.dephaseInitialState ? ", eigenbasis dephasing enabled." : ".")
							<< std::endl;
			}

			if (!context.initialStates.empty())
				totalPreparedPopulation += context.initialPopulation;
			_totalDimension += context.superDimension;
			_contexts.push_back(std::move(context));
		}

		if (_contexts.empty() || _totalDimension == 0)
		{
			this->Log() << "StaticSS-MultiSystem requires at least one nonempty SpinSystem." << std::endl;
			return false;
		}
		if (this->diagnostics)
		{
			this->Log() << "Total prepared direct-sum population=" << totalPreparedPopulation << ".";
			if (std::abs(totalPreparedPopulation - 1.0) > 1.0e-10)
				this->Log() << " Integrated observables scale linearly with this value.";
			this->Log() << std::endl;
		}
		return true;
	}

	// ---------------------------------------------------------------------
	// Orientation-specific physics
	// ---------------------------------------------------------------------
	bool TaskMultiStaticSS::BuildStaticHamiltonian(SystemContext &_context, const arma::mat &_rotation, arma::sp_cx_mat &_hamiltonian)
	{
		_context.space->UseSuperoperatorSpace(false);

		bool built = false;
		if (this->hamiltonianMode == HamiltonianMode::FullFixed)
			built = _context.space->StaticHamiltonian(_hamiltonian);
		else if (this->hamiltonianMode == HamiltonianMode::RotatedFull)
			built = _context.space->StaticHamiltonianRotatedZYZ(_rotation, _hamiltonian);
		else
			built = _context.space->StaticHamiltonianRotatedSA(_rotation, _hamiltonian);

		if (!built)
		{
			this->Log() << "Failed to build " << this->HamiltonianModeName()
						<< " Hamiltonian for SpinSystem \"" << _context.system->Name()
						<< "\". Semiclassical-field interactions are not supported by this steady-state powder builder."
						<< std::endl;
			return false;
		}

		const arma::cx_mat dense(_hamiltonian);
		const double scale = std::max(1.0, arma::norm(dense, "fro"));
		if (arma::norm(dense - dense.t(), "fro") > 1.0e-10 * scale)
		{
			this->Log() << "Hamiltonian for SpinSystem \"" << _context.system->Name()
						<< "\" is not Hermitian after orientation assembly." << std::endl;
			return false;
		}
		return true;
	}

	bool TaskMultiStaticSS::PrepareInitialDensity(SystemContext &_context,
												  const arma::mat &_rotation,
												  const arma::sp_cx_mat &_hamiltonian,
												  arma::cx_mat &_density)
	{
		const arma::uword hilbertDimension = _context.space->HilbertSpaceDimensions();
		_density.zeros(hilbertDimension, hilbertDimension);
		if (_context.initialStates.empty())
			return true;

		for (size_t index = 0; index < _context.initialStates.size(); ++index)
		{
			arma::cx_mat contribution;
			const SpinAPI::state_ptr &state = _context.initialStates[index];
			if (state == nullptr)
			{
				std::vector<std::string> thermalList = _context.system->ThermalHamiltonianList();
				arma::cx_mat thermalHamiltonian;
				if (thermalList.empty())
				{
					thermalHamiltonian = arma::cx_mat(_hamiltonian);
				}
				else
				{
					arma::sp_cx_mat selected;
					bool built = false;
					if (this->hamiltonianMode == HamiltonianMode::FullFixed)
						built = _context.space->ThermalHamiltonian(thermalList, selected);
					else if (this->hamiltonianMode == HamiltonianMode::RotatedFull)
						built = _context.space->BaseHamiltonianRotatedZYZ(thermalList, _rotation, selected);
					else
						built = _context.space->BaseHamiltonianRotated_SA(thermalList, _rotation, selected);
					if (!built)
					{
						this->Log() << "Failed to build the thermal Hamiltonian for SpinSystem \""
									<< _context.system->Name() << "\"." << std::endl;
						return false;
					}
					thermalHamiltonian = arma::cx_mat(selected);
				}

				if (!_context.space->ThermalStateFromHamiltonian(
						thermalHamiltonian, _context.system->Temperature(), contribution))
				{
					this->Log() << "Failed to build the thermal initial state for SpinSystem \""
								<< _context.system->Name() << "\"." << std::endl;
					return false;
				}
			}
			else if (!this->OrientedStateProjector(
						 _context, state, _rotation, _context.initialFrame, contribution))
			{
				this->Log() << "Failed to build initial state \"" << state->Name()
							<< "\" for SpinSystem \"" << _context.system->Name() << "\"." << std::endl;
				return false;
			}

			_density += _context.initialWeights[index] * contribution;
		}

		const arma::cx_double trace = arma::trace(_density);
		if (!std::isfinite(trace.real()) || !std::isfinite(trace.imag()) ||
			std::abs(trace.imag()) > 1.0e-10 * std::max(1.0, std::abs(trace.real())) ||
			trace.real() <= 0.0)
		{
			this->Log() << "Initial density matrix for SpinSystem \"" << _context.system->Name()
						<< "\" has invalid trace " << trace << "." << std::endl;
			return false;
		}
		_density /= trace.real();
		_density = 0.5 * (_density + _density.t());

		if (_context.dephaseInitialState)
		{
			arma::cx_mat dephased;
			if (!_context.space->DephaseStateInEigenbasis(_density, arma::cx_mat(_hamiltonian), dephased))
			{
				this->Log() << "Failed to dephase the initial state of SpinSystem \""
							<< _context.system->Name() << "\" in its orientation-specific eigenbasis." << std::endl;
				return false;
			}
			_density = std::move(dephased);
		}

		// A reaction network may start in more than one independently prepared
		// manifold (for example PP and triplet-pair encounter populations).
		// The state mixture inside each manifold is trace-normalized first;
		// initialpopulation then assigns its population in the direct sum.
		_density *= _context.initialPopulation;

		return true;
	}

	bool TaskMultiStaticSS::BuildReactionLoss(SystemContext &_context,
											  const arma::mat &_rotation,
											  arma::sp_cx_mat &_loss)
	{
		_context.space->UseSuperoperatorSpace(true);
		_loss.zeros(_context.superDimension, _context.superDimension);

		for (const auto &transition : _context.system->Transitions())
		{
			if (transition->SourceState() == nullptr || !transition->IsActive())
				continue;

			arma::cx_mat projectorDense;
			const SpinAPI::StateFrame frame =
				this->TransitionStateFrame(transition, _context, false);
			if (!this->OrientedStateProjector(
					_context, transition->SourceState(), _rotation, frame, projectorDense))
				return false;

			const arma::sp_cx_mat projector =
				arma::conv_to<arma::sp_cx_mat>::from(projectorDense);
			arma::sp_cx_mat left;
			arma::sp_cx_mat right;
			if (!_context.space->SuperoperatorFromLeftOperator(projector, left) ||
				!_context.space->SuperoperatorFromRightOperator(projector, right))
				return false;

			// For a direct-sum reaction network, both Haberkorn loss and the
			// loss half of a Lindblad jump are -k/2{P,rho}. If a target system
			// exists, BuildCreationOperator adds the matching +k C rho C^dagger
			// block, making the represented intersystem transfer trace preserving.
			_loss += 0.5 * transition->Rate() * (left + right);
		}

		return true;
	}

	bool TaskMultiStaticSS::BuildCreationOperator(const SpinAPI::transition_ptr &_transition,
												  SystemContext &_source,
												  SystemContext &_target,
												  const arma::mat &_rotation,
												  arma::sp_cx_mat &_creation)
	{
		if (_transition == nullptr || _transition->TargetState() == nullptr || !_transition->IsActive())
			return false;

		arma::cx_vec sourceState;
		arma::cx_vec targetState;
		if (!this->OrientedStateVector(
				_source, _transition->SourceState(), _rotation,
				this->TransitionStateFrame(_transition, _source, false), sourceState))
			return false;
		if (!this->OrientedStateVector(
				_target, _transition->TargetState(), _rotation,
				this->TransitionStateFrame(_transition, _target, true), targetState))
			return false;

		const double sourceNorm = arma::norm(sourceState);
		const double targetNorm = arma::norm(targetState);
		if (!(sourceNorm > 0.0) || !(targetNorm > 0.0))
			return false;
		sourceState /= sourceNorm;
		targetState /= targetNorm;

		const arma::sp_cx_mat jump =
			arma::conv_to<arma::sp_cx_mat>::from(targetState * sourceState.t());
		_source.space->UseSuperoperatorSpace(true);
		if (!_source.space->SuperoperatorFromOperators(jump, jump.t(), _creation))
			return false;

		_creation *= _transition->Rate();
		return true;
	}

	bool TaskMultiStaticSS::BuildLiouvillian(std::vector<SystemContext> &_contexts,
											 const arma::mat &_rotation,
											 arma::sp_cx_mat &_liouvillian,
											 arma::cx_vec &_initialDensity)
	{
		const arma::uword totalDimension =
			_contexts.back().offset + _contexts.back().superDimension;
		_liouvillian.zeros(totalDimension, totalDimension);
		_initialDensity.zeros(totalDimension);

		// Diagonal blocks contain coherent motion, reaction loss, and optional
		// relaxation. The initial density is prepared from the same Hamiltonian
		// and crystallite rotation before conversion to Liouville space.
		for (auto &context : _contexts)
		{
			arma::sp_cx_mat hamiltonian;
			if (!this->BuildStaticHamiltonian(context, _rotation, hamiltonian))
				return false;

			arma::cx_mat density;
			if (!this->PrepareInitialDensity(context, _rotation, hamiltonian, density))
				return false;

			context.space->UseSuperoperatorSpace(true);
			arma::cx_vec densityVector;
			if (!context.space->OperatorToSuperspace(density, densityVector))
				return false;
			_initialDensity.subvec(
				context.offset, context.offset + context.superDimension - 1) = densityVector;

			arma::sp_cx_mat left;
			arma::sp_cx_mat right;
			if (!context.space->SuperoperatorFromLeftOperator(hamiltonian, left) ||
				!context.space->SuperoperatorFromRightOperator(hamiltonian, right))
				return false;
			arma::sp_cx_mat diagonal =
				arma::cx_double(0.0, -1.0) * (left - right);

			arma::sp_cx_mat reactionLoss;
			if (!this->BuildReactionLoss(context, _rotation, reactionLoss))
				return false;
			diagonal -= reactionLoss;

			if (context.system->operators_cbegin() != context.system->operators_cend())
			{
				arma::vec eigenvalues;
				arma::cx_mat eigenvectors;
				if (!arma::eig_sym(eigenvalues, eigenvectors, arma::cx_mat(hamiltonian)))
				{
					this->Log() << "Failed to diagonalize the Hamiltonian for relaxation in SpinSystem \""
								<< context.system->Name() << "\"." << std::endl;
					return false;
				}

				for (auto iterator = context.system->operators_cbegin();
					 iterator != context.system->operators_cend(); ++iterator)
				{
					arma::sp_cx_mat relaxation;
					if (!context.space->PowderRelaxationOperator(
							*iterator, eigenvectors, _rotation, relaxation))
					{
						this->Log() << "Failed to build relaxation operator \"" << (*iterator)->Name()
									<< "\" for SpinSystem \"" << context.system->Name() << "\"." << std::endl;
						return false;
					}
					diagonal += relaxation;
				}
			}

			_liouvillian.submat(
				context.offset, context.offset,
				context.offset + context.superDimension - 1,
				context.offset + context.superDimension - 1) += diagonal;
		}

		// Off-diagonal blocks are jump maps from a source SpinSystem into the
		// target SpinSystem named by each transition. Multiple transitions
		// between the same pair are accumulated, never overwritten.
		for (auto &source : _contexts)
		{
			for (const auto &transition : source.system->Transitions())
			{
				if (transition->Target() == nullptr)
					continue;

				auto targetIterator = std::find_if(
					_contexts.begin(), _contexts.end(),
					[&](const SystemContext &candidate)
					{ return candidate.system == transition->Target(); });
				if (targetIterator == _contexts.end())
				{
					this->Log() << "Transition \"" << transition->Name()
								<< "\" targets a SpinSystem outside this task." << std::endl;
					return false;
				}

				arma::sp_cx_mat creation;
				if (!this->BuildCreationOperator(
						transition, source, *targetIterator, _rotation, creation))
				{
					this->Log() << "Failed to build creation operator for transition \""
								<< transition->Name() << "\"." << std::endl;
					return false;
				}

				_liouvillian.submat(
					targetIterator->offset, source.offset,
					targetIterator->offset + targetIterator->superDimension - 1,
					source.offset + source.superDimension - 1) += creation;
			}
		}

		if (this->diagnostics)
		{
			const double entries = static_cast<double>(totalDimension) * static_cast<double>(totalDimension);
			const double density = entries > 0.0
									   ? static_cast<double>(_liouvillian.n_nonzero) / entries
									   : 0.0;
			this->Log() << "Assembled Liouvillian: dimension=" << totalDimension
						<< ", nonzeros=" << _liouvillian.n_nonzero
						<< ", density=" << density << "." << std::endl;
		}

		return true;
	}

	// ---------------------------------------------------------------------
	// Linear solve and observable projection
	// ---------------------------------------------------------------------
	bool TaskMultiStaticSS::SolveIntegratedDensity(const arma::sp_cx_mat &_liouvillian,
												   const arma::cx_vec &_initialDensity,
												   arma::cx_vec &_integratedDensity)
	{
		if (_liouvillian.n_rows != _liouvillian.n_cols ||
			_liouvillian.n_rows != _initialDensity.n_rows)
			return false;

		const arma::cx_vec rhs = -_initialDensity;
		const double entryCount =
			static_cast<double>(_liouvillian.n_rows) * static_cast<double>(_liouvillian.n_cols);
		const double matrixDensity = entryCount > 0.0
										 ? static_cast<double>(_liouvillian.n_nonzero) / entryCount
										 : 1.0;

		bool useSparse = this->linearSolver == LinearSolver::Sparse;
		if (this->linearSolver == LinearSolver::Automatic)
			useSparse = _liouvillian.n_rows > this->denseSolverThreshold && matrixDensity < 0.35;

		bool solved = false;
		const char *usedSolver = "dense LAPACK";
		if (useSparse)
		{
			usedSolver = "sparse SuperLU";
			try
			{
				solved = arma::spsolve(_integratedDensity, _liouvillian, rhs, "superlu");
			}
			catch (const std::runtime_error &)
			{
				solved = false;
			}

			if (!solved && this->linearSolver == LinearSolver::Automatic)
			{
				usedSolver = "dense LAPACK fallback";
				solved = arma::solve(
					_integratedDensity, arma::cx_mat(_liouvillian), rhs,
					arma::solve_opts::no_approx);
			}
		}
		else
		{
			solved = arma::solve(
				_integratedDensity, arma::cx_mat(_liouvillian), rhs,
				arma::solve_opts::no_approx);
		}

		if (!solved || !_integratedDensity.is_finite())
		{
			this->Log() << "Failed to solve the integrated steady-state equation with "
						<< usedSolver << ". The reaction network may contain a nondecaying subspace."
						<< std::endl;
			return false;
		}

		const double rhsNorm = std::max(1.0, arma::norm(rhs, 2));
		const double relativeResidual =
			arma::norm(_liouvillian * _integratedDensity - rhs, 2) / rhsNorm;
		if (!std::isfinite(relativeResidual) ||
			relativeResidual > this->solverResidualTolerance)
		{
			this->Log() << "Integrated-density solve failed its residual check: "
						<< relativeResidual << " > " << this->solverResidualTolerance << "."
						<< std::endl;
			return false;
		}

		if (this->diagnostics)
			this->Log() << "Solved with " << usedSolver
						<< "; relative residual=" << relativeResidual << "." << std::endl;
		return true;
	}

	bool TaskMultiStaticSS::ProjectOrientation(std::vector<SystemContext> &_contexts,
											   const arma::mat &_rotation,
											   const arma::cx_vec &_integratedDensity,
											   std::vector<double> &_values)
	{
		_values.clear();
		double sumYield = 0.0;
		bool projectionsValid = true;

		for (auto &context : _contexts)
		{
			const arma::cx_vec vector = _integratedDensity.subvec(
				context.offset, context.offset + context.superDimension - 1);
			arma::cx_mat density;
			context.space->UseSuperoperatorSpace(true);
			if (!context.space->OperatorFromSuperspace(vector, density))
				return false;

			if (this->productYieldsOnly)
			{
				for (const auto &transition : context.system->Transitions())
				{
					if (transition->SourceState() == nullptr)
						continue;

					arma::cx_mat projector;
					if (!this->OrientedStateProjector(
							context, transition->SourceState(), _rotation,
							this->TransitionStateFrame(transition, context, false), projector))
						return false;

					const double population = this->RealProjection(
						arma::trace(projector * density),
						context.system->Name() + "." + transition->Name() + ".yield",
						projectionsValid);
					const double yield = transition->Rate() * population;
					_values.push_back(yield);
					sumYield += yield;
				}
			}
			else
			{
				for (const auto &state : context.system->States())
				{
					arma::cx_mat projector;
					if (!this->OrientedStateProjector(
							context, state, _rotation,
							this->ObservableStateFrame(context, state), projector))
						return false;
					_values.push_back(this->RealProjection(
						arma::trace(projector * density),
						context.system->Name() + "." + state->Name(),
						projectionsValid));
				}
			}
		}

		if (this->productYieldsOnly)
			_values.push_back(sumYield);
		return projectionsValid;
	}

	// ---------------------------------------------------------------------
	// Frame-aware state helpers
	// ---------------------------------------------------------------------
	bool TaskMultiStaticSS::OrientedStateProjector(SystemContext &_context,
												   const SpinAPI::state_ptr &_state,
												   const arma::mat &_rotation,
												   SpinAPI::StateFrame _frame,
												   arma::cx_mat &_projector)
	{
		if (_state == nullptr || !_context.space->GetState(_state, _projector))
			return false;

		if (_frame != SpinAPI::StateFrame::Molecular ||
			this->hamiltonianMode == HamiltonianMode::FullFixed)
			return true;

		arma::cx_mat rotated;
		if (!_context.space->RotateState(_projector, _rotation, rotated))
			return false;
		_projector = std::move(rotated);
		return true;
	}

	bool TaskMultiStaticSS::OrientedStateVector(SystemContext &_context,
												const SpinAPI::state_ptr &_state,
												const arma::mat &_rotation,
												SpinAPI::StateFrame _frame,
												arma::cx_vec &_vector)
	{
		if (_state == nullptr || !_context.space->GetState(_state, _vector))
			return false;

		if (_frame != SpinAPI::StateFrame::Molecular ||
			this->hamiltonianMode == HamiltonianMode::FullFixed)
			return true;

		// State-vector phases are irrelevant to C rho C^dagger. Rotating the
		// rank-one projector and recovering its occupied eigenvector avoids a
		// second, task-local implementation of spin rotation generators.
		arma::cx_mat projector = _vector * _vector.t();
		arma::cx_mat rotated;
		if (!_context.space->RotateState(projector, _rotation, rotated))
			return false;

		arma::vec eigenvalues;
		arma::cx_mat eigenvectors;
		if (!arma::eig_sym(eigenvalues, eigenvectors, rotated) || eigenvalues.empty())
			return false;
		_vector = eigenvectors.col(eigenvalues.index_max());
		return true;
	}

	SpinAPI::StateFrame TaskMultiStaticSS::SystemStateFrame(
		const SystemContext &_context,
		const std::string &_kind,
		SpinAPI::StateFrame _fallback) const
	{
		if (_context.system->GetProperties() == nullptr)
			return _fallback;

		std::string value;
		bool found = false;
		if (_kind == "transition")
			found = _context.system->GetProperties()->Get("transitionstateframe", value) ||
					_context.system->GetProperties()->Get("transition_state_frame", value);
		else if (_kind == "observable")
			found = _context.system->GetProperties()->Get("observablestateframe", value) ||
					_context.system->GetProperties()->Get("observable_state_frame", value);
		if (!found)
			return _fallback;

		return this->ParseStateFrame(value, _fallback);
	}

	SpinAPI::StateFrame TaskMultiStaticSS::TransitionStateFrame(
		const SpinAPI::transition_ptr &_transition,
		const SystemContext &_context,
		bool _target) const
	{
		SpinAPI::StateFrame fallback =
			this->SystemStateFrame(_context, "transition", SpinAPI::StateFrame::Fixed);
		if (_transition == nullptr || _transition->Properties() == nullptr)
			return fallback;

		std::string value;
		const bool found = _target
							   ? (_transition->Properties()->Get("targetframe", value) ||
								  _transition->Properties()->Get("target_state_frame", value))
							   : (_transition->Properties()->Get("sourceframe", value) ||
								  _transition->Properties()->Get("source_state_frame", value));
		if (!found &&
			!_transition->Properties()->Get("transitionstateframe", value) &&
			!_transition->Properties()->Get("stateframe", value))
			return fallback;

		return this->ParseStateFrame(value, fallback);
	}

	SpinAPI::StateFrame TaskMultiStaticSS::ObservableStateFrame(
		const SystemContext &_context,
		const SpinAPI::state_ptr &_state) const
	{
		SpinAPI::StateFrame fallback =
			this->SystemStateFrame(_context, "observable", SpinAPI::StateFrame::Fixed);
		if (_state == nullptr || _state->Properties() == nullptr)
			return fallback;

		std::string value;
		if (!_state->Properties()->Get("observableframe", value) &&
			!_state->Properties()->Get("observable_state_frame", value))
			return fallback;

		return this->ParseStateFrame(value, fallback);
	}

	SpinAPI::StateFrame TaskMultiStaticSS::ParseStateFrame(
		const std::string &_value,
		SpinAPI::StateFrame _fallback) const
	{
		std::string value = _value;
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c)
					   { return static_cast<char>(std::tolower(c)); });
		if (value == "molecular" || value == "mol" || value == "rotating")
			return SpinAPI::StateFrame::Molecular;
		if (value == "eigen" || value == "thermal")
			return SpinAPI::StateFrame::Eigen;
		if (value == "fixed" || value == "lab" || value == "laboratory")
			return SpinAPI::StateFrame::Fixed;
		return _fallback;
	}

	double TaskMultiStaticSS::RealProjection(
		const arma::cx_double &_value,
		const std::string &_label,
		bool &_valid)
	{
		const double scale = std::max(1.0, std::abs(_value.real()));
		if (!std::isfinite(_value.real()) || !std::isfinite(_value.imag()) ||
			std::abs(_value.imag()) > 1.0e-8 * scale)
		{
			this->Log() << "Observable \"" << _label << "\" has a non-real projection "
						<< _value << "." << std::endl;
			_valid = false;
		}

		return std::abs(_value.real()) < 1.0e-14 ? 0.0 : _value.real();
	}

	const char *TaskMultiStaticSS::HamiltonianModeName() const
	{
		if (this->hamiltonianMode == HamiltonianMode::RotatedFull)
			return "rotated_zyz (exact/nonsecular)";
		if (this->hamiltonianMode == HamiltonianMode::RotatedSecular)
			return "rotated_sa (high-field secular)";
		return "full (legacy fixed frame)";
	}
}
