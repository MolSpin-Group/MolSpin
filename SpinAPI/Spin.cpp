/////////////////////////////////////////////////////////////////////////
// Spin class (SpinAPI Module)
// ------------------
// Basic spin class to represent e.g. a radical.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <memory>
#include <iostream>
#include <mutex>
#include "ObjectParser.h"
#include "Spin.h"

namespace SpinAPI
{
	namespace
	{
		std::mutex SxCollectionMutex;
		std::mutex SyCollectionMutex;
		std::mutex SzCollectionMutex;
		std::mutex SpCollectionMutex;
		std::mutex SmCollectionMutex;
	}
	// -----------------------------------------------------
	// Spin Constructors and Destructor
	// -----------------------------------------------------
	Spin::Spin(std::string _name, std::string _contents) : tensor(2.0), type(SpinType::NotSpecified), properties(std::make_shared<MSDParser::ObjectParser>(_name, _contents)), s(1),
														   quantizationAxis1({1.0, 0.0, 0.0}), quantizationAxis2({0.0, 1.0, 0.0}), quantizationAxis3({0.0, 0.0, 1.0}), trajectory(),
														   trjHasTime(false), trjHasQAxis1(false), trjHasQAxis2(false), trjHasQAxis3(false), trjTime(0),
														   trjQAxis1X(0), trjQAxis1Y(0), trjQAxis1Z(0), trjQAxis2X(0), trjQAxis2Y(0), trjQAxis2Z(0), trjQAxis3X(0), trjQAxis3Y(0), trjQAxis3Z(0)
	{
		std::string str = "";

		// Get the spin quantum number and get the spin matrices
		if (!this->properties->GetSpin("spin", s))
		{
			// If "spin" was specified, but with an invalid value (e.g. negative), then set spin to 0 as that will make Spin::IsValid return false
			if (this->properties->Get("spin", str))
				s = 0;
		}

		// See if a tensor is specified
		this->properties->Get("tensor", this->tensor);

		// Get the type of the spin
		this->properties->Get("type", str);
		if (str.compare("electron") == 0 || str.compare("e") == 0)
			this->type = SpinType::Electron;
		else if (str.compare("nucleus") == 0 || str.compare("n") == 0)
			this->type = SpinType::Nucleus;

		// Is a trajectory specified?
		if (this->properties->Get("trajectory", str))
		{
			// Get the directory
			std::string directory;
			if (!this->properties->Get("FileReaderDirectory", directory))
			{
				std::cout << "Warning: Failed to obtain input file directory while initializing Spin " << this->Name() << "!\n";
				std::cout << "         This may lead to problems when trying to localize the trajectory file." << std::endl;
			}

			// Load the trajectory, and check whether something was loaded
			if (this->trajectory.Load(directory + str))
			{
				if (this->trajectory.Length() > 0)
				{
					// Get column numbers
					this->trjHasTime = this->trajectory.HasColumn("time", trjTime);
					this->trjHasQAxis1 = this->trajectory.HasColumn("axis1.x", trjQAxis1X);
					this->trjHasQAxis1 &= this->trajectory.HasColumn("axis1.y", trjQAxis1Y);
					this->trjHasQAxis1 &= this->trajectory.HasColumn("axis1.z", trjQAxis1Z);
					this->trjHasQAxis2 = this->trajectory.HasColumn("axis2.x", trjQAxis2X);
					this->trjHasQAxis2 &= this->trajectory.HasColumn("axis2.y", trjQAxis2Y);
					this->trjHasQAxis2 &= this->trajectory.HasColumn("axis2.z", trjQAxis2Z);
					this->trjHasQAxis3 = this->trajectory.HasColumn("axis3.x", trjQAxis3X);
					this->trjHasQAxis3 &= this->trajectory.HasColumn("axis3.y", trjQAxis3Y);
					this->trjHasQAxis3 &= this->trajectory.HasColumn("axis3.z", trjQAxis3Z);
				}
			}
			else
			{
				std::cout << "ERROR: Failed to load trajectory \"" << (directory + str) << "\" for Spin object \"" << this->Name() << "\"!" << std::endl;
			}
		}

		// Quantization axes can be specified either directly or through a trajectory
		arma::vec inAxis = arma::zeros<arma::vec>(3);
		if (!this->trjHasQAxis1 && this->properties->Get("quantizationaxis1", inAxis))
			this->quantizationAxis1 = inAxis;
		if (!this->trjHasQAxis2 && this->properties->Get("quantizationaxis2", inAxis))
			this->quantizationAxis2 = inAxis;
		if (!this->trjHasQAxis3 && this->properties->Get("quantizationaxis3", inAxis))
			this->quantizationAxis3 = inAxis;
	}

	Spin::Spin(const Spin &_spin) : tensor(_spin.tensor), type(_spin.type), properties(std::make_shared<MSDParser::ObjectParser>(*(this->properties))), s(_spin.s),
									quantizationAxis1(_spin.quantizationAxis1), quantizationAxis2(_spin.quantizationAxis2), quantizationAxis3(_spin.quantizationAxis3),
									trajectory(_spin.trajectory), trjHasTime(_spin.trjHasTime), trjHasQAxis1(_spin.trjHasQAxis1), trjHasQAxis2(_spin.trjHasQAxis2),
									trjHasQAxis3(_spin.trjHasQAxis3), trjTime(_spin.trjTime), trjQAxis1X(_spin.trjQAxis1X), trjQAxis1Y(_spin.trjQAxis1Y),
									trjQAxis1Z(_spin.trjQAxis1Z), trjQAxis2X(_spin.trjQAxis2X), trjQAxis2Y(_spin.trjQAxis2Y), trjQAxis2Z(_spin.trjQAxis2Z),
									trjQAxis3X(_spin.trjQAxis3X), trjQAxis3Y(_spin.trjQAxis3Y), trjQAxis3Z(_spin.trjQAxis3Z)
	{
	}

	Spin::~Spin()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	// Copy-assignment
	const Spin &Spin::operator=(const Spin &_spin)
	{
		this->tensor = _spin.tensor;
		this->type = _spin.type;
		this->properties = std::make_shared<MSDParser::ObjectParser>(*(_spin.properties));
		this->s = _spin.s;
		this->quantizationAxis1 = _spin.quantizationAxis1;
		this->quantizationAxis2 = _spin.quantizationAxis2;
		this->quantizationAxis3 = _spin.quantizationAxis3;
		this->trajectory = _spin.trajectory;

		this->trjHasTime = _spin.trjHasTime;
		this->trjHasQAxis1 = _spin.trjHasQAxis1;
		this->trjHasQAxis2 = _spin.trjHasQAxis2;
		this->trjHasQAxis3 = _spin.trjHasQAxis3;
		this->trjTime = _spin.trjTime;
		this->trjQAxis1X = _spin.trjQAxis1X;
		this->trjQAxis1Y = _spin.trjQAxis1Y;
		this->trjQAxis1Z = _spin.trjQAxis1Z;
		this->trjQAxis2X = _spin.trjQAxis2X;
		this->trjQAxis2Y = _spin.trjQAxis2Y;
		this->trjQAxis2Z = _spin.trjQAxis2Z;
		this->trjQAxis3X = _spin.trjQAxis3X;
		this->trjQAxis3Y = _spin.trjQAxis3Y;
		this->trjQAxis3Z = _spin.trjQAxis3Z;

		return (*this);
	}
	// -----------------------------------------------------
	// Private methods
	// -----------------------------------------------------
	// Makes sure the quantization axes are orthonormal
	void Spin::OrthonormalizeQAxes()
	{
		// Make Axis1 and Axis2 orthogonal
		arma::vec tmp = dot(this->quantizationAxis1, this->quantizationAxis2) * this->quantizationAxis1;
		this->quantizationAxis2 -= tmp;

		// Make Axis3 orthogonal to Axis1 and Axis2
		tmp = dot(this->quantizationAxis1, this->quantizationAxis3) * this->quantizationAxis1 + dot(this->quantizationAxis2, this->quantizationAxis3) * this->quantizationAxis2;
		this->quantizationAxis3 -= tmp;

		// Normalize the axes
		this->quantizationAxis1 /= arma::norm(this->quantizationAxis1);
		this->quantizationAxis2 /= arma::norm(this->quantizationAxis2);
		this->quantizationAxis3 /= arma::norm(this->quantizationAxis3);
	}
	// -----------------------------------------------------
	// Name and validation
	// -----------------------------------------------------
	std::string Spin::Name() const
	{
		return this->properties->Name();
	}

	// The spin is valid if the spin quantum number has an allowed value
	// WRONG STATEMENT: A spin-0 is not allowed as it would not make sense (it has no spin-space dynamics)
	// We have included spin-0 as it is still correct and also helpful when dealing with multispin systems of electronic levels
	bool Spin::IsValid()
	{
		// Commented out as we are now beyond this restriction
		// if (s == 0)
		//	return false;

		// Please include incoming validation steps in this function here.

		return true;
	}
	// -----------------------------------------------------
	// Public Spin methods
	// -----------------------------------------------------
	// Sets the trajectory step for the Spin object and its tensor
	bool Spin::SetTrajectoryStep(unsigned int _step)
	{
		// Update g-tensor from its trajectory, based on a single step in the trajectory
		bool status = this->tensor.SetTrajectoryStep(_step);

		// Make sure that we have a trajectory in order to do anything further
		if (this->trajectory.Length() < 1)
			return status;

		// Make sure that the requested step exists
		if (_step >= this->trajectory.Length())
			return status;

		// Get the axes
		if (this->trjHasQAxis1)
		{
			this->quantizationAxis1(0) = this->trajectory.Get(_step, this->trjQAxis1X);
			this->quantizationAxis1(1) = this->trajectory.Get(_step, this->trjQAxis1Y);
			this->quantizationAxis1(2) = this->trajectory.Get(_step, this->trjQAxis1Z);
		}
		if (this->trjHasQAxis2)
		{
			this->quantizationAxis2(0) = this->trajectory.Get(_step, this->trjQAxis2X);
			this->quantizationAxis2(1) = this->trajectory.Get(_step, this->trjQAxis2Y);
			this->quantizationAxis2(2) = this->trajectory.Get(_step, this->trjQAxis2Z);
		}
		if (this->trjHasQAxis3)
		{
			this->quantizationAxis3(0) = this->trajectory.Get(_step, this->trjQAxis3X);
			this->quantizationAxis3(1) = this->trajectory.Get(_step, this->trjQAxis3Y);
			this->quantizationAxis3(2) = this->trajectory.Get(_step, this->trjQAxis3Z);
		}

		// Make sure that the axes are still orthonormal
		if (this->trjHasQAxis1 || this->trjHasQAxis2 || this->trjHasQAxis3)
			this->OrthonormalizeQAxes();

		return true;
	}

	// Sets the time for the Spin object and its tensor
	bool Spin::SetTime(double _time)
	{
		// Update g-tensor from its trajectory, based on time (if specified in trajectory)
		bool status = this->tensor.SetTime(_time);

		// The time must be specified in the trajectory in order to do anything further
		if (!this->trjHasTime)
			return status;

		// Get the row corresponding to the time right after (or at) the requested time
		unsigned int row = 0;
		double timestepAbove = 0.0;
		if (!this->trajectory.FirstRowEqGreaterThan(_time, this->trjTime, row, timestepAbove))
		{
			// If no such time was found, i.e. we are past the last element of the trajectory, use the last step of the trajectory
			status |= this->SetTrajectoryStep(this->trajectory.Length() - 1);
		}
		else if (row == 0)
		{
			// If the specified time is before the first element of the trajectory, use the first step of the trajectory
			status |= this->SetTrajectoryStep(0);
		}
		else
		{
			double timestepBelow = this->trajectory.Get(row - 1, trjTime);			  // Get time at previous step
			double l = 1 - (timestepAbove - _time) / (timestepAbove - timestepBelow); // Get linear interpolation factor, i.e. a number between 0 and 1, where 0 = previous time step and 1 = next time step in the trajectory

			// Get the axes
			if (this->trjHasQAxis1)
			{
				this->quantizationAxis1(0) = this->trajectory.Get(row, this->trjQAxis1X) * l + this->trajectory.Get(row - 1, this->trjQAxis1X) * (1 - l);
				this->quantizationAxis1(1) = this->trajectory.Get(row, this->trjQAxis1Y) * l + this->trajectory.Get(row - 1, this->trjQAxis1Y) * (1 - l);
				this->quantizationAxis1(2) = this->trajectory.Get(row, this->trjQAxis1Z) * l + this->trajectory.Get(row - 1, this->trjQAxis1Z) * (1 - l);
			}
			if (this->trjHasQAxis2)
			{
				this->quantizationAxis2(0) = this->trajectory.Get(row, this->trjQAxis2X) * l + this->trajectory.Get(row - 1, this->trjQAxis2X) * (1 - l);
				this->quantizationAxis2(1) = this->trajectory.Get(row, this->trjQAxis2Y) * l + this->trajectory.Get(row - 1, this->trjQAxis2Y) * (1 - l);
				this->quantizationAxis2(2) = this->trajectory.Get(row, this->trjQAxis2Z) * l + this->trajectory.Get(row - 1, this->trjQAxis2Z) * (1 - l);
			}
			if (this->trjHasQAxis3)
			{
				this->quantizationAxis3(0) = this->trajectory.Get(row, this->trjQAxis3X) * l + this->trajectory.Get(row - 1, this->trjQAxis3X) * (1 - l);
				this->quantizationAxis3(1) = this->trajectory.Get(row, this->trjQAxis3Y) * l + this->trajectory.Get(row - 1, this->trjQAxis3Y) * (1 - l);
				this->quantizationAxis3(2) = this->trajectory.Get(row, this->trjQAxis3Z) * l + this->trajectory.Get(row - 1, this->trjQAxis3Z) * (1 - l);
			}

			// Make sure that the axes are still orthonormal
			if (this->trjHasQAxis1 || this->trjHasQAxis2 || this->trjHasQAxis3)
				this->OrthonormalizeQAxes();

			status = true;
		}

		return status;
	}

	// Returns the length of the g-tensor trajectory
	unsigned int Spin::TrajectoryLength() const
	{
		return std::max(this->tensor.TrajectoryLength(), this->trajectory.Length());
	}

	// Returns the first quantization axis
	arma::vec Spin::QuantizationAxis1() const
	{
		return this->quantizationAxis1;
	}

	// Returns the second quantization axis
	arma::vec Spin::QuantizationAxis2() const
	{
		return this->quantizationAxis2;
	}

	// Returns the third quantization axis
	arma::vec Spin::QuantizationAxis3() const
	{
		return this->quantizationAxis3;
	}

	// Returns a matrix where the quantization axes are the columns
	arma::mat Spin::QuantizationAxes() const
	{
		arma::mat axes = arma::eye<arma::mat>(3, 3);
		axes.col(0) = this->quantizationAxis1;
		axes.col(1) = this->quantizationAxis2;
		axes.col(2) = this->quantizationAxis3;

		return axes;
	}
	// -----------------------------------------------------
	// Access to custom properties
	// -----------------------------------------------------
	std::shared_ptr<const MSDParser::ObjectParser> Spin::Properties() const
	{
		return this->properties;
	}
	// -----------------------------------------------------
	// Public spin matrix methods
	// -----------------------------------------------------
	// Returns the Sx matrix for the spin
	const arma::sp_cx_mat Spin::Sx() const
	{
		return (*(SxFromCollection(this->s)));
	}

	// And similarly for the Sy spin matrix
	const arma::sp_cx_mat Spin::Sy() const
	{
		return (*(SyFromCollection(this->s)));
	}

	// And the Sz spin matrix
	const arma::sp_cx_mat Spin::Sz() const
	{
		return (*(SzFromCollection(this->s)));
	}

	// The S+ spin ladder operator
	const arma::sp_cx_mat Spin::Sp() const
	{
		return (*(SpFromCollection(this->s)));
	}

	// The S- spin ladder operator
	const arma::sp_cx_mat Spin::Sm() const
	{
		return (*(SmFromCollection(this->s)));
	}

	// Returns the transformed magnetic moment; (Sx,Sy,Sz)-vector multiplied by the tensor
	const arma::sp_cx_mat Spin::Tx() const
	{
		// First quantization axis
		arma::sp_cx_mat QASx = this->quantizationAxis1(0) * this->Sx() + this->quantizationAxis1(1) * this->Sy() + this->quantizationAxis1(2) * this->Sz();

		if (IsIsotropic(this->tensor))
			return (QASx * this->tensor.Isotropic());

		// Get the other quantization axes
		arma::sp_cx_mat QASy = this->quantizationAxis2(0) * this->Sx() + this->quantizationAxis2(1) * this->Sy() + this->quantizationAxis2(2) * this->Sz();
		arma::sp_cx_mat QASz = this->quantizationAxis3(0) * this->Sx() + this->quantizationAxis3(1) * this->Sy() + this->quantizationAxis3(2) * this->Sz();

		auto mat = this->tensor.LabFrame();
		return (QASx * mat(0, 0) + QASy * mat(0, 1) + QASz * mat(0, 2));
	}

	// Returns the transformed magnetic moment; (Sx,Sy,Sz)-vector multiplied by the tensor
	const arma::sp_cx_mat Spin::Ty() const
	{
		// Second quantization axis
		arma::sp_cx_mat QASy = this->quantizationAxis2(0) * this->Sx() + this->quantizationAxis2(1) * this->Sy() + this->quantizationAxis2(2) * this->Sz();

		if (IsIsotropic(this->tensor))
			return (QASy * this->tensor.Isotropic());

		// Get the other quantization axes
		arma::sp_cx_mat QASx = this->quantizationAxis1(0) * this->Sx() + this->quantizationAxis1(1) * this->Sy() + this->quantizationAxis1(2) * this->Sz();
		arma::sp_cx_mat QASz = this->quantizationAxis3(0) * this->Sx() + this->quantizationAxis3(1) * this->Sy() + this->quantizationAxis3(2) * this->Sz();

		auto mat = this->tensor.LabFrame();
		return (QASx * mat(1, 0) + QASy * mat(1, 1) + QASz * mat(1, 2));
	}

	// Returns the transformed magnetic moment; (Sx,Sy,Sz)-vector multiplied by the tensor
	const arma::sp_cx_mat Spin::Tz() const
	{
		// Third quantization axis
		arma::sp_cx_mat QASz = this->quantizationAxis3(0) * this->Sx() + this->quantizationAxis3(1) * this->Sy() + this->quantizationAxis3(2) * this->Sz();

		if (IsIsotropic(this->tensor))
			return (QASz * this->tensor.Isotropic());

		// Get the other quantization axes
		arma::sp_cx_mat QASx = this->quantizationAxis1(0) * this->Sx() + this->quantizationAxis1(1) * this->Sy() + this->quantizationAxis1(2) * this->Sz();
		arma::sp_cx_mat QASy = this->quantizationAxis2(0) * this->Sx() + this->quantizationAxis2(1) * this->Sy() + this->quantizationAxis2(2) * this->Sz();

		auto mat = this->tensor.LabFrame();
		return (QASx * mat(2, 0) + QASy * mat(2, 1) + QASz * mat(2, 2));
	}
	// -----------------------------------------------------
	// Public method to create ActionTarget objects
	// -----------------------------------------------------
	// Create ActionTargets
	void Spin::GetActionTargets(std::vector<RunSection::NamedActionScalar> &_scalars, std::vector<RunSection::NamedActionVector> &_vectors, std::string _system)
	{
		// Get ActionTargets from the associated Tensor
		this->tensor.GetActionTargets(_scalars, _vectors, _system + "." + this->Name());

		// Get ActionTargets for the Spin object
		std::vector<RunSection::NamedActionVector> vectors;

		// Quantization axes form one orthonormal frame. Expose them to output
		// objects, but do not let independent vector actions corrupt that frame.
		// A future mutable interface must rotate all three axes transactionally.
		RunSection::ActionVector qaxis1av = RunSection::ActionVector(this->quantizationAxis1, nullptr, true);
		RunSection::ActionVector qaxis2av = RunSection::ActionVector(this->quantizationAxis2, nullptr, true);
		RunSection::ActionVector qaxis3av = RunSection::ActionVector(this->quantizationAxis3, nullptr, true);
		vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".quantizationaxis1", qaxis1av));
		vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".quantizationaxis2", qaxis2av));
		vectors.push_back(RunSection::NamedActionVector(_system + "." + this->Name() + ".quantizationaxis3", qaxis3av));

		// Insert the ActionTargets
		_vectors.insert(_vectors.end(), vectors.begin(), vectors.end());
	}
	// -----------------------------------------------------
	// Static private members
	// -----------------------------------------------------
	// Initialize the collections
	std::map<unsigned int, std::shared_ptr<arma::sp_cx_mat>> Spin::SxCollection;
	std::map<unsigned int, std::shared_ptr<arma::sp_cx_mat>> Spin::SyCollection;
	std::map<unsigned int, std::shared_ptr<arma::sp_cx_mat>> Spin::SzCollection;
	std::map<unsigned int, std::shared_ptr<arma::sp_cx_mat>> Spin::SpCollection;
	std::map<unsigned int, std::shared_ptr<arma::sp_cx_mat>> Spin::SmCollection;

	// Get the Sx matrix, or create it from S+ and S-
	std::shared_ptr<arma::sp_cx_mat> Spin::SxFromCollection(unsigned int _s)
	{
		// Sx can be created from S+ and S-
		auto sp = Spin::SpFromCollection(_s);
		auto sm = Spin::SmFromCollection(_s);

		// If S+ or S- could not be obtained, we cannot create Sx
		if (sp == nullptr || sm == nullptr)
			return nullptr;

		std::lock_guard<std::mutex> lock(SxCollectionMutex);
		auto it = Spin::SxCollection.find(_s);
		if (it == Spin::SxCollection.end())
		{
			it = Spin::SxCollection.emplace(_s, std::make_shared<arma::sp_cx_mat>(((*sp) + (*sm)) / 2.0)).first;
		}

		// Return the matrix
		return it->second;
	}

	// Get the Sy matrix, or create it from S+ and S-
	std::shared_ptr<arma::sp_cx_mat> Spin::SyFromCollection(unsigned int _s)
	{
		// Sy can be created from S+ and S-
		auto sp = Spin::SpFromCollection(_s);
		auto sm = Spin::SmFromCollection(_s);

		// If S+ or S- could not be obtained, we cannot create Sy
		if (sp == nullptr || sm == nullptr)
			return nullptr;

		std::lock_guard<std::mutex> lock(SyCollectionMutex);
		auto it = Spin::SyCollection.find(_s);
		if (it == Spin::SyCollection.end())
		{
			it = Spin::SyCollection.emplace(_s, std::make_shared<arma::sp_cx_mat>(((*sp) - (*sm)) / arma::cx_double(0.0, 2.0))).first;
		}

		// Return the matrix
		return it->second;
	}

	// Get the Sz matrix
	std::shared_ptr<arma::sp_cx_mat> Spin::SzFromCollection(unsigned int _s)
	{
		std::lock_guard<std::mutex> lock(SzCollectionMutex);
		auto it = Spin::SzCollection.find(_s);
		if (it == Spin::SzCollection.end())
		{
			// Create and fill a vector containing the diagonal values
			arma::cx_vec diagvec(_s + 1);
			for (unsigned int i = 0; i < _s + 1; i++)
				diagvec[i] = ((double)_s) / 2.0 - ((double)i);

			// Create the matrix and set the diagonal values
			std::shared_ptr<arma::sp_cx_mat> tempmat = std::make_shared<arma::sp_cx_mat>(_s + 1, _s + 1);
			tempmat->diag() = diagvec;

			// Put the matrix into the collection
			it = Spin::SzCollection.emplace(_s, tempmat).first;
		}

		// Return the matrix
		return it->second;
	}

	// Get the S+ matrix
	std::shared_ptr<arma::sp_cx_mat> Spin::SpFromCollection(unsigned int _s)
	{
		std::lock_guard<std::mutex> lock(SpCollectionMutex);
		auto it = Spin::SpCollection.find(_s);
		if (it == Spin::SpCollection.end())
		{
			// Create a matrix and fill it will the right values
			std::shared_ptr<arma::sp_cx_mat> tempmat = std::make_shared<arma::sp_cx_mat>(_s + 1, _s + 1);
			double s = static_cast<double>(_s) / 2.0; // Get "s" quantum number in units of hbar
			for (unsigned int i = 1; i < _s + 1; i++)
			{
				double m = s - static_cast<double>(i);
				(*tempmat)(i - 1, i) = sqrt(s * (s + 1.0) - m * (m + 1.0)); // S+|s,m> = hbar * sqrt(s(s+1) - m(m+1))|s,m+1>
			}

			// Put the matrix into the collection
			it = Spin::SpCollection.emplace(_s, tempmat).first;
		}

		// Return the matrix
		return it->second;
	}

	// Get the S- matrix
	std::shared_ptr<arma::sp_cx_mat> Spin::SmFromCollection(unsigned int _s)
	{
		std::lock_guard<std::mutex> lock(SmCollectionMutex);
		auto it = Spin::SmCollection.find(_s);
		if (it == Spin::SmCollection.end())
		{
			// Create a matrix and fill it will the right values
			std::shared_ptr<arma::sp_cx_mat> tempmat = std::make_shared<arma::sp_cx_mat>(_s + 1, _s + 1);
			double s = static_cast<double>(_s) / 2.0; // Get "s" quantum number in units of hbar
			for (unsigned int i = 0; i < _s; i++)
			{
				double m = s - static_cast<double>(i);
				(*tempmat)(i + 1, i) = sqrt(s * (s + 1.0) - m * (m - 1.0)); // S-|s,m> = hbar * sqrt(s(s+1) - m(m-1))|s,m-1>
			}

			// Put the matrix into the collection
			it = Spin::SmCollection.emplace(_s, tempmat).first;
		}

		// Return the matrix
		return it->second;
	}
	// -----------------------------------------------------
	// Non-member non-friend functions
	// -----------------------------------------------------
	bool HasTrajectory(const Spin &_spin)
	{
		return (_spin.TrajectoryLength() > 0);
	}

	bool IsIsotropic(const Spin &_spin)
	{
		return IsIsotropic(_spin.GetTensor());
	}
	// -----------------------------------------------------
}
