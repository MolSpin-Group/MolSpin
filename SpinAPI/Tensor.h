/////////////////////////////////////////////////////////////////////////
// Tensor class (SpinAPI Module)
// ------------------
// A basic tensor class. Stores the data as isotropic and anisotropic
// values, and principal axes.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Tensor
#define MOD_SpinAPI_Tensor

#include <array>
#include <armadillo>
#include "Trajectory.h"
#include "ActionTarget.h"

namespace SpinAPI
{
	class Tensor
	{
	private:
		// Implementation details
		double isotropic;	   // Isotropic part of principal values (trace / 3.0)
		arma::vec anisotropic; // Traceless principal values / eigenvalues
		arma::vec axis1;	   // Principal axis definitions in the lab frame
		arma::vec axis2;
		arma::vec axis3;
		arma::vec mat1;
		arma::vec mat2;
		arma::vec mat3;
		Trajectory trajectory;

		// Trajectory columns
		bool trjHasTime;
		bool trjHasIsotropic;
		bool trjHasAnisotropic;
		bool trjHasAxis1;
		bool trjHasAxis2;
		bool trjHasAxis3;

		bool trjHasMatXX;
		bool trjHasMatXY;
		bool trjHasMatXZ;
		bool trjHasMatYX;
		bool trjHasMatYY;
		bool trjHasMatYZ;
		bool trjHasMatZX;
		bool trjHasMatZY;
		bool trjHasMatZZ;

		unsigned int trjTime;
		unsigned int trjIsotropic;
		unsigned int trjAnisotropicX;
		unsigned int trjAnisotropicY;
		unsigned int trjAnisotropicZ;
		unsigned int trjAxis1X;
		unsigned int trjAxis1Y;
		unsigned int trjAxis1Z;
		unsigned int trjAxis2X;
		unsigned int trjAxis2Y;
		unsigned int trjAxis2Z;
		unsigned int trjAxis3X;
		unsigned int trjAxis3Y;
		unsigned int trjAxis3Z;

		unsigned int trjMatXX;
		unsigned int trjMatXY;
		unsigned int trjMatXZ;
		unsigned int trjMatYX;
		unsigned int trjMatYY;
		unsigned int trjMatYZ;
		unsigned int trjMatZX;
		unsigned int trjMatZY;
		unsigned int trjMatZZ;

		// Private methods
		void DiagonalizeMatrix(const arma::mat &);					// Set the tensor from a matrix
		void SeparateIsotropy();									// Separate the isotropic value from the anisotropic values
		void OrthonormalizeAxes();									// Makes axes orthonormal
		double DoubleFromString(std::string) const;					// Convert string to double, or return 0.0 if no conversion was possible
		bool ParseTensor(const std::string &, const std::string &); // Get the values from the first string - the second string sets the path used to load a trajectory

		// Private methods to create ActionTargets
		std::vector<RunSection::NamedActionScalar> CreateActionScalars(const std::string &);
		std::vector<RunSection::NamedActionVector> CreateActionVectors(const std::string &);

	public:
		// Constructors / Destructors
		explicit Tensor(double);											 // Normal constructor with isotropic tensor
		Tensor(double, double, double, double);								 // Normal constructor with isotropic+anisotropic part
		Tensor(double, const arma::vec &);									 // Normal constructor with isotropic+anisotropic part
		Tensor(double, const arma::vec &, const arma::mat &);				 // Normal constructor with isotropic+anisotropic part + axes
		explicit Tensor(const arma::mat &);									 // Normal constructor from matrix (iso+aniso+axes are deduced)
		explicit Tensor(const std::string &, const std::string &_path = ""); // Normal constructor parsing tensor from string - the second string sets the path used to load a trajectory
		Tensor(const Tensor &);												 // Copy-constructor
		~Tensor();															 // Destructor

		// Operators
		const Tensor &operator=(const Tensor &); // Copy-assignment

		// Public methods
		double Isotropic() const;
		arma::vec Anisotropic() const;
		arma::vec Axis1() const;
		arma::vec Axis2() const;
		arma::vec Axis3() const;
		arma::mat Axes() const;
		arma::mat LabFrame() const; // Returns the matrix representing the tensor in the lab frame

		// Spherical tensor representation components
		arma::cx_double SphericalT0() const;  // T0
		arma::cx_double SphericalTp1() const; // T+1
		arma::cx_double SphericalTm1() const; // T-1
		arma::cx_double SphericalTp2() const; // T+2
		arma::cx_double SphericalTm2() const; // T-2

		// Loads a trajectory and gets the column numbers of the headers. Fails if there already was loaded a trajectory unless _overwrite is set to true, which removes the old trajectory
		bool LoadTrajectory(const std::string &, const std::string &_path = "", bool _overwrite = false);

		// Trajectory handling
		unsigned int TrajectoryLength() const; // Number of steps in the trajectory for tensor (0 means no trajectory)
		bool SetTrajectoryStep(unsigned int);  // Set value from trajectory based on step number
		bool SetTime(double);				   // Set value from trajectory based on time

		// Methods for time-dependent interactions
		void SetTensor(arma::mat &); // Set value from lab frame matrix
		arma::mat GetTensor();		 // return lab frame matrix

		// Public method for creating ActionTargets
		void GetActionTargets(std::vector<RunSection::NamedActionScalar> &, std::vector<RunSection::NamedActionVector> &, const std::string &);
	};

	// Non-member non-friend functions
	bool HasTrajectory(const Tensor &);
	bool IsIsotropic(const Tensor &); // Tensor isotropy

	//SemiClassical specific functions 
	typedef std::array<std::array<double,3>,3> SCMatrix3x3;
	///A SemiClassical operator specific function that parses a string defined in the semi-classical field and returns a 3x3 matrix
	///@param input_str A string defined as "isotropic(...)" or "anisotropic(... ... ...)" or "matrix(... ... ...; ... ... ...; ... ... ...)"
	///@param return_matrix A 3x3 double array
	///@return a boolean to confirm correct parsing - will currently only accept isotropic tensors due to incomplete theory
	bool Parse3x3TensorLikeArrayFromString(const std::string& input_str, SCMatrix3x3& return_matrix);

	// Non-member non-friend functions for ActionTarget validation
	bool CheckActionScalarTensorIsotropicPart(const double &);
}

#endif
