/////////////////////////////////////////////////////////////////////////
// Trajectory class (SpinAPI Module)
// ------------------
// A general trajectory class used to load all kinds of trajectories.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_SpinAPI_Trajectory
#define MOD_SpinAPI_Trajectory

#include <map>
#include <istream>
#include <string>
#include <vector>

namespace SpinAPI
{
	class Trajectory
	{
	private:
		// Implementation details
		std::string filename;
		std::map<std::string, unsigned int> headers;
		std::vector<std::vector<double>> data;

		// Private character validation methods
		bool isAlphanumericCharacter(const char &) const;
		bool isNumericCharacter(const char &) const;
		bool isDelimiter(const char &) const;

		void AddNumber(unsigned int, std::string); // Puts a number into the next column of the row, provided there are enough headers
		void SetHeader(unsigned int, std::string);

	public:
		// Constructors / Destructors
		Trajectory();
		Trajectory(const Trajectory &); // Copy-constructor
		~Trajectory();

		// Operators
		const Trajectory &operator=(const Trajectory &); // Copy-assignment

		// Public methods
		bool Load(const std::string &); // Loads the given trajectory file
		// Strict opt-in reader for experimental data. The historical Load()
		// remains permissive for backward compatibility.
		bool LoadStrict(const std::string &_filename, std::string &_error);
		bool LoadStrict(std::istream &_input, std::string &_error);
		bool Clear();					// Removes any loaded trajectory data

		// TODO: Consider using size_t or vector::size_type instead of unsigned int
		// Note that row and column numbers are zero-based
		// Get methods return zero if the row or column does not exist
		double Get(const unsigned int _row, const unsigned int _column) const;
		double Get(const unsigned int _row, const std::string &_column) const;
		double Get(const int _row, const int _column) const;
		double Get(const int _row, const std::string &_column) const;

		bool HasColumn(std::string _header) const;
		bool HasColumn(std::string _header, unsigned int &_headerColumn) const;

		// Sets the uint& to the first row that is equal or greater/less than the given value,
		// and sets the double& to the value found in that row
		// Returns false if not value was found satifying the constraint (greater/less than)
		bool FirstRowEqGreaterThan(double _value, unsigned int _column, unsigned int &_row, double &_firstGreaterValue) const;
		bool FirstRowEqGreaterThan(double _value, std::string _column, unsigned int &_row, double &_firstGreaterValue) const;
		bool FirstRowEqLessThan(double _value, unsigned int _column, unsigned int &_row, double &_firstLessValue) const;
		bool FirstRowEqLessThan(double _value, std::string _column, unsigned int &_row, double &_firstLessValue) const;

		unsigned int Length() const; // Returns number of rows
	};

	// Non-member non-friend functions
}

#endif
