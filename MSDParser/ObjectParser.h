/////////////////////////////////////////////////////////////////////////
// ObjectParser class (MSDParser Module)
// ------------------
// Parses the contents of objects, to provide easy access to object
// contents for the SpinAPI classes.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_MSDParser_ObjectParser
#define MOD_MSDParser_ObjectParser

#include <map>
#include <vector>
#include <armadillo>
#include "SpinAPIfwd.h"

namespace MSDParser
{
	class ObjectParser
	{
	private:
		// Implementation
		std::map<std::string, std::string> fields;
		std::string name;

	public:
		// Constructors / Destructors
		ObjectParser(std::string, std::string); // Normal constructor
		ObjectParser(const ObjectParser &);		// Default Copy-constructor
		~ObjectParser();						// Destructor

		// Operators
		ObjectParser &operator=(const ObjectParser &); // Default Copy-assignment
		// Public methods to get a property
		bool Get(const std::string &, std::string &) const;
		bool Get(const std::string &, double &) const;
		bool Get(const std::string &, unsigned int &) const;
		bool Get(const std::string &, int &) const;
		bool Get(const std::string &, bool &) const;
		bool Get(const std::string &, SpinAPI::Tensor &) const;
		bool Get(const std::string &, arma::vec &) const;
		bool GetList(const std::string &, std::vector<std::string> &, const char &_delimiter = ',') const;
		bool GetList(const std::string &, std::vector<double> &, const char &_delimiter = ',') const;
		bool GetList(const std::string &, std::vector<unsigned int> &, const char &_delimiter = ',') const;
		bool GetList(const std::string &, std::vector<int> &, const char &_delimiter = ',') const;
		bool GetList(const std::string &, std::vector<arma::vec> &, const char &_delimiter = ',') const;
		bool GetList(const std::string &, std::vector<bool> &, const char &_delimiter = ',') const;
		bool GetMatrix(const std::string &_str, arma::mat &_out) const;
		bool GetPulseSequence(const std::string &_str, std::vector<std::tuple<std::string, double>> &_out) const;
		bool GetSpin(const std::string &, unsigned int &) const; // Allows for notation such as "1/2"->1, "1"->2, "3/2"->3, etc.

		std::vector<std::pair<std::string, std::string>> GetFunction(std::string) const; // TODO: Invent a better name for this method

		std::string Name() const;
	};
}

#endif
