/////////////////////////////////////////////////////////////////////////
// Object Parser implementations, provides easy access to properties for
// the loaded objects.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <regex>
#include "Tensor.h"
#include "ObjectParser.h"
#include "SpinAPIfwd.h"

namespace MSDParser
{
	// -----------------------------------------------------
	// ObjectParser Constructors and Destructor
	// -----------------------------------------------------
	// Initialize the ObjectParser with a string
	ObjectParser::ObjectParser(std::string _name, std::string _contents) : fields(), name(_name)
	{
		int parantheses = 0;
		std::string key("");
		std::string value("");
		bool isKey = true;
		auto i = _contents.cbegin();

		while (i != _contents.cend())
		{
			if (isKey)
			{
				if ((*i) == '=') // Finding '=' means we got the key and then need the value
					isKey = false;
				else if ((*i) == ';') // Finding ';' in a key is wrong, so skip the key
					key.clear();
				else
					key += (*i);
			}
			else
			{
				// Track parantheses such that a value of a matrix string, e.g. "1 2;3 4", also can be read (specifed by "matrix(...)" for Tensor objects)
				if ((*i) == '(')
				{
					++parantheses;
				}
				else if ((*i) == ')')
				{
					--parantheses;
				}

				if ((*i) == ';' && parantheses < 1)
				{
					// Make sure that neither the key nor the value is empty before using them
					if (key.size() > 0 && value.size() > 0)
					{
						this->fields[key] = value;
					}
					// Clear buffers and reset
					key.clear();
					value.clear();
					isKey = true;
				}
				else
				{
					value += (*i);
				}
			}
			i++;
		}
	}

	// Copy-constructor
	// Copies the values through the initializer list
	ObjectParser::ObjectParser(const ObjectParser &_objParser) : fields(_objParser.fields), name(_objParser.name)
	{
	}

	// Destructor
	ObjectParser::~ObjectParser()
	{
	}

	// -----------------------------------------------------
	// ObjectParser public Get method overloads
	// -----------------------------------------------------
	// Attempt to find a keyword matching the given name
	bool ObjectParser::Get(const std::string &_str, std::string &_out) const
	{
		auto i = this->fields.find(_str);
		if (i != this->fields.end())
		{
			_out = i->second;
		}
		else
			return false;
		return true;
	}

	// Attemp to find a double with the given name
	bool ObjectParser::Get(const std::string &_str, double &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
			return false;

		double tmp;

		try
		{
			tmp = std::stod(str);
		}
		catch (const std::exception &)
		{
			return false;
		}
		// vector<double>
		_out = tmp;
		return true;
	}

	// Attemp to find an unsigned integer with the given name
	bool ObjectParser::Get(const std::string &_str, unsigned int &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
			return false;

		unsigned int tmp;

		try
		{
			if (std::stoi(str) < 0)
				return false;
			tmp = static_cast<unsigned int>(std::stoi(str));
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// Attemp to find an integer with the given name
	bool ObjectParser::Get(const std::string &_str, int &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
			return false;

		int tmp;

		try
		{
			tmp = std::stoi(str);
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// Attemp to find a bool with the given name
	bool ObjectParser::Get(const std::string &_str, bool &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
			return false;

		if (str.compare("true") == 0 || str.compare("yes") == 0 || str.compare("1") == 0)
			_out = true;
		else if (str.compare("false") == 0 || str.compare("no") == 0 || str.compare("0") == 0)
			_out = false;
		else
			return false;

		return true;
	}

	// Attemp to find a tensor with the given name
	bool ObjectParser::Get(const std::string &_str, SpinAPI::Tensor &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
			return false;

		std::string directory = "";
		this->Get("FileReaderDirectory", directory);

		_out = SpinAPI::Tensor(str, directory);
		return true;
	}

	// Attemp to find a vector with the given name
	bool ObjectParser::Get(const std::string &_str, arma::vec &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
		{
			return false;
		}

		// Parse string and make sure that we have a 3D vector
		arma::vec tmpVec = arma::vec(str);

		if (tmpVec.n_elem != 3)
			return false;

		_out = tmpVec;
		return true;
	}

	// -----------------------------------------------------
	// ObjectParser public GetList method overloads
	// - Similar to Get, but allows multiple values to be specifed
	// -----------------------------------------------------
	// Attemp to find a keyword matching the given name
	bool ObjectParser::GetList(const std::string &_str, std::vector<std::string> &_out, const char &_delimiter) const
	{
		auto i = this->fields.find(_str);

		// If the keyword was found
		if (i != this->fields.end())
		{

			// Get all the values separated by the delimiter
			std::istringstream stream(i->second);
			for (std::string s; std::getline(stream, s, _delimiter);)
				_out.push_back(s);

			return true;
		}

		return false;
	}

	// Attemp to find a list of doubles with the given name
	bool ObjectParser::GetList(const std::string &_str, std::vector<double> &_out, const char &_delimiter) const
	{
		// Get a list of strings from the keyword
		std::vector<std::string> strs;
		if (!this->GetList(_str, strs, _delimiter))
			return false;

		std::vector<double> tmp;
		tmp.reserve(strs.size());

		try
		{
			// Attemp to parse the values
			for (auto i = strs.cbegin(); i != strs.cend(); i++)
				tmp.push_back(std::stod(*i));
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// Attemp to find a list of unsigned integers with the given keyword
	bool ObjectParser::GetList(const std::string &_str, std::vector<unsigned int> &_out, const char &_delimiter) const
	{
		// Get a list of strings from the keyword
		std::vector<std::string> strs;
		if (!this->GetList(_str, strs, _delimiter))
			return false;

		std::vector<unsigned int> tmp;
		tmp.reserve(strs.size());

		try
		{
			// Attemp to parse the values
			for (auto i = strs.cbegin(); i != strs.cend(); i++)
			{
				if (std::stoi(*i) < 0)
					return false;
				tmp.push_back(static_cast<unsigned int>(std::stoi(*i)));
			}
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// Attemp to find a list of integers with the given keyword
	bool ObjectParser::GetList(const std::string &_str, std::vector<int> &_out, const char &_delimiter) const
	{
		// Get a list of strings from the keyword
		std::vector<std::string> strs;
		if (!this->GetList(_str, strs, _delimiter))
			return false;

		std::vector<int> tmp;
		tmp.reserve(strs.size());

		try
		{
			// Attemp to parse the values
			for (auto i = strs.cbegin(); i != strs.cend(); i++)
				tmp.push_back(std::stoi(*i));
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// Attemp to find a list of vectors with the given keyword
	bool ObjectParser::GetList(const std::string &_str, std::vector<arma::vec> &_out, const char &_delimiter) const
	{
		// Get a list of strings from the keyword
		std::vector<std::string> strs;
		if (!this->GetList(_str, strs, _delimiter))
			return false;

		std::vector<arma::vec> tmp;
		tmp.reserve(strs.size());

		// Parse string and make sure that we have a 3D vector
		for (auto i = strs.cbegin(); i != strs.cend(); i++)
		{
			arma::vec tmpVec = arma::vec(*i);
			if (tmpVec.n_elem != 3)
				return false;
			tmp.push_back(tmpVec);
		}

		_out = tmp;
		return true;
	}

	bool ObjectParser::GetList(const std::string &_str, std::vector<bool> &_out, const char &_delimiter) const
	{
		// Get a list of strings from the keyword
		std::vector<std::string> strs;
		if (!this->GetList(_str, strs, _delimiter))
			return false;

		std::vector<bool> tmp;
		bool tmpBool;
		tmp.reserve(strs.size());

		try
		{
			// Attemp to parse the values
			for (auto i = strs.cbegin(); i != strs.cend(); i++)
			{
				if ((*i).compare("true") == 0 || (*i).compare("yes") == 0 || (*i).compare("1") == 0)
					tmpBool = true;
				else if ((*i).compare("false") == 0 || (*i).compare("no") == 0 || (*i).compare("0") == 0)
					tmpBool = false;
				else
					return false;

				tmp.push_back(tmpBool);
			}
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// -----------------------------------------------------
	// ObjectParser public Getmatrix method overloads
	// - Similar to Get, but allows multiple values to be specifed
	// -----------------------------------------------------
	// Attemp to find a keyword matching the given name	y
	bool ObjectParser::GetMatrix(const std::string &_str, arma::mat &_out) const
	{
		// Step 1: Output the initial message indicating the start of the reading process.
		std::cout << "Starting with reading: " << _str << std::endl;

		// Step 2: Initialize an empty vector to hold the split strings.
		std::vector<std::string> strs;

		// Step 3: Populate 'strs' by splitting '_str' using the delimiter ','.
		if (!this->GetList(_str, strs, ','))
			return false;

		// Step 4: Check if the resulting list is empty and return false if it is.
		if (strs.empty())
			return false;

		// Step 5: Initialize variable to keep track of the number of elements in each row.
		size_t numElements = 0;

		// Step 6: Create a lambda function to trim whitespace from both ends of a string.
		auto trim = [](std::string &str)
		{
			str.erase(0, str.find_first_not_of(' ')); // Prefix
			str.erase(str.find_last_not_of(' ') + 1); // Suffix
		};

		// Parse numeric values robustly, even if whitespace was stripped upstream.
		auto parse_numbers = [](const std::string &row, std::vector<double> &values) -> bool
		{
			values.clear();
			static const std::regex num_re(R"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)");
			for (std::sregex_iterator it(row.begin(), row.end(), num_re), end; it != end; ++it)
			{
				try
				{
					values.push_back(std::stod((*it).str()));
				}
				catch (const std::exception &)
				{
					return false;
				}
			}
			return !values.empty();
		};

		// Step 7: Validate the number of elements in each row (numElements).
		for (const std::string &str : strs)
		{
			std::string modified_str = str;
			// Remove square brackets
			modified_str.erase(std::remove(modified_str.begin(), modified_str.end(), '['), modified_str.end());
			modified_str.erase(std::remove(modified_str.begin(), modified_str.end(), ']'), modified_str.end());

			// Trim leading and trailing whitespaces
			trim(modified_str);

			std::vector<double> row_values;
			if (!parse_numbers(modified_str, row_values))
				return false;
			size_t count = row_values.size();

			// Check if the number of elements in each row is consistent
			if (numElements == 0)
				numElements = count;
			else if (count != numElements)
				return false;
		}

		// Step 8: Output the number of rows and columns.
		std::cout << "Number of rows: " << strs.size() << ", Number of columns: " << numElements << std::endl;

		// Step 9: Create a temporary matrix to store the values.
		arma::mat tmp(strs.size(), numElements);

		// Step 10: Populate the temporary matrix with values.
		for (size_t i = 0; i < strs.size(); ++i)
		{
			std::string str = strs[i];
			// Remove square brackets
			str.erase(std::remove(str.begin(), str.end(), '['), str.end());
			str.erase(std::remove(str.begin(), str.end(), ']'), str.end());

			// Trim leading and trailing whitespaces
			trim(str);

			std::vector<double> row_values;
			size_t j = 0;

			// Parse each value and populate the matrix
			if (!parse_numbers(str, row_values))
			{
				std::cout << "Invalid number format while parsing matrix " << _str << std::endl;
				return false;
			}
			for (double value : row_values)
			{
				if (j >= numElements)
				{
					std::cout << "Invalid matrix row width while parsing " << _str << std::endl;
					return false;
				}
				tmp(i, j) = value;
				j++;
			}
			if (j != numElements)
			{
				std::cout << "Invalid number format while parsing matrix " << _str << std::endl;
				return false;
			}
		}

		// Step 11: Output the populated matrix for debugging.
		// std::cout << tmp << std::endl;

		// Step 12: Handle small numbers close to zero.
		for (int i = 0; i < (int)strs.size(); i++)
		{
			for (int j = 0; j < (int)numElements; j++)
			{
				if (std::abs(tmp(i, j)) <= 1e-20)
				{
					tmp(i, j) = 0.0;
				}
			}
		}

		// Step 13: Assign the populated temporary matrix to the output matrix.
		_out = tmp;

		// Step 14: Output a message indicating the successful parsing.
		std::cout << "Read the list of multiexponentials for " << _str << std::endl;

		// Step 15: Return true to indicate successful parsing.
		return true;
	}

	// Old version: less robust
	// bool ObjectParser::GetMatrix(const std::string& _str, arma::mat& _out) const
	// {
	// 	std::cout << "Starting with reading: " << _str << std::endl;

	// 	// Step 1: Get a list of strings by splitting _str using the delimiter ','
	// 	std::vector<std::string> strs;
	// 	if (!this->GetList(_str, strs, ','))
	// 		return false;

	// 	// Step 2: Check if the list of strings is empty
	// 	if (strs.empty())
	// 		return false;

	// 	size_t numElements = 0;

	// 	// Step 3: Iterate through each string in the list to determine the number of elements
	// 	for (const std::string& str : strs)
	// 	{
	// 		size_t count = std::count(str.begin(), str.end(), ' ') + 1;

	// 		// If it's the first string, set the numElements
	// 		if (numElements == 0)
	// 			numElements = count;
	// 		// If the count is different from numElements, return false (rows should have the same number of elements)
	// 		else if (count != numElements)
	// 			return false;
	// 	}

	// 	std::cout << numElements << std::endl;
	// 	std::cout << strs.size() << std::endl;

	// 	// Step 4: Create a matrix with the size of the list of strings (rows) and numElements (columns)
	// 	arma::mat tmp(strs.size(), numElements);

	// 	// Step 5: Iterate through each string in the list
	// 	for (size_t i = 0; i < strs.size(); i++)
	// 	{
	// 		std::string str = strs[i];

	// 		// Step 6: Remove '[' and ']' characters from the string
	// 		str.erase(std::remove(str.begin(), str.end(), '['), str.end());
	// 		str.erase(std::remove(str.begin(), str.end(), ']'), str.end());

	// 		// Step 7: Output the modified string
	// 		//std::cout << "Substring: " << str << std::endl;

	// 		std::istringstream stream(str);
	// 		std::string token;
	// 		size_t j = 0;

	// 		// Step 8: Split the string using the delimiter ' ' and parse the values into the matrix
	// 		while (std::getline(stream, token, ' '))
	// 		{
	// 			try {
	// 				tmp(i, j) = std::stod(token);
	// 				//std::cout << "Parsed value: " << tmp(i, j) << std::endl;
	// 			} catch (const std::exception& e) {
	// 				//std::cout << "Invalid number format: " << token << std::endl;
	// 				return false;
	// 			}

	// 			j++;
	// 		}
	// 	}

	// 	for (int i = 0; i < (int) strs.size();i++){
	// 		for(int j = 0; j < (int) numElements;j++){
	// 			if(tmp(i,j) <= 1e-20)
	// 			{
	// 				tmp(i,j) *= 0.0;
	// 			}
	// 		}
	// 	}

	// 	std::cout << tmp << std::endl;
	// 	// Step 9: Assign the temporary matrix to the output matrix (_out)
	// 	_out = tmp;

	// 	// Step 10: Output a message indicating the successful parsing
	// 	std::cout << "Read the list of multiexponentials for " << _str << std::endl;

	// 	// Step 11: Return true to indicate successful parsing
	// 	return true;
	// }

	// -----------------------------------------------------
	// Specialized Get method to get a pulse sequence
	// -----------------------------------------------------
	// Attempt to find a pulse sequence
	bool ObjectParser::GetPulseSequence(const std::string &_str, std::vector<std::tuple<std::string, double>> &_out) const
	{
		// Step 1: Output the initial message indicating the start of the reading process.
		// std::cout << "Starting with reading: " << _str << std::endl;

		// Step 2: Initialize an empty vector to hold the split strings.
		std::vector<std::string> strs;

		// Step 3: Populate 'strs' by splitting '_str' using the delimiter ','.
		if (!this->GetList(_str, strs, ','))
			return false;

		// Step 4: Check if the resulting list is empty and return false if it is.
		if (strs.empty())
			return false;

		// Step 5: Initialize variable to keep track of the number of elements in each row.
		size_t numElements = 0;

		// Step 6: Create a lambda function to trim whitespace from both ends of a string.
		auto trim = [](std::string &str)
		{
			str.erase(0, str.find_first_not_of(' ')); // Prefix
			str.erase(str.find_last_not_of(' ') + 1); // Suffix
		};

		// Step 7: Validate the number of elements in each row (numElements).
		for (const std::string &str : strs)
		{
			std::string modified_str = str;
			// Remove square brackets
			modified_str.erase(std::remove(modified_str.begin(), modified_str.end(), '['), modified_str.end());
			modified_str.erase(std::remove(modified_str.begin(), modified_str.end(), ']'), modified_str.end());

			// Trim leading and trailing whitespaces
			trim(modified_str);

			std::istringstream stream(modified_str);
			size_t count = std::count(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>(), ' ') + 1;

			// Check if the number of elements in each row is consistent
			if (numElements == 0)
				numElements = count;
			else if (count < numElements)
				return false;
		}

		// Step 8: Create tuple/pair that store the pulse, and the free evolution time t_evo
		for (const std::string &str : strs)
		{
			std::string modified_str = str;

			modified_str.erase(std::remove(modified_str.begin(), modified_str.end(), '['), modified_str.end());
			modified_str.erase(std::remove(modified_str.begin(), modified_str.end(), ']'), modified_str.end());
			trim(modified_str);

			std::istringstream stream(modified_str);
			std::vector<std::string> tokens;
			std::string token;
			while (stream >> token)
			{
				tokens.push_back(token);
			}

			if (tokens.size() < 2)
			{
				std::cerr << "Invalid number of elements in sequence. Expected 2, got " << tokens.size() << std::endl;
				return false;
			}

			double t_evo;
			try
			{
				t_evo = std::stod(tokens[1]);
			}
			catch (const std::exception &e)
			{
				std::cerr << "Error converting string to double: " << e.what() << std::endl;
				return false;
			}

			_out.emplace_back(tokens[0], t_evo);
		}

		return true;
	}
	// -----------------------------------------------------
	// Specialized Get method to get a spin quantum number
	// -----------------------------------------------------
	// Attempt to find a spin quantum number with the given name
	bool ObjectParser::GetSpin(const std::string &_str, unsigned int &_out) const
	{
		std::string str("");
		if (!this->Get(_str, str))
			return false;

		unsigned int tmp;

		try
		{
			int tmp2 = -1;

			// Check whether the spin is written in the form "1/2", "2/2", "3/2", etc.
			if (str.size() > 2 && (*(str.cend() - 1) == '2' && *(str.cend() - 2) == '/'))
			{
				tmp2 = std::stoi(str.substr(0, str.size() - 2).c_str());
			}
			else
			{
				// Use a factor of two if the spin quantum number is not specified with "/2"
				tmp2 = 2 * std::stoi(str.c_str());
			}

			if (std::stoi(str) < 0)
				return false;
			tmp = static_cast<unsigned int>(tmp2);
		}
		catch (const std::exception &)
		{
			return false;
		}

		_out = tmp;
		return true;
	}

	// -----------------------------------------------------
	// ObjectParser public GetFunction method
	// -----------------------------------------------------
	std::vector<std::pair<std::string, std::string>> ObjectParser::GetFunction(std::string _keyword) const
	{
		std::vector<std::pair<std::string, std::string>> result;

		// Loop through all the fields
		for (auto i = this->fields.cbegin(); i != this->fields.cend(); i++)
		{
			// Search for parantheses
			auto pstart = i->first.find_first_of("(");
			if (pstart != std::string::npos)
			{
				// If the keyword was found, put it into the results collection
				if (i->first.substr(0, pstart).compare(_keyword) == 0)
				{
					auto pend = i->first.find_first_of(")");
					result.push_back(std::pair<std::string, std::string>(i->first.substr(pstart + 1, pend - pstart - 1), i->second));
				}
			}
		}

		return result;
	}
	// -----------------------------------------------------
	// ObjectParser other public methods
	// -----------------------------------------------------
	std::string ObjectParser::Name() const
	{
		return this->name;
	}
	// -----------------------------------------------------
}
