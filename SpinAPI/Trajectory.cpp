/////////////////////////////////////////////////////////////////////////
// Trajectory class (SpinAPI Module)
// ------------------
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <cmath>
#include <set>
#include <stdexcept>
#include "Trajectory.h"

namespace SpinAPI
{
	// -----------------------------------------------------
	// Spin Constructors and Destructor
	// -----------------------------------------------------
	Trajectory::Trajectory() : filename(), headers(), data()
	{
	}

	Trajectory::Trajectory(const Trajectory &_trajectory) : filename(_trajectory.filename), headers(_trajectory.headers), data(_trajectory.data)
	{
	}

	Trajectory::~Trajectory()
	{
	}
	// -----------------------------------------------------
	// Operators
	// -----------------------------------------------------
	const Trajectory &Trajectory::operator=(const Trajectory &_trajectory)
	{
		this->filename = _trajectory.filename;
		this->headers = _trajectory.headers;
		this->data = _trajectory.data;

		return (*this);
	}
	// -----------------------------------------------------
	// Private methods
	// -----------------------------------------------------
	// Checks whether a character is a letter, number (see method below) or an underscore
	bool Trajectory::isAlphanumericCharacter(const char &_c) const
	{
		if (isNumericCharacter(_c))
			return true;
		else if (_c >= 0x41 && _c <= 0x5A) // '_c' is a capital letter (i.e. A-Z)
			return true;
		else if (_c >= 0x61 && _c <= 0x7A) // '_c' is a lower case letter (i.e. a-z)
			return true;
		else if (_c == '_') // '_c' is an underscore
			return true;

		return false;
	}

	// Checks whether a character is a number (including '.+-' and 'e' or 'E' for scientific notation)
	bool Trajectory::isNumericCharacter(const char &_c) const
	{
		const std::string allowed(".-+");

		if (_c >= 0x30 && _c <= 0x39) // '_c' is a number (i.e. 0-9)
			return true;
		else if (_c == 0x45 || _c == 0x65) // '_c' is 'e' or 'E'
			return true;
		else if (allowed.find(_c) != std::string::npos) // '_c' is one of the allowed special characters
			return true;

		return false;
	}

	// Checks whether a character is a space or a tab
	bool Trajectory::isDelimiter(const char &_c) const
	{
		const std::string delimiters(" \t");

		if (delimiters.find(_c) != std::string::npos)
			return true;

		return false;
	}

	// Method to add data (a number) to the trajectory
	void Trajectory::AddNumber(unsigned int _row, std::string _value)
	{
		// If we need to add more than a single new row, don't do it (intermediate columns would become empty)
		if (_row > this->data.size())
			return;

		// Create a new row if we need it
		if (_row == this->data.size())
			this->data.push_back(std::vector<double>());

		// Do not load any data if there is no header for the column
		if (this->data[_row].size() >= this->headers.size())
			return;

		// Convert from string to double
		std::istringstream stream(_value);
		double x;

		if (!(stream >> x))
			x = 0.0;

		// Store the value
		this->data[_row].push_back(x);
	}

	// Sets the header for a column in the trajectory
	void Trajectory::SetHeader(unsigned int _column, std::string _name)
	{
		this->headers.insert(std::pair<std::string, unsigned int>(_name, _column));
	}
	// -----------------------------------------------------
	// Public methods
	// -----------------------------------------------------
	// Clears any previously loaded data, and loads a trajectory with the given filename
	bool Trajectory::Load(const std::string &_filename)
	{
		// Clean-up in case we already had a trajectory loaded
		this->Clear();

		// Open trajectory file and check that the stream is ok
		std::ifstream filehandle(_filename);
		if (filehandle.is_open())
		{
			char c;
			bool hasHeader = false; // Set to true after first newline character (first line contains the headers)
			std::string value = "";
			unsigned int linenumber = 0;
			unsigned int column = 0;

			while (filehandle.get(c))
			{
				if (c == '\n')
				{
					// Was this the header line?
					if (hasHeader)
					{
						this->AddNumber(linenumber - 1, value);
					}
					else
					{
						hasHeader = true;
						this->SetHeader(column, value);
					}

					// Reset value, ready to read the next
					value = "";
					column = 0;

					// We found a newline character!
					linenumber++;
				}
				else if (this->isDelimiter(c))
				{
					if (!value.empty())
					{
						// Was this the header line?
						if (hasHeader)
						{
							this->AddNumber(linenumber - 1, value);
						}
						else
						{
							this->SetHeader(column, value);
						}

						// Reset value, ready to read the next
						value = "";
						column++;
					}
				}
				else if (!hasHeader && this->isAlphanumericCharacter(c))
				{
					value += c;
				}
				else if (hasHeader && this->isNumericCharacter(c))
				{
					value += c;
				}
			}

			return true;
		}

		std::cout << "ERROR: Failed to open trajectory file \"" << _filename << "\"! Trajectory was not loaded!" << std::endl;
		return false;
	}

	bool Trajectory::LoadStrict(const std::string &_filename, std::string &_error)
	{
		std::ifstream input(_filename);
		if (!input)
		{
			_error = "cannot open experimental trajectory \"" + _filename + "\"";
			this->Clear();
			return false;
		}
		return LoadStrict(input, _error);
	}

	bool Trajectory::LoadStrict(std::istream &_input, std::string &_error)
	{
		this->Clear();
		_error.clear();
		std::string line;
		unsigned int lineNumber = 0;
		bool haveHeader = false;
		unsigned int columns = 0;
		while (std::getline(_input, line))
		{
			++lineNumber;
			std::istringstream row(line);
			std::vector<std::string> fields;
			std::string field;
			while (row >> field) fields.push_back(field);
			if (fields.empty()) continue;
			if (!haveHeader)
			{
				std::set<std::string> seen;
				for (const auto &header : fields)
				{
					if (header.empty() || !(std::isalpha(static_cast<unsigned char>(header[0])) || header[0] == '_'))
					{ _error = "invalid experimental trajectory header on line " + std::to_string(lineNumber); this->Clear(); return false; }
					for (unsigned char c : header)
						if (!(std::isalnum(c) || c == '_'))
						{ _error = "invalid experimental trajectory header on line " + std::to_string(lineNumber); this->Clear(); return false; }
					if (!seen.insert(header).second)
					{ _error = "duplicate experimental trajectory header \"" + header + "\""; this->Clear(); return false; }
					this->SetHeader(columns++, header);
				}
				haveHeader = true;
				continue;
			}
			if (fields.size() != columns)
			{ _error = "experimental trajectory row " + std::to_string(lineNumber) + " has " + std::to_string(fields.size()) + " columns; expected " + std::to_string(columns); this->Clear(); return false; }
			std::vector<double> values;
			values.reserve(columns);
			for (const auto &token : fields)
			{
				try
				{
					size_t used = 0;
					const double value = std::stod(token, &used);
					if (used != token.size() || !std::isfinite(value))
						throw std::invalid_argument("invalid numeric token");
					values.push_back(value);
				}
				catch (const std::exception &)
				{ _error = "invalid or non-finite number \"" + token + "\" on experimental trajectory line " + std::to_string(lineNumber); this->Clear(); return false; }
			}
			this->data.push_back(std::move(values));
		}
		if (!_input.eof() && _input.fail())
		{ _error = "failed while reading experimental trajectory"; this->Clear(); return false; }
		if (!haveHeader || this->data.size() < 2)
		{ _error = "experimental trajectory requires a header and at least two data rows"; this->Clear(); return false; }
		return true;
	}

	// Removes all loaded trajectory data
	bool Trajectory::Clear()
	{
		this->headers.clear();
		this->data.clear();
		return true;
	}
	// -----------------------------------------------------
	// Public Get methods
	// -----------------------------------------------------
	// Returns the value at the specified row and column number
	double Trajectory::Get(const unsigned int _row, const unsigned int _column) const
	{
		if (_row >= this->data.size())
			return 0.0;

		if (_column >= this->data[_row].size())
			return 0.0;

		return this->data[_row][_column];
	}

	// Overload of Get, with a string to specify column header name
	double Trajectory::Get(const unsigned int _row, const std::string &_column) const
	{
		auto i = this->headers.find(_column);
		if (i != this->headers.end())
			return this->Get(_row, i->second);

		return 0.0;
	}

	// Overload of Get, with signed ints
	double Trajectory::Get(const int _row, const int _column) const
	{
		if (_row >= 0 && _column >= 0)
			return this->Get((unsigned)_row, (unsigned)_column);

		return 0.0;
	}

	// Overload of Get, with signed int and string
	double Trajectory::Get(const int _row, const std::string &_column) const
	{
		auto i = this->headers.find(_column);
		if (_row >= 0 && i != this->headers.end())
			return this->Get((unsigned)_row, i->second);

		return 0.0;
	}
	// -----------------------------------------------------
	// Public Header management methods
	// -----------------------------------------------------
	// Checks whether a column header exists in the trajectory
	bool Trajectory::HasColumn(std::string _header) const
	{
		auto i = this->headers.find(_header);
		return (i != this->headers.end());
	}

	// Checks whether a column header exists in the trajectory, and get the corresponding column index
	bool Trajectory::HasColumn(std::string _header, unsigned int &_headerColumn) const
	{
		auto i = this->headers.find(_header);
		if (i != this->headers.end())
		{
			_headerColumn = i->second;
			return true;
		}

		return false;
	}
	// -----------------------------------------------------
	// Public methods to get row number
	// -----------------------------------------------------
	// For a given column, find the first row with a value greater than or equal to the specified _value
	// Sets the _row to the found row number, and _firstGreaterValue to the value of the column in that row
	bool Trajectory::FirstRowEqGreaterThan(double _value, unsigned int _column, unsigned int &_row, double &_firstGreaterValue) const
	{
		// Make sure that we are checking a valid column
		if (_column >= this->headers.size())
			return false;

		// Iterate throught the rows
		unsigned int row = 0;
		for (auto i = this->data.cbegin(); i != this->data.cend(); i++)
		{
			// Check if the column has a value (note that _column is zero-based, hence don't use ">=")
			if ((*i).size() > _column)
			{
				// Check the Equal-to-or-Greater-than condition
				if ((*i)[_column] >= _value)
				{
					_row = row;
					_firstGreaterValue = (*i)[_column];
					return true;
				}
			}

			row++;
		}

		return false;
	}

	// Similar to the above, but with a string to specify the column
	bool Trajectory::FirstRowEqGreaterThan(double _value, std::string _column, unsigned int &_row, double &_firstGreaterValue) const
	{
		auto i = this->headers.find(_column);
		if (i != this->headers.end())
			return this->FirstRowEqGreaterThan(_value, i->second, _row, _firstGreaterValue);

		return false;
	}

	// For a given column, find the first row with a value less than or equal to the specified _value
	// Sets the _row to the found row number, and _firstLessValue to the value of the column in that row
	bool Trajectory::FirstRowEqLessThan(double _value, unsigned int _column, unsigned int &_row, double &_firstLessValue) const
	{
		// Make sure that we are checking a valid column
		if (_column >= this->headers.size())
			return false;

		// Iterate throught the rows
		unsigned int row = 0;
		for (auto i = this->data.cbegin(); i != this->data.cend(); i++)
		{
			// Check if the column has a value (note that _column is zero-based, hence don't use ">=")
			if ((*i).size() > _column)
			{
				// Check the Equal-to-or-Less-than condition
				if ((*i)[_column] <= _value)
				{
					_row = row;
					_firstLessValue = (*i)[_column];
					return true;
				}
			}

			row++;
		}

		return false;
	}

	// Similar to the above, but with a string to specify the column
	bool Trajectory::FirstRowEqLessThan(double _value, std::string _column, unsigned int &_row, double &_firstLessValue) const
	{
		auto i = this->headers.find(_column);
		if (i != this->headers.end())
			return this->FirstRowEqLessThan(_value, i->second, _row, _firstLessValue);

		return false;
	}
	// -----------------------------------------------------
	// Other public methods
	// -----------------------------------------------------
	// Returns the number of rows in the trajectory
	unsigned int Trajectory::Length() const
	{
		return this->data.size();
	}
	// -----------------------------------------------------
}
