/////////////////////////////////////////////////////////////////////////
// FileReader implementations, used to read objects from MSD files.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include "MSDParser.h"
#include "FileReader.h"

namespace MSDParser
{
	// -----------------------------------------------------
	// Static member definitions
	// -----------------------------------------------------
	std::vector<std::string> FileReader::filelist;
	std::map<std::string, std::string> FileReader::define_directives;

	// Returns a copy of the file list
	std::vector<std::string> FileReader::GetFileList()
	{
		return FileReader::filelist;
	}

	// Returns a copy of all the definitions
	std::map<std::string, std::string> FileReader::GetDefinitions()
	{
		return FileReader::define_directives;
	}

	// Extracts the path from the filename
	std::string FileReader::PathFromFileName(const std::string &_filename)
	{
		// Search for '/' from the end towards the beginning
		for (auto i = _filename.size(); i > 0; i--)
			if (_filename[i] == '/')
				return _filename.substr(0, i + 1); // The '+1' adds the '/'

		// If no '/' was found, the file must be in the current directory
		return "./";
	}

	// Checks whether a string is absolute (returns false for a relative path)
	bool FileReader::PathIsAbsolute(const std::string &_filename)
	{
		// Absolute paths start with '/'
		if (_filename.size() > 0 && _filename[0] == '/')
			return true;

		return false;
	}

	// Add a define directive
	bool FileReader::AddDefinition(const std::string &_name, const std::string &_value)
	{
		// Make sure that a name was specified
		if (_name.empty())
		{
			// If no name is specified, notify that the directive was invalid
			std::cout << "WARNING: Could not make definition because no name was given!" << std::endl;
			std::cout << " ---->   Definition was requested on the commandline." << std::endl;
			return false;
		}
		else if (define_directives.find(_name) != define_directives.cend())
		{
			// If the name was already defined
			std::cout << "ERROR: FileReader failed to make definition because \"" << _name << "\" was already defined!" << std::endl;
			std::cout << " ---->   Definition was requested on the commandline." << std::endl;
			return false;
		}

		// Add the definition
		define_directives[_name] = _value;
		return true;
	}
	// -----------------------------------------------------
	// FileReader Constructors and Destructor
	// -----------------------------------------------------
	// Initialize the FileReader with an MSD file
	FileReader::FileReader(const std::string &_filename) : objectGroupType(ObjectGroup::None), objectGroupName(""), subfile(nullptr), filehandle(_filename), filename(_filename), linenumber(1), inObjectGroup(false), inExcludedDirective(false)
	{
		// Save the filename/path to avoid circular dependencies
		FileReader::filelist.push_back(_filename);
	}

	// Destructor
	FileReader::~FileReader()
	{
		// Note: file is automatically closed by ifstream when the ifstream is destroyed
	}

	// -----------------------------------------------------
	// FileReader private methods to validate characters
	// -----------------------------------------------------
	// In specifying keywords and names, no special characters are allowed.
	// NOTE: Underscore is allowed as well for its use in names.
	bool FileReader::isAlphanumericCharacter(const char &_c) const
	{
		if (_c >= 0x30 && _c <= 0x39) // '_c' is a number (i.e. 0-9)
			return true;
		else if (_c >= 0x41 && _c <= 0x5A) // '_c' is a capital letter (i.e. A-Z)
			return true;
		else if (_c >= 0x61 && _c <= 0x7A) // '_c' is a lower case letter (i.e. a-z)
			return true;
		else if (_c == '_') // '_c' is an underscore
			return true;

		return false;
	}

	// In contents, more than alphanumerics are allowed
	bool FileReader::isAllowedCharacter(const char &_c) const
	{
		const std::string allowed("().-+,;=|>/*^");

		if (this->isAlphanumericCharacter(_c))
			return true;
		else if (allowed.find(_c) != std::string::npos) // '_c' is one of the allowed special characters
			return true;

		return false;
	}

	// Additional characters may be allowed in strings
	bool FileReader::isAllowedCharacterInString(const char &_c) const
	{
		const std::string allowed(" "); // Additional allowed characters

		if (this->isAllowedCharacter(_c))
			return true;
		else if (allowed.find(_c) != std::string::npos)
			return true;

		return false;
	}

	// Returns the char in lowercase if it is a letter
	char FileReader::getLowerCase(const char &_c) const
	{
		if (_c >= 0x41 && _c <= 0x5A) // '_c' is a capital letter (i.e. A-Z)
			return (_c + 0x20);		  // Add 0x20 to convert to lowercase

		return _c;
	}

	// -----------------------------------------------------
	// FileReader private method to handle '#'-directives
	// -----------------------------------------------------
	void FileReader::handleDirective(const std::string &_keyword, const std::string &_name, const std::string &_contents)
	{
		// End an "else" if encountered
		if (_keyword.compare("else") == 0)
		{
			inExcludedDirective = (inExcludedDirective == false);
			return;
		}

		// End an "endif" if encountered
		if (this->inExcludedDirective)
		{
			if (_keyword.compare("endif") == 0)
				inExcludedDirective = false;

			return;
		}

		// If we encounter and "endif" when !inExcludedDirective, just return
		if (_keyword.compare("endif") == 0)
			return;

		// Check for the other directives
		if (_keyword.compare("include") == 0)
		{
			// Make sure that a file was specified
			if (_name.empty())
			{
				// If no file is specified, terminate the program
				std::cout << "ERROR: FileReader failed to include file, because no file name was given!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				std::exit(1);
			}

			// If a file was specified, make sure we use a file path relative to the current file, unless an absolute path was given
			std::string include_file = _name;
			if (!FileReader::PathIsAbsolute(_name))
			{
				include_file = FileReader::PathFromFileName(this->filename) + _name;
			}

			if (std::find(filelist.cbegin(), filelist.cend(), include_file) != filelist.cend())
			{
				// If the file was already included, terminate the program
				// This prevents circular includes
				std::cout << "ERROR: FileReader failed to include file, because the file was already included! Files can only be included once!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				std::exit(1);
			}
			else if (this->inObjectGroup)
			{
				// Cannot include files from inside an object group
				std::cout << "ERROR: FileReader failed to include file, because the directive was called inside an object group! Please move include directive outside the object group." << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				std::exit(1);
			}
			else if (!_contents.empty())
			{
				// The include directive only needs a filename, so if extra information is supplied it might be a mistake
				std::cout << "WARNING: FileReader found redundant data after specified include file. This information will be ignored!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
			}

			// Create FileReader for subfile
			this->subfile = std::make_shared<FileReader>(include_file);
			if (!this->subfile->filehandle.good())
			{
				// If the file could not be loaded, terminate the program
				std::cout << "ERROR: FileReader failed to open include file \"" << include_file << "\" for reading!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				std::exit(1);
			}
		}
		else if (_keyword.compare("define") == 0)
		{
			// Make sure that a name was specified
			if (_name.empty())
			{
				// If no name is specified, notify that the directive was invalid
				std::cout << "WARNING: Could not make definition because no name was given!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}
			else if (define_directives.find(_name) != define_directives.cend())
			{
				// If the name was already defined
				std::cout << "ERROR: FileReader failed to make definition because \"" << _name << "\" was already defined!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Add the definition
			define_directives[_name] = _contents;
		}
		else if (_keyword.compare("undef") == 0)
		{
			// Make sure that a name was specified
			if (_name.empty())
			{
				// If no name is specified, notify that the directive was invalid
				std::cout << "WARNING: Could not remove definition because no name was given!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Get an iterator to the definition
			auto tmp = define_directives.find(_name);

			if (tmp == define_directives.cend())
			{
				// If the name was not defined
				std::cout << "WARNING: FileReader failed to remove definition because \"" << _name << "\" was not defined!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Remove the definition
			define_directives.erase(tmp);
		}
		else if (_keyword.compare("ifdef") == 0)
		{
			// Make sure that a name was specified
			if (_name.empty())
			{
				// If no name is specified, notify that the directive was invalid
				std::cout << "WARNING: Could not evaluate conditional directive as no definition was given!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Nested conditions are not supported
			if (inExcludedDirective)
			{
				// If no name is specified, notify that the directive was invalid
				std::cout << "WARNING: Nested conditional directives are not supported!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Check whether the given name is defined
			if (define_directives.find(_name) == define_directives.cend())
			{
				inExcludedDirective = true;
			}
		}
		else if (_keyword.compare("ifndef") == 0)
		{
			// Make sure that a name was specified
			if (_name.empty())
			{
				// If no name is specified, notify that the directive was invalid
				std::cout << "WARNING: Could not evaluate conditional directive as no definition was given!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Nested conditions are not supported
			if (inExcludedDirective)
			{
				// If no name is specified, notify that the directive was invalid
				std::cout << "WARNING: Nested conditional directives are not supported!" << std::endl;
				std::cout << " ---->   See line " << (linenumber - 1) << " in file \"" << this->filename << "\"." << std::endl;
				return;
			}

			// Check whether the given name is defined
			if (define_directives.find(_name) != define_directives.cend())
			{
				inExcludedDirective = true;
			}
		}
		else
		{
			std::cout << "Found unknown directive of type \"" << _keyword << "\" with name \"" << _name << "\"!" << std::endl;
		}
	}

	// -----------------------------------------------------
	// FileReader private method to set the object group
	// -----------------------------------------------------
	void FileReader::setObjectGroup(const std::string &_keyword, const std::string &_name)
	{
		if (_keyword.compare("spinsystem") == 0)
		{
			if (_name.empty())
			{
				// SpinSystem objects must have a name
				std::cout << "ERROR: FileReader failed to create SpinSystem, because no name was provided!" << std::endl;
				std::cout << " ---->   See line " << linenumber << " in file \"" << this->filename << "\"." << std::endl;
				std::exit(1);
			}

			// Set the object group
			this->objectGroupType = ObjectGroup::SpinSystem;
			this->objectGroupName = _name;
		}
		else if (_keyword.compare("settings") == 0)
		{
			// Set the object group
			this->objectGroupType = ObjectGroup::Settings;
			this->objectGroupName = "";
		}
		else if (_keyword.compare("run") == 0)
		{
			// Set the object group
			this->objectGroupType = ObjectGroup::Run;
			this->objectGroupName = "";
		}
		else
		{
			// Undefined object group type
			std::cout << "ERROR: FileReader did not recognize object group of type \"" << _keyword << "\"!" << std::endl;
			std::cout << " ---->   See line " << linenumber << " in file \"" << this->filename << "\"." << std::endl;
			std::exit(1);
		}
	}

	// -----------------------------------------------------
	// FileReader private method to create an object
	// -----------------------------------------------------
	MSDFileObject FileReader::createObject(const std::string &_keyword, const std::string &_name, std::string _contents)
	{
		if (_name.empty())
		{
			// If no file is specified, terminate the program
			std::cout << "ERROR: FileReader failed to create object of type \"" << _keyword << "\", because no object name was given!" << std::endl;
			std::cout << " ---->   See line " << linenumber << " in file \"" << this->filename << "\"." << std::endl;
			std::exit(1);
		}

		// Replace defined names in _contents
		for (auto i = define_directives.cbegin(); i != define_directives.cend(); i++)
		{
			// Only do the replacement if a value is defined
			if (i->second.empty())
				continue;

			std::string::size_type n = _contents.find(i->first, 0); // Find first occurence
			while (n != std::string::npos)
			{
				// Replace a found occurence
				_contents.replace(n, i->first.size(), i->second);

				// Find next occurence
				n = _contents.find(i->first, n + i->second.size());
			}
		}

		// Add path to the directory of the input file that read this object, such that objects can access it
		// Also make sure that the contents block ends with a ';'
		if (_contents.size() > 0 && _contents[_contents.size() - 1] != ';')
		{
			_contents += ";";
		}
		_contents += "FileReaderDirectory=" + FileReader::PathFromFileName(this->filename) + ";";

		// Get the object type and return the object
		if (_keyword.compare("spin") == 0)
		{
			return MSDFileObject(ObjectType::Spin, _name, _contents);
		}
		else if (_keyword.compare("interaction") == 0)
		{
			return MSDFileObject(ObjectType::Interaction, _name, _contents);
		}
		else if (_keyword.compare("transition") == 0)
		{
			return MSDFileObject(ObjectType::Transition, _name, _contents);
		}
		else if (_keyword.compare("operator") == 0)
		{
			return MSDFileObject(ObjectType::Operator, _name, _contents);
		}
		else if (_keyword.compare("state") == 0)
		{
			return MSDFileObject(ObjectType::State, _name, _contents);
		}
		else if (_keyword.compare("pulse") == 0)
		{
			return MSDFileObject(ObjectType::Pulse, _name, _contents);
		}
		else if (_keyword.compare("pulsesequence") == 0)
		{
			return MSDFileObject(ObjectType::PulseSequence, _name, _contents);
		}
		else if (_keyword.compare("task") == 0)
		{
			return MSDFileObject(ObjectType::Task, _name, _contents);
		}
		else if (_keyword.compare("action") == 0)
		{
			return MSDFileObject(ObjectType::Action, _name, _contents);
		}
		else if (_keyword.compare("subsystem") == 0)
		{
			return MSDFileObject(ObjectType::SubSystem, _name, _contents);
		}
		else if (_keyword.compare("output") == 0)
		{
			return MSDFileObject(ObjectType::Output, _name, _contents);
		}
		else if (_keyword.compare("settings") == 0)
		{
			return MSDFileObject(ObjectType::Settings, _name, _contents);
		}
		else if (_keyword.compare("properties") == 0)
		{
			return MSDFileObject(ObjectType::Properties, _name, _contents);
		}

		return MSDFileObject(ObjectType::Undefined, _name, _contents);
	}

	// -----------------------------------------------------
	// FileReader public ReadObject method
	// -----------------------------------------------------
	MSDFileObject FileReader::ReadObject()
	{
		// If we have included a "subfile", see if there are any more objects in that file
		if (this->subfile != nullptr)
		{
			// Get next object
			MSDFileObject obj(this->subfile->ReadObject());

			// Return the object if it is not the last in the subfile
			if (obj.Type() != ObjectType::EOFObject)
			{
				this->objectGroupType = this->subfile->ObjectGroupType();
				this->objectGroupName = this->subfile->ObjectGroupName();
				return obj;
			}
			else
			{
				// If there are no more objects in the subfile, stop reading from the subfile and proceed with the current file below
				this->subfile = nullptr;
			}
		}

		// Helper definitions for parsing the file
		MSDFileObject outObj;
		std::string keyword("");
		std::string name("");
		std::string contents("");
		bool hasObject = false;
		bool inSLComment = false; // Single-line comment ("// ...")
		bool inMLComment = false; // Multi-line comment ("/* ... */")
		bool inString = false;
		bool inDirective = false;
		bool inObject = false;
		char c;

		const int in_keyword = 0;
		const int in_name = 1;
		const int in_contents = 2;
		int state = in_keyword;

		while (!hasObject)
		{
			// Get the next character
			if (!this->filehandle.get(c))
			{
				// If we hit the EOFObject with an unfinished object, notify the user but ignore the object
				if (!keyword.empty() || !name.empty())
				{
					std::cout << "Failed to read object \"" << name << "\" of type \"" << keyword << "\". Unexpected end of file! Object ignored." << std::endl;
				}

				// If we hit the end of the file, return EOFObject
				return outObj;
			}

			// Count line numbers (the '\n' character itself is handled below)
			if (c == '\n')
			{
				linenumber++;
			}

			// If we are in an excluded part of a conditional directive (e.g. an "ifdef SOMENAME" where SOMENAME was *not* defined), skip everything until we encounter an "endif"
			if (this->inExcludedDirective && !inDirective && (c != '#' || state != in_keyword || !keyword.empty() || inObject))
			{
				continue;
			}

			// Single-line comments are only stopped by newline characters
			if (inSLComment)
			{
				if (c == '\n')
					inSLComment = false;

				if (!inDirective || c != '\n')
					continue;
			}

			// Multi-line comments are only stopped by "*/"
			if (inMLComment)
			{
				if (c == '*' && this->filehandle.peek() == '/')
				{
					inMLComment = false;
					this->filehandle.get(c); // Skip the next character since it is part of the "stop comment" symbol
				}

				continue;
			}

			// When we enter/exit a string
			if (c == '\"')
			{
				// Toggle inString variable
				inString = (inString == false);
				continue;
			}

			if (inString)
			{
				if (state == in_contents)
				{
					// Strings are only valid in object or directive contents...
					if (this->isAllowedCharacterInString(c))
					{
						contents += c;
					}
				}
				else if (state == in_name && inDirective)
				{
					// ... or names of directives.
					// Note that this is necessary for the '#include' directive.
					if (this->isAllowedCharacterInString(c))
					{
						name += c;
					}
				}
			}
			else if (c == '/' && this->filehandle.peek() == '*') // Found a multi-line comment
			{
				inMLComment = true;
			}
			else if (c == '/' && this->filehandle.peek() == '/') // Found a single-line comment
			{
				inSLComment = true;
			}
			else if (c == '#' && state == in_keyword && keyword.empty() && !inDirective && !inObject)
			{
				// We found a directive (special command to the FileReader object, similar to pre-compiler directives)
				inDirective = true;
			}
			else if (c == ' ')
			{
				// Spaces may separate keywords from names, and name from contents in directives.
				// Otherwise, spaces are ignored (except for strings which is handled above).
				if (state == in_keyword && !keyword.empty())
				{
					state = in_name;
				}
				else if (state == in_name && inDirective)
				{
					state = in_contents;
				}
			}
			else if (c == '\n')
			{
				// New-line characters only have significance for directives, which they end
				if (inDirective)
				{
					// Do what the directive specifies
					this->handleDirective(keyword, name, contents);

					// Maybe the directive was an "include" that just opened a new file
					if (this->subfile != nullptr)
					{
						// Get an object from the newly opened file
						outObj = this->subfile->ReadObject();

						// Check whether we found an object
						if (outObj.Type() != ObjectType::EOFObject)
						{
							this->objectGroupType = this->subfile->ObjectGroupType();
							this->objectGroupName = this->subfile->ObjectGroupName();

							// Break the loop and return the object
							hasObject = true;
						}
						else
						{
							// If there are no more objects in the subfile, stop reading from the subfile and proceed with the current file instead
							this->subfile = nullptr;
						}
					}

					// Clear the state
					keyword.clear();
					name.clear();
					contents.clear();
					inDirective = false;
					state = in_keyword;
				}
			}
			else if (c == '{')
			{
				// We are entering an object or object-group
				if ((state == in_keyword || state == in_name) && !this->inObjectGroup)
				{
					// We have entered an object group, so set the current FileReader ObjectGroup
					this->setObjectGroup(keyword, name);

					// Clear state to read objects within the ObjectGroup
					keyword.clear();
					name.clear();
					contents.clear(); // TODO: Check whether this is redundant...
					state = in_keyword;
					this->inObjectGroup = true;
				}
				else if ((state == in_keyword || state == in_name) && this->inObjectGroup)
				{
					// We have entered an object, but do we have a name?
					if (state == in_keyword)
					{
						// We do not have a name!
						std::cout << "ERROR: FileReader did not find a name for object of type \"" << keyword << "\"!" << std::endl;
						std::cout << " ---->   See line " << linenumber << " in file \"" << this->filename << "\"." << std::endl;
						std::exit(1);
					}

					// So, we have a name... Change state to get the contents of the object!
					state = in_contents;
					inObject = true;
				}
			}
			else if (c == '}')
			{
				// If we were inside an object, this is where it ends
				if (inObject)
				{
					// Get the object
					outObj = this->createObject(keyword, name, contents);

					// Break the loop and return the object
					hasObject = true;
				}
				else if (inObjectGroup)
				{
					// We are ending an object group instead
					this->inObjectGroup = false;

					// Make sure that the state is reset
					keyword.clear();
					name.clear();
					contents.clear();
					inDirective = false;
					state = in_keyword;
				}
			}
			else if (state == in_keyword)
			{
				// Only add alphanumerics to keywords
				if (this->isAlphanumericCharacter(c))
				{
					keyword += this->getLowerCase(c);
				}
			}
			else if (state == in_name)
			{
				// Only add alphanumerics to names (or strings for directives)
				if (this->isAlphanumericCharacter(c))
				{
					name += this->getLowerCase(c);
				}
			}
			else if (state == in_contents)
			{
				// Add to contents if it is an allowed character
				if (this->isAllowedCharacter(c))
				{
					contents += this->getLowerCase(c);
				}
			}
		}

		return outObj;
	}

	// -----------------------------------------------------
	// Other FileReader public methods
	// -----------------------------------------------------
	// Checks whether the file is open for reading
	bool FileReader::IsOpen()
	{
		return this->filehandle.good();
	}
	// -----------------------------------------------------
}
