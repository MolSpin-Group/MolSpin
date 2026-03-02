/////////////////////////////////////////////////////////////////////////
// OutputHandler implementation (RunSection module)
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#include <iomanip>
#include <iostream>
#include "OutputHandler.h"

namespace RunSection
{
	// -----------------------------------------------------
	// Settings Constructors and Destructor
	// -----------------------------------------------------
	OutputHandler::OutputHandler() : log(nullptr), data(nullptr), logstream(nullptr), datastream(nullptr), logPrecision(6), dataPrecision(6), ignored_messages_stream(std::make_shared<std::ostringstream>())
	{
	}

	OutputHandler::~OutputHandler()
	{
	}
	// -----------------------------------------------------
	// Public methods
	// -----------------------------------------------------
	bool OutputHandler::SetLogFile(const std::string &_filename, bool _append)
	{
		// If we already have a logfile, make sure to close it
		if (this->log != nullptr && this->log->is_open())
			this->log->close();

		// Check whether we have to append
		auto mode = std::ofstream::out;
		if (_append)
			mode |= std::ofstream::app;

		// Create the ofstream object
		this->log = std::make_shared<std::ofstream>(_filename, mode);

		// If the file could not be opened, reset the pointer
		if (!this->log->is_open())
		{
			this->log = nullptr;
			return false;
		}

		// Set the logstream to the file stream
		this->logstream = &(*log);
		this->ApplyPrecision(this->logstream, this->logPrecision);

		return true;
	}

	bool OutputHandler::SetDataFile(const std::string &_filename, bool _append)
	{
		// If we already have a data file, make sure to close it
		if (this->data != nullptr && this->data->is_open())
			this->data->close();

		// Check whether we have to append
		auto mode = std::ofstream::out;
		if (_append)
			mode |= std::ofstream::app;

		// Create the ofstream object
		this->data = std::make_shared<std::ofstream>(_filename, mode);

		// If the file could not be opened, reset the pointer
		if (!this->data->is_open())
		{
			this->data = nullptr;
			return false;
		}

		// Set the datastream to the file stream
		this->datastream = &(*data);
		this->ApplyPrecision(this->datastream, this->dataPrecision);

		return true;
	}

	// Returns an ostream reference of the logfile (if any; std::cout otherwise)
	std::ostream &OutputHandler::Log(const MessageType &_msgtype) const
	{
		// First line: If we have a normal message, and if the message if of high enough priority,
		// OR Second line: If we do not have an error, and if we have a warning, and if warnings should not be displayed,
		// OR Third line: If we have an error, and if errors should not be displayed
		if ((((_msgtype & NormalMessage) == _msgtype) && (_msgtype < (this->notificationLevel & NormalMessage))) || (!(_msgtype & MessageType_Error) && (_msgtype & MessageType_Warning) && !(this->notificationLevel & MessageType_Warning)) || ((_msgtype & MessageType_Error) && !(this->notificationLevel & MessageType_Error)))
		{
			// Create a temporary "garbage stream" to catch all the unwanted text
			// TODO: A more elegant solution could be implemented...
			this->ignored_messages_stream->str() = ""; // Clean-up in case there has been written many times to this stream
			this->ignored_messages_stream->clear();
			return *(this->ignored_messages_stream);
		}

		if (this->logstream == nullptr)
			return std::cout;

		return (*logstream);
	}

	// Returns an ostream reference to the data file (if any; the logfile otherwise)
	std::ostream &OutputHandler::Data() const
	{
		if (this->datastream == nullptr)
			return this->Log();

		return (*datastream);
	}

	// Sets the log stream (i.e. when using something else than a file or std::cout)
	bool OutputHandler::SetLogStream(std::ostream &_stream)
	{
		// If we have a logfile, make sure to close it
		if (this->log != nullptr && this->log->is_open())
			this->log->close();

		// And reset the file stream pointer
		this->log = nullptr;

		// Set the new stream
		this->logstream = &_stream;
		this->ApplyPrecision(this->logstream, this->logPrecision);

		return true;
	}

	// Sets the data stream (i.e. when using something else than a file or std::cout)
	bool OutputHandler::SetDataStream(std::ostream &_stream)
	{
		// If we have a datafile, make sure to close it
		if (this->data != nullptr && this->data->is_open())
			this->data->close();

		// And reset the file stream pointer
		this->data = nullptr;

		// Set the new stream
		this->datastream = &_stream;
		this->ApplyPrecision(this->datastream, this->dataPrecision);

		return true;
	}

	bool OutputHandler::SetLogPrecision(unsigned int _precision)
	{
		if (_precision == 0)
			return false;
		this->logPrecision = _precision;
		this->ApplyPrecision(this->logstream, this->logPrecision);
		return true;
	}

	bool OutputHandler::SetDataPrecision(unsigned int _precision)
	{
		if (_precision == 0)
			return false;
		this->dataPrecision = _precision;
		this->ApplyPrecision(this->datastream, this->dataPrecision);
		return true;
	}

	void OutputHandler::ApplyPrecision(std::ostream *_stream, unsigned int _precision) const
	{
		if (_stream == nullptr)
			return;
		(*_stream) << std::setprecision(_precision);
	}

	// Sets the notification level (i.e. which types of messages to show)
	bool OutputHandler::SetNotificationLevel(const MessageType &_msgtype)
	{
		// Only set the bits that are used
		this->notificationLevel = _msgtype & (NormalMessage | MessageType_Warning | MessageType_Error);
		return (this->notificationLevel == _msgtype);
	}
	// -----------------------------------------------------
}
