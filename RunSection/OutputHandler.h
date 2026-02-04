/////////////////////////////////////////////////////////////////////////
// OutputHandler (RunSection module)
// ------------------
// Handles the output streams that are available to task classes.
//
// TODO: Consider parallel access to files when implementing MPI version.
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_OutputHandler
#define MOD_RunSection_OutputHandler

#include <memory>
#include <fstream>
#include <sstream>
#include "RunSectionDefines.h"

namespace RunSection
{
	class OutputHandler
	{
	private:
		// Implementation
		std::shared_ptr<std::ofstream> log;							 // File stream
		std::shared_ptr<std::ofstream> data;						 // File stream
		std::ostream *logstream;									 // Can be another type of stream
		std::ostream *datastream;									 // Can be another type of stream
		unsigned int logPrecision;
		unsigned int dataPrecision;
		std::shared_ptr<std::ostringstream> ignored_messages_stream; // A "garbage stream" for messages that should not be written to the other streams
		MessageType notificationLevel;

		void ApplyPrecision(std::ostream *, unsigned int) const;

	public:
		// Constructors / Destructors
		OutputHandler();							   // Normal constructor
		OutputHandler(const OutputHandler &) = delete; // Default Copy-constructor, no need to copy, so better delete than end up with splicing problems
		~OutputHandler();							   // Destructor

		// Operators
		const OutputHandler &operator=(const OutputHandler &) = delete; // Default Copy-assignment

		// Public methods
		bool SetLogFile(const std::string &_filename, bool _append = false);  // Write to a file
		bool SetDataFile(const std::string &_filename, bool _append = false); // Write to a file
		bool SetLogStream(std::ostream &);									  // Use an existing stream (used by testing module)
		bool SetDataStream(std::ostream &);									  // Use an existing stream (used by testing module)
		bool SetLogPrecision(unsigned int);
		bool SetDataPrecision(unsigned int);
		bool SetNotificationLevel(const MessageType &);						  // Sets the notification level (i.e. which types of messages to show)

		// Public const methods
		std::ostream &Log(const MessageType &_msgtype = MessageType_Normal) const;
		std::ostream &Data() const;
	};
}

#endif
