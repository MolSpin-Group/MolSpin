#ifndef MOD_RunSection_General_TaskSpec
#define MOD_RunSection_General_TaskSpec

#include "ObjectParser.h"

#include <iosfwd>
#include <map>
#include <string>
#include <vector>

namespace RunSection::General
{
	struct TaskSpecProperty
	{
		std::vector<std::string> aliases;
		std::string type;
		std::string defaultJson;
		std::vector<std::string> choices;
		std::string description;
	};

	const char *TaskSpecVersion();
	const std::map<std::string, TaskSpecProperty> &TaskSpecProperties(const std::string &taskName);
	std::vector<std::string> TaskSpecInputKeys(const std::string &taskName, const std::string &propertyName);
	bool IsTaskSpecAlias(const std::string &taskName, const std::string &alias);
	bool WriteTaskSpec(std::ostream &output, const std::string &taskName = "");

	template <typename T>
	bool GetTaskProperty(const MSDParser::ObjectParser &properties,
		const std::string &taskName, const std::string &propertyName, T &value)
	{
		for (const auto &key : TaskSpecInputKeys(taskName, propertyName))
			if (properties.Get(key, value))
				return true;
		return false;
	}

	template <typename T>
	bool GetTaskPropertyList(const MSDParser::ObjectParser &properties,
		const std::string &taskName, const std::string &propertyName,
		std::vector<T> &value, const char delimiter = ',')
	{
		for (const auto &key : TaskSpecInputKeys(taskName, propertyName))
			if (properties.GetList(key, value, delimiter))
				return true;
		return false;
	}

	template <typename T>
	bool GetTaskPulseSequence(const MSDParser::ObjectParser &properties,
		const std::string &taskName, const std::string &propertyName, T &value)
	{
		for (const auto &key : TaskSpecInputKeys(taskName, propertyName))
			if (properties.GetPulseSequence(key, value))
				return true;
		return false;
	}
}

#endif
