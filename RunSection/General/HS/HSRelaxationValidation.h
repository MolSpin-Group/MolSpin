// Shared policy for legacy propagation representations without relaxation.
#ifndef MOLSPIN_HS_RELAXATION_VALIDATION_H
#define MOLSPIN_HS_RELAXATION_VALIDATION_H
#include "Operator.h"
#include "SpinSystem.h"
#include "SpinSpace.h"
#include <string>
#include <vector>
namespace RunSection::General::HS
{
	inline bool ValidateNoRelaxation(const std::vector<SpinAPI::system_ptr> &systems,
		const std::string &representation, std::string &error)
	{
		error.clear();
		for (const auto &system : systems)
			for (const auto &op : system->Operators())
				if (!op || !op->IsValid() || SpinAPI::HasNonzeroRelaxationRate(op))
				{
					error = representation + " does not support relaxation Operator \"" +
						(op ? op->Name() : std::string("<null>")) +
						"\"; use a compatible HSGeneral or SSGeneral calculation";
					return false;
				}
		return true;
	}
}
#endif
