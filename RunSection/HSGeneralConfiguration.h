/////////////////////////////////////////////////////////////////////////
// HSGeneralConfiguration (RunSection module)
//
// Resolves the orthogonal physical choices of the additive HSGeneral task
// to an existing, validated MolSpin task implementation. Keeping this
// decision in one place prevents orchestration layers from duplicating the
// task compatibility matrix while legacy task names remain fully supported.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_HSGeneralConfiguration
#define MOD_RunSection_HSGeneralConfiguration

#include <string>
#include <random>
#include <iosfwd>
#include <vector>
#include "MSDParserfwd.h"
#include "SpinAPIfwd.h"
#include "SpinSpace.h"

namespace RunSection
{
	enum class HSGeneralTarget
	{
		StaticDirectTimeEvolution,
		StaticDirectYields,
		StaticStochasticTimeEvolution,
		StaticStochasticYields,
		DynamicDirectTimeEvolution,
		DynamicDirectYields,
		DynamicStochasticTimeEvolution,
		DynamicStochasticYields,
		StaticDirectSpectra,
		StaticResonanceSpectra
	};

	struct HSGeneralConfiguration
	{
		HSGeneralTarget target = HSGeneralTarget::StaticDirectTimeEvolution;
		std::string dynamics = "static";
		std::string calculation = "timeevolution";
		std::string sampling = "direct";
		std::string approximation = "full";
		bool powderAveraging = false;
	};

	bool ResolveHSGeneralConfiguration(const MSDParser::ObjectParser &_properties,
									 HSGeneralConfiguration &_configuration,
									 std::string &_error);
	bool IsHSGeneralTask(const MSDParser::ObjectParser &_properties);
	bool ValidateHSGeneralTraceSamplingSystems(const std::vector<SpinAPI::system_ptr> &_systems,
										 std::string &_error);
	void SeedHSGeneralRandomGenerator(const MSDParser::ObjectParser &_properties,
								  std::mt19937 &_generator,
								  std::ostream &_log);
	bool BuildHSGeneralTraceSamples(const MSDParser::ObjectParser &_properties,
									const SpinAPI::system_ptr &_system,
									SpinAPI::SpinSpace &_space,
									arma::uword _sampleCount,
									std::mt19937 &_generator,
									SpinAPI::HilbertTraceSampleSet &_samples,
										std::ostream &_log,
										std::string &_error);
	bool BuildHSGeneralInitialDensityMatrix(const SpinAPI::system_ptr &_system,
										 SpinAPI::SpinSpace &_space,
										 arma::cx_mat &_density,
										 std::string &_error);
	bool BuildHSGeneralInitialStateFactors(const SpinAPI::system_ptr &_system,
										SpinAPI::SpinSpace &_space,
										arma::cx_mat &_factors,
										std::ostream &_log,
										std::string &_error);
	const char *HSGeneralTargetName(HSGeneralTarget _target);
}

#endif
