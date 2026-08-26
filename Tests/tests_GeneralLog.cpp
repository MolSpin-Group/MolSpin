//////////////////////////////////////////////////////////////////////////////
// Shared General logging utility tests.
//////////////////////////////////////////////////////////////////////////////
#include "General/GeneralLog.h"
#include "Spin.h"
#include "SpinSystem.h"
#include "State.h"
#include "ObjectParser.h"
#include <sstream>

bool test_general_log_progress_is_throttled()
{
    size_t reports=0;
    for(size_t i=0;i<3200;++i)
        if(RunSection::General::Log::ShouldReportOrientation(i,3200)) ++reports;
    return reports>=10 && reports<=12 &&
        RunSection::General::Log::ShouldReportOrientation(0,3200) &&
        RunSection::General::Log::ShouldReportOrientation(3199,3200);
}

bool test_general_log_system_inventory_contains_dimensions_and_initial_state()
{
    auto spin=std::make_shared<SpinAPI::Spin>("E","type=electron;spin=1/2;");
    auto up=std::make_shared<SpinAPI::State>("Up","spin(E)=|1/2>;");
    auto system=std::make_shared<SpinAPI::SpinSystem>("System");
    system->Add(spin); system->Add(up);
    if(!up->ParseFromSystem(*system)) return false;
    system->SetProperties(std::make_shared<MSDParser::ObjectParser>("properties","initialstate=Up;"));
    std::ostringstream log;
    RunSection::General::Log::PrintSystemInventory(log,{system},"test inventory");
    const std::string text=log.str();
    return text.find("Hilbert dimension=2")!=std::string::npos &&
        text.find("Liouville dimension=4")!=std::string::npos &&
        text.find("1*Up")!=std::string::npos;
}

void AddGeneralLogTests(std::vector<test_case> &cases)
{
    cases.push_back({"GeneralLog throttles orientation progress",test_general_log_progress_is_throttled});
    cases.push_back({"GeneralLog reports dimensions and initial state",test_general_log_system_inventory_contains_dimensions_and_initial_state});
}
