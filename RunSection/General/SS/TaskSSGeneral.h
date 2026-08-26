/////////////////////////////////////////////////////////////////////////
// TaskSSGeneral — thin RunSection shell for one-system superspace calculations.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_General_SS_TaskSSGeneral
#define MOD_RunSection_General_SS_TaskSSGeneral
#include "BasicTask.h"
#include "SSExecutionPlan.h"
namespace RunSection::General::SS
{
    class TaskSSGeneral final:public ::RunSection::BasicTask
    {
    public:TaskSSGeneral(const MSDParser::ObjectParser&,const ::RunSection::RunSection&);~TaskSSGeneral()override=default;
    protected:bool RunLocal()override;bool Validate()override;
    private:SSExecutionPlan plan;bool planResolved=false;bool WriteHeader(const std::vector<std::string>&,bool);
    };
}
#endif
