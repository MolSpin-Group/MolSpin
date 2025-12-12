#include "HeapProfiler.h"

static RunSection::ProfilerStats s_ProfilerStats;
#include <cstdlib>
#include <iostream>
#include <algorithm>

namespace RunSection
{
    void Profiler::CalcTotal()
    {
        uint32_t total = s_ProfilerStats.m_TotallAllocated - s_ProfilerStats.m_TotalFreed;
        if(total > s_ProfilerStats.m_Peak)
        {
            s_ProfilerStats.m_Peak = total;
        }
    }

    void Profiler::GetAllocated(size_t size)
    {
        if(s_ProfilerStats.m_Profile)
        {
            s_ProfilerStats.m_TotallAllocated += size;
            s_ProfilerStats.m_Total += size;
        }
    }

    void Profiler::GetFreed(size_t size)
    {
        if(s_ProfilerStats.m_Profile)
        {
            s_ProfilerStats.m_TotalFreed += size;
            s_ProfilerStats.m_Total -= size;
        }
    }

    uint32_t Profiler::GetPeak()
    {
        return s_ProfilerStats.m_Peak;
    }

    void Profiler::Update()
    {
        if(s_ProfilerStats.m_Profile)
        {
            CalcTotal();
            std::cout << s_ProfilerStats.m_Total << std::endl;
        }
    }

    void Profiler::StartProfiling()
    {
        s_ProfilerStats.m_Profile = true;
    }

    void Profiler::StopProfiling()
    {
        s_ProfilerStats.m_Profile = false;
    }
    
    void Profiler::UpdatePtrTracker(void* ptr, size_t size)
    {
        bool fre = false;
        if(size == 0)
            fre = true;
        //else
        //    s_ProfilerStats.m_PtrTracker.push_back({ptr, size});
        
        if(!fre)
            return;

        //auto i = std::find_if(s_ProfilerStats.m_PtrTracker.begin(), s_ProfilerStats.m_PtrTracker.end(), [ptr](Ptr a) {
        //    return a.ptr == ptr;
        //});

        //f(i == s_ProfilerStats.m_PtrTracker.end())
        //   return;
        
        //RunSection::Profiler::GetFreed((*i).size);
        //s_ProfilerStats.m_PtrTracker.erase(i);

        //RunSection::Profiler::Update();
        return;
    }
}

void* mallocLog(size_t __size)
{
    RunSection::Profiler::GetAllocated(__size);
    RunSection::Profiler::Update();
    #undef malloc
    void* ptr = malloc(__size);
    RunSection::Profiler::UpdatePtrTracker(ptr, __size);
    return ptr;
}

void* callocLog(size_t nmemb, size_t __size)
{
    RunSection::Profiler::GetAllocated(__size);
    RunSection::Profiler::Update();
    #undef calloc
    void* ptr = calloc(nmemb,__size);
    RunSection::Profiler::UpdatePtrTracker(ptr, __size);
    return ptr;
}

int posix_memalignLog(void** _memptr, size_t _alignment, size_t _size)
{
    RunSection::Profiler::GetAllocated(_size);
    RunSection::Profiler::Update();
    #undef posix_memalign
    return posix_memalign(_memptr, _alignment, _size);
}

void freeLog(void *ptr)
{
    RunSection::Profiler::UpdatePtrTracker(ptr);
    #undef free
    free(ptr);
}

void *operator new(size_t size)
{
    //RunSection::Profiler::GetAllocated(size);
    //RunSection::Profiler::Update();
    return mallocLog(size);
    //return malloc(size);
}

void operator delete(void *ptr, size_t size)
{
    RunSection::Profiler::GetFreed(size);
    RunSection::Profiler::Update();
    freeLog(ptr);
}


