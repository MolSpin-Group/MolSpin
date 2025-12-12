/////////////////////////////////////////////////////////////////////////
// HeapProfiler (RunSection module)
// ------------------
// Tracks the heap allocation for given task
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_Heapprofiler
#define MOD_RunSection_Heapprofiler

#include <stdint.h>
#include <memory>
#include <vector>
#include <cstdlib>


namespace RunSection
{
    struct Ptr
    {
        void* ptr;
        size_t size;
    };
    struct ProfilerStats
    {
        uint32_t m_TotallAllocated = 0;
        uint32_t m_TotalFreed = 0;
        uint32_t m_Peak = 0;
        uint32_t m_Total = 0;
        bool m_Profile = false;

        std::vector<Ptr> m_PtrTracker= {};
    };

    class Profiler
    {
    private:
        static void CalcTotal();
    public:
        static void GetAllocated(size_t size);
        static void GetFreed(size_t size);
        static uint32_t GetPeak();
        static void Update();

        static void StartProfiling();
        static void StopProfiling();

        static void UpdatePtrTracker(void* ptr, size_t size = 0);
    };
}

void* operator new(size_t size);
void operator delete(void* ptr, size_t size);
void* mallocLog(size_t __size);
void* callocLog(size_t nmemb, size_t __size);
int posix_memalignLog(void** _memptr, size_t _alignment, size_t _size);
void freeLog(void* ptr);

#ifndef MemoryProfilingOverloads
#define MemoryProfilingOverloads
#define malloc(size) (mallocLog(size))
#define calloc(num, size) (callocLog(num, size))
#define posix_memalign(ptr, align, size) (posix_memalignLog(ptr, align, size))
#define free(ptr) (freeLog(ptr))
#endif


#endif



