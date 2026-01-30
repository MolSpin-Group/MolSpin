/////////////////////////////////////////////////////////////////////////
// HeapProfiler
// ------------------
// Tracks the heap allocation for given task
//
// Molecular Spin Dynamics Software - developed by Claus Nielsen and Luca Gerhards.
// (c) 2025 Quantum Biology and Computational Physics Group.
// See LICENSE.txt for license information.
/////////////////////////////////////////////////////////////////////////
#ifndef MOD_RunSection_Heapprofiler
#define MOD_RunSection_Heapprofiler

#define _GNU_SOURCE

#include <stdint.h>
#include <memory>
#include <map>
#include <cstdlib>
#include <stacktrace>
#include <string>
#include <stdio.h>
#include <filesystem>
#include <mutex>

namespace RunSection
{
    struct Ptr
    {
        std::string ptr;
        size_t size;
    };

    struct Function
    {
        std::string name;
        int allocation_calls = 0;
        int free_calls = 0;
        size_t allocation = 0;
        size_t freed = 0;

        
    };

    enum class Mode
    {
        def = 0,
        allocate,
        free
    };
    struct ProfilerStats
    {
        uint32_t m_TotallAllocated = 0;
        uint32_t m_TotalFreed = 0;
        uint32_t m_Peak = 0;
        uint32_t m_Total = 0;
        uint32_t m_LastAllocation = 0;
        uint32_t m_NumberOfMemoryCalls = 0;
        bool m_Profile = true;
        Mode m_mode = Mode::def;

        //std::vector<Ptr> m_AllocatedPointers;
        std::map<std::string, size_t> m_AllocatedPointers;
        std::map<std::string, Function> m_FunctionProfile;

        std::mutex m_MemoryLock;
        std::mutex m_StackLock;
        std::mutex m_WriteLock;
        std::mutex m_FunctionLock;

        std::string directory = "";
        bool m_started = false;
        FILE* m_StatsFile = nullptr;
        FILE* m_MemoryTracking = nullptr;
        FILE* m_Stack = nullptr;
    };

    class Profiler
    {
    private:
        static void CalcTotal();

        static void CreateDirectory();
        static void OpenFiles();
        static void CloseFiles();
        static void WriteMemory();
    public:
        static void GetAllocated(size_t size, std::string ptr_address);
        static void GetFreed(size_t size);
        static uint32_t GetPeak();
        static void Update();
        static void UpdateFunctionProfile(std::string function);

        static void WriteStack(char*);

        static void StartProfiling();
        static void StopProfiling();
    };
}

void StartProfiling();
void StopProfiling();

void* operator new(size_t size);//, std::source_location location = std::source_location::current());
void operator delete(void* ptr, size_t size);//, std::source_location location = std::source_location::current());
void* malloc(size_t __size);//, std::source_location location = std::source_location::current());
void free(void* ptr);//, std::source_location location = std::source_location::current());
int posix_memalign(void** _memptr, size_t _alignment, size_t _size);//, std::source_location location = std::source_location::current());
void* realloc(void* ptr, size_t new_size);
//void* calloc(size_t num, size_t size);


std::string ConvertToString(void* ptr);

bool InternalFrame(void* addr);
void PrintStackTrace();

void bt_error_callback(void* data, const char* msg, int errnum);
int bt_full_callback(void* data, uintptr_t pc, const char* filename, int lineno, const char* function);

#endif



