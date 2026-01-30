#include "HeapProfiler.h"

static RunSection::ProfilerStats s_ProfilerStats;
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <complex>
#include <execinfo.h>
#include <backtrace.h>
#include <cstring>
#include <cxxabi.h>
#include <chrono>
#include <ctime>

typedef void* (*malloc_fn)(size_t);
typedef int (*posix_memalign_fn)(void**, size_t, size_t);
typedef void (*free_fn)(void*);
typedef void* (*calloc_fn)(size_t, size_t);
typedef void*(*realloc_fn)(void*, size_t);

static malloc_fn realmalloc = NULL;
static malloc_fn current = NULL;
static posix_memalign_fn realposix = NULL;
static free_fn realfree = NULL;
static realloc_fn realrealloc = NULL;
static calloc_fn current2 = NULL;
static calloc_fn realcalloc = NULL;

static bool InProfiler = false;
static bool FirstCall = true;
static bool BottomFrame = true; //last molspin specific frame to be called i.e the important one

static backtrace_state *bt_state = nullptr;
static int level = 0;

static void mtrace_init(void)
{
    realmalloc = reinterpret_cast<malloc_fn>(dlsym(RTLD_NEXT, "malloc"));
    if (NULL == realmalloc) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
}

static void posixtrace_init(void)
{
    realposix = reinterpret_cast<posix_memalign_fn>(dlsym(RTLD_NEXT, "posix_memalign"));
    if (NULL == realposix) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
}

static void freetrace_init(void)
{
    realfree = reinterpret_cast<free_fn>(dlsym(RTLD_NEXT, "free"));
    if (NULL == realfree) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
}

static void calloctrace_init(void)
{
    realcalloc = reinterpret_cast<calloc_fn>(dlsym(RTLD_NEXT, "calloc"));
    if (NULL == realcalloc) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
}

static void realloctrace_init(void)
{
    realrealloc = reinterpret_cast<realloc_fn>(dlsym(RTLD_NEXT, "realloc"));
    if (NULL == realrealloc)
    {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
    }
}

namespace RunSection
{
    void Profiler::CalcTotal()
    {
        uint32_t total = s_ProfilerStats.m_TotallAllocated - s_ProfilerStats.m_TotalFreed;
        if(total > s_ProfilerStats.m_Peak)
        {
            s_ProfilerStats.m_Peak = total;
        }
        if(s_ProfilerStats.m_TotalFreed > s_ProfilerStats.m_TotallAllocated)
        {
            std::cin.get();
        }
    }

    void Profiler::CreateDirectory()
    {
        auto now = std::chrono::system_clock::now();
        std::time_t time = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << time;
        std::string ts = ss.str();
        std::filesystem::create_directory(ts);
        s_ProfilerStats.directory = ts;
    }

    void Profiler::OpenFiles()
    {
        if(s_ProfilerStats.m_started == false)
        {
            CreateDirectory();
            //s_ProfilerStats.m_started = true;
        }
        std::string stats_file = s_ProfilerStats.directory + "/functionstats.csv";
        std::string memory = s_ProfilerStats.directory + "/memory.csv";
        std::string stack = s_ProfilerStats.directory + "/stack.txt";
        s_ProfilerStats.m_StatsFile = fopen(stats_file.c_str(), "ab+");
        s_ProfilerStats.m_MemoryTracking = fopen(memory.c_str(), "ab+");
        s_ProfilerStats.m_Stack = fopen(stack.c_str(), "ab+");
        if(s_ProfilerStats.m_started == false)
        {
            fprintf(s_ProfilerStats.m_MemoryTracking, "call,total (bytes),\n");
            fprintf(s_ProfilerStats.m_StatsFile, "function name,allocation calls,total allocated,free calls,total freed,\n");
            s_ProfilerStats.m_started = true;
        }
    }

    void Profiler::CloseFiles()
    {
        fclose(s_ProfilerStats.m_StatsFile);
        fclose(s_ProfilerStats.m_MemoryTracking);
        fclose(s_ProfilerStats.m_Stack);
    }

    void Profiler::WriteMemory()
    {
        s_ProfilerStats.m_WriteLock.lock();
        if(!InProfiler)
        {
            s_ProfilerStats.m_WriteLock.unlock();
            return;
        }
        OpenFiles();
        fprintf(s_ProfilerStats.m_MemoryTracking, "%d,%d,\n", s_ProfilerStats.m_NumberOfMemoryCalls, s_ProfilerStats.m_Total);
        CloseFiles();
        s_ProfilerStats.m_WriteLock.unlock();
        return;
    }

    void Profiler::WriteStack(char* buffer)
    {
        s_ProfilerStats.m_WriteLock.lock();
        if(!InProfiler)
        {
            s_ProfilerStats.m_WriteLock.unlock();
            return;
        }
        OpenFiles();
        fprintf(s_ProfilerStats.m_Stack, "%s", buffer);
        CloseFiles();
        s_ProfilerStats.m_WriteLock.unlock();
        return;
    }

    void Profiler::GetAllocated(size_t size, std::string ptr_address)
    {
        if(s_ProfilerStats.m_Profile)
        {
            s_ProfilerStats.m_MemoryLock.lock();
            s_ProfilerStats.m_TotallAllocated += size;
            s_ProfilerStats.m_Total += size;
            s_ProfilerStats.m_mode = Mode::allocate;
            s_ProfilerStats.m_LastAllocation = size;
            //std::cout << "Allocated: " << ptr_address << " size: " << size << std::endl;
            //fprintf(stderr, ptr_address.c_str());
            s_ProfilerStats.m_AllocatedPointers[ptr_address] = size;
            s_ProfilerStats.m_MemoryLock.unlock();
        }
        return;
    }

    void Profiler::GetFreed(size_t size)
    {
        if(s_ProfilerStats.m_Profile)
        {
            s_ProfilerStats.m_MemoryLock.lock();
            s_ProfilerStats.m_mode = Mode::free;
            s_ProfilerStats.m_LastAllocation = size;
            s_ProfilerStats.m_TotalFreed += size;
            s_ProfilerStats.m_Total -= size;
            s_ProfilerStats.m_MemoryLock.unlock();
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
            s_ProfilerStats.m_MemoryLock.lock();
            CalcTotal();
            WriteMemory();
            s_ProfilerStats.m_NumberOfMemoryCalls += 1;
            //std::cout << s_ProfilerStats.m_Total << std::endl;
            //fprintf(stderr,"%d \n",s_ProfilerStats.m_Total);
            s_ProfilerStats.m_MemoryLock.unlock();
        }
        return;
    }

    void Profiler::UpdateFunctionProfile(std::string function)
    {
        s_ProfilerStats.m_FunctionLock.lock();
        if(BottomFrame == false)
        {
            s_ProfilerStats.m_FunctionLock.unlock();
            return;
        }
        //BottomFrame = false;
        Mode current_mode = s_ProfilerStats.m_mode;
        size_t allocation = s_ProfilerStats.m_LastAllocation;
        bool present = false;
        auto it = s_ProfilerStats.m_FunctionProfile.find(function);
        if(it != s_ProfilerStats.m_FunctionProfile.end())
        {
            present = true;
        }
        if(present)
        {
            if(current_mode == Mode::allocate)
            {
                s_ProfilerStats.m_FunctionProfile[function].allocation_calls += 1;
                s_ProfilerStats.m_FunctionProfile[function].allocation += allocation;
            }
            else
            {
                s_ProfilerStats.m_FunctionProfile[function].free_calls += 1;
                s_ProfilerStats.m_FunctionProfile[function].freed += allocation;
            }
            //BottomFrame = true;
            s_ProfilerStats.m_FunctionProfile[function].LevelCall(level);
            s_ProfilerStats.m_FunctionLock.unlock();
            return;
        }
        Function functionstats;
        functionstats.name = function;
        if(current_mode == Mode::allocate)
        {
            functionstats.allocation += allocation;
            functionstats.allocation_calls += 1;
        }
        else
        {
            functionstats.free_calls += 1;
            functionstats.freed += allocation;
        }
        s_ProfilerStats.m_FunctionProfile[function] = functionstats;
        s_ProfilerStats.m_FunctionProfile[function].LevelCall(level);
        s_ProfilerStats.m_FunctionLock.unlock();
        return;
    }

    void Profiler::StartProfiling()
    {
        s_ProfilerStats.m_Profile = true;
    }

    void Profiler::StopProfiling()
    {
        s_ProfilerStats.m_Profile = false;
        //std::cout << s_ProfilerStats.m_Peak << std::endl;
        fprintf(stderr, "Peak memory: %d bytes\n",s_ProfilerStats.m_Peak);
        fprintf(stderr, "Function stats\n");
        OpenFiles();
        for(auto it = s_ProfilerStats.m_FunctionProfile.begin(); it != s_ProfilerStats.m_FunctionProfile.end(); it++)
        {
            auto f = it->second;
            fprintf(stderr, "\"%s\": %d %d %d %d\n", f.name.c_str(), f.allocation_calls, f.allocation, f.free_calls, f.freed);
            fprintf(s_ProfilerStats.m_StatsFile,"%s,%d,%d,%d,%d, ,",f.name.c_str(), f.allocation_calls, f.allocation, f.free_calls, f.freed);
            for(int i = 0; i < f.level_calls.size(); i++)
            {
                fprintf(s_ProfilerStats.m_StatsFile, "%s,",f.level_calls[i]);
            }
            fprintf(s_ProfilerStats.m_StatsFile, "\n");
        }
        CloseFiles();
    }
}

void* my_malloc(size_t __size)
{
    if(realmalloc == NULL)
    {
        mtrace_init();
        fprintf(stderr, "real malloc retrieved\n");
        return realmalloc(__size);
    }
    if(InProfiler)
    {
        //fprintf(stderr, "in profiler\n");
        void* ptr = realmalloc(__size);
        //fprintf(stderr, "out profiler\n");
        return ptr;
    }
    InProfiler = true;
    void* ptr = realmalloc(__size);

    //fprintf(stderr, "malloc called from:\n");
    PrintStackTrace();

    RunSection::Profiler::GetAllocated(__size, ConvertToString(ptr));
    RunSection::Profiler::Update();

    //auto trace = std::sta
    //std::cout << ConvertToString(ptr) << " size: " << __size << std::endl;
    //std::cout << "from file: " << location.file_name() << " line: " << location.line() << std::endl;
    //std::cin.get();
    InProfiler = false;
    return ptr;
}

void* malloc(size_t __size)//, std::source_location location)
{
    if(realmalloc == NULL)
    {
        mtrace_init();
        fprintf(stderr, "real malloc retrieved\n");
        current = realmalloc;
        return realmalloc(__size);
    }
    return current(__size);
}

int posix_memalign(void** _memptr, size_t _alignment, size_t _size)//, std::source_location location)
{
    if(realposix == NULL)
    {
        posixtrace_init();
    }
    if(InProfiler)
    {
        return realposix(_memptr, _alignment, _size);
    }
    InProfiler = true;
    RunSection::Profiler::GetAllocated(_size, ConvertToString(*_memptr));
    RunSection::Profiler::Update();

    //void* buffer[20];
    //int nptrs = backtrace(buffer, 20);
    //fprintf(stderr, "posix_memalign called from:\n");
    //backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);

    PrintStackTrace();

    //std::cout << "Aligned Allocated: " << ConvertToString(*_memptr) << " size: " << _size << std::endl;
    //std::cout << "from file: " << std::get<0>(frame) << " line: " << std::get<1>(frame) << std::endl;
    //std::cin.get();
    InProfiler = false;
    return realposix(_memptr, _alignment, _size);
}

//void* my_calloc(size_t num, size_t size)
//{;
//    if(realcalloc == NULL)
//    {
//        calloctrace_init();
//    }
//    if(InProfiler)
//    {
//        return realcalloc(num, size);
//    }
//    InProfiler = true;
//    //fprintf(stderr, "calloc called");
//    void* ptr = realcalloc(num, size);
//    //RunSection::Profiler::GetAllocated(size * num, ConvertToString(ptr));
//    //RunSection::Profiler::Update();
//
//    //fprintf(stderr, "calloc called from:\n");
//    //PrintStackTrace();
//    InProfiler = false;
//    return ptr;
//}
//
//void* calloc(size_t num, size_t size)//, std::source_location location)
//{
//    if(realcalloc == NULL)
//    {
//        calloctrace_init();
//        fprintf(stderr, "real calloc retrieved\n");
//        current2 = realcalloc;
//        return realcalloc(num, size);
//    }
//    return current2(num, size);
//}

void* realloc(void* ptr, size_t new_size)
{
    if(realcalloc == NULL)
    {
        realloctrace_init();
    }
    if(InProfiler)
    {
        return realrealloc(ptr, new_size);
    }
    InProfiler = true;
    std::string address = ConvertToString(ptr);
    void* new_ptr = realrealloc(ptr, new_size);
    if(new_ptr != NULL)
    {
        size_t size = 0;
        auto it = s_ProfilerStats.m_AllocatedPointers.find(address);
        if (it != s_ProfilerStats.m_AllocatedPointers.end())
        {
            size = it->second;
            RunSection::Profiler::GetFreed(size);
            RunSection::Profiler::Update();
            s_ProfilerStats.m_AllocatedPointers.erase(it);
        }
        RunSection::Profiler::GetAllocated(new_size, ConvertToString(new_ptr));
        RunSection::Profiler::Update();
    }

    //fprintf(stderr, "realloc called from:\n");
    PrintStackTrace();
    InProfiler = false;
    return new_ptr;
}

void free(void* ptr)//, std::source_location location)
{
    if(realfree == NULL)
    {
        freetrace_init();
    }
    if(InProfiler)
    {
        realfree(ptr);
        return;
    }
    InProfiler = true;
    std::string address = ConvertToString(ptr);
    size_t size = 0;
    auto it = s_ProfilerStats.m_AllocatedPointers.find(address);
    if (it != s_ProfilerStats.m_AllocatedPointers.end())
    {
        size = it->second;
        RunSection::Profiler::GetFreed(size);
        RunSection::Profiler::Update();
        s_ProfilerStats.m_AllocatedPointers.erase(it);
    }
    InProfiler = false;
    realfree(ptr);
}

std::string ConvertToString(void *ptr)
{
    std::string address;
    std::stringstream ss;
    ss << ptr;
    address = ss.str();
    return address;
}

bool InternalFrame(const char* filename)
{
    auto find_char = [](const char* buf1, size_t buf1l, const char* buf2, size_t buf2l) {
        if(buf2l == 0 || buf2l > buf1l)
        {
            //fprintf(stderr, "check string bigger than filename");
            return (char*)nullptr;
        }
        for(size_t i = 0; i <= buf1l - buf2l; i++) {
            if(memcmp(buf1 + i, buf2, buf2l) == 0)
            {
                return (char*)(buf1 + i);
            }
        }
        //fprintf(stderr, "not internal frame");
        return (char*)nullptr;
    };

    unsigned int l1 = strlen(filename);
    char lower[l1];
    for(int i = 0; i < l1; i++)
    {
        lower[i] = tolower(filename[i]);
    }
    char* pos;
    char internalframe[] = "heapprofiler"; 
    pos = find_char(lower, strlen(lower), internalframe, strlen(internalframe));
    if(pos)
        return true;
    char SPINAPI[] = "/spinapi/";
    char RUNSECTION[] = "/runsection/";
    char MSDPARSER[] = "/msdparser/";
    char MS[] = "/MolSpin/";
    char armadillo[] = "/armadillo_bits/";
    char armadillo2[] = "/armadillo_bits/memory.hpp";
    pos = find_char(lower, strlen(lower), SPINAPI, strlen(SPINAPI));
    if(pos)
        return false;
    pos = find_char(lower, strlen(lower), RUNSECTION, strlen(RUNSECTION));
    if(pos)
        return false;
    pos = find_char(lower, strlen(lower), MSDPARSER, strlen(MSDPARSER));
    if(pos)
        return false;
    pos = find_char(lower, strlen(lower), MS, strlen(MS));
    if(pos)
        return false;
    pos = find_char(lower, strlen(lower), armadillo2, strlen(armadillo2));
    if(pos)
        return true;
    pos = find_char(lower, strlen(lower), armadillo, strlen(armadillo));
    if(pos)
        return false;
    char nonMS[] = "/usr/";
    pos = find_char(lower, strlen(lower), nonMS, strlen(nonMS));
    if(pos)
        return true;
    
    return true;
}

void PrintStackTrace()
{
    s_ProfilerStats.m_StackLock.lock();
    if(bt_state == nullptr)
    {
        s_ProfilerStats.m_StackLock.unlock();
        return;
    }
    void* buffer[32] = {};
    int nptrs = backtrace(buffer, 32);
    char hdr[128];
    int len = snprintf(hdr, sizeof(hdr), "Memory allocation stack trace (most recent call last), total frames: %d\n", nptrs);
    //write(STDERR_FILENO, hdr, len);
    RunSection::Profiler::WriteStack(hdr);
    BottomFrame = true;
    level = 0;
    for(int i = 0; i < nptrs; i++)
    {
        backtrace_pcinfo(bt_state, (uintptr_t)buffer[i], bt_full_callback, bt_error_callback, nullptr);
    }
    s_ProfilerStats.m_StackLock.unlock();
}

char* demangle(const char* s)
{
    int status = 0;
    char* demangle = abi::__cxa_demangle(s,nullptr,nullptr,&status);
    if(status == 0 && demangle != nullptr) {
        return demangle;
    }
    return strdup(s);

} 

void bt_error_callback(void *data, const char *msg, int errnum)
{
    const char prefix[] = "libbacktrace error: ";
    write(STDERR_FILENO, prefix, sizeof(prefix) - 1);
    if(msg)
    {
        write(STDERR_FILENO, msg, strlen(msg));
    }
    const char newline = '\n';
    write(STDERR_FILENO, &newline, 1);
}

int bt_full_callback(void *data, uintptr_t pc, const char *filename, int lineno, const char *function)
{
    if(filename == nullptr || lineno == 0 || function == nullptr)
    {
        return 0;
    }
    if(!InternalFrame(filename))
    {
        char buffer[1024];
        int len = 0;
        char* func = demangle(function);
        std::string function_name(func);
        len += snprintf(buffer, sizeof(buffer), " %s:%d (%s)\n", filename, lineno, func);
        //free(func);
        RunSection::Profiler::WriteStack(buffer);
        RunSection::Profiler::UpdateFunctionProfile(function_name);
        level = level + 1;
    }
    return 0;
}

void StartProfiling()
{
    fprintf(stderr,"Started profiling \n");
    RunSection::Profiler::StartProfiling();
}

void StopProfiling()
{
    RunSection::Profiler::StopProfiling();
}

void *operator new(size_t size)//, std::source_location location)
{
    return malloc(size);
}

void operator delete(void *ptr, size_t size)//, std::source_location location)
{
    free(ptr);
}

void __attribute__((constructor)) run_at_start() {
    InProfiler = true;
    bt_state = backtrace_create_state(nullptr, 0, bt_error_callback, nullptr);
    InProfiler = false;
    FirstCall = false;
    StartProfiling();
    void* tmp[32];
    int nptrs = backtrace(tmp,32);
    fprintf(stderr,"Swapping malloc func\n");
    current = reinterpret_cast<malloc_fn>(my_malloc);
    s_ProfilerStats.m_AllocatedPointers = {};
    // Mutexes are default-initialized; do not unlock unheld mutexes here.
    s_ProfilerStats.m_MemoryLock.unlock();
    s_ProfilerStats.m_StackLock.unlock();
    s_ProfilerStats.m_WriteLock.unlock();
    s_ProfilerStats.m_FunctionLock.unlock();
    //fprintf(stderr, "Swapping calloc function\n");
    //current2 = reinterpret_cast<calloc_fn>(my_calloc);
}

void __attribute__((destructor)) run_on_end() {
    InProfiler = true;
    fprintf(stderr,"Stopped profiling \n");
    StopProfiling();
}



