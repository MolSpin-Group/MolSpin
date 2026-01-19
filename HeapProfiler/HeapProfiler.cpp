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

typedef void* (*malloc_fn)(size_t);
typedef int (*posix_memalign_fn)(void**, size_t, size_t);
typedef void (*free_fn)(void*);
static malloc_fn realmalloc = NULL;
static malloc_fn current = NULL;
static posix_memalign_fn realposix = NULL;
static free_fn realfree = NULL;
static bool InProfiler = false;
static bool FirstCall = true;

static backtrace_state *bt_state = nullptr;

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

    void Profiler::GetAllocated(size_t size, std::string ptr_address)
    {
        if(s_ProfilerStats.m_Profile)
        {
            s_ProfilerStats.m_TotallAllocated += size;
            s_ProfilerStats.m_Total += size;
            //std::cout << "Allocated: " << ptr_address << " size: " << size << std::endl;
            //fprintf(stderr, ptr_address.c_str());
            s_ProfilerStats.m_AllocatedPointers[ptr_address] = size;
        }
        return;
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
            //std::cout << s_ProfilerStats.m_Total << std::endl;
            //fprintf(stderr,"%d \n",s_ProfilerStats.m_Total);
        }
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
        fprintf(stderr, "in profiler\n");
        void* ptr = realmalloc(__size);
        fprintf(stderr, "out profiler\n");
        return ptr;
    }
    InProfiler = true;
    void* ptr = realmalloc(__size);
    RunSection::Profiler::GetAllocated(__size, ConvertToString(ptr));
    RunSection::Profiler::Update();

    fprintf(stderr, "malloc called from:\n");
    PrintStackTrace();

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
    fprintf(stderr, "posix_memalign called from:\n");
    //backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);

    PrintStackTrace();

    //std::cout << "Aligned Allocated: " << ConvertToString(*_memptr) << " size: " << _size << std::endl;
    //std::cout << "from file: " << std::get<0>(frame) << " line: " << std::get<1>(frame) << std::endl;
    //std::cin.get();
    InProfiler = false;
    return realposix(_memptr, _alignment, _size);
}

void free(void* ptr)//, std::source_location location)
{
    if(InProfiler)
    {
        realfree(ptr);
        return;
    }
    if(realfree == NULL)
    {
        freetrace_init();
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
    char nonMS[] = "/usr/";
    pos = find_char(lower, strlen(lower), nonMS, strlen(nonMS));
    if(pos)
        return true;
    return false;
}

void PrintStackTrace()
{
    if(bt_state == nullptr)
    {
        return;
    }
    fprintf(stderr, "in PrintStackTrace\n");
    void* buffer[32] = {};
    fprintf(stderr, "BufferCreated\n");
    int nptrs = backtrace(buffer, 32);
    //if(nptrs == 0)
    //{
    //    return;
    //}
    fprintf(stderr, "nptrs collected\n");
    char hdr[128];
    int len = snprintf(hdr, sizeof(hdr), "Memory allocation stack trace (most recent call last), total frames: %d\n", nptrs);
    write(STDERR_FILENO, hdr, len);
    //fprintf(stderr, hdr);
    for(int i = 0; i < nptrs; i++)
    {
        backtrace_pcinfo(bt_state, (uintptr_t)buffer[i], bt_full_callback, bt_error_callback, nullptr);
    }
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
    char buffer[1024];
    int len = 0;
    len += snprintf(buffer, sizeof(buffer), " %s:%d (%s)\n", filename, lineno, function);
    if(!InternalFrame(filename))
    //InternalFrame(filename);  
        write(STDERR_FILENO, buffer, len);
    //fprintf(stderr, buffer);
    //std::cin.get();
    
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
    //RunSection::Profiler::GetAllocated(size);
    //RunSection::Profiler::Update();
    return malloc(size);
    //return malloc(size);
}

void operator delete(void *ptr, size_t size)//, std::source_location location)
{
    //RunSection::Profiler::GetFreed(size);
    //RunSection::Profiler::Update();
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
}

void __attribute__((destructor)) run_on_end() {
    fprintf(stderr,"Stopped profiling \n");
    InProfiler = true;
    StopProfiling();
}



