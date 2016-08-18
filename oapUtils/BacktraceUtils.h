
#ifndef BACKTRACEUTILS_H
#define	BACKTRACEUTILS_H

#include <string>
#include <vector>

class BacktraceUtils {
public:
    static BacktraceUtils& GetInstance();
    const std::vector<std::string>& readBacktrace(bool showThisObjectFrames = false);
    void printBacktrace(bool showThisObjectFrames = false);
protected:
    BacktraceUtils();
    BacktraceUtils(const BacktraceUtils& orig);
    virtual ~BacktraceUtils();
private:
    size_t m_size;
    void** m_buffer;
    int m_ptrsCount;
    char** m_strings;

    typedef std::vector<std::string> Backtrace;
    Backtrace m_backtrace;
    static BacktraceUtils* m_backtraceUtils;

    char** repeatReadBacktrace();
};

#endif	/* BACKTRACEUTILS_H */

