
#ifndef PARAMETERS_H
#define	PARAMETERS_H

#include "DebugLogs.h"
#include "Math.h"

class Parameters {
public:

    template <typename T>class Parameter {
    public:

        Parameter() : isAvailable(false) {
        }
        T value;
        bool isAvailable;
    };

    Parameters(bool isThreadsCount = true, bool isSerieLimit = false);
    virtual ~Parameters();

    bool isParamThreadsCount() const;
    void setThreadsCount(uintt count);
    uintt getThreadsCount() const;

    bool isParamSerieLimit() const;
    void setSerieLimit(uintt limit);
    uintt getSerieLimit() const;
private:
    Parameter<uintt> threadsCount;
    Parameter<uintt> serieLimit;
    Parameters(const Parameters& orig);
};

#endif	/* PARAMETERS_H */

