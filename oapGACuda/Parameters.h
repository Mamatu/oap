/* 
 * File:   ParametersInterface.h
 * Author: mmatula
 *
 * Created on September 27, 2013, 9:58 PM
 */

#ifndef PARAMETERSINTERFACE_H
#define	PARAMETERSINTERFACE_H
#include "GACore.h"
#include <string.h>

class Parameters {
public:
    Parameters();
    virtual ~Parameters();

    virtual void setThreadsCount(uint threadsCount);
    virtual uint getThreadsCount();

    virtual void setSeed(long long int seed);
    virtual long long int getSeed();

    virtual void setMutationType(uint mutationType);
    virtual uint getMutationType();

    virtual void setSelectionType(uint selectionType);
    virtual uint getSelectionType();

    virtual void setCrossoverType(uint crossoverType);
    virtual uint getCrossoverType();

    virtual void setRandomsFactor(uint randomsFactor);
    virtual uint getRandomsFactor();

    virtual void setMutationsCount(uint count);
    virtual void setRangesCount(uint count);
    virtual void setRandomsRange(floatt  min, floatt  max, uint index);

    virtual uint getMutationsCount();
    virtual uint getRangesCount();
    virtual void getRandomsRange(floatt & min, floatt & max, uint index);

    
    void copy(Parameters& parameters);

private:
    uint threadsCount;
    long long int seed;
    uint randomsFactor;
    uint mutationType;
    uint selectionType;
    uint crossoverType;
    uint mutationsCount;
    uint rangesCount;
    floatt * ranges;
    synchronization::Mutex mutex;

};

#endif	/* PARAMETERSINTERFACE_H */

