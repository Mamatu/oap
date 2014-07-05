
#ifndef FUNCTIONS_H
#define	FUNCTIONS_H

#include "WrapperInterfaces.h"
#include "Parameters.h"
namespace ga {
    class GARatingExecutorCreatorCPU;
    class GAProcessCPU;
    class GAProcessConfiguratorCPU;
}

class ParametersDeliver {
    Parameters* parameters;
public:
    ParametersDeliver();
    virtual ~ParametersDeliver();
    virtual void setParameters(Parameters* parameters);
    Parameters& getParameters() const;
};

class SetMutationType : public utils::OglaFunction, public ParametersDeliver {
public:
    SetMutationType();

protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class BoundaryMutation : public utils::OglaObject, public ParametersDeliver {

    class SetMutationsCount : public utils::OglaFunction, public ParametersDeliver {
    public:
        uint steps;
        SetMutationsCount();
    protected:
        void invoked(utils::Reader& reader, utils::Putter* writer);
    };

    class SetRandomsRanges : public utils::OglaFunction, public ParametersDeliver {
    public:
        floatt * bounds;
        uint count;
        SetRandomsRanges();
        ~SetRandomsRanges();
    protected:
        void invoked(utils::Reader& reader, utils::Putter* writer);
    };

    SetMutationsCount setMutationsCount;
    SetRandomsRanges setRandomsRange;
    utils::FunctionsContainer functionsContainer;
public:
    BoundaryMutation();
    void setParameters(Parameters* parameters);
};

class SetCrossoverType : public utils::OglaFunction, public ParametersDeliver {
    ga::GARatingExecutorCreatorCPU* gcc;
public:
    SetCrossoverType();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetSelectionType : public utils::OglaFunction, public ParametersDeliver {
    ga::GARatingExecutorCreatorCPU* gcc;
public:
    SetSelectionType();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetThreadsCount : public utils::OglaFunction, public ParametersDeliver {
    ga::GAProcessConfiguratorCPU* gcc;
public:
    SetThreadsCount();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetSeed : public utils::OglaFunction, public ParametersDeliver {
    ga::GAProcessConfiguratorCPU* ggc;
public:
    SetSeed();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetRandomsFactor : public utils::OglaFunction, public ParametersDeliver {
    ga::GAProcessConfiguratorCPU* ggc;
public:
    SetRandomsFactor();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

/*
class SetRandomNumbers : public utils::Function {
    ga::cpu::GeneticComponentImpl* gpi;
public:
    SetRandomNumbers();
    void init(ga::cpu::GeneticComponentImpl* gci);
protected:
    void invoked(utils::Reader& reader, utils::Setter* writer);
};

class OpenRandomNumbersFile : public utils::Function {
    ga::cpu::GeneticComponentImpl* gpi;
public:
    OpenRandomNumbersFile(ga::cpu::GeneticComponentImpl* _gpi);
protected:
    void invoked(utils::Reader& reader, utils::Setter* writer);

};*/



#endif

