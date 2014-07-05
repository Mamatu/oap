
#ifndef FUNCTIONS_H
#define	FUNCTIONS_H

#include "WrapperInterfaces.h"
#include "Parameters.h"
namespace ga {
    class GARatingEntityCUDA;
    class GAProcessCUDA;
    class GAModuleInstanceCUDA;
}

class ParametersDeliver {
    Parameters* parameters;
public:
    ParametersDeliver();
    virtual ~ParametersDeliver();
    virtual void setParameters(Parameters* parameters);
    Parameters& getParameters() const;
};

class SetMutationType : public utils::Function, public ParametersDeliver {
public:
    SetMutationType();

protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class BoundaryMutation : public utils::Object, public ParametersDeliver {

    class SetMutationsCount : public utils::Function, public ParametersDeliver {
    public:
        uint steps;
        SetMutationsCount();
    protected:
        void invoked(utils::Reader& reader, utils::Putter* writer);
    };

    class SetRandomsRanges : public utils::Function, public ParametersDeliver {
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

class SetCrossoverType : public utils::Function, public ParametersDeliver {
    ga::GARatingEntityCUDA* gcc;
public:
    SetCrossoverType();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetSelectionType : public utils::Function, public ParametersDeliver {
    ga::GARatingEntityCUDA* gcc;
public:
    SetSelectionType();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetSeed : public utils::Function, public ParametersDeliver {
    ga::GAModuleInstanceCUDA* ggc;
public:
    SetSeed();
protected:
    void invoked(utils::Reader& reader, utils::Putter* writer);
};

class SetRandomsFactor : public utils::Function, public ParametersDeliver {
    ga::GAModuleInstanceCUDA* ggc;
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

