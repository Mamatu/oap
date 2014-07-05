
#include "Functions.h"
#include "GAProcessCPU.h"
#include "GAModuleCPU.h"

ParametersDeliver::ParametersDeliver() : parameters(NULL) {
}

ParametersDeliver::~ParametersDeliver() {
}

void ParametersDeliver::setParameters(Parameters* parameters) {
    this->parameters = parameters;
}

Parameters& ParametersDeliver::getParameters() const {
    return *this->parameters;
}

utils::ArgumentType uintArgs[] = {utils::ARGUMENT_TYPE_INT};
utils::ArgumentType longArgs[] = {utils::ARGUMENT_TYPE_LONG};
utils::ArgumentType llArgs[] = {utils::ARGUMENT_TYPE_LONG};
utils::ArgumentType doublesArgs[] = {utils::ARGUMENT_TYPE_INT, utils::ARGUMENT_TYPE_ARRAY_DOUBLES};

SetMutationType::SetMutationType() : utils::OglaFunction("setMutationType", uintArgs, 1) {
}

void SetMutationType::invoked(utils::Reader& reader, __attribute__((unused)) utils::Putter* writer) {
    uint type = reader.readInt();
    this->getParameters().setMutationType(type);
}

BoundaryMutation::SetMutationsCount::SetMutationsCount() : utils::OglaFunction("setMutationStep", uintArgs, 1), steps(1) {
}

void BoundaryMutation::SetMutationsCount::invoked(utils::Reader& reader, __attribute__((unused)) utils::Putter* writer) {
    steps = reader.readInt();
}

BoundaryMutation::SetRandomsRanges::SetRandomsRanges() : utils::OglaFunction("setRandomsRange", doublesArgs, 2), bounds(NULL), count(0) {
}

BoundaryMutation::SetRandomsRanges::~SetRandomsRanges() {
    if (bounds) {
        delete[] bounds;
    }
    this->count = 0;
}

void BoundaryMutation::SetRandomsRanges::invoked(utils::Reader& reader, __attribute__((unused)) utils::Putter* writer) {
    if (this->bounds) {
        delete[] this->bounds;
    }
    this->count = reader.readInt();
    this->bounds = new floatt [count * 2];
    this->getParameters().setRangesCount(this->count);
    for (uint fa = 0; fa < count; fa++) {
        floatt  min = (floatt ) reader.getDouble();
        floatt  max = (floatt ) reader.getDouble();
        this->getParameters().setRandomsRange(min, max, fa);
    }
}

BoundaryMutation::BoundaryMutation() : utils::OglaObject("boundaryMutation") {
    functionsContainer.add(&(this->setMutationsCount));
    functionsContainer.add(&(this->setRandomsRange));
    this->setFunctionsContainer(&functionsContainer);
}

void BoundaryMutation::setParameters(Parameters* parameters) {
    ParametersDeliver::setParameters(parameters);
    setMutationsCount.setParameters(parameters);
    setRandomsRange.setParameters(parameters);
}

SetCrossoverType::SetCrossoverType() : utils::OglaFunction("setCrossoverType", uintArgs, 1), gcc(NULL) {
}

void SetCrossoverType::invoked(utils::Reader& reader, utils::Putter* writer) {
    uint type = reader.readInt();
    this->getParameters().setCrossoverType(type);
}

SetSelectionType::SetSelectionType() : utils::OglaFunction("setSelectionType", uintArgs, 1), gcc(NULL) {
}

void SetSelectionType::invoked(utils::Reader& reader, utils::Putter* writer) {
    uint type = reader.readInt();
    this->getParameters().setSelectionType(type);
}

SetThreadsCount::SetThreadsCount() : utils::OglaFunction("setThreadsCount", uintArgs, 1) {
}

void SetThreadsCount::invoked(utils::Reader& reader, utils::Putter* writer) {
    uint threadsNumber = reader.readInt();
    this->getParameters().setThreadsCount(threadsNumber);
}

SetSeed::SetSeed() : utils::OglaFunction("setSeed", llArgs, 1) {
}

void SetSeed::invoked(utils::Reader& reader, utils::Putter* writer) {
    long long int seed = reader.getLong();
    this->getParameters().setSeed(seed);
}

SetRandomsFactor::SetRandomsFactor() : utils::OglaFunction("setRandomsFactor", uintArgs, 1) {
}

void SetRandomsFactor::invoked(utils::Reader& reader, utils::Putter* writer) {
    uint randomsFactor = reader.readInt();
    this->getParameters().setRandomsFactor(randomsFactor);
}