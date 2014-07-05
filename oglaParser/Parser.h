/* 
 * File:   Parser.h
 * Author: mmatula
 *
 * Created on March 16, 2014, 3:59 PM
 */

#ifndef PARSER_H
#define	PARSER_H

#include <vector>
#include <string>
#include <map>

#include "Brackets.h"
#include "Function.h"
#include "MathStructure.h"
#include "Operator.h"
#include "Code.h"

class Parser {
public:

    Parser();
    Parser(const Parser& orig);
    virtual ~Parser();
private:
    typedef std::vector<MathStructure> MathStructures;
    std::vector<Operator*> operators;
    std::vector<MathStructure::Allocator*> mathStructuresCreators;
    std::vector<Function*> functions;
    std::vector<Brackets*> brackets;
    std::map<std::string, MathStructure*> variables;
    std::map<void*, std::string> codes;

    void setVariableValue(const std::string& name, MathStructure* complex);

    void setVariable(const std::string& name);

    void removeVariable(const std::string& name);

    void addOperator(Operator* operatorImpl);

    void removeOperator(Operator* operatorImpl);

    void addBracket(Brackets* bracket);

    void removeBracket(Brackets* bracket);

    void addMathStructureCreator(MathStructure::Allocator* allocator);
    void removeMathStructureCreator(MathStructure::Allocator* creator);

    void addFunction(Function* function);

    void removeFunction(Function* function);

    int findClosingBracket(Brackets& bracket, const std::string& equation, int begin);

    template<typename T> void insert(std::vector<T> objs, int index, T obj) {
        
    }

    void parse(Code& code);

    int findOperators(Code& code, int index);

    MathStructure* getParam(Code& code, int index);

    void executeOperators(Code& code, int index);

    void executeFunctions(Code& code, int index);

    void executeBracket(Code& code, int index);

    void executeMathStructure(Code& code, int index);

    bool parseComplex(const std::string& equation, int fa, Code& code, int na);

    bool parseVariable(const std::string& equation, int fa, Code& code, int na);

    bool parseBrackets(const std::string& equation, int fa, Code& code, int na);

    bool parseMathStructure(const std::string& equation, int fa, Code& code, int n);

    bool parseOperators(const std::string& equation, int fa, Code& code, int na);

    bool parseFunctions(const std::string& equation, int fa, Code& code, int na);

    void execute(MathStructures& mathStructures);

    void execute1(Code& code, MathStructures& mathStructures);

    void* parse(const std::string& equation);

    void calculate(const std::string& equation, MathStructures& mathStructures);

    int examineOperators(OperatorsItem* operatorsItem,
            MathStructure* param1, MathStructure* param2);
};

#endif	/* PARSER_H */

