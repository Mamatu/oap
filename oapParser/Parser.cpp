/* 
 * File:   Parser.cpp
 * Author: mmatula
 * 
 * Created on March 16, 2014, 3:59 PM
 */

#include "Parser.h"
#include <algorithm>
#include "Types.h"

#define REMOVE(type, vec, item)\
type::iterator it = std::find(vec.begin(),vec.end(),item);\
if(it != vec.end()){vec.erase(it);}\


Parser::Parser() {
}

Parser::Parser(const Parser& orig) {
}

Parser::~Parser() {
}

void Parser::setVariableValue(const std::string& name, MatrixStructureUtils* complex) {
    this->variables[name] = complex;
}

void Parser::setVariable(const std::string& name) {
    this->variables[name] = NULL;
}

void Parser::removeVariable(const std::string& name) {
    std::map<std::string, MatrixStructureUtils*>::iterator it = this->variables.find(name);
    this->variables.erase(it);
}

void Parser::addOperator(Operator* operatorImpl) {
    this->operators.push_back(operatorImpl);
}

void Parser::removeOperator(Operator* operatorImpl) {
    REMOVE(std::vector<Operator*>, this->operators, operatorImpl);
}

void Parser::addBracket(Brackets* bracket) {
    if (bracket->getLeftSymbol() == bracket->getRightSymbol()) {
        debug("Invalid bracket declration. Left symbol can't be equal to right.");
        return;
    }
    this->brackets.push_back(bracket);
}

void Parser::removeBracket(Brackets* bracket) {
    REMOVE(std::vector<Brackets*>, this->brackets, bracket);
}

void Parser::addFunction(Function* function) {
    this->functions.push_back(function);
}

void Parser::removeFunction(Function* function) {
    REMOVE(std::vector<Function*>, this->functions, function);
}

int Parser::findClosingBracket(Brackets& bracket, const std::string& equation, int begin) {
    int index = -1;
    int stackCounter = 0;
    for (int fa = begin; fa < equation.length(); fa++) {
        if (equation[fa] == bracket.getLeftSymbol()) {
            stackCounter++;
        }
        if (equation[fa] == bracket.getRightSymbol()) {
            stackCounter--;
            if (stackCounter == 0) {
                index = fa;
                break;
            }
        }
    }
    if (index == -1) {
    }
    return index;
}

int Parser::findOperators(Code& code, int index) {
}

void Parser::parse(Code& code) {
    int na = 0;
    do {
        Code::Item obj = code.get(na);
        if (obj.getType() == Code::EQUATION_ITEM) {
            EquationItem* equationItem = reinterpret_cast<EquationItem*> (obj.getGenericPtr());
            std::string equation = equationItem->string;
            //equation = equation.replaceAll("\\s+", "");
            int fa = 0;
            bool parsed = true;
            do {
                parsed = true;
                if (this->parseComplex(equation, fa, code, na) == false) {
                    if (this->parseVariable(equation, fa, code, na) == false) {
                        if (this->parseBrackets(equation, fa, code, na) == false) {
                            if (this->parseMatrixStructure(equation, fa, code, na) == false) {
                                if (this->parseOperators(equation, fa, code, na) == false) {
                                    if (this->parseFunctions(equation, fa, code, na) == false) {
                                        parsed = false;
                                        fa++;
                                    }
                                }
                            }
                        }
                    }
                }
            } while (parsed == false);
            if (fa > 0) {
                code.insert(na, new std::string(equation.substr(0, fa)));
            }
            na = 0;
        } else {
            na++;
        }
    } while (na != code.size());
}

MatrixStructure* Parser::getParam(Code& code, int index) {
    if (index < 0 || index > code.size()) {
        return NULL;
    }
    MatrixStructure* param1 = NULL;
    bool pb1 = (index >= 0);
    if (pb1) {
        Code::Item objParam1 = code.get(index);
        if (objParam1.getType() == Code::BRACKET_ITEM) {
            this->executeBracket(code, index);
        } else if (objParam1.getType() == Code::FUNCTION_ITEM) {
            this->executeFunctions(code, index);
        }
        objParam1 = code.get(index);
        if (objParam1.getType() == Code::MATRIX_STRUCTURE_ITEM) {
            param1 = reinterpret_cast<MatrixStructure*> (objParam1.getGenericPtr());
        } else if (objParam1.getType() == Code::VARIABLE_ITEM) {
            Variable* variableItem = reinterpret_cast<Variable*> (objParam1.getGenericPtr());
            Variable mathStructure = variableItem->mathStructure;
            Variable.updateVariable(v, variables);
            param1 = v.mathStructure;
        }
    }
    return param1;
}

int Parser::examineOperators(OperatorsItem* operatorsItem,
        MatrixStructure* param1, MatrixStructure* param2) {
    int index = -1;
    bool invalid = false;
    for (int fa = 0; fa < operatorsItem->operators.size(); fa++) {
        Operator* operatorPtr = operatorsItem.get(fa).operator;
        if (operatorPtr->setParams(param1, param2) == true) {
            if (index != -1) {
                invalid = true;
            }
            index = fa;
        }
    }
    return index;
}

void Parser::executeOperators(Code& code, int index) {
    Code::Item object = code.get(index);
    OperatorsItem* operatorsItem = reinterpret_cast<OperatorsItem*> (object.getGenericPtr());
    int nindex = this->findOperators(code, index + 1);
    if (nindex > -1) {
        OperatorsItem* nextOperatorsItem = reinterpret_cast<OperatorsItem*> (code.get(nindex).getGenericPtr());
        if (nextOperatorsItem->isStronger(operatorsItem)) {
            this->executeOperators(code, nindex);
        }
    }
    MatrixStructure* param1 = getParam(code, index - 1);
    MatrixStructure* param2 = getParam(code, index + 1);
    MatrixStructure* cparam1 = NULL;
    if (param1 != NULL) {
        cparam1 = param1.createCopy();
    }
    MatrixStructure cparam2 = NULL;
    if (param2 != NULL) {
        cparam2 = param2.createCopy();
    }
    Integer[] operatorsIndieces = this->examineOperators(operators(std::vector, cparam1, cparam2);
            Operator operatorImpl = this->getMostImportantOperator(operators(std::vector, operatorsIndieces);
    if (operatorImpl.setParams(cparam1, cparam2)) {
        MatrixStructure result = operatorImpl.execute();
        if (cparam1 != NULL) {
            code.set(index - 1, new MatrixStructure(result));
                    code.remove(index);
            if (cparam2 != NULL) {
                code.remove(index);
            }
        } else {
            code.set(index, new MatrixStructure(result));
            if (cparam2 != NULL) {

                code.remove(index + 1);
            }
        }
    }
}

void Parser::executeFunctions(Code& code, int index) {
    if (code.size() - 1 == index) {
    }
    FunctionItem.std::vector functions(std::vector = (FunctionItem.std::vector) code.get(index);
    if (code.get(index + 1).getType() == Code.Type.BRACKET_ITEM) {
        Code& codeInBrackets = ((BracketsItem) code.get(index + 1)).getCode();
                std::vector<MatrixStructure> params = new Arraystd::vector<MatrixStructure>();
                bool mustBeRawCodeFunction = false;
        try {
            MatrixStructure[] ms = this->execute1(codeInBrackets);
                    params.addAll(Arrays.as(std::vector(ms));
        } catch (Exception ex) {
            mustBeRawCodeFunction = true;
        }

        for (int fa = 0; fa < functions(std::vector.size(); fa++) {
                bool canBeContinued = false;
                        bool hasRawCodeParams = functions(std::vector.get(fa).function.rawCodeAsParams();
                if ((mustBeRawCodeFunction == true && hasRawCodeParams == true) || mustBeRawCodeFunction == false) {
                    canBeContinued = true;
                }
                if (hasRawCodeParams == false) {

                    if (functions(std::vector.get(fa).function.getParamsCount() == params.size()) {

                            if (functions(std::vector.get(fa).function.setParams(params, NULL) == true) {
                                    MatrixStructure object = functions(std::vector.get(fa).function.execute();
                                    code.set(index, new MatrixStructure(object));
                                    code.remove(index + 1);
                                    break;
                                }
                        }
                } else {
                    std::vector<ParamItem> paramsItems = new Arraystd::vector<ParamItem>();
                            Code tempCode = new Code();
                    for (int fb = 0; fb < codeInBrackets.size(); fb++) {
                        if (codeInBrackets.get(fb).getType() == Code.Type.OPERATOR_ITEMS) {
                            OperatorItem.std::vector operatorItems = (OperatorItem.std::vector) codeInBrackets.get(fb);
                            if (operatorItems.size() == 1) {
                                if (operatorItems.get(0).operatorImpl instanceof OperatorsImpls.Seprator) {
                                    paramsItems.add(new ParamItem(tempCode));
                                            tempCode.clear();
                                } else {
                                    tempCode.add(codeInBrackets.get(fb));
                                }
                            } else {
                                tempCode.add(codeInBrackets.get(fb));
                            }
                        } else {
                            tempCode.add(codeInBrackets.get(fb));
                        }
                    }
                    paramsItems.add(new ParamItem(tempCode));
                            tempCode.clear();

                    if (functions(std::vector.get(fa).function.setParams(paramsItems, this) == true) {
                            MatrixStructure object = functions(std::vector.get(fa).function.execute();
                            code.set(index, new MatrixStructure(object));
                            code.remove(index + 1);
                            break;
                        } else {
                    }
                }
            }
    }
}

void Parser::executeBracket(Code& code, int index) {
    BracketsItem bracketItem = (BracketsItem) code.get(index);
            Code subCode = bracketItem.getCode();
            MatrixStructure[] mathStructures = this->execute1(subCode);
            code.set(index, new MatrixStructure(mathStructures[0]));
    for (int fa = 1; fa < mathStructures.length; fa++) {

        code.insert(index + fa, new MatrixStructure(mathStructures[fa]));
    }
}

void Parser::executeMatrixStructure(Code& code, int index) {
    std::vector<MatrixStructure> mathStructures = new Arraystd::vector<MatrixStructure>();
            mathStructures.clear();
            MatrixStructureCreatorItem.std::vector creatorItems = (MatrixStructureCreatorItem.std::vector) code.get(index);
            MatrixStructure.Creator creator = creatorItems.get(0).creator;
            Code objs = creatorItems.code;
            MatrixStructure[] mathStructuresArray = this->execute1(objs);
            MatrixStructure mathStructure1 = NULL;
            mathStructures.addAll(Arrays.as(std::vector(mathStructuresArray));
    if (creator.setParams(mathStructures) == true) {

        mathStructure1 = creator.create();
    }
    code.remove(index);
            code.set(index, new MatrixStructure(mathStructure1));
}

bool Parser::parseComplex(const std::string& equation, int fa, Code& code, int na) {
    Complex complex = NULL;
            Complex complex1 = NULL;
            int complexEndIndex = 0;
    for (int fb = fa + 1; fb <= equation.length() + 1 && (complex1 != NULL || fb == fa + 1); fb++) {
        complex = complex1;
        if (fb <= equation.length()) {
            const std::string& substring = equation.substring(fa, fb);
                    complex1 = Complex.parseComplex(substring);
        }
        complexEndIndex = fb;
    }
    if (complex != NULL) {
        code.set(na, new MatrixStructure(complex));
                codes.put(complex, complex.toconst std::string & ());
        if (complexEndIndex < equation.length()) {

            complexEndIndex--;
                    const std::string& part2 = equation.substring(complexEndIndex, equation.length());
                    code.insert(na + 1, new EquationItem(part2));
                    equation = part2;
        }
    }
    return (complex != NULL);
}

bool Parser::parseVariable(const std::string& equation, int fa, Code& code, int na) {
    const std::string& variableName = NULL;
            const std::string& variableName1 = NULL;
            int variableEndIndex = 0;
    for (int fb = fa + 1; fb <= equation.length() + 1 && (variableName1 != NULL || fb == fa + 1); fb++) {
        variableName = variableName1;
        if (fb <= equation.length()) {
            const std::string& substring = equation.substring(fa, fb);
            if (this->variables.containsKey(substring)) {
                variableName1 = substring;
            } else {
                variableName1 = NULL;
            }
        }
        variableEndIndex = fb;
    }
    if (variableName.length() != 0) {
        Variable variable = new Variable(variableName);
                code.set(na, new Variable(variable));
                codes.put(variable, variable.toconst std::string & ());
        if (variableEndIndex < equation.length()) {

            variableEndIndex--;
                    const std::string& part2 = equation.substring(variableEndIndex, equation.length());
                    code.insert(na + 1, new EquationItem(part2));
                    equation = part2;
        }
    }
    return (variableName.length() != 0);
}

bool Parser::parseBrackets(const std::string& equation, int fa, Code& code, int na) {
    for (size_t fa = 0; fa< this->brackets.size(); ++fa) {
        Brackets* bracket = this->brackets[fa];
                char symbol = equation.at(fa);
        if (symbol == bracket.getLeftSymbol()) {
            int index1 = findClosingBracket(bracket, equation, fa);
                    const std::string& part2 = equation.substr(fa + 1, index1);
                    const std::string& part3 = equation.substr(index1 + 1, equation.length());
                    //objs.set(na, bracket);
                    Code subCode = new Code();
                    subCode.add(new EquationItem(part2));
                    this->parse(subCode);
                    codes.put(subCode, part2);
                    code.set(na, new BracketsItem(bracket, subCode));
            if (part3.length() > 0) {
                code.insert(na + 1, new EquationItem(part3));
            }
            fa = 0;

            return true;
        }
    }
    return false;
}

bool Parser::parseMatrixStructure(const std::string& equation, int fa, Code& code, int na) {
    bool status = false;
            bool isFirst = true;
            MatrixStructureCreatorItem.List creatorList = null;
    for (int f = 0; f<this->mathStructuresUtils.size(); f++) {
        MatrixStructure::Allocator* mathStructureCreator = this->mathStructuresUtils[f];
                int fa1 = fa;
                char symbol = equation[fa1];
        if (isFirst == true) {
            if (symbol == mathStructureCreator.getBoundary().getLeftSymbol()) {
                int index1 = this.findClosingBracket(mathStructureCreator.getBoundary(), equation, fa1);
                        String part2 = equation.substring(fa1 + 1, index1);
                        String part3 = equation.substring(index1 + 1, equation.length());
                        Code subCode = new Code();
                        subCode.add(new EquationItem(part2));
                        this.parse(subCode);
                        MatrixStructureCreatorItem.List creatorList1 = new MatrixStructureCreatorItem.List(subCode);
                        creatorList1.add(new MatrixStructureCreatorItem(mathStructureCreator));
                        creatorList = creatorList1;
                        code.set(na, creatorList1);
                if (part3.length() > 0) {
                    code.insert(na + 1, new EquationItem(part3));
                }
                isFirst = false;
                        status = true;
            }
        } else {
            if (creatorList.get(0).creator.getBoundary().getLeftSymbol() == mathStructureCreator.getBoundary().getLeftSymbol()
                    && creatorList.get(0).creator.getBoundary().getRightSymbol() == mathStructureCreator.getBoundary().getRightSymbol()) {

                creatorList.add(new MatrixStructureCreatorItem(mathStructureCreator));
            }
        }
    }
    return status;
}

bool Parser::parseOperators(const std::string& equation, int fa, Code& code, int na) {
    for (Operator operatorImpl : operators) {
        char symbol = equation.at(fa);
        if (symbol == operatorImpl.getSymbol()) {

            if (operators.size() != 0) {
                operators(std::vector = new OperatorItem.std::vector();
            }
            operators(std::vector.add(new OperatorItem(operatorImpl));
        }
    }

    if (operators.size() != 0) {

        const std::string& sign = equation.substring(fa, fa + 1);
                const std::string& part2 = equation.substring(fa + 1, equation.length());
                code.set(na, operators(std::vector);
                code.insert(na + 1, new EquationItem(part2));
                equation = part2;
                codes.put(operators(std::vector, sign);
    }
    return (operators(std::vector != NULL);
}

bool Parser::parseFunctions(const std::string& equation, int fa, Code& code, int na) {
    FunctionItem.std::vector functions(std::vector = NULL;
    for (Function function : functions) {
        if (fa + function.getName().length() < equation.length()) {
            const std::string& comp = equation.substring(fa, fa + function.getName().length());
            if (comp.equals(function.getName())) {

                if (functions(std::vector == NULL) {
                        functions(std::vector = new FunctionItem.std::vector();
                    }
                functions(std::vector.add(new FunctionItem(function));
            }
        }
    }

    if (functions(std::vector != NULL) {
            const std::string& functionName = equation.substring(fa, fa + functions(std::vector.get(0).function.getName().length());
            const std::string& part2 = equation.substring(fa + functions(std::vector.get(0).function.getName().length(), equation.length());
            code.set(na, functions(std::vector);
            code.insert(na + 1, new EquationItem(part2));
            codes.put(functions(std::vector, functionName);
        }
    return (functions(std::vector != NULL);
}

MatrixStructure* Parser::execute(Object codeObject) {
    Code& code1 = (Code) codeObject;
            Code& code = code1.createCopy();
            MatrixStructure[] ms = execute1(code);

    return ms;
}

MatrixStructure* Parser::execute1(Code& code, MatrixStructures& mathStructures) {
    try {
        bool next = true;
        while (next) {
            next = false;
                    int fa = 0;
            while (fa < code.size()) {
                Code.Item object = code.get(fa);
                if (object.getType() == Code.Type.STRING_ITEM) {
                    const std::string&Item stringItem = (const std::string & Item) object;
                    if (variables.containsKey(stringItem.string)) {
                        MatrixStructure mathStructure = variables.get(stringItem.string);
                        if (mathStructure != NULL) {
                            code.set(fa, new Variable(new Variable(stringItem.string)));
                        }
                    }
                } else if (object.getType() == Code.Type.OPERATOR_ITEMS) {
                    next = true;
                            this->executeOperators(code, fa);
                            fa = 0;
                } else if (object.getType() == Code.Type.FUNCTION_ITEMS) {
                    next = true;
                            this->executeFunctions(code, fa);
                            fa = 0;
                } else if (object.getType() == Code.Type.BRACKET_ITEM) {
                    next = true;
                            this->executeBracket(code, fa);
                            fa = 0;
                } else if (object.getType() == Code.Type.MATH_STRUCTURE_CREATOR_ITEMS) {
                    next = true;
                            this->executeMatrixStructure(code, fa);
                            fa = 0;
                }
                fa++;
            }
        }
        std::vector<MatrixStructure> mathStructures = new Arraystd::vector<MatrixStructure>();
        for (int fa = 0; fa < code.size(); fa++) {
            if (code.get(fa).getType() == Code.Type.VARIABLE_ITEM) {
                Variable variableItem = (Variable) code.get(fa);
                        Variable.updateVariable(variableItem.variable, variables);
                        mathStructures.add(variableItem.variable.mathStructure);
            }
            if (code.get(fa).getType() == Code.Type.MATH_STRUCTURE_ITEM) {
                MatrixStructure mathStructure = ((MatrixStructure) code.get(fa)).mathStructure;
                        mathStructures.add(mathStructure);
            }
        }
        MatrixStructure[] a = new MatrixStructure[mathStructures.size()];
        return mathStructures.toArray(a);
    } catch (SyntaxErrorException see) {

        throw see;
    }
}

MatrixStructure* Parser::parse(const std::string& equation) {
    Code& code = new Code();
            code.add(new EquationItem(equation));
            this->parse(code);

    return code;
}

MatrixStructure* Parser::calculate(const std::string& equation) {
    MatrixStructure* out = NULL;
            codes.clear();
            Code& code = new Code();
            code.add(new EquationItem(equation));
            this->parse(code);
            out = this->execute1(code);
    return out;
}