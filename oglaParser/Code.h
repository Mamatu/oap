/* 
 * File:   Code.h
 * Author: mmatula
 *
 * Created on April 12, 2014, 11:58 PM
 */

#ifndef CODE_H
#define	CODE_H

#include <vector>
#include <string>

#include "Function.h"
#include "Operator.h"
#include "Variable.h"
#include "MatrixStructure.h"

class Code {
public:

    enum Type {
        INVALID_ITEM,
        EQUATION_ITEM,
        BRACKET_ITEM,
        MATRIX_STRUCTURE_ITEM,
        FUNCTION_ITEM,
        FUNCTIONS_ITEM,
        OPERATOR_ITEM,
        OPERATORS_ITEM,
        VARIABLE_ITEM,
        STRING_ITEM,
        PARAM_ITEM
    };


    Code(const char* string);
    virtual ~Code();

    class Item {
        void* genericPtr;
        Code::Type type;
    public:

        Item() : genericPtr(NULL), type(INVALID_ITEM) {
        }

        Item(void* _genericPtr, Code::Type _type) :
        genericPtr(_genericPtr), type(_type) {
        }

        Code::Type getType() {
            return type;
        }

        void* getGenericPtr() {
            return genericPtr;
        }
    };

    Item get(int fa) const;
    int size() const;

    template<typename T> void insert(int index, T* obj) {
        items.insert(items.begin() + index, Item(obj, obj->getType()));
    }
private:
    Code(const Code& orig);
    void setString(const char* string);
    std::vector<Item> items;
};

#define GET_TYPE(type) Code::Type getType() const { return type;}

class EquationItem {
public:

    GET_TYPE(Code::EQUATION_ITEM);

    EquationItem(const std::string& _string) : string(_string) {
    }
    std::string string;
};

class BracketsItem {
public:

    GET_TYPE(Code::BRACKET_ITEM);
    Code code;
    std::string string;
};

class FunctionsItem {
public:

    GET_TYPE(Code::FUNCTIONS_ITEM);
    std::vector<Function*> functions;
};

class OperatorItem {
public:

    GET_TYPE(Code::OPERATOR_ITEM);
    Operator* operatorPtr;
};

class OperatorsItem {
public:

    GET_TYPE(Code::OPERATORS_ITEM);
    std::vector<OperatorItem> operators;

    bool isStronger(OperatorsItem* comp) {
        const int w1 = comp->operators[0].operatorPtr->getWeight();
        const int w2 = operators[0].operatorPtr->getWeight();
        return w2>w1;
    }
};

class VariableItem {
public:

    GET_TYPE(Code::VARIABLE_ITEM);
    Variable* variable;

    VariableItem(Variable* _variable = NULL) : variable(_variable) {
    }
};

class MatrixStructureItem {
public:
    MatrixStructure* matrixStructure;
};

#endif	/* CODE_H */

