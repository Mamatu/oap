/* 
 * File:   Operator.h
 * Author: mmatula
 *
 * Created on March 16, 2014, 4:14 PM
 */

#ifndef OGLA_OPERATOR_H
#define	OGLA_OPERATOR_H

#include "MatrixStructure.h"

class Operator {
protected:
    Operator(const Operator& orig);
public:
    Operator();
    virtual ~Operator();
    virtual int getWeight() = 0;
    virtual char getSymbol() = 0;
    virtual bool setParams(const MatrixStructure* param1, const MatrixStructure* param2) = 0;
    virtual bool execute(MatrixStructure** mathStructure) = 0;
};

#endif	/* OPERATOR_H */

