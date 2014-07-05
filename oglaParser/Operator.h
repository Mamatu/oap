/* 
 * File:   Operator.h
 * Author: mmatula
 *
 * Created on March 16, 2014, 4:14 PM
 */

#ifndef OPERATOR_H
#define	OPERATOR_H

#include "MathStructure.h"

class Operator {
protected:
    Operator(const Operator& orig);
public:
    Operator();
    virtual ~Operator();
    virtual int getWeight() = 0;
    virtual char getSymbol() = 0;
    virtual bool setParams(const MathStructure* param1, const MathStructure* param2) = 0;
    virtual bool execute(MathStructure** mathStructure) = 0;
};

#endif	/* OPERATOR_H */

