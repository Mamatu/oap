/* 
 * File:   Variable.h
 * Author: mmatula
 *
 * Created on May 4, 2014, 7:19 PM
 */

#ifndef VARIABLE_H
#define	VARIABLE_H


#include "MathStructure.h"
#include <string>

class Variable {
public:

    std::string name;
    MathStructure* mathStructure;
    std::string path;

    Variable() : name(""), mathStructure(NULL), path("") {
    }

    Variable(const std::string& _name, MathStructure* _mathStructure) :
    name(_name), mathStructure(_mathStructure), path("") {
    }

    Variable(const std::string& _name, const std::string& _path) :
    name(_name), mathStructure(NULL), path(_path) {
    }
};

#endif	/* VARIABLE_H */

