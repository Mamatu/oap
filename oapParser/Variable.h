/* 
 * File:   Variable.h
 * Author: mmatula
 *
 * Created on May 4, 2014, 7:19 PM
 */

#ifndef VARIABLE_H
#define	VARIABLE_H


#include "MatrixStructure.h"
#include <string>

class Variable {
public:

    std::string name;
    MatrixStructure* mathStructure;
    std::string path;

    Variable() : name(""), mathStructure(NULL), path("") {
    }

    Variable(const std::string& _name, MatrixStructure* _mathStructure) :
    name(_name), mathStructure(_mathStructure), path("") {
    }

    Variable(const std::string& _name, const std::string& _path) :
    name(_name), mathStructure(NULL), path(_path) {
    }
};

#endif	/* VARIABLE_H */

