/* 
 * File:   Brackets.h
 * Author: mmatula
 *
 * Created on March 16, 2014, 10:39 PM
 */

#ifndef BRACKETS_H
#define	BRACKETS_H

class Brackets {
public:
    Brackets();
    Brackets(const Brackets& orig);
    virtual ~Brackets();
    char getLeftSymbol();
    char getRightSymbol();
private:

};

#endif	/* BRACKETS_H */

