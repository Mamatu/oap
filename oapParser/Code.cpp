/* 
 * File:   Code.cpp
 * Author: mmatula
 * 
 * Created on April 12, 2014, 11:58 PM
 */

#include "Code.h"
#include "Types.h"

Code::Code(const char* string) {
    this->setString(string);
}

Code::Code(const Code& orig) {
}

Code::~Code() {
}

void Code::setString(const char* string) {
    this->items.push_back(Item(new EquationItem(string), EQUATION_ITEM));
}

Code::Item Code::get(int fa) const {
    if (this->items.size() < fa) {
        debug("Index %d is higher than size of code types buffor: %luu \n", fa, items.size());
        return Item();
    }
    return this->items[fa];
}

int Code::size() const {
}

