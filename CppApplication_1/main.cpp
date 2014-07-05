#include <stdlib.h>
#include <stdio.h>
#include <string.h>

class B {
public:
    int b;
};

class C {
public:
    int c;
};

class A : public B, public C {
public:
    A():d(new int()){}
    ~A(){delete d;}
    int a;
    const int*d;
};


int main() {

    A* a = new A();
    B* b = a;
    C* c = a;
    //printf("p == %p, %p, %p",&a->a,&a->b,&a->c);
    printf("p == %d \n",sizeof(int));

}