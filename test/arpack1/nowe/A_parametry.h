#ifndef paramerty_INCLUDE
#define paramerty_INCLUDE

//*********** pliki ***********
//*****F
#define plikFz "a.dat"
//*********** stale ***********
#define s1 0.5
#define s2 0.5
#define M 5    //Uwaga!! M=2*k
#define cn  16 //4096 //Uwaga!! nn1= pow((2*s1+1),2*M); -> nn1=pow((2*s1+1),M+1)
#define mgB 1.0//0.671713012
//************* J **************
#define jx1 1.0
#define jz1 1.0
#define jx2 1.0
#define jz2 1.0
//************ D **************
#define Dz1 0.0
#define Dz2 0.0
//************ g **************
#define gx1 1.0 
#define gz1 1.0
#define gx2 1.0
#define gz2 1.0
//************ T **************
#define stalaBg 0.0 

#define T0pod 0.1 //0.10
#define krTpod 0.01
#define iloscTpod 1000 //600 //30
#define ModeP "w"

//#define T0pod 4.0
//#define krTpod 0.50
//#define iloscTpod 93
//#define ModeP "a"

//#define T0pod 45.0
//#define krTpod 5.0
//#define iloscTpod 46
//#define ModeP "a"
//*****************************************************
#endif
