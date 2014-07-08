#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=f95
AS=as

# Macros
CND_PLATFORM=GNU-Linux-x86
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/_ext/1855438775/IAdditionImpl.o \
	${OBJECTDIR}/_ext/1855438775/IDotProductImpl.o \
	${OBJECTDIR}/_ext/1855438775/IExpImpl.o \
	${OBJECTDIR}/_ext/1855438775/IMagnitudeImpl.o \
	${OBJECTDIR}/_ext/1855438775/IMultiplicationConstImpl.o \
	${OBJECTDIR}/_ext/1855438775/ISimpleDiagonalizationImpl.o \
	${OBJECTDIR}/_ext/1855438775/ISubstractionImpl.o \
	${OBJECTDIR}/_ext/1855438775/ITensorProductImpl.o \
	${OBJECTDIR}/_ext/1855438775/ITransposeImpl.o \
	${OBJECTDIR}/_ext/1855438775/MathOperations.o \
	${OBJECTDIR}/_ext/1855438775/Matrix.o \
	${OBJECTDIR}/_ext/1855438775/MatrixModules.o \
	${OBJECTDIR}/IDeterminant.o \
	${OBJECTDIR}/IIraMethod.o \
	${OBJECTDIR}/IQRDecomposition.o \
	${OBJECTDIR}/MatrixStructureUtils.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-Wl,-rpath,../oglaUtils/dist/Release/GNU-Linux-x86 -L../oglaUtils/dist/Release/GNU-Linux-x86 -loglaUtils -Wl,-rpath,../oglaMath/dist/Release/GNU-Linux-x86 -L../oglaMath/dist/Release/GNU-Linux-x86 -loglaMath

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrix.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrix.${CND_DLIB_EXT}: ../oglaUtils/dist/Release/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrix.${CND_DLIB_EXT}: ../oglaMath/dist/Release/GNU-Linux-x86/liboglaMath.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrix.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrix.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/1855438775/IAdditionImpl.o: /home/mmatula/Ogla/oglaMatrix/IAdditionImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/IAdditionImpl.o /home/mmatula/Ogla/oglaMatrix/IAdditionImpl.cpp

${OBJECTDIR}/_ext/1855438775/IDotProductImpl.o: /home/mmatula/Ogla/oglaMatrix/IDotProductImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/IDotProductImpl.o /home/mmatula/Ogla/oglaMatrix/IDotProductImpl.cpp

${OBJECTDIR}/_ext/1855438775/IExpImpl.o: /home/mmatula/Ogla/oglaMatrix/IExpImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/IExpImpl.o /home/mmatula/Ogla/oglaMatrix/IExpImpl.cpp

${OBJECTDIR}/_ext/1855438775/IMagnitudeImpl.o: /home/mmatula/Ogla/oglaMatrix/IMagnitudeImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/IMagnitudeImpl.o /home/mmatula/Ogla/oglaMatrix/IMagnitudeImpl.cpp

${OBJECTDIR}/_ext/1855438775/IMultiplicationConstImpl.o: /home/mmatula/Ogla/oglaMatrix/IMultiplicationConstImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/IMultiplicationConstImpl.o /home/mmatula/Ogla/oglaMatrix/IMultiplicationConstImpl.cpp

${OBJECTDIR}/_ext/1855438775/ISimpleDiagonalizationImpl.o: /home/mmatula/Ogla/oglaMatrix/ISimpleDiagonalizationImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/ISimpleDiagonalizationImpl.o /home/mmatula/Ogla/oglaMatrix/ISimpleDiagonalizationImpl.cpp

${OBJECTDIR}/_ext/1855438775/ISubstractionImpl.o: /home/mmatula/Ogla/oglaMatrix/ISubstractionImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/ISubstractionImpl.o /home/mmatula/Ogla/oglaMatrix/ISubstractionImpl.cpp

${OBJECTDIR}/_ext/1855438775/ITensorProductImpl.o: /home/mmatula/Ogla/oglaMatrix/ITensorProductImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/ITensorProductImpl.o /home/mmatula/Ogla/oglaMatrix/ITensorProductImpl.cpp

${OBJECTDIR}/_ext/1855438775/ITransposeImpl.o: /home/mmatula/Ogla/oglaMatrix/ITransposeImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/ITransposeImpl.o /home/mmatula/Ogla/oglaMatrix/ITransposeImpl.cpp

${OBJECTDIR}/_ext/1855438775/MathOperations.o: /home/mmatula/Ogla/oglaMatrix/MathOperations.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/MathOperations.o /home/mmatula/Ogla/oglaMatrix/MathOperations.cpp

${OBJECTDIR}/_ext/1855438775/Matrix.o: /home/mmatula/Ogla/oglaMatrix/Matrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/Matrix.o /home/mmatula/Ogla/oglaMatrix/Matrix.cpp

${OBJECTDIR}/_ext/1855438775/MatrixModules.o: /home/mmatula/Ogla/oglaMatrix/MatrixModules.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1855438775
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1855438775/MatrixModules.o /home/mmatula/Ogla/oglaMatrix/MatrixModules.cpp

${OBJECTDIR}/IDeterminant.o: IDeterminant.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/IDeterminant.o IDeterminant.cpp

${OBJECTDIR}/IIraMethod.o: IIraMethod.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/IIraMethod.o IIraMethod.cpp

${OBJECTDIR}/IQRDecomposition.o: IQRDecomposition.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/IQRDecomposition.o IQRDecomposition.cpp

${OBJECTDIR}/MatrixStructureUtils.o: MatrixStructureUtils.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/MatrixStructureUtils.o MatrixStructureUtils.cpp

# Subprojects
.build-subprojects:
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Release
	cd ../oglaMath && ${MAKE}  -f Makefile CONF=Release

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrix.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Release clean
	cd ../oglaMath && ${MAKE}  -f Makefile CONF=Release clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
