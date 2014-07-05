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
	${OBJECTDIR}/_ext/852485519/AdditionImpl.o \
	${OBJECTDIR}/_ext/852485519/DotProductImpl.o \
	${OBJECTDIR}/_ext/852485519/ExpImpl.o \
	${OBJECTDIR}/_ext/852485519/HostMatrixModules.o \
	${OBJECTDIR}/_ext/852485519/MagnitudeImpl.o \
	${OBJECTDIR}/_ext/852485519/MathOperationsCpu.o \
	${OBJECTDIR}/_ext/852485519/MultiplicationConstImpl.o \
	${OBJECTDIR}/_ext/852485519/SimpleDiagonalizationImpl.o \
	${OBJECTDIR}/_ext/852485519/SubstractionImpl.o \
	${OBJECTDIR}/_ext/852485519/TensorProductImpl.o \
	${OBJECTDIR}/_ext/852485519/TransposeImpl.o \
	${OBJECTDIR}/DeterminantImpl.o \
	${OBJECTDIR}/HostMatrixStructure.o \
	${OBJECTDIR}/IRAMOperationsImpl.o \
	${OBJECTDIR}/QRDecompositionImpl.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-ggdb3
CXXFLAGS=-ggdb3

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-Wl,-rpath,/home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86 -loglaUtils -Wl,-rpath,/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -loglaMath -lpthread -Wl,-rpath,/home/mmatula/Ogla/oglaMatrix/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrix/dist/Debug/GNU-Linux-x86 -loglaMatrix

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86/liboglaMath.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMatrix/dist/Debug/GNU-Linux-x86/liboglaMatrix.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/852485519/AdditionImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/AdditionImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/AdditionImpl.o /home/mmatula/Ogla/oglaMatrixCpu/AdditionImpl.cpp

${OBJECTDIR}/_ext/852485519/DotProductImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/DotProductImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/DotProductImpl.o /home/mmatula/Ogla/oglaMatrixCpu/DotProductImpl.cpp

${OBJECTDIR}/_ext/852485519/ExpImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/ExpImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/ExpImpl.o /home/mmatula/Ogla/oglaMatrixCpu/ExpImpl.cpp

${OBJECTDIR}/_ext/852485519/HostMatrixModules.o: /home/mmatula/Ogla/oglaMatrixCpu/HostMatrixModules.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/HostMatrixModules.o /home/mmatula/Ogla/oglaMatrixCpu/HostMatrixModules.cpp

${OBJECTDIR}/_ext/852485519/MagnitudeImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/MagnitudeImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/MagnitudeImpl.o /home/mmatula/Ogla/oglaMatrixCpu/MagnitudeImpl.cpp

${OBJECTDIR}/_ext/852485519/MathOperationsCpu.o: /home/mmatula/Ogla/oglaMatrixCpu/MathOperationsCpu.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/MathOperationsCpu.o /home/mmatula/Ogla/oglaMatrixCpu/MathOperationsCpu.cpp

${OBJECTDIR}/_ext/852485519/MultiplicationConstImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/MultiplicationConstImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/MultiplicationConstImpl.o /home/mmatula/Ogla/oglaMatrixCpu/MultiplicationConstImpl.cpp

${OBJECTDIR}/_ext/852485519/SimpleDiagonalizationImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/SimpleDiagonalizationImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/SimpleDiagonalizationImpl.o /home/mmatula/Ogla/oglaMatrixCpu/SimpleDiagonalizationImpl.cpp

${OBJECTDIR}/_ext/852485519/SubstractionImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/SubstractionImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/SubstractionImpl.o /home/mmatula/Ogla/oglaMatrixCpu/SubstractionImpl.cpp

${OBJECTDIR}/_ext/852485519/TensorProductImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/TensorProductImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/TensorProductImpl.o /home/mmatula/Ogla/oglaMatrixCpu/TensorProductImpl.cpp

${OBJECTDIR}/_ext/852485519/TransposeImpl.o: /home/mmatula/Ogla/oglaMatrixCpu/TransposeImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/852485519
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/852485519/TransposeImpl.o /home/mmatula/Ogla/oglaMatrixCpu/TransposeImpl.cpp

${OBJECTDIR}/DeterminantImpl.o: DeterminantImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DeterminantImpl.o DeterminantImpl.cpp

${OBJECTDIR}/HostMatrixStructure.o: HostMatrixStructure.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/HostMatrixStructure.o HostMatrixStructure.cpp

${OBJECTDIR}/IRAMOperationsImpl.o: IRAMOperationsImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/IRAMOperationsImpl.o IRAMOperationsImpl.cpp

${OBJECTDIR}/QRDecompositionImpl.o: QRDecompositionImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Wall -I../oglaMath -I../oglaUtils -I../oglaServerUtils -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/QRDecompositionImpl.o QRDecompositionImpl.cpp

# Subprojects
.build-subprojects:
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMatrix && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCpu.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMatrix && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
