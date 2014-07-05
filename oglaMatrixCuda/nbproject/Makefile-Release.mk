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
CND_CONF=Release
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/_ext/657242938/DeviceMatrixModules.o \
	${OBJECTDIR}/_ext/657242938/Kernels.o \
	${OBJECTDIR}/_ext/657242938/KernelsOperations.o \
	${OBJECTDIR}/_ext/657242938/MathOperationsCuda.o \
	${OBJECTDIR}/_ext/657242938/Parameters.o \
	${OBJECTDIR}/DHMatrixCopier.o \
	${OBJECTDIR}/DeviceMatrixCopier.o \
	${OBJECTDIR}/DeviceMatrixStructure.o \
	${OBJECTDIR}/DeviceUtils.o \
	${OBJECTDIR}/HDMatrixCopier.o


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
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCuda.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCuda.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCuda.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/657242938/DeviceMatrixModules.o: /home/mmatula/Ogla/oglaMatrixCuda/DeviceMatrixModules.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/657242938
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/657242938/DeviceMatrixModules.o /home/mmatula/Ogla/oglaMatrixCuda/DeviceMatrixModules.cpp

${OBJECTDIR}/_ext/657242938/Kernels.o: /home/mmatula/Ogla/oglaMatrixCuda/Kernels.cu 
	${MKDIR} -p ${OBJECTDIR}/_ext/657242938
	${RM} "$@.d"
	$(COMPILE.c) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/657242938/Kernels.o /home/mmatula/Ogla/oglaMatrixCuda/Kernels.cu

${OBJECTDIR}/_ext/657242938/KernelsOperations.o: /home/mmatula/Ogla/oglaMatrixCuda/KernelsOperations.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/657242938
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/657242938/KernelsOperations.o /home/mmatula/Ogla/oglaMatrixCuda/KernelsOperations.cpp

${OBJECTDIR}/_ext/657242938/MathOperationsCuda.o: /home/mmatula/Ogla/oglaMatrixCuda/MathOperationsCuda.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/657242938
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/657242938/MathOperationsCuda.o /home/mmatula/Ogla/oglaMatrixCuda/MathOperationsCuda.cpp

${OBJECTDIR}/_ext/657242938/Parameters.o: /home/mmatula/Ogla/oglaMatrixCuda/Parameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/657242938
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/657242938/Parameters.o /home/mmatula/Ogla/oglaMatrixCuda/Parameters.cpp

${OBJECTDIR}/DHMatrixCopier.o: DHMatrixCopier.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DHMatrixCopier.o DHMatrixCopier.cpp

${OBJECTDIR}/DeviceMatrixCopier.o: DeviceMatrixCopier.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DeviceMatrixCopier.o DeviceMatrixCopier.cpp

${OBJECTDIR}/DeviceMatrixStructure.o: DeviceMatrixStructure.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DeviceMatrixStructure.o DeviceMatrixStructure.cpp

${OBJECTDIR}/DeviceUtils.o: DeviceUtils.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DeviceUtils.o DeviceUtils.cpp

${OBJECTDIR}/HDMatrixCopier.o: HDMatrixCopier.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/HDMatrixCopier.o HDMatrixCopier.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaMatrixCuda.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
