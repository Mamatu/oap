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
	${OBJECTDIR}/_ext/657242938/KernelExecutor.o


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
LDLIBSOPTIONS=-Wl,-rpath,../oglaMatrix/dist/Debug/GNU-Linux-x86 -L../oglaMatrix/dist/Debug/GNU-Linux-x86 -loglaMatrix -Wl,-rpath,../oglaMath/dist/Debug/GNU-Linux-x86 -L../oglaMath/dist/Debug/GNU-Linux-x86 -loglaMath -Wl,-rpath,../oglaUtils/dist/Debug/GNU-Linux-x86 -L../oglaUtils/dist/Debug/GNU-Linux-x86 -loglaUtils

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT}: ../oglaMatrix/dist/Debug/GNU-Linux-x86/liboglaMatrix.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT}: ../oglaMath/dist/Debug/GNU-Linux-x86/liboglaMath.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT}: ../oglaUtils/dist/Debug/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/657242938/KernelExecutor.o: /home/mmatula/Ogla/oglaMatrixCuda/KernelExecutor.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/657242938
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaMath -I../oglaMatrix -I../oglaUtils -I/usr/local/cuda-6.0/include -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/657242938/KernelExecutor.o /home/mmatula/Ogla/oglaMatrixCuda/KernelExecutor.cpp

# Subprojects
.build-subprojects:
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Debug
	cd ../oglaMath && ${MAKE}  -f Makefile CONF=Debug
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaCuda.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Debug clean
	cd ../oglaMath && ${MAKE}  -f Makefile CONF=Debug clean
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
