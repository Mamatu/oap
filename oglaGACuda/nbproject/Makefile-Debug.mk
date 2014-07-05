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
	${OBJECTDIR}/_ext/1652654143/CudaExecution.o \
	${OBJECTDIR}/_ext/1652654143/Functions.o \
	${OBJECTDIR}/_ext/1652654143/GAModuleCUDA.o \
	${OBJECTDIR}/_ext/1652654143/GAProcessCUDA.o \
	${OBJECTDIR}/_ext/1652654143/GARatingCUDA.o \
	${OBJECTDIR}/_ext/1652654143/Parameters.o \
	${OBJECTDIR}/_ext/1652654143/cu_kernel.o


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
LDLIBSOPTIONS=-Wl,-rpath,/home/mmatula/wProject/oglaGA/dist/Debug/GNU-Linux-x86 -L/home/mmatula/wProject/oglaGA/dist/Debug/GNU-Linux-x86 -loglaGA /home/mmatula/wProject/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a -Wl,-rpath,/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -loglaServerUtils

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT}: /home/mmatula/wProject/oglaGA/dist/Debug/GNU-Linux-x86/liboglaGA.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT}: /home/mmatula/wProject/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86/liboglaServerUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/1652654143/CudaExecution.o: /home/mmatula/Ogla/oglaGACuda/CudaExecution.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/CudaExecution.o /home/mmatula/Ogla/oglaGACuda/CudaExecution.cpp

${OBJECTDIR}/_ext/1652654143/Functions.o: /home/mmatula/Ogla/oglaGACuda/Functions.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/Functions.o /home/mmatula/Ogla/oglaGACuda/Functions.cpp

${OBJECTDIR}/_ext/1652654143/GAModuleCUDA.o: /home/mmatula/Ogla/oglaGACuda/GAModuleCUDA.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/GAModuleCUDA.o /home/mmatula/Ogla/oglaGACuda/GAModuleCUDA.cpp

${OBJECTDIR}/_ext/1652654143/GAProcessCUDA.o: /home/mmatula/Ogla/oglaGACuda/GAProcessCUDA.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/GAProcessCUDA.o /home/mmatula/Ogla/oglaGACuda/GAProcessCUDA.cpp

${OBJECTDIR}/_ext/1652654143/GARatingCUDA.o: /home/mmatula/Ogla/oglaGACuda/GARatingCUDA.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/GARatingCUDA.o /home/mmatula/Ogla/oglaGACuda/GARatingCUDA.cpp

${OBJECTDIR}/_ext/1652654143/Parameters.o: /home/mmatula/Ogla/oglaGACuda/Parameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/Parameters.o /home/mmatula/Ogla/oglaGACuda/Parameters.cpp

${OBJECTDIR}/_ext/1652654143/cu_kernel.o: /home/mmatula/Ogla/oglaGACuda/cu_kernel.cu 
	${MKDIR} -p ${OBJECTDIR}/_ext/1652654143
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I../oglaGA -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1652654143/cu_kernel.o /home/mmatula/Ogla/oglaGACuda/cu_kernel.cu

# Subprojects
.build-subprojects:
	cd /home/mmatula/wProject/oglaGA && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/wProject/oglaUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGACuda.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd /home/mmatula/wProject/oglaGA && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/wProject/oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
