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
	${OBJECTDIR}/_ext/1941267477/Brackets.o \
	${OBJECTDIR}/_ext/1941267477/Code.o \
	${OBJECTDIR}/_ext/1941267477/Function.o \
	${OBJECTDIR}/_ext/1941267477/Operator.o \
	${OBJECTDIR}/_ext/1941267477/Parser.o \
	${OBJECTDIR}/Variable.o


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
LDLIBSOPTIONS=-Wl,-rpath,/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -loglaMath -Wl,-rpath,/home/mmatula/Ogla/oglaMatrixCpu/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrixCpu/dist/Debug/GNU-Linux-x86 -loglaMatrixCpu -Wl,-rpath,/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86 -loglaMatrixCuda -Wl,-rpath,../oglaUtils/dist/Debug/GNU-Linux-x86 -L../oglaUtils/dist/Debug/GNU-Linux-x86 -loglaUtils -Wl,-rpath,../oglaMatrix/dist/Debug/GNU-Linux-x86 -L../oglaMatrix/dist/Debug/GNU-Linux-x86 -loglaMatrix

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86/liboglaMath.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMatrixCpu/dist/Debug/GNU-Linux-x86/liboglaMatrixCpu.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}: ../oglaUtils/dist/Debug/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}: ../oglaMatrix/dist/Debug/GNU-Linux-x86/liboglaMatrix.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/1941267477/Brackets.o: /home/mmatula/Ogla/oglaParser/Brackets.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1941267477
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -I../oglaMath -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1941267477/Brackets.o /home/mmatula/Ogla/oglaParser/Brackets.cpp

${OBJECTDIR}/_ext/1941267477/Code.o: /home/mmatula/Ogla/oglaParser/Code.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1941267477
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -I../oglaMath -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1941267477/Code.o /home/mmatula/Ogla/oglaParser/Code.cpp

${OBJECTDIR}/_ext/1941267477/Function.o: /home/mmatula/Ogla/oglaParser/Function.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1941267477
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -I../oglaMath -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1941267477/Function.o /home/mmatula/Ogla/oglaParser/Function.cpp

${OBJECTDIR}/_ext/1941267477/Operator.o: /home/mmatula/Ogla/oglaParser/Operator.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1941267477
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -I../oglaMath -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1941267477/Operator.o /home/mmatula/Ogla/oglaParser/Operator.cpp

${OBJECTDIR}/_ext/1941267477/Parser.o: /home/mmatula/Ogla/oglaParser/Parser.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1941267477
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -I../oglaMath -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1941267477/Parser.o /home/mmatula/Ogla/oglaParser/Parser.cpp

${OBJECTDIR}/Variable.o: Variable.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -I../oglaMath -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/Variable.o Variable.cpp

# Subprojects
.build-subprojects:
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMatrixCuda && ${MAKE}  -f Makefile CONF=Debug
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Debug
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaParser.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMatrixCuda && ${MAKE}  -f Makefile CONF=Debug clean
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
