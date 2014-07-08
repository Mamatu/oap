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
	${OBJECTDIR}/main.o


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
LDLIBSOPTIONS=-ldl -lpthread -Wl,-rpath,../oglaMatrix/dist/Release/GNU-Linux-x86 -L../oglaMatrix/dist/Release/GNU-Linux-x86 -loglaMatrix -Wl,-rpath,/home/mmatula/Ogla/oglaShibataCpu/dist/Release/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaShibataCpu/dist/Release/GNU-Linux-x86 -loglaShibataCpu -Wl,-rpath,/home/mmatula/Ogla/oglaMatrixCpu/dist/Release/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrixCpu/dist/Release/GNU-Linux-x86 -loglaMatrixCpu /home/mmatula/Ogla/oglaMath/dist/Release/GNU-Linux-x86/liboglamath.a -Wl,-rpath,../oglaShibataCuda/dist/Release/GNU-Linux-x86 -L../oglaShibataCuda/dist/Release/GNU-Linux-x86 -loglaShibataCuda -Wl,-rpath,../oglaMatrixCuda/dist/Release/GNU-Linux-x86 -L../oglaMatrixCuda/dist/Release/GNU-Linux-x86 -loglaMatrixCuda -Wl,-rpath,../oglaUtils/dist/Release/GNU-Linux-x86 -L../oglaUtils/dist/Release/GNU-Linux-x86 -loglaUtils

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: ../oglaMatrix/dist/Release/GNU-Linux-x86/liboglaMatrix.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: /home/mmatula/Ogla/oglaShibataCpu/dist/Release/GNU-Linux-x86/liboglaShibataCpu.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: /home/mmatula/Ogla/oglaMatrixCpu/dist/Release/GNU-Linux-x86/liboglaMatrixCpu.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: /home/mmatula/Ogla/oglaMath/dist/Release/GNU-Linux-x86/liboglamath.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: ../oglaShibataCuda/dist/Release/GNU-Linux-x86/liboglaShibataCuda.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: ../oglaMatrixCuda/dist/Release/GNU-Linux-x86/liboglaMatrixCuda.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: ../oglaUtils/dist/Release/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaShibataCpu -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaUtils -I../oglaMatrix -I../oglaShibataCuda -I../oglaCuda -I../oglaMatrixCuda -I../oglaCore -I/usr/local/cuda-6.0/include -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Release
	cd /home/mmatula/Ogla/oglaShibataCpu && ${MAKE}  -f Makefile CONF=Release
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Release
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Release
	cd ../oglaShibataCuda && ${MAKE}  -f Makefile CONF=Release
	cd ../oglaMatrixCuda && ${MAKE}  -f Makefile CONF=Release
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Release

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglashibatamgr

# Subprojects
.clean-subprojects:
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Release clean
	cd /home/mmatula/Ogla/oglaShibataCpu && ${MAKE}  -f Makefile CONF=Release clean
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Release clean
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Release clean
	cd ../oglaShibataCuda && ${MAKE}  -f Makefile CONF=Release clean
	cd ../oglaMatrixCuda && ${MAKE}  -f Makefile CONF=Release clean
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Release clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
