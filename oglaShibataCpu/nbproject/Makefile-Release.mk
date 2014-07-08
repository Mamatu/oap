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
	${OBJECTDIR}/_ext/846718754/Parameters.o \
	${OBJECTDIR}/_ext/846718754/RealTransferMatrixCpu.o \
	${OBJECTDIR}/_ext/846718754/TransferMatrix.o \
	${OBJECTDIR}/_ext/846718754/TransferMatrixCpu.o \
	${OBJECTDIR}/HostTreePointerCreator.o \
	${OBJECTDIR}/TreePointer.o


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
LDLIBSOPTIONS=/home/mmatula/Ogla/oglaMath/dist/Release/GNU-Linux-x86/liboglamath.a -Wl,-rpath,/home/mmatula/Ogla/oglaMatrixCpu/dist/Release/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrixCpu/dist/Release/GNU-Linux-x86 -loglaMatrixCpu -Wl,-rpath,../oglaMatrix/dist/Release/GNU-Linux-x86 -L../oglaMatrix/dist/Release/GNU-Linux-x86 -loglaMatrix -Wl,-rpath,../oglaUtils/dist/Release/GNU-Linux-x86 -L../oglaUtils/dist/Release/GNU-Linux-x86 -loglaUtils

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMath/dist/Release/GNU-Linux-x86/liboglamath.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMatrixCpu/dist/Release/GNU-Linux-x86/liboglaMatrixCpu.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}: ../oglaMatrix/dist/Release/GNU-Linux-x86/liboglaMatrix.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}: ../oglaUtils/dist/Release/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/846718754/Parameters.o: /home/mmatula/Ogla/oglaShibataCpu/Parameters.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/846718754
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/846718754/Parameters.o /home/mmatula/Ogla/oglaShibataCpu/Parameters.cpp

${OBJECTDIR}/_ext/846718754/RealTransferMatrixCpu.o: /home/mmatula/Ogla/oglaShibataCpu/RealTransferMatrixCpu.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/846718754
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/846718754/RealTransferMatrixCpu.o /home/mmatula/Ogla/oglaShibataCpu/RealTransferMatrixCpu.cpp

${OBJECTDIR}/_ext/846718754/TransferMatrix.o: /home/mmatula/Ogla/oglaShibataCpu/TransferMatrix.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/846718754
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/846718754/TransferMatrix.o /home/mmatula/Ogla/oglaShibataCpu/TransferMatrix.cpp

${OBJECTDIR}/_ext/846718754/TransferMatrixCpu.o: /home/mmatula/Ogla/oglaShibataCpu/TransferMatrixCpu.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/846718754
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/846718754/TransferMatrixCpu.o /home/mmatula/Ogla/oglaShibataCpu/TransferMatrixCpu.cpp

${OBJECTDIR}/HostTreePointerCreator.o: HostTreePointerCreator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/HostTreePointerCreator.o HostTreePointerCreator.cpp

${OBJECTDIR}/TreePointer.o: TreePointer.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -O2 -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaMath -I/home/mmatula/Ogla/oglaMatrixCpu -I../oglaMatrix -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/TreePointer.o TreePointer.cpp

# Subprojects
.build-subprojects:
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Release
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Release
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Release
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Release

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCpu.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Release clean
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Release clean
	cd ../oglaMatrix && ${MAKE}  -f Makefile CONF=Release clean
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Release clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
