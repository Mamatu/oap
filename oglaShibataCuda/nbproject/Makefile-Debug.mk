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
	${OBJECTDIR}/_ext/478473223/Kernels.o \
	${OBJECTDIR}/_ext/478473223/RealTransferMatrixCuda.o \
	${OBJECTDIR}/DeviceTreePointerCreator.o


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
LDLIBSOPTIONS=/home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a -Wl,-rpath,/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -loglaServerUtils -Wl,-rpath,/home/mmatula/Ogla/oglaMatrixCpu/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrixCpu/dist/Debug/GNU-Linux-x86 -loglaMatrixCpu -Wl,-rpath,/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86 -loglaMatrixCuda -Wl,-rpath,/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -loglaMatrix -Wl,-rpath,/home/mmatula/Ogla/oglaShibataCpu/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaShibataCpu/dist/Debug/GNU-Linux-x86 -loglaShibataCpu

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86/liboglaServerUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMatrixCpu/dist/Debug/GNU-Linux-x86/liboglaMatrixCpu.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMatrixCuda/dist/Debug/GNU-Linux-x86/liboglaMatrixCuda.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86/liboglaMatrix.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaShibataCpu/dist/Debug/GNU-Linux-x86/liboglaShibataCpu.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/478473223/Kernels.o: /home/mmatula/Ogla/oglaShibataCuda/Kernels.cu 
	${MKDIR} -p ${OBJECTDIR}/_ext/478473223
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -I/usr/local/cuda-6.0/include -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -I../oglaMath -I../oglaShibataCpu -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/478473223/Kernels.o /home/mmatula/Ogla/oglaShibataCuda/Kernels.cu

${OBJECTDIR}/_ext/478473223/RealTransferMatrixCuda.o: /home/mmatula/Ogla/oglaShibataCuda/RealTransferMatrixCuda.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/478473223
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -I/usr/local/cuda-6.0/include -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -I../oglaMath -I../oglaShibataCpu -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/478473223/RealTransferMatrixCuda.o /home/mmatula/Ogla/oglaShibataCuda/RealTransferMatrixCuda.cpp

${OBJECTDIR}/DeviceTreePointerCreator.o: DeviceTreePointerCreator.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaUtils -I/home/mmatula/Ogla/oglaMath -I/usr/local/cuda-6.0/include -I../oglaMatrix -I../oglaMatrixCpu -I../oglaMatrixCuda -I../oglaMath -I../oglaShibataCpu -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/DeviceTreePointerCreator.o DeviceTreePointerCreator.cpp

# Subprojects
.build-subprojects:
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMatrixCuda && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaShibataCpu && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaShibataCuda.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMatrixCpu && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMatrixCuda && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaShibataCpu && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
