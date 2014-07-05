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
	${OBJECTDIR}/_ext/1679335344/GAProcess.o \
	${OBJECTDIR}/_ext/1679335344/GATypes.o


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
LDLIBSOPTIONS=/home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a -Wl,-rpath,/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -loglaServerUtils -Wl,-rpath,/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86 -loglaMath

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86/liboglaServerUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT}: /home/mmatula/Ogla/oglaMath/dist/Debug/GNU-Linux-x86/liboglaMath.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/1679335344/GAProcess.o: /home/mmatula/Ogla/oglaGA/GAProcess.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1679335344
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1679335344/GAProcess.o /home/mmatula/Ogla/oglaGA/GAProcess.cpp

${OBJECTDIR}/_ext/1679335344/GATypes.o: /home/mmatula/Ogla/oglaGA/GATypes.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1679335344
	${RM} "$@.d"
	$(COMPILE.cc) -g -I../oglaServerUtils -I../oglaUtils -I/home/mmatula/Ogla/oglaMath -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1679335344/GATypes.o /home/mmatula/Ogla/oglaGA/GATypes.cpp

# Subprojects
.build-subprojects:
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaGA.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaMath && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
