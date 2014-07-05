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
LDLIBSOPTIONS=-Wl,-rpath,/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -L/home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86 -loglaServerUtils /home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a -lpthread -ldl

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglaserver

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglaserver: /home/mmatula/Ogla/oglaServerUtils/dist/Debug/GNU-Linux-x86/liboglaServerUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglaserver: /home/mmatula/Ogla/oglaUtils/dist/Debug/GNU-Linux-x86/liboglautils.a

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglaserver: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglaserver ${OBJECTFILES} ${LDLIBSOPTIONS}

${OBJECTDIR}/main.o: main.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I/home/mmatula/Ogla/oglaServerUtils -I/home/mmatula/Ogla/oglaUtils -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/main.o main.cpp

# Subprojects
.build-subprojects:
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/oglaserver

# Subprojects
.clean-subprojects:
	cd /home/mmatula/Ogla/oglaServerUtils && ${MAKE}  -f Makefile CONF=Debug clean
	cd /home/mmatula/Ogla/oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
