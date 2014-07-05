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
	${OBJECTDIR}/_ext/708470008/CoreInterfaces.o \
	${OBJECTDIR}/_ext/708470008/Loader.o \
	${OBJECTDIR}/_ext/708470008/OglaInterface.o \
	${OBJECTDIR}/_ext/708470008/OglaServer.o \
	${OBJECTDIR}/_ext/708470008/Status.o


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
LDLIBSOPTIONS=-ldl -lpthread -Wl,-rpath,../oglaUtils/dist/Debug/GNU-Linux-x86 -L../oglaUtils/dist/Debug/GNU-Linux-x86 -loglaUtils

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaServerUtils.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaServerUtils.${CND_DLIB_EXT}: ../oglaUtils/dist/Debug/GNU-Linux-x86/liboglaUtils.so

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaServerUtils.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaServerUtils.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/708470008/CoreInterfaces.o: /home/mmatula/Ogla/oglaServerUtils/CoreInterfaces.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/708470008
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/708470008/CoreInterfaces.o /home/mmatula/Ogla/oglaServerUtils/CoreInterfaces.cpp

${OBJECTDIR}/_ext/708470008/Loader.o: /home/mmatula/Ogla/oglaServerUtils/Loader.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/708470008
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/708470008/Loader.o /home/mmatula/Ogla/oglaServerUtils/Loader.cpp

${OBJECTDIR}/_ext/708470008/OglaInterface.o: /home/mmatula/Ogla/oglaServerUtils/OglaInterface.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/708470008
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/708470008/OglaInterface.o /home/mmatula/Ogla/oglaServerUtils/OglaInterface.cpp

${OBJECTDIR}/_ext/708470008/OglaServer.o: /home/mmatula/Ogla/oglaServerUtils/OglaServer.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/708470008
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/708470008/OglaServer.o /home/mmatula/Ogla/oglaServerUtils/OglaServer.cpp

${OBJECTDIR}/_ext/708470008/Status.o: /home/mmatula/Ogla/oglaServerUtils/Status.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/708470008
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -I../oglaUtils -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/708470008/Status.o /home/mmatula/Ogla/oglaServerUtils/Status.cpp

# Subprojects
.build-subprojects:
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Debug

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaServerUtils.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:
	cd ../oglaUtils && ${MAKE}  -f Makefile CONF=Debug clean

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
