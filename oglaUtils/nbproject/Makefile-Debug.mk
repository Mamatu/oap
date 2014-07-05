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
	${OBJECTDIR}/_ext/1314722299/Argument.o \
	${OBJECTDIR}/_ext/1314722299/ArgumentReader.o \
	${OBJECTDIR}/_ext/1314722299/ArgumentWriter.o \
	${OBJECTDIR}/_ext/1314722299/ArrayTools.o \
	${OBJECTDIR}/_ext/1314722299/Async.o \
	${OBJECTDIR}/_ext/1314722299/Buffer.o \
	${OBJECTDIR}/_ext/1314722299/Callbacks.o \
	${OBJECTDIR}/_ext/1314722299/DynamicLoader.o \
	${OBJECTDIR}/_ext/1314722299/FunctionInfo.o \
	${OBJECTDIR}/_ext/1314722299/FunctionInfoImpl.o \
	${OBJECTDIR}/_ext/1314722299/Module.o \
	${OBJECTDIR}/_ext/1314722299/NumbersGenerator.o \
	${OBJECTDIR}/_ext/1314722299/ObjectInfo.o \
	${OBJECTDIR}/_ext/1314722299/ObjectInfoImpl.o \
	${OBJECTDIR}/_ext/1314722299/Reader.o \
	${OBJECTDIR}/_ext/1314722299/RpcBase.o \
	${OBJECTDIR}/_ext/1314722299/RpcImpl.o \
	${OBJECTDIR}/_ext/1314722299/Serializable.o \
	${OBJECTDIR}/_ext/1314722299/Socket.o \
	${OBJECTDIR}/_ext/1314722299/ThreadUtils.o \
	${OBJECTDIR}/_ext/1314722299/ThreadsMapper.o \
	${OBJECTDIR}/_ext/1314722299/WrapperInterfaces.o \
	${OBJECTDIR}/_ext/1314722299/Writer.o \
	${OBJECTDIR}/LHandle.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-m64 -ggdb3
CXXFLAGS=-m64 -ggdb3

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=-ldl -lpthread

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaUtils.${CND_DLIB_EXT}

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaUtils.${CND_DLIB_EXT}: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaUtils.${CND_DLIB_EXT} ${OBJECTFILES} ${LDLIBSOPTIONS} -shared -fPIC

${OBJECTDIR}/_ext/1314722299/Argument.o: /home/mmatula/Ogla/oglaUtils/Argument.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Argument.o /home/mmatula/Ogla/oglaUtils/Argument.cpp

${OBJECTDIR}/_ext/1314722299/ArgumentReader.o: /home/mmatula/Ogla/oglaUtils/ArgumentReader.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ArgumentReader.o /home/mmatula/Ogla/oglaUtils/ArgumentReader.cpp

${OBJECTDIR}/_ext/1314722299/ArgumentWriter.o: /home/mmatula/Ogla/oglaUtils/ArgumentWriter.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ArgumentWriter.o /home/mmatula/Ogla/oglaUtils/ArgumentWriter.cpp

${OBJECTDIR}/_ext/1314722299/ArrayTools.o: /home/mmatula/Ogla/oglaUtils/ArrayTools.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ArrayTools.o /home/mmatula/Ogla/oglaUtils/ArrayTools.cpp

${OBJECTDIR}/_ext/1314722299/Async.o: /home/mmatula/Ogla/oglaUtils/Async.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Async.o /home/mmatula/Ogla/oglaUtils/Async.cpp

${OBJECTDIR}/_ext/1314722299/Buffer.o: /home/mmatula/Ogla/oglaUtils/Buffer.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Buffer.o /home/mmatula/Ogla/oglaUtils/Buffer.cpp

${OBJECTDIR}/_ext/1314722299/Callbacks.o: /home/mmatula/Ogla/oglaUtils/Callbacks.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Callbacks.o /home/mmatula/Ogla/oglaUtils/Callbacks.cpp

${OBJECTDIR}/_ext/1314722299/DynamicLoader.o: /home/mmatula/Ogla/oglaUtils/DynamicLoader.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/DynamicLoader.o /home/mmatula/Ogla/oglaUtils/DynamicLoader.cpp

${OBJECTDIR}/_ext/1314722299/FunctionInfo.o: /home/mmatula/Ogla/oglaUtils/FunctionInfo.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/FunctionInfo.o /home/mmatula/Ogla/oglaUtils/FunctionInfo.cpp

${OBJECTDIR}/_ext/1314722299/FunctionInfoImpl.o: /home/mmatula/Ogla/oglaUtils/FunctionInfoImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/FunctionInfoImpl.o /home/mmatula/Ogla/oglaUtils/FunctionInfoImpl.cpp

${OBJECTDIR}/_ext/1314722299/Module.o: /home/mmatula/Ogla/oglaUtils/Module.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Module.o /home/mmatula/Ogla/oglaUtils/Module.cpp

${OBJECTDIR}/_ext/1314722299/NumbersGenerator.o: /home/mmatula/Ogla/oglaUtils/NumbersGenerator.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/NumbersGenerator.o /home/mmatula/Ogla/oglaUtils/NumbersGenerator.cpp

${OBJECTDIR}/_ext/1314722299/ObjectInfo.o: /home/mmatula/Ogla/oglaUtils/ObjectInfo.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ObjectInfo.o /home/mmatula/Ogla/oglaUtils/ObjectInfo.cpp

${OBJECTDIR}/_ext/1314722299/ObjectInfoImpl.o: /home/mmatula/Ogla/oglaUtils/ObjectInfoImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ObjectInfoImpl.o /home/mmatula/Ogla/oglaUtils/ObjectInfoImpl.cpp

${OBJECTDIR}/_ext/1314722299/Reader.o: /home/mmatula/Ogla/oglaUtils/Reader.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Reader.o /home/mmatula/Ogla/oglaUtils/Reader.cpp

${OBJECTDIR}/_ext/1314722299/RpcBase.o: /home/mmatula/Ogla/oglaUtils/RpcBase.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/RpcBase.o /home/mmatula/Ogla/oglaUtils/RpcBase.cpp

${OBJECTDIR}/_ext/1314722299/RpcImpl.o: /home/mmatula/Ogla/oglaUtils/RpcImpl.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/RpcImpl.o /home/mmatula/Ogla/oglaUtils/RpcImpl.cpp

${OBJECTDIR}/_ext/1314722299/Serializable.o: /home/mmatula/Ogla/oglaUtils/Serializable.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Serializable.o /home/mmatula/Ogla/oglaUtils/Serializable.cpp

${OBJECTDIR}/_ext/1314722299/Socket.o: /home/mmatula/Ogla/oglaUtils/Socket.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Socket.o /home/mmatula/Ogla/oglaUtils/Socket.cpp

${OBJECTDIR}/_ext/1314722299/ThreadUtils.o: /home/mmatula/Ogla/oglaUtils/ThreadUtils.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ThreadUtils.o /home/mmatula/Ogla/oglaUtils/ThreadUtils.cpp

${OBJECTDIR}/_ext/1314722299/ThreadsMapper.o: /home/mmatula/Ogla/oglaUtils/ThreadsMapper.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/ThreadsMapper.o /home/mmatula/Ogla/oglaUtils/ThreadsMapper.cpp

${OBJECTDIR}/_ext/1314722299/WrapperInterfaces.o: /home/mmatula/Ogla/oglaUtils/WrapperInterfaces.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/WrapperInterfaces.o /home/mmatula/Ogla/oglaUtils/WrapperInterfaces.cpp

${OBJECTDIR}/_ext/1314722299/Writer.o: /home/mmatula/Ogla/oglaUtils/Writer.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/1314722299
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/_ext/1314722299/Writer.o /home/mmatula/Ogla/oglaUtils/Writer.cpp

${OBJECTDIR}/LHandle.o: LHandle.cpp 
	${MKDIR} -p ${OBJECTDIR}
	${RM} "$@.d"
	$(COMPILE.cc) -g -Werror -fPIC  -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/LHandle.o LHandle.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}
	${RM} ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/liboglaUtils.${CND_DLIB_EXT}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
