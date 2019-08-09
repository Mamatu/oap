include ../makefile-core.mk

OAP_LIBS_PATHS_TEMP = $(shell ls $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/)
OAP_LIBS_PATHS = $(addprefix $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/, $(OAP_LIBS_PATHS_TEMP))

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin/
	$(CXX) -c $(SANITIZER_COMPILATION) $(CXXOPTIONS) -isystem ${OAP_GTEST_PATH}/include -I${OAP_GTEST_PATH} -pthread ${OAP_GTEST_PATH}/src/gtest-all.cc
	$(CXX) -c $(SANITIZER_COMPILATION) $(CXXOPTIONS) -isystem ${OAP_GTEST_PATH}/include -I${OAP_GTEST_PATH} -pthread ${OAP_GTEST_PATH}/src/gtest_main.cc
	$(CXX) -c $(SANITIZER_COMPILATION) $(CXXOPTIONS) -isystem ${OAP_GMOCK_PATH}/include -I${OAP_GMOCK_PATH} -I${OAP_GTEST_PATH}/include -pthread -c ${OAP_GMOCK_PATH}/src/gmock-all.cc
	ar -rv libgmock.a gmock-all.o
	ar -rv libgtest.a gtest-all.o gtest_main.o
	$(CXX) $(SANITIZER_LINKING) $(OCPP_FILES) -L/usr/lib -L/usr/local/lib $(OAP_LIBS_PATHS) $(LIBS_DIRS) $(LIBS) libgtest.a libgmock.a -lpthread -o $@
	cp $@ $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/bin/
build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(SANITIZER_LINKING) $(CXXOPTIONS) -I $(OAP_GTEST_PATH)/include -I $(OAP_GMOCK_PATH)/include $(INCLUDE_DIRS) -lpthread $< -o $@
clean:
	rm -rf dist/$(MODE)/$(PLATFORM)/*
	rm -rf build/$(MODE)/$(PLATFORM)/*
	rm -f libgtest.a
	rm -f libgmock.a
