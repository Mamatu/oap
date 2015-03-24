include ../makefile-core.mk

OGLA_LIBS_PATHS_TEMP = $(shell ls $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/)
OGLA_LIBS_PATHS = $(addprefix $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/, $(OGLA_LIBS_PATHS_TEMP))

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin/
	$(CXX) $(SANITIZER_LINK) $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC -L/usr/lib -L/usr/local/lib -lpthread -ldl $(OGLA_LIBS_PATHS) $(LIBS_DIRS) $(LIBS) -o $@
	cp $@ $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/bin/
build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
clean:
	rm -f dist/$(MODE)/$(PLATFORM)/*
	rm -f build/$(MODE)/$(PLATFORM)/*