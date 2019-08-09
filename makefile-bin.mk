include ../makefile-core.mk

OAP_LIBS_PATHS_TEMP = $(shell ls $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/)
OAP_LIBS_PATHS = $(addprefix $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/, $(OAP_LIBS_PATHS_TEMP))

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin/
	$(CXX) $(SANITIZER_LINKING) $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC -L/usr/lib -L/usr/local/lib -lpthread -ldl $(OAP_LIBS_PATHS) $(LIBS_DIRS) $(LIBS) -o $@
	cp $@ $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/bin/

-include $(OCPP_FILES:.o=.d)

build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(SANITIZER_COMPILATION) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
	$(CXX) -MM  $(SANITIZER_COMPILATION) $(CXXOPTIONS) $(INCLUDE_DIRS) $*.cpp > build/$(MODE)/$(PLATFORM)/$*.d
	@mv -f  build/$(MODE)/$(PLATFORM)/$*.d  build/$(MODE)/$(PLATFORM)/$*.d.tmp
	@sed -e 's|.*:|build/$(MODE)/$(PLATFORM)/$*.o:|' < build/$(MODE)/$(PLATFORM)/$*.d.tmp > build/$(MODE)/$(PLATFORM)/$*.d
	@sed -e 's/.*://' -e 's/\\$$//' < build/$(MODE)/$(PLATFORM)/$*.d.tmp | fmt -1 | \
		sed -e 's/^ *//' -e 's/$$/:/' >> build/$(MODE)/$(PLATFORM)/$*.d
	@rm -f build/$(MODE)/$(PLATFORM)/$*.d.tmp
clean:
	rm -rf dist/$(MODE)/$(PLATFORM)/*
	rm -rf build/$(MODE)/$(PLATFORM)/*
