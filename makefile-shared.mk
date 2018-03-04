include ../makefile-core.mk

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	$(CXX) $(SANITIZER_LINKING) -shared $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC -lpthread -ldl $(LIBS_DIRS) $(LIBS) -o $@.so
	cp $@.so $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/
build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(SANITIZER_COMPILATION) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
clean:
	rm -r dist/$(MODE)/$(PLATFORM)/*
	rm -r build/$(MODE)/$(PLATFORM)/*
