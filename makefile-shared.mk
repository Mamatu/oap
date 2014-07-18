include ../makefile-core.mk

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	$(CXX) -shared $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC -lpthread -ldl $(LIBS_DIRS) $(LIBS) -o $@.so
	cp $@.so $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/
build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
clean:
	rm -f dist/$(MODE)/$(PLATFORM)/*
	rm -f build/$(MODE)/$(PLATFORM)/*
