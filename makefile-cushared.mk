include ../makefile-core.mk

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES) $(CU_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/
	$(CXX) $(SANITIZER_LINK) -shared $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC -lpthread -ldl $(LIBS_DIRS) $(LIBS) -o $@.so
	$(NVCC) $(NVCCOPTIONS) $(NVCC_INCLUDE_DIRS) $(INCLUDE_DIRS) --cubin $(CU_FILES) -o $@.cubin
	cp $@.so $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/lib/
	cp $@.cubin $(OGLA_PATH)/dist/$(MODE)/$(PLATFORM)/cubin/
build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(LIBS_DIRS) $(LIBS) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
clean:
	rm -f dist/$(MODE)/$(PLATFORM)/*
	rm -f build/$(MODE)/$(PLATFORM)/*
