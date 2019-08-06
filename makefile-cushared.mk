include ../makefile-core.mk

dist/$(MODE)/$(PLATFORM)/$(TARGET) : $(OCPP_FILES) $(CU_FILES)
	mkdir -p dist/$(MODE)/$(PLATFORM)/
	$(CXX) $(SANITIZER_LINKING) -shared $(INCLUDE_DIRS) $(OCPP_FILES) -fPIC -lpthread -ldl $(LIBS_DIRS) $(LIBS) -o $@.so
	$(NVCC) -DCUDA $(NVCCOPTIONS) $(NVCC_INCLUDE_DIRS) $(INCLUDE_DIRS) --cubin $(CU_FILES) -o $@.cubin
	cp $@.so $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/lib/
	cp $@.cubin $(OAP_PATH)/dist/$(MODE)/$(PLATFORM)/cubin/

-include $(OCPP_FILES:.o=.d)

build/$(MODE)/$(PLATFORM)/%.o : %.cpp
	mkdir -p build/$(MODE)/$(PLATFORM)/
	$(CXX) $(SANITIZER_COMPILATION) $(LIBS_DIRS) $(LIBS) $(CXXOPTIONS) $(INCLUDE_DIRS) $< -o $@
	$(CXX) -MM  $(SANITIZER_COMPILATION) $(CXXOPTIONS) $(INCLUDE_DIRS) $*.cpp > build/$(MODE)/$(PLATFORM)/$*.d
	@mv -f  build/$(MODE)/$(PLATFORM)/$*.d  build/$(MODE)/$(PLATFORM)/$*.d.tmp
	@sed -e 's|.*:|build/$(MODE)/$(PLATFORM)/$*.o:|' < build/$(MODE)/$(PLATFORM)/$*.d.tmp > build/$(MODE)/$(PLATFORM)/$*.d
	@sed -e 's/.*://' -e 's/\\$$//' < build/$(MODE)/$(PLATFORM)/$*.d.tmp | fmt -1 | \
		sed -e 's/^ *//' -e 's/$$/:/' >> build/$(MODE)/$(PLATFORM)/$*.d
	@rm -f build/$(MODE)/$(PLATFORM)/$*.d.tmp
clean:
	rm -f dist/$(MODE)/$(PLATFORM)/*
	rm -f build/$(MODE)/$(PLATFORM)/*
