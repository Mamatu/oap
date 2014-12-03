include project_generic.mk

.PHONY: subdirs $(OGLA_INCLUDES)
subdirs: $(OGLA_INCLUDES)
.PHONY: clean
$(OGLA_INCLUDES):
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	mkdir -p dist/$(MODE)/$(PLATFORM)/cubin
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin
	$(MAKE) -C $@
clean:	
	rm -rf */dist/$(MODE)/$(PLATFORM)/*
	rm -rf */build/$(MODE)/$(PLATFORM)/*
	rm -rf dist/$(MODE)/$(PLATFORM)/lib/*
	rm -rf dist/$(MODE)/$(PLATFORM)/cubin/*
	rm -rf dist/$(MODE)/$(PLATFORM)/bin/*
