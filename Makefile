include project.mk

.PHONY: subdirs $(OGLA_INCLUDES)
subdirs: $(OGLA_INCLUDES)
.PHONY: clean
$(OGLA_INCLUDES):
	mkdir -p dist/$(MODE)/$(PLATFORM)/dist
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	mkdir -p dist/$(MODE)/$(PLATFORM)/cubin
	$(MAKE) -C $@
clean:	
	rm -rf */dist
	rm -rf */build
	rm -rf dist
