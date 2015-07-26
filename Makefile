include project_generic.mk

.PHONY: subdirs $(OGLA_MODULES)
subdirs: $(OGLA_MODULES)
$(OGLA_MODULES):
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	mkdir -p dist/$(MODE)/$(PLATFORM)/cubin
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin
	$(MAKE) -C $@
.PHONY: clean
clean:
	for dir in $(OGLA_MODULES); do \
	$(MAKE) -C $$dir clean; \
	done
	rm -rf */dist/$(MODE)/$(PLATFORM)/*
	rm -rf */build/$(MODE)/$(PLATFORM)/*
	rm -rf dist/$(MODE)/$(PLATFORM)/lib/*
	rm -rf dist/$(MODE)/$(PLATFORM)/cubin/*
	rm -rf dist/$(MODE)/$(PLATFORM)/bin/*
