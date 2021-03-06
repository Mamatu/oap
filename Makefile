include project_generic.mk

.PHONY: subdirs $(OAP_MODULES)
subdirs: $(OAP_MODULES)

$(OAP_MODULES):
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	mkdir -p dist/$(MODE)/$(PLATFORM)/cubin
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin
	$(MAKE) -C $@ -j$(OAP_BUILD_THREADS)
.PHONY: clean
clean:
	for dir in $(OAP_MODULES); do \
	$(MAKE) -C $$dir clean; \
	done
	rm -rf /tmp/Oap
	rm -rf */dist/$(MODE)/$(PLATFORM)/*
	rm -rf */build/$(MODE)/$(PLATFORM)/*
	rm -rf dist/$(MODE)/$(PLATFORM)/lib/*
	rm -rf dist/$(MODE)/$(PLATFORM)/cubin/*
	rm -rf dist/$(MODE)/$(PLATFORM)/bin/*
cuclean:
	for dir in $(CU_OAP_MODULES); do \
	$(MAKE) -C $$dir clean; \
	done
	rm -rf */dist/$(MODE)/$(PLATFORM)/*
	rm -rf */build/$(MODE)/$(PLATFORM)/*
	rm -rf dist/$(MODE)/$(PLATFORM)/lib/*
	rm -rf dist/$(MODE)/$(PLATFORM)/cubin/*
	rm -rf dist/$(MODE)/$(PLATFORM)/bin/*

