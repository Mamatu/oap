include project_generic.mk

.PHONY: subdirs $(OAP_MODULES)
subdirs: $(OAP_MODULES)
$(OAP_MODULES):
	mkdir -p /tmp/Oap/tests_data
	tar -xzf ./oap2dt3d/data/images_monkey.tar.gz ./oap2dt3d/data/images_monkey
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	mkdir -p dist/$(MODE)/$(PLATFORM)/cubin
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin
	$(MAKE) -C $@
.PHONY: clean
clean:
	for dir in $(OAP_MODULES); do \
	$(MAKE) -C $$dir clean; \
	done
	rm -rf /tmp/Oap/tests_data
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

