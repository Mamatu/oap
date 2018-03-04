include project_generic.mk

.PHONY: subdirs $(OAP_MODULES)
subdirs: $(OAP_MODULES)

$(OAP_MODULES):
	mkdir -p dist/$(MODE)/$(PLATFORM)/lib
	mkdir -p dist/$(MODE)/$(PLATFORM)/cubin
	mkdir -p dist/$(MODE)/$(PLATFORM)/bin
	$(MAKE) -C $@
.PHONY: clean
clean:
	rm -r /tmp/Oap
	rm -f dist/$(MODE)/$(PLATFORM)/lib/*
	rm -f dist/$(MODE)/$(PLATFORM)/cubin/*
	rm -f dist/$(MODE)/$(PLATFORM)/bin/*
	for dir in $(OAP_MODULES); do \
	$(MAKE) -C $$dir clean; \
	done
cuclean:
	for dir in $(CU_OAP_MODULES); do \
	$(MAKE) -C $$dir clean; \
	done
	rm dist/$(MODE)/$(PLATFORM)/cubin/*

