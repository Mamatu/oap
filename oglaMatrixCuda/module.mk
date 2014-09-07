include ../project_generic.mk
TARGET := liboglaMatrixCuda
INCLUDE_PATHS :=
EXTRA_LIBS := -L/usr/local/cuda-6.0/lib -lcudart -lcuda
