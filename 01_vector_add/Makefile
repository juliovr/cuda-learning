all: build

build:
	mkdir -p bin/
	nvcc -o bin/vecadd vecadd.cu

run: build
	./bin/vecadd
