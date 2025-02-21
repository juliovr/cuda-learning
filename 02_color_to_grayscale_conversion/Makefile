all: build

build:
	mkdir -p bin/
	nvcc -g -G -o bin/main main.cu

run: build
	./bin/main
