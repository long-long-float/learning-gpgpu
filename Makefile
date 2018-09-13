COMPILER = sudo nvidia-docker run --rm -v `pwd`:/gpu -it gpu g++ -std=c++11

all: image-process

image-process: image-process.cpp
ray-tracing: ray-tracing.cpp

%: %.cpp
	# `libpng-config ...` expands at outer docker container.
	# $(COMPILER) /gpu/$< -lOpenCL `libpng-config --cppflags` `libpng-config --ldflags` -o /gpu/bin/$@
	$(COMPILER) /gpu/$< -lOpenCL -lpng12 -o /gpu/bin/$@

.PHONY: clean
clean:
	rm -rf bin
