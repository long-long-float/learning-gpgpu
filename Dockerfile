FROM nvidia/opencl:devel

RUN apt update
RUN apt install -y g++ libpng++-dev

