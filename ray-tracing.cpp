#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <png++/png.hpp>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

typedef unsigned char byte;

using namespace std;

int main(int argc, char *argv[])
{
  if (argc != 2) {
    cerr << "Usage: " << argv[0] << " kernel_name" << endl;
    return 1;
  }
  const string kernel_name(argv[1]);

  try {
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) {
      cerr << "No platform found." << endl;
      return 1;
    }

    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0
    };
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

    cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << endl;
    cout << "\tLocal memory size: " << devices[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
    cout << "\tMax workgroup size: " << devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;

    ifstream kernel_ifs("ray-tracing.cl");
    const string code = string(istreambuf_iterator<char>(kernel_ifs), istreambuf_iterator<char>());
    cl::Program program(context, code);
    try {
      program.build({devices[0]});
    } catch (cl::Error const& ex) {
      if (ex.err() == CL_BUILD_PROGRAM_FAILURE) {
        auto buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
        cerr << "Build Error: " << buildlog << endl;
        return 1;
      }
      else {
        throw ex;
      }
    }

    cl::Kernel kernel(program, kernel_name.c_str());

    const png::uint_32 width = 256;
    const png::uint_32 height = 256;
    const int data_count = width * height * 3; // rgb

    const int data_size = sizeof(byte) * data_count;

    cl::Buffer result(context, CL_MEM_READ_WRITE, data_size);

    kernel.setArg(0, result);
    kernel.setArg(1, (int)width);
    kernel.setArg(2, (int)height);

    cl::Event event;
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    queue.enqueueNDRangeKernel(
        kernel, cl::NullRange,
        cl::NDRange(width, height),
        cl::NullRange,
        nullptr, &event);
    event.wait();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    std::cout << "execution time: " << static_cast<double>(end - start) * 1e-3f << " us" << std::endl;

    vector<byte> raw_result(data_size);
    queue.enqueueReadBuffer(result, CL_TRUE, 0, data_size, reinterpret_cast<void*>(raw_result.data()));

    png::image<png::rgb_pixel> result_image(width, height);
    for (png::uint_32 y = 0; y < height; y++) {
      for (png::uint_32 x = 0; x < width; x++) {
        const int idx = (y * width + x) * 3;
        result_image[y][x] = png::rgb_pixel(
            raw_result[idx + 0],
            raw_result[idx + 1],
            raw_result[idx + 2]);
      }
    }
    result_image.write("result-" + kernel_name + ".png");

    cout << "done!" << endl;

    return 0;
  } catch (cl::Error const& ex) {
    cerr << "OpenCL Error: " << ex.what() << " (code " << ex.err() << ")" << endl;
    return 1;
  } catch (exception const& ex) {
    cerr << "Exception: " << ex.what() << endl;
    return 1;
  }
}
