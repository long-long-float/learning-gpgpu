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
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " kernel_name file" << endl;
    return 1;
  }

  const string kernel_name(argv[1]);
  const bool use_workgroup = kernel_name == "edge_use_local_mem";

  const string image_path(argv[2]);
  png::image<png::rgb_pixel> image(image_path);

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

    ifstream kernel_ifs("image-process.cl");
    const string code = string(istreambuf_iterator<char>(kernel_ifs), istreambuf_iterator<char>());
    cl::Program program(context, code);
    program.build({devices[0]});

    cl::Kernel kernel(program, kernel_name.c_str());

    const png::uint_32 width = image.get_width();
    const png::uint_32 height = image.get_height();
    const int data_count = width * height * 3; // rgb
    const png::uint_32 local_work_col = 32;
    const png::uint_32 local_work_row = 32;

    // convert image to byte sequence
    vector<byte> raw_data(data_count);
    for (png::uint_32 y = 0; y < height; y++) {
      for (png::uint_32 x = 0; x < width; x++) {
        auto pix = image[y][x];
        const int idx = (y * width + x) * 3;
        raw_data[idx + 0] = pix.red;
        raw_data[idx + 1] = pix.green;
        raw_data[idx + 2] = pix.blue;
      }
    }

    const int data_size = sizeof(byte) * data_count;

    cl::Buffer data(context, CL_MEM_READ_WRITE, data_size);
    cl::Buffer result(context, CL_MEM_READ_WRITE, data_size);

    kernel.setArg(0, data);
    kernel.setArg(1, result);
    kernel.setArg(2, (int)width);
    kernel.setArg(3, (int)height);

    cl::Event event;
    cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

    queue.enqueueWriteBuffer(data, CL_TRUE, 0, data_size, reinterpret_cast<void*>(raw_data.data()));

    queue.enqueueNDRangeKernel(
        kernel, cl::NullRange,
        (use_workgroup ? cl::NDRange(ceil(static_cast<float>(width) / local_work_col) * local_work_col, ceil(static_cast<float>(height) / local_work_row) * local_work_row) : cl::NDRange(width, height)),
        (use_workgroup ? cl::NDRange(local_work_col, local_work_row) : cl::NullRange),
        nullptr, &event);
    event.wait();

    cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    std::cout << "execution time: " << static_cast<double>(end - start) * 1e-3f << " us" << std::endl;

    vector<byte> raw_result(raw_data.size());
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
