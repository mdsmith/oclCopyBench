
#define FLOAT
#define DOUBLE
#define ULTRA
#define TEN
//#define BUF_SIZE 4096
#define BUF_SIZE 8192
#define VERBOSITY_LEVEL 1
//#define BUF_SIZE 1024

#include <iostream>
#include <sys/time.h>

#if defined(__APPLE__) || defined(APPLE)
    #include <OpenCL/OpenCL.h>
#else
    #include <CL/opencl.h>
#endif
using namespace std;

#define STRINGIFY(src) #src

#define kernel_header "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n typedef struct tag_UltraFloat{float mantissa; int base; int exponent;} UltraFloat; typedef struct tag_TenFloat{float mantissa; int exponent;} TenFloat;"

inline const char* Kernels()
{
    static const char* kernels =
        kernel_header
        #include "kernel.cl"
        ;
    return kernels;
}

cl_context ctx;
cl_kernel float_kernel;
cl_kernel double_kernel;
cl_kernel ultra_kernel;
cl_kernel ten_kernel;
cl_command_queue queue;
size_t global_work_size = BUF_SIZE;
size_t local_work_size;
cl_int err_num;
cl_program prog;

cl_mem d_buf1;
cl_mem d_buf2;

// XXX do these but with ints instead of floats
struct UltraFloat
{
    cl_float mantissa;
    cl_int base;
    cl_int exponent;
};

struct TenFloat
{
    cl_float mantissa;
    // Base == 10;
    cl_int exponent;
};

int setup_context();
int compile_kernel(const char* kernel_name, cl_kernel &kernel);
int create_buffer(cl_mem &d_buf, void* buffer, size_t size);
int launch_kernel(cl_kernel kernel);
int read_buffer(cl_mem &d_buf, void* data, size_t size);

int main()
{
    setup_context();
    const char* kernel_name;
    size_t size;

    float runtime = 0;
    timeval t1, t2;

#if defined(FLOAT) || defined(DOUBLE)
    int* exponents = new int[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        exponents[i] = 0;
    }
#endif

    // ******* BURN RUN SO THE FIRST ONE DOESN'T SUCK *******
    float* burn_floatDataset = new float[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        burn_floatDataset[i] = 2;
    }
    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);
    size = sizeof(float) * BUF_SIZE;

    gettimeofday(&t1, NULL);

    create_buffer(d_buf1, burn_floatDataset, size);
    create_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
    err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
    err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }
    launch_kernel(float_kernel);
    read_buffer(d_buf1, burn_floatDataset, size);
    read_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    // ******* BURN RUN SO THE FIRST ONE DOESN'T SUCK *******

#ifdef FLOAT
    runtime = 0;
    float* floatDataset = new float[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        floatDataset[i] = 2;
    }
    kernel_name = "floatTest";
    compile_kernel(kernel_name, float_kernel);
    size = sizeof(float) * BUF_SIZE;

    gettimeofday(&t1, NULL);

    create_buffer(d_buf1, floatDataset, size);
    create_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
    err_num  = clSetKernelArg(float_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
    err_num  |= clSetKernelArg(float_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }
    launch_kernel(float_kernel);
    read_buffer(d_buf1, floatDataset, size);
    read_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float results:" << endl;
        for (int i = 0; i < BUF_SIZE; i++)
        {
            cout << floatDataset[i];
            cout << "10";
            cout << exponents[i];
        }
        cout << endl;
    }
    cout << "Float runtime: " << runtime << endl;
#endif

#ifdef DOUBLE
    runtime = 0;
    double* doubleDataset = new double[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        doubleDataset[i] = 3;
        exponents[i] = 0;
    }
    kernel_name = "doubleTest";
    compile_kernel(kernel_name, double_kernel);
    size = sizeof(double) * BUF_SIZE;

    gettimeofday(&t1, NULL);

    create_buffer(d_buf1, doubleDataset, size);
    create_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);
    err_num  = clSetKernelArg(double_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
    err_num  |= clSetKernelArg(double_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }
    launch_kernel(double_kernel);
    read_buffer(d_buf1, doubleDataset, size);
    read_buffer(d_buf2, exponents, sizeof(int) * BUF_SIZE);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Double results:" << endl;
        for (int i = 0; i < BUF_SIZE; i++)
        {
            cout << doubleDataset[i];
            cout << "10";
            cout << exponents[i];
        }
        cout << endl;
    }
    cout << "Double runtime: " << runtime << endl;
#endif

#ifdef ULTRA
    runtime = 0;
    UltraFloat* ultraDataset = new UltraFloat[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        ultraDataset[i].mantissa = 4;
        ultraDataset[i].base = 10;
        ultraDataset[i].exponent = 0;
    }
    kernel_name = "ultraTest";
    compile_kernel(kernel_name, ultra_kernel);
    size = sizeof(UltraFloat) * BUF_SIZE;

    gettimeofday(&t1, NULL);

    create_buffer(d_buf1, ultraDataset, size);
    err_num  = clSetKernelArg(ultra_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }
    launch_kernel(ultra_kernel);
    read_buffer(d_buf1, ultraDataset, size);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Ultra dataset" << endl;
        for (int i = 0; i < BUF_SIZE; i++)
        {
            cout << ultraDataset[i].mantissa;
            cout << ultraDataset[i].base;
            cout << ultraDataset[i].exponent;
        }
        cout << endl;
    }
    cout << "Ultra runtime: " << runtime << endl;
#endif

#ifdef TEN
    runtime = 0;
    TenFloat* tenDataset = new TenFloat[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
    {
        tenDataset[i].mantissa = 5;
        tenDataset[i].exponent = 0;
    }
    kernel_name = "tenTest";
    compile_kernel(kernel_name, ten_kernel);
    size = sizeof(TenFloat) * BUF_SIZE;

    gettimeofday(&t1, NULL);

    create_buffer(d_buf1, tenDataset, size);
    err_num  = clSetKernelArg(ten_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel arg set fail" << endl;
        exit(err_num);
    }
    launch_kernel(ten_kernel);
    read_buffer(d_buf1, tenDataset, size);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Ten dataset" << endl;
        for (int i = 0; i < BUF_SIZE; i++)
        {
            cout << tenDataset[i].mantissa;
            cout << 10;
            cout << tenDataset[i].exponent;
        }
        cout << endl;
    }
    cout << "Ten runtime: " << runtime << endl;
#endif
    return 0;
}

int create_buffer(cl_mem &d_buf, void* data, size_t size)
{
    d_buf = clCreateBuffer( ctx,
                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    size,
                    data,
                    &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
    /*
    err_num = clEnqueueWriteBuffer( queue,
                                    d_buf,
                                    CL_TRUE,
                                    0,
                                    size,
                                    data,
                                    0,
                                    NULL,
                                    NULL
            );
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
            */
    return 0;
}

int read_buffer(cl_mem &d_buf, void* data, size_t size)
{
    err_num = clEnqueueReadBuffer(  queue,
                                    d_buf,
                                    CL_TRUE,
                                    0,
                                    size,
                                    data,
                                    0,
                                    NULL,
                                    NULL
            );
    if (err_num != CL_SUCCESS)
    {
        cout << "read fail" << endl;
        exit(err_num);
    }
    return 0;
}

int launch_kernel(cl_kernel kernel)
{
    err_num = clEnqueueNDRangeKernel(   queue,
                                        kernel,
                                        1,
                                        0,
                                        &global_work_size,
                                        &local_work_size,
                                        0,
                                        NULL,
                                        NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "kernel launch fail" << endl;
        exit(err_num);
    }
    clFinish(queue);
    return 0;
}


int setup_context()
{
    cl_platform_id plat = NULL;
    cl_device_id *devices = NULL;
    cl_device_id device = NULL;
    cl_uint dev_count = 0;
    err_num = CL_SUCCESS;

    err_num = clGetPlatformIDs(1, &plat, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "Plat fail" << endl;
        exit(err_num);
    }

    // Dev setup
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
    devices = (cl_device_id *)malloc(dev_count * sizeof(cl_device_id));
    err_num = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, dev_count, devices, NULL);
    device = devices[0]; // XXX set back down to 0
    if (err_num != CL_SUCCESS)
    {
        cout << "Dev fail" << endl;
        exit(err_num);
    }

    // Context setup
    // 1 == my device count (arbitrary)
    ctx = clCreateContext(0, 1, &device, NULL, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "Ctx fail" << endl;
        exit(err_num);
    }

    // get device info
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    size_t wg_max;
    err_num = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name),
                             vendor_name, &returned_size);
    err_num |= clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                              device_name, &returned_size);
    err_num |= clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wg_max),
                              &wg_max, &returned_size);
    if (err_num != CL_SUCCESS)
    {
        cout << "Name fetch fail" << endl;
        exit(err_num);
    }
    printf("Connecting to %s %s...\n", vendor_name, device_name);
    printf("Max work group size: %"PRIuPTR"\n", wg_max);
    local_work_size = wg_max;

    // queue setup
    queue = clCreateCommandQueue(   ctx,
                                    device,
                                    0,
                                    &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "queue fail" << endl;
        exit(err_num);
    }

    // prog setup
    const char* source = Kernels();
    prog = clCreateProgramWithSource(ctx, 1, &source, NULL, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "compile fail" << endl;
        exit(err_num);
    }

    // build program
    err_num = clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", NULL, NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "build fail " << err_num << endl;
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        exit(err_num);
    }
    return 0;
}

int compile_kernel(const char* kernel_name, cl_kernel &kernel)
{
    // kernel setup
    kernel = clCreateKernel(prog, kernel_name, &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make kernel fail" << endl;
        exit(err_num);
    }
    return 0;
}
