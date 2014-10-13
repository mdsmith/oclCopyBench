// I haven't added in any code to take non power of two dimensions into
// account. This would add complexity but minimal cost

#define FLOAT_ZERO_THREE_MAT_TILED
#define FLOAT_MAT_CPU
// This is the sitebuffer for the CPU version
#define BUF_SIZE 64*2048
// Size of site buffer
// This is equivalent to a one branch tree with 2048 sites. The *2 is in
// there because results have to be stored somewhere!
#define MAT_SIZE1 64*2048*2
// Size of transition matrix buffer, for just this one branch
#define MAT_SIZE2 64*64
#define VERBOSITY_LEVEL 2
#define REPS 1000

#include <iostream>
#include <string.h>
#include <sys/time.h>

#if defined(__APPLE__) || defined(APPLE)
    #include <OpenCL/OpenCL.h>
#else
    //#include <CL/opencl.h>
    #include <CL/cl.h>
    #include <CL/cl_ext.h>
    //#define FLOAT_ZERO_TWO
#endif
using namespace std;

#define STRINGIFY(src) #src

#define kernel_header "#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n typedef struct tag_UltraFloat{float mantissa; int base; int exponent;} UltraFloat; typedef struct tag_TenFloat{float mantissa; int exponent;} TenFloat;"

inline const char* Kernels()
{
    static const char* kernels =
        kernel_header
        #include "simpleKernel.cl"
        ;
    return kernels;
}

cl_context ctx;
cl_kernel float_kernel;
cl_kernel float_mat_tiled_kernel;
cl_command_queue queue;
size_t global_work_size = BUF_SIZE;
size_t local_work_size;
cl_int err_num;
cl_program prog;
void* mapPtrA;
void* mapPtrB;
void* mapPtrAb;
void* mapPtrBb;

cl_mem d_buf1;
cl_mem d_buf1b;
cl_mem d_buf2;
cl_mem d_buf2b;

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
int create_buffer_zero(cl_mem &d_buf, size_t size);
void* map_buffer_zero(cl_mem d_buf, size_t size);
int unmap_buffer_zero(cl_mem &d_buf, void* buffer);
int launch_kernel(cl_kernel kernel);
int read_buffer(cl_mem &d_buf, void* data, size_t size);

int main()
{
    setup_context();
    const char* kernel_name;
    size_t size;
    size_t size_e;

    float runtime = 0;
    timeval t1, t2;

    int* exponents = new int[BUF_SIZE];
    float* floatCPUDataset = new float[BUF_SIZE];
    float* floatZeroDataset4 = new float[MAT_SIZE1];
    float* floatZeroDataset4b = new float[MAT_SIZE2];
    int* floatZeroExponents4 = new int[MAT_SIZE1];
    int* floatZeroExponents4b = new int[MAT_SIZE2];
    cl_int size1 = sizeof(float) * MAT_SIZE1;
    cl_int size_e1 = sizeof(cl_int) * MAT_SIZE1;
    cl_int size2 = sizeof(float) * MAT_SIZE2;
    cl_int size_e2 = sizeof(cl_int) * MAT_SIZE2;

#if defined(FLOAT) || defined(DOUBLE)
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

    // ******* END BURN RUN SO THE FIRST ONE DOESN'T SUCK *******

#ifdef FLOAT_MAT_CPU
    runtime = 0;
    gettimeofday(&t1, NULL);

    float* mat1 = new float[MAT_SIZE1];
    float* mat1e = new float[MAT_SIZE1];
    float* mat2 = new float[MAT_SIZE2];
    float* mat2e = new float[MAT_SIZE2];
    int nRows = 2048;
    int nCols = 64;
    int m = 64;
    for (int i = 0; i < MAT_SIZE2; i++) {
        mat1[i] = i;
        mat1e[i] = 0;
    }
    for (int i = 0; i < MAT_SIZE2; i++) {
        mat2[i] = i;
        mat2e[i] = 0;
    }
    /*
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_CPU_Mat results:" << endl;
        for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << mat1[i];
            cout << " ";
        }
        cout << endl;
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            cout << mat2[i];
            cout << " ";
        }
        cout << endl;
    }
    */
    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < MAT_SIZE1; i++) {
            mat1[i] = i;
            mat1e[i] = 0;
        }
        for (int i = 0; i < MAT_SIZE2; i++) {
            mat2[i] = i;
            mat2e[i] = 0;
        }

#pragma omp parallel for
        /* Normal Matrix multiplication
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < m; j++) {
                float sum = 0.0;
                for (int k = 0; k < nCols; k++) {
                    sum += mat1[i*nCols + k] * mat2[k*m + j];
                }
                mat1[MAT_SIZE1/2 + i*nCols + j] = sum;
            }
        }
        */
        /* Transposed Matrix Multiplication */
        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < m; j++) {
                float sum = 0.0;
                for (int k = 0; k < nCols; k++) {
                    sum += mat1[i*nCols + k] * mat2[j*m + k];
                }
                mat1[MAT_SIZE1/2 + i*nCols + j] = sum;
            }
        }
    }

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_CPU_Mat runtime: " << runtime/REPS << endl;

    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_CPU_Mat results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 20; i++)
        //for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << mat1[i + MAT_SIZE1/2 - 10];
            //cout << mat1[i];
            //cout << "x10^";
            //cout << exponents[i];
            cout << " ";
        }
        cout << endl;
        /*
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            cout << mat2[i];
            //cout << "x10^";
            //cout << exponents[i];
            cout << " ";
        }
        cout << endl;
        */
    }
#endif

#ifdef FLOAT_ZERO_THREE_MAT_TILED
    runtime = 0;

    gettimeofday(&t1, NULL);

    floatZeroDataset4 = new float[MAT_SIZE1];
    floatZeroDataset4b = new float[MAT_SIZE2];
    floatZeroExponents4 = new int[MAT_SIZE1];
    floatZeroExponents4b = new int[MAT_SIZE2];

    size1 = sizeof(float) * MAT_SIZE1;
    size_e1 = sizeof(cl_int) * MAT_SIZE1;
    size2 = sizeof(float) * MAT_SIZE2;
    size_e2 = sizeof(cl_int) * MAT_SIZE2;

    kernel_name = "floatMatTiledTest";
    compile_kernel(kernel_name, float_mat_tiled_kernel);

    create_buffer_zero(d_buf1, size1);
    create_buffer_zero(d_buf2, size_e1);
    create_buffer_zero(d_buf1b, size2);
    create_buffer_zero(d_buf2b, size_e2);

    for (int iter = 0; iter < REPS; iter++) {
        for (int i = 0; i < MAT_SIZE1; i++)
        {
            floatZeroDataset4[i] = i;
            floatZeroExponents4[i] = 0;
        }
        for (int i = 0; i < MAT_SIZE2; i++)
        {
            floatZeroDataset4b[i] = i;
            floatZeroExponents4b[i] = 0;
        }
        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
        memcpy(mapPtrA, floatZeroDataset4, size1);
        memcpy(mapPtrB, floatZeroExponents4, size_e1);
        memcpy(mapPtrAb, floatZeroDataset4b, size2);
        memcpy(mapPtrBb, floatZeroExponents4b, size_e2);
        clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
        clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

        int temp_block_number = 4;
        int temp_block_size = 16;

        err_num  = clSetKernelArg(float_mat_tiled_kernel, 0, sizeof(cl_mem), (void *) &d_buf1);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 1, sizeof(cl_mem), (void *) &d_buf2);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 2, sizeof(cl_mem), (void *) &d_buf1b);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 3, sizeof(cl_mem), (void *) &d_buf2b);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 4, sizeof(cl_int), (void *) &temp_block_number);
        err_num  |= clSetKernelArg(float_mat_tiled_kernel, 5, sizeof(cl_int), (void *) &temp_block_size);
        if (err_num != CL_SUCCESS)
        {
            cout << "kernel arg set fail" << endl;
            exit(err_num);
        }

        launch_kernel(float_mat_tiled_kernel);

        mapPtrA = (float*)map_buffer_zero(d_buf1, size1);
        mapPtrB = (float*)map_buffer_zero(d_buf2, size_e1);
        mapPtrAb = (float*)map_buffer_zero(d_buf1b, size2);
        mapPtrBb = (float*)map_buffer_zero(d_buf2b, size_e2);
    }
    if (VERBOSITY_LEVEL > 1)
    {
        cout << "Float_Zero_Three_Mat_Tiled results:" << endl;
        //for (int i = 0; i < BUF_SIZE; i++)
        for (int i = 0; i < 20; i++)
        //for (int i = 0; i < MAT_SIZE1; i++)
        {
            cout << ((float*)mapPtrA)[i + MAT_SIZE1/2 - 10];
            //cout << ((float*)mapPtrA)[i];
            //cout << "x10^";
            //cout << ((int*)mapPtrB)[i + MAT_SIZE1/2];
            cout << " ";
        }
        cout << endl;
    }

    clEnqueueUnmapMemObject(queue, d_buf1, mapPtrA, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2, mapPtrB, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf1b, mapPtrAb, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queue, d_buf2b, mapPtrBb, 0, NULL, NULL);

    gettimeofday(&t2, NULL);
    runtime += (t2.tv_sec -t1.tv_sec) * 1000.0;
    runtime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    cout << "Float_Zero_Three_Mat_Tiled runtime: " << runtime/REPS << endl;
#endif
    return 0;
}

void* map_buffer_zero(cl_mem d_buf, size_t size)
{
    void* temp = clEnqueueMapBuffer(  queue,
                                d_buf,
                                CL_TRUE,
                                CL_MAP_READ | CL_MAP_WRITE,
                                0,
                                size,
                                0,
                                NULL,
                                NULL,
                                NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "map buffer fail" << endl;
        exit(err_num);
    }
    return temp;
}

int unmap_buffer_zero(cl_mem &d_buf, void* data)
{
     err_num = clEnqueueUnmapMemObject( queue,
                                d_buf,
                                data,
                                0,
                                NULL,
                                NULL);
    if (err_num != CL_SUCCESS)
    {
        cout << "unmap buffer fail" << endl;
        exit(err_num);
    }
    return 0;
}

int create_buffer_zero(cl_mem &d_buf, size_t size)
{
    d_buf = clCreateBuffer( ctx,
                            CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR,
                            size,
                            NULL,
                            &err_num);
    if (err_num != CL_SUCCESS)
    {
        cout << "make buffer fail" << endl;
        exit(err_num);
    }
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

    device = devices[1]; // XXX set back down to 0
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
    //printf("Connecting to %s %s...\n", vendor_name, device_name);
    cout << "Connecting to " << vendor_name << " " << device_name << "..." << endl;
    //printf("Max work group size: %"PRIuPTR"\n", wg_max);
    cout << "Max work group size: " << wg_max << endl;
    //local_work_size = wg_max;
    local_work_size = 256;

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
        //printf("%s\n", log);
        cout << log << endl;
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
        cout << err_num << endl;
        exit(err_num);
    }
    return 0;
}
