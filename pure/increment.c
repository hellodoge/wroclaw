#include <stdio.h>
#include <stdlib.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

const char *program_name = "inc";
const char *program_code = _CL_STRINGIFY(
    __kernel void inc(int in, __global int *out) {
        *out = in + 1;
    }
);

typedef cl_int result_t;

void assert_success(result_t, const char *operation);

// WARNING
// OpenCL entities partitioning as shown here is not recommended.
// I'm just experimenting with different segregations, and what problems they cause. 

typedef struct {
    cl_platform_id platforms;
    cl_device_id devices;
    cl_uint num_platforms;
    cl_uint num_devices;
} device_t;

typedef struct {
    cl_context context;
    cl_command_queue queue;
} context_t;

typedef struct {
    cl_program program;
    cl_kernel kernel;
} kernel_t;

typedef struct {
    void *arg;
    size_t size;
} argument_t;

// facade that picks one of any available devices
result_t get_device(device_t *);

result_t get_context(context_t *, device_t *);
result_t get_kernel(kernel_t *k, context_t *c, device_t *d, 
                    const char *name, const char *code);

result_t release_device(device_t *);
result_t release_context(context_t *);
result_t release_kernel(kernel_t *);

result_t enqueue_kernel(context_t *c, kernel_t *k, size_t args_count, argument_t args[]);


int main() {
    result_t err;

    device_t device = {};
    err = get_device(&device);
    assert_success(err, "getting device");

    context_t context = {};
    err = get_context(&context, &device);
    assert_success(err, "creating context");

    kernel_t kernel = {};
    err = get_kernel(&kernel, &context, &device, program_name, program_code);
    assert_success(err, "building kernel");

    cl_int input = 123;
    cl_mem output = clCreateBuffer(context.context, CL_MEM_READ_WRITE, sizeof(cl_int),
                                   NULL /* no host-specified initial value */, &err);
    assert_success(err, "creating output buffer");
    
    err = enqueue_kernel(&context, &kernel, 2 /* number of args */, (argument_t[]){
        (argument_t){.arg=&input, .size=sizeof(input)},
        (argument_t){.arg=&output, .size=sizeof(output)},
    });
    assert_success(err, "enqueuing kernel");

    err = clFinish(context.queue);

    cl_int result;
    const cl_bool blocking_read = CL_TRUE;
    clEnqueueReadBuffer(context.queue, output, blocking_read, 0 /* offset */,
                        sizeof(result), &result, /* last 3 arguments are 
                        responsible of order of kernels execution */ 0, NULL, NULL);

    printf("%d + 1 is %d\n", input, (int)result);

    assert_success(clReleaseMemObject(output), "release buffer");
    assert_success(release_kernel(&kernel), "release kernel");
    assert_success(release_context(&context), "release context");
    assert_success(release_device(&device), "release device");
}

// for such a simple program it does not matter which device we'll pick
result_t get_device(device_t *d) {
    result_t err;

    err = clGetPlatformIDs(1 /* max number of platforms to find */, 
                           &d->platforms, &d->num_platforms);
    if (err != CL_SUCCESS)
        return err;

    err = clGetDeviceIDs(d->platforms, CL_DEVICE_TYPE_ALL,
                         1 /* pick one available device */,
                         &d->devices, &d->num_devices);
    return err;
}

result_t get_context(context_t *c, device_t *d) {
    result_t err;

    c->context = clCreateContext(NULL /* let the driver pick platform itself*/,
                                 d->num_devices, &d->devices, NULL, NULL /* previous two
                                 arguments specify async callback for errors reporting */,
                                 &err);
    if (err != CL_SUCCESS)
        return err;
    
    c->queue = clCreateCommandQueueWithProperties(c->context, d->devices, 
                                                  NULL /* default properties */, &err);
    return err;
}

result_t get_kernel(kernel_t *k, context_t *c, device_t *d, const char *name, const char *code) {
    result_t err;

    k->program = clCreateProgramWithSource(c->context, 1 /* code is one string */,
                                           &code, NULL /* code is null terminated*/, &err);
    if (err != CL_SUCCESS)
        return err;

    err = clBuildProgram(k->program, d->num_devices, &d->devices, "" /* default options*/, NULL, NULL 
                         /* last two arguments specify async callback for errors reporting */);

    k->kernel = clCreateKernel(k->program, name, &err);
    return err;
}

result_t enqueue_kernel(context_t *c, kernel_t *k, size_t args_count, argument_t args[]) {
    result_t err;

    for (int i = 0; i < args_count; i++) {
        err = clSetKernelArg(k->kernel, i, args[i].size, args[i].arg);
        if (err != CL_SUCCESS)
            return err;
    }

    /* clEnqueueTask is deprecated, see Notes section at
     * https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueTask.html */
    const size_t global_work_size = 1;
    const size_t local_work_size = 1;
    err = clEnqueueNDRangeKernel(c->queue, k->kernel, 1, NULL, &global_work_size, 
                                 &local_work_size, 0, NULL /* last two arguments point that we
                                 should not wait for any events to complete before executing 
                                 the kernel */, NULL /* out event variable*/);
    if (err != CL_SUCCESS)
        return err;

    return clFlush(c->queue);
}

result_t release_device(device_t *d) {
    return clReleaseDevice(d->devices);
}

result_t release_context(context_t *c) {
    // warning: err2 result might be lost
    result_t err1 = clReleaseCommandQueue(c->queue);
    result_t err2 = clReleaseContext(c->context);
    if (err1 != CL_SUCCESS)
        return err1;
    return err2;
}

result_t release_kernel(kernel_t *k) {
    // warning: err2 result might be lost
    result_t err1 = clReleaseProgram(k->program);
    result_t err2 = clReleaseKernel(k->kernel);
    if (err1 != CL_SUCCESS)
        return err1;
    return err2;
}


void assert_success(result_t err, const char *operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s failed with code %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}