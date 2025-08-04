#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace std::chrono;

constexpr size_t DATA_SIZE = 1e9;

sycl::event compute(sycl::queue& q, double* device_buf, sycl::event dep_event) {
  // Simple kernel that just wastes some time. In reality could be blocks of a large matrix multiplication
    return q.submit([&](sycl::handler& h) {
        h.depends_on(dep_event);
        h.parallel_for(sycl::range<1>{DATA_SIZE}, [=](sycl::id<1> i) {
            for (int j = 0; j < 1e4; j++)
              device_buf[i] += 1.0f;
        });
    });
}

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    double* host_buf = sycl::malloc_host<double>(DATA_SIZE, q);
    double* gpu_buf_1 = sycl::malloc_device<double>(DATA_SIZE, q);
    double* gpu_buf_2 = sycl::malloc_device<double>(DATA_SIZE, q);

    for (size_t i = 0; i < DATA_SIZE; ++i)
        host_buf[i] = static_cast<double>(i);

    auto t0 = high_resolution_clock::now();

    // First async copy and compute using gpu_buf_1
    auto copy1_start = high_resolution_clock::now();
    sycl::event copy_event1 = q.memcpy(gpu_buf_1, host_buf, sizeof(double) * DATA_SIZE);
    auto compute_event1 = compute(q, gpu_buf_1, copy_event1);
    copy_event1.wait();
    auto copy1_end = high_resolution_clock::now();

    // Overwrite host_buf with some new "data"
    for (size_t i = 0; i < DATA_SIZE; i+=512)
        host_buf[i] *= 2.0;

    // Uncomment to remove copy<->compute parlellism
    //compute_event1.wait();
    
    // Second async copy and compute using gpu_buf_2
    auto copy2_start = high_resolution_clock::now();
    sycl::event copy_event2 = q.memcpy(gpu_buf_2, host_buf, sizeof(double) * DATA_SIZE);
    auto compute_event2 = compute(q, gpu_buf_2, copy_event2);
    copy_event2.wait();
    auto copy2_end = high_resolution_clock::now();

    compute_event1.wait();
    compute_event2.wait();
    auto total_end = high_resolution_clock::now();

    std::cout << "\n--- Timing Results ---\n";
    std::cout << "Copy 1: "
              << duration_cast<milliseconds>(copy1_end - copy1_start).count() << " ms\n";
    std::cout << "CPU stuff: "
              << duration_cast<milliseconds>(copy2_start - copy1_end).count() << " ms\n";
    std::cout << "Copy 2: "
              << duration_cast<milliseconds>(copy2_end - copy2_start).count() << " ms\n";
    std::cout << "Total execution: "
              << duration_cast<milliseconds>(total_end - t0).count() << " ms\n";

    // Cleanup
    sycl::free(host_buf, q);
    sycl::free(gpu_buf_1, q);
    sycl::free(gpu_buf_2, q);

    return 0;
}

