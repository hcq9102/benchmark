#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/execution_base/this_thread.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>

#include <hpx/config.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/datapar.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/parallel/algorithms/rotate.hpp>


#include <random>
#include <utility>
#include <string>
#include <type_traits>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstddef>
#include <chrono>
#include <iostream>

struct gen_float_t{
    std::mt19937 mersenne_engine {42};
    std::uniform_real_distribution<float> dist_float {1, 1024};
    auto operator()()
    {
        return dist_float(mersenne_engine);
   }
} gen_float{};

std::size_t mid = 0;

template <typename ExPolicy> 
double rotate(ExPolicy policy, std::size_t n)
{
    using allocator_type = hpx::compute::host::block_allocator<float>;
    using executor_type = hpx::compute::host::block_executor<>;
    
    auto numa_domains = hpx::compute::host::numa_domains();
    allocator_type alloc(numa_domains);
    executor_type executor(numa_domains);

    hpx::compute::vector<float, allocator_type> c(n, 0.0, alloc);
    // generate numbers according to gen() function
    if constexpr (hpx::is_parallel_execution_policy_v<std::decay_t<ExPolicy>>){
        hpx::generate(hpx::execution::par, c.begin(), c.end(), gen_float_t{});
    }
    else
    {
        hpx::generate(hpx::execution::seq, c.begin(), c.end(), gen_float_t{});
    }
 
    // count time begin
    auto t1 = std::chrono::high_resolution_clock::now();
    hpx::rotate(policy, c.begin(), c.begin()+ mid, c.end());
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

template <typename ExPolicy>
auto test(ExPolicy policy, std::size_t iterations, std::size_t n)
{
    double avg_time = 0.0;
    for (std::size_t i = 0; i < iterations; i++)
    {
        avg_time += rotate<ExPolicy>(policy, n);
    }
    avg_time /= (double) iterations;
    return avg_time;
}


int hpx_main()
{
    auto& seq_pol = hpx::execution::seq;
    auto& par_pol = hpx::execution::par;
    //auto par_sr_pol = hpx::execution::par.on(exec);
   // auto par_task_sr_pol = hpx::execution::par(task).on(exec);

    std::cout << "new benchmark rotate test \n";
    std::size_t threadsNum = hpx::get_os_thread_count();
    std::cout << "Threads : " << threadsNum << '\n';
    std::ofstream fout("01rotate_newbench_medusa01_threads=40.csv");

    fout << "threadsNum = " << threadsNum <<'\n';
    //fout << "n,i,seq,par,par_SR,par_task_SR,seq/par, seq/par_SR, seq/par_task_SR \n";
    fout << "n,i,seq,par \n";
    for (std::size_t i = 6; i <= 28; i++)
    {
        std::size_t n = std::pow(2, i);
        mid = n/8;
        double SEQ = test(seq_pol, 10, n);
        double PAR = test(par_pol, 10, n);
        
        #if defined(OUTPUT_TO_CSV)
        // std::cout << "N : " << i << '\n';
        // std::cout << "SEQ: " << seq << '\n';
        // std::cout << "PAR: " << par << "\n\n";
        #endif
        
        fout << n << ","
            << i << "," 
            << SEQ << ","
            << PAR  << "," << "\n";
        fout.flush();    
    }
    fout.close();
    return hpx::finalize();

}

int main(int argc, char *argv[]) {
    namespace po = hpx::program_options;

    po::options_description desc_commandline;
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args); 
  //return hpx::local::init(hpx_main, argc, argv);
}
