//#include <hpx/hpx_main.hpp>
#include <hpx/modules/algorithms.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>
//#include <hpx/execution/executors/num_cores.hpp>
#include <hpx/parallel/algorithms/rotate.hpp>

#include <random>
#include <chrono>
#include <array>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <numeric>

//#define OUTPUT_TO_CSV
enum class ALGORITHM
{
    ROTATE,
    ROTATE_COPY
};

void measureRotateAlgorithms()
{
//#if defined(OUTPUT_TO_CSV)
    std::map<ALGORITHM, std::string> filenames = {
        {ALGORITHM::ROTATE, "origin_rotate_mid_8_threads=40.csv"},
        {ALGORITHM::ROTATE_COPY, "origin_rotateCopy_mid_8_threads=40.csv"},
    };
//#endif

    for (int alg = (int) ALGORITHM::ROTATE;
         alg <= (int) ALGORITHM::ROTATE_COPY; alg++)
    {
        std::size_t start = 32;
        std::size_t till = 1 << 29;

        const auto NUM_ITERATIONS = 5;

        std::vector<std::array<double, 3>> data;

        for (std::size_t s = start; s <= till; s *= 2)
        {
            std::vector<int> arr(s);
            std::iota(std::begin(arr), std::end(arr), 1);

	    std::size_t mid = s/8;

            double seqTime = 0;
            double parTime = 0;

            for (int i = 0; i < NUM_ITERATIONS + 5; i++)
            {
                std::vector<int> res(s);
                auto t1 = std::chrono::high_resolution_clock::now();

                switch ((ALGORITHM) alg)
                {
                    case ALGORITHM::ROTATE:
                        hpx::rotate(hpx::execution::seq, arr.begin(), arr.begin()+ mid, arr.end());
                        break;

                    case ALGORITHM::ROTATE_COPY:
                        hpx::rotate_copy(hpx::execution::seq, arr.begin(),arr.begin()+ mid, arr.end(), res.begin());
                        break;
                };
                auto end1 = std::chrono::high_resolution_clock::now();

                // don't consider first 5 iterations
                if (NUM_ITERATIONS < 5)
                {
                    continue;
                }

                std::chrono::duration<double> time_span1 =
                        std::chrono::duration_cast<std::chrono::duration<double>>(end1 - t1);

                seqTime += time_span1.count();
            }

            for (int i = 0; i < NUM_ITERATIONS + 5; i++)
            {
                std::vector<int> res1(s);
                auto t2 = std::chrono::high_resolution_clock::now();
                switch ((ALGORITHM) alg)
                {
                    case ALGORITHM::ROTATE:
                        hpx::rotate(hpx::execution::par, arr.begin(),arr.begin()+ mid, arr.end());
                        break;

                    case ALGORITHM::ROTATE_COPY:
                        hpx::rotate_copy(hpx::execution::par, arr.begin(),arr.begin()+ mid, arr.end(), res1.begin());
                        break;
                };
                auto end2 = std::chrono::high_resolution_clock::now();

                // don't consider first 5 iterations
                if (NUM_ITERATIONS < 5)
                {
                    continue;
                }

                std::chrono::duration<double> time_span2 =
                        std::chrono::duration_cast<std::chrono::duration<double>>(end2 - t2);

                parTime += time_span2.count();
            }

            seqTime /= NUM_ITERATIONS;
            parTime /= NUM_ITERATIONS;

//#if defined(OUTPUT_TO_CSV)
            data.push_back(std::array<double, 3>{(double) s, seqTime, parTime});
#if defined(OUTPUT_TO_CSV)
            std::cout << "N : " << s << '\n';
            std::cout << "SEQ: " << seqTime << '\n';
            std::cout << "PAR: " << parTime << "\n\n";
#endif
        }

//#if defined(OUTPUT_TO_CSV)
        std::ofstream outputFile(filenames[(ALGORITHM) alg]);
        for (auto& d : data)
        {
            outputFile << d[0] << "," << d[1] << "," << d[2] << ","
                       << ",\n";
        }
//#endif
    }
}

int hpx_main(hpx::program_options::variables_map&)
{
    measureRotateAlgorithms();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    // Initialize and run HPX.
   HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
