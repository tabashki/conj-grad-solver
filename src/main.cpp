#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <string_view>

#include "utils.h"
#include "serial.h"
#include "parallel.h"

using std::cout;
using std::endl;
using std::vector;

void print_help()
{
    cout << "conjgrad [-s|-p] <INFILE> <OUTFILE>" << endl
        << endl
        << "\tINFILE:\tList of newline separated float values for the `b` vector" << endl
        << endl
        << "\t-s\tExecute serial version of the algorithm" << endl
        << endl
        << "\t-p\tExecute parallel version of the algorithm" << endl;
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        cout << "Insufficient number of arguments." << endl << endl;
        print_help();
        return 1;
    }

    bool parallel = false;
    if (std::string(argv[1]) == "-p")
    {
        parallel = true;
    }

    std::ifstream infile(argv[2]);
    if (!infile)
    {
        cout << "Failed to open file: " << argv[2] << endl << endl;
        print_help();
        return 1;
    }

    std::ofstream outfile(argv[3]);
    if (!outfile)
    {
        cout << "Failed to open file: " << argv[3] << endl << endl;
        print_help();
        return 1;
    }

    std::vector<float> b_vec;
    read_from_file(infile, b_vec);

    const unsigned int N = b_vec.size();
    constexpr float threshold = 1e-6f;
    std::vector<float> result;
    result.reserve(N);

    constexpr int max_iters = 100000;
    int64_t start_time = 0;
    int64_t end_time = 0;

    if (parallel)
    {
        start_time = timestamp_ns();
        bool success = parallel_conj_grad(b_vec, result, threshold, max_iters);
        end_time = timestamp_ns();

        if (!success)
        {
            cout << "Failed to calculate parallel result for threshold: " << threshold << endl;
            return 2;
        }   
    }
    else
    {
        start_time = timestamp_ns();
        bool success = serial_conj_grad(b_vec, result, threshold, max_iters);
        end_time = timestamp_ns();

        if (!success)
        {
            cout << "Failed to calculate serial result for threshold: " << threshold << endl;
            return 2;
        }    
    }

    write_to_file(outfile, result);

    double duration_ms = (end_time - start_time) * 1e-6;
    cout << "Successfully calculated solution in " << duration_ms << " ms" << endl;
    return 0;
}