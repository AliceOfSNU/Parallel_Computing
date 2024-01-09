#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);
void sgemm(int M, int N, int K, const float* A, const float* B, float* C);
void printCudaInfo();


// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}


int main(int argc, char** argv)
{

    int N = 20 * 1000 * 1000;

    // parse commandline options ////////////////////////////////////////////
    int opt;
    std::string task;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {"task",    1,  0, 'm'},
        {0 ,0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "m:?n:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'n':
            N = atoi(optarg);
            break;
        case 'm':
            task = optarg;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////
    if(task.compare("vec") == 0){

        const float alpha = 2.0f;
        float* xarray = new float[N];
        float* yarray = new float[N];
        float* resultarray = new float[N];

        // load X, Y, store result
        int totalBytes = sizeof(float) * 3 * N;

        for (int i=0; i<N; i++) {
            xarray[i] = yarray[i] = i % 10;
            resultarray[i] = 0.f;
        }

        printCudaInfo();

        for (int i=0; i<3; i++) {
        saxpyCuda(N, alpha, xarray, yarray, resultarray);
        }

        delete [] xarray;
        delete [] yarray;
        delete [] resultarray;

    } else if (task.compare("gemm") == 0){
        int N = 256, M = 256, K = 16;
        float* A = new float[M * K];
        float* B = new float[K * N];
        float* C = new float[M * N];
        float* C_ref = new float[M * N]; //cpu

        //populate A
        for(int y = 0; y < M; y += K){
            for(int x = 0; x < K; ++x){
                A[y + x] = x % 2;
            }
        }

        //populate A
        for(int y = 0; y < K; y += N){
            for(int x = 0; x < N; ++x){
                A[y + x] = x % 2;
            }
        }

        //multiply
        for(int y = 0; y < M; ++y){
            for(int x = 0; x < N; ++x){
                C_ref[y * N + x] = .0f;
                for(int k = 0; k < K; ++k){
                    C_ref[y * N + x] += A[y*K +k] * B[k*N + x];
                }
            }
        }

        //gpu impl
        sgemm(M, N, K, A, B, C);

        // check correctness
        for(int y = 0; y < M; ++y){
            for(int x = 0; x < N; ++x){
                if(C[y * N + x] != C_ref[y * N + x]){
                    printf("C[%d][%d] expected %f, but got %f \n", y, x, C_ref[y * N + x], C[y * N + x]);
                    exit(1);
                }
            }
        }

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_ref;
    }
    return 0;
}
