/*
 Please include compiler name below (you may also include any other modules you would like to be loaded)

 COMPILER= gnu

 Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines

 CC = cc
 OPT = -O3
 CFLAGS = -Wall -std=gnu99 $(OPT)
 MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
 LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
 wagner ayllon
 */

const char* dgemm_desc = "Simple blocked dgemm.";
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 1024
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
            /* Compute C(i,j) */
            double cij = C[i+j*lda];
            for (int k = 0; k < K; ++k)
                cij += A[i+(k*lda)] * B[k+j*lda];
            C[i+j*lda] = cij;
        }
}

void do_block_fast(int lda, int M, int N, int K, double *A, double *B, double *C) {
    static double a[BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (16))); // make a local aligned copy of A's block
    for (int j = 0; j < K; j++)
        for (int i = 0; i < M; i++)
            a[i + j * BLOCK_SIZE] = A[i + j * lda];

    /* For each row i of A */
    for (int i = 0; i < M; ++i) /* For each column j of B */
        for (int j = 0; j < N; ++j) {
/* Compute C(i,j) */
            double cij = C[i + j * lda];
            for (int k = 0; k < K; k+= 2) {
                cij += a[i+k*BLOCK_SIZE] * B[k+j*lda];
                cij += a[i+(k+1)*BLOCK_SIZE] * B[(k+1)+j*lda];
            }
            C[i + j * lda] = cij;
        }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
//void do_block_fast2(int lda, int M, int N, int K, double *A, double *B, double *C) {


    /* For each block-row of A */
//copy A here
    static double a[
            BLOCK_SIZE * BLOCK_SIZE] __attribute__ ((aligned (16))); // make a local aligned copy of A's block


    for (int i = 0; i < lda; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
            /* Accumulate block dgemms into block of C */

            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE, lda - i);
                int N = min (BLOCK_SIZE, lda - j);
                int K = min (BLOCK_SIZE, lda - k);

                double a1, a2, b1, b2, a3, a4, b3, b4;
                double a;
                double c1, c3, c4, cij;

                c1 = a1 * b1;
                double  c2 = a2 * b2;
                c3 = a3 * b3;
                c4 = a4 * b4;
                cij += c1;
                cij += c2;
                cij += c3;
                cij += c4;

                //         a1 = a[i + k * BLOCK_SIZE];
                //         a2 = a[i + (k + 1) * BLOCK_SIZE];
                //         a3 = a[i + (k + 2) * BLOCK_SIZE];
                //         a4 = a[i + (k + 3) * BLOCK_SIZE];

                //        b1 = B[k + j * lda];;
                //        b2 = B[(k + 1) + j * lda];
                //        b3 = B[(k + 2) + j * lda];
                //        b4 = B[(k + 3) + j * lda];

                /* Perform individual block dgemm */
                if ((M % BLOCK_SIZE == 0) && (N % BLOCK_SIZE == 0) && (K % BLOCK_SIZE == 0)) {
                    do_block_fast(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                } else {
                    do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
                }
            }
}


/*

void square_dgemm (int n, double* A, double* B, double* C)
    {
        // For each row i of A
        for (int i = 0; i < n; ++i)
            // For each column j of B
            for (int j = 0; j < n; ++j)
            {
                // Compute C(i,j)
                double cij = C[i+j*n];
                for( int k = 0; k < n; k++ )
                    cij += A[i+k*n] * B[k+j*n];
                C[i+j*n] = cij;

                double* pa = &A[i];
                double* pb = &B[j*n];
                double* pc = &C[i+j*n];
                for( int k = 0; k < n; k++ )
                {
                    *pc += *pa++ * *pb;
                    pb += n;
                }
            }
    }

*/
