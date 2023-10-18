#ifndef NCL_H
#define NCL_H

#include <stddef.h>
#include <math.h>

#ifndef NCL_MALLOC
# include <stdlib.h>
# define NCL_MALLOC malloc
# ifndef NCL_FREE
#  define NCL_FREE free
# endif
#endif // NCL_MALLOC

#ifndef NCL_ASSERT
# include <assert.h>
# define NCL_ASSERT assert
#endif // NCL_ASSERT

#ifndef NCL_PRINT
# include <stdio.h>
# define NCL_PRINT printf
#endif // NCL_PRINT

#ifndef NCL_RAND
# include <stdlib.h>
# define NCL_RAND rand
# ifndef NCL_RAND_MAX
#  define NCL_RAND_MAX RAND_MAX
# endif // NCL_RAND_MAX
#endif // NCL_RAND

struct Matrix {
        size_t rows;
        size_t cols;
        size_t stride;
        float *arr;
};

#define MATRIX_AT(m, i, j) ((m).arr[(i)*(m).stride+(j)])
#define MATRIX_PRINT(m)    Matrix_print(m, #m, 0)

struct Matrix Matrix_create(size_t rows, size_t cols);
void          Matrix_copy(struct Matrix dst, struct Matrix src);
struct Matrix Matrix_row(struct Matrix mat, size_t row);
void          Matrix_fill(struct Matrix mat, float value);
void          Matrix_rand(struct Matrix mat, float low, float high);
void          Matrix_dot(struct Matrix dst, struct Matrix a, struct Matrix b);
void          Matrix_sum(struct Matrix dst, struct Matrix a);
void          Matrix_sig(struct Matrix mat);
void          Matrix_print(struct Matrix mat, const char *name, int padding);

struct NeuNet {
        size_t count;
        struct Matrix *as;
        struct Matrix *ws;
        struct Matrix *bs;
};

#define NEUNET_INPUT(nn)  ((nn).as[0])
#define NEUNET_OUTPUT(nn) ((nn).as[(nn).count])
#define NEUNET_PRINT(m)   NeuNet_print(m, #m)

struct NeuNet NeuNet_create(size_t *arch, size_t arch_size);
void          NeuNet_clear(struct NeuNet nn);
void          NeuNet_rand(struct NeuNet nn, float low, float high);
void          NeuNet_forward(struct NeuNet nn);
float         NeuNet_cost(struct NeuNet nn, struct Matrix ti, struct Matrix to);
void          NeuNet_fiiff(struct NeuNet nn, struct NeuNet g, float eps, struct Matrix ti, struct Matrix to);
void          NeuNet_learn(struct NeuNet nn, struct NeuNet g, float rate);
void          NeuNet_backprop(struct NeuNet nn, struct NeuNet g, struct Matrix ti, struct Matrix to);
void          NeuNet_print(struct NeuNet mat, const char *name);

#endif // NCL_H


/* after this it's the function implementations,
 * please don't use function that are not in the previous part
 * they have been created specifically for the framework
 * so they can lead to undefined behaviour
 */


// #define NCL_IMPLEMENTATION
#ifdef NCL_IMPLEMENTATION


struct Matrix Matrix_create(size_t rows, size_t cols) {
        struct Matrix m = (struct Matrix) {
                .rows = rows,
                        .cols = cols,
                        .stride = cols,
                        .arr  = (float *)NCL_MALLOC(sizeof(*m.arr)*rows*cols),
        };
        NCL_ASSERT(m.arr != NULL);
        return m;
}

void Matrix_copy(struct Matrix dst, struct Matrix src) {
        NCL_ASSERT(dst.rows == src.rows);
        NCL_ASSERT(dst.cols == src.cols);
        for(size_t i = 0; i < dst.rows; i++) {
                for(size_t j = 0; j < dst.cols; j++) {
                        MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
                }
        }
}

struct Matrix Matrix_row(struct Matrix mat, size_t row) {
        return (struct Matrix){
                .rows   = 1,
                        .cols   = mat.cols,
                        .stride = mat.stride,
                        .arr    = &MATRIX_AT(mat, row, 0),
        };
}

void Matrix_fill(struct Matrix mat, float value) {
        for (size_t i = 0; i < mat.rows; i++) {
                for (size_t j = 0; j < mat.cols; j++) {
                        MATRIX_AT(mat, i, j) = value;
                }
        }
}

float randFloat() {
        return (float)NCL_RAND() / (float)NCL_RAND_MAX;
}

void Matrix_rand(struct Matrix mat, float low, float high) {
        for (size_t i = 0; i < mat.rows; i++) {
                for (size_t j = 0; j < mat.cols; j++) {
                        MATRIX_AT(mat, i, j) = randFloat() * (high - low) + low;
                }
        }
}

void Matrix_dot(struct Matrix dst, struct Matrix a, struct Matrix b) {
        NCL_ASSERT(a.cols == b.rows);
        NCL_ASSERT(dst.rows == a.rows);
        NCL_ASSERT(dst.cols == b.cols);
        for (size_t i = 0; i < dst.rows; i++) {
                for (size_t j = 0; j < dst.cols; j++) {
                        MATRIX_AT(dst, i, j) = 0;
                        for (size_t k = 0; k < a.cols; k++) MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j);
                }
        }
}

void Matrix_sum(struct Matrix dst, struct Matrix a) {
        NCL_ASSERT(dst.rows == a.rows);
        NCL_ASSERT(dst.cols == a.cols);
        for (size_t i = 0; i < dst.rows; i++) {
                for (size_t j = 0; j < dst.cols; j++) {
                        MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
                }
        }
}

float sigf(float x) {
        return 1.0f/(1.0f+expf(-x));
}

void Matrix_sig(struct Matrix mat) {
        for (size_t i = 0; i < mat.rows; i++) {
                for (size_t j = 0; j < mat.cols; j++) {
                        MATRIX_AT(mat, i, j) = sigf(MATRIX_AT(mat, i, j));
                }
        }

}

void Matrix_print(struct Matrix mat, const char *name, int padding) {
        NCL_PRINT("%*s%s = [\n", padding, "", name);
        for (size_t i = 0; i < mat.rows; i++) {
                NCL_PRINT("%*s", padding + 8, "");
                for (size_t j = 0; j < mat.cols; j++) {
                        NCL_PRINT(" %f", MATRIX_AT(mat, i, j));
                }
                NCL_PRINT("\n");
        }
        NCL_PRINT("%*s]\n", padding, "");
}

struct NeuNet NeuNet_create(size_t *arch, size_t arch_size) {
        NCL_ASSERT(arch_size > 0);

        struct NeuNet ret;

        ret.count = arch_size - 1;
        ret.ws = (struct Matrix*)NCL_MALLOC(sizeof(*ret.ws) * ret.count);
        NCL_ASSERT(ret.ws != NULL);
        ret.bs = (struct Matrix*)NCL_MALLOC(sizeof(*ret.bs) * ret.count);
        NCL_ASSERT(ret.bs != NULL);
        ret.as = (struct Matrix*)NCL_MALLOC(sizeof(*ret.as) * (ret.count + 1));
        NCL_ASSERT(ret.as != NULL);

        ret.as[0] = Matrix_create(1, arch[0]);

        for(size_t i = 1; i < arch_size; i++) {
                ret.ws[i - 1] = Matrix_create(ret.as[i - 1].cols, arch[i]);
                ret.bs[i - 1] = Matrix_create(1, arch[i]);
                ret.as[i]     = Matrix_create(1, arch[i]);
        }

        return ret;
}

void NeuNet_clear(struct NeuNet nn) {
        for(size_t i = 0; i < nn.count; i++) {
                Matrix_fill(nn.ws[i], 0);
                Matrix_fill(nn.bs[i], 0);
                Matrix_fill(nn.as[i], 0);
        }
        Matrix_fill(nn.as[nn.count], 0);
}

void NeuNet_rand(struct NeuNet nn, float low, float high) {
        for(size_t i = 0; i < nn.count; i++) {
                Matrix_rand(nn.ws[i], low, high);
                Matrix_rand(nn.bs[i], low, high);
        }
}

void NeuNet_forward(struct NeuNet nn) {
        for(size_t i = 0; i < nn.count; i++) {
                Matrix_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
                Matrix_sum(nn.as[i + 1], nn.bs[i]);
                Matrix_sig(nn.as[i + 1]);
        }
}

float NeuNet_cost(struct NeuNet nn, struct Matrix ti, struct Matrix to) {
        NCL_ASSERT(ti.rows == to.rows);
        NCL_ASSERT(to.cols == NEUNET_OUTPUT(nn).cols);
        float c = 0;
        for(size_t i = 0; i < ti.rows; i++) {
                struct Matrix x = Matrix_row(ti, i);
                struct Matrix y = Matrix_row(to, i);

                Matrix_copy(NEUNET_INPUT(nn), x);
                NeuNet_forward(nn);

                size_t q = to.cols;
                struct Matrix out = NEUNET_OUTPUT(nn);
                for(size_t j = 0; j < q; j++) {
                        float d = MATRIX_AT(out, 0, j) - MATRIX_AT(y, 0, j);
                        c += d * d;
                }
        }
        return (c/(float)ti.rows);
}

void NeuNet_fiiff(struct NeuNet nn, struct NeuNet g, float eps, struct Matrix ti, struct Matrix to) {
        float saved;
        float c = NeuNet_cost(nn, ti, to);
        for(size_t i = 0; i < nn.count; i++) {
                for(size_t j = 0; j < nn.ws[i].rows; j++) {
                        for(size_t k = 0; k < nn.ws[i].cols; k++) {
                                saved = MATRIX_AT(nn.ws[i], j, k);
                                MATRIX_AT(nn.ws[i], j, k) += eps;
                                MATRIX_AT(g.ws[i], j, k) = (NeuNet_cost(nn, ti, to) - c) / eps;
                                MATRIX_AT(nn.ws[i], j, k) = saved;
                        }
                }
                for(size_t j = 0; j < nn.bs[i].rows; j++) {
                        for(size_t k = 0; k < nn.bs[i].cols; k++) {
                                saved = MATRIX_AT(nn.bs[i], j, k);
                                MATRIX_AT(nn.bs[i], j, k) += eps;
                                MATRIX_AT(g.bs[i], j, k) = (NeuNet_cost(nn, ti, to) - c) / eps;
                                MATRIX_AT(nn.bs[i], j, k) = saved;
                        }
                }
        }
}

void NeuNet_backprop(struct NeuNet nn, struct NeuNet g, struct Matrix ti, struct Matrix to) {
        NCL_ASSERT(ti.rows == to.rows);
        NCL_ASSERT(to.cols == NEUNET_OUTPUT(nn).cols);
        NeuNet_clear(g);
        size_t n = ti.rows;
        for(size_t i = 0; i < n; i++) {
                Matrix_copy(NEUNET_INPUT(nn), Matrix_row(ti, i));
                NeuNet_forward(nn);
                for(size_t j = 0; j <= nn.count; j++) Matrix_fill(g.as[j], 0);
                for(size_t j = 0; j < to.cols; j++) {
                        MATRIX_AT(NEUNET_OUTPUT(g), 0, j) = MATRIX_AT(NEUNET_OUTPUT(nn), 0, j) - MATRIX_AT(to, i, j);
                }
                for(size_t l = nn.count; l > 0; l--) {
                        for(size_t j = 0; j < nn.as[l].cols; j++) {
                                float a = MATRIX_AT(nn.as[l], 0, j);
                                float da = MATRIX_AT(g.as[l], 0, j);
                                MATRIX_AT(g.bs[l - 1], 0, j) += 2 * da * a * (1 - a);
                                for(size_t k = 0; k < nn.as[l - 1].cols; k++) {
                                        float pa = MATRIX_AT(nn.as[l - 1], 0, k);
                                        float w = MATRIX_AT(nn.ws[l - 1], k, j);
                                        MATRIX_AT(g.ws[l - 1], k, j) += 2 * da * a * (1 - a) * pa;
                                        MATRIX_AT(g.as[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
                                }
                        }
                }
        }
        for(size_t i = 0; i < g.count; i++) {
                for(size_t j = 0; j < g.ws[i].rows; j++) {
                        for(size_t k = 0; k < g.ws[i].cols; k++) {
                                MATRIX_AT(g.ws[i], j, k) /= n;
                        }
                }
                for(size_t j = 0; j < g.bs[i].rows; j++) {
                        for(size_t k = 0; k < g.bs[i].cols; k++) {
                                MATRIX_AT(g.bs[i], j, k) /= n;
                        }
                }
        }
}

void NeuNet_learn(struct NeuNet nn, struct NeuNet g, float rate) {
        for(size_t i = 0; i < nn.count; i++) {
                for(size_t j = 0; j < nn.ws[i].rows; j++) {
                        for(size_t k = 0; k < nn.ws[i].cols; k++) {
                                MATRIX_AT(nn.ws[i], j, k) -= rate * MATRIX_AT(g.ws[i], j, k);
                        }
                }
                for(size_t j = 0; j < nn.bs[i].rows; j++) {
                        for(size_t k = 0; k < nn.bs[i].cols; k++) {
                                MATRIX_AT(nn.bs[i], j, k) -= rate * MATRIX_AT(g.bs[i], j, k);
                        }
                }
        }
}

void NeuNet_print(struct NeuNet nn, const char *name) {
        char buf[256];
        NCL_PRINT("%s = [\n", name);
        for(size_t i = 0; i < nn.count; i++) {
                snprintf(buf, sizeof(buf), "%s.w%zu", name, i);
                Matrix_print(nn.ws[i], buf, 4);
                snprintf(buf, sizeof(buf), "%s.b%zu", name, i);
                Matrix_print(nn.bs[i], buf, 4);
        }
        NCL_PRINT("]\n");
}

#endif // NCL_IMPLEMENTATION
