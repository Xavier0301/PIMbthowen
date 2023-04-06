#pragma once

#include <stdlib.h>
#include <math.h>

/************* UNIFORM ***********/
long unif_rand(long max);

long unif_rand_range(long min, long max);

void shuffle_array(size_t* array, size_t length);

/************* GAUSSIAN ***********/
// Returns a random number sampled from N(0,1)
double gauss_rand();

// Inverse of the error function erf
double erf_inv(double x);

// Quantile of Gaussian distribution  
double gauss_inv(double p);
