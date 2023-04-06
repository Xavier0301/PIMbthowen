#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "common.h"

typedef struct Params {
    unsigned int   input_size;
    T     alpha;
    int   n_warmup;
    int   n_reps;
}Params;

static void usage() {
    fprintf(stderr,
        "\nUsage:  ./program [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -w <W>    # of untimed warmup iterations (default=0)"
        "\n    -e <E>    # of timed repetition iterations (default=1)"
        "\n"
        "\nWorkload-specific options:"
        "\n    -i <I>    input size (default=2621440 elements)"
        "\n    -a <A>    alpha (default=100)"
        "\n");
}

struct Params input_params(int argc, char **argv) {
    struct Params p;
    p.input_size    = 2621440;
    p.alpha         = 100;
    p.n_warmup      = 0;
    p.n_reps        = 1;

    int opt;
    while((opt = getopt(argc, argv, "hi:a:w:e:")) >= 0) {
        switch(opt) {
        case 'h':
        usage();
        exit(0);
        break;
        case 'i': p.input_size    = atoi(optarg); break;
        case 'a': p.alpha         = atoi(optarg); break;
        case 'w': p.n_warmup      = atoi(optarg); break;
        case 'e': p.n_reps        = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }
    assert(NR_DPUS > 0 && "Invalid # of dpus!");

    return p;
}
#endif
