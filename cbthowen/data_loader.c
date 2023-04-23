#include "data_loader.h"

bmatrix_t train_images; // Of shape (#Samples, 784) -> flattened
bmatrix_t test_images; // Of shape (#Samples, 784) -> flattened
unsigned char* train_labels; // Of shape (#Samples,) 
unsigned char* test_labels; // Of shape (#Samples,)

bmatrix_t binarized_train; // Of shape (#Samples, 784 * bits_per_pixel)
bmatrix_t binarized_test; // Of shape (#Samples, 784 * bits_per_pixel)

bmatrix_t reordered_binarized_train; // Of shape (#Samples, 784 * bits_per_pixel)
bmatrix_t reordered_binarized_test; // Of shape (#Samples, 784 * bits_per_pixel)

void reverse_bytes(uint32_t* element) {
    unsigned char tmp;
    char* ptr = (char*) element;

    tmp = ptr[0];
    ptr[0] = ptr[3];
    ptr[3] = tmp;

    tmp = ptr[1];
    ptr[1] = ptr[2];
    ptr[2] = tmp;
}

void read_mnist_file(char* file_path, size_t num_samples, size_t stride, size_t len_info, unsigned char* data, uint32_t* info) {
    FILE* fd = fopen(file_path, "r");

    if(fd == NULL) printf("Not able to read the file at path %s\n", file_path);
    
    fread(info, sizeof(uint32_t), len_info, fd);
    for(size_t it = 0; it < len_info; ++it) reverse_bytes(info + it);
    
    fread(data, stride, num_samples, fd);

    fclose(fd);
}

void load_mnist() {
    // Allocating buffers
    bmatrix_init(&train_images, MNIST_NUM_TRAIN, MNIST_IM_SIZE);
    bmatrix_init(&test_images, MNIST_NUM_TEST, MNIST_IM_SIZE);

    train_labels = calloc(MNIST_NUM_TRAIN, sizeof(*train_labels));
    test_labels = calloc(MNIST_NUM_TRAIN, sizeof(*test_labels));

    uint32_t* info_buffer = calloc(MNIST_LEN_INFO_IMAGE, sizeof(*info_buffer));

    // Reading data
    read_mnist_file(MNIST_TRAIN_IMAGE, MNIST_NUM_TRAIN, MNIST_IM_SIZE, MNIST_LEN_INFO_IMAGE, train_images.data, info_buffer);
    assert(info_buffer[0] == 2051);
    assert(info_buffer[1] == MNIST_NUM_TRAIN);

    read_mnist_file(MNIST_TEST_IMAGE, MNIST_NUM_TEST, MNIST_IM_SIZE, MNIST_LEN_INFO_IMAGE, test_images.data, info_buffer);
    assert(info_buffer[0] == 2051);
    assert(info_buffer[1] == MNIST_NUM_TEST);
    
    read_mnist_file(MNIST_TRAIN_LABEL, MNIST_NUM_TRAIN, 1, MNIST_LEN_INFO_LABEL, train_labels, info_buffer);  
    assert(info_buffer[0] == 2049);
    assert(info_buffer[1] == MNIST_NUM_TRAIN);

    read_mnist_file(MNIST_TEST_LABEL, MNIST_NUM_TEST, 1, MNIST_LEN_INFO_LABEL, test_labels, info_buffer);
    assert(info_buffer[0] == 2049);
    assert(info_buffer[1] == MNIST_NUM_TEST);

    free(info_buffer);
}

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  ((byte) & 0x80 ? '1' : '0'), \
  ((byte) & 0x40 ? '1' : '0'), \
  ((byte) & 0x20 ? '1' : '0'), \
  ((byte) & 0x10 ? '1' : '0'), \
  ((byte) & 0x08 ? '1' : '0'), \
  ((byte) & 0x04 ? '1' : '0'), \
  ((byte) & 0x02 ? '1' : '0'), \
  ((byte) & 0x01 ? '1' : '0') 

void print_pixel(unsigned char value, int raw) {
    if(raw == 1) printf(BYTE_TO_BINARY_PATTERN" ", BYTE_TO_BINARY(value));
    else if(raw == 2) printf("%d ", value);
    else {
        char map[10]= " .,:;ox%#@";
        size_t index = (255 - value) * 10 / 256;
        printf("%c ", map[index]);
    }
}

void print_binarized_value(unsigned char value, size_t num_bits) {
    char map[2]= " @";
    printf("%c ", map[value - 1]);
}

void print_mnist_image_(bmatrix_t* images, unsigned char* labels, size_t index, int raw) {
    printf("Image %zu (Label %d)\n", index, labels[index]);
    for (size_t j = 0; j < MNIST_IM_SIZE; ++j) {
        print_pixel(*MATRIX(*images, index, j), raw);
        if ((j+1) % 28 == 0) putchar('\n');
    }
    putchar('\n');
}

void print_binarized_mnist_image(size_t index, size_t num_bits) {
    printf("Image %zu (Label %d) (Binarized)\n", index, train_labels[index]);
    for (size_t j = 0; j < MNIST_IM_SIZE; ++j) {
        char value = *MATRIX(binarized_test, index, j * num_bits);
        for(size_t b = 1; b < num_bits; ++b)
            value |= (*MATRIX(binarized_test, index, j * num_bits + b) << b);

        print_binarized_value(value, num_bits);
        if ((j+1) % 28 == 0) putchar('\n');
    }
    putchar('\n'); 
}

void print_mnist_image(size_t index) {
    print_mnist_image_(&train_images, train_labels, index, 0);
}

void print_mnist_image_raw(size_t index) {
    print_mnist_image_(&train_images, train_labels, index, 1);
}

unsigned char thermometer_encode(unsigned char val, double mean, double std, size_t num_bits, double* skews, unsigned char* encodings) {
    size_t skew_index = 0;
    for(; skew_index < num_bits - 1 && val > skews[skew_index] * std + mean; ++skew_index);

    // printf("val: %d, index: %d\n", val, skew_index);
        
    return encodings[skew_index];
}

void binarize_sample(bmatrix_t* result, bmatrix_t* dataset, size_t sample_it, size_t num_bits, double* mean, double* variance, double* skews, unsigned char* encodings) {
    for(size_t offset_it = 0; offset_it < dataset->stride; ++offset_it) {
        char packed_encoding = thermometer_encode(*MATRIX(*dataset, sample_it, offset_it), mean[offset_it], sqrt(variance[offset_it]), num_bits, skews, encodings);
        // printf("packed "BYTE_TO_BINARY_PATTERN"\n", BYTE_TO_BINARY(packed_encoding));
        for(size_t bit_it = 0; bit_it < num_bits; ++bit_it) {
            // char x = (packed_encoding >> bit_it) & 0x1;
            // printf("    "BYTE_TO_BINARY_PATTERN"\n", BYTE_TO_BINARY(x));
            *MATRIX(*result, sample_it, offset_it*num_bits + bit_it) = (packed_encoding >> bit_it) & 0x1;
        }
    }
}

void binarize_matrix(bmatrix_t* result, bmatrix_t* dataset, size_t sample_size, size_t num_samples, size_t num_bits) {
    double skews[num_bits - 1];
    for(size_t it = 1; it < num_bits; ++it) {
        skews[it] = gauss_inv((((double) it)) / (((double) num_bits)));
        // printf("skew: %lf\n", skews[it]);
    }

    unsigned char encodings[num_bits];
    for(size_t it = 0; it < num_bits; ++it) {
        encodings[it] = (((unsigned char) 0xff) << it) & (((unsigned char) 0xff) >> (8 - num_bits));
        // printf("encoding: "BYTE_TO_BINARY_PATTERN"\n", BYTE_TO_BINARY(encodings[it]));
    }

    double mean[sample_size];
    double variance[sample_size];

    bmatrix_mean(mean, dataset, sample_size, num_samples);
    bmatrix_variance(variance, dataset, sample_size, num_samples, mean);

    for(size_t sample_it = 0; sample_it < num_samples; ++sample_it)
        binarize_sample(result, dataset, sample_it, num_bits, mean, variance, skews, encodings);
}

void binarize_mnist(size_t num_bits) {
    // Allocating buffers
    bmatrix_init(&binarized_train, MNIST_NUM_TRAIN, MNIST_IM_SIZE * num_bits);
    bmatrix_init(&binarized_test, MNIST_NUM_TEST, MNIST_IM_SIZE * num_bits);

    // Binarize
    binarize_matrix(&binarized_train, &train_images, MNIST_IM_SIZE, MNIST_NUM_TRAIN, num_bits);
    binarize_matrix(&binarized_test, &test_images, MNIST_IM_SIZE, MNIST_NUM_TEST, num_bits);
}

void fill_input_random(unsigned char* input, size_t input_length) {
    for(size_t it = 0; it < input_length; ++it) {
        input[it] = rand() % 2;
    }
}

void reorder_dataset(bmatrix_t* result, bmatrix_t* dataset, size_t* order, size_t num_samples, size_t num_elements) {
    for(size_t it = 0; it < num_samples; ++it) {
        reorder_array(MATRIX_AXIS1(*result, it), MATRIX_AXIS1(*dataset, it), order, num_elements);
    }
}

void reorder_binarized_mnist(size_t* order, size_t num_bits) {
    // Allocating buffers
    bmatrix_init(&reordered_binarized_train, MNIST_NUM_TRAIN, MNIST_IM_SIZE * num_bits);
    bmatrix_init(&reordered_binarized_test, MNIST_NUM_TEST, MNIST_IM_SIZE * num_bits);

    // Reordering
    reorder_dataset(&reordered_binarized_train, &binarized_train, order, MNIST_NUM_TRAIN, MNIST_IM_SIZE * num_bits);
    reorder_dataset(&reordered_binarized_test, &binarized_test, order, MNIST_NUM_TEST, MNIST_IM_SIZE * num_bits);
}

