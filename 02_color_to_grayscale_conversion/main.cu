#include <stdio.h>
#include <stdlib.h>

#include "../util.h"

#define CHANNELS 3

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"


__global__
void color_to_grayscale_conversion(u8 *p_out, u8 *p_in, int width, int height)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        // Get 1D offset for the grayscale image (output)
        int gray_offset = row*width + col;

        int rgb_offset = gray_offset*CHANNELS;
        u8 r = p_in[rgb_offset + 0];
        u8 g = p_in[rgb_offset + 1];
        u8 b = p_in[rgb_offset + 2];

        p_out[gray_offset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

u8 *color_to_grayscale_cpu(u8 *p_in, int width, int height, int channels)
{
    int size = width*height;
    u8 *p_out = (u8 *)malloc(size);
    if (p_out) {
        for (int i = 0; i < size; i++) {
            int offset = i*channels;
            u8 r = p_in[offset + 0];
            u8 g = p_in[offset + 1];
            u8 b = p_in[offset + 2];

            u8 pixel = 0.21f*r + 0.71f*g + 0.07f*b;
            p_out[i] = pixel;
        }
    }

    return p_out;
}


#define COLOR_TABLE_SIZE (256)
static u32 color_table[COLOR_TABLE_SIZE];

static void fill_color_table()
{
    for (int i = 0; i < COLOR_TABLE_SIZE; i++) {
        u8 r = i;
        u8 g = i;
        u8 b = i;
        u8 reserved = 0;

        color_table[i] = (reserved << 24) | (b << 16) | (g << 8) | (r << 0);
    }
}

int main()
{
    fill_color_table();

    int width;
    int height;
    int channels_in_file;

    const char *filename = "image.jpg";
    u8 *image = stbi_load(filename, &width, &height, &channels_in_file, CHANNELS);
    if (image == NULL) {
        fprintf(stderr, "Failed to load image %s: %s\n", filename, stbi_failure_reason());
        exit(EXIT_FAILURE);
    }

    // printf("image = %d, x = %d, y = %d, channels_in_file = %d\n", image, x, y, channels_in_file);


#if 0
    // Reorder the R and B channels because the stb and bmp format are reverse.
    int image_size = x*y;
    for (int i = 0; i < image_size; i++) {
        int offset = i*channels_in_file;
        u8 tmp = image[offset + 0];
        image[offset + 0] = image[offset + 2];
        image[offset + 2] = tmp;
    }
#endif


#if 1
    cudaError_t err = cudaSuccess;

    int source_image_size = width*height*CHANNELS;
    int dest_image_size = width*height;

    u8 *dest_image = (u8 *)malloc(dest_image_size);
    
    u8 *source_image_d;
    u8 *dest_image_d;

    cudaMalloc((void **)&source_image_d, source_image_size);
    cudaMalloc((void **)&dest_image_d, dest_image_size);

    cudaMemcpy(source_image_d, image, source_image_size, cudaMemcpyHostToDevice);

    dim3 dim_grid(ceil(width / 32.0f), ceil(height / 32.0f), 1);
    dim3 dim_block(32, 32, 1);
    color_to_grayscale_conversion<<<dim_grid, dim_block>>>(dest_image_d, source_image_d, width, height);

    cudaMemcpy(dest_image, dest_image_d, dest_image_size, cudaMemcpyDeviceToHost);

    cudaFree(source_image_d);
    cudaFree(dest_image_d);

    save_to_bitmap("grayscale_gpu.bmp", dest_image, width, height, 1, COLOR_TABLE_SIZE, color_table);

    printf("Done\n");

#else
    u8 *image_grayscale = color_to_grayscale_cpu(image, x, y, CHANNELS);
    if (image_grayscale) {
        save_to_bitmap("grayscale.bmp", image_grayscale, x, y, 1);
        printf("Done\n");
        free(image_grayscale);
    }
#endif

    stbi_image_free(image);

    return 0;
}
