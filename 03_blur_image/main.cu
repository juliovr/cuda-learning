#include <stdio.h>
#include <stdlib.h>

#include "../util.h"


#define CHANNELS 3
#define BLUR_SIZE 10

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"


__global__
void blur_kernel(u8 *out, u8 *in, int width, int height)
{
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int pix_val_r = 0;
        int pix_val_g = 0;
        int pix_val_b = 0;
        int pixels = 0;

        // Get the average of the surroundings BLUR_SIZE x BLUR_SIZE box.
        for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row) {
            for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col) {
                int cur_row = row + blur_row;
                int cur_col = col + blur_col;

                if (cur_row >= 0 && cur_row < height &&
                    cur_col >= 0 && cur_col < width)
                {
                    int offset = (cur_row*width + cur_col)*CHANNELS;
                    pix_val_r += in[offset + 0];
                    pix_val_g += in[offset + 1];
                    pix_val_b += in[offset + 2];

                    ++pixels; // Keep track of number of pixels in the avg.
                }
            }
        }

        int offset = (row*width + col)*CHANNELS;
        out[offset + 2] = (u8)((f32)pix_val_r / (f32)pixels);
        out[offset + 1] = (u8)((f32)pix_val_g / (f32)pixels);
        out[offset + 0] = (u8)((f32)pix_val_b / (f32)pixels);
    }
}

void blur(u8 *out, u8 *in, int width, int height)
{
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int pix_val_r = 0;
            int pix_val_g = 0;
            int pix_val_b = 0;
            int pixels = 0;

            // Get the average of the surroundings BLUR_SIZE x BLUR_SIZE box.
            for (int blur_row = -BLUR_SIZE; blur_row < BLUR_SIZE + 1; ++blur_row) {
                for (int blur_col = -BLUR_SIZE; blur_col < BLUR_SIZE + 1; ++blur_col) {
                    int cur_row = row + blur_row;
                    int cur_col = col + blur_col;

                    if (cur_row >= 0 && cur_row < height &&
                        cur_col >= 0 && cur_col < width)
                    {
                        int offset = (cur_row*width + cur_col)*CHANNELS;
                        pix_val_r += in[offset + 0];
                        pix_val_g += in[offset + 1];
                        pix_val_b += in[offset + 2];

                        ++pixels; // Keep track of number of pixels in the avg.
                    }
                }
            }

            int offset = (row*width + col)*CHANNELS;
            out[offset + 2] = (u8)((f32)pix_val_r / (f32)pixels);
            out[offset + 1] = (u8)((f32)pix_val_g / (f32)pixels);
            out[offset + 0] = (u8)((f32)pix_val_b / (f32)pixels);
        }
    }
}


int main()
{
    int width;
    int height;
    int channels_in_file;

    const char *filename = "image.jpg";
    u8 *image = stbi_load(filename, &width, &height, &channels_in_file, CHANNELS);
    if (image == NULL) {
        fprintf(stderr, "Failed to load image %s: %s\n", filename, stbi_failure_reason());
        exit(EXIT_FAILURE);
    }

    int image_size = width*height*channels_in_file;
    u8 *dest_image = (u8 *)malloc(image_size);

#if 1
    cudaError_t err = cudaSuccess;    
    
    u8 *source_image_d;
    u8 *dest_image_d;

    cudaMalloc((void **)&source_image_d, image_size);
    cudaMalloc((void **)&dest_image_d, image_size);

    cudaMemcpy(source_image_d, image, image_size, cudaMemcpyHostToDevice);

    dim3 dim_grid(ceil(width / 32.0f), ceil(height / 32.0f), 1);
    dim3 dim_block(32, 32, 1);
    blur_kernel<<<dim_grid, dim_block>>>(dest_image_d, source_image_d, width, height);

    cudaMemcpy(dest_image, dest_image_d, image_size, cudaMemcpyDeviceToHost);

    cudaFree(source_image_d);
    cudaFree(dest_image_d);

    save_to_bitmap("blur.bmp", dest_image, width, height, CHANNELS, 0, NULL);

    printf("Done\n");

#else
    blur(dest_image, image, width, height);

    save_to_bitmap("blur_cpu.bmp", dest_image, width, height, CHANNELS, 0, NULL);

    printf("Done\n");
#endif

    stbi_image_free(image);

    return 0;
}
