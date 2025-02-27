#include <stdio.h>
#include <stdlib.h>

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef char s8;
typedef short s16;
typedef int s32;
typedef float f32;

#define cudaMalloc(dev_ptr, size)                                                                           \
    err = cudaMalloc(dev_ptr, size);                                                                        \
    if (err != cudaSuccess) {                                                                               \
        fprintf(stderr, "Failed to allocate memory (line %d): %s\n", __LINE__, cudaGetErrorString(err));    \
        exit(EXIT_FAILURE);                                                                                 \
    }

#define cudaMemcpy(dst, src, count, kind)                                                                   \
    err = cudaMemcpy(dst, src, count, kind);                                                                \
    if (err != cudaSuccess) {                                                                               \
        fprintf(stderr, "Failed to copy memory (line %d): %s\n", __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                                                                 \
    }

#define cudaFree(dev_ptr)                                                                                   \
    err = cudaFree(dev_ptr);                                                                                \
    if (err != cudaSuccess) {                                                                               \
        fprintf(stderr, "Failed to free memory (line %d): %s\n", __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                                                                 \
    }


#define CHANNELS 3
#define BLUR_SIZE 10

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


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
            out[offset + 2] = (u32)((f32)pix_val_r / (f32)pixels);
            out[offset + 1] = (u32)((f32)pix_val_g / (f32)pixels);
            out[offset + 0] = (u32)((f32)pix_val_b / (f32)pixels);
        }
    }
}


#pragma pack(push, 1)
struct BitmapHeader {
    u16 signature;
    u32 file_size;
    u16 reserved_1;
    u16 reserved_2;
    u32 data_offset;
};

struct BitmapInfoHeader {
    u32 info_header_size;
    s32 width;
    s32 height;
    u16 planes;
    u16 bits_per_pixel;
    u32 compression;
    u32 image_size;
    s32 horizontal_resolution;
    s32 vertical_resolution;
    u32 colors_used;
    u32 important_colors;
};

struct Bitmap {
    BitmapHeader header;
    BitmapInfoHeader info_header;
};
#pragma pack(pop)

void save_to_bitmap(const char *output_filename, u8 *image, int x, int y, int channels_in_file)
{
    s32 width = x;
    s32 height = y;
    s32 image_size_bytes = width*height*channels_in_file;

    BitmapHeader header = {};
    header.signature = 0x4D42;
    header.file_size = sizeof(Bitmap) + image_size_bytes;
    header.data_offset = sizeof(Bitmap);

    BitmapInfoHeader info_header = {};
    info_header.info_header_size = sizeof(BitmapInfoHeader);
    info_header.width = width;
    info_header.height = -height;
    info_header.planes = 1;
    info_header.bits_per_pixel = channels_in_file*8;
    info_header.compression = 0;
    info_header.image_size = image_size_bytes;
    info_header.colors_used = 0;
    info_header.important_colors = 0;

    Bitmap bitmap = {};
    bitmap.header = header;
    bitmap.info_header = info_header;


    FILE *output = fopen(output_filename, "wb");
    if (output) {
        fwrite(&bitmap, sizeof(Bitmap), 1, output);
        fwrite(image, image_size_bytes, 1, output);
    }

    fclose(output);
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

#if 1
    cudaError_t err = cudaSuccess;

    int image_size = width*height*CHANNELS;
    
    u8 *dest_image = (u8 *)malloc(image_size);
    
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

    save_to_bitmap("blur.bmp", dest_image, width, height, CHANNELS);

    printf("Done\n");

#else
    int image_size = width*height*CHANNELS;
    u8 *dest_image = (u8 *)malloc(image_size);

    blur(dest_image, image, width, height);

    save_to_bitmap("blur_cpu.bmp", dest_image, width, height, CHANNELS);

    printf("Done\n");
#endif

    stbi_image_free(image);

    return 0;
}
