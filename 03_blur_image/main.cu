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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


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


#define COLOR_TABLE_SIZE  (256)
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
    info_header.colors_used = COLOR_TABLE_SIZE;
    info_header.important_colors = 0;

    Bitmap bitmap = {};
    bitmap.header = header;
    bitmap.info_header = info_header;

#if 0
    FILE *reference = fopen("reference.bmp", "rb");
    Bitmap *reference_bitmap;
    if (reference) {
        fseek(reference, 0, SEEK_END);
        int size = ftell(reference);
        fseek(reference, 0, SEEK_SET);

        u8 *memory = (u8 *)malloc(size);
        fread(memory, size, 1, reference);

        fclose(reference);

        reference_bitmap = (Bitmap *)memory;
    }
#endif

    FILE *output = fopen(output_filename, "wb");
    if (output) {
        fwrite(&bitmap, sizeof(Bitmap), 1, output);
        fwrite(color_table, COLOR_TABLE_SIZE * sizeof(u32), 1, output);
        fwrite(image, image_size_bytes, 1, output);
    }

    fclose(output);
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

    save_to_bitmap("grayscale_gpu.bmp", dest_image, width, height, 1);

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
