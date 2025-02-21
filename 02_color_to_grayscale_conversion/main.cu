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

#if 1
        p_out[gray_offset] = 0.21f*r + 0.71f*g + 0.07f*b;
#else
        p_out[gray_offset] =
#endif
    }
}

u8 *color_to_grayscale_cpu(u8 *p_in, int width, int height, int channels)
{
    int size = width*height;
    u8 *p_out = (u8 *)malloc(size);
    if (p_out) {
        for (int i = 0; i < size; i++) {
            int offset = i*channels;
            u8 r = p_in[offset + 2];
            u8 g = p_in[offset + 1];
            u8 b = p_in[offset + 0];

            u8 pixel = 0.21f*r + 0.71f*g + 0.07f*b;
            p_out[i] = pixel;
            // p_out[offset + 0] = 0.3f*r;
            // p_out[offset + 1] = 0.59f*g;
            // p_out[offset + 2] = 0.11f*b;
        }
    }

    return p_out;
}


struct V3 {
    f32 x, y, z;
};

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
    u32 red_mask;
    u32 green_mask;
    u32 blue_mask;
    u32 alpha_mask;
    u32 color_space_type;
    struct {
        V3 color_space_red;
        V3 color_space_green;
        V3 color_space_blue;
    } color_space_endpoints;
    u32 gamma_red;
    u32 gamma_green;
    u32 gamma_blue;
    u32 intent;
    u32 profile_data;
    u32 profile_size;
    u32 reserved;
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
        fwrite(image, image_size_bytes, 1, output);
    }

    fclose(output);
}


int main()
{
    int x;
    int y;
    int channels_in_file;
    int comp;

    const char *filename = "image.jpg";
    u8 *image = stbi_load(filename, &x, &y, &channels_in_file, CHANNELS);
    if (image == NULL) {
        fprintf(stderr, "Failed to load image %s: %s\n", filename, stbi_failure_reason());
        exit(EXIT_FAILURE);
    }

    printf("image = %d, x = %d, y = %d, channels_in_file = %d\n", image, x, y, channels_in_file);


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

    u8 *image_grayscale = color_to_grayscale_cpu(image, x, y, CHANNELS);
    if (image_grayscale) {
        save_to_bitmap("grayscale.bmp", image_grayscale, x, y, 1);
        printf("Done\n");
        free(image_grayscale);
    }

    stbi_image_free(image);

    return 0;
}
