#ifndef UTIL_H
#define UTIL_H

//
// Types
//

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef char s8;
typedef short s16;
typedef int s32;
typedef float f32;

//
// CUDA wrappers
//

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


//
// Bitmap
//

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

void save_to_bitmap(const char *output_filename, u8 *image, int x, int y, int channels_in_file, u32 colors_used, u32 *color_table)
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
    info_header.colors_used = colors_used;
    info_header.important_colors = 0;

    Bitmap bitmap = {};
    bitmap.header = header;
    bitmap.info_header = info_header;


    FILE *output = fopen(output_filename, "wb");
    if (output) {
        fwrite(&bitmap, sizeof(Bitmap), 1, output);
        if (colors_used) {
            fwrite(color_table, colors_used * sizeof(u32), 1, output);
        }
        fwrite(image, image_size_bytes, 1, output);
    }

    fclose(output);
}


#endif