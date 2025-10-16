#ifndef DSP_UTILS_H
#define DSP_UTILS_H

#include <hailo/hailodsp.h>
#include <iostream>
#include <string>

// Struct Definitions
struct image_arguments
{
    const char *path;
    size_t width;
    size_t height;
    size_t stride; // 0 means auto
    dsp_image_format_t format;
    dsp_memory_type_t memory_type;
};

// Function Declarations
void cleanup_planes(dsp_data_plane_t *planes, size_t planes_count);
void cleanup_image(dsp_image_properties_t *image);
int allocate_dma_heap_buffer(int heap_fd, dsp_data_plane_t &plane, const dsp_data_plane_t &input_plane);
dsp_image_properties_t *generic_alloc_image_dmabuf(const image_arguments *args, const dsp_data_plane_t *planes, size_t planes_count);
void save_raw_image(dsp_image_properties_t *image, const std::string &filename);
int generic_read_image_dmabuf(dsp_image_properties_t *image, struct image_arguments *args);

#endif // DSP_UTILS_H
