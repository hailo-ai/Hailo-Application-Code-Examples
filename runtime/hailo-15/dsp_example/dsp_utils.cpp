#include "dsp_utils.h"
#include <fstream>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <cerrno>


void cleanup_planes(dsp_data_plane_t *planes, size_t planes_count)
{
    if (planes)
    {
        for (size_t i = 0; i < planes_count; ++i)
        {
            if (planes[i].fd != -1)
            {
                close(planes[i].fd);
            }
        }
        delete[] planes;
    }
}

void cleanup_image(dsp_image_properties_t *image)
{
    if (image)
    {
        cleanup_planes(image->planes, image->planes_count);
        delete image;
    }
}

int allocate_dma_heap_buffer(int heap_fd, dsp_data_plane_t &plane, const dsp_data_plane_t &input_plane)
{
    struct dma_heap_allocation_data alloc_data = {
        .len = input_plane.bytesused,
        .fd_flags = O_RDWR | O_CLOEXEC,
        .heap_flags = 0,
    };

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0)
    {
        std::cerr << "Failed to allocate dma heap buffer: " << strerror(errno) << std::endl;
        return -1;
    }

    plane.fd = alloc_data.fd;
    plane.bytesused = input_plane.bytesused;
    plane.bytesperline = input_plane.bytesperline;
    return 0;
}

dsp_image_properties_t *generic_alloc_image_dmabuf(const image_arguments *args,
                                                   const dsp_data_plane_t *planes,
                                                   size_t planes_count)
{
    int heap_fd = -1;
    auto image = new dsp_image_properties_t();
    memset(image, 0, sizeof(*image));

    auto ret_planes = new dsp_data_plane_t[planes_count];
    memset(ret_planes, 0, sizeof(dsp_data_plane_t) * planes_count);

    for (size_t i = 0; i < planes_count; ++i)
    {
        ret_planes[i].fd = -1;
    }

    heap_fd = open("/dev/dma_heap/hailo_media_buf,cma", O_RDWR);
    if (heap_fd < 0)
    {
        std::cerr << "Failed to open dma heap: " << strerror(errno) << std::endl;
        cleanup_planes(ret_planes, planes_count);
        delete image;
        return nullptr;
    }

    for (size_t i = 0; i < planes_count; ++i)
    {
        if (allocate_dma_heap_buffer(heap_fd, ret_planes[i], planes[i]) < 0)
        {
            cleanup_planes(ret_planes, planes_count);
            delete image;
            close(heap_fd);
            return nullptr;
        }
    }

    image->planes = ret_planes;
    image->planes_count = planes_count;
    image->format = args->format;
    image->width = args->width;
    image->height = args->height;
    image->memory = DSP_MEMORY_TYPE_DMABUF;

    close(heap_fd);
    return image;
}

void save_raw_image(dsp_image_properties_t *image, const std::string &filename)
{
    if (!image || image->planes_count == 0)
    {
        std::cerr << "Invalid image passed to save_raw_image." << std::endl;
        return;
    }

    std::ofstream out_file(filename, std::ios::binary);
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open output file." << std::endl;
        return;
    }

    for (size_t i = 0; i < image->planes_count; ++i)
    {
        size_t size = image->planes[i].bytesused;
        int fd = image->planes[i].fd;

        void *data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED)
        {
            std::cerr << "Failed to mmap buffer for saving plane " << i << std::endl;
            continue;
        }

        out_file.write(reinterpret_cast<const char *>(data), size);
        munmap(data, size);
    }

    out_file.close();
    std::cout << "Saved raw image to " << filename << std::endl;
}

int generic_read_image_dmabuf(dsp_image_properties_t *image, struct image_arguments *args)
{
    void *addr = nullptr;
    size_t mapped_size = 0;
    FILE *file = fopen(args->path, "rb");
    if (!file)
    {
        std::cerr << "Failed to open file " << args->path << std::endl;
        return 1;
    }

    for (size_t i = 0; i < image->planes_count; ++i)
    {
        mapped_size = image->planes[i].bytesused;
        addr = mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, image->planes[i].fd, 0);
        if (addr == MAP_FAILED)
        {
            std::cerr << "Failed to mmap dma buf" << std::endl;
            fclose(file);
            return 1;
        }

        struct dma_buf_sync sync_start = {.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE};
        if (ioctl(image->planes[i].fd, DMA_BUF_IOCTL_SYNC, &sync_start) < 0)
        {
            std::cerr << "Failed to sync dma buf" << std::endl;
            munmap(addr, mapped_size);
            fclose(file);
            return 1;
        }

        if (fread(addr, mapped_size, 1, file) != 1)
        {
            std::cerr << "Failed to read " << mapped_size << " bytes from file" << std::endl;
            munmap(addr, mapped_size);
            fclose(file);
            return 1;
        }

        struct dma_buf_sync sync_end = {.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE};
        if (ioctl(image->planes[i].fd, DMA_BUF_IOCTL_SYNC, &sync_end) < 0)
        {
            std::cerr << "Failed to sync dma buf" << std::endl;
            munmap(addr, mapped_size);
            fclose(file);
            return 1;
        }

        munmap(addr, mapped_size);
    }

    fclose(file);
    return 0;
}
