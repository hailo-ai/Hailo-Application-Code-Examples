#include <iostream>
#include <cstdlib> // for std::atoi
#include <getopt.h> // for getopt_long
#include <memory>
#include <vector>
#include "dsp_utils.h"

class DSPImage {
public:
    DSPImage() : image(nullptr) {}
    
    ~DSPImage() {
        if (image) {
            cleanup_image(image);
            image = nullptr;
        }
    }

    // Delete copy constructor and assignment operator
    DSPImage(const DSPImage&) = delete;
    DSPImage& operator=(const DSPImage&) = delete;

    // Allow move operations
    DSPImage(DSPImage&& other) noexcept : image(other.image) {
        other.image = nullptr;
    }

    DSPImage& operator=(DSPImage&& other) noexcept {
        if (this != &other) {
            if (image) {
                cleanup_image(image);
            }
            image = other.image;
            other.image = nullptr;
        }
        return *this;
    }

    dsp_image_properties_t* get() { return image; }
    dsp_image_properties_t** address() { return &image; }

private:
    dsp_image_properties_t* image;
};

void print_usage(const char *prog_name)
{
    std::cerr << "Usage: " << prog_name << " --input-width <width> --input-height <height> --output-width <output_width> --output-height <output_height> <raw_image_path>" << std::endl;
    std::cerr << "Example: " << prog_name << " --input-width 810 --input-height 1080 --output-width 640 --output-height 640 image.raw" << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc < 9)
    {
        print_usage(argv[0]);
        return 1;
    }

    // Default values
    size_t input_width = 0, input_height = 0;
    size_t output_width = 640, output_height = 640;
    std::string path;

    struct option long_options[] = {
        {"input-width", required_argument, nullptr, 'w'},
        {"input-height", required_argument, nullptr, 'h'},
        {"output-width", required_argument, nullptr, 'W'},
        {"output-height", required_argument, nullptr, 'H'},
        {nullptr, 0, nullptr, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "w:h:W:H:", long_options, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'w':
            input_width = std::atoi(optarg);
            break;
        case 'h':
            input_height = std::atoi(optarg);
            break;
        case 'W':
            output_width = std::atoi(optarg);
            break;
        case 'H':
            output_height = std::atoi(optarg);
            break;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    if (input_width == 0 || input_height == 0 || optind >= argc)
    {
        print_usage(argv[0]);
        return 1;
    }

    path = argv[optind]; // Path is the last argument after all options

    if (input_width <= 0 || input_height <= 0 || output_width <= 0 || output_height <= 0)
    {
        std::cerr << "Invalid dimensions." << std::endl;
        return 1;
    }

    dsp_device device;
    dsp_create_device(&device);

    // Create a vector to store all DSP images
    std::vector<DSPImage> images;

    // Original image
    image_arguments args = {
        path.c_str(),
        input_width,
        input_height,
        input_width * input_height * 3,
        DSP_IMAGE_FORMAT_RGB,
        DSP_MEMORY_TYPE_DMABUF};

    dsp_data_plane_t plane;
    plane.bytesused = args.width * args.height * 3;
    plane.bytesperline = args.width * 3;

    images.emplace_back();
    dsp_image_properties_t *original_image = generic_alloc_image_dmabuf(&args, &plane, 1);
    if (!original_image)
    {
        std::cerr << "Source DMA buffer allocation failed." << std::endl;
        return 1;
    }
    *images.back().address() = original_image;

    if (generic_read_image_dmabuf(original_image, &args) != DSP_SUCCESS)
    {
        std::cerr << "Failed to read image into DMA buffer." << std::endl;
        return 1;
    }

    // Cropped and resized image
    image_arguments cropped_resized_args = {
        path.c_str(),
        output_width,
        output_height,
        output_width * output_height * 3,
        DSP_IMAGE_FORMAT_RGB,
        DSP_MEMORY_TYPE_DMABUF};

    dsp_data_plane_t cropped_resized_plane;
    cropped_resized_plane.bytesused = cropped_resized_args.width * cropped_resized_args.height * 3;
    cropped_resized_plane.bytesperline = cropped_resized_args.width * 3;

    images.emplace_back();
    dsp_image_properties_t *cropped_resized_image = generic_alloc_image_dmabuf(&cropped_resized_args, &cropped_resized_plane, 1);
    if (!cropped_resized_image)
    {
        std::cerr << "Output DMA buffer allocation failed." << std::endl;
        return 1;
    }
    *images.back().address() = cropped_resized_image;

    dsp_letterbox_properties_t letterbox_params = {
        .alignment = DSP_LETTERBOX_MIDDLE,
        .color = {.r = 0, .g = 0, .b = 0},
    };

    dsp_roi_t crop_params = {
        .start_x = 0,
        .start_y = 0,
        .end_x = input_width / 2,
        .end_y = input_height / 2,
    };

    dsp_resize_params_t resize_params;
    resize_params.interpolation = INTERPOLATION_TYPE_BILINEAR;
    resize_params.src = original_image;
    resize_params.dst = cropped_resized_image;

    if (dsp_crop_and_resize_letterbox(device, &resize_params, &crop_params, &letterbox_params) != DSP_SUCCESS)
    {
        std::cerr << "Failed to process image." << std::endl;
        return 1;
    }

    save_raw_image(cropped_resized_image, "cropped_and_resized_image.raw");

    // NV12 image
    image_arguments nv12_args = {
        path.c_str(),
        output_width,
        output_height,
        output_width * output_height * 3 / 2,
        DSP_IMAGE_FORMAT_NV12,
        DSP_MEMORY_TYPE_DMABUF};

    dsp_data_plane_t planes[2] = {
        [0] = {
            .bytesperline = nv12_args.width,
            .bytesused = nv12_args.width * nv12_args.height,
        },
        [1] = {
            .bytesperline = nv12_args.width,
            .bytesused = nv12_args.width * nv12_args.height / 2,
        },
    };

    images.emplace_back();
    dsp_image_properties_t *nv12_image = generic_alloc_image_dmabuf(&nv12_args, planes, 2);
    if (!nv12_image)
    {
        std::cerr << "NV12 image allocation failed." << std::endl;
        return 1;
    }
    *images.back().address() = nv12_image;

    if (dsp_convert_format(device, cropped_resized_image, nv12_image) != DSP_SUCCESS)
    {
        std::cerr << "Failed to convert image format." << std::endl;
        return 1;
    }

    // Affine rotation image
    images.emplace_back();
    dsp_image_properties_t *affine_rotation_image = generic_alloc_image_dmabuf(&nv12_args, planes, 2);
    if (!affine_rotation_image)
    {
        std::cerr << "Affine rotation image allocation failed." << std::endl;
        return 1;
    }
    *images.back().address() = affine_rotation_image;

    dsp_affine_rotation_params_t rotate_params;
    rotate_params.interpolation = INTERPOLATION_TYPE_BILINEAR;
    rotate_params.src = nv12_image;
    rotate_params.dst = affine_rotation_image;
    rotate_params.theta = 30;

    if (dsp_rotate(device, &rotate_params) != DSP_SUCCESS)
    {
        std::cerr << "Failed to rotate image." << std::endl;
        return 1;
    }

    save_raw_image(affine_rotation_image, "affine_rotation_image.raw");

    return 0;
}
