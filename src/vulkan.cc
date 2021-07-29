#include <glslang/SPIRV/GlslangToSpv.h>

#define VK_USE_PLATFORM_XCB_KHR 1
#include <vulkan/vulkan.hpp>
#include <xcb/xcb.h>
#include <vulkan/vulkan_xcb.h>
#include <vulkan/vulkan_raii.hpp>

#include "geometries.hpp"
#include "math.hpp"
#include "shaders.hpp"
#include "utils.hpp"
#include "wsi.hpp"

#include <iostream>
#include <thread>

static char const * application_name = "render";
static char const * engine_name = "render";

int main() {
    vk::DynamicLoader dl;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    vk::raii::Context context;
    vk::raii::Instance instance = vk::raii::su::makeInstance(
        context,
        application_name,
        engine_name,
        {},
        vk::su::getInstanceExtensions()
    );
    //VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
#if !defined(NDEBUG)
    vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger(instance, vk::su::makeDebugUtilsMessengerCreateInfoEXT());
#endif
    vk::raii::PhysicalDevice physical_device = std::move(vk::raii::PhysicalDevices(instance).front());

    vk::raii::su::SurfaceData surface_data(instance, application_name, vk::Extent2D(500, 500));

    std::pair<uint32_t, uint32_t> graphics_and_present_queue_family_index =
        vk::raii::su::findGraphicsAndPresentQueueFamilyIndex(physical_device, *surface_data.pSurface);
    vk::raii::Device device = vk::raii::su::makeDevice(
            physical_device, graphics_and_present_queue_family_index.first, vk::su::getDeviceExtensions());

    vk::raii::CommandPool command_pool = vk::raii::CommandPool(
        device,
        {
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            graphics_and_present_queue_family_index.first
        }
    );
    vk::raii::CommandBuffer command_buffer = vk::raii::su::makeCommandBuffer(device, command_pool);

    vk::raii::Queue graphics_queue(device, graphics_and_present_queue_family_index.first, 0);
    vk::raii::Queue present_queue(device, graphics_and_present_queue_family_index.second, 0);

    //FIXME
    vk::raii::su::SwapChainData swap_chain_data(
        physical_device,
        device,
        *surface_data.pSurface,
        surface_data.extent,
        vk::ImageUsageFlagBits::eColorAttachment |
        vk::ImageUsageFlagBits::eTransferSrc,
        {},
        graphics_and_present_queue_family_index.first,
        graphics_and_present_queue_family_index.second
    );

    vk::raii::su::DepthBufferData depth_buffer_data(
        physical_device,
        device,
        vk::Format::eD16Unorm,
        surface_data.extent
    );

    vk::raii::su::TextureData texture_data(physical_device, device);
    //FIXME

    vk::raii::su::BufferData uniform_buffer_data(
        physical_device,
        device,
        sizeof(glm::mat4x4),
        vk::BufferUsageFlagBits::eUniformBuffer
    );
    glm::mat4x4 mvpc_matrix = vk::su::createModelViewProjectionClipMatrix(surface_data.extent);
    vk::raii::su::copyToDevice(uniform_buffer_data.deviceMemory, mvpc_matrix);

    vk::raii::DescriptorSetLayout descriptor_set_layout = vk::raii::su::makeDescriptorSetLayout(
        device,
        {
            {vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex},
            {vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment}
        }
    );
    vk::raii::PipelineLayout pipeline_layout(device, {{}, *descriptor_set_layout});

    vk::Format color_format = vk::su::pickSurfaceFormat(physical_device.getSurfaceFormatsKHR(**surface_data.pSurface)).format;
    vk::raii::RenderPass render_pass = vk::raii::su::makeRenderPass(device, color_format, depth_buffer_data.format);

    glslang::InitializeProcess();
    vk::raii::ShaderModule vertex_shader_module = vk::raii::su::makeShaderModule(
        device,
        vk::ShaderStageFlagBits::eVertex,
        vertexShaderText_PT_T
    );
    vk::raii::ShaderModule fragment_shader_module = vk::raii::su::makeShaderModule(
        device,
        vk::ShaderStageFlagBits::eFragment,
        fragmentShaderText_T_C
    );
    glslang::FinalizeProcess();

    std::vector<vk::raii::Framebuffer> framebuffers = vk::raii::su::makeFramebuffers(
        device,
        render_pass,
        swap_chain_data.imageViews,
        &*depth_buffer_data.pImageView,
        surface_data.extent
    );

    vk::raii::su::BufferData vertex_buffer_data(
        physical_device,
        device,
        sizeof(texturedCubeData),
        vk::BufferUsageFlagBits::eVertexBuffer
    );
    vk::raii::su::copyToDevice(
        vertex_buffer_data.deviceMemory,
        texturedCubeData,
        sizeof(texturedCubeData) / sizeof(texturedCubeData[0])
    );

    vk::raii::DescriptorPool descriptor_pool = vk::raii::su::makeDescriptorPool(
        device,
        {
            {vk::DescriptorType::eUniformBuffer, 1},
            {vk::DescriptorType::eCombinedImageSampler, 1}
        }
    );
    vk::raii::DescriptorSet descriptor_set = std::move(vk::raii::DescriptorSets(device, {*descriptor_pool, *descriptor_set_layout}).front());

    vk::raii::su::updateDescriptorSets(
        device,
        descriptor_set,
        {{vk::DescriptorType::eUniformBuffer, uniform_buffer_data.buffer, nullptr}},
        {texture_data}
    );

    vk::raii::PipelineCache pipeline_cache(device, vk::PipelineCacheCreateInfo());
    vk::raii::Pipeline graphics_pipeline = vk::raii::su::makeGraphicsPipeline(
        device,
        pipeline_cache,
        vertex_shader_module,
        nullptr,
        fragment_shader_module,
        nullptr,
        sizeof(texturedCubeData[0]),
        {{vk::Format::eR32G32B32A32Sfloat, 0}, {vk::Format::eR32G32Sfloat, 16}},
        vk::FrontFace::eClockwise,
        true,
        pipeline_layout,
        render_pass
    );

    while (true) {
        vk::raii::Semaphore image_acquired_semaphore(device, vk::SemaphoreCreateInfo());
        auto[result, image_index] = swap_chain_data.pSwapChain->acquireNextImage(vk::su::FenceTimeout, *image_acquired_semaphore);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("unknown error from acquireNextImage");
        }
        if (image_index >= swap_chain_data.images.size()) {
            throw std::runtime_error("image_index returned by acquireNextImage is greater than expected");
        }

        std::array<vk::ClearValue, 2> clear_values;
        clear_values[0].color = vk::ClearColorValue(std::array<float, 4>({{0.2f, 0.2f, 0.2f, 0.2f}}));
        clear_values[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);
        vk::RenderPassBeginInfo render_pass_begin_info(
            *render_pass, *framebuffers[image_index], vk::Rect2D(vk::Offset2D(0, 0), surface_data.extent), clear_values
        );

        command_buffer.begin(vk::CommandBufferBeginInfo());
        //texture_data.setImage(command_buffer, vk::su::CheckerboardImageGenerator());
        /*
        vk::raii::su::setImageLayout(
            command_buffer,
            *(texture_data.pImageData->image),
            texture_data.format,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );
        */

        command_buffer.beginRenderPass(render_pass_begin_info, vk::SubpassContents::eInline);
        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphics_pipeline);
        command_buffer.bindDescriptorSets(
            vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, {*descriptor_set}, nullptr
        );

        command_buffer.bindVertexBuffers(0, {*vertex_buffer_data.buffer}, {0});
        command_buffer.setViewport(
            0,
            vk::Viewport(
                0.0f,
                0.0f,
                static_cast<float>(surface_data.extent.width),
                static_cast<float>(surface_data.extent.height),
                0.0f,
                1.0f
            )
        );
        command_buffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), surface_data.extent));

        command_buffer.draw(12 * 3, 1, 0, 0);
        command_buffer.endRenderPass();
        command_buffer.end();

        vk::raii::Fence draw_fence(device, vk::FenceCreateInfo());

        vk::PipelineStageFlags wait_destination_stage_mask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        vk::SubmitInfo submit_info(*image_acquired_semaphore, wait_destination_stage_mask, *command_buffer);
        graphics_queue.submit(submit_info, *draw_fence);

        while (vk::Result::eTimeout == device.waitForFences({*draw_fence}, VK_TRUE, vk::su::FenceTimeout)) {
        }

        vk::PresentInfoKHR present_info(nullptr, **swap_chain_data.pSwapChain, image_index);
        result = present_queue.presentKHR(present_info);
        if (result == vk::Result::eSuccess) {
        } else if (result == vk::Result::eSuboptimalKHR) {
            std::cout << "vk::Queue::presentKHR returned vk::Result::eSuboptimalKHR !\n";
        } else {
            throw std::runtime_error("unexpected result is returned!");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::cout << "frame\n";
    }

    device.waitIdle();
    return 0;
}
