#pragma once

#if defined(__ANDROID__)
#include <android/asset_manager.h>
#include <android/log.h>
// os window, glfw, sdi, ...
#include <android/native_window.h>
#include <android/native_window_jni.h>
// To use volk, you have to include volk.h instead of vulkan/vulkan.h;
// this is necessary to use function definitions from volk.
#include <vulkan/vulkan.h>
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <vector>
#include <optional>
#include <numeric>
#include <assert.h>
#include <format>
#include <utility>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <filesystem> // for shader
#include <optional>

// must ahead of <vk_mem_alloc.h>, or else it will crash on vk functions
#ifndef __ANDROID__
#define VK_NO_PROTOTYPES // for volk
#define NOMINMAX
#include "volk.h"
#endif
// To do it properly:
//
// Include "vk_mem_alloc.h" file in each CPP file where you want to use the library. This includes declarations of all members of the library.
// In exactly one CPP file define following macro before this include. It enables also internal definitions.
#include <vk_mem_alloc.h>
#include <misc.h>

#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>

#if defined(__ANDROID__)
// functor for custom deleter for unique_ptr
struct AndroidNativeWindowDeleter
{
    void operator()(ANativeWindow *window) { ANativeWindow_release(window); }
};
#endif

#ifdef _WIN32
#if !defined(VK_USE_PLATFORM_WIN32_KHR)
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#endif

#ifdef __linux__
#if !defined(VK_USE_PLATFORM_XLIB_KHR)
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#endif

enum COMMAND_SEMANTIC : int
{
    RENDERING = 0,
    IO,
    MIPMAP,
    TRANSFER,
    COMMAND_SEMANTIC_SIZE
};

class Window;

using CommandBufferEntity = std::tuple<VkCommandPool, VkCommandBuffer, VkFence, uint32_t, VkQueue>;
enum COMMAND_BUFFER_ENTITY_OFFSET : int
{
    COMMAND_POOL = 0,
    COMMAND_BUFFER,
    FENCE,
    QUEUE_FAMILY_INDEX,
    QUEUE
};

using ImageEntity = std::tuple<VkImage, VkImageView, VmaAllocation, VmaAllocationInfo, uint32_t, VkExtent3D, VkFormat>;
enum IMAGE_ENTITY_OFFSET : int
{
    IMAGE = 0,
    IMAGE_VIEW,
    IMAGE_VMA_ALLOCATION,
    IMAGE_VMA_ALLOCATION_INFO,
    MIPMAP_COUNT = 4,
    IMAGE_EXTENT,
    IMAGE_FORMAT
};

using MappingAddressType = void *;

// VkDeviceOrHostAddressConstKHR: Union specifying a const device or host address
// Gpu device local buffer: 64bit address could be accessed by shader
// staging buffer: void*
using BufferEntity = std::tuple<VkBuffer, VmaAllocation, VmaAllocationInfo, MappingAddressType, VkDeviceSize,
                                VkDeviceOrHostAddressConstKHR,
                                void *>;
enum BUFFER_ENTITY_UID : int
{
    BUFFER = 0,
    VMA_ALLOCATION,
    VMA_ALLOCATION_INFO,
    MAPPING_ADDRESS,
    BUFFER_SIZE,
    DEVICE_HOST_ADDRESS,
    EXPORT_HANDLE,
};

using ASEntity = std::tuple<BufferEntity, VkAccelerationStructureKHR, VkDeviceAddress>;
enum AS_ENTITY_UID : int
{
    BUFFER_ENTITY = 0,
    AS,
    DEVICE_ADDRESS
};

enum GRAPHICS_PIPELINE_SEMANTIC : int
{
    NORMAL = 0,
    WIREFRAME,
    GRAPHICS_PIPELINE_SEMANTIC_SIZE
};

class VkContext
{
    class Impl;

public:
    VkContext() = delete;
    VkContext(
        const Window &window,
        const std::vector<const char *> &instanceValidationLayers,
        const std::set<std::string> &instanceExtensions,
        const std::vector<const char *> deviceExtensions);
    VkContext(const VkContext &) = delete;
    VkContext &operator=(const VkContext &) = delete;
    VkContext(VkContext &&) noexcept = default;
    VkContext &operator=(VkContext &&) noexcept = default;

    ~VkContext();

    void createSwapChain();

    // generic to swapchain image and non-swapchain images
    VkRenderPass createSwapChainRenderPass();
    // renderPass, imagecount
    VkFramebuffer createFramebuffer(
        const std::string &name,
        VkRenderPass renderPass,
        const std::vector<VkImageView> &colors,
        const VkImageView depth,
        const VkImageView stencil,
        uint32_t width,
        uint32_t height);

    void initDefaultCommandBuffers();

    std::vector<CommandBufferEntity> createGraphicsCommandBuffers(
        const std::string &name,
        uint32_t count,
        uint32_t inflightCount,
        VkFenceCreateFlags flags);

    std::vector<CommandBufferEntity> createTransferCommandBuffers(
        const std::string &name,
        uint32_t count,
        uint32_t inflightCount,
        VkFenceCreateFlags flags);

    void BeginRecordCommandBuffer(CommandBufferEntity &cmdBuffer);
    void EndRecordCommandBuffer(CommandBufferEntity &cmdBuffer);

    BufferEntity createPersistentBuffer(
        const std::string &name,
        VkDeviceSize bufferSizeInBytes,
        VkBufferUsageFlags bufferUsageFlag,
        VkMemoryPropertyFlags requiredMemoryProperties,
        VkMemoryPropertyFlags preferredMemoryProperties,
        bool mapping = true);

    BufferEntity createStagingBuffer(
        const std::string &name,
        VkDeviceSize bufferSizeInBytes);

    // device local buffer
    BufferEntity createDeviceLocalBuffer(
        const std::string &name,
        VkDeviceSize bufferSizeInBytes,
        VkBufferUsageFlags bufferUsageFlag);

    // buffer specially for SBT
    std::tuple<BufferEntity, VkStridedDeviceAddressRegionKHR> createShaderBindTableBuffer(
        const std::string &name,
        VkDeviceSize bufferSizeInBytes,        // not consider alignment
        VkDeviceSize alignedBufferSizeInBytes, // consider alignment
        VkDeviceSize alignedStrideSizeInBytes, // in case of multiple ray miss stages, considering aligneded bytes
        bool mapping);

    // most generic api which should be used when none of the other three buffer creation apis could apply to
    BufferEntity createBuffer(
        const std::string &name,
        VkDeviceSize bufferSizeInBytes,
        VkBufferUsageFlags bufferUsageFlag,
        VkSharingMode bufferSharingMode,
        VmaAllocationCreateFlags memoryAllocationFlag,
        VkMemoryPropertyFlags requiredMemoryProperties,
        VkMemoryPropertyFlags preferredMemoryProperties,
        VmaMemoryUsage memoryUsage,
        bool mapping = false);

    BufferEntity createExportableBuffer(
        const std::string &name,
        VkDeviceSize bufferSizeInBytes,
        VkBufferUsageFlags bufferUsageFlag,
        VkSharingMode bufferSharingMode,
        bool mapping = false);

    ImageEntity createImage(
        const std::string &name,
        VkImageType imageType,
        VkFormat format,
        VkExtent3D extent,
        uint32_t textureMipLevelCount,
        uint32_t textureLayersCount,
        VkSampleCountFlagBits textureMultiSampleCount,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags memoryFlags,
        bool generateMips);

    void createExportableImage(
        const std::string &name,
        VkImageType imageType,
        VkFormat format,
        VkExtent3D extent,
        uint32_t textureMipLevelCount,
        VkSampleCountFlagBits textureMultiSampleCount,
        VkImageUsageFlags usage);

    std::tuple<VkSampler> createSampler(const std::string &name);

    // uint32_t: set id
    std::vector<VkDescriptorSetLayout> createDescriptorSetLayout(std::vector<std::vector<VkDescriptorSetLayoutBinding>> &setBindings);

    VkDescriptorPool createDescriptorSetPool(
        const std::unordered_map<VkDescriptorType, uint32_t> &dsBudgets,
        uint32_t dsCap = 100);

    std::tuple<std::unordered_map<GRAPHICS_PIPELINE_SEMANTIC, VkPipeline>, VkPipelineLayout> createGraphicsPipeline(
        std::unordered_map<VkShaderStageFlagBits, std::tuple<VkShaderModule,
                                                             const char *,
                                                             // std::string, /** dangling pointer issues */
                                                             const VkSpecializationInfo *>>
            vsShaderEntities,
        const std::vector<VkDescriptorSetLayout> &dsLayouts,
        const std::vector<VkPushConstantRange> &pushConstants,
        const VkRenderPass &renderPass);

    std::tuple<VkPipeline, VkPipelineLayout> createComputePipeline(
        std::unordered_map<VkShaderStageFlagBits, std::tuple<VkShaderModule,
                                                             const char *,
                                                             // std::string, /** dangling pointer issues */
                                                             const VkSpecializationInfo *>>
            vsShaderEntities,
        const std::vector<VkDescriptorSetLayout> &dsLayouts,
        const std::vector<VkPushConstantRange> &pushConstants);

    std::tuple<VkPipeline, VkPipelineLayout, std::vector<VkRayTracingShaderGroupCreateInfoKHR>> createRayTracingPipeline(
        std::unordered_map<VkShaderStageFlagBits, std::tuple<VkShaderModule,
                                                             const char *,
                                                             // std::string, /** dangling pointer issues */
                                                             const VkSpecializationInfo *>>
            vsShaderEntities,
        const std::vector<VkDescriptorSetLayout> &dsLayouts,
        const std::vector<VkPushConstantRange> &pushConstants);

    std::unordered_map<VkDescriptorSetLayout *, std::vector<VkDescriptorSet>> allocateDescriptorSet(
        const VkDescriptorPool pool,
        const std::unordered_map<VkDescriptorSetLayout *, uint32_t> &dsAllocation);

    void bindBufferToDescriptorSet(
        const VkBuffer bufferHandle,
        VkDeviceSize offset,
        VkDeviceSize sizeInBytes,
        VkDescriptorSet descriptorSetToBind,
        VkDescriptorType descriptorSetType,
        uint32_t descriptorSetBindingPoint = 0);

    // uint32_t dstArrayElement = 0 useful for async io case
    void bindTextureToDescriptorSet(
        const std::vector<ImageEntity> &images,
        VkDescriptorSet descriptorSetToBind,
        VkDescriptorType descriptorSetType,
        uint32_t dstArrayElement = 0,
        uint32_t descriptorSetBindingPoint = 0);

    void bindSamplerToDescriptorSet(
        const std::vector<VkSampler> &samplers,
        VkDescriptorSet descriptorSetToBind,
        VkDescriptorType descriptorSetType,
        uint32_t descriptorSetBindingPoint = 0);

    void writeBuffer(
        const BufferEntity &stagingBuffer,
        const BufferEntity &deviceLocalBuffer,
        const CommandBufferEntity &cmdBuffer,
        const void *rawData,
        uint32_t sizeInBytes,
        uint32_t srcOffset = 0,
        uint32_t dstOffset = 0);

    void writeImage(
        const ImageEntity &image,
        const BufferEntity &stagingBuffer,
        const CommandBufferEntity &cmdBuffer,
        void *rawData);

    void submitWriteImageCommand(
        ImageEntity &image,
        BufferEntity &stagingBuffer,
        CommandBufferEntity &cmdBuffer,
        void *rawData,
        const std::vector<VkSemaphore> &semaphoresToWait,
        std::optional<VkPipelineStageFlags> waitStage,
        const std::vector<VkSemaphore> &semaphoresToSignal);

    // // simple version which does not involve transfer<->graphics queue ownership transfer
    // // cmdBuffer: all the commands record there
    // // image: which will generate mipmap pyrimad from
    void generateMipmaps(
        const ImageEntity &image,
        const CommandBufferEntity &cmdBuffer);

    VkInstance getInstance() const;
    VkDevice getLogicDevice() const;
    VmaAllocator getVmaAllocator() const;
    VkQueue getGraphicsComputeQueue() const;
    VkQueue getPresentationQueue() const;

    VkPhysicalDevice getSelectedPhysicalDevice() const;
    VkPhysicalDeviceProperties getSelectedPhysicalDeviceProp() const;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR getSelectedPhysicalDeviceRayTracingProperties() const;

    VkSurfaceKHR getSurfaceKHR() const;

    uint32_t getGraphicsComputeQueueFamilyIndex() const;
    uint32_t getPresentQueueFamilyIndex() const;

    VkPhysicalDeviceVulkan12Features getVk12FeatureCaps() const;

    VkSwapchainKHR getSwapChain() const;
    VkExtent2D getSwapChainExtent() const;

    // per-frame rendering op
    const std::vector<VkImage> &getSwapChainImages() const;
    const std::vector<VkImageView> &getSwapChainImageViews() const;
    // legacy sync io (still needs graphics)
    const CommandBufferEntity &getCommandBufferForIO() const;
    // graphics without transfer
    const CommandBufferEntity &getCommandBufferForMipmapOnly() const;
    // non-graphics ops
    const CommandBufferEntity &getCommandBufferForTransferOnly() const;
    // get command buffer for active frame: fits non pre-recorded commandbuffer
    std::pair<uint32_t, CommandBufferEntity> getCommandBufferForRendering() const;
    std::vector<CommandBufferEntity> &getCommandBufferForRenderingForAllSwapChains();

    // let application to access the tracy
    TracyVkCtx getTracyContext() const;

    void advanceCommandBuffer();

    void submitCommand();

    void present(uint32_t swapChainImageIndex);

    uint32_t getSwapChainImageIndexToRender() const;

    // memory barrier: explicitly control access to buffer and image subresource ranges.
    // Image memory barriers can also be used to define image layout transitions or a queue family ownership transfer
    // queue family ownership transfer operation:

    // A queue family ownership transfer consists of two distinct parts:
    // Release exclusive ownership from the source queue family: thread 1
    // Acquire exclusive ownership for the destination queue family: thread 2
    // a pass through
    // An application must ensure that these operations occur in the correct order
    // by defining an execution dependency between them, e.g. using a semaphore.

    // typedef struct VkImageMemoryBarrier {
    //     VkStructureType            sType;
    //     const void*                pNext;
    //     VkAccessFlags              srcAccessMask;
    //     VkAccessFlags              dstAccessMask;
    //     VkImageLayout              oldLayout;
    //     VkImageLayout              newLayout;
    //     uint32_t                   srcQueueFamilyIndex;
    //     uint32_t                   dstQueueFamilyIndex;
    //     VkImage                    image;
    //     VkImageSubresourceRange    subresourceRange;
    // } VkImageMemoryBarrier;

    // // Provided by VK_VERSION_1_3
    // typedef struct VkImageMemoryBarrier2 {
    //     VkStructureType            sType;
    //     const void*                pNext;
    //     VkPipelineStageFlags2      srcStageMask;
    //     VkAccessFlags2             srcAccessMask;
    //     VkPipelineStageFlags2      dstStageMask;
    //     VkAccessFlags2             dstAccessMask;
    //     VkImageLayout              oldLayout;
    //     VkImageLayout              newLayout;
    //     uint32_t                   srcQueueFamilyIndex;
    //     uint32_t                   dstQueueFamilyIndex;
    //     VkImage                    image;
    //     VkImageSubresourceRange    subresourceRange;
    // } VkImageMemoryBarrier2;

    // step1: release part of queue family ownership transfer
    void releaseQueueFamilyOwnership(
        const CommandBufferEntity &cmdBuffer,
        const ImageEntity &image,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex);

    // step2: acquire part of queue family ownership transfer
    void acquireQueueFamilyOwnership(
        const CommandBufferEntity &cmdBuffer,
        const ImageEntity &image,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex);

    // features chains
    // now is to toggle features selectively
    // enable features
    static VkPhysicalDeviceFeatures sPhysicalDeviceFeatures; // cannot fly without sPhysicalDeviceFeatures2
    static VkPhysicalDeviceFeatures2 sPhysicalDeviceFeatures2;

    static VkPhysicalDeviceVulkan11Features sEnable11Features;
    static VkPhysicalDeviceVulkan12Features sEnable12Features;
    static VkPhysicalDeviceVulkan13Features sEnable13Features;
    static VkPhysicalDeviceFragmentDensityMapFeaturesEXT sFragmentDensityMapFeatures;
    // for ray-tracing
    static VkPhysicalDeviceAccelerationStructureFeaturesKHR sAccelStructFeatures;
    static VkPhysicalDeviceRayTracingPipelineFeaturesKHR sRayTracingPipelineFeatures;
    static VkPhysicalDeviceRayQueryFeaturesKHR sRayQueryFeatures;

private:
    std::unique_ptr<Impl> _pimpl;
};
