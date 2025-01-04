#pragma once

// #define GLM_SWIZZLE // for xyz Swizzle Operators
// #define GLM_SWIZZLE_XYZW
// #define GLM_SWIZZLE_STQP

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <misc.h>
#include <renderPassBase.h>

class CullFustrum : public RenderPassBase,
                    public VkContextAccessor,
                    public SceneAccessor,
                    public CameraAccessor,
                    public DescriptorPoolAccessor
{
public:
    CullFustrum()
    {
    }

    ~CullFustrum()
    {
    }

    virtual void setContext(VkContext *ctx) override
    {
        _ctx = ctx;
    }

    virtual const VkContext &context() const override
    {
        return *_ctx;
    }

    virtual void setScene(std::shared_ptr<Scene> scene) override
    {
        _scene = scene;
    }

    virtual const Scene &scene() const override
    {
        return *_scene;
    }

    virtual const CameraBase &camera() const override
    {
        return *_camera;
    }

    virtual void setCamera(const CameraBase *camera) override
    {
        _camera = camera;
    }

    virtual void setDescriptorPool(const VkDescriptorPool dsPool) override
    {
        _dsPool = dsPool;
    }

    virtual const VkDescriptorPool descriptorPool() const override
    {
        return _dsPool;
    }

    // algorithm specific
    // indirectDrawBuffer to be cull against
    inline void setIndirectDrawBuffer(BufferEntity *idb)
    {
        _indirectDrawBuffer = idb;
    }

    virtual void finalizeInit() override
    {
        initShaderModules();
        createDescriptorSetLayout();
        initComputePipeline();
        allocateDescriptorSets();

        initFustrumBuffer();
        initMeshBoundingBoxBuffer();
        initCulledIndirectDrawBuffer();
        // step1: bind res to ds, then later on bind ds to the compute pipeline
        bindResourceToDescriptorSets();

        uploadResource();
    }

    inline BufferEntity getCulledIDR() const
    {
        return this->_culledIndirectDrawBuffer;
    }

    inline BufferEntity getCulledIDRCount() const
    {
        return this->_culledIndirectDrawCountBuffer;
    }

    virtual void execute(CommandBufferEntity cmd, int currentFrameId) override
    {
        auto commandBufferHandle = std::get<1>(cmd);
        // for buffer memory barrier (two idr which shader writes into)
        // shared compute/graphics queue family
        auto commandQueueFamilyIndex = std::get<3>(cmd);
        auto computePipelineHandle = std::get<0>(_computePipelineEntity);
        auto computePipelineLayout = std::get<1>(_computePipelineEntity);
        auto vmaAllocator = _ctx->getVmaAllocator();

        const auto &fustrumBuffers = std::get<0>(_fustrumBuffers);
        ASSERT(
            currentFrameId >= 0 && currentFrameId < fustrumBuffers.size(),
            "execute:: currentFrameId should be in a valid range");

        auto vmaAllocation = std::get<1>(fustrumBuffers[currentFrameId]);
        auto mappedMemory = std::get<3>(fustrumBuffers[currentFrameId]);

        // update uniform buffer
        auto frustrum = _camera->fustrumPlanes();
        if (mappedMemory)
        {
            memcpy(mappedMemory, &frustrum, sizeof(Fustrum));
        }
        else
        {
            // racing condition due to vmaAllocator
            void *mappedMemory{nullptr};
            VK_CHECK(vmaMapMemory(vmaAllocator, vmaAllocation, &mappedMemory));
            memcpy(mappedMemory, &frustrum, sizeof(Fustrum));
            // memcpy(mappedMemory, &ubo, sizeof(ubo));
            vmaUnmapMemory(vmaAllocator, vmaAllocation);
        }
        // update push constants
        const auto numMeshesToCull = uint32_t(_bb.size());
        vkCmdBindPipeline(commandBufferHandle, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineHandle);
        vkCmdPushConstants(commandBufferHandle, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numMeshesToCull);

        // resource and ds to the shaders of this pipeline
        // IDR = 0,
        // BOUNDING_BOX,
        // FUSTRUMS,
        // CULLED_IDR,
        // CULLED_IDR_COUNTER,
        // DESC_LAYOUT_SEMANTIC_SIZE
        vkCmdBindDescriptorSets(commandBufferHandle,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 0, 1,
                                &_descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::IDR]][0],
                                0,
                                nullptr);
        vkCmdBindDescriptorSets(commandBufferHandle,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 1, 1,
                                &_descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX]][0],
                                0,
                                nullptr);
        vkCmdBindDescriptorSets(commandBufferHandle,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 2, 1,
                                &_descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::FUSTRUMS]][currentFrameId],
                                0,
                                nullptr);
        vkCmdBindDescriptorSets(commandBufferHandle,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 3, 1,
                                &_descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::CULLED_IDR]][0],
                                0,
                                nullptr);
        vkCmdBindDescriptorSets(commandBufferHandle,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                computePipelineLayout, 4, 1,
                                &_descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER]][0],
                                0,
                                nullptr);
        // thread group x,y,z
        vkCmdDispatch(commandBufferHandle, (numMeshesToCull / 64) + 1, 1, 1);

        const auto culledIDRBufferHandle = std::get<0>(_culledIndirectDrawBuffer);
        const auto culledIDRBufferSizeInBytes = std::get<4>(_culledIndirectDrawBuffer);
        const auto culledIDRCountBufferHandle = std::get<0>(_culledIndirectDrawCountBuffer);
        const auto culledIDRCountBufferSizeInBytes = std::get<4>(_culledIndirectDrawCountBuffer);

        // from shader write to idr buffer read
        std::array<VkBufferMemoryBarrier, 2> bufferMemoryBarriers{
            VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                .srcQueueFamilyIndex = commandQueueFamilyIndex,
                .dstQueueFamilyIndex = commandQueueFamilyIndex,
                .buffer = culledIDRBufferHandle,
                .size = culledIDRBufferSizeInBytes,
            },
            VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                .srcQueueFamilyIndex = commandQueueFamilyIndex,
                .dstQueueFamilyIndex = commandQueueFamilyIndex,
                .buffer = culledIDRCountBufferHandle,
                .size = culledIDRCountBufferSizeInBytes,
            },
        };

        vkCmdPipelineBarrier(
            commandBufferHandle,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0,
            0, nullptr,                                                         // memory
            (uint32_t)bufferMemoryBarriers.size(), bufferMemoryBarriers.data(), // buffer
            0, nullptr                                                          // image
        );

        // cpu testing
        for (const auto &bb : _bb)
        {
            for (int i = 0; i < 1; ++i)
            {
                const auto p = frustrum.planes[i];
                glm::vec3 n(p);
                glm::vec3 extents(bb.extents[0], bb.extents[1],
                                  bb.extents[2]);
                glm::vec3 center(bb.center[0], bb.center[1],
                                 bb.center[2]);
                float radiusEffective = 0.5 * glm::dot(glm::abs(n), extents);
                float distFromCenter = glm::dot(center, n) + p.w;
                const bool shouldBeCulled = (distFromCenter <= -radiusEffective);

                if (shouldBeCulled)
                {
                    log(Level::Info, "shouldBeCulled: ", shouldBeCulled);
                }
            }
        }
    }

private:
    void initShaderModules()
    {
        ASSERT(_ctx, "vk context should be defined");
        auto logicalDevice = _ctx->getLogicDevice();
        const auto shadersPath = getAssetPath();
        const auto computeShaderPath = shadersPath + "/cullFustrum.comp";
        _csShaderModule = createShaderModule(
            logicalDevice,
            computeShaderPath,
            "main",
            "cullFustrum.comp");
    }

    void initFustrumBuffer()
    {
        ASSERT(_ctx, "vk context should be defined");
        const auto numFramesInFlight = _ctx->getSwapChainImageViews().size();

        // sizeof(Fustrum) * numFramesInFlight
        std::vector<BufferEntity> buffers;
        buffers.reserve(numFramesInFlight);

        for (size_t i = 0; i < numFramesInFlight; ++i)
        {
            buffers.emplace_back(_ctx->createPersistentBuffer(
                "Uniform Fustrum Buffer" + std::to_string(i),
                numFramesInFlight * sizeof(Fustrum),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT));
        }

        _fustrumBuffers = std::make_tuple(buffers, numFramesInFlight);
    }

    void initMeshBoundingBoxBuffer()
    {
        ASSERT(_ctx, "vk context should be defined");
        ASSERT(_scene, "scene should be defined");
        _bb.reserve(_scene->meshes.size());

        for (const auto &mesh : _scene->meshes)
        {

            _bb.emplace_back(BoundingBox{
                .center = glm::vec4(
                    mesh.center[0],
                    mesh.center[1],
                    mesh.center[2],
                    1.0f),
                .extents = glm::vec4(
                    mesh.extents[0],
                    mesh.extents[1],
                    mesh.extents[2],
                    1.0f)});

            log(Level::Info, "BoundingBox Center: ", _bb.back().center);
            log(Level::Info, "BoundingBox Extents: ", _bb.back().extents);
        }
        // build up the combo buffer
        const auto bytesize = sizeof(BoundingBox) * _bb.size();
        _meshBoundBoxComboStagingBuffer = _ctx->createStagingBuffer(
            "Combo BoundingBox Staging Buffer",
            bytesize);

        _meshBoundBoxComboDeviceBuffer = _ctx->createDeviceLocalBuffer(
            "Combo BoundingBox Device Local Buffer",
            bytesize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    void initCulledIndirectDrawBuffer()
    {
        ASSERT(_ctx, "vk context should be defined");
        ASSERT(_indirectDrawBuffer, "indirect draw buffer should be defined");
        const auto bufferSizeInBytes = std::get<4>(*_indirectDrawBuffer);

        // VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT: specifies that the buffer can be used to retrieve a buffer device address via vkGetBufferDeviceAddress
        // and use that address to access the buffer’s memory from a shader.
        // VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT must be provided
        _culledIndirectDrawBuffer = _ctx->createDeviceLocalBuffer(
            "Culled Indirect Draw Buffer",
            bufferSizeInBytes,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

        _culledIndirectDrawCountBuffer = _ctx->createDeviceLocalBuffer(
            "Culled Indirect Draw Counter Buffer",
            sizeof(uint32_t),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    }

    // refer to section in cs
    // #define IDR_SETID 0
    // #define BOUNDINGBOX_SETID 1
    // #define FUSTRUMS_SETID 2
    // #define CULLED_IDR 3
    // #define CULLED_IDR_COUNTER 4

    enum DESC_LAYOUT_SEMANTIC : int
    {
        IDR = 0,
        BOUNDING_BOX,
        FUSTRUMS,
        CULLED_IDR,
        CULLED_IDR_COUNTER,
        DESC_LAYOUT_SEMANTIC_SIZE
    };

    void createDescriptorSetLayout()
    {
        ASSERT(_ctx, "vk context should be defined");
        std::vector<std::vector<VkDescriptorSetLayoutBinding>> setBindings(DESC_LAYOUT_SEMANTIC_SIZE);

        setBindings[DESC_LAYOUT_SEMANTIC::IDR].resize(1);
        setBindings[DESC_LAYOUT_SEMANTIC::IDR][0].binding = 0; // depends on the shader: set 0, binding = 0
        setBindings[DESC_LAYOUT_SEMANTIC::IDR][0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        setBindings[DESC_LAYOUT_SEMANTIC::IDR][0].descriptorCount = 1;
        setBindings[DESC_LAYOUT_SEMANTIC::IDR][0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        setBindings[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX].resize(1);
        setBindings[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX][0].binding = 0; // depends on the shader: set 0, binding = 0
        setBindings[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX][0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        setBindings[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX][0].descriptorCount = 1;
        setBindings[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX][0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        setBindings[DESC_LAYOUT_SEMANTIC::FUSTRUMS].resize(1);
        setBindings[DESC_LAYOUT_SEMANTIC::FUSTRUMS][0].binding = 0; // depends on the shader: set 0, binding = 0
        setBindings[DESC_LAYOUT_SEMANTIC::FUSTRUMS][0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        setBindings[DESC_LAYOUT_SEMANTIC::FUSTRUMS][0].descriptorCount = 1;
        setBindings[DESC_LAYOUT_SEMANTIC::FUSTRUMS][0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR].resize(1);
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR][0].binding = 0; // depends on the shader: set 0, binding = 0
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR][0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR][0].descriptorCount = 1;
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR][0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER].resize(1);
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER][0].binding = 0; // depends on the shader: set 0, binding = 0
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER][0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER][0].descriptorCount = 1;
        setBindings[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER][0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        _descriptorSetLayouts = _ctx->createDescriptorSetLayout(setBindings);
    }

    void initComputePipeline()
    {
        const std::string entryPoint{"main"};
        // layout(push_constant) uniform PushConsts {
        // 	uint count;
        // } MeshesToCull;
        _computePipelineEntity = _ctx->createComputePipeline(
            {{VK_SHADER_STAGE_COMPUTE_BIT,
              std::make_tuple(_csShaderModule, entryPoint.c_str(), nullptr)}},
            _descriptorSetLayouts,
            {{
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = sizeof(uint32_t),
            }});
    }

    void allocateDescriptorSets()
    {
        ASSERT(_ctx, "vk context should be defined");
        ASSERT(_dsPool, "descriptorset pool should be defined");
        const auto numFramesInFlight = _ctx->getSwapChainImageViews().size();
        _descriptorSets = _ctx->allocateDescriptorSet(_dsPool,
                                                      {{&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::IDR],
                                                        1},
                                                       {&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX],
                                                        1},
                                                       {&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::FUSTRUMS],
                                                        numFramesInFlight},
                                                       {&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::CULLED_IDR],
                                                        1},
                                                       {&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER],
                                                        1}});
    }

    void bindResourceToDescriptorSets()
    {
        ASSERT(_ctx, "vk context should be defined");
        auto logicalDevice = _ctx->getLogicDevice();
        const auto numFramesInFlight = _ctx->getSwapChainImageViews().size();

        // idr as input (readonly)
        {
            const auto &dstSets = _descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::IDR]];
            ASSERT(dstSets.size() == 1, "IDR descriptor set size is 1");
            const auto bufferSizeInBytes = std::get<4>(*_indirectDrawBuffer);
            _ctx->bindBufferToDescriptorSet(
                std::get<0>(*_indirectDrawBuffer),
                0,
                bufferSizeInBytes,
                dstSets[0],
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0);
        }

        // boundingbox buffer
        {
            const auto &dstSets = _descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::BOUNDING_BOX]];
            ASSERT(dstSets.size() == 1, "boundingbox descriptor set size is 1");
            const auto bufferSizeInBytes = std::get<4>(_meshBoundBoxComboDeviceBuffer);
            _ctx->bindBufferToDescriptorSet(
                std::get<0>(_meshBoundBoxComboDeviceBuffer),
                0,
                bufferSizeInBytes,
                dstSets[0],
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0);
        }

        // fustrum uniform buffer
        {
            const auto &dstSets = _descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::FUSTRUMS]];
            ASSERT(dstSets.size() == numFramesInFlight, "FUSTRUMS descriptor set size should equal # of frames in flight");
            const auto &fustrumBuffers = std::get<0>(_fustrumBuffers);
            ASSERT(fustrumBuffers.size() == numFramesInFlight, "fustrumBuffers' size should equal # of frames in flight");

            for (size_t i = 0; i < numFramesInFlight; i++)
            {
                const auto bufferSizeInBytes = std::get<4>(fustrumBuffers[i]);
                _ctx->bindBufferToDescriptorSet(
                    std::get<0>(fustrumBuffers[i]),
                    0,
                    bufferSizeInBytes,
                    _descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::FUSTRUMS]][i],
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    0);
            }
        }

        // culled idr buffer (writable)
        {
            const auto &dstSets = _descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::CULLED_IDR]];
            ASSERT(dstSets.size() == 1, "culled idr descriptor set size is 1");
            const auto bufferSizeInBytes = std::get<4>(_culledIndirectDrawBuffer);
            _ctx->bindBufferToDescriptorSet(
                std::get<0>(_culledIndirectDrawBuffer),
                0,
                bufferSizeInBytes,
                dstSets[0],
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0);
        }

        // culled idr counter buffer (writable)
        {
            const auto &dstSets = _descriptorSets[&_descriptorSetLayouts[DESC_LAYOUT_SEMANTIC::CULLED_IDR_COUNTER]];
            ASSERT(dstSets.size() == 1, "culled idr counter descriptor set size is 1");
            const auto bufferSizeInBytes = std::get<4>(_culledIndirectDrawCountBuffer);
            _ctx->bindBufferToDescriptorSet(
                std::get<0>(_culledIndirectDrawCountBuffer),
                0,
                bufferSizeInBytes,
                dstSets[0],
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                0);
        }
    }

    void uploadResource()
    {
        ASSERT(_ctx, "vk context should be defined");
        auto logicalDevice = _ctx->getLogicDevice();
        // this io belongs to the graphics queue, so no explict ownership acq and release needed
        auto cmdBuffersForIO = _ctx->getCommandBufferForIO();
        auto graphicsComputeQueue = _ctx->getGraphicsComputeQueue();
        _ctx->BeginRecordCommandBuffer(cmdBuffersForIO);
        _ctx->writeBuffer(
            _meshBoundBoxComboStagingBuffer,
            _meshBoundBoxComboDeviceBuffer,
            cmdBuffersForIO,
            reinterpret_cast<const void *>(_bb.data()),
            _bb.size() * sizeof(BoundingBox),
            0,
            0);
        _ctx->EndRecordCommandBuffer(cmdBuffersForIO);

        const auto uploadCmdBuffer = std::get<1>(cmdBuffersForIO);
        const auto uploadCmdBufferFence = std::get<2>(cmdBuffersForIO);

        const VkPipelineStageFlags flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
        // wait for no body
        VkSubmitInfo submitInfo{};
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = VK_NULL_HANDLE;
        // only useful when having waitsemaphore
        submitInfo.pWaitDstStageMask = &flags;
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &uploadCmdBuffer;
        // no body needs to be notified
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = VK_NULL_HANDLE;

        VK_CHECK(vkResetFences(logicalDevice, 1, &uploadCmdBufferFence));
        VK_CHECK(vkQueueSubmit(graphicsComputeQueue, 1, &submitInfo, uploadCmdBufferFence));
        // sync io
        const auto result = vkWaitForFences(logicalDevice, 1, &uploadCmdBufferFence, VK_TRUE,
                                            100000000000);
        if (result == VK_TIMEOUT)
        {
            vkDeviceWaitIdle(logicalDevice);
        }
    }
    // ownership be careful
    BufferEntity *_indirectDrawBuffer{nullptr};
    BufferEntity _culledIndirectDrawBuffer;
    // vkCmdDrawIndexedIndirectCount vs vkCmdDrawIndexedIndirect
    // vkCmdDrawIndexedIndirectCount: extra buffer for draw counter, which is filled in in the gpu
    BufferEntity _culledIndirectDrawCountBuffer;
    // refer to frame in fight
    std::tuple<std::vector<BufferEntity>, size_t> _fustrumBuffers;
    // interleave all the bounding box of meshes into one big buffer.
    BufferEntity _meshBoundBoxComboDeviceBuffer;
    BufferEntity _meshBoundBoxComboStagingBuffer;
    // life cycle of host buffer matters when gpu uploading process is done
    std::vector<BoundingBox> _bb;
    // for pipeline and binding resource
    std::vector<VkDescriptorSetLayout> _descriptorSetLayouts;
    VkShaderModule _csShaderModule{VK_NULL_HANDLE};
    std::tuple<VkPipeline, VkPipelineLayout> _computePipelineEntity;
    // do i use individual ds pool ?
    std::unordered_map<VkDescriptorSetLayout *, std::vector<VkDescriptorSet>> _descriptorSets;
};