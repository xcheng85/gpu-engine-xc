#pragma once

#include <context.h>
#include <scene.h>      // for scene accessor
#include <cameraBase.h> // for camera accessor

class RenderPassBase
{
public:
    RenderPassBase() = default;
    virtual ~RenderPassBase() = default;
    virtual void finalizeInit() = 0;
    virtual void execute(CommandBufferEntity cmd, int frameIndex) = 0;

protected:
    VkContext *_ctx{nullptr};
    std::shared_ptr<Scene> _scene{nullptr};
    const CameraBase *_camera{nullptr};
    // descriptorset pool, every render pass needs to allocate ds from it
    VkDescriptorPool _dsPool{VK_NULL_HANDLE};
};

class VkContextAccessor
{
public:
    virtual void setContext(VkContext *) = 0;
    virtual const VkContext &context() const = 0;
};

// mixing class to access scene
class SceneAccessor
{
public:
    virtual void setScene(std::shared_ptr<Scene>) = 0;
    virtual const Scene &scene() const = 0;
};

class CameraAccessor
{
public:
    virtual void setCamera(const CameraBase *) = 0;
    virtual const CameraBase &camera() const = 0;
};

// design for shared ds pool
class DescriptorPoolAccessor
{
public:
    virtual void setDescriptorPool(const VkDescriptorPool) = 0;
    virtual const VkDescriptorPool descriptorPool() const = 0;
};