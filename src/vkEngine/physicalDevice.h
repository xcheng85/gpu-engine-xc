#pragma once

#ifndef __ANDROID__
#define VK_NO_PROTOTYPES // for volk
#include "volk.h"
#endif

class PhysicalDevice final
{
public:
    PhysicalDevice() = delete;
    explicit PhysicalDevice(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

    inline VkPhysicalDevice getVKHandle() const {
        return _vkHandle;
    }

    inline const VkSurfaceCapabilitiesKHR& getSurfaceCaps() const {
        return _surfaceCapabilities;
    }
    
private:
    VkPhysicalDevice _vkHandle;
    VkSurfaceCapabilitiesKHR _surfaceCapabilities;
};