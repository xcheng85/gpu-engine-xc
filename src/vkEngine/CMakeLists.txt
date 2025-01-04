MESSAGE(STATUS "${PLATFORM}")

file(GLOB BASE_SRC "*.cpp" "*.hpp" "*.h")
file(GLOB BASE_HEADERS "*.hpp" "*.h")

add_library(gpuVkEngine ${BASE_SRC} ${BASE_HEADERS})
target_include_directories(gpuVkEngine PUBLIC .)
target_include_directories(gpuVkEngine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${gltfsdk_SOURCE_DIR}/GLTFSDK/Inc 
  ${stb_SOURCE_DIR}
  ${fetch_ktx_SOURCE_DIR}/include
  $<INSTALL_INTERFACE:include>
)

message(${stb_SOURCE_DIR})

# volk_headers or volk, refer to the github
target_link_libraries(
  gpuVkEngine
    GLTFSDK
    ${Vulkan_LIBRARIES} 
    volk 
    ${RequiredVulkanSDKLIBS}
    TracyClient
    ktx
)

#glm
target_compile_definitions(gpuVkEngine PUBLIC -DGLM_ENABLE_EXPERIMENTAL)
#-DVK_PRERECORD_COMMANDS)

get_target_property(dirs gpuVkEngine INTERFACE_INCLUDE_DIRECTORIES)
foreach(dir IN LISTS dirs)
    message(STATUS "dir='${dir}'")
endforeach()