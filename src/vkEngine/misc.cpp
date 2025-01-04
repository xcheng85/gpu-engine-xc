#include <filesystem>
#include <fstream>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h> //c
#include <glslang/Public/ResourceLimits.h>    //c++
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <DirStackFileIncluder.h>

#include <misc.h>

std::string getAssetPath()
{
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    return "";
#else
    const auto path = std::filesystem::current_path().parent_path() / "assets";
    return path.string();
#endif
}

std::vector<char> readFile(const std::string &filePath, bool isBinary)
{
    std::ios_base::openmode mode = std::ios::ate;
    if (isBinary)
    {
        mode |= std::ios::binary;
    }
    std::ifstream file(filePath, mode);
    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    if (!isBinary)
    {
        fileSize += 1; // add extra for null char at end
    }
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(buffer.data()), fileSize);
    file.close();
    if (!isBinary)
    {
        buffer[buffer.size() - 1] = '\0';
    }
    return buffer;
}

EShLanguage shaderStageFromFileName(const std::filesystem::path &path)
{
    const auto ext = path.extension().string();
    if (ext == ".vert")
    {
        return EShLangVertex;
    }
    else if (ext == ".frag")
    {
        return EShLangFragment;
    }
    else if (ext == ".comp")
    {
        return EShLangCompute;
    }
    else if (ext == ".rgen")
    {
        return EShLangRayGen;
    }
    else if (ext == ".rmiss")
    {
        return EShLangMiss;
    }
    else if (ext == ".rchit")
    {
        return EShLangClosestHit;
    }
    else
    {
        ASSERT(false, "unsupported shader stage");
    }
}

// 1. load spv as binary, easy
// 2. build from glsl in the runtime; complicated
// data: txt array
// shader stage
// shaderDir for include
// entryPoint: main()
// std::vector<char>: binary array
std::vector<char> glslToSpirv(const std::vector<char> &shaderText,
                              EShLanguage shaderStage,
                              const std::string &shaderDir,
                              const char *entryPoint)
{
    glslang::TShader tmp(shaderStage);
    const char *data = shaderText.data();
    // c style: array + size
    tmp.setStrings(&data, 1);

    glslang::EshTargetClientVersion clientVersion = glslang::EShTargetVulkan_1_3;
    glslang::EShTargetLanguageVersion langVersion = glslang::EShTargetSpv_1_6;
    // raytracing
    if (shaderStage == EShLangRayGen || shaderStage == EShLangAnyHit ||
        shaderStage == EShLangClosestHit || shaderStage == EShLangMiss)
    {
        langVersion = glslang::EShTargetSpv_1_4;
    }

    // use opengl 4.6
    tmp.setEnvInput(glslang::EShSourceGlsl, shaderStage, glslang::EShClientVulkan, 460);
    tmp.setEnvClient(glslang::EShClientVulkan, clientVersion);
    tmp.setEnvTarget(glslang::EShTargetSpv, langVersion);
    // // validate only
    // tmp.setEnvClient(EShClientNone, 0);
    // tmp.setEnvTarget(EShTargetNone, 0);
    tmp.setEntryPoint(entryPoint);
    tmp.setSourceEntryPoint(entryPoint);

    // parsing include in the shader file
    const TBuiltInResource *resources = GetDefaultResources();
    // Message choices for what errors and warnings are given.
    // https://chromium.googlesource.com/external/github.com/KhronosGroup/glslang/+/refs/heads/SPIR-V_1.4/glslang/Public/ShaderLang.h
    EShMessages messages = (EShMessages)(EShMsgDefault | EShMsgSpvRules | EShMsgVulkanRules | EShMsgDebugInfo);

    DirStackFileIncluder includer;
    std::vector<std::string> IncludeDirectoryList;

    const auto sharedShaderPath = getAssetPath();
    IncludeDirectoryList.push_back(sharedShaderPath);

    std::for_each(IncludeDirectoryList.rbegin(), IncludeDirectoryList.rend(), [&includer](const std::string &dir)
                  { includer.pushExternalLocalDirectory(dir); });
    std::string preprocessedGLSL;
    if (!tmp.preprocess(resources, 460, ENoProfile, false, false, messages,
                        &preprocessedGLSL, includer))
    {
        std::cout << "Preprocessing failed for shader: " << std::endl;
        std::cout << std::endl;
        std::cout << tmp.getInfoLog() << std::endl;
        std::cout << tmp.getInfoDebugLog() << std::endl;
        ASSERT(false, "Error occured");
        return std::vector<char>();
    }

    // preprocessedGLSL = removeUnnecessaryLines(preprocessedGLSL);

    const char *preprocessedGLSLStr = preprocessedGLSL.c_str();
    std::cout << preprocessedGLSL << std::endl;

    // without include
    glslang::TShader tshader(shaderStage);
    tshader.setEnvInput(glslang::EShSourceGlsl, shaderStage, glslang::EShClientVulkan, 460);
    tshader.setEnvClient(glslang::EShClientVulkan, clientVersion);
    tshader.setEnvTarget(glslang::EShTargetSpv, langVersion);
    tshader.setEntryPoint(entryPoint);
    tshader.setSourceEntryPoint(entryPoint);
    tshader.setStrings(&preprocessedGLSLStr, 1);

    if (!tshader.parse(resources, 460, false, messages))
    {
        std::cout << "Parsing failed for shader: " << std::endl;
        std::cout << std::endl;
        std::cout << tshader.getInfoLog() << std::endl;
        std::cout << tshader.getInfoDebugLog() << std::endl;
        ASSERT(false, "parse failed");
        return std::vector<char>();
    }

    glslang::SpvOptions spvOptions;

    // use debug settings
    tshader.setDebugInfo(true);
    spvOptions.emitNonSemanticShaderDebugInfo = true;
    spvOptions.emitNonSemanticShaderDebugSource = true;
    spvOptions.generateDebugInfo = true;
    spvOptions.disableOptimizer = true;
    spvOptions.optimizeSize = false;
    spvOptions.stripDebugInfo = false;

    glslang::TProgram program;
    program.addShader(&tshader);
    if (!program.link(messages))
    {
        std::cout << "Parsing failed for shader " << std::endl;
        std::cout << program.getInfoLog() << std::endl;
        std::cout << program.getInfoDebugLog() << std::endl;
        ASSERT(false, "Failed to link shader stage to program");
    }

    std::vector<uint32_t> spirvArtifacts;
    spv::SpvBuildLogger spvLogger;
    // write result to
    glslang::GlslangToSpv(*program.getIntermediate(shaderStage),
                          spirvArtifacts,
                          &spvLogger,
                          &spvOptions);

    std::vector<char> byteCode;
    byteCode.resize(spirvArtifacts.size() * (sizeof(uint32_t) / sizeof(char)));
    std::memcpy(byteCode.data(), spirvArtifacts.data(), byteCode.size());

    return byteCode;
}

VkShaderModule createShaderModule(
    VkDevice logicalDevice,
    const std::string &filePath,
    const std::string &entryPoint,
    const std::string &correlationId)
{
    VkShaderModule res;
    const auto path = std::filesystem::path(filePath);
    const bool isBinary = path.extension().string() == ".spv";
    std::vector<char> data = readFile(filePath, isBinary);
    if (!isBinary)
    {
        data = glslToSpirv(data,
                           shaderStageFromFileName(path),
                           path.parent_path().string(),
                           entryPoint.c_str());
    }
    const VkShaderModuleCreateInfo shaderModule = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = data.size(),
        .pCode = (const uint32_t *)data.data(),
    };
    VK_CHECK(vkCreateShaderModule(logicalDevice, &shaderModule, nullptr, &res));
    return res;
}

// input: shaderModule Meta
// output: to meet the vk api
std::vector<VkPipelineShaderStageCreateInfo> gatherPipelineShaderStageCreateInfos(
    const std::unordered_map<VkShaderStageFlagBits, std::tuple<VkShaderModule, const char *, const VkSpecializationInfo *>> &shaderModuleEntities)
{
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
    shaderStages.reserve(shaderModuleEntities.size());

    for (const auto &[shaderModuleStage, shaderModuleEntity] : shaderModuleEntities)
    {
        const auto shaderModuleHandle = std::get<0>(shaderModuleEntity);
        const auto entryFunctionName = std::get<1>(shaderModuleEntity);
        const auto pSpecializationInfo = std::get<2>(shaderModuleEntity);

        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = shaderModuleStage;
        shaderStageInfo.module = shaderModuleHandle;
        shaderStageInfo.pName = entryFunctionName;
        shaderStageInfo.pSpecializationInfo = pSpecializationInfo;

        shaderStages.emplace_back(shaderStageInfo);
    }
    return shaderStages;
}

// for shader group mapping which is required when creating rt pipeine
uint32_t findShaderStageIndex(
    const std::vector<VkPipelineShaderStageCreateInfo> &shaderStages, 
    const VkShaderModule shaderModule)
{
    if (auto it = std::find_if(std::begin(shaderStages), std::end(shaderStages), [=](const VkPipelineShaderStageCreateInfo& shaderStage){
        return shaderModule == shaderStage.module;
    }); it != std::end(shaderStages))
    {
        return it - std::begin(shaderStages);
    }
    else
    {
        return -1;
    }
}

VkImageViewType getImageViewType(VkImageType imageType)
{
    switch (imageType)
    {
    case VK_IMAGE_TYPE_1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case VK_IMAGE_TYPE_2D:
        return VK_IMAGE_VIEW_TYPE_2D;
    case VK_IMAGE_TYPE_3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    default:
        break;
    }
    log(Level::Error, "unsupported imageType: ", imageType);
    ASSERT(false, "unsupported imageType");
    return VK_IMAGE_VIEW_TYPE_2D;
}

uint32_t get2DImageSizeInBytes(VkExtent2D extent, VkFormat imageFormat)
{
    switch (imageFormat)
    {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return 4 * extent.width * extent.height;
    default:
        break;
    }
    log(Level::Error, "unsupported imageFormat: ", imageFormat);
    ASSERT(false, "unsupported imageFormat");
    return 1;
}

uint32_t get3DImageSizeInBytes(VkExtent3D extent, VkFormat imageFormat)
{
    switch (imageFormat)
    {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return 4 * extent.width * extent.height * extent.depth;
    default:
        break;
    }
    log(Level::Error, "unsupported imageFormat: ", imageFormat);
    ASSERT(false, "unsupported imageFormat");
    return 1;
}