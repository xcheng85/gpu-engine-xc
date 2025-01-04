#pragma once

#include <string>
#include <vector>
#include <numeric>
#include <stb_image.h>
#include <misc.h>

#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

// mimic
// struct VkDrawIndexedIndirectCommand {
//    uint32_t    indexCount;
//    uint32_t    instanceCount;
//    uint32_t    firstIndex;
//    int32_t     vertexOffset;
//    uint32_t    firstInstance;
//};

struct IndirectDrawDef1
{
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    uint32_t vertexOffset;
    uint32_t firstInstance;
    uint32_t meshId;
    int materialIndex;
};

inline std::ostream &operator<<(std::ostream &os, const IndirectDrawDef1 &id)
{
    log(Level::Info, "meshId: ", id.meshId);
    log(Level::Info, "indexCount: ", id.indexCount);
    log(Level::Info, "instanceCount: ", id.instanceCount);
    log(Level::Info, "firstIndex: ", id.firstIndex);
    log(Level::Info, "vertexOffset: ", id.vertexOffset);
    log(Level::Info, "firstInstance: ", id.firstInstance);
    log(Level::Info, "materialIndex: ", id.materialIndex);
    return os;
}

//
// struct Vertex {
//    vec3f pos;
//    vec2f texCoord;
//    uint32_t material;
//
//    void transform(const mat4x4f &m) {
//        auto newp = MatrixMultiplyVector4x4(m, vec4f(std::array<float, 4>{
//                pos[COMPONENT::X],
//                pos[COMPONENT::Y],
//                pos[COMPONENT::Z],
//                1.0
//        }));
//
//        pos = vec3f(std::array{
//                newp[COMPONENT::X],
//                newp[COMPONENT::Y],
//                newp[COMPONENT::Z]
//        });
//    }
//};

struct Vertex
{
    float vx;
    float vy;
    float vz;
    float ux;
    float uy;
    uint32_t material;

    void transform(const glm::mat4 &m)
    {
        auto newp = m * glm::vec4(vx, vy, vz, 1.0);
        vx = newp[0];
        vy = newp[1];
        vz = newp[2];
    }
};

struct BoundingBox
{
    glm::vec4 center;
    glm::vec4 extents;
};

struct Mesh
{
    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
    int32_t materialIdx{-1};
    glm::vec3 minAABB{(std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)(), (std::numeric_limits<float>::max)()};
    glm::vec3 maxAABB{-(std::numeric_limits<float>::max)(), -(std::numeric_limits<float>::max)(), -(std::numeric_limits<float>::max)()};
    glm::vec3 extents;
    glm::vec3 center;
};

// https://github.com/KhronosGroup/glTF/blob/2.0/specification/2.0/schema/material.schema.json
// struct Material : glTFChildOfRootProperty
// struct PBRMetallicRoughness : glTFProperty
struct Material
{
    // refer to PBRMetallicRoughness
    int basecolorTextureId{-1};
    int basecolorSamplerId{-1};
    int metallicRoughnessTextureId{-1};
    glm::vec4 basecolor;
};

#include <ktx.h>
#include <ktxvulkan.h>

class ITexture
{
public:
    virtual ~ITexture()
    {
    }

    virtual void *data()
    {
        return _data;
    };

    inline uint32_t width() const
    {
        return _width;
    };
    inline uint32_t height() const
    {
        return _height;
    }
    inline uint8_t channels() const
    {
        return _channels;
    }

protected:
    void *_data{nullptr};
    int _width{0};
    int _height{0};
    int _channels{0};

    ITexture() {};
};

class Texture : public ITexture
{
public:
    // glb version, no resource ownership
    Texture() = delete;
    explicit Texture(const std::vector<uint8_t> &rawBuffer);
    explicit Texture(unsigned char *rawBuffer, size_t sizeInBytes);
    ~Texture();
};

class TextureKtx : public ITexture
{
public:
    explicit TextureKtx(std::string path);
    explicit TextureKtx(unsigned char *rawBuffer, int sizeInBytes);

    ~TextureKtx();

private:
    ktxTexture *_ktxTexture{nullptr};
};

struct Scene
{
    ~Scene();

    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<std::unique_ptr<Texture>> textures;
    std::vector<IndirectDrawDef1> indirectDraw;
    uint32_t totalVerticesByteSize{0};
    uint32_t totalIndexByteSize{0};
};