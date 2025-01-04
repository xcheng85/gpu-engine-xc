#pragma once

#include <memory>

#include <GLTFSDK/Deserialize.h>
#include <GLTFSDK/GLBResourceReader.h>
#include <GLTFSDK/GLTF.h>
#include <GLTFSDK/GLTFResourceReader.h>
#include <scene.h>

class GltfBinaryIOReader {
public:
    std::shared_ptr <Scene> read(const std::string &filePath);

    // for android
    std::shared_ptr <Scene> read(const std::vector<char> &binarybuffer);

private:
};
