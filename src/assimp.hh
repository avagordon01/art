#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <string>
#include <vector>
#include <array>

struct mesh {
    std::vector<unsigned> indices;
    std::vector<std::array<float, 3>> vertices;
    mesh(const std::string& filename) {
        Assimp::Importer importer;

        const aiScene* scene = importer.ReadFile(filename,
            aiProcess_Triangulate |
            aiProcess_JoinIdenticalVertices |
            aiProcess_SortByPType
        );

        if (!scene) {
            throw std::runtime_error("assimp failed to load \"" + filename + "\" with error: " + importer.GetErrorString());
        }

        if (scene->mNumMeshes == 0) {
            throw std::runtime_error("error! no mesh in the file");
        }
        if (scene->mNumMeshes > 1) {
            throw std::runtime_error("error! more than one mesh in the file");
        }
        const aiMesh* mesh = scene->mMeshes[0];
        for (size_t i = 0; i < mesh->mNumFaces; i++) {
            const aiFace& face = mesh->mFaces[i];
            for (size_t j = 0; j < face.mNumIndices; j++) {
                unsigned index = face.mIndices[j];
                indices.push_back(index);
            }
        }
        for (size_t i = 0; i < mesh->mNumVertices; i++) {
            const aiVector3D& vertex = mesh->mVertices[i];
            vertices.push_back({vertex.x, vertex.y, vertex.z});
        }
    }
};
