#ifndef MODEL_H
#define MODEL_H

#include "math_calc.h"
#include <vector>
#include <string>
#include <vector_types.h>

struct TriangleMesh
{
	std::vector<glm::vec3> vertex;
	std::vector<glm::vec3> normal;
	std::vector<glm::vec2> texcoord;
	std::vector<glm::ivec3> index;

	glm::vec3 diffuse;
	glm::vec3 specular;
	glm::vec3 emission;
	int diffuseTexID{ -1 };
	int roughnessTexID{ -1 };
	int metellicTexID{ -1 };
	int emissionTexID{ -1 };
};
struct Texture
{
	~Texture() {
		if(pixel)	delete[] pixel;
	}

	uint32_t* pixel{ nullptr };
	glm::ivec2 resolution{ -1 };
};
struct HDRTexture
{
	~HDRTexture() {
		if (pixel)	delete[] pixel;
	}

	float4* pixel{ nullptr };
	glm::ivec2 resolution{ -1 };
};
class Model {
public:
	std::vector<TriangleMesh*> meshes;
	std::vector<Texture*> textures;
	BoundingBox box;
public:
	~Model() {
		for (auto mesh : meshes)
			delete mesh;
		for (auto texture : textures)
			delete texture;
	}
	static Model* loadOBJ(const std::string& path);
	static Texture* loadBackground(const std::string& path);
	static HDRTexture* loadHDRTexture(const std::string& path);
};

#endif // MODEL_H
