#ifndef LAUNCHPARAMS_H
#define LAUNCHPARAMS_H

#include "optix_add.h"
#include "math_calc.h"

enum{RADIANCE_RAY_TYPE = 0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT};

struct DenoiserSettings
{
	bool enableProgressiveRefinement;
	bool enableDenoiser;
};

struct Frame
{
	float4* colorBuffer;
	glm::vec2 size;
	int frameCount;
	int maxTracingDepth;
	float specular;
	float roughness;
	float metallic;
	float subSurface;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float IOR;
	float transmission;
	bool enableProgressiveRefinement;
	bool enableDenoiser;
	bool enablePBR;
	bool useRoughnessTexture;
	bool useMetallicTexture;
	bool enableBSDF;
	Frame() {
		frameCount = 0;
		maxTracingDepth = 4;
		roughness = 0.0f;
		metallic = 0.0f;
		specular = 0.0f;
		subSurface = 0.0f;
		specularTint = 0.0f;
		anisotropic = 0.0f;
		sheen = 0.0f;
		sheenTint = 0.0f;
		clearcoat = 0.0f;
		clearcoatGloss = 0.0f;
		IOR = 0.0f;
		transmission = 0.0f;
		enableProgressiveRefinement = true;
		enableDenoiser = false;
		enablePBR = true;
		useRoughnessTexture = false;
		useMetallicTexture = false;
	}
};
struct CameraProp
{
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 horizontal;
	glm::vec3 vertical;
};
struct TriangleMeshSBTData {
	glm::vec3 color;
	glm::vec3 specular;
	glm::vec3 emission;
	glm::vec3* vertex;
	glm::vec3* normal;
	glm::vec2* texcoord;
	glm::ivec3* index;
	bool hasATexture;
	bool hasRTexture;
	bool hasMTexture;
	bool hasETexture;
	cudaTextureObject_t texture;
	cudaTextureObject_t roughnessT;
	cudaTextureObject_t metellicT;
	cudaTextureObject_t emissionT;
};

struct LaunchParams
{
	Frame frame;
	CameraProp camera;
	OptixTraversableHandle traversable;
};

#endif // !LAUNCHPARAMS_H