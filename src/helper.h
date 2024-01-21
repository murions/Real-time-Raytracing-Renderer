#ifndef __CUDACC__
#define __CUDACC__
#endif
// 邪道去除报错
#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <vector_types.h>
#include <optix_device.h>
#include "LaunchParams.h"

struct Sphere {
	glm::vec3 position;
	float radius;
	glm::vec3 albedo;
	glm::vec3 specular;
	float smoothness;
	glm::vec3 emission;
};

extern "C" __constant__ LaunchParams optixLaunchParams;

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1) {
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}
static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1) {
	const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}
template<typename T> static __forceinline__ __device__ T* getPRD() {
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}
static __forceinline__ __device__ float3 convert_vec3(glm::vec3 v) {
	return make_float3(v.x, v.y, v.z);
}
static __forceinline__ __device__ float4 convert_vec4(glm::vec4 v) {
	return make_float4(v.x, v.y, v.z, v.w);
}
static __forceinline__ __device__ glm::vec3 convert_float3(float3 v) {
	return glm::vec3(v.x, v.y, v.z);
}
static __forceinline__ __device__ glm::vec4 convert_float4(float4 v) {
	return glm::vec4(v.x, v.y, v.z, v.w);
}
static __forceinline__ __device__ glm::vec3 sResult(glm::vec3 res) {
	return glm::vec3(glm::saturate(res.x), glm::saturate(res.y), glm::saturate(res.z));
}
static __forceinline__ __device__ glm::vec2 getSphericalCoord(float3 dir) {
	glm::vec3 rayDir = glm::normalize(convert_float3(dir));
	float theta = asin(rayDir.y) / PI;
	float phi = atan2(rayDir.z, rayDir.x) / PI * 0.5f;
	return glm::vec2(phi + 0.5f, theta + 0.5f);
}
static __forceinline__ __device__ unsigned int wang_hash(unsigned int seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}
// 产生0-1的随机浮点数
static __forceinline__ __device__ float getRandomFloat(unsigned int seed) {
	return (wang_hash(seed) & 0xFFFFFF) / 16777216.0f;
}
static __forceinline__ __device__ float RadicalInverse(unsigned int bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555) << 1u) | ((bits & 0xAAAAAAAA) >> 1u);
	bits = ((bits & 0x33333333) << 2u) | ((bits & 0xCCCCCCCC) >> 2u);
	bits = ((bits & 0x0F0F0F0F) << 4u) | ((bits & 0xF0F0F0F0) >> 4u);
	bits = ((bits & 0x00FF00FF) << 8u) | ((bits & 0xFF00FF00) >> 8u);
	return  float(bits) * 2.3283064365386963e-10;
}
static __forceinline__ __device__ glm::vec2 Hammersley(unsigned int i, unsigned int N) {
	return glm::vec2(float(i) / float(N), RadicalInverse(i));
}

// 计算切线空间
static __forceinline__ __device__ glm::mat3 getTangentSpace(glm::vec3 normal) {
	glm::vec3 helper = glm::vec3(0, 1, 0);
	if (abs(normal.y) > 0.999f)
		helper = glm::vec3(0, 0, -1);
	glm::vec3 tangent = glm::normalize(glm::cross(normal, helper));
	glm::vec3 binormal = glm::normalize(glm::cross(normal, tangent));
	return glm::mat3(tangent, binormal, normal);
}
// 余弦半球采样
static __forceinline__ __device__ glm::vec3 sampleHemiSphere(glm::vec3 normal, unsigned int seed) {
	float cosTheta = sqrtf(glm::max(0.001f, getRandomFloat(seed)));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	float phi = 2.0f * PI * getRandomFloat(seed);
	glm::vec3 tangentSpaceDir = glm::vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

	return tangentSpaceDir * getTangentSpace(normal);

}
// 余弦与均匀混合采样
static __forceinline__ __device__ glm::vec3 sampleHemiSphereAlpha(glm::vec3 normal, float alpha, unsigned int seed) {
	float cosTheta = powf(glm::max(0.001f, getRandomFloat(seed)), 1.0f / (1.0f + alpha));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	float phi = 2.0f * PI * getRandomFloat(seed);
	glm::vec3 tangentSpaceDir = glm::vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

	return tangentSpaceDir * getTangentSpace(normal);

}
// GGX重要性采样
static __forceinline__ __device__ glm::vec3 sampleHemiSphereGGX(glm::vec3 normal, float alpha, float alpha2, unsigned int seed) {
	float rx = getRandomFloat(seed);
	float ry = getRandomFloat(seed);
	float cosTheta = sqrtf((1.0f - ry) / (1.0f + ry * (alpha2 - 1.0f)));
	float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
	float phi = 2.0f * PI * rx;
	glm::vec3 tangentSpaceDir = glm::vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

	return tangentSpaceDir * getTangentSpace(normal);
}
static __forceinline__ __device__ glm::vec3 calcF0(const glm::vec3 albedo, const float metallic) {
	float s_metallic = glm::saturate(metallic);
	return s_metallic * albedo + (1.0f - s_metallic) * glm::vec3(0.04f);
}
static __forceinline__ __device__ glm::vec3 calcF(const glm::vec3 F0, const float VdotH) {
	float tmp = (-5.55473f * VdotH - 6.98316f) * VdotH;
	return F0 + (glm::vec3(1.0f) - F0) * exp2(tmp);
}
static __forceinline__ __device__ float calcD(const float NdotH, const float alpha2) {
	const float NdotH2 = NdotH * NdotH;
	const float tmp = NdotH2 * (alpha2 - 1.0f) + 1.0f;
	return glm::saturate(alpha2 / (PI * tmp * tmp));
}
static __forceinline__ __device__ float calcG(const float NdotL, const float NdotV, const float alpha2) {
	const float k = alpha2 / 2.0f;

	const float ggx0 = NdotL / (NdotV * (1.0f - k) + k);
	const float ggx1 = NdotV / (NdotL * (1.0f - k) + k);

	return glm::saturate(ggx0 * ggx1);
}
static __forceinline__ __device__ float SchlickFresnel(float u) {
	float m = glm::clamp(1.0f - u, 0.0f, 1.0f);
	float m2 = m * m;
	return m2 * m2 * m;
}
static __forceinline__ __device__ float calcD2(const float NdotH, const float alpha2) {
	if (alpha2 >= 1.0f)
		return PI_INV;
	const float NdotH2 = NdotH * NdotH;
	const float tmp = NdotH2 * (alpha2 - 1.0f) + 1.0f;
	return glm::saturate((alpha2 - 1.0f) / (PI * logf(alpha2) * tmp));
}
static __forceinline__ __device__ float SmithG_GGX(float NdotV, float alphaG2) {
	float b = NdotV * NdotV;
	return 1.0f / (NdotV + sqrtf(alphaG2 + b - alphaG2 * b));
}