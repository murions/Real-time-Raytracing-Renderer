#ifndef RENDERER_H
#define RENDERER_H

#define _CRT_SECURE_NO_WARNINGS
#include <optix_host.h>
#include "CUDABuffer.h"
#include "LaunchParams.h"
#include "Model.h"

class Camera {
public:
	glm::vec3 pos, at, up;
};

class Renderer {
public:
	Renderer(Model *model, Texture *background);
	~Renderer();
	void render();
	void resize(const glm::ivec2 &size);
	void downloadPixels(float4 h_pixels[]);
	void setCamera(const Camera& camera);
	glm::ivec2 getWindowSize();
	void setMaxDepth(int i);
	int getMaxDepth();
	void enablePR();
	void disablePR();
	void enableDenoiser();
	void disableDenoiser();
	int getFrameCount();

	void cleanupState();
	void rebuildOptix(Model* model);
	void rebuildOptix(Texture* background);
protected:
	void initOptix();
	void createContext();
	void createModule();
	void createRaygenPrograms();
	void createMissPrograms();
	void createHitgroupPrograms();
	void createPipeline();
	void buildSBT();

	OptixTraversableHandle buildAccel();
	void createTextures();
protected:
	CUcontext cudaContext;
	CUstream stream;
	cudaDeviceProp deviceProps;

	OptixDeviceContext optixContext;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};

	OptixModule module;
	OptixModuleCompileOptions moduleCompileOptions = {};

	std::vector<OptixProgramGroup> raygenPGs;
	//CUDABuffer raygenRecordsBuffer;
	std::vector<OptixProgramGroup> missPGs;
	//CUDABuffer missRecordsBuffer;
	std::vector<OptixProgramGroup> hitgroupPGs;
	//CUDABuffer hitgroupRecordsBuffer;
	OptixShaderBindingTable sbt = {};

	CUDABuffer launchParamsBuffer;

	Camera lastSetCamera;

	Model* model;
	Texture* background;
	HDRTexture* bg;

	std::vector<CUDABuffer> vertexBuffer;
	std::vector<CUDABuffer> normalBuffer;
	std::vector<CUDABuffer> texcoordBuffer;
	std::vector<CUDABuffer> indexBuffer;
	CUDABuffer accelStructureBuffer;

	std::vector<cudaArray_t> textureArrays;
	std::vector<cudaTextureObject_t> textureObjects;
	cudaTextureObject_t bgObject;
	cudaArray_t bgArray;

	DenoiserSettings denoiserSettings;
	CUDABuffer denoiserBuffer;
	OptixDenoiser denoiser = nullptr;
	CUDABuffer denoiserScratch;
	CUDABuffer denoiserState;

	CUDABuffer renderBuffer;
public:
	LaunchParams launchParams;
};


#endif // RENDERER_H