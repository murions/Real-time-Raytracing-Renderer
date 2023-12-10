#include "Renderer.h"
#include <optix_function_table_definition.h>

extern "C" char embedded_ptx_code[];

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	void* data;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	cudaTextureObject_t data;
};
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	TriangleMeshSBTData data;
};
Renderer::~Renderer() {
}
Renderer::Renderer(Model *model, Texture *background) : model(model), background(background) {
	launchParams.frame.frameCount = 0;
	launchParams.frame.maxTracingDepth = 8;
	
	denoiserSettings.enableProgressiveRefinement = true;
	launchParams.frame.enableProgressiveRefinement = true;
	denoiserSettings.enableDenoiser = false;
	launchParams.frame.enableDenoiser = false;
	launchParams.frame.enablePBR = true;
	launchParams.frame.useRoughnessTexture = false;
	launchParams.frame.useMetallicTexture = false;
	launchParams.frame.enableBSDF = true;
	
	initOptix();
	
	std::cout << "creating optix context..." << std::endl;
	createContext();
	
	std::cout << "setting up module..." << std::endl;
	createModule();
	
	std::cout << "creating raygen programs..." << std::endl;
	createRaygenPrograms();
	std::cout << "creating miss programs..." << std::endl;
	createMissPrograms();
	std::cout << "creating hitgroup programs..." << std::endl;
	createHitgroupPrograms();
	
	launchParams.traversable = buildAccel();
	
	std::cout << "setting up optix pipeline..." << std::endl;
	createPipeline();
	
	createTextures();
	
	std::cout << "building SBT..." << std::endl;
	buildSBT();
	launchParamsBuffer.alloc(sizeof(launchParams));
	std::cout << "context, module, pipeline, etc, all set up..." << std::endl;
	
	std::cout << PRINT_GREEN;
	std::cout << "Optix 7 fully set up!" << std::endl;
	std::cout << PRINT_RESET << std::endl;
}

void Renderer::createTextures() {
	int numTex = (int)model->textures.size();

	textureArrays.resize(numTex);
	textureObjects.resize(numTex);
	if (background) {
		cudaResourceDesc bgRes_Desc = {};
		cudaChannelFormatDesc bgChannel_desc;
		int32_t bgWidth = background->resolution.x;
		int32_t bgHeight = background->resolution.y;
		int32_t numBgComp = 4;
		int32_t bgPitch = bgWidth * numBgComp * sizeof(uint8_t);
		bgChannel_desc = cudaCreateChannelDesc<uchar4>();
		CUDA_CHECK(MallocArray(&bgArray, &bgChannel_desc, bgWidth, bgHeight));
		CUDA_CHECK(Memcpy2DToArray(bgArray, 0, 0, background->pixel, bgPitch, bgPitch, bgHeight, cudaMemcpyHostToDevice));
		bgRes_Desc.resType = cudaResourceTypeArray;
		bgRes_Desc.res.array.array = bgArray;
		cudaTextureDesc bgTex_desc = {};
		bgTex_desc.addressMode[0] = cudaAddressModeWrap;
		bgTex_desc.addressMode[1] = cudaAddressModeWrap;
		bgTex_desc.filterMode = cudaFilterModeLinear;
		bgTex_desc.readMode = cudaReadModeNormalizedFloat;
		bgTex_desc.normalizedCoords = 1;
		bgTex_desc.maxAnisotropy = 1;
		bgTex_desc.maxMipmapLevelClamp = 99;
		bgTex_desc.minMipmapLevelClamp = 0;
		bgTex_desc.mipmapFilterMode = cudaFilterModePoint;
		bgTex_desc.borderColor[0] = 1;
		bgTex_desc.sRGB = 0;
		CUDA_CHECK(CreateTextureObject(&bgObject, &bgRes_Desc, &bgTex_desc, nullptr));
	}
	
	for (int texID = 0; texID < numTex; texID++) {
		Texture* texture;
		texture = model->textures[texID];
	
		cudaResourceDesc res_Desc = {};
	
		cudaChannelFormatDesc channel_desc;
		int32_t width = texture->resolution.x;
		int32_t height = texture->resolution.y;
		int32_t numComp = 4;
		int32_t pitch = width * numComp * sizeof(uint8_t);
		channel_desc = cudaCreateChannelDesc<uchar4>();
	
		cudaArray_t& pixelArray = textureArrays[texID];
		CUDA_CHECK(MallocArray(&pixelArray, &channel_desc, width, height));
		
		CUDA_CHECK(Memcpy2DToArray(pixelArray, 0, 0, texture->pixel, pitch, pitch, height, cudaMemcpyHostToDevice));
	
		res_Desc.resType = cudaResourceTypeArray;
		res_Desc.res.array.array = pixelArray;
	
		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1;
		tex_desc.sRGB = 1;
		if (texID != 0)
			tex_desc.sRGB = 0;
	
		cudaTextureObject_t cuda_tex = 0;
		CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_Desc, &tex_desc, nullptr));
		textureObjects[texID] = cuda_tex;
	}
}

OptixTraversableHandle Renderer::buildAccel() {
	const int numMeshes = (int)model->meshes.size();

	vertexBuffer.resize(numMeshes);
	normalBuffer.resize(numMeshes);
	texcoordBuffer.resize(numMeshes);
	indexBuffer.resize(numMeshes);
	
	OptixTraversableHandle asHandle{ 0 };
	
	std::vector<OptixBuildInput> triangleInput(numMeshes);
	std::vector<CUdeviceptr> d_vertices(numMeshes);
	std::vector<CUdeviceptr> d_indices(numMeshes);
	std::vector<uint32_t> triangleInputFlags(numMeshes);
	
	for (int i = 0; i < numMeshes; i++) {
		TriangleMesh& mesh = *model->meshes[i];
		vertexBuffer[i].alloc_and_upload(mesh.vertex);
		indexBuffer[i].alloc_and_upload(mesh.index);
		if (!mesh.normal.empty())
			normalBuffer[i].alloc_and_upload(mesh.normal);
		if (!mesh.texcoord.empty())
			texcoordBuffer[i].alloc_and_upload(mesh.texcoord);
	
		triangleInput[i] = {};
		triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	
		d_vertices[i] = vertexBuffer[i].d_pointer();
		d_indices[i] = indexBuffer[i].d_pointer();
	
		triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
		triangleInput[i].triangleArray.numVertices = (int)mesh.vertex.size();
		triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];
	
		triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[i].triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
		triangleInput[i].triangleArray.numIndexTriplets = (int)mesh.index.size();
		triangleInput[i].triangleArray.indexBuffer = d_indices[i];
	
		triangleInputFlags[i] = { 0 };
	
		triangleInput[i].triangleArray.flags = &triangleInputFlags[i];
		triangleInput[i].triangleArray.numSbtRecords = 1;
		triangleInput[i].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	
	OptixAccelBufferSizes blasBufferSizes;	// BLAS: Bottom Level Acceleration Structure
	OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions, triangleInput.data(), (int)numMeshes, &blasBufferSizes));
	
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t));
	
	OptixAccelEmitDesc emitDesc;
	emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitDesc.result = compactedSizeBuffer.d_pointer();
	
	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);
	
	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
	
	OPTIX_CHECK(optixAccelBuild(optixContext, 0, &accelOptions, triangleInput.data(), (int)numMeshes, tempBuffer.d_pointer(), tempBuffer.sizeInBytes, outputBuffer.d_pointer(), outputBuffer.sizeInBytes, &asHandle, &emitDesc, 1));
	CUDA_SYNC_CHECK();
	
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1);
	accelStructureBuffer.alloc(compactedSize);
	OPTIX_CHECK(optixAccelCompact(optixContext, 0, asHandle, accelStructureBuffer.d_pointer(), accelStructureBuffer.sizeInBytes, &asHandle));
	CUDA_SYNC_CHECK();
	
	outputBuffer.free();
	tempBuffer.free();
	compactedSizeBuffer.free();
	
	return asHandle;
}

void Renderer::initOptix() {
	std::cout << "initializing optix..." << std::endl;

	cudaFree(0);
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	if (numDevices == 0)
		throw std::runtime_error("no CUDA capable devices found!");
	std::cout << PRINT_CYAN << "found " << numDevices << " CUDA devices " << PRINT_RESET << std::endl;
	
	OPTIX_CHECK(optixInit());
	std::cout << PRINT_GREEN << "successfully initialized optix." << PRINT_RESET << std::endl;
}
static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}
void Renderer::createContext() {
	const int deviceID = 0;
	CUDA_CHECK(SetDevice(deviceID));
	CUDA_CHECK(StreamCreate(&stream));

	cudaGetDeviceProperties(&deviceProps, deviceID);
	std::cout << "running on device: " << deviceProps.name << std::endl;
	
	CUresult cuRes = cuCtxGetCurrent(&cudaContext);
	if (cuRes != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: error code %d\n", cuRes);
	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
}
void Renderer::createModule() {
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
	
	pipelineLinkOptions.maxTraceDepth = 16;
	
	const std::string ptxCode = embedded_ptx_code;
	
	char log[2048];
	size_t sizeof_log = sizeof(log);
#if (OPTIX_VERSION%10000)/100 < 5
	OPTIX_CHECK(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &module));
			if (sizeof_log > 1)	std::cout << log << std::endl;
#else
	OPTIX_CHECK(optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), log, &sizeof_log, &module));
			if (sizeof_log > 1)	std::cout << log << std::endl;
#endif
}
void Renderer::createRaygenPrograms() {
	raygenPGs.resize(1);

	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	pgDesc.raygen.module = module;
	pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";
	
	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0]));
	if(sizeof_log > 1)	std::cout << log << std::endl;
}
void Renderer::createMissPrograms() {
	missPGs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	pgDesc.miss.module = module;
	pgDesc.miss.entryFunctionName = "__miss__radiance";
	
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)	std::cout << log << std::endl;
	
	pgDesc.miss.entryFunctionName = "__miss__shadow";
	
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)	std::cout << log << std::endl;
}
void Renderer::createHitgroupPrograms() {
	hitgroupPGs.resize(RAY_TYPE_COUNT);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	
	OptixProgramGroupOptions pgOptions = {};
	OptixProgramGroupDesc pgDesc = {};
	pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	pgDesc.hitgroup.moduleCH = module;
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	pgDesc.hitgroup.moduleAH = module;
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
	
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPGs[RADIANCE_RAY_TYPE]));
	if (sizeof_log > 1)	std::cout << log << std::endl;
	
	pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
	pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";
	
	OPTIX_CHECK(optixProgramGroupCreate(optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPGs[SHADOW_RAY_TYPE]));
	if (sizeof_log > 1)	std::cout << log << std::endl;
}
void Renderer::createPipeline() {
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : raygenPGs)
		programGroups.push_back(pg);
	for (auto pg : missPGs)
		programGroups.push_back(pg);
	for (auto pg : hitgroupPGs)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(), (int)programGroups.size(), log, &sizeof_log, &pipeline));
	if (sizeof_log > 1)	std::cout << log << std::endl;
	
	OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 32 * 1024, 32 * 1024, 32 * 1024, 1));
	if (sizeof_log > 1)	std::cout << log << std::endl;
}
void Renderer::buildSBT() {
	std::vector<RaygenRecord> raygenRecords;
	CUDABuffer hitgroupRecordsBuffer;
	CUDABuffer missRecordsBuffer;
	CUDABuffer raygenRecordsBuffer;
	for (int i = 0; i < raygenPGs.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
		rec.data = nullptr;
		raygenRecords.push_back(rec);
	}
	raygenRecordsBuffer.alloc_and_upload(raygenRecords);
	sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

	std::vector<MissRecord> missRecords;
	for (int i = 0; i < missPGs.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
		rec.data = bgObject;
		missRecords.push_back(rec);
	}
	missRecordsBuffer.alloc_and_upload(missRecords);
	sbt.missRecordBase = missRecordsBuffer.d_pointer();
	sbt.missRecordStrideInBytes = sizeof(MissRecord);
	sbt.missRecordCount = (int)missRecords.size();
	
	std::vector<HitgroupRecord> hitgroupRecords;
	int numObjects = (int)model->meshes.size();
	for (int i = 0; i < numObjects; i++) 
		for(int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++){
			auto mesh = model->meshes[i];
			HitgroupRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[rayID], &rec));
			rec.data.color = mesh->diffuse;
			rec.data.specular = mesh->specular;
			rec.data.emission = mesh->emission;
			if (mesh->diffuseTexID >= 0) {
				rec.data.hasATexture = true;
				rec.data.texture = textureObjects[mesh->diffuseTexID];
			}
			else
			{
				rec.data.hasATexture = false;
			}
			if (mesh->metellicTexID >= 0) {
				rec.data.hasMTexture = true;
				rec.data.metellicT = textureObjects[mesh->metellicTexID];
			}
			else
			{
				rec.data.hasMTexture = false;
			}
			if (mesh->roughnessTexID >= 0) {
				rec.data.hasRTexture = true;
				rec.data.roughnessT = textureObjects[mesh->roughnessTexID];
			}
			else
			{
				rec.data.hasRTexture = false;
			}
			if (mesh->emissionTexID >= 0) {
				rec.data.hasETexture = true;
				rec.data.emissionT = textureObjects[mesh->emissionTexID];
			}
			else
			{
				rec.data.hasETexture = false;
			}
			rec.data.vertex = (glm::vec3*)vertexBuffer[i].d_pointer();
			rec.data.index = (glm::ivec3*)indexBuffer[i].d_pointer();
			rec.data.normal = (glm::vec3*)normalBuffer[i].d_pointer();
			rec.data.texcoord = (glm::vec2*)texcoordBuffer[i].d_pointer();
			hitgroupRecords.push_back(rec);
		}
	hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
	sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
	sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}
void Renderer::render() {
	launchParams.frame.frameCount++;
	if (launchParams.frame.size.x == 0) return;

	if (!denoiserSettings.enableProgressiveRefinement)
		launchParams.frame.enableProgressiveRefinement = false;
	launchParamsBuffer.upload(&launchParams, 1);
	
	OPTIX_CHECK(optixLaunch(pipeline, stream, launchParamsBuffer.d_pointer(), launchParamsBuffer.sizeInBytes, &sbt, launchParams.frame.size.x, launchParams.frame.size.y, 1));


	OptixDenoiserParams denoiserParams;
#if OPTIX_VERSION > 70500
	denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
	denoiserParams.hdrIntensity = (CUdeviceptr)0;
	if (denoiserSettings.enableProgressiveRefinement)
		denoiserParams.blendFactor = 1.0f / (launchParams.frame.frameCount);
	else
		denoiserParams.blendFactor = 0.0f;

	OptixImage2D inputLayer;
	inputLayer.data = renderBuffer.d_pointer();
	inputLayer.width = launchParams.frame.size.x;
	inputLayer.height = launchParams.frame.size.y;
	inputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
	inputLayer.pixelStrideInBytes = sizeof(float4);
	inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	
	OptixImage2D outputLayer;
	outputLayer.data = denoiserBuffer.d_pointer();
	outputLayer.width = launchParams.frame.size.x;
	outputLayer.height = launchParams.frame.size.y;
	outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
	outputLayer.pixelStrideInBytes = sizeof(float4);
	outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;
	
	if (denoiserSettings.enableDenoiser) {
#if OPTIX_VERSION >= 70300
		OptixDenoiserGuideLayer denoiserGuideLayer = {};

		OptixDenoiserLayer denoiserLayer = {};
		denoiserLayer.input = inputLayer;
		denoiserLayer.output = outputLayer;
	
		OPTIX_CHECK(optixDenoiserInvoke(denoiser,
			/*stream*/0,
			&denoiserParams,
			denoiserState.d_pointer(),
			denoiserState.sizeInBytes,
			&denoiserGuideLayer,
			&denoiserLayer, 1,
			/*inputOffsetX*/0,
			/*inputOffsetY*/0,
			denoiserScratch.d_pointer(),
			denoiserScratch.sizeInBytes));
#else
		OPTIX_CHECK(optixDenoiserInvoke(denoiser,
			/*stream*/0,
			&denoiserParams,
			denoiserState.d_pointer(),
			denoiserState.size(),
			&inputLayer, 1,
			/*inputOffsetX*/0,
			/*inputOffsetY*/0,
			&outputLayer,
			denoiserScratch.d_pointer(),
			denoiserScratch.size()));
#endif
	}
	else
		cudaMemcpy((void*)outputLayer.data, (void*)inputLayer.data, outputLayer.width * outputLayer.height * sizeof(float4), cudaMemcpyDeviceToDevice);

	CUDA_SYNC_CHECK();
}
void Renderer::setCamera(const Camera& camera) {
	//launchParams.frame.frameCount = 0;
	lastSetCamera = camera;
	launchParams.camera.position = camera.pos;
	launchParams.camera.direction = normalize(camera.at - camera.pos);
	const float cosFovy = 0.66f;
	const float aspect = launchParams.frame.size.x / (float)launchParams.frame.size.y;
	launchParams.camera.horizontal = cosFovy * aspect * normalize(cross(launchParams.camera.direction, camera.up));
	launchParams.camera.vertical = cosFovy * normalize(cross(launchParams.camera.horizontal, launchParams.camera.direction));
}
void Renderer::resize(const glm::ivec2& newSize) {
	if (newSize.x == 0 | newSize.y == 0) return;

	if (denoiser)
		OPTIX_CHECK(optixDenoiserDestroy(denoiser));
	OptixDenoiserOptions denoiserOptions = {};
#if OPTIX_VERSION >= 70300
	OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR, &denoiserOptions, &denoiser));
#else
	denoiserOptions.inputKind = OPTIX_DENOISER_INPUT_RGB;

#if OPTIX_VERSION < 70100
	// these only exist in 7.0, not 7.1
	denoiserOptions.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

	OPTIX_CHECK(optixDenoiserCreate(optixContext, &denoiserOptions, &denoiser));
	OPTIX_CHECK(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, NULL, 0));
#endif

	// .. then compute and allocate memory resources for the denoiser
	OptixDenoiserSizes denoiserReturnSizes;
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, newSize.x, newSize.y,
		&denoiserReturnSizes));

#if OPTIX_VERSION < 70100
	denoiserScratch.resize(denoiserReturnSizes.recommendedScratchSizeInBytes);
#else
	denoiserScratch.resize(std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
		denoiserReturnSizes.withoutOverlapScratchSizeInBytes));
#endif
	denoiserState.resize(denoiserReturnSizes.stateSizeInBytes);

	denoiserBuffer.resize(newSize.x * newSize.y * sizeof(float4));
	renderBuffer.resize(newSize.x * newSize.y * sizeof(float4));
	
	launchParams.frame.size = newSize;
	launchParams.frame.colorBuffer = (float4*)renderBuffer.d_pointer();
	launchParams.frame.frameCount = 0;

 	setCamera(lastSetCamera);

	OPTIX_CHECK(optixDenoiserSetup(
		denoiser, 0, newSize.x, newSize.y, denoiserState.d_pointer(), denoiserState.sizeInBytes, denoiserScratch.d_pointer(), denoiserScratch.sizeInBytes
	));
}
glm::ivec2 Renderer::getWindowSize() {
	return glm::ivec2(launchParams.frame.size.x, launchParams.frame.size.y);
}
void Renderer::setMaxDepth(int i) {
	if (i > 10000)
		std::cout << PRINT_YELLOW << "Warnings: maxPathTracingDepth is too large!" << PRINT_RESET << std::endl;
	launchParams.frame.maxTracingDepth = i;
}
int Renderer::getMaxDepth() {
	return launchParams.frame.maxTracingDepth;
}
void Renderer::enablePR() {
	launchParams.frame.enableProgressiveRefinement = true;
	denoiserSettings.enableProgressiveRefinement = true;
}
void Renderer::disablePR() {
	launchParams.frame.enableProgressiveRefinement = false;
	denoiserSettings.enableProgressiveRefinement = false;
}
void Renderer::enableDenoiser() {
	denoiserSettings.enableDenoiser = true;
	launchParams.frame.enableDenoiser = true;
}
void Renderer::disableDenoiser() {
	denoiserSettings.enableDenoiser = false;
	launchParams.frame.enableDenoiser = false;
}
int Renderer::getFrameCount() {
	return launchParams.frame.enableProgressiveRefinement ? launchParams.frame.frameCount : 0;
}
void Renderer::downloadPixels(float4 h_pixels[]) {
	denoiserBuffer.download(h_pixels, launchParams.frame.size.x * launchParams.frame.size.y);
}

void Renderer::cleanupState() {
	accelStructureBuffer.free();
	OPTIX_CHECK(optixPipelineDestroy(pipeline));
	for (int i = 0; i < raygenPGs.size(); i++)
		OPTIX_CHECK(optixProgramGroupDestroy(raygenPGs[i]));
	for (int i = 0; i < missPGs.size(); i++)
		OPTIX_CHECK(optixProgramGroupDestroy(missPGs[i]));
	for (int i = 0; i < hitgroupPGs.size(); i++)
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroupPGs[i]));
	OPTIX_CHECK(optixModuleDestroy(module));
	OPTIX_CHECK(optixDeviceContextDestroy(optixContext));

	CUDA_CHECK(Free(reinterpret_cast<void*>(sbt.raygenRecord)));
	CUDA_CHECK(Free(reinterpret_cast<void*>(sbt.missRecordBase)));
	CUDA_CHECK(Free(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
	launchParamsBuffer.free();
	renderBuffer.free();
	denoiserBuffer.free();
	std::cout << PRINT_GREEN << "Safely exit pathTracer!" << PRINT_RESET << std::endl;
}
void Renderer::rebuildOptix(Model* model) {
	this->model = model;

	launchParams.traversable = buildAccel(); 
	createTextures();
	buildSBT();
	launchParamsBuffer.alloc(sizeof(launchParams));

	launchParamsBuffer.upload(&launchParams, 1);
	OPTIX_CHECK(optixLaunch(pipeline, stream, launchParamsBuffer.d_pointer(), launchParamsBuffer.sizeInBytes, &sbt, launchParams.frame.size.x, launchParams.frame.size.y, 1));
	std::cout << "context, module, pipeline, etc, all set up" << std::endl;

	std::cout << PRINT_GREEN;
	std::cout << "Optix 7 fully set up" << std::endl;
	std::cout << PRINT_RESET;
}
void Renderer::rebuildOptix(Texture* background) {
	this->background = background;

	launchParams.traversable = buildAccel();
	createTextures();
	buildSBT();
	launchParamsBuffer.alloc(sizeof(launchParams));

	launchParamsBuffer.upload(&launchParams, 1);
	OPTIX_CHECK(optixLaunch(pipeline, stream, launchParamsBuffer.d_pointer(), launchParamsBuffer.sizeInBytes, &sbt, launchParams.frame.size.x, launchParams.frame.size.y, 1));
	std::cout << "context, module, pipeline, etc, all set up" << std::endl;

	std::cout << PRINT_GREEN;
	std::cout << "Optix 7 fully set up" << std::endl;
	std::cout << PRINT_RESET;
}