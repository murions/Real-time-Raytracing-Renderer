#include "helper.h"

struct RadiancePRD
{
	glm::vec3       radiance;
	glm::vec3		viewPoint;
	glm::vec3       energy;
	glm::vec3		lastEnergy;
	glm::vec3       origin;
	glm::vec3       direction;
	unsigned int seed;
	float pad;
};
extern "C" __global__ void __closesthit__radiance() {
	const TriangleMeshSBTData & sbtData = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
	RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();	// prd: Per Ray Data
	glm::vec3 emission = sResult(sbtData.emission);
	//if (emission.x > 0 || emission.y > 0 || emission.z > 0) {
	//	prd.radiance = glm::vec3(min(emission.x, 1.0f), min(emission.y, 1.0f), min(emission.z, 1.0f));
	//	prd.lastEnergy = prd.energy;
	//	prd.energy = glm::vec3(0);
	//	return;
	//}

	const int primID = optixGetPrimitiveIndex();
	const glm::ivec3 index = sbtData.index[primID];
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	glm::vec3 N;
	if (sbtData.normal) {
		N = (1.0f - u - v) * sbtData.normal[index.x] + u * sbtData.normal[index.y] + v * sbtData.normal[index.z];
		N = glm::normalize(N);
	}
	else
	{
		const glm::vec3& A = sbtData.vertex[index.x];
		const glm::vec3& B = sbtData.vertex[index.y];
		const glm::vec3& C = sbtData.vertex[index.z];
		N = glm::normalize(glm::cross(B - A, C - A));
	}

	const glm::vec3 rayDir = glm::normalize(convert_float3(optixGetWorldRayDirection()));

	glm::vec3 specular = sbtData.specular;
	glm::vec3 diffuse = glm::vec3(glm::min(1.0f - specular.x, sbtData.color.x), glm::min(1.0f - specular.y, sbtData.color.y), glm::min(1.0f - specular.z, sbtData.color.z));
	float metallic = optixLaunchParams.frame.metallic;
	float roughness = optixLaunchParams.frame.roughness;
	if (sbtData.texcoord) {
		const glm::vec2 texcoord = (1.0f - u - v) * sbtData.texcoord[index.x] + u * sbtData.texcoord[index.y] + v * sbtData.texcoord[index.z];
		if (sbtData.hasATexture) {

			float4 fromTexture = tex2D<float4>(sbtData.texture, texcoord.x, texcoord.y);
			diffuse = sResult(glm::vec3(fromTexture.x, fromTexture.y, fromTexture.z));
		}
		if (sbtData.hasMTexture && optixLaunchParams.frame.useMetallicTexture) {

			float4 fromTexture = tex2D<float4>(sbtData.metellicT, texcoord.x, texcoord.y);
			metallic = fromTexture.x;
		}
		if (sbtData.hasRTexture && optixLaunchParams.frame.useRoughnessTexture) {

			float4 fromTexture = tex2D<float4>(sbtData.roughnessT, texcoord.x, texcoord.y);
			roughness = fromTexture.x;
		}
		if (sbtData.hasETexture) {

			float4 fromTexture = tex2D<float4>(sbtData.emissionT, texcoord.x, texcoord.y);
			emission = sResult(glm::vec3(fromTexture.x, fromTexture.y, fromTexture.z));
		}
	}

	glm::vec3 newOri = convert_float3(optixGetWorldRayOrigin()) + optixGetRayTmax() * convert_float3(optixGetWorldRayDirection());
	//glm::vec3 newDir = sampleHemiSphere(N, glm::ivec2(optixGetLaunchIndex().x, optixGetLaunchIndex().y));

	//N = convert_float3(optixTransformNormalFromObjectToWorldSpace(convert_vec3(N)));

	glm::vec3 V = -rayDir;
	glm::vec3 F0 = calcF0(diffuse, metallic);
	glm::vec3 kd = glm::vec3(0.5f) * (1.0f - metallic);
	if (!optixLaunchParams.frame.enablePBR) {	// Phong Model
		prd.origin = newOri + 1e-3f * N;
		prd.lastEnergy = prd.energy;
		float roulette = getRandomFloat(prd.seed);
		float diffChance = glm::saturate(diffuse.x + diffuse.y + diffuse.z) / 3.0f + 0.0000001f;
		float specChance = glm::saturate(specular.x + specular.y + specular.z) / 3.0f + 0.0000001f;
		if (roulette < diffChance) {	//diffuse
			prd.direction = sampleHemiSphere(N, prd.seed);
			prd.energy *= diffuse * sbtData.color * (1.0f / diffChance);
		}
		else if (roulette < diffChance + specChance) {	// specular
			float smoothness = powf(1000.0f, optixLaunchParams.frame.roughness * optixLaunchParams.frame.roughness);
			prd.direction = sampleHemiSphereAlpha(glm::reflect(rayDir, N), smoothness, prd.seed);
			float f = 1.0f / (smoothness + 1.0f) + 1.0f;
			prd.energy *= specular * glm::saturate(glm::dot(N, glm::normalize(prd.direction)) * f) * (1.0f / specChance);
		}
		else	// dropout
		{
			prd.energy = glm::vec3(0);
		}
		prd.radiance = emission;
	}
	else if(!optixLaunchParams.frame.enableBSDF)	// BRDF model
	{
		float alpha = roughness * roughness;
		float alpha2 = alpha * alpha;

		prd.origin = newOri + 1e-3f * N;
		prd.lastEnergy = prd.energy;
		//prd.energy *= diffuse + F / 4.0f * PI * G * D;
		float roulette = getRandomFloat(prd.seed);
		float reflChance = glm::saturate(specular.x + specular.y + specular.z) / 3.0f + 0.0000001f;
		float diffChance = (glm::saturate(kd.x + kd.y + kd.z) / 3.0f + 0.0000001f) * (1.0f - reflChance);
		float specChance = (1.0f - diffChance) + 0.0000001f;
		if (roulette < diffChance) {	// diffuse
			glm::vec3 ddir = sampleHemiSphere(N, prd.seed);

			prd.direction = ddir;
			prd.energy *= sResult((1.0f / diffChance) * diffuse);
		}
		else if (roulette < diffChance + reflChance) {	// reflect
			float smoothness = powf(1000.0f, (1.0f - optixLaunchParams.frame.roughness) * (1.0f - optixLaunchParams.frame.roughness));
			prd.direction = glm::reflect(rayDir, N);
			float f = 1.0f / (smoothness + 1.0f) + 1.0f;
			prd.energy *= specular * (1.0f / reflChance);
		}
		else if (roulette < specChance + reflChance + diffChance)	// specular
		{
			glm::vec3 sdir = sampleHemiSphereGGX(N, alpha, alpha2, prd.seed);
			glm::vec3 L = glm::normalize(sdir);
			glm::vec3 H = glm::normalize(L + V + glm::vec3(0.001f));
			float VdotH = glm::max(glm::dot(V, H), 0.001f);
			float NdotV = glm::max(glm::dot(N, V), 0.001f);
			float NdotL = glm::max(glm::dot(N, L), 0.001f);
			float LdotV = glm::max(glm::dot(L, V), 0.001f);
			glm::vec3 F = calcF(F0, VdotH);

			float G = calcG(NdotL, NdotV, alpha2);
			//float D = calcD(glm::max(glm::dot(N, H), 0.0f), optixLaunchParams.frame.roughness);

			prd.direction = sdir;
			prd.energy *= sResult((1.0f / specChance) * F * G / NdotL / NdotV * LdotV);
		}
		else	// dropout
		{
			prd.energy = glm::vec3(0);
		}
		prd.radiance = emission;
	}
	else	// BSDF model
	{
		float alpha = roughness * roughness;
		float alpha2 = alpha * alpha;
		glm::vec3 sdir = sampleHemiSphereGGX(N, alpha, alpha2, prd.seed);
		glm::vec3 L = glm::normalize(sdir);
		glm::vec3 H = glm::normalize(L + V + glm::vec3(0.001f));
		float VdotH = glm::max(glm::dot(V, H), 0.001f);
		float NdotV = glm::max(glm::dot(N, V), 0.001f);
		float NdotL = glm::max(glm::dot(N, L), 0.001f);
		float LdotV = glm::max(glm::dot(L, V), 0.001f);
		float LdotH = glm::max(glm::dot(L, H), 0.001f);
		float NdotH = glm::max(glm::dot(N, H), 0.001f);

		// diffuse
		float Fd90 = 0.5f + 2.0f * LdotH * LdotH * roughness;
		float FL = SchlickFresnel(NdotL);
		float FV = SchlickFresnel(NdotV);
		float Fd = glm::mix(1.0f, Fd90, FL) * glm::mix(1.0f, Fd90, FV);

		// specular
		glm::vec3 Cdlin = diffuse;
		float Cdlum = Cdlin.x + Cdlin.y + Cdlin.z;
		glm::vec3 Ctint = Cdlum > 0 ? (Cdlin / Cdlum) : glm::vec3(1);
		glm::vec3 Cspec = optixLaunchParams.frame.specular * glm::mix(glm::vec3(1.0f), Ctint, optixLaunchParams.frame.specularTint);
		glm::vec3 Cspec0 = glm::mix(0.08f * Cspec, Cdlin, metallic);
		float Ds = calcD(NdotH, alpha2);
		float FH = SchlickFresnel(LdotH);
		glm::vec3 Fs = glm::mix(Cspec0, glm::vec3(1.0f), FH);
		float Gs = SmithG_GGX(NdotV, alpha) * SmithG_GGX(NdotL, alpha);

		// SSS
		float Fss90 = LdotH * LdotH * roughness;
		float Fss = glm::mix(1.0f, Fss90, FL) * glm::mix(1.0f, Fss90, FV);
		float sss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - 0.5f) + 0.5f);

		// clearcoat
		float clearcoatGloss = glm::mix(0.1f, 0.001f, optixLaunchParams.frame.clearcoatGloss);
		float Dr = calcD2(NdotH, clearcoatGloss * clearcoatGloss);
		float Fr = glm::mix(0.04f, 1.0f, FH);
		float Gr = SmithG_GGX(NdotL, 0.0625f) * SmithG_GGX(NdotV, 0.0625f);
		glm::vec3 clearcoat = glm::vec3(0.25f * Gr * Dr * Fr * optixLaunchParams.frame.clearcoat);
		glm::vec3 Csheen = glm::mix(glm::vec3(1.0f), Ctint, optixLaunchParams.frame.sheenTint);
		glm::vec3 Fsheen = FH * optixLaunchParams.frame.sheen * Csheen;

		// total: diffuse  =   PI_INV * glm::mix(Fd, sss, optixLaunchParams.subSurface) * Cdlin + Fsheen;
		//		  specular =   Gs * Fs * Ds;
		//		  BSDF     =   diffuse * (1.0f - metallic) + specular + clearcoat;

		prd.origin = newOri + 1e-5f * N;
		prd.lastEnergy = prd.energy;
		//prd.energy *= diffuse + F / 4.0f * PI * G * D;
		float roulette = getRandomFloat(prd.seed);
		float diffChance = glm::saturate(kd.x + kd.y + kd.z) / 3.0f + 0.0000001f;
		float coatChance = glm::saturate(optixLaunchParams.frame.clearcoat * optixLaunchParams.frame.clearcoat * 0.25f);
		float specChance = glm::saturate((1.0f - diffChance - coatChance) + 0.0000001f);
		if (roulette < specChance) {	// specular
			glm::vec3 F = calcF(F0, VdotH);

			float G = calcG(NdotL, NdotV, alpha2);
			//float D = calcD(glm::max(glm::dot(N, H), 0.0f), optixLaunchParams.frame.roughness);

			prd.direction = sdir;
			prd.energy *= sResult((1.0f / specChance) * Fs * Gs / NdotL / NdotV * LdotV) * PI;
			
		}
		else if (roulette < specChance + diffChance)	// diffuse
		{
			glm::vec3 ddir = sampleHemiSphere(N, prd.seed);
			prd.direction = ddir;

			prd.energy *= sResult((1.0f / diffChance) * Cdlin * glm::lerp(sss, Fd, optixLaunchParams.frame.subSurface) + Fsheen * PI);
		}
		//else if (roulette < specChance + diffChance + sssChance)	// subsurface
		//{
		//	glm::vec3 ddir = sampleHemiSphere(N, prd.seed);

		//	prd.direction = ddir;
		//	prd.energy *= sResult((1.0f / sssChance) * glm::vec3(glm::abs(Fd - sss)));
		//}
		else if (roulette < specChance + diffChance + coatChance)	// clearcoat 
		{
			prd.direction = sdir;
			prd.energy *= sResult((1.0f / coatChance) * clearcoat / Ds / NdotL / NdotV * LdotV) * PI;
		}
		else	// dropout
		{
			prd.energy = glm::vec3(0);
		}
		prd.radiance = emission;
	}
	glm::vec3 lightVisibility = glm::vec3(0);
	const glm::vec3 surfPos
		= (1.f - u - v) * sbtData.vertex[index.x]
		+ u * sbtData.vertex[index.y]
		+ v * sbtData.vertex[index.z];
	const glm::vec3 lightPos
		= glm::vec3(0, 100, 0)
		+ (getRandomFloat(prd.seed) * 2.0f - 1.0f) * glm::vec3(400, 0, 0)
		+ (getRandomFloat(prd.seed) * 2.0f - 1.0f) * glm::vec3(0, 400, 0);
	glm::vec3 lightDir = lightPos - surfPos;
	float lightDist = glm::length(lightDir);
	lightDir = normalize(lightDir);
	uint32_t u0, u1;
	packPointer(&lightVisibility, u0, u1);
	optixTrace(optixLaunchParams.traversable,
		convert_vec3(surfPos + 1e-5f * N),
		convert_vec3(lightDir),
		1e-5f,      // tmin
		lightDist * (1.f - 1e-5f),  // tmax
		0.0f,       // rayTime
		OptixVisibilityMask(255),
		// For shadow rays: skip any/closest hit shaders and terminate on first
		// intersection with anything. The miss shader is used to mark if the
		// light was visible.
		OPTIX_RAY_FLAG_DISABLE_ANYHIT
		| OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
		| OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		SHADOW_RAY_TYPE,            // SBT offset
		RAY_TYPE_COUNT,               // SBT stride
		SHADOW_RAY_TYPE,            // missSBTIndex 
		u0, u1);
	prd.energy *= glm::max(lightVisibility, 0.1f);
}
extern "C" __global__ void __closesthit__shadow() {}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __anyhit__shadow() {}
extern "C" __global__ void __miss__radiance() {
	const cudaTextureObject_t& sbtData = *(const cudaTextureObject_t*)optixGetSbtDataPointer();
	RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

	const float3 rayDir = optixGetWorldRayDirection();
	glm::vec2 uv = getSphericalCoord(rayDir);
	float4 back = tex2DLod<float4>(sbtData, uv.x, uv.y, 0.0f);
	prd.radiance = sResult(glm::vec3(powf(back.x, 2.2f), powf(back.y, 2.2f), powf(back.z, 2.2f)));
	prd.lastEnergy = prd.energy;
	prd.energy = glm::vec3(0);
}
extern "C" __global__ void __miss__shadow() {
	RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();
	prd.radiance = glm::vec3(1.0f, 1.0f, 1.0f);
}
extern "C" __global__ void __raygen__renderFrame() {
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;

	const auto& camera = optixLaunchParams.camera;

	RadiancePRD prd;
	prd.energy = glm::vec3(1);
	prd.lastEnergy = glm::vec3(1);
	prd.radiance = glm::vec3(0);

	int frame = optixLaunchParams.frame.enableDenoiser == false ? optixLaunchParams.frame.frameCount : 0;
	unsigned int seed = (ix * 1973 + iy * 9277 + frame * 26699) | 1;
	float rx = getRandomFloat(seed);
	float ry = getRandomFloat(seed);

	prd.seed = seed;

	const glm::vec2 screen(glm::vec2(ix + rx, iy + ry) / optixLaunchParams.frame.size);

	glm::vec3 rayOri = camera.position;
	glm::vec3 rayDir = glm::normalize(camera.direction + (screen.x - 0.5f) * camera.horizontal + (screen.y - 0.5f) * camera.vertical);

	prd.viewPoint = rayOri;

	glm::vec3 result = glm::vec3(0);
	for (int i = 0; i < optixLaunchParams.frame.maxTracingDepth; i++) {
		uint32_t u0, u1;
		packPointer(&prd, u0, u1);
		optixTrace(optixLaunchParams.traversable, convert_vec3(rayOri), convert_vec3(rayDir), 0.01f, 1e16f, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT, RADIANCE_RAY_TYPE, RAY_TYPE_COUNT, RADIANCE_RAY_TYPE, u0, u1);
		result += prd.radiance * prd.lastEnergy;
		if (prd.energy.x == 0 || prd.energy.y == 0 || prd.energy.z == 0)
			break;
		rayOri = prd.origin;
		rayDir = prd.direction;
	}
	result = sResult(result);
	glm::vec4 rgba = glm::vec4(result, 1.0f);

	const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;

	if (optixLaunchParams.frame.enableProgressiveRefinement) {
		rgba += float(optixLaunchParams.frame.frameCount) * convert_float4(optixLaunchParams.frame.colorBuffer[fbIndex]);
		rgba /= (float)optixLaunchParams.frame.frameCount + 1.0f;
	}

	optixLaunchParams.frame.colorBuffer[fbIndex] = convert_vec4(rgba);
}