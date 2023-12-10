#include "Model.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
#include <set>

namespace std {
	inline bool operator<(const tinyobj::index_t& a,
		const tinyobj::index_t& b)
	{
		if (a.vertex_index < b.vertex_index) return true;
		if (a.vertex_index > b.vertex_index) return false;

		if (a.normal_index < b.normal_index) return true;
		if (a.normal_index > b.normal_index) return false;

		if (a.texcoord_index < b.texcoord_index) return true;
		if (a.texcoord_index > b.texcoord_index) return false;

		return false;
	}
}

int addVertex(TriangleMesh* mesh, tinyobj::attrib_t& attributes, const tinyobj::index_t& index, std::map<tinyobj::index_t, int>& vertices) {
	if (vertices.find(index) != vertices.end())
		return vertices[index];

	const glm::vec3* vertex_array = (const glm::vec3*)attributes.vertices.data();
	const glm::vec3* normal_array = (const glm::vec3*)attributes.normals.data();
	const glm::vec2* texcoord_array = (const glm::vec2*)attributes.texcoords.data();

	int newID = (int)mesh->vertex.size();
	vertices[index] = newID;

	mesh->vertex.push_back(vertex_array[index.vertex_index]);
	if (index.normal_index >= 0) {
		while (mesh->normal.size() < mesh->vertex.size())
		{
			mesh->normal.push_back(normal_array[index.normal_index]);
		}
	}
	if (index.texcoord_index >= 0) {
		while (mesh->texcoord.size() < mesh->vertex.size())
		{
			mesh->texcoord.push_back(texcoord_array[index.texcoord_index]);
		}
	}
	if (mesh->normal.size() > 0)
		mesh->normal.resize(mesh->vertex.size());
	if (mesh->texcoord.size() > 0)
		mesh->texcoord.resize(mesh->vertex.size());

	return newID;
}

Texture* Model::loadBackground(const std::string& path) {
	Texture* background = nullptr;

	if (path == "")
		exit(2);

	std::string file = path;
	for (auto& c : file)
		if (c == '\\') c = '/';

	glm::ivec2 res;
	int comp;
	unsigned char* image = stbi_load(file.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha);

	if (image) {
		Texture* texture = new Texture;
		texture->resolution = res;
		texture->pixel = (uint32_t*)image;

		for (int y = 0; y < res.y / 2; y++) {
			uint32_t* line_y = texture->pixel + y * res.x;
			uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
			int mirror_y = res.y - 1 - y;
			for (int x = 0; x < res.x; x++)
				std::swap(line_y[x], mirrored_y[x]);
		}

		background = texture;
	}
	else
	{
		std::cout << PRINT_RED << "Failed to load texture!" << PRINT_RESET << std::endl;
	}
	return background;
}
HDRTexture* Model::loadHDRTexture(const std::string& path) {
	HDRTexture* background = nullptr;

	if (path == "")
		exit(2);

	std::string file = path;
	for (auto& c : file)
		if (c == '\\') c = '/';

	glm::ivec2 res;
	int comp;
	float* image = stbi_loadf(file.c_str(), &res.x, &res.y, &comp, 0);

	if (image) {
		HDRTexture* texture = new HDRTexture;
		texture->resolution = res;
		texture->pixel = (float4*)image;
		background = texture;
	}
	else
	{
		std::cout << PRINT_RED << "Failed to load texture!" << PRINT_RESET << std::endl;
	}
	return background;
}

int loadTexture(Model* model, std::map<std::string, int>& textures, const std::string& fileName, const std::string& path) {
	if (fileName == "")
		return -1;

	if (textures.find(fileName) != textures.end())
		return textures[fileName];

	std::string file = fileName;
	for (auto& c : file)
		if (c == '\\') c = '/';
	file = path + "/" + file;

	glm::ivec2 res;
	int comp;
	unsigned char* image = stbi_load(file.c_str(), &res.x, &res.y, &comp, STBI_rgb_alpha);

	int textureID = -1;
	if (image) {
		textureID = (int)model->textures.size();
		Texture* texture = new Texture;
		texture->resolution = res;
		texture->pixel = (uint32_t*)image;
		// 直接导入会使图像翻转
		for (int y = 0; y < res.y / 2; y++) {
			uint32_t* line_y = texture->pixel + y * res.x;
			uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
			int mirror_y = res.y - 1 - y;
			for (int x = 0; x < res.x; x++)
				std::swap(line_y[x], mirrored_y[x]);
		}

		model->textures.push_back(texture);
	}
	else
	{
		std::cout << PRINT_RED << "Failed to load texture!" <<PRINT_RESET << std::endl;
	}

	textures[fileName] = textureID;
	return textureID;
}

Model* Model::loadOBJ(const std::string& path) {
	Model* model = new Model;

	const std::string mtlPath = path.substr(0, path.rfind('/') + 1);
	std::cout << "From " << mtlPath << " loading models..." << std::endl;

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	if (!tinyobj::LoadObj(&attributes, &shapes, &materials, &err, &err, path.c_str(), mtlPath.c_str(), true))
		throw std::runtime_error("Failed to load OBJ files.");
	
	if (materials.empty())
		throw std::runtime_error("Failed to parse materials.");

	std::cout << PRINT_BLUE << "Finished to load OBJ files." << PRINT_RESET << std::endl;
	std::cout << PRINT_CYAN << "Done loading obj file - found " << PRINT_YELLOW << shapes.size() << PRINT_CYAN << " shapes with " << PRINT_YELLOW << materials.size() << PRINT_CYAN << " materials." << PRINT_RESET << std::endl;
	std::map<std::string, int> textures;

	for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
		tinyobj::shape_t& shape = shapes[shapeID];

		std::set<int> materialIDs;
		for (auto faceMatID : shape.mesh.material_ids)
			materialIDs.insert(faceMatID);
		for (auto materialID : materialIDs) {
			std::map<tinyobj::index_t, int> vertices;
			TriangleMesh* mesh = new TriangleMesh;

			for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
				if (shape.mesh.material_ids[faceID] != materialID) continue;
				tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
				tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
				tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

				glm::ivec3 index(addVertex(mesh, attributes, idx0, vertices), addVertex(mesh, attributes, idx1, vertices), addVertex(mesh, attributes, idx2, vertices));
				mesh->index.push_back(index);
				mesh->diffuse = (const glm::vec3&)materials[materialID].diffuse;
				mesh->specular = (const glm::vec3&)materials[materialID].specular;
				mesh->emission = (const glm::vec3&)materials[materialID].emission;
				mesh->diffuseTexID = loadTexture(model, textures, materials[materialID].diffuse_texname, mtlPath);
				mesh->metellicTexID = loadTexture(model, textures, materials[materialID].metallic_texname, mtlPath);
				mesh->roughnessTexID = loadTexture(model, textures, materials[materialID].roughness_texname, mtlPath);
				mesh->emissionTexID = loadTexture(model, textures, materials[materialID].emissive_texname, mtlPath);
			}

			if (mesh->vertex.empty())
				delete mesh;
			else
				model->meshes.push_back(mesh);
		}
	}

	for (auto mesh : model->meshes)
		for (auto vertex : mesh->vertex)
			model->box.extend(vertex);
	std::cout << PRINT_CYAN << "created a total of " << PRINT_YELLOW << model->meshes.size() << PRINT_CYAN << " meshes." << PRINT_RESET << std::endl;

	int numA = 0, numM = 0, numR = 0, numE = 0;
	for (auto mesh : model->meshes) {
		if (mesh->diffuseTexID >= 0)
			numA++;
		if (mesh->metellicTexID >= 0)
			numM++;
		if (mesh->roughnessTexID >= 0)
			numR++;
		if (mesh->emissionTexID >= 0)
			numE++;
	}
	std::cout << PRINT_CYAN << "loaded " << PRINT_YELLOW << numA << PRINT_CYAN << " albedo texture(s)." << std::endl;
	std::cout << "loaded " << PRINT_YELLOW << numM << PRINT_CYAN << " metellic texture(s)." << std::endl;
	std::cout << "loaded " << PRINT_YELLOW << numR << PRINT_CYAN << " roughness texture(s)." << std::endl;
	std::cout << "loaded " << PRINT_YELLOW << numE << PRINT_CYAN << " emission texture(s)." << PRINT_RESET << std::endl;

	return model;
}