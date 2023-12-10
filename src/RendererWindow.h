#ifndef RENDERERWINDOW_H
#define RENDERERWINDOW_H

#include <glad/glad.h>
#include "GLWindow.h"
#include "Shader.h"
#include "Renderer.h"
#include <vector>

class RendererWindow : public GLCameraWindow {
public:
	GLuint fbTexture{ 0 };
	Renderer renderer;
	std::vector<float4> pixels;
	glm::ivec2 fbsize;
	float moveSpeed = 0.01f, rotationSpeed = 0.01f, scaleSpeed = 0.2f;

	bool enablePR = true;
	bool enableDenoiser = false;
	int modelVertices = 0;

	bool enableBF = true;
	float x_radius = 5.0f;
	float y_radius = 5.0f;
	float x_factor = 0.81f;
	float y_factor = 0.81f;

	unsigned int VAO;
	unsigned int VBO;
	Shader shader;

	float backgroundCol[4]{ 0.5f, 0.5f, 0.5f, 0.15f };
	float textCol[4]{ 0, 0, 0, 1 };

public:
	RendererWindow(const std::string& title, Model *model, Texture *background, const Camera &camera, const float moveSpeed, const float rotateSpeed, const float scaleSpeed) : GLCameraWindow(title, camera.pos, camera.at, camera.up), renderer(model, background){
		renderer.setCamera(camera);
		this->moveSpeed = moveSpeed;
		this->rotationSpeed = rotateSpeed;
		this->scaleSpeed = scaleSpeed;
		for (auto mesh : model->meshes) {
			modelVertices += mesh->vertex.size();
		}
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << PRINT_RED << "Failed to initialize GLAD!" << PRINT_RESET << std::endl;
		}
		//input data
		std::cout << PRINT_GREEN << "Successfully init GLAD." << PRINT_RESET << std::endl;
		float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
	   // positions   // texCoords
	   -1.0f,  1.0f,  0.0f, 1.0f,
	   -1.0f, -1.0f,  0.0f, 0.0f,
		1.0f, -1.0f,  1.0f, 0.0f,

	   -1.0f,  1.0f,  0.0f, 1.0f,
		1.0f, -1.0f,  1.0f, 0.0f,
		1.0f,  1.0f,  1.0f, 1.0f
		};
		//opengl init
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glBindVertexArray(VAO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

		shader = Shader("../shader/vertex.glsl", "../shader/fragment.glsl");
	}

	virtual void render() override {
		ImGuiIO& io = ImGui::GetIO();
		static float phi = 0, theta = 0.0f, fov = acos(0.66f);
		float dx = 0, dy = 0, dz = 0;

		static int model_idx = 0, bg_idx = 0;
		if (ImGui::Begin("Scene Control", (bool*)0, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoResize)) 
		{
			if (ImGui::Combo("select model", &model_idx, "Helmet\0Revolver\0Lost Empire\0Spawn\0Sponza\0Dragon\0Living room\0Mitsuba\0"))
			{
				switch (model_idx)
				{
					Model* models;
				case 0:
					models = Model::loadOBJ(helmet);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 1:
					models = Model::loadOBJ(cerberus);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 2:
					models = Model::loadOBJ(minecraft);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 3:
					models = Model::loadOBJ(spawn);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 4:
					models = Model::loadOBJ(sponza);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 5:
					models = Model::loadOBJ(dragon);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 6:
					models = Model::loadOBJ(living_room);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				case 7:
					models = Model::loadOBJ(mitsuba);
					rebuild(models);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					modelVertices = 0;
					for (auto mesh : models->meshes) {
						modelVertices += mesh->vertex.size();
					}
					break;
				}
			}
			if (ImGui::Combo("select background", &bg_idx, "Golf\0Rain forest\0Garden\0Gazebo\0Night\0Fireplace\0Museum\0Cape_hill\0"))
			{
				switch (bg_idx)
				{
					Texture* backgrounds;
				case 0:
					backgrounds = Model::loadBackground(golf);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 1:
					backgrounds = Model::loadBackground(rainforest);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 2:
					backgrounds = Model::loadBackground(garden);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 3:
					backgrounds = Model::loadBackground(gazebo);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 4:
					backgrounds = Model::loadBackground(night);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 5:
					backgrounds = Model::loadBackground(fireplace);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 6:
					backgrounds = Model::loadBackground(museum);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				case 7:
					backgrounds = Model::loadBackground(cape_hill);
					rebuild(backgrounds);
					renderer.launchParams.frame.frameCount = 0;
					cameraFrame.modified = true;
					break;
				}
			}
		}
		ImGui::End();
		if (!ImGui::IsAnyItemActive())
		{
			// 鼠标中键拖动平移
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
			{
				dx -= io.MouseDelta.x * moveSpeed;
				dy += io.MouseDelta.y * moveSpeed;
				renderer.launchParams.frame.frameCount = 0;
				cameraFrame.modified = true;
			}
			// 鼠标右键拖动旋转
			else if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				phi += io.MouseDelta.y * rotationSpeed;
				theta += io.MouseDelta.x * rotationSpeed;
				phi = cameraFrame.scalarModAngle(phi);
				theta = cameraFrame.scalarModAngle(theta);
				renderer.launchParams.frame.frameCount = 0;
				cameraFrame.modified = true;
			}
			// 鼠标滚轮缩放
			else if (io.MouseWheel != 0.0f)
			{
				dz += scaleSpeed * io.MouseWheel;
				renderer.launchParams.frame.frameCount = 0;
				cameraFrame.modified = true;
			}
		}
		cameraFrame.camPitch = phi + PI;
		cameraFrame.camYaw = theta;
		cameraFrame.moveLeftRight = dx;
		cameraFrame.moveBackForward = dz;
		cameraFrame.moveUp = dy;
		cameraFrame.camRotationMatrix = glm::eulerAngleXYZ(cameraFrame.camPitch, cameraFrame.camYaw, cameraFrame.camRoll);
		cameraFrame.camTarget = cameraFrame.DefaultForward * cameraFrame.camRotationMatrix;
		cameraFrame.camTarget = normalize(cameraFrame.camTarget);
		cameraFrame.camRight =  cameraFrame.DefaultRight * cameraFrame.camRotationMatrix;
		cameraFrame.camForward = cameraFrame.DefaultForward * cameraFrame.camRotationMatrix;
		glm::vec3 cam_Up = -glm::cross(glm::vec3(cameraFrame.camForward.x, cameraFrame.camForward.y, cameraFrame.camForward.z),
			glm::vec3(cameraFrame.camRight.x, cameraFrame.camRight.y, cameraFrame.camRight.z));
		cameraFrame.camUp = glm::vec4(cam_Up.x, cam_Up.y, cam_Up.z, 0);
		cameraFrame.camPosition += cameraFrame.moveLeftRight * cameraFrame.camRight;
		cameraFrame.camPosition += cameraFrame.moveBackForward * cameraFrame.camForward;
		cameraFrame.camPosition += cameraFrame.moveUp * cameraFrame.camUp;
		cameraFrame.moveLeftRight = 0.0f;
		cameraFrame.moveBackForward = 0.0f;
		cameraFrame.moveUp = 0.0f;
		cameraFrame.camTarget = cameraFrame.camPosition + cameraFrame.camTarget;

		cameraFrame.setOrientation(cameraFrame.camPosition, cameraFrame.camTarget, cameraFrame.camUp);

		ImGuiWindowFlags wf = 0;
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(backgroundCol[0], backgroundCol[1], backgroundCol[2], backgroundCol[3]));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(textCol[0], textCol[1], textCol[2], textCol[3]));
		wf |= ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar;
		if (ImGui::Begin("Properties", nullptr, wf)) {
			//if (ImGui::BeginMenuBar()) {
			//	ImGui::Text("Properties");
			//}
			//ImGui::EndMenuBar();
			ImGui::Text("Background:");
			ImGui::SameLine();
			ImGui::ColorEdit4("##9", backgroundCol, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoBorder | ImGuiColorEditFlags_AlphaBar);
			ImGui::SameLine(180, 0);
			ImGui::Text("Text:");
			ImGui::SameLine();
			ImGui::ColorEdit4("##10", textCol, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoBorder | ImGuiColorEditFlags_AlphaBar);
			if (ImGui::BeginChild("Model", ImVec2(240, 50), true, wf)) {
				ImGui::Text("Model:");
				ImGui::Text("Vertices: %d", modelVertices);
			}
			ImGui::EndChild();
			if (ImGui::BeginChild("Camera", ImVec2(240, 80), true, wf)) {
				ImGui::Text("Camera:");
				ImGui::Text("Position: (%.2f, %.2f, %.2f)", cameraFrame.get_pos().x, cameraFrame.get_pos().y, cameraFrame.get_pos().z);
				ImGui::Text("Rotation: (%.2f, %.2f, 0.00)", cameraFrame.convertToDegrees(phi), cameraFrame.convertToDegrees(theta));
				//char deg[20];
				//sprintf(deg, "%.2f", cameraFrame.convertToDegrees(fov));
				//ImGui::Text("FOV:");
				//ImGui::SameLine();
				//ImGui::Text(deg);
				glm::ivec2 size = renderer.getWindowSize();
				ImGui::Text("Resolution: %d x %d", size.x, size.y);
			}
			ImGui::EndChild();
			ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
			ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
			ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0, 0, 0, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0, 0, 0, 1.0f));
			ImGui::PushStyleVar(ImGuiStyleVar_GrabMinSize, 1.0f);
			if (ImGui::BeginChild("Camera settings", ImVec2(240, 100), true, wf)) {
				ImGui::Text("Camera settings:");
				ImGui::Text("moveSpeed: ");
				ImGui::SameLine();
				ImGui::SliderFloat("##1", &moveSpeed, 0, 5, "");
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", moveSpeed);
				ImGui::Text("rotateSpeed: ");
				ImGui::SameLine();
				ImGui::SliderFloat("##2", &rotationSpeed, 0, 0.1f, "");
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", rotationSpeed);
				ImGui::Text("scaleSpeed: ");
				ImGui::SameLine();
				ImGui::SliderFloat("##3", &scaleSpeed, 0, 15, "");
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", scaleSpeed);
			}
			ImGui::EndChild();
			if (ImGui::BeginChild("Basic settings", ImVec2(240, 300), true, wf)) {
				ImGui::Text("Basic settings:");
				ImGui::Text("maxTracingDepth:");
				ImGui::SameLine(0, 0);
				int depth = renderer.getMaxDepth();
				if (ImGui::DragInt("##4", &depth, 0.25f, 0, 100, ""))
					renderer.setMaxDepth(depth);
				ImGui::SameLine(200, 0);
				ImGui::Text("%d", depth);

				ImGui::Text("FrameSamples: %d", renderer.getFrameCount());
				if (ImGui::Checkbox("Enable Progressive Sampling", &enablePR)) {
					enablePR == true ? renderer.enablePR() : renderer.disablePR();
					renderer.launchParams.frame.frameCount = 0;
				}
				if (ImGui::Checkbox("Enable Denoiser", &enableDenoiser)) {
					enableDenoiser == true ? renderer.enableDenoiser() : renderer.disableDenoiser();
					//renderer.launchParams.frame.frameCount = 0;
				}
				if (ImGui::Checkbox("Enable Bilateral-Filtering", &enableBF)) {
					renderer.launchParams.frame.frameCount = 0;
				}
				ImGui::Text("Radius_x:");
				if (ImGui::SliderFloat("##5", &x_factor, 0, 1, ""))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", x_factor);
				ImGui::Text("Factor_x:");
				if (ImGui::SliderFloat("##6", &x_radius, 0, 10, ""))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", x_radius);
				ImGui::Text("Radius_y:");
				if (ImGui::SliderFloat("##7", &y_factor, 0, 1, ""))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", y_factor);
				ImGui::Text("Factor_y:");
				if (ImGui::SliderFloat("##8", &y_radius, 0, 10, ""))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", y_radius);
			}
			ImGui::EndChild();
			if (ImGui::BeginChild("Material settings", ImVec2(240, 120), true, wf)) {
				ImGui::Text("Material settings:");
				if (ImGui::Checkbox("Enable PBR", &renderer.launchParams.frame.enablePBR)) 
					renderer.launchParams.frame.frameCount = 0;
				if (ImGui::Checkbox("Use Roughness Texture", &renderer.launchParams.frame.useRoughnessTexture))
					renderer.launchParams.frame.frameCount = 0;
				if (ImGui::Checkbox("Use Metallic Texture", &renderer.launchParams.frame.useMetallicTexture))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::Text("roughness:");
				ImGui::SameLine(0, 0);
				if (ImGui::SliderFloat("##5", &renderer.launchParams.frame.roughness, 0, 1, ""))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", renderer.launchParams.frame.roughness);
				ImGui::Text("metellic:");
				ImGui::SameLine(0, 0);
				if (ImGui::SliderFloat("##6", &renderer.launchParams.frame.metallic, 0, 1, ""))
					renderer.launchParams.frame.frameCount = 0;
				ImGui::SameLine(200, 0);
				ImGui::Text("%.2f", renderer.launchParams.frame.metallic);
			}
			ImGui::EndChild();
			ImGui::PopStyleVar();
			ImGui::PopStyleColor(4);
		}
		ImGui::End();
		ImGui::PopStyleColor(2);
		wf = ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoInputs;
		if (ImGui::Begin("Tips", nullptr, wf)) {
			ImGui::Text("Drag mouse left or middle to move.");
			ImGui::Text("Drag mouse right to rotate.");
			ImGui::Text("Slider mouse wheel to scale.");
		}
		ImGui::End();
		if (ImGui::Begin("Mat")) {
			if (ImGui::Checkbox("Enable BSDF", &renderer.launchParams.frame.enableBSDF))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::Text("roughness:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##5", &renderer.launchParams.frame.roughness, 0, 1, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.roughness);
			ImGui::Text("metellic:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##6", &renderer.launchParams.frame.metallic, 0, 1, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.metallic);
			ImGui::Text("subSurface:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##55", &renderer.launchParams.frame.subSurface, 0, 1, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.subSurface);
			ImGui::Text("specular:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##15", &renderer.launchParams.frame.specular, 0, 1, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.specular);
			ImGui::Text("specularTint:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##16", &renderer.launchParams.frame.specularTint, 0, 100, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.specularTint);
			ImGui::Text("clearcoat:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##25", &renderer.launchParams.frame.clearcoat, 0, 3, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.clearcoat);
			ImGui::Text("clearcoatGloss:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##26", &renderer.launchParams.frame.clearcoatGloss, 0, 100, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.clearcoatGloss);
			ImGui::Text("sheen:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##35", &renderer.launchParams.frame.sheen, 0, 100, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.sheen);
			ImGui::Text("sheenTint:");
			ImGui::SameLine(0, 0);
			if (ImGui::SliderFloat("##36", &renderer.launchParams.frame.sheenTint, 0, 100, ""))
				renderer.launchParams.frame.frameCount = 0;
			ImGui::SameLine(200, 0);
			ImGui::Text("%.2f", renderer.launchParams.frame.sheenTint);
		}
		ImGui::End();
		if (cameraFrame.modified) {
			renderer.setCamera(Camera{ cameraFrame.get_pos(), cameraFrame.get_at(), cameraFrame.get_up() });
			cameraFrame.modified = false;
		}
		renderer.render();
	}
	virtual void draw() override {
		renderer.downloadPixels(pixels.data());
		if (fbTexture == 0)
			glGenTextures(1, &fbTexture);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		GLenum texFormat = GL_RGBA;
		GLenum texelType = GL_FLOAT;
		glTexImage2D(GL_TEXTURE_2D, 0, texFormat, fbsize.x, fbsize.y, 0, GL_RGBA, texelType, pixels.data());

		//glDisable(GL_LIGHTING);
		//glColor3f(1, 1, 1);

		//glMatrixMode(GL_MODELVIEW);
		//glLoadIdentity();

		glEnable(GL_TEXTURE_2D);
		//glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glDisable(GL_DEPTH_TEST);

		glViewport(0, 0, fbsize.x, fbsize.y);

		//glMatrixMode(GL_PROJECTION);
		//glLoadIdentity();
		//glOrtho(0, (float)fbsize.x, 0, (float)fbsize.y, -1, 1);

		//glBegin(GL_QUADS);
		//{
		//	glTexCoord2f(0, 0);
		//	glVertex3f(0, 0, 0);

		//	glTexCoord2f(0, 1);
		//	glVertex3f(0, (float)fbsize.y, 0);

		//	glTexCoord2f(1, 1);
		//	glVertex3f((float)fbsize.x, (float)fbsize.y, 0);

		//	glTexCoord2f(1, 0);
		//	glVertex3f((float)fbsize.x, 0, 0);
		//}
		//glEnd();
		shader.use(); 
		shader.setInt("screenTexture", 0);
		shader.setBool("enableBF", enableBF);
		shader.setFloat("x_radius", x_radius);
		shader.setFloat("x_factor", x_factor);
		shader.setFloat("y_radius", y_radius);
		shader.setFloat("y_factor", y_factor);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessary actually, since we won't be able to see behind the quad anyways)
		glClear(GL_COLOR_BUFFER_BIT);

		glBindVertexArray(VAO);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glDrawArrays(GL_TRIANGLES, 0, 6);
	}

	virtual void resize(const glm::ivec2& newSize) override {
		fbsize = newSize;
		renderer.resize(newSize);
		pixels.resize(newSize.x * newSize.y);
	}

	void cleanup() {
		renderer.cleanupState();
	}
	void rebuild(Model* model) {
		renderer.rebuildOptix(model);
	}
	void rebuild(Texture* background) {
		renderer.rebuildOptix(background);
	}
};
#endif // RENDERERWINDOW_H