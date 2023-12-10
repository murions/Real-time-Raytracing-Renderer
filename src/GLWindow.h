#ifndef GLWINDOW_H
#define GLWINDOW_H

//#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <ImGui/imgui.h>
#include <ImGui/imgui_impl_glfw.h>
#include <ImGui/imgui_impl_opengl3.h>
#include <iostream>
#include "math_calc.h"

// TODO: override the render, draw, resize, key, mouseMotion, mouseButton, etc. functions.
class GLWindow {
public:
	GLFWwindow* handle{ nullptr };
	float currentTime = 0;
	float lastTime = 0;
	float lastCountTime = 0;
	char strFrameRate[50];		// fps count
public:
	GLWindow(const std::string& title);
	GLWindow(const std::string& title, int width, int height);
	~GLWindow();

	virtual void init(const std::string& title, int width, int height);
	virtual void draw() {}
	virtual void resize(const glm::ivec2& newSize) {}
	virtual void key(int key, int mods){}
	virtual void mouseMotion(const int32_t& size_x, const int32_t& size_y) {}
	virtual void mouseButton(int button, int action, int mods){}
	inline glm::ivec2 getMousePos() const {
		double x, y;
		glfwGetCursorPos(handle, &x, &y);
		return glm::ivec2((int)x, (int)y);
	}
	virtual void render(){}
	virtual void run();
};
struct CameraFrame {
	CameraFrame(){}
	glm::vec3 getPOI() const {
		return position - poiDistance * frame[2];
	}
	void setOrientation(const glm::vec3& origin, const glm::vec3& interest, const glm::vec3& up) {
		position = origin;
		upVector = up;
		frame[2] = (interest == origin) ? glm::vec3(0, 0, 1) : glm::normalize(origin - interest);
		frame[0] = glm::cross(up, frame[2]);
		if (dot(frame[0], frame[0]) < 1e-8f) {
			frame[0] = glm::vec3(0, 1, 0);
		}
		else {
			frame[0] = glm::normalize(frame[0]);
		}
		frame[1] = normalize(cross(frame[2], frame[0]));
		poiDistance = glm::length(interest - origin);
		
		if (fabsf(dot(frame[2], upVector)) < 1e-6f)
			return;
		frame[0] = normalize(cross(upVector, frame[2]));
		frame[1] = normalize(cross(frame[2], frame[0]));
		modified = true;
	}
	inline glm::vec3 get_pos() const { return position; }
	inline glm::vec3 get_at() const { return getPOI(); }
	inline glm::vec3 get_up() const { return upVector; }

	inline float scalarModAngle(float Angle){
		Angle = Angle + PI;

		float fTemp = fabsf(Angle);
		fTemp = fTemp - (PI_MUL2 * static_cast<float>(static_cast<int32_t>(fTemp / PI_MUL2)));

		fTemp = fTemp - PI;

		if (Angle < 0.0f) {
			fTemp = -fTemp;
		}
		return fTemp;
	}
	inline float convertToDegrees(float fRadians) { return fRadians * (180.0f / PI); }

	glm::vec4 camPosition = glm::vec4(0, 2, 11, 1);
	glm::vec4 camTarget = glm::vec4(0, 0, 1, 1);
	glm::vec4 camUp = glm::vec4(0, 1, 0, 0);
	glm::vec4 DefaultForward = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
	glm::vec4 DefaultRight = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
	glm::vec4 camForward = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
	glm::vec4 camRight = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
	glm::mat4 camRotationMatrix;
	float moveLeftRight = 0.0f;
	float moveBackForward = 0.0f;
	float moveUp = 0.0f;
	float camYaw = 0.0f;
	float camPitch = 0.0f;
	float camRoll = 0.0f;

	glm::mat3 frame{ glm::vec3(1,0,0),glm::vec3(0,1,0),glm::vec3(0,0,1) };
	glm::vec3 position{ 0, -1, 0 };
	float poiDistance{ 1.0f };
	glm::vec3 upVector{ 0, 1, 0 };
	bool forceUp{ true };
	bool modified{ true };
};

struct GLCameraWindow : public GLWindow
{
	GLCameraWindow(const std::string& title,
		const glm::vec3& camera_from, const glm::vec3& camera_at, const glm::vec3& camera_up) :
		GLWindow(title){
		cameraFrame.setOrientation(camera_from, camera_at, camera_up);
	}
	
	CameraFrame cameraFrame;

};


#endif // GLWINDOW_H