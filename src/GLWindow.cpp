#include "GLWindow.h"
#include <cassert>
#include <stdlib.h>

static void error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}
static void resize_callback(GLFWwindow* window, int width, int height) {
	GLWindow* glw = static_cast<GLWindow*>(glfwGetWindowUserPointer(window));
	assert(glw);
	glw->resize(glm::ivec2(width, height));
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	GLWindow* glw = static_cast<GLWindow*>(glfwGetWindowUserPointer(window));
	assert(glw);
	if(action == GLFW_PRESS)
		glw->key(key, mods);
}
static void mouseMotion_callback(GLFWwindow* window, double x, double y) {
	GLWindow* glw = static_cast<GLWindow*>(glfwGetWindowUserPointer(window));
	assert(glw);
	glw->mouseMotion((int)x, (int)y);
}
static void mouseButton_callback(GLFWwindow* window, int button, int action, int mods) {
	GLWindow* glw = static_cast<GLWindow*>(glfwGetWindowUserPointer(window));
	assert(glw);
	glw->mouseButton(button, action, mods);
}

GLWindow::~GLWindow() {
	glfwDestroyWindow(handle);
	glfwTerminate();
}
GLWindow::GLWindow(const std::string& title) {
	glfwSetErrorCallback(error_callback);

	init(title, 1280, 720);
}
GLWindow::GLWindow(const std::string& title, int width, int height) {
	glfwSetErrorCallback(error_callback);

	init(title, width, height);
}
void GLWindow::init(const std::string& title, int width, int height) {
	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	const char* glsl_version = "#version 130";

	handle = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

	lastTime = glfwGetTime();
	lastCountTime = lastTime;

	if (!handle) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetWindowUserPointer(handle, this);
	glfwMakeContextCurrent(handle);
	glfwSwapInterval(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	
	ImGui::StyleColorsLight();
	ImGui::GetStyle().WindowBorderSize = 0.0f;
	ImGui_ImplGlfw_InitForOpenGL(handle, true);
	ImGui_ImplOpenGL3_Init(glsl_version);
}
void GLWindow::run() {
	int width, height;
	glfwGetFramebufferSize(handle, &width, &height);
	resize(glm::ivec2(width, height));

	glfwSetFramebufferSizeCallback(handle, resize_callback);
	glfwSetMouseButtonCallback(handle, ImGui_ImplGlfw_MouseButtonCallback);
	glfwSetKeyCallback(handle, key_callback);
	glfwSetCursorPosCallback(handle, ImGui_ImplGlfw_CursorPosCallback);

	while (!glfwWindowShouldClose(handle))
	{
		float currentTime = glfwGetTime();
		float deltaTime = currentTime - lastTime;
		float fps = 1.0f / deltaTime;
		lastTime = currentTime;
		sprintf_s(strFrameRate, "%.2f fps        %.2f ms", fps, deltaTime * 1000.0f);
		if (currentTime - lastCountTime > 1.0f)
			lastCountTime = currentTime;
		if (lastCountTime >= currentTime - FLT_EPSILON)
			glfwSetWindowTitle(handle, strFrameRate);
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		render();
		draw();
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(handle);
		glfwPollEvents();
	}
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}