#include "Renderer.h"
#include "RendererWindow.h"

int main() {
	try {
		Model* model = Model::loadOBJ(helmet);
		Texture* background = Model::loadBackground("E:/studv/Renderer/PathTracer/texture/sunflowers_puresky_4k.hdr");
		
		Camera camera = { glm::vec3(0, 2, 11), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0) };
		RendererWindow* rendererWindow = new RendererWindow("Path Tracer", model, background, camera, 0.01f, 0.01f, 0.2f);
		rendererWindow->run();
	
		rendererWindow->cleanup();
	}
	catch (std::runtime_error& e) {
		std::cout << PRINT_RED << "FETAL ERROR: " << e.what() << PRINT_RESET << std::endl;
		exit(1);
	}
	return 0;
}
