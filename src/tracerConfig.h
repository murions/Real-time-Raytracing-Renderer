#ifndef TRACERCONFIG_H
#define TRACERCONFIG_H


#define PRINT_RESET		"\033[0m"
#define PRINT_GREY		"\033[1;30m"
#define PRINT_RED		"\033[1;31m"
#define PRINT_GREEN		"\033[1;32m"
#define PRINT_YELLOW	"\033[1;33m"
#define PRINT_BLUE		"\033[1;34m"
#define PRINT_MAGENTA	"\033[1;35m"
#define PRINT_CYAN		"\033[1;36m"
#define PRINT_WHITE		"\033[1;37m"

#include <iostream>
#include <string>
#ifndef PROJECT_DIR
    const std::string project_dir = "../";
#else
    const std::string project_dir = PROJECT_DIR"/";
#endif
#ifndef RESOURCE_DIR
    const std::string resource_dir = "./";
#else
    const std::string resource_dir = RESOURCE_DIR"/";
#endif
// model
const std::string minecraft = resource_dir + "model/lost-empire/lost_empire.obj";
const std::string cornellBox = resource_dir + "model/CornellBox/CornellBox-Original.obj";
const std::string sponza = resource_dir + "model/sponza/really_new_sponza.obj";
const std::string mitsuba = resource_dir + "model/mitsuba/mitsuba.obj";
const std::string cerberus = resource_dir + "model/Cerberus_LP/Cerberus_LP.obj";
const std::string helmet = resource_dir + "model/helmet/helmet.obj";
const std::string dragon = resource_dir + "model/dragon3.obj";
const std::string living_room = resource_dir + "model/living_room/living_room.obj";
const std::string spawn = resource_dir + "model/vokselia_spawn/vokselia_spawn.obj";
// texture
const std::string cape_hill = resource_dir + "texture/cape_hill_4k.hdr";
const std::string golf = resource_dir + "texture/limpopo_golf_course_4k.hdr";
const std::string rainforest = resource_dir + "texture/rainforest_trail_4k.hdr";
const std::string garden = resource_dir + "texture/studio_garden_4k.hdr";
const std::string gazebo = resource_dir + "texture/whipple_creek_gazebo_4k.hdr";
const std::string night = resource_dir + "texture/dikhololo_night_4k.hdr";
const std::string fireplace = resource_dir + "texture/fireplace_4k.hdr";
const std::string museum = resource_dir + "texture/museum_of_ethnography_4k.hdr";
#endif // TRACERCONFIG_H
