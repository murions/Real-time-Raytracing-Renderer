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
// model
const std::string minecraft = project_dir + "model/lost-empire/lost_empire.obj";
const std::string cornellBox = project_dir + "model/CornellBox/CornellBox-Original.obj";
const std::string sponza = project_dir + "model/sponza/really_new_sponza.obj";
const std::string mitsuba = project_dir + "model/mitsuba/mitsuba.obj";
const std::string cerberus = project_dir + "model/Cerberus_LP/Cerberus_LP.obj";
const std::string helmet = project_dir + "model/helmet/helmet.obj";
const std::string dragon = project_dir + "model/dragon3.obj";
const std::string living_room = project_dir + "model/living_room/living_room.obj";
const std::string spawn = project_dir + "model/vokselia_spawn/vokselia_spawn.obj";
// texture
const std::string cape_hill = project_dir + "texture/cape_hill_4k.hdr";
const std::string golf = project_dir + "texture/limpopo_golf_course_4k.hdr";
const std::string rainforest = project_dir + "texture/rainforest_trail_4k.hdr";
const std::string garden = project_dir + "texture/studio_garden_4k.hdr";
const std::string gazebo = project_dir + "texture/whipple_creek_gazebo_4k.hdr";
const std::string night = project_dir + "texture/dikhololo_night_4k.hdr";
const std::string fireplace = project_dir + "texture/fireplace_4k.hdr";
const std::string museum = project_dir + "texture/museum_of_ethnography_4k.hdr";
#endif // TRACERCONFIG_H
