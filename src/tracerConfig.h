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

#include <string>
// model
const std::string minecraft = "../model/lost-empire/lost_empire.obj";
const std::string cornellBox = "../model/CornellBox/CornellBox-Original.obj";
const std::string sponza = "../model/sponza/really_new_sponza.obj";
const std::string mitsuba = "../model/mitsuba/mitsuba.obj";
const std::string cerberus = "../model/Cerberus_LP/Cerberus_LP.obj";
const std::string helmet = "../model/helmet/helmet.obj";
const std::string dragon = "../model/dragon3.obj";
const std::string living_room = "../model/living_room/living_room.obj";
const std::string spawn = "../model/vokselia_spawn/vokselia_spawn.obj";
// texture
const std::string cape_hill = "../texture/cape_hill_4k.hdr";
const std::string golf = "../texture/limpopo_golf_course_4k.hdr";
const std::string rainforest = "../texture/rainforest_trail_4k.hdr";
const std::string garden = "../texture/studio_garden_4k.hdr";
const std::string gazebo = "../texture/whipple_creek_gazebo_4k.hdr";
const std::string night = "../texture/dikhololo_night_4k.hdr";
const std::string fireplace = "../texture/fireplace_4k.hdr";
const std::string museum = "../texture/museum_of_ethnography_4k.hdr";
#endif // TRACERCONFIG_H
