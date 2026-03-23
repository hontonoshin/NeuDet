#============================================================
# CMakeLists.txt  –  Li6ZnS:Ag neutron detector (Geant4 11.x)
#
#============================================================
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(NeutronDetector VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#------------------------------------------------------------
# Find Geant4 (requires data libraries: G4NDL, G4EMLOW, etc.)
#------------------------------------------------------------
find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

#------------------------------------------------------------
# Sources
#------------------------------------------------------------
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/include/*.hh)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(neutron_sim ${SOURCES} ${HEADERS})
target_link_libraries(neutron_sim ${Geant4_LIBRARIES})

#------------------------------------------------------------
# Copy macro files to build directory
#------------------------------------------------------------
configure_file(${PROJECT_SOURCE_DIR}/run.mac
               ${PROJECT_BINARY_DIR}/run.mac COPYONLY)

#------------------------------------------------------------
# Install
#------------------------------------------------------------
install(TARGETS neutron_sim DESTINATION bin)
