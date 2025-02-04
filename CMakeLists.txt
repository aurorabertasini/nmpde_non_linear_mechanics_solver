cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(./common/cmake-common.cmake)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# Create the library with the shared code
add_library(navier_stokes
    src/UncoupledNavierStokes.cpp
    src/MonolithicNavierStokes.cpp
    src/SteadyNavierStokes.cpp
    src/ConfigReader.cpp
)
deal_ii_setup_target(navier_stokes)

# main_monolithic
add_executable(main_monolithic src/main_monolithic.cpp)
deal_ii_setup_target(main_monolithic)
target_link_libraries(main_monolithic navier_stokes)

# main_uncoupled
add_executable(main_uncoupled src/main_uncoupled.cpp)
deal_ii_setup_target(main_uncoupled)
target_link_libraries(main_uncoupled navier_stokes)
