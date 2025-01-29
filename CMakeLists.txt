cmake_minimum_required(VERSION 3.12.0)
project(project_title LANGUAGES CXX C)

include(./common/cmake-common.cmake)

add_executable(main src/main.cpp src/ChorinTemam.cpp src/IncrementalChorinTemam.cpp src/MonolithicNavierStokes.cpp src/SteadyNavierStokes.cpp src/ConfigReader.cpp)
deal_ii_setup_target(main)
