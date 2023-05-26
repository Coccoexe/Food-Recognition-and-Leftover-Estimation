# CMakeList.txt : CMake project for Food-Recognition-and-Leftover-Estimation, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Food-Recognition-and-Leftover-Estimation")

# Add source to this project's executable.
add_executable (Food-Recognition-and-Leftover-Estimation "Food-Recognition-and-Leftover-Estimation.cpp" "Food-Recognition-and-Leftover-Estimation.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Food-Recognition-and-Leftover-Estimation PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
