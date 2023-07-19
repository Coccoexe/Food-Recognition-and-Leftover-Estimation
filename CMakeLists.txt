# CMakeList.txt : CMake project for Food-Recognition-and-Leftover-Estimation, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

#project
project ("Food-Recognition-and-Leftover-Estimation")

# opencv
find_package(OpenCV REQUIRED)
find_package(Python REQUIRED Development) #NEW

# include
include_directories(${OpenCV_INCLUDE_DIRS})

# executables & libraries
add_executable (${PROJECT_NAME}  "src/main.cpp" "src/BoundingBoxes.cpp" "src/Segmentation.cpp" "src/Metrics.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Python::Python) #NEW

# check
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
