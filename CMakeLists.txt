cmake_minimum_required(VERSION 3.16)
project(sign_language_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(mediapipe REQUIRED)

add_executable(sign_language_recognition sign_language_recognition.cpp)

target_include_directories(sign_language_recognition PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(sign_language_recognition PRIVATE ${OpenCV_LIBS} mediapipe::mediapipe)
