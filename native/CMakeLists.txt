cmake_minimum_required(VERSION 3.20)
project(svr_roughness_native VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT VCPKG_TARGET_TRIPLET)
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows")
    elseif(APPLE)
        set(VCPKG_TARGET_TRIPLET "x64-osx")
    else()
        set(VCPKG_TARGET_TRIPLET "x64-linux")
    endif()
endif()

if(UNIX AND NOT APPLE)
    file(GLOB_RECURSE PCL_HINTS "/usr/lib/*/cmake/pcl*/PCLConfig.cmake" "/usr/share/pcl*/PCLConfig.cmake")
    foreach(HINT_FILE ${PCL_HINTS})
        get_filename_component(HINT_DIR "${HINT_FILE}" DIRECTORY)
        list(APPEND CMAKE_PREFIX_PATH "${HINT_DIR}")
    endforeach()
    list(APPEND CMAKE_PREFIX_PATH "/usr/include/eigen3")
    find_package(MPI QUIET)
endif()

set(PACKAGE_VCPKG_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}")
if(EXISTS "${PACKAGE_VCPKG_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "${PACKAGE_VCPKG_PREFIX}")
endif()

add_compile_definitions(CGAL_SILENCE_DEPRECATION_WARNINGS)
if(APPLE)
    add_compile_options("-Wno-unknown-directives" "-Wno-cpp" "-Wno-deprecated-declarations")
endif()

find_package(PCL REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)
find_package(TBB REQUIRED)

add_library(SurfInspectNative SHARED
    abi_bridge.cpp
    surfinspect_native.h
    cloud_viewer.cpp
    cloud_viewer.h
    types.h
    pcl_dependencies.h
    pcl_dependencies.cpp
)

target_compile_definitions(SurfInspectNative PRIVATE SURFINSPECT_NATIVE_EXPORTS NOMINMAX WIN32_LEAN_AND_MEAN BOOST_BIND_GLOBAL_PLACEHOLDERS CGAL_SILENCE_DEPRECATION_WARNINGS)
target_include_directories(SurfInspectNative PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PCL_INCLUDE_DIRS}
)
target_link_directories(SurfInspectNative PRIVATE ${PCL_LIBRARY_DIRS})
target_compile_definitions(SurfInspectNative PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(SurfInspectNative PRIVATE
    ${PCL_LIBRARIES}
    Boost::filesystem
    CGAL::CGAL
    TBB::tbb
)

install(TARGETS SurfInspectNative
    RUNTIME DESTINATION svr_roughness/native
    LIBRARY DESTINATION svr_roughness/native
    ARCHIVE DESTINATION svr_roughness/native
)
