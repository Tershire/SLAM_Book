# Automatically generated by scripts/boost/generate-ports.ps1

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO boostorg/date_time
    REF boost-1.81.0
    SHA512 07eeeb09bab0e64a662a1ea92a24c93e8c119d804e3b63f71ccbb533348c6dbeaf8f759847409fe23c902cadcfd025fee804d67f9a4182eb336094f0c1f37d9a
    HEAD_REF master
)

include(${CURRENT_HOST_INSTALLED_DIR}/share/boost-build/boost-modular-build.cmake)
boost_modular_build(SOURCE_PATH ${SOURCE_PATH})
include(${CURRENT_INSTALLED_DIR}/share/boost-vcpkg-helpers/boost-modular-headers.cmake)
boost_modular_headers(SOURCE_PATH ${SOURCE_PATH})