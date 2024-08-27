# Automatically generated by scripts/boost/generate-ports.ps1

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO boostorg/filesystem
    REF boost-1.78.0
    SHA512 5a17a3084db574325394e56dc5fb740dfe6d0aa53026040ce98e6248a62501cf7b8f37721d7f6d6d0366ab5bdcbc52be9084ebf3147c91c6208fd9a67bfbac1c
    HEAD_REF master
)

if(NOT DEFINED CURRENT_HOST_INSTALLED_DIR)
    message(FATAL_ERROR "boost-filesystem requires a newer version of vcpkg in order to build.")
endif()
include(${CURRENT_HOST_INSTALLED_DIR}/share/boost-build/boost-modular-build.cmake)
boost_modular_build(SOURCE_PATH ${SOURCE_PATH})
include(${CURRENT_INSTALLED_DIR}/share/boost-vcpkg-helpers/boost-modular-headers.cmake)
boost_modular_headers(SOURCE_PATH ${SOURCE_PATH})
