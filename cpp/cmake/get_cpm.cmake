# Downloads CPM.cmake into the build tree once (https://github.com/cpm-cmake/CPM.cmake).
set(CPM_DOWNLOAD_VERSION 0.43.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
  message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
  file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION} STATUS cpm_status)
  list(GET cpm_status 0 cpm_code)
  if(NOT cpm_code EQUAL 0)
    file(REMOVE ${CPM_DOWNLOAD_LOCATION})
    message(FATAL_ERROR "CPM.cmake download failed: ${cpm_status}")
  endif()
endif()
include(${CPM_DOWNLOAD_LOCATION})
