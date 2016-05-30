cmake_minimum_required(VERSION 2.6)

project( ht )

if( UNIX )
  set( CMAKE_CXX_FLAGS "-Wall -std=c++14 -Werror=return-type -g" )
endif( UNIX )

aux_source_directory( . ht_src )
include_directories( . )

find_package( OpenCV REQUIRED )
include_directories( SYSTEM ${OpenCV_INCLUDE_DIR} )
set ( requiredLibs ${requiredLibs}  ${OpenCV_LIBS} )

include( FindCURL REQUIRED )
include_directories( SYSTEM ${CURL_INCLUDE_DIR} )
set( requiredLibs ${requiredLibs} ${CURL_LIBRARIES} )


message ( ${ht_src} )
add_executable( ht ${ht_src} )

target_link_libraries( ht ${CMAKE_DL_LIBS} ${requiredLibs})


aux_source_directory ( server/ ht_srv )
include_directories( server )

find_package( OpenCV REQUIRED )

message ( ${ht_srv} )
add_library( hts_tmp MODULE ${ht_srv} )

target_link_libraries( hts_tmp ${OpenCV_LIBS} )

add_custom_command(TARGET hts_tmp POST_BUILD
  COMMAND mv libhts_tmp.so libhts.so
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT Updated libhts.so)
