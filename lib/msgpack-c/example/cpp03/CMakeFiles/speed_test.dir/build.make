# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c

# Include any dependencies generated for this target.
include example/cpp03/CMakeFiles/speed_test.dir/depend.make

# Include the progress variables for this target.
include example/cpp03/CMakeFiles/speed_test.dir/progress.make

# Include the compile flags for this target's objects.
include example/cpp03/CMakeFiles/speed_test.dir/flags.make

example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o: example/cpp03/CMakeFiles/speed_test.dir/flags.make
example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o: example/cpp03/speed_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o"
	cd /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/speed_test.dir/speed_test.cpp.o -c /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03/speed_test.cpp

example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/speed_test.dir/speed_test.cpp.i"
	cd /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03/speed_test.cpp > CMakeFiles/speed_test.dir/speed_test.cpp.i

example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/speed_test.dir/speed_test.cpp.s"
	cd /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03/speed_test.cpp -o CMakeFiles/speed_test.dir/speed_test.cpp.s

example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.requires:

.PHONY : example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.requires

example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.provides: example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.requires
	$(MAKE) -f example/cpp03/CMakeFiles/speed_test.dir/build.make example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.provides.build
.PHONY : example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.provides

example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.provides.build: example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o


# Object files for target speed_test
speed_test_OBJECTS = \
"CMakeFiles/speed_test.dir/speed_test.cpp.o"

# External object files for target speed_test
speed_test_EXTERNAL_OBJECTS =

example/cpp03/speed_test: example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o
example/cpp03/speed_test: example/cpp03/CMakeFiles/speed_test.dir/build.make
example/cpp03/speed_test: /usr/lib/x86_64-linux-gnu/libboost_timer.so
example/cpp03/speed_test: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
example/cpp03/speed_test: /usr/lib/x86_64-linux-gnu/libboost_system.so
example/cpp03/speed_test: example/cpp03/CMakeFiles/speed_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable speed_test"
	cd /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/speed_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example/cpp03/CMakeFiles/speed_test.dir/build: example/cpp03/speed_test

.PHONY : example/cpp03/CMakeFiles/speed_test.dir/build

example/cpp03/CMakeFiles/speed_test.dir/requires: example/cpp03/CMakeFiles/speed_test.dir/speed_test.cpp.o.requires

.PHONY : example/cpp03/CMakeFiles/speed_test.dir/requires

example/cpp03/CMakeFiles/speed_test.dir/clean:
	cd /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 && $(CMAKE_COMMAND) -P CMakeFiles/speed_test.dir/cmake_clean.cmake
.PHONY : example/cpp03/CMakeFiles/speed_test.dir/clean

example/cpp03/CMakeFiles/speed_test.dir/depend:
	cd /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03 /media/ceoebenezer/INFORMATION/EBZER_CODES/motion_detection/msgpack-c/example/cpp03/CMakeFiles/speed_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example/cpp03/CMakeFiles/speed_test.dir/depend

