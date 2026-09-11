cmake_minimum_required(VERSION 3.16)

# change this to your desired addon name
# make sure, that the name matches exactly the argument of PYBIND11_MODULE()
set(addon_name ngsdiffgeo)

project(${addon_name})
include(ngsolve_addon.cmake)

# change the source file arguments to your source files
add_ngsolve_addon(${addon_name} src/ngsdiffgeo.cpp src/riemannian_manifold.cpp src/tensor_fields.cpp src/symbolic_expression.cpp src/coefficient_grad.cpp src/kforms.cpp)

include(CTest)
if(BUILD_TESTING)
  # NGSolve's C++ libraries use the Python C API.  Extension modules obtain
  # those symbols from the interpreter that imports them, but these tests are
  # standalone executables and therefore need to embed/link Python explicitly.
  find_package(Python3 REQUIRED COMPONENTS Development)

  add_executable(test_tensor_meta tests/test_tensor_meta.cpp)
  target_include_directories(
    test_tensor_meta
    BEFORE PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      $<TARGET_PROPERTY:ngsolve,INTERFACE_INCLUDE_DIRECTORIES>
  )
  target_link_libraries(test_tensor_meta PRIVATE ngsolve Python3::Python)
  set_target_properties(test_tensor_meta PROPERTIES CXX_STANDARD 17)
  add_test(NAME tensor_meta COMMAND test_tensor_meta)
  set(cpp_test_targets test_tensor_meta)

  if(NOT WIN32)
    add_executable(
      test_tensor_field_forwarding
      tests/test_tensor_field_forwarding.cpp
    )
    target_include_directories(
      test_tensor_field_forwarding
      BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<TARGET_PROPERTY:ngsolve,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(
      test_tensor_field_forwarding
      PRIVATE ngsdiffgeo ngsolve Python3::Python
    )
    set_target_properties(
      test_tensor_field_forwarding
      PROPERTIES CXX_STANDARD 17
    )
    add_test(NAME tensor_field_forwarding COMMAND test_tensor_field_forwarding)

    add_executable(
      test_coefficient_grad
      tests/test_coefficient_grad.cpp
    )
    target_include_directories(
      test_coefficient_grad
      BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<TARGET_PROPERTY:ngsolve,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(
      test_coefficient_grad
      PRIVATE ngsdiffgeo ngsolve Python3::Python
    )
    set_target_properties(
      test_coefficient_grad
      PROPERTIES CXX_STANDARD 17
    )
    add_test(NAME coefficient_grad COMMAND test_coefficient_grad)

    add_executable(test_kforms tests/test_kforms.cpp)
    target_include_directories(
      test_kforms BEFORE PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      $<TARGET_PROPERTY:ngsolve,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(test_kforms PRIVATE ngsdiffgeo ngsolve Python3::Python)
    set_target_properties(test_kforms PROPERTIES CXX_STANDARD 17)
    foreach(check properties equivalence patterns codegen validation transforms shared_gradient shared_wedge simd)
      add_test(NAME kforms_${check} COMMAND test_kforms ${check})
    endforeach()

    add_executable(
      test_riemannian_manifold
      tests/test_riemannian_manifold.cpp
    )
    target_include_directories(
      test_riemannian_manifold BEFORE PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      $<TARGET_PROPERTY:ngsolve,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(
      test_riemannian_manifold PRIVATE ngsdiffgeo ngsolve Python3::Python
    )
    set_target_properties(
      test_riemannian_manifold PROPERTIES CXX_STANDARD 17
    )
    add_test(NAME riemannian_manifold COMMAND test_riemannian_manifold)

    list(
      APPEND cpp_test_targets
      test_tensor_field_forwarding
      test_coefficient_grad
      test_kforms
      test_riemannian_manifold
    )
  endif()

  add_custom_target(cpp_tests DEPENDS ${cpp_test_targets})
endif()

# install the compiled python module and __init__.py ( don't change this )
set(install_dir ${ADDON_INSTALL_DIR_PYTHON}/${addon_name})
install(TARGETS ${addon_name} DESTINATION ${install_dir})
install(FILES src/__init__.py src/wrappers.py src/manifolds.py DESTINATION ${install_dir})

# install additional python files/demos/examples
#install(FILES demos/exploremesh.py DESTINATION ${install_dir}/demos)

# generate stub files for autocomplete in IDEs
# this must be done at the very end (such that the stubgen generation happens after the python modules are installed)
ngsolve_generate_stub_files(${addon_name})
