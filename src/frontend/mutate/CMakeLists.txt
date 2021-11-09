set(LLVM_LINK_COMPONENTS support)
set(LLVM_USED_LIBS clangTooling clangBasic clangAST)

add_clang_executable(mutate
  Mutate.cpp
  )

target_link_libraries(mutate
  clangTooling
  clangBasic
  clangASTMatchers
  smtlib2
  )
