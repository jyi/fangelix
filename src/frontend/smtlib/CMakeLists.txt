set(LLVM_LINK_COMPONENTS support)
set(LLVM_USED_LIBS clangTooling clangBasic clangAST)

add_clang_library(smtlib2
  SMTLIB2.cpp
  )

target_link_libraries(smtlib2
  clangTooling
  clangBasic
  clangASTMatchers
  )
