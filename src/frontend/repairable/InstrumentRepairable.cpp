#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <fstream>

#include "../AngelixCommon.h"
#include "../smtlib/SMTLIB2.h"

enum VarTypes { ALL, INT, PTR, INT_AND_PTR };
enum DefectClass { UNKNOWN, ASSIGNMENTS, PTR_ASSIGNMENTS, IFS, LOOPS, GUARDS };


std::string DCtoString(DefectClass dc) {
  switch(dc) {
  case ASSIGNMENTS:
    return "\"A\"";
  case PTR_ASSIGNMENTS:
    return "\"P\"";
  case IFS:
    return "\"I\"";
  case LOOPS:
    return "\"L\"";
  case GUARDS:
    return "\"G\"";
  default:
    return "\"U\"";
  }
}


bool suitableVarDecl(VarDecl* vd, VarTypes collectedTypes) {
  bool ret = (collectedTypes == ALL ||
              (collectedTypes == INT &&
               (vd->getType().getTypePtr()->isIntegerType() ||
                vd->getType().getTypePtr()->isCharType())) ||
              (collectedTypes == PTR &&
               vd->getType().getTypePtr()->isPointerType()) ||
              (collectedTypes == INT_AND_PTR &&
               (vd->getType().getTypePtr()->isIntegerType() ||
                vd->getType().getTypePtr()->isCharType() ||
                vd->getType().getTypePtr()->isPointerType())));
  return ret;
}

bool suitableExpr(const Expr *expr) {
  const Type *typePtr = expr->getType().getTypePtr();

  if (getenv("ANGELIX_POINTER_VARIABLES")) {
    return typePtr->isIntegerType() ||
      typePtr->isCharType() ||
      typePtr->isPointerType();
  } else {
    return typePtr->isIntegerType() ||
      typePtr->isCharType();
  }
}

bool consistentExpr(const Expr *expr) {
  const Type *typePtr = expr->getType().getTypePtr();
  if (getenv("ANGELIX_ASSIGNMENTS_DEFECT_CLASS")) {
    if (typePtr->isPointerType()) return false;
  }

  if (getenv("ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS")) {
    if (typePtr->isIntegerType()) return false;
  }

  return true;
}

class CollectVariables : public StmtVisitor<CollectVariables> {
  std::unordered_set<VarDecl*> *VSet;
  std::unordered_set<MemberExpr*> *MSet;
  std::unordered_set<ArraySubscriptExpr*> *ASet;
  VarTypes Types;

public:
  CollectVariables(std::unordered_set<VarDecl*> *vset,
                   std::unordered_set<MemberExpr*> *mset,
                   std::unordered_set<ArraySubscriptExpr*> *aset, VarTypes t): VSet(vset), MSet(mset), ASet(aset), Types(t) {}

  void Collect(Expr *E) {
    if (E)
      Visit(E);
  }

  void Visit(Stmt* S) {
    StmtVisitor<CollectVariables>::Visit(S);
  }

  void VisitBinaryOperator(BinaryOperator *Node) {
    Collect(Node->getLHS());
    Collect(Node->getRHS());
  }

  void VisitUnaryOperator(UnaryOperator *Node) {
    Collect(Node->getSubExpr());
  }

  void VisitImplicitCastExpr(ImplicitCastExpr *Node) {
    Collect(Node->getSubExpr());
  }

  void VisitParenExpr(ParenExpr *Node) {
    Collect(Node->getSubExpr());
  }

  void VisitIntegerLiteral(IntegerLiteral *Node) {
  }

  void VisitCharacterLiteral(CharacterLiteral *Node) {
  }

  void VisitMemberExpr(MemberExpr *Node) {
    if (MSet && !getenv("ANGELIX_EXLUCDE_MEMBER_EXPR")) {
      MSet->insert(Node); // TODO: check memeber type?
    }
  }

  void VisitDeclRefExpr(DeclRefExpr *Node) {
    if (VSet && isa<VarDecl>(Node->getDecl())) {
      VarDecl* vd;
      if ((vd = cast<VarDecl>(Node->getDecl())) != NULL) {
        if (suitableVarDecl(vd, Types)) {
          VSet->insert(vd);
        }
      }
    }
  }

  void VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
    if (ASet) {
      ASet->insert(Node);
    }
  }


};

std::unordered_set<VarDecl*> collectVarsFromExpr(const Stmt* stmt, VarTypes types) {
  std::unordered_set<VarDecl*> set;
  CollectVariables T(&set, NULL, NULL, types);
  T.Visit(const_cast<Stmt*>(stmt));
  return set;
}

std::unordered_set<MemberExpr*> collectMemberExprFromExpr(const Stmt* stmt, VarTypes types) {
  std::unordered_set<MemberExpr*> set;
  CollectVariables T(NULL, &set, NULL, types);
  T.Visit(const_cast<Stmt*>(stmt));
  return set;
}

std::unordered_set<ArraySubscriptExpr*> collectArraySubscriptExprFromExpr(const Stmt* stmt, VarTypes types) {
  std::unordered_set<ArraySubscriptExpr*> set;
  CollectVariables T(NULL, NULL, &set, types);
  T.Visit(const_cast<Stmt*>(stmt));
  return set;
}

VarTypes
getVarTypes(const Type *typePtr) {
  VarTypes collectedTypes;
  if (getenv("ANGELIX_POINTER_VARIABLES")) {
    if (typePtr->isPointerType()) {
      collectedTypes = PTR;
      // std::cout << "[InstrumentRepariable.cpp] collectedTypes = PTR at line " << line << "\n";
    } else {
      collectedTypes = INT_AND_PTR;
      // std::cout << "[InstrumentRepariable.cpp] collectedTypes = INT_AND_PTR at line " << line << "\n";
    }
  } else {
    collectedTypes = INT;
    // std::cout << "[InstrumentRepariable.cpp] collectedTypes = INT at line " << line << "\n";
  }
  return collectedTypes;
}

std::pair< std::unordered_set<VarDecl*>, std::unordered_set<MemberExpr*> >
collectVarsFromScope(VarTypes collectedTypes, const ast_type_traits::DynTypedNode node,
                     ASTContext* context, unsigned line,
                     Rewriter &Rewrite) {
  const FunctionDecl* fd;
  if ((fd = node.get<FunctionDecl>()) != NULL) {
    std::unordered_set<VarDecl*> var_set;
    std::unordered_set<MemberExpr*> member_set;
    if (getenv("ANGELIX_FUNCTION_PARAMETERS")) {
      for (auto it = fd->param_begin(); it != fd->param_end(); ++it) {
        auto vd = cast<VarDecl>(*it);
        if (suitableVarDecl(vd, collectedTypes)) {
          var_set.insert(vd);
        }
      }
    }

    if (getenv("ANGELIX_GLOBAL_VARIABLES")) {
      ArrayRef<ast_type_traits::DynTypedNode> parents = context->getParents(node);
      if (parents.size() > 0) {
        const ast_type_traits::DynTypedNode parent = *(parents.begin()); // TODO: for now only first
        const TranslationUnitDecl* tu;
        if ((tu = parent.get<TranslationUnitDecl>()) != NULL) {
          for (auto it = tu->decls_begin(); it != tu->decls_end(); ++it) {
            if (isa<VarDecl>(*it)) {
              VarDecl* vd = cast<VarDecl>(*it);
              unsigned beginLine = getDeclExpandedLine(vd, context->getSourceManager());
              if (line > beginLine && suitableVarDecl(vd, collectedTypes)) {
                var_set.insert(vd);
              }
            }
          }
        }
      }
    }
    std::pair< std::unordered_set<VarDecl*>, std::unordered_set<MemberExpr*> > result(var_set, member_set);
    return result;

  } else {

    std::unordered_set<VarDecl*> var_set;
    std::unordered_set<MemberExpr*> member_set;
    const CompoundStmt* cstmt;
    if ((cstmt = node.get<CompoundStmt>()) != NULL) {
      for (auto it = cstmt->body_begin(); it != cstmt->body_end(); ++it) {

        if (isa<BinaryOperator>(*it)) {
          BinaryOperator* op = cast<BinaryOperator>(*it);
          SourceRange expandedLoc = getExpandedLoc(op, context->getSourceManager());
          unsigned beginLine = context->getSourceManager().getExpansionLineNumber(expandedLoc.getBegin());
          if (line > beginLine &&
              BinaryOperator::getOpcodeStr(op->getOpcode()).lower() == "=" &&
              isa<DeclRefExpr>(op->getLHS())) {
            DeclRefExpr* dref = cast<DeclRefExpr>(op->getLHS());
            VarDecl* vd;
            if ((vd = cast<VarDecl>(dref->getDecl())) != NULL && suitableVarDecl(vd, collectedTypes)) {
              var_set.insert(vd);
            }
          }
        }

        if (isa<DeclStmt>(*it)) {
          DeclStmt* dstmt = cast<DeclStmt>(*it);
          SourceRange expandedLoc = getExpandedLoc(dstmt, context->getSourceManager());
          unsigned beginLine = context->getSourceManager().getExpansionLineNumber(expandedLoc.getBegin());
          if (dstmt->isSingleDecl()) {
            Decl* d = dstmt->getSingleDecl();
            if (isa<VarDecl>(d)) {
              VarDecl* vd = cast<VarDecl>(d);
              if (line > beginLine && vd->hasInit() && suitableVarDecl(vd, collectedTypes)) {
                var_set.insert(vd);
              } else {
                if (getenv("ANGELIX_INIT_UNINIT_VARS")) {
                  if (line > beginLine && suitableVarDecl(vd, collectedTypes)) {
                    std::ostringstream stringStream;
                    stringStream << vd->getType().getAsString() << " "
                                 << vd->getNameAsString()
                                 << " = 0;";
                    std::string replacement = stringStream.str();
                    Rewrite.ReplaceText(expandedLoc, replacement);
                    if (suitableVarDecl(vd, collectedTypes))
                      var_set.insert(vd);
                  }
                }
              }
            }
          }
        }

        if (getenv("ANGELIX_USED_VARIABLES")) {
          Stmt* stmt = cast<Stmt>(*it);
          SourceRange expandedLoc = getExpandedLoc(stmt, context->getSourceManager());
          unsigned beginLine = context->getSourceManager().getExpansionLineNumber(expandedLoc.getBegin());
          if (line > beginLine) {
            std::unordered_set<VarDecl*> varsFromExpr = collectVarsFromExpr(*it, collectedTypes);
            var_set.insert(varsFromExpr.begin(), varsFromExpr.end());

            //TODO: should be generalized for other cases:
            if (isa<IfStmt>(*it)) {
              IfStmt* ifStmt = cast<IfStmt>(*it);
              Stmt* thenStmt = ifStmt->getThen();
              if (isa<CallExpr>(*thenStmt)) {
                CallExpr* callExpr = cast<CallExpr>(thenStmt);
                for (auto a = callExpr->arg_begin(); a != callExpr->arg_end(); ++a) {
                  auto e = cast<Expr>(*a);
                  std::unordered_set<MemberExpr*> membersFromArg = collectMemberExprFromExpr(e, collectedTypes);
                  member_set.insert(membersFromArg.begin(), membersFromArg.end());
                }
              }
            }
          }
        }

      }
    }

    ArrayRef<ast_type_traits::DynTypedNode> parents = context->getParents(node);
    if (parents.size() > 0) {
      const ast_type_traits::DynTypedNode parent = *(parents.begin()); // TODO: for now only first
      std::pair< std::unordered_set<VarDecl*>, std::unordered_set<MemberExpr*> > parent_vars =
        collectVarsFromScope(collectedTypes, parent, context, line, Rewrite);
      var_set.insert(parent_vars.first.cbegin(), parent_vars.first.cend());
      member_set.insert(parent_vars.second.cbegin(), parent_vars.second.cend());
    }
    std::pair< std::unordered_set<VarDecl*>, std::unordered_set<MemberExpr*> > result(var_set, member_set);
    return result;
  }
}

class ExpressionHandler : public MatchFinder::MatchCallback {
public:
  ExpressionHandler(Rewriter &Rewrite, clang::CompilerInstance &CI, std::string type) :
    Rewrite(Rewrite), CI(CI), DC(UNKNOWN), type(type) {}

  ExpressionHandler(Rewriter &Rewrite, clang::CompilerInstance &CI, DefectClass DC,
                    std::string type) :
    Rewrite(Rewrite), CI(CI), DC(DC), type(type) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    const Expr *expr = Result.Nodes.getNodeAs<clang::Expr>("repairable");
    if (!expr) {
      return;
    }

    SourceManager &srcMgr = Rewrite.getSourceMgr();
    const LangOptions &langOpts = Rewrite.getLangOpts();
    SourceRange expandedLoc = getExpandedLoc(expr, srcMgr);

    unsigned beginLine = srcMgr.getExpansionLineNumber(expandedLoc.getBegin());
    unsigned beginColumn = srcMgr.getExpansionColumnNumber(expandedLoc.getBegin());
    unsigned endLine = srcMgr.getExpansionLineNumber(expandedLoc.getEnd());
    unsigned endColumn = srcMgr.getExpansionColumnNumber(expandedLoc.getEnd());

    if (suitableExpr(expr) && consistentExpr(expr)) {
      std::pair<FileID, unsigned> decLoc =
        srcMgr.getDecomposedExpansionLoc(expandedLoc.getBegin());

      if (srcMgr.getMainFileID() != decLoc.first)
        return;

      if (insideMacro(expr, srcMgr, langOpts)) {
        std::cout << "[InstrumentRepariable.cpp] skip: "
                  << beginLine << " " << beginColumn << " " << endLine << " " << endColumn << ": "
                  << toString(expr) << "\n";
        return;
      }

      std::cout << "[InstrumentRepariable.cpp] instrument: "
                << beginLine << " " << beginColumn << " " << endLine << " " << endColumn << ": "
                << toString(expr) << " / type: " << expr->getType().getCanonicalType().getAsString() << "\n";

      // // FIXME: a better solution is to instrument non-overlapping parts
      // std::cout << "check macro overlap: " << toString(expr) << "\n";
      // if (overlapWithMacro(expr, srcMgr, langOpts,
      //                      CI.getPreprocessor(), Result.Context)) {
      //   return;
      // }

      VarTypes collectedTypes = getVarTypes(expr->getType().getTypePtr());

      const ast_type_traits::DynTypedNode node = ast_type_traits::DynTypedNode::create(*expr);
      std::pair< std::unordered_set<VarDecl*>, std::unordered_set<MemberExpr*> > varsFromScope =
        collectVarsFromScope(collectedTypes, node, Result.Context, beginLine, Rewrite);
      std::unordered_set<VarDecl*> varsFromExpr = collectVarsFromExpr(expr, collectedTypes);
      std::unordered_set<MemberExpr*> memberFromExpr = collectMemberExprFromExpr(expr, collectedTypes);
      std::unordered_set<ArraySubscriptExpr*> arraySubscriptFromExpr =
        collectArraySubscriptExprFromExpr(expr, collectedTypes);

      std::unordered_set<VarDecl*> vars;
      vars.insert(varsFromScope.first.begin(), varsFromScope.first.end());
      vars.insert(varsFromExpr.begin(), varsFromExpr.end());
      std::unordered_set<MemberExpr*> members;
      members.insert(varsFromScope.second.begin(), varsFromScope.second.end());
      members.insert(memberFromExpr.begin(), memberFromExpr.end());

      std::ostringstream exprStream;
      std::ostringstream nameStream;
      std::string first_expr = "";
      bool first = true;
      for (auto it = vars.begin(); it != vars.end(); ++it) {
        if (first) {
          first = false;
          first_expr = (*it)->getName().str();
        } else {
          exprStream << ", ";
          nameStream << ", ";
        }
        VarDecl* var = *it;
        exprStream << var->getName().str();
        nameStream << "\"" << var->getName().str() << "\"";
      }

      for (auto it = members.begin(); it != members.end(); ++it) {
        if (first) {
          first = false;
          first_expr = toString(*it);
        } else {
          exprStream << ", ";
          nameStream << ", ";
        }
        MemberExpr* me = *it;
        exprStream << toString(me);
        nameStream << "\"" << toString(me) << "\"";
      }

      for (auto it = arraySubscriptFromExpr.begin(); it != arraySubscriptFromExpr.end(); ++it) {
        if (first) {
          first = false;
        } else {
          exprStream << ", ";
          nameStream << ", ";
        }
        ArraySubscriptExpr* ae = *it;
        exprStream << toString(ae->getLHS()) << "[" << toString(ae->getRHS()) << "]";
        nameStream << "\"" <<  toString(ae->getLHS()) << "_LBRSQR_" << toString(ae->getRHS()) << "_RBRSQR_" << "\"";
      }

      const Type *typePtr = expr->getType().getTypePtr();
      std::string typeStr = typePtr->isPointerType()? "void_ptr" : "int";
      std::string otherTypesStr = typePtr->isPointerType()? "void*[]" : "int[]";

      std::string env_exp_names;
      std::string env_exps;
      int size;
      if (getenv("ANGELIX_EMPTY_ENV_EXPS")) {
        env_exp_names = "";
        env_exps = "";
        size = 0;
      } else {
        env_exp_names = nameStream.str();
        env_exps = exprStream.str();
        size = vars.size() + members.size() + arraySubscriptFromExpr.size();
      }

      std::ostringstream stringStream;
      stringStream << "ANGELIX_EXPR("
                   << DCtoString(DC) << ", "
                   << typeStr << ", "
                   << toString(expr) << ", "
                   << beginLine << ", "
                   << beginColumn << ", "
                   << endLine << ", "
                   << endColumn << ", "
                   << "((char*[]){" << env_exp_names << "}), "
                   << "((" << otherTypesStr << "){" << env_exps << "}), "
                   << size
                   << ")";
      std::string replacement = stringStream.str();

      Rewrite.ReplaceText(expandedLoc, replacement);

      // extract a smt2 file
      char *angelix_extracted = getenv("ANGELIX_EXTRACTED");
      if (angelix_extracted) {
        std::ostringstream exprId;
        exprId << beginLine << "-" << beginColumn << "-" << endLine << "-" << endColumn;
        std::string extractedDir(angelix_extracted);
        std::ofstream fs(extractedDir + "/" + exprId.str() + ".smt2");
        if (getenv("ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS")) {
          if (!first_expr.empty())
            fs << "(assert " << first_expr << ")\n";
          else
            fs << "(assert 0)\n";
        } else {
          fs << "(assert " << toSMTLIB2(expr) << ")\n";
        }
      }
    } else {
      std::cout << "[InstrumentRepariable.cpp] skip: "
                << beginLine << " " << beginColumn << " " << endLine << " " << endColumn << ": "
                << toString(expr) << " / type: " << expr->getType().getCanonicalType().getAsString() << "\n";
    }
  }

protected:
  Rewriter &Rewrite;
  CompilerInstance &CI;
  DefectClass DC;
  std::string type;
};


class RHSHandler : public ExpressionHandler {
public:
  RHSHandler(Rewriter &Rewrite, clang::CompilerInstance &CI) :
    ExpressionHandler(Rewrite, CI, ASSIGNMENTS, "int") {}
};

class PtrRHSHandler : public ExpressionHandler {
public:
  PtrRHSHandler(Rewriter &Rewrite, clang::CompilerInstance &CI) :
    ExpressionHandler(Rewrite, CI, PTR_ASSIGNMENTS, "int") {}
};

class IFHandler : public ExpressionHandler {
public:
  IFHandler(Rewriter &Rewrite, clang::CompilerInstance &CI) :
    ExpressionHandler(Rewrite, CI, IFS, "bool") {}
};


class LoopHandler : public ExpressionHandler {
public:
  LoopHandler(Rewriter &Rewrite, clang::CompilerInstance &CI) :
    ExpressionHandler(Rewrite, CI, LOOPS, "bool") {}
};

class StatementHandler : public MatchFinder::MatchCallback {
public:
  StatementHandler(Rewriter &Rewrite, DefectClass DC) : Rewrite(Rewrite), DC(DC) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const Stmt *stmt = Result.Nodes.getNodeAs<clang::Stmt>("repairable")) {
      SourceManager &srcMgr = Rewrite.getSourceMgr();
      const LangOptions &langOpts = Rewrite.getLangOpts();

      if (insideMacro(stmt, srcMgr, langOpts))
        return;

      SourceRange expandedLoc = getExpandedLoc(stmt, srcMgr);

      std::pair<FileID, unsigned> decLoc = srcMgr.getDecomposedExpansionLoc(expandedLoc.getBegin());
      if (srcMgr.getMainFileID() != decLoc.first)
        return;

      unsigned beginLine = srcMgr.getExpansionLineNumber(expandedLoc.getBegin());
      unsigned beginColumn = srcMgr.getExpansionColumnNumber(expandedLoc.getBegin());
      unsigned endLine = srcMgr.getExpansionLineNumber(expandedLoc.getEnd());
      unsigned endColumn = srcMgr.getExpansionColumnNumber(expandedLoc.getEnd());

      std::cout << beginLine << " " << beginColumn << " " << endLine << " " << endColumn << "\n"
                << toString(stmt) << "\n";

      std::ostringstream stringStream;
      stringStream << "if ("
                   << "ANGELIX_EXPR("
                   << DCtoString(DC) << ", "
                   << "bool ,"
                   << 1 << ", "
                   << beginLine << ", "
                   << beginColumn << ", "
                   << endLine << ", "
                   << endColumn
                   << ")"
                   << ") "
                   << toString(stmt);
      std::string replacement = stringStream.str();

      Rewrite.ReplaceText(expandedLoc, replacement);
    }
  }

protected:
  Rewriter &Rewrite;
  DefectClass DC;
};


class GuardHandler : public StatementHandler {
public:
  GuardHandler(Rewriter &Rewrite) :
    StatementHandler(Rewrite, GUARDS) {}
};


class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R, clang::CompilerInstance &CI) :
    CI(CI),
    HandlerForRHS(R, CI), HandlerForPtrRHS(R, CI),  HandlerForIF(R, CI),
    HandlerForLoop(R, CI), HandlerForGuard(R) {
    if (getenv("ANGELIX_IGNORE_TRIVIAL")) {
      if (getenv("ANGELIX_IF_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableIfCondition, &HandlerForIF);
      if (getenv("ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableLoopCondition, &HandlerForLoop);
      if (getenv("ANGELIX_ASSIGNMENTS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableAssignment, &HandlerForRHS);
      if (getenv("ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableAssignment, &HandlerForPtrRHS);
    } else {
      if (getenv("ANGELIX_IF_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableIfCondition, &HandlerForIF);

      if (getenv("ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableLoopCondition, &HandlerForLoop);

      if (getenv("ANGELIX_ASSIGNMENTS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableAssignment, &HandlerForRHS);

      if (getenv("ANGELIX_PTR_ASSIGNMENTS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableAssignment, &HandlerForPtrRHS);
    }
    if (getenv("ANGELIX_GUARDS_DEFECT_CLASS"))
      Matcher.addMatcher(InterestingStatement, &HandlerForGuard);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Matcher.matchAST(Context);
  }

private:
  CompilerInstance &CI;
  RHSHandler HandlerForRHS;
  PtrRHSHandler HandlerForPtrRHS;
  IFHandler HandlerForIF;
  LoopHandler HandlerForLoop;
  GuardHandler HandlerForGuard;
  MatchFinder Matcher;
};


class InstrumentRepairableAction : public ASTFrontendAction {
public:
  InstrumentRepairableAction() {}

  void EndSourceFileAction() override {
    FileID ID = TheRewriter.getSourceMgr().getMainFileID();
    if (INPLACE_MODIFICATION) {
      //overwriteMainChangedFile(TheRewriter);
      TheRewriter.overwriteChangedFiles();
    } else {
      TheRewriter.getEditBuffer(ID).write(llvm::outs());
    }
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter, getCompilerInstance());
  }

private:
  Rewriter TheRewriter;
};


// Apply a custom category to all command-line options so that they are the only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("angelix options");


int main(int argc, const char **argv) {
  // CommonOptionsParser constructor will parse arguments and create a
  // CompilationDatabase.  In case of error it will terminate the program.
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);

  // We hand the CompilationDatabase we created and the sources to run over into the tool constructor.
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  return Tool.run(newFrontendActionFactory<InstrumentRepairableAction>().get());
}
