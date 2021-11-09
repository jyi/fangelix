#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <fstream>
#include <random>

#include "Mutate.h"


class Count {
public:
  void inc() {
    this->data++;
  }

  int get() {
    return this->data;
  }

private:
  int data = 0;
};


class CountExpressions: public MatchFinder::MatchCallback {
public:
  CountExpressions(Count &C) : C(C) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const BinaryOperator *binExpr =
         Result.Nodes.getNodeAs<clang::BinaryOperator>("mutable")) {
      if (!binExpr->isRelationalOp()) {
        return;
      }

      std::cout << "mutable: " << toString(binExpr) << "\n";
      C.inc();
    }
  }

private:
  Count &C;
};


class MutateExpression : public MatchFinder::MatchCallback {
public:
  typedef BinaryOperatorKind Opcode;

  MutateExpression(Rewriter &Rewrite, std::set<int> &OS, std::set<int> &MS,
                   int &matchID, int max_match_id) :
    Rewrite(Rewrite), OS(OS), MS(MS), matchID(matchID), max_match_id(max_match_id) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
    if (const BinaryOperator *binExpr =
         Result.Nodes.getNodeAs<clang::BinaryOperator>("mutable")) {
      if (!binExpr->isRelationalOp()) {
        return;
      }

      SourceManager &srcMgr = Rewrite.getSourceMgr();
      const LangOptions &langOpts = Rewrite.getLangOpts();
      std::cout << "match ID: " << matchID << "\n";
      if (OS.find(matchID) != OS.end()) {
        if (insideMacro(binExpr, srcMgr, langOpts)) {
          std::cout << "inside macro\n";
          OS.erase(matchID);
          for (int i = matchID + 1; i < max_match_id ; i++) {
            if (OS.find(i) != OS.end()) {
              OS.insert(i);
              break;
            }
          }
          return;
        }

        // found a match
        std::cout << "match found\n";

        std::ostringstream stringStream;
        stringStream << toString(binExpr->getLHS()) << " " <<
          BinaryOperator::getOpcodeStr(mutateRelatonalOp(binExpr->getOpcode())).str() <<
          " " << toString(binExpr->getRHS());
        std::string replacement = stringStream.str();
        std::cout << "replacement: " << replacement << "\n";

        Rewrite.ReplaceText(getExpandedLoc(binExpr, srcMgr), replacement);
        MS.insert(matchID);
      } else {
        std::cout << "match not found\n";
      }
      matchID++;
    }
  }

  Opcode mutateRelatonalOp(Opcode opc) {
    switch (opc) {
    default:
      llvm_unreachable("Not a supported operator.");
    case BO_LT: return BO_GE;
    case BO_GT: return BO_LE;
    case BO_LE: return BO_GT;
    case BO_GE: return BO_LT;
    case BO_EQ: return BO_NE;
    case BO_NE: return BO_EQ;
    case BO_LAnd: return BO_LOr;
    case BO_LOr: return BO_LAnd;
    }
  }

private:
  Rewriter &Rewrite;
  std::set<int> &OS;
  std::set<int> &MS;
  int &matchID;
  int max_match_id;
};


class MutableCountConsumer: public ASTConsumer {
public:
  MutableCountConsumer(Count &C) : C(C), CountExp(CountExpressions(C)) {
    if (getenv("ANGELIX_IGNORE_TRIVIAL")) {
      if (getenv("ANGELIX_IF_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableIfCondition, &CountExp);

      if (getenv("ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableLoopCondition, &CountExp);

      if (getenv("ANGELIX_ASSIGNMENTS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableAssignment, &CountExp);
    } else {
      if (getenv("ANGELIX_IF_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableIfCondition, &CountExp);

      if (getenv("ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableLoopCondition, &CountExp);
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Matcher.matchAST(Context);
  }

private:
  MatchFinder Matcher;
  Count &C;
  CountExpressions CountExp;
};


class MutateConsumer : public ASTConsumer {
public:
  MutateConsumer(Rewriter &R, std::set<int> &OS, std::set<int> &MS, int o) :
    OS(OS), MS(MS), matchID(0), max_match_id(max_match_id),
    MutExp(R, OS, MS, matchID, max_match_id) {
    if (getenv("ANGELIX_IGNORE_TRIVIAL")) {
      if (getenv("ANGELIX_IF_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableIfCondition, &MutExp);

      if (getenv("ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableLoopCondition, &MutExp);

      if (getenv("ANGELIX_ASSIGNMENTS_DEFECT_CLASS"))
        Matcher.addMatcher(NonTrivialRepairableAssignment, &MutExp);
    } else {
      if (getenv("ANGELIX_IF_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableIfCondition, &MutExp);

      if (getenv("ANGELIX_LOOP_CONDITIONS_DEFECT_CLASS"))
        Matcher.addMatcher(RepairableLoopCondition, &MutExp);
    }
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    Matcher.matchAST(Context);
  }

private:
  std::set<int> &OS;
  std::set<int> &MS;
  MutateExpression MutExp;
  MatchFinder Matcher;
  int matchID;
  int max_match_id;
};


class CountRepairableAction: public ASTFrontendAction {
public:
  CountRepairableAction(Count &C) : C(C) {}

  virtual std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::unique_ptr<clang::ASTConsumer>(new MutableCountConsumer(C));
  }

private:
  Count &C;
};


class MutateAction : public ASTFrontendAction {
public:
  MutateAction(std::set<int> &OS, std::set<int> &MS, int max_match_id) :
    OS(OS), MS(MS), max_match_id(max_match_id) {}

  void EndSourceFileAction() override {
    FileID ID = TheRewriter.getSourceMgr().getMainFileID();
    if (INPLACE_MODIFICATION) {
      //overwriteMainChangedFile(TheRewriter);
      TheRewriter.overwriteChangedFiles();
    } else {
      TheRewriter.getEditBuffer(ID).write(llvm::outs());
    }
  }

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::unique_ptr<clang::ASTConsumer>(new MutateConsumer(TheRewriter,
                                                                  OS, MS, max_match_id));
  }

private:
  Rewriter TheRewriter;
  std::set<int> &OS;
  std::set<int> &MS;
  int max_match_id;
};


class CountMutableActionFactory : public FrontendActionFactory {
public:
  CountMutableActionFactory(Count &C) : C(C) {}

  FrontendAction *create() override {
    return new CountRepairableAction(C);
  }

private:
  Count &C;
};


class MutateActionFactory : public FrontendActionFactory {
public:
  MutateActionFactory(std::set<int> &OS, std::set<int> &MS, int max_match_id) :
    OS(OS), MS(MS), max_match_id(max_match_id) {}

  FrontendAction *create() override {
    return new MutateAction(OS, MS, max_match_id);
  }

private:
  std::set<int> &OS;
  std::set<int> &MS;
  int max_match_id;
};

// Apply a custom category to all command-line options so that they are the only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("mutate options");


int main(int argc, const char **argv) {
  // CommonOptionsParser constructor will parse arguments and create a
  // CompilationDatabase.  In case of error it will terminate the program.
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);

  // We hand the CompilationDatabase we created and the sources to run over into the tool constructor.
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  // Phase 1
  Count C;
  CountMutableActionFactory *fac1 = new CountMutableActionFactory(C);
  Tool.run(fac1);
  delete fac1;
  std::cout << "# of mutables: " << C.get() << "\n";

  if (C.get() <= 0) {
    std::cerr << "No mutable found\n";
    exit(1);
  }

  unsigned int mut_num = (unsigned int) atoi(getenv("ANGELIX_MUTATION_NUM"));
  std::cout << "mut_num: " << mut_num << "\n";

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(0, C.get() - 1);

  std::set<int> occurrence_set;
  while (occurrence_set.size() < mut_num) {
    int occurrence = dist(gen);
    occurrence_set.insert(occurrence);
  }

  std::set<int> mutated_set;

  // Phase 2
  MutateActionFactory *fac2 = new MutateActionFactory(occurrence_set, mutated_set, C.get());
  Tool.run(fac2);
  delete fac2;
  if (mutated_set.size() < mut_num) {
    std::cerr << "Error: only " << mutated_set.size() << " are muated\n";
    exit(1);
  }
}
