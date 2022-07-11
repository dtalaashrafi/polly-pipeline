//===- CodeGeneration.cpp - Code generate the Scops using ISL. ---------======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The CodeGeneration pass takes a Scop created by ScopInfo and translates it
// back to LLVM-IR using the ISL code generator.
//
// The Scop describes the high level memory behavior of a control flow region.
// Transformation passes can update the schedule (execution order) of statements
// in the Scop. ISL is used to generate an abstract syntax tree that reflects
// the updated execution order. This clast is used to create new LLVM-IR that is
// computationally equivalent to the original control flow region, but executes
// its code in the new execution order defined by the changed schedule.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/CodeGeneration.h"
#include "polly/CodeGen/IRBuilder.h"
#include "polly/CodeGen/IslAst.h"
#include "polly/CodeGen/IslNodeBuilder.h"
#include "polly/CodeGen/PerfMonitor.h"
#include "polly/CodeGen/Utils.h"
#include "polly/DependenceInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/ast.h"
#include <cassert>
// pipeline (*)
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-codegen"

static cl::opt<bool> Verify("polly-codegen-verify",
                            cl::desc("Verify the function generated by Polly"),
                            cl::Hidden, cl::init(false), cl::ZeroOrMore,
                            cl::cat(PollyCategory));
static cl::opt<std::string>
    PipelineCodeGen("polly-pipeline-codegen",
                  cl::desc("Code generation for pipeline pattern between iterations of "
                           "different loops (yes/no)"),
                  cl::Hidden, cl::init("no"), cl::ZeroOrMore,
                  cl::cat(PollyCategory));

bool polly::PerfMonitoring;

static cl::opt<bool, true>
    XPerfMonitoring("polly-codegen-perf-monitoring",
                    cl::desc("Add run-time performance monitoring"), cl::Hidden,
                    cl::location(polly::PerfMonitoring), cl::init(false),
                    cl::ZeroOrMore, cl::cat(PollyCategory));

STATISTIC(ScopsProcessed, "Number of SCoP processed");
STATISTIC(CodegenedScops, "Number of successfully generated SCoPs");
STATISTIC(CodegenedAffineLoops,
          "Number of original affine loops in SCoPs that have been generated");
STATISTIC(CodegenedBoxedLoops,
          "Number of original boxed loops in SCoPs that have been generated");

namespace polly {

/// Mark a basic block unreachable.
///
/// Marks the basic block @p Block unreachable by equipping it with an
/// UnreachableInst.
void markBlockUnreachable(BasicBlock &Block, PollyIRBuilder &Builder) {
  auto *OrigTerminator = Block.getTerminator();
  Builder.SetInsertPoint(OrigTerminator);
  Builder.CreateUnreachable();
  OrigTerminator->eraseFromParent();
}
} // namespace polly

static void verifyGeneratedFunction(Scop &S, Function &F, IslAstInfo &AI) {
  if (!Verify || !verifyFunction(F, &errs()))
    return;

  LLVM_DEBUG({
    errs() << "== ISL Codegen created an invalid function ==\n\n== The "
              "SCoP ==\n";
    errs() << S;
    errs() << "\n== The isl AST ==\n";
    AI.print(errs());
    errs() << "\n== The invalid function ==\n";
    F.print(errs());
  });

  llvm_unreachable("Polly generated function could not be verified. Add "
                   "-polly-codegen-verify=false to disable this assertion.");
}

// CodeGeneration adds a lot of BBs without updating the RegionInfo
// We make all created BBs belong to the scop's parent region without any
// nested structure to keep the RegionInfo verifier happy.
static void fixRegionInfo(Function &F, Region &ParentRegion, RegionInfo &RI) {
  for (BasicBlock &BB : F) {
    if (RI.getRegionFor(&BB))
      continue;

    RI.setRegionFor(&BB, &ParentRegion);
  }
}

/// Remove all lifetime markers (llvm.lifetime.start, llvm.lifetime.end) from
/// @R.
///
/// CodeGeneration does not copy lifetime markers into the optimized SCoP,
/// which would leave the them only in the original path. This can transform
/// code such as
///
///     llvm.lifetime.start(%p)
///     llvm.lifetime.end(%p)
///
/// into
///
///     if (RTC) {
///       // generated code
///     } else {
///       // original code
///       llvm.lifetime.start(%p)
///     }
///     llvm.lifetime.end(%p)
///
/// The current StackColoring algorithm cannot handle if some, but not all,
/// paths from the end marker to the entry block cross the start marker. Same
/// for start markers that do not always cross the end markers. We avoid any
/// issues by removing all lifetime markers, even from the original code.
///
/// A better solution could be to hoist all llvm.lifetime.start to the split
/// node and all llvm.lifetime.end to the merge node, which should be
/// conservatively correct.
static void removeLifetimeMarkers(Region *R) {
  for (auto *BB : R->blocks()) {
    auto InstIt = BB->begin();
    auto InstEnd = BB->end();

    while (InstIt != InstEnd) {
      auto NextIt = InstIt;
      ++NextIt;

      if (auto *IT = dyn_cast<IntrinsicInst>(&*InstIt)) {
        switch (IT->getIntrinsicID()) {
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
          BB->getInstList().erase(InstIt);
          break;
        default:
          break;
        }
      }

      InstIt = NextIt;
    }
  }
}

// pipeline (*)
// TODO: add an option for running this function.
// We should run it ONLY if we are pipelining.
void extractPipeline(PollyIRBuilder &Builder, Function *F)
{
  errs() << "*************** "<< "\n";
  
  auto M = F->getParent();

  std::vector<BasicBlock *> bb_vec;
  for(auto bb = F->begin() ; bb != F->end() ; bb++)
        bb_vec.push_back(&*bb);

  ArrayRef<BasicBlock *> bb_arr(bb_vec);

  CodeExtractorAnalysisCache CEAC(*F);
  CodeExtractor extractor(bb_arr,nullptr ,true , nullptr, nullptr, nullptr, true, true, "pipeline");
  auto new_F = extractor.extractCodeRegion(CEAC);
  auto arg_type = new_F->getArg(0)->getType();


  const std::string Name = "make_parallel_region";
  Function *P = M->getFunction(Name);
  if (!P) 
  {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {PointerType::getUnqual(FunctionType::get(
                      Builder.getVoidTy(), arg_type, false)),
                      arg_type};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    P = Function::Create(Ty, Linkage, Name, M);
  }

  auto old_fn = cast<CallInst>(new_F->user_back());
  auto arg_val = old_fn->getArgOperand(0);

  Value *Args[] = {new_F, arg_val};

  Builder.SetInsertPoint(cast<CallInst>(old_fn));
  Builder.CreateCall(P, Args);

  old_fn->eraseFromParent();
  
  new_F->addFnAttr(PollySkipFnAttr);

  return;
}


static bool CodeGen(Scop &S, IslAstInfo &AI, LoopInfo &LI, DominatorTree &DT,
                    ScalarEvolution &SE, RegionInfo &RI) {
  // Check whether IslAstInfo uses the same isl_ctx. Since -polly-codegen
  // reports itself to preserve DependenceInfo and IslAstInfo, we might get
  // those analysis that were computed by a different ScopInfo for a different
  // Scop structure. When the ScopInfo/Scop object is freed, there is a high
  // probability that the new ScopInfo/Scop object will be created at the same
  // heap position with the same address. Comparing whether the Scop or ScopInfo
  // address is the expected therefore is unreliable.
  // Instead, we compare the address of the isl_ctx object. Both, DependenceInfo
  // and IslAstInfo must hold a reference to the isl_ctx object to ensure it is
  // not freed before the destruction of those analyses which might happen after
  // the destruction of the Scop/ScopInfo they refer to.  Hence, the isl_ctx
  // will not be freed and its space not reused as long there is a
  // DependenceInfo or IslAstInfo around.

  errs() << "************* \n Enter codegen \n *************" << "\n";

  IslAst &Ast = AI.getIslAst();
  if (Ast.getSharedIslCtx() != S.getSharedIslCtx()) {
    LLVM_DEBUG(dbgs() << "Got an IstAst for a different Scop/isl_ctx\n");
    return false;
  }

  // Check if we created an isl_ast root node, otherwise exit.
  isl::ast_node AstRoot = Ast.getAst();
  if (AstRoot.is_null())
    return false;

  // Collect statistics. Do it before we modify the IR to avoid having it any
  // influence on the result.
  auto ScopStats = S.getStatistics();
  ScopsProcessed++;

  auto &DL = S.getFunction().getParent()->getDataLayout();
  Region *R = &S.getRegion();
  assert(!R->isTopLevelRegion() && "Top level regions are not supported");

  ScopAnnotator Annotator;

  simplifyRegion(R, &DT, &LI, &RI);
  assert(R->isSimple());
  BasicBlock *EnteringBB = S.getEnteringBlock();
  assert(EnteringBB);
  PollyIRBuilder Builder(EnteringBB->getContext(), ConstantFolder(),
                         IRInserter(Annotator));
  Builder.SetInsertPoint(EnteringBB->getTerminator());

  // Only build the run-time condition and parameters _after_ having
  // introduced the conditional branch. This is important as the conditional
  // branch will guard the original scop from new induction variables that
  // the SCEVExpander may introduce while code generating the parameters and
  // which may introduce scalar dependences that prevent us from correctly
  // code generating this scop.
  BBPair StartExitBlocks =
      std::get<0>(executeScopConditionally(S, Builder.getTrue(), DT, RI, LI));
  BasicBlock *StartBlock = std::get<0>(StartExitBlocks);
  BasicBlock *ExitBlock = std::get<1>(StartExitBlocks);


  removeLifetimeMarkers(R);
  auto *SplitBlock = StartBlock->getSinglePredecessor();

  IslNodeBuilder NodeBuilder(Builder, Annotator, DL, LI, SE, DT, S, StartBlock);

  // All arrays must have their base pointers known before
  // ScopAnnotator::buildAliasScopes.
  NodeBuilder.allocateNewArrays(StartExitBlocks);
  Annotator.buildAliasScopes(S);

  if (PerfMonitoring) {
    PerfMonitor P(S, EnteringBB->getParent()->getParent());
    P.initialize();
    P.insertRegionStart(SplitBlock->getTerminator());

    BasicBlock *MergeBlock = ExitBlock->getUniqueSuccessor();
    P.insertRegionEnd(MergeBlock->getTerminator());
  }

  // First generate code for the hoisted invariant loads and transitively the
  // parameters they reference. Afterwards, for the remaining parameters that
  // might reference the hoisted loads. Finally, build the runtime check
  // that might reference both hoisted loads as well as parameters.
  // If the hoisting fails we have to bail and execute the original code.
  Builder.SetInsertPoint(SplitBlock->getTerminator());
  if (!NodeBuilder.preloadInvariantLoads()) {
    // Patch the introduced branch condition to ensure that we always execute
    // the original SCoP.
    auto *FalseI1 = Builder.getFalse();
    auto *SplitBBTerm = Builder.GetInsertBlock()->getTerminator();
    SplitBBTerm->setOperand(0, FalseI1);

    // Since the other branch is hence ignored we mark it as unreachable and
    // adjust the dominator tree accordingly.
    auto *ExitingBlock = StartBlock->getUniqueSuccessor();
    assert(ExitingBlock);
    auto *MergeBlock = ExitingBlock->getUniqueSuccessor();
    assert(MergeBlock);
    markBlockUnreachable(*StartBlock, Builder);
    markBlockUnreachable(*ExitingBlock, Builder);
    auto *ExitingBB = S.getExitingBlock();
    assert(ExitingBB);
    DT.changeImmediateDominator(MergeBlock, ExitingBB);
    DT.eraseNode(ExitingBlock);
  } else {
    NodeBuilder.addParameters(S.getContext().release());
    Value *RTC = NodeBuilder.createRTC(AI.getRunCondition().release());

    Builder.GetInsertBlock()->getTerminator()->setOperand(0, RTC);

    // Explicitly set the insert point to the end of the block to avoid that a
    // split at the builder's current
    // insert position would move the malloc calls to the wrong BasicBlock.
    // Ideally we would just split the block during allocation of the new
    // arrays, but this would break the assumption that there are no blocks
    // between polly.start and polly.exiting (at this point).

    Builder.SetInsertPoint(StartBlock->getTerminator());

    NodeBuilder.create(AstRoot.release());

    NodeBuilder.finalize();
    
    fixRegionInfo(*EnteringBB->getParent(), *R->getParent(), RI);
    errs() << "EXIT\n";

    CodegenedScops++;
    CodegenedAffineLoops += ScopStats.NumAffineLoops;
    CodegenedBoxedLoops += ScopStats.NumBoxedLoops;
  }

  Function *F = EnteringBB->getParent();
  // pipeline (*)
  if(PipelineCodeGen == "yes")
    extractPipeline(Builder, F);

  // //
  verifyGeneratedFunction(S, *F, AI);
  for (auto *SubF : NodeBuilder.getParallelSubfunctions())
    verifyGeneratedFunction(S, *SubF, AI);

  // Mark the function such that we run additional cleanup passes on this
  // function (e.g. mem2reg to rediscover phi nodes).
  F->addFnAttr("polly-optimized");

  errs() << "************************************ END ******************************** \n";

  return true;
}

namespace {

class CodeGeneration : public ScopPass {
public:
  static char ID;

  /// The data layout used.
  const DataLayout *DL;

  /// @name The analysis passes we need to generate code.
  ///
  ///{
  LoopInfo *LI;
  IslAstInfo *AI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  RegionInfo *RI;
  ///}

  CodeGeneration() : ScopPass(ID) {}

  /// Generate LLVM-IR for the SCoP @p S.
  bool runOnScop(Scop &S) override {
    // Skip SCoPs in case they're already code-generated by PPCGCodeGeneration.
    if (S.isToBeSkipped())
      return false;

    AI = &getAnalysis<IslAstInfoWrapperPass>().getAI();
    LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
    DL = &S.getFunction().getParent()->getDataLayout();
    RI = &getAnalysis<RegionInfoPass>().getRegionInfo();
    return CodeGen(S, *AI, *LI, *DT, *SE, *RI);
  }

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ScopPass::getAnalysisUsage(AU);

    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<IslAstInfoWrapperPass>();
    AU.addRequired<RegionInfoPass>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<ScopDetectionWrapperPass>();
    AU.addRequired<ScopInfoRegionPass>();
    AU.addRequired<LoopInfoWrapperPass>();

    AU.addPreserved<DependenceInfo>();
    AU.addPreserved<IslAstInfoWrapperPass>();

    // FIXME: We do not yet add regions for the newly generated code to the
    //        region tree.
  }
};
} // namespace

PreservedAnalyses CodeGenerationPass::run(Scop &S, ScopAnalysisManager &SAM,
                                          ScopStandardAnalysisResults &AR,
                                          SPMUpdater &U) {
  auto &AI = SAM.getResult<IslAstAnalysis>(S, AR);
  if (CodeGen(S, AI, AR.LI, AR.DT, AR.SE, AR.RI)) {
    U.invalidateScop(S);
    return PreservedAnalyses::none();
  }

  return PreservedAnalyses::all();
}

char CodeGeneration::ID = 1;

Pass *polly::createCodeGenerationPass() { return new CodeGeneration(); }

INITIALIZE_PASS_BEGIN(CodeGeneration, "polly-codegen",
                      "Polly - Create LLVM-IR from SCoPs", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetectionWrapperPass);
INITIALIZE_PASS_END(CodeGeneration, "polly-codegen",
                    "Polly - Create LLVM-IR from SCoPs", false, false)
