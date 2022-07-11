//===- Schedule.cpp - Calculate an optimized schedule ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass generates an entirely new schedule tree from the data dependences
// and iteration domains. The new schedule tree is computed in two steps:
//
// 1) The isl scheduling optimizer is run
//
// The isl scheduling optimizer creates a new schedule tree that maximizes
// parallelism and tileability and minimizes data-dependence distances. The
// algorithm used is a modified version of the ``Pluto'' algorithm:
//
//   U. Bondhugula, A. Hartono, J. Ramanujam, and P. Sadayappan.
//   A Practical Automatic Polyhedral Parallelizer and Locality Optimizer.
//   In Proceedings of the 2008 ACM SIGPLAN Conference On Programming Language
//   Design and Implementation, PLDI ’08, pages 101–113. ACM, 2008.
//
// 2) A set of post-scheduling transformations is applied on the schedule tree.
//
// These optimizations include:
//
//  - Tiling of the innermost tilable bands
//  - Prevectorization - The choice of a possible outer loop that is strip-mined
//                       to the innermost level to enable inner-loop
//                       vectorization.
//  - Some optimizations for spatial locality are also planned.
//
// For a detailed description of the schedule tree itself please see section 6
// of:
//
// Polyhedral AST generation is more than scanning polyhedra
// Tobias Grosser, Sven Verdoolaege, Albert Cohen
// ACM Transactions on Programming Languages and Systems (TOPLAS),
// 37(4), July 2015
// http://www.grosser.es/#pub-polyhedral-AST-generation
//
// This publication also contains a detailed discussion of the different options
// for polyhedral loop unrolling, full/partial tile separation and other uses
// of the schedule tree.
//
//===----------------------------------------------------------------------===//

#include "polly/ScheduleOptimizer.h"
#include "polly/CodeGen/CodeGeneration.h"
#include "polly/DependenceInfo.h"
#include "polly/ManualOptimizer.h"
#include "polly/MatmulOptimizer.h"
#include "polly/Options.h"
#include "polly/ScheduleTreeTransform.h"
#include "polly/Support/ISLOStream.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "isl/options.h"
// pipeline (*) may be removed
#include "isl/schedule.h"

using namespace llvm;
using namespace polly;

namespace llvm {
class Loop;
class Module;
} // namespace llvm

#define DEBUG_TYPE "polly-opt-isl"

static cl::opt<std::string>
    OptimizeDeps("polly-opt-optimize-only",
                 cl::desc("Only a certain kind of dependences (all/raw)"),
                 cl::Hidden, cl::init("all"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

static cl::opt<std::string>
    SimplifyDeps("polly-opt-simplify-deps",
                 cl::desc("Dependences should be simplified (yes/no)"),
                 cl::Hidden, cl::init("yes"), cl::ZeroOrMore,
                 cl::cat(PollyCategory));

// pipeline (*)
static cl::opt<std::string>
    PipelineLoops("polly-pipeline-loops",
                  cl::desc("Detecting pipeline pattern between iterations of "
                           "different loops (yes/no)"),
                  cl::Hidden, cl::init("no"), cl::ZeroOrMore,
                  cl::cat(PollyCategory));

static cl::opt<int> MaxConstantTerm(
    "polly-opt-max-constant-term",
    cl::desc("The maximal constant term allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> MaxCoefficient(
    "polly-opt-max-coefficient",
    cl::desc("The maximal coefficient allowed (-1 is unlimited)"), cl::Hidden,
    cl::init(20), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> FusionStrategy(
    "polly-opt-fusion", cl::desc("The fusion strategy to choose (min/max)"),
    cl::Hidden, cl::init("min"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string>
    MaximizeBandDepth("polly-opt-maximize-bands",
                      cl::desc("Maximize the band depth (yes/no)"), cl::Hidden,
                      cl::init("yes"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<std::string> OuterCoincidence(
    "polly-opt-outer-coincidence",
    cl::desc("Try to construct schedules where the outer member of each band "
             "satisfies the coincidence constraints (yes/no)"),
    cl::Hidden, cl::init("no"), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> PrevectorWidth(
    "polly-prevect-width",
    cl::desc(
        "The number of loop iterations to strip-mine for pre-vectorization"),
    cl::Hidden, cl::init(4), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> FirstLevelTiling("polly-tiling",
                                      cl::desc("Enable loop tiling"),
                                      cl::init(true), cl::ZeroOrMore,
                                      cl::cat(PollyCategory));

static cl::opt<int> FirstLevelDefaultTileSize(
    "polly-default-tile-size",
    cl::desc("The default tile size (if not enough were provided by"
             " --polly-tile-sizes)"),
    cl::Hidden, cl::init(32), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    FirstLevelTileSizes("polly-tile-sizes",
                        cl::desc("A tile size for each loop dimension, filled "
                                 "with --polly-default-tile-size"),
                        cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                        cl::cat(PollyCategory));

static cl::opt<bool>
    SecondLevelTiling("polly-2nd-level-tiling",
                      cl::desc("Enable a 2nd level loop of loop tiling"),
                      cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> SecondLevelDefaultTileSize(
    "polly-2nd-level-default-tile-size",
    cl::desc("The default 2nd-level tile size (if not enough were provided by"
             " --polly-2nd-level-tile-sizes)"),
    cl::Hidden, cl::init(16), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    SecondLevelTileSizes("polly-2nd-level-tile-sizes",
                         cl::desc("A tile size for each loop dimension, filled "
                                  "with --polly-default-tile-size"),
                         cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                         cl::cat(PollyCategory));

static cl::opt<bool> RegisterTiling("polly-register-tiling",
                                    cl::desc("Enable register tiling"),
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::cat(PollyCategory));

static cl::opt<int> RegisterDefaultTileSize(
    "polly-register-tiling-default-tile-size",
    cl::desc("The default register tile size (if not enough were provided by"
             " --polly-register-tile-sizes)"),
    cl::Hidden, cl::init(2), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::list<int>
    RegisterTileSizes("polly-register-tile-sizes",
                      cl::desc("A tile size for each loop dimension, filled "
                               "with --polly-register-tile-size"),
                      cl::Hidden, cl::ZeroOrMore, cl::CommaSeparated,
                      cl::cat(PollyCategory));

static cl::opt<bool> PragmaBasedOpts(
    "polly-pragma-based-opts",
    cl::desc("Apply user-directed transformation from metadata"),
    cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedOpts("polly-pattern-matching-based-opts",
                cl::desc("Perform optimizations based on pattern matching"),
                cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> OptimizedScops(
    "polly-optimized-scops",
    cl::desc("Polly - Dump polyhedral description of Scops optimized with "
             "the isl scheduling optimizer and the set of post-scheduling "
             "transformations is applied on the schedule tree"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

STATISTIC(ScopsProcessed, "Number of scops processed");
STATISTIC(ScopsRescheduled, "Number of scops rescheduled");
STATISTIC(ScopsOptimized, "Number of scops optimized");

STATISTIC(NumAffineLoopsOptimized, "Number of affine loops optimized");
STATISTIC(NumBoxedLoopsOptimized, "Number of boxed loops optimized");

#define THREE_STATISTICS(VARNAME, DESC)                                        \
  static Statistic VARNAME[3] = {                                              \
      {DEBUG_TYPE, #VARNAME "0", DESC " (original)"},                          \
      {DEBUG_TYPE, #VARNAME "1", DESC " (after scheduler)"},                   \
      {DEBUG_TYPE, #VARNAME "2", DESC " (after optimizer)"}}

THREE_STATISTICS(NumBands, "Number of bands");
THREE_STATISTICS(NumBandMembers, "Number of band members");
THREE_STATISTICS(NumCoincident, "Number of coincident band members");
THREE_STATISTICS(NumPermutable, "Number of permutable bands");
THREE_STATISTICS(NumFilters, "Number of filter nodes");
THREE_STATISTICS(NumExtension, "Number of extension nodes");

STATISTIC(FirstLevelTileOpts, "Number of first level tiling applied");
STATISTIC(SecondLevelTileOpts, "Number of second level tiling applied");
STATISTIC(RegisterTileOpts, "Number of register tiling applied");
STATISTIC(PrevectOpts, "Number of strip-mining for prevectorization applied");
STATISTIC(MatMulOpts,
          "Number of matrix multiplication patterns detected and optimized");

namespace {
/// Additional parameters of the schedule optimizer.
///
/// Target Transform Info and the SCoP dependencies used by the schedule
/// optimizer.
struct OptimizerAdditionalInfoTy {
  const llvm::TargetTransformInfo *TTI;
  const Dependences *D;
};

class ScheduleTreeOptimizer {
public:
  /// Apply schedule tree transformations.
  ///
  /// This function takes an (possibly already optimized) schedule tree and
  /// applies a set of additional optimizations on the schedule tree. The
  /// transformations applied include:
  ///
  ///   - Tiling
  ///   - Prevectorization
  ///
  /// @param Schedule The schedule object the transformations will be applied
  ///                 to.
  /// @param OAI      Target Transform Info and the SCoP dependencies.
  /// @returns        The transformed schedule.
  static isl::schedule
  optimizeSchedule(isl::schedule Schedule,
                   const OptimizerAdditionalInfoTy *OAI = nullptr);

  /// Apply schedule tree transformations.
  ///
  /// This function takes a node in an (possibly already optimized) schedule
  /// tree and applies a set of additional optimizations on this schedule tree
  /// node and its descendants. The transformations applied include:
  ///
  ///   - Tiling
  ///   - Prevectorization
  ///
  /// @param Node The schedule object post-transformations will be applied to.
  /// @param OAI  Target Transform Info and the SCoP dependencies.
  /// @returns    The transformed schedule.
  static isl::schedule_node
  optimizeScheduleNode(isl::schedule_node Node,
                       const OptimizerAdditionalInfoTy *OAI = nullptr);

  /// Decide if the @p NewSchedule is profitable for @p S.
  ///
  /// @param S           The SCoP we optimize.
  /// @param NewSchedule The new schedule we computed.
  ///
  /// @return True, if we believe @p NewSchedule is an improvement for @p S.
  static bool isProfitableSchedule(polly::Scop &S, isl::schedule NewSchedule);

  /// Isolate a set of partial tile prefixes.
  ///
  /// This set should ensure that it contains only partial tile prefixes that
  /// have exactly VectorWidth iterations.
  ///
  /// @param Node A schedule node band, which is a parent of a band node,
  ///             that contains a vector loop.
  /// @return Modified isl_schedule_node.
  static isl::schedule_node isolateFullPartialTiles(isl::schedule_node Node,
                                                    int VectorWidth);

private:
  /// Check if this node is a band node we want to tile.
  ///
  /// We look for innermost band nodes where individual dimensions are marked as
  /// permutable.
  ///
  /// @param Node The node to check.
  static bool isTileableBandNode(isl::schedule_node Node);

  /// Pre-vectorizes one scheduling dimension of a schedule band.
  ///
  /// prevectSchedBand splits out the dimension DimToVectorize, tiles it and
  /// sinks the resulting point loop.
  ///
  /// Example (DimToVectorize=0, VectorWidth=4):
  ///
  /// | Before transformation:
  /// |
  /// | A[i,j] -> [i,j]
  /// |
  /// | for (i = 0; i < 128; i++)
  /// |    for (j = 0; j < 128; j++)
  /// |      A(i,j);
  ///
  /// | After transformation:
  /// |
  /// | for (it = 0; it < 32; it+=1)
  /// |    for (j = 0; j < 128; j++)
  /// |      for (ip = 0; ip <= 3; ip++)
  /// |        A(4 * it + ip,j);
  ///
  /// The goal of this transformation is to create a trivially vectorizable
  /// loop.  This means a parallel loop at the innermost level that has a
  /// constant number of iterations corresponding to the target vector width.
  ///
  /// This transformation creates a loop at the innermost level. The loop has
  /// a constant number of iterations, if the number of loop iterations at
  /// DimToVectorize can be divided by VectorWidth. The default VectorWidth is
  /// currently constant and not yet target specific. This function does not
  /// reason about parallelism.
  static isl::schedule_node prevectSchedBand(isl::schedule_node Node,
                                             unsigned DimToVectorize,
                                             int VectorWidth);

  /// Apply additional optimizations on the bands in the schedule tree.
  ///
  /// We are looking for an innermost band node and apply the following
  /// transformations:
  ///
  ///  - Tile the band
  ///      - if the band is tileable
  ///      - if the band has more than one loop dimension
  ///
  ///  - Prevectorize the schedule of the band (or the point loop in case of
  ///    tiling).
  ///      - if vectorization is enabled
  ///
  /// @param Node The schedule node to (possibly) optimize.
  /// @param User A pointer to forward some use information
  ///        (currently unused).
  static isl_schedule_node *optimizeBand(isl_schedule_node *Node, void *User);

  /// Apply additional optimizations on the bands in the schedule tree.
  ///
  /// We apply the following
  /// transformations:
  ///
  ///  - Tile the band
  ///  - Prevectorize the schedule of the band (or the point loop in case of
  ///    tiling).
  ///      - if vectorization is enabled
  ///
  /// @param Node The schedule node to (possibly) optimize.
  /// @param User A pointer to forward some use information
  ///        (currently unused).
  static isl::schedule_node standardBandOpts(isl::schedule_node Node,
                                             void *User);
};

isl::schedule_node
ScheduleTreeOptimizer::isolateFullPartialTiles(isl::schedule_node Node,
                                               int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  Node = Node.child(0).child(0);
  isl::union_map SchedRelUMap = Node.get_prefix_schedule_relation();
  isl::union_set ScheduleRangeUSet = SchedRelUMap.range();
  isl::set ScheduleRange{ScheduleRangeUSet};
  isl::set IsolateDomain = getPartialTilePrefixes(ScheduleRange, VectorWidth);
  auto AtomicOption = getDimOptions(IsolateDomain.get_ctx(), "atomic");
  isl::union_set IsolateOption = getIsolateOptions(IsolateDomain, 1);
  Node = Node.parent().parent();
  isl::union_set Options = IsolateOption.unite(AtomicOption);
  Node = Node.band_set_ast_build_options(Options);
  return Node;
}

isl::schedule_node ScheduleTreeOptimizer::prevectSchedBand(
    isl::schedule_node Node, unsigned DimToVectorize, int VectorWidth) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  isl_size ScheduleDimensions = Space.dim(isl::dim::set);
  assert((isl_size)DimToVectorize < ScheduleDimensions);

  if (DimToVectorize > 0) {
    Node = isl::manage(
        isl_schedule_node_band_split(Node.release(), DimToVectorize));
    Node = Node.child(0);
  }
  if ((isl_size)DimToVectorize < ScheduleDimensions - 1)
    Node = isl::manage(isl_schedule_node_band_split(Node.release(), 1));
  Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Sizes = isl::multi_val::zero(Space);
  Sizes = Sizes.set_val(0, isl::val(Node.get_ctx(), VectorWidth));
  Node =
      isl::manage(isl_schedule_node_band_tile(Node.release(), Sizes.release()));
  Node = isolateFullPartialTiles(Node, VectorWidth);
  Node = Node.child(0);
  // Make sure the "trivially vectorizable loop" is not unrolled. Otherwise,
  // we will have troubles to match it in the backend.
  Node = Node.band_set_ast_build_options(
      isl::union_set(Node.get_ctx(), "{ unroll[x]: 1 = 0 }"));
  Node = isl::manage(isl_schedule_node_band_sink(Node.release()));
  Node = Node.child(0);
  if (isl_schedule_node_get_type(Node.get()) == isl_schedule_node_leaf)
    Node = Node.parent();
  auto LoopMarker = isl::id::alloc(Node.get_ctx(), "SIMD", nullptr);
  PrevectOpts++;
  return Node.insert_mark(LoopMarker);
}

static bool isSimpleInnermostBand(const isl::schedule_node &Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  assert(isl_schedule_node_n_children(Node.get()) == 1);

  auto ChildType = isl_schedule_node_get_type(Node.child(0).get());

  if (ChildType == isl_schedule_node_leaf)
    return true;

  if (ChildType != isl_schedule_node_sequence)
    return false;

  auto Sequence = Node.child(0);

  for (int c = 0, nc = isl_schedule_node_n_children(Sequence.get()); c < nc;
       ++c) {
    auto Child = Sequence.child(c);
    if (isl_schedule_node_get_type(Child.get()) != isl_schedule_node_filter)
      return false;
    if (isl_schedule_node_get_type(Child.child(0).get()) !=
        isl_schedule_node_leaf)
      return false;
  }
  return true;
}

bool ScheduleTreeOptimizer::isTileableBandNode(isl::schedule_node Node) {
  if (isl_schedule_node_get_type(Node.get()) != isl_schedule_node_band)
    return false;

  if (isl_schedule_node_n_children(Node.get()) != 1)
    return false;

  if (!isl_schedule_node_band_get_permutable(Node.get()))
    return false;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);

  if (Dims <= 1)
    return false;

  return isSimpleInnermostBand(Node);
}

__isl_give isl::schedule_node
ScheduleTreeOptimizer::standardBandOpts(isl::schedule_node Node, void *User) {
  if (FirstLevelTiling) {
    Node = tileNode(Node, "1st level tiling", FirstLevelTileSizes,
                    FirstLevelDefaultTileSize);
    FirstLevelTileOpts++;
  }

  if (SecondLevelTiling) {
    Node = tileNode(Node, "2nd level tiling", SecondLevelTileSizes,
                    SecondLevelDefaultTileSize);
    SecondLevelTileOpts++;
  }

  if (RegisterTiling) {
    Node =
        applyRegisterTiling(Node, RegisterTileSizes, RegisterDefaultTileSize);
    RegisterTileOpts++;
  }

  if (PollyVectorizerChoice == VECTORIZER_NONE)
    return Node;

  auto Space = isl::manage(isl_schedule_node_band_get_space(Node.get()));
  auto Dims = Space.dim(isl::dim::set);

  for (int i = Dims - 1; i >= 0; i--)
    if (Node.band_member_get_coincident(i)) {
      Node = prevectSchedBand(Node, i, PrevectorWidth);
      break;
    }

  return Node;
}

__isl_give isl_schedule_node *
ScheduleTreeOptimizer::optimizeBand(__isl_take isl_schedule_node *Node,
                                    void *User) {
  if (!isTileableBandNode(isl::manage_copy(Node)))
    return Node;

  const OptimizerAdditionalInfoTy *OAI =
      static_cast<const OptimizerAdditionalInfoTy *>(User);

  if (PMBasedOpts && User) {
    if (isl::schedule_node PatternOptimizedSchedule = tryOptimizeMatMulPattern(
            isl::manage_copy(Node), OAI->TTI, OAI->D)) {
      MatMulOpts++;
      isl_schedule_node_free(Node);
      return PatternOptimizedSchedule.release();
    }
  }

  return standardBandOpts(isl::manage(Node), User).release();
}

isl::schedule
ScheduleTreeOptimizer::optimizeSchedule(isl::schedule Schedule,
                                        const OptimizerAdditionalInfoTy *OAI) {
  auto Root = Schedule.get_root();
  Root = optimizeScheduleNode(Root, OAI);
  return Root.get_schedule();
}

isl::schedule_node ScheduleTreeOptimizer::optimizeScheduleNode(
    isl::schedule_node Node, const OptimizerAdditionalInfoTy *OAI) {
  Node = isl::manage(isl_schedule_node_map_descendant_bottom_up(
      Node.release(), optimizeBand,
      const_cast<void *>(static_cast<const void *>(OAI))));
  return Node;
}

bool ScheduleTreeOptimizer::isProfitableSchedule(Scop &S,
                                                 isl::schedule NewSchedule) {
  // To understand if the schedule has been optimized we check if the schedule
  // has changed at all.
  // TODO: We can improve this by tracking if any necessarily beneficial
  // transformations have been performed. This can e.g. be tiling, loop
  // interchange, or ...) We can track this either at the place where the
  // transformation has been performed or, in case of automatic ILP based
  // optimizations, by comparing (yet to be defined) performance metrics
  // before/after the scheduling optimizer
  // (e.g., #stride-one accesses)
  auto NewScheduleMap = NewSchedule.get_map();
  auto OldSchedule = S.getSchedule();
  assert(OldSchedule && "Only IslScheduleOptimizer can insert extension nodes "
                        "that make Scop::getSchedule() return nullptr.");
  bool changed = !OldSchedule.is_equal(NewScheduleMap);
  return changed;
}

class IslScheduleOptimizerWrapperPass : public ScopPass {
public:
  static char ID;

  explicit IslScheduleOptimizerWrapperPass() : ScopPass(ID) {}

  /// Optimize the schedule of the SCoP @p S.
  bool runOnScop(Scop &S) override;

  /// Print the new schedule for the SCoP @p S.
  void printScop(raw_ostream &OS, Scop &S) const override;

  /// Register all analyses and transformation required.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Release the internal memory.
  void releaseMemory() override {
    LastSchedule = {};
    IslCtx.reset();
  }

private:
  std::shared_ptr<isl_ctx> IslCtx;
  isl::schedule LastSchedule;
};

char IslScheduleOptimizerWrapperPass::ID = 0;

#ifndef NDEBUG
static void printSchedule(llvm::raw_ostream &OS, const isl::schedule &Schedule,
                          StringRef Desc) {
  isl::ctx Ctx = Schedule.get_ctx();
  isl_printer *P = isl_printer_to_str(Ctx.get());
  P = isl_printer_set_yaml_style(P, ISL_YAML_STYLE_BLOCK);
  P = isl_printer_print_schedule(P, Schedule.get());
  char *Str = isl_printer_get_str(P);
  OS << Desc << ": \n" << Str << "\n";
  free(Str);
  isl_printer_free(P);
}
#endif

/// Collect statistics for the schedule tree.
///
/// @param Schedule The schedule tree to analyze. If not a schedule tree it is
/// ignored.
/// @param Version  The version of the schedule tree that is analyzed.
///                 0 for the original schedule tree before any transformation.
///                 1 for the schedule tree after isl's rescheduling.
///                 2 for the schedule tree after optimizations are applied
///                 (tiling, pattern matching)
static void walkScheduleTreeForStatistics(isl::schedule Schedule, int Version) {
  auto Root = Schedule.get_root();
  if (!Root)
    return;

  isl_schedule_node_foreach_descendant_top_down(
      Root.get(),
      [](__isl_keep isl_schedule_node *nodeptr, void *user) -> isl_bool {
        isl::schedule_node Node = isl::manage_copy(nodeptr);
        int Version = *static_cast<int *>(user);

        switch (isl_schedule_node_get_type(Node.get())) {
        case isl_schedule_node_band: {
          NumBands[Version]++;
          if (isl_schedule_node_band_get_permutable(Node.get()) ==
              isl_bool_true)
            NumPermutable[Version]++;

          int CountMembers = isl_schedule_node_band_n_member(Node.get());
          NumBandMembers[Version] += CountMembers;
          for (int i = 0; i < CountMembers; i += 1) {
            if (Node.band_member_get_coincident(i))
              NumCoincident[Version]++;
          }
          break;
        }

        case isl_schedule_node_filter:
          NumFilters[Version]++;
          break;

        case isl_schedule_node_extension:
          NumExtension[Version]++;
          break;

        default:
          break;
        }

        return isl_bool_true;
      },
      &Version);
}

// pipeline (*)

// TODO: remove this function.
isl::map get_pipeline_relation(isl::map rmap, isl::map wmap) {
  errs() << "***"
         << "\n";
  errs() << "Enter get_pipeline_relation\n";

  // isl_ctx * ctx = rmap.get_ctx().get();
  // auto map_read = rmap.copy();
  //  auto map_write = wmap.copy();

  auto write_reverse = wmap.reverse();
  auto K = rmap.apply_range(write_reverse);
  auto K_temp = K.lexmax();
  auto D = K.domain();
  auto D1 = D.lex_ge_set(D);
  auto L1 = D1.apply_range(K_temp);
  auto L = L1.lexmax();
  auto L_reverse = L.reverse();
  auto T = L_reverse.lexmax();

  return T;
}

// TODO: remove this function.
isl::map get_blocks(isl::set s, isl::map m, int choose) {
  //  isl_set *s_copy = isl_set_copy(s);
  //  isl_map *m_copy = isl_map_copy(m);

  isl::set D;
  if (choose == 0)
    D = m.domain();
  else if (choose == 1)
    D = m.range();

  auto Dp1 = D.lex_ge_set(s);
  auto Dp = Dp1.reverse();
  auto E = Dp.lexmin();
  auto Ed = E.domain();

  // if the whole domain is not covered.
  if (s.is_equal(Ed) == 0) {
    auto last_elem = s.lexmax();
    auto rem_elems = s.subtract(Ed);
    auto comp_E = isl::map::from_domain_and_range(rem_elems, last_elem);
    E = E.unite(comp_E);
  }

  return E;
}

// TODO: to be removed
struct dummys {
  int in_out;
  isl_pw_multi_aff *d;
};

// TODO: replace this in the file
typedef struct depends
{
  int is_source_only = 0;
  int out_index;
  int *in_index;
  int num_in_index;
  isl_pw_multi_aff *out_pw; //writes
  isl_pw_multi_aff_list *in_pw; //reads
} depends;


// Map T is between source iterations and target iterations.
// It is interpreted as: having I[i,j]->J[i',j'] in T, where
// i' and j' are functions of i and j, after the iteration [i,j]
// of the source, we can run up to [i',j'] of target.
// What we do in this function is to map I[i,j] and J[i',j']
// to an "imaginary" array [i,j]. Then in the code generation
// using IN and OUT dependencies, we can syncronize blocks.
// TODO: rest of iterations, kind of an early exit.

isl::id get_depend_pw(isl::map T, isl::set I, isl::set J, int choose) {
  errs() << "Enter computing pw\n";
  T = T.coalesce();

  // the lexmax is for considering the last block and all.
  isl::map Er = (T.domain().unite(I.lexmax())).flatten_map().coalesce();

  errs() << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
            "$$$$$$$$$$$$$$$\n";
  T.dump();

  auto ctx = T.get_ctx();
  isl::val valm1(ctx, -1);

  isl::set Erdom = Er.domain();
  isl::set ErmI1 = I.subtract(Erdom);              // uncovered
  isl::set ErmI = ErmI1.unite(ErmI1.complement()); // drop all constraints

  auto dim_count = ErmI1.dim(isl::dim::set);

  for (int dim = 0; dim < dim_count; dim++)
    ErmI = ErmI.fix_val(isl::dim::set, dim, valm1);

  struct dummys *dummy = (struct dummys *)malloc(sizeof(struct dummys));
  auto new_id = isl::id::alloc(ctx, "task", dummy);

  if (choose == 0) {
    isl::map final = isl::map::from_domain_and_range(ErmI1, ErmI);
    final = final.unite(Er);
    Er = Er.set_tuple_name(isl::dim::out, "D");
    isl::pw_multi_aff pw2 = isl::pw_multi_aff::from_map(final);
    dummy->d = pw2.release();
    dummy->in_out = 0;
    return new_id;
  }

  if (choose == 1) {
    isl::map Tr = T.reverse();
    isl::set Tdom = Tr.domain();
    isl::set TmJ1 = J.subtract(Tdom); // uncovered
    isl::map final =
        isl::map::from_domain_and_range(TmJ1, ErmI); // use I for complement
    final = final.unite(Tr);
    T = T.set_tuple_name(isl::dim::in, "D");
    isl::pw_multi_aff pw1 = isl::pw_multi_aff::from_map(final);
    dummy->d = pw1.release();
    dummy->in_out = 1;
    return new_id;
  }

  // add an error else
}

isl::schedule get_pipeline_schedule_per_loop_nest(isl::map E, isl::id new_id) {

  errs() << "Enter scheduling\n\n";

  isl::ctx ctx = E.get_ctx();
  // auto new_id = isl::id::alloc(ctx, "task", &Epw);
  isl::id new_id_begin(ctx, "taskBegin");

  // auto pw = (struct dummys*)new_id.get_user();
  // errs() << "from scheduling\n" << pw[0].n <<"\n";
  // isl_pw_multi_aff_dump(pw[0].d);
  E = E.coalesce();
  isl::set rE = E.range();
  isl::set dE = E.domain();

  isl::union_set rE_uset(rE);
  isl::union_set dE_uset(dE);

  errs() << "Enter coalesce \n";
  isl::map rE_map = rE.flatten_map().coalesce();
  errs() << "first\n";
  isl::map dE_map = dE.flatten_map().coalesce();
  errs() << "second\n";

  isl::union_map E_umap(E);
  isl::union_map rE_umap(rE_map);
  isl::union_map dE_umap(dE_map);

  errs() << "finish prepration\n";

  // auto ps = isl::multi_union_pw_aff::from_union_map(E_umap);
  // errs() << "first\n";
  auto ps1 = isl::multi_union_pw_aff::from_union_map(dE_umap);
  errs() << "first\n";
  auto ps2 = isl::multi_union_pw_aff::from_union_map(rE_umap);

  auto r_sch_node = isl::schedule_node::from_domain(rE_uset);
  isl::schedule_node r_sch_node_child = r_sch_node.child(0);

  auto r_h_node = r_sch_node_child.insert_partial_schedule(ps2);
  isl::union_set options(
      isl::set(ctx, "{ atomic[i0] : 0 <= i0 <= 1 }")); // TODO: generalize dimensions
  r_h_node = r_h_node.band_set_ast_build_options(options);
  r_h_node = r_h_node.insert_mark(new_id_begin);
  auto r_h = r_h_node.get_schedule();

  auto exp_sch_node_2 = isl::schedule_node::from_domain(dE_uset);
  auto exp_sch_node_child = exp_sch_node_2.child(0);
  auto exp_sch_node_1 = exp_sch_node_child.insert_partial_schedule(ps1);
  auto exp_sch_node = exp_sch_node_1.insert_mark(new_id);
  auto exp_sch = exp_sch_node.get_schedule();

  auto contraction = isl::union_pw_multi_aff(E_umap);

  auto final_sch = r_h.expand(contraction, exp_sch);

  // isl_schedule_copy(r_h.get());
  // errs() << "sch domain  " << final_sch.get_domain();

  //  return isl::manage(final_sch);
  return final_sch;
}

// TODO: remove this function
isl::schedule schedule_pipeline(isl::map mem_acc_rd, isl::map mem_acc_wr,
                                isl::set rd_dom, isl::set wr_dom) {
  errs() << "****"
         << "\n";

  errs() << "Enter schedule_pipeline\n";

  auto rd_dom_map = rd_dom.flatten_map();
  auto wr_dom_map = wr_dom.flatten_map();

  mem_acc_rd = mem_acc_rd.apply_domain(rd_dom_map);
  mem_acc_wr = mem_acc_wr.apply_domain(wr_dom_map);

  isl::map T = get_pipeline_relation(mem_acc_rd, mem_acc_wr);
  isl::map E = get_blocks(wr_dom, T, 0);
  isl::map F = get_blocks(rd_dom, T, 1);

  auto Epw = get_depend_pw(T, wr_dom, rd_dom, 0);
  auto Fpw = get_depend_pw(T, wr_dom, rd_dom, 1);

  isl::schedule E_sch = get_pipeline_schedule_per_loop_nest(E, Epw);
  isl::schedule F_sch = get_pipeline_schedule_per_loop_nest(F, Fpw);

  auto final_map = E_sch.sequence(F_sch);

  return final_map;
}


// TODO: take care of the old version
isl::id get_depend_pw_new(ScopStmt &stmt) 
{

  // to generelize, the number 1 should be equal to the total
  // number of stmts in the scop. get that number from somewhere.
  auto ctx = stmt.getIslCtx(); 
  depends *dep = (depends *)malloc(sizeof(depends));
  dep->num_in_index = stmt.sources_id.size();
  dep->in_pw = isl_pw_multi_aff_list_alloc(ctx.get(),dep->num_in_index);
  dep->in_index = (int*) malloc(sizeof(int) * dep->num_in_index);
  auto new_id = isl::id::alloc(ctx, "task", dep);

  auto I = stmt.getDomain();

  // for considering the last block.
  isl::map Er = (stmt.write_dependency_map).unite(I.lexmax().flatten_map());
  isl::val valm1(ctx, -1);

  errs() << "************MMMMAAAAPPP:\n";
  Er.dump();

  isl::set Erdom = Er.domain();
  isl::set ErmI1 = I.subtract(Erdom);              // uncovered
  isl::set ErmI = ErmI1.unite(ErmI1.complement()); // drop all constraints
  auto dim_count = ErmI1.dim(isl::dim::set);
  for (int dim = 0; dim < dim_count; dim++)
    ErmI = ErmI.fix_val(isl::dim::set, dim, valm1);

  // setup write dependency
  isl::map write_final = isl::map::from_domain_and_range(ErmI1, ErmI);
  write_final = write_final.unite(Er);
  isl::pw_multi_aff pw2 = isl::pw_multi_aff::from_map(write_final);

  dep->out_pw = pw2.release();

  errs() << "finish getting out dependencies\n";
  ///////////////////////////////////

  dep->out_index = stmt.pipeline_id;
  for(int i=0 ; i < dep->num_in_index ; i++)
    dep->in_index[i] = stmt.sources_id[i]; // order should be kept.

  ////////////////////////////////////////
  
  if(dep->num_in_index != 0)
  {
    errs() << "NUMBER:  " << dep->num_in_index << "\n";
    for(int i=0 ; i < dep->num_in_index ; i++)
    {
      errs() << "Enter for " << i << "\n";
      auto JI = stmt.sources_domain[i];
      JI = JI.unite(JI.complement());
      errs() << "source domain\n";
      JI.dump();
      auto dim_count = JI.dim(isl::dim::set);
      for (int dim = 0; dim < dim_count; dim++)
        JI = JI.fix_val(isl::dim::set, dim, valm1);
      
      auto J = stmt.getDomain();
      errs() << "domain\n";
      J.unite(J.complement()).dump();

      isl::map Tr = stmt.read_dependency_maps[i].reverse();
      errs() << "************MMMMAAAAPPP:\n";
      Tr.dump();

      // errs() << "Tr domain\n";
      // Tr.domain().unite(Tr.domain().complement()).dump();

      isl::set Tdom = Tr.domain();
      isl::set TmJ1 = J.subtract(Tdom); // uncovered
      isl::map final = isl::map::from_domain_and_range(TmJ1, JI); // use I for complement
      errs() << "begin unite\n";
      final = final.unite(Tr);
      errs() << "************MMMMAAAAPPP:\n";
      final.dump();


      isl::pw_multi_aff pw1 = isl::pw_multi_aff::from_map(final);
      errs() << "adding to list\n";
      isl_pw_multi_aff_list_add(dep->in_pw, pw1.release());
    }
  }
  else
  {
    dep->is_source_only = 1;
  }
 
  return new_id;
}





// TODO: remove the old function and change the name
isl::schedule schedule_pipeline_new(Scop &S) 
{
  errs() << "****" << "\n";

  errs() << "Enter schedule_pipeline\n";

  // do these for statements.
  // auto ctx = S.getIslCtx(); 
  std::vector<isl::schedule> sch_vec;

  for(auto stmt = S.begin() ; stmt != S.end() ; stmt++)
  {
    errs() << "**************** enter the for loop\n";
    auto new_id = get_depend_pw_new(*stmt);
    errs() << "finish computing id\n";
    isl::schedule sch = get_pipeline_schedule_per_loop_nest(stmt->final_E, new_id);
    sch_vec.push_back(sch);
    errs() << "***************** exit the for loop\n";
  }
  errs() << "Exit loop of computing schedules\n";

  auto final_sch = sch_vec[0];
  for(int i=1 ; i < sch_vec.size() ; i++)
    final_sch = final_sch.sequence(sch_vec[i]);

  return final_sch;
}




static bool runIslScheduleOptimizer(
    Scop &S,
    function_ref<const Dependences &(Dependences::AnalysisLevel)> GetDeps,
    TargetTransformInfo *TTI, isl::schedule &LastSchedule) {

  errs() << "***********************&&&&&&&&&&&&"
         << "\n";

  // Skip SCoPs in case they're already optimised by PPCGCodeGeneration
  if (S.isToBeSkipped())
    return false;

  // Skip empty SCoPs but still allow code generation as it will delete the
  // loops present but not needed.
  if (S.getSize() == 0) {
    S.markAsOptimized();
    return false;
  }

  ScopsProcessed++;

  // Schedule without optimizations.
  isl::schedule Schedule = S.getScheduleTree();
  walkScheduleTreeForStatistics(S.getScheduleTree(), 0);
  LLVM_DEBUG(printSchedule(dbgs(), Schedule, "Original schedule tree"));

  bool HasUserTransformation = false;
  if (PragmaBasedOpts) {
    isl::schedule ManuallyTransformed =
        applyManualTransformations(&S, Schedule);
    if (!ManuallyTransformed) {
      LLVM_DEBUG(dbgs() << "Error during manual optimization\n");
      return false;
    }

    if (ManuallyTransformed.get() != Schedule.get()) {
      // User transformations have precedence over other transformations.
      HasUserTransformation = true;
      Schedule = std::move(ManuallyTransformed);
      LLVM_DEBUG(
          printSchedule(dbgs(), Schedule, "After manual transformations"));
    }
  }

  // Only continue if either manual transformations have been applied or we are
  // allowed to apply heuristics.
  // TODO: Detect disabled heuristics and no user-directed transformation
  // metadata earlier in ScopDetection.
  if (!HasUserTransformation && S.hasDisableHeuristicsHint()) {
    LLVM_DEBUG(dbgs() << "Heuristic optimizations disabled by metadata\n");
    return false;
  }

  // Get dependency analysis.
  const Dependences &D = GetDeps(Dependences::AL_Statement);
  if (D.getSharedIslCtx() != S.getSharedIslCtx()) {
    LLVM_DEBUG(dbgs() << "DependenceInfo for another SCoP/isl_ctx\n");
    return false;
  }
  if (!D.hasValidDependences()) {
    LLVM_DEBUG(dbgs() << "Dependency information not available\n");
    return false;
  }

  // pipeline (*)
  /*
  Schedule = getPipelineWithIsl(?..?);
  Don't do anything else
  Assign schedule to the Scop and return
  */

  errs() << "PipelineLoops: " << PipelineLoops << '\n';
  if (PipelineLoops == "yes") {
    errs() << S << "\n";

    int temp_pipe_id = 0;
    for(auto stmt = S.begin() ; stmt != S.end() ; stmt++)
    {
      stmt->pipeline_id = temp_pipe_id;
      temp_pipe_id++;
    }

    // S.multiStmt();
    S.getPipelineGraph(); //we can't use dependencies, as we want stmt->MemRef.
    S.getPipelineRelations();
    S.getPipelineBlocks();
    S.getFinalF();
    S.getFinalE();
    S.getDependencyMaps();
    auto new_sch = schedule_pipeline_new(S);
    S.setScheduleTree(new_sch);
    S.markAsOptimized();

    errs() << "new schedule" << "\n";
    errs() << S << "\n";
    // errs() << S.getScheduleTree() << "\n";

    // auto s_acc_func = S.access_functions();
    // auto s_domains = S.getDomains().get_set_list();

    // // FIXEIT: find a general way.
    // auto rd_dom = s_domains.get_at(1);
    // auto wr_dom = s_domains.get_at(0);

    // auto raw_dep = D.getDependences(2);
    // errs() << "*********** \n" << raw_dep << "\n";
    // // can not use dependencies, as they don't cover the whole
    // // iteration domain.(maybe if I change st in the high level func.)
    // // errs() << "domain\n";
    // // errs() << raw_dep.domain();
    // // errs() << "range\n";
    // // errs() << raw_dep.range();

    // isl_ctx *i_ctx = isl_ctx_alloc();
    // isl::ctx ctx(i_ctx);

    // // apply read and write maps to the dependency.
    // // can we get the source map and destination map of dependency?

    // // this part about memory access should be reconsidered
    // // TODO: generalize this
    // auto begin_acc = s_acc_func.begin(); //<<== it's a vector
    // auto mem_acc_wr =
    //     begin_acc[1].get()->getAccessRelation(); //<<== isl::noexceptions
    // auto mem_acc_rd =
    //     begin_acc[3].get()->getAccessRelation(); //<<== isl::noexceptions

    // auto sch_map = schedule_pipeline(mem_acc_rd, mem_acc_wr, rd_dom, wr_dom);
    // S.setScheduleTree(sch_map);
    // S.markAsOptimized();

    // errs() << "new schedule" << "\n";
    // errs() << S.getScheduleTree() << "\n";

    return false;

  } else if (PipelineLoops != "no") {
    errs() << "-polly-pipeline-loops should be (yes/no)"
           << "\n";
    PipelineLoops = "no";
  }

  // Apply ISL's algorithm only if not overriden by the user. Note that
  // post-rescheduling optimizations (tiling, pattern-based, prevectorization)
  // rely on the coincidence/permutable annotations on schedule tree bands that
  // are added by the rescheduling analyzer. Therefore, disabling the
  // rescheduler implicitly also disables these optimizations.
  if (HasUserTransformation) {
    LLVM_DEBUG(
        dbgs() << "Skipping rescheduling due to manual transformation\n");
  } else {
    // Build input data.
    int ValidityKinds =
        Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
    int ProximityKinds;

    if (OptimizeDeps == "all")
      ProximityKinds =
          Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
    else if (OptimizeDeps == "raw")
      ProximityKinds = Dependences::TYPE_RAW;
    else {
      errs() << "Do not know how to optimize for '" << OptimizeDeps << "'"
             << " Falling back to optimizing all dependences.\n";
      ProximityKinds =
          Dependences::TYPE_RAW | Dependences::TYPE_WAR | Dependences::TYPE_WAW;
    }

    isl::union_set Domain = S.getDomains();

    if (!Domain)
      return false;

    isl::union_map Validity = D.getDependences(ValidityKinds);
    isl::union_map Proximity = D.getDependences(ProximityKinds);

    // Simplify the dependences by removing the constraints introduced by the
    // domains. This can speed up the scheduling time significantly, as large
    // constant coefficients will be removed from the dependences. The
    // introduction of some additional dependences reduces the possible
    // transformations, but in most cases, such transformation do not seem to be
    // interesting anyway. In some cases this option may stop the scheduler to
    // find any schedule.
    if (SimplifyDeps == "yes") {
      Validity = Validity.gist_domain(Domain);
      Validity = Validity.gist_range(Domain);
      Proximity = Proximity.gist_domain(Domain);
      Proximity = Proximity.gist_range(Domain);
    } else if (SimplifyDeps != "no") {
      errs()
          << "warning: Option -polly-opt-simplify-deps should either be 'yes' "
             "or 'no'. Falling back to default: 'yes'\n";
    }

    LLVM_DEBUG(dbgs() << "\n\nCompute schedule from: ");
    LLVM_DEBUG(dbgs() << "Domain := " << Domain << ";\n");
    LLVM_DEBUG(dbgs() << "Proximity := " << Proximity << ";\n");
    LLVM_DEBUG(dbgs() << "Validity := " << Validity << ";\n");

    unsigned IslSerializeSCCs;

    if (FusionStrategy == "max") {
      IslSerializeSCCs = 0;
    } else if (FusionStrategy == "min") {
      IslSerializeSCCs = 1;
    } else {
      errs() << "warning: Unknown fusion strategy. Falling back to maximal "
                "fusion.\n";
      IslSerializeSCCs = 0;
    }

    int IslMaximizeBands;

    if (MaximizeBandDepth == "yes") {
      IslMaximizeBands = 1;
    } else if (MaximizeBandDepth == "no") {
      IslMaximizeBands = 0;
    } else {
      errs()
          << "warning: Option -polly-opt-maximize-bands should either be 'yes'"
             " or 'no'. Falling back to default: 'yes'\n";
      IslMaximizeBands = 1;
    }

    int IslOuterCoincidence;

    if (OuterCoincidence == "yes") {
      IslOuterCoincidence = 1;
    } else if (OuterCoincidence == "no") {
      IslOuterCoincidence = 0;
    } else {
      errs() << "warning: Option -polly-opt-outer-coincidence should either be "
                "'yes' or 'no'. Falling back to default: 'no'\n";
      IslOuterCoincidence = 0;
    }

    isl_ctx *Ctx = S.getIslCtx().get();

    isl_options_set_schedule_outer_coincidence(Ctx, IslOuterCoincidence);
    isl_options_set_schedule_serialize_sccs(Ctx, IslSerializeSCCs);
    isl_options_set_schedule_maximize_band_depth(Ctx, IslMaximizeBands);
    isl_options_set_schedule_max_constant_term(Ctx, MaxConstantTerm);
    isl_options_set_schedule_max_coefficient(Ctx, MaxCoefficient);
    isl_options_set_tile_scale_tile_loops(Ctx, 0);

    auto OnErrorStatus = isl_options_get_on_error(Ctx);
    isl_options_set_on_error(Ctx, ISL_ON_ERROR_CONTINUE);

    auto SC = isl::schedule_constraints::on_domain(Domain);
    SC = SC.set_proximity(Proximity);
    SC = SC.set_validity(Validity);
    SC = SC.set_coincidence(Validity);
    Schedule = SC.compute_schedule();
    isl_options_set_on_error(Ctx, OnErrorStatus);

    ScopsRescheduled++;
    LLVM_DEBUG(printSchedule(dbgs(), Schedule, "After rescheduling"));
  }

  walkScheduleTreeForStatistics(Schedule, 1);

  // In cases the scheduler is not able to optimize the code, we just do not
  // touch the schedule.
  if (!Schedule)
    return false;

  // Apply post-rescheduling optimizations.
  const OptimizerAdditionalInfoTy OAI = {TTI, const_cast<Dependences *>(&D)};
  Schedule = ScheduleTreeOptimizer::optimizeSchedule(Schedule, &OAI);
  Schedule = hoistExtensionNodes(Schedule);
  LLVM_DEBUG(printSchedule(dbgs(), Schedule, "After post-optimizations"));
  walkScheduleTreeForStatistics(Schedule, 2);

  if (!ScheduleTreeOptimizer::isProfitableSchedule(S, Schedule))
    return false;

  auto ScopStats = S.getStatistics();
  ScopsOptimized++;
  NumAffineLoopsOptimized += ScopStats.NumAffineLoops;
  NumBoxedLoopsOptimized += ScopStats.NumBoxedLoops;
  LastSchedule = Schedule;

  S.setScheduleTree(Schedule);
  S.markAsOptimized();

  if (OptimizedScops)
    errs() << S;

  return false;
}

bool IslScheduleOptimizerWrapperPass::runOnScop(Scop &S) {
  releaseMemory();

  Function &F = S.getFunction();
  IslCtx = S.getSharedIslCtx();

  auto getDependences =
      [this](Dependences::AnalysisLevel) -> const Dependences & {
    return getAnalysis<DependenceInfo>().getDependences(
        Dependences::AL_Statement);
  };
  // auto &Deps  = getAnalysis<DependenceInfo>();
  TargetTransformInfo *TTI =
      &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  // getAnalysis<ScopDetectionWrapperPass>(F);

  // call to a function to find read/write relation as we want for
  // pipelining...?
  return runIslScheduleOptimizer(S, getDependences, TTI, LastSchedule);
}

static void runScheduleOptimizerPrinter(raw_ostream &OS,
                                        isl::schedule LastSchedule) {
  isl_printer *p;
  char *ScheduleStr;

  OS << "Calculated schedule:\n";

  if (!LastSchedule) {
    OS << "n/a\n";
    return;
  }

  p = isl_printer_to_str(LastSchedule.get_ctx().get());
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule(p, LastSchedule.get());
  ScheduleStr = isl_printer_get_str(p);
  isl_printer_free(p);

  OS << ScheduleStr << "\n";

  free(ScheduleStr);
}

void IslScheduleOptimizerWrapperPass::printScop(raw_ostream &OS, Scop &) const {
  runScheduleOptimizerPrinter(OS, LastSchedule);
}

void IslScheduleOptimizerWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  ScopPass::getAnalysisUsage(AU);
  AU.addRequired<DependenceInfo>();
  AU.addRequired<TargetTransformInfoWrapperPass>();

  AU.addPreserved<DependenceInfo>();
}

} // namespace

Pass *polly::createIslScheduleOptimizerWrapperPass() {
  return new IslScheduleOptimizerWrapperPass();
}

INITIALIZE_PASS_BEGIN(IslScheduleOptimizerWrapperPass, "polly-opt-isl",
                      "Polly - Optimize schedule of SCoP", false, false);
INITIALIZE_PASS_DEPENDENCY(DependenceInfo);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass);
INITIALIZE_PASS_END(IslScheduleOptimizerWrapperPass, "polly-opt-isl",
                    "Polly - Optimize schedule of SCoP", false, false)

static llvm::PreservedAnalyses
runIslScheduleOptimizerUsingNPM(Scop &S, ScopAnalysisManager &SAM,
                                ScopStandardAnalysisResults &SAR, SPMUpdater &U,
                                raw_ostream *OS) {
  DependenceAnalysis::Result &Deps = SAM.getResult<DependenceAnalysis>(S, SAR);
  auto GetDeps = [&Deps](Dependences::AnalysisLevel) -> const Dependences & {
    return Deps.getDependences(Dependences::AL_Statement);
  };
  TargetTransformInfo *TTI = &SAR.TTI;
  isl::schedule LastSchedule;
  bool Modified = runIslScheduleOptimizer(S, GetDeps, TTI, LastSchedule);
  if (OS) {
    *OS << "Printing analysis 'Polly - Optimize schedule of SCoP' for region: '"
        << S.getName() << "' in function '" << S.getFunction().getName()
        << "':\n";
    runScheduleOptimizerPrinter(*OS, LastSchedule);
  }

  if (!Modified)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<AllAnalysesOn<Module>>();
  PA.preserveSet<AllAnalysesOn<Function>>();
  PA.preserveSet<AllAnalysesOn<Loop>>();
  return PA;
}

llvm::PreservedAnalyses
IslScheduleOptimizerPass::run(Scop &S, ScopAnalysisManager &SAM,
                              ScopStandardAnalysisResults &SAR, SPMUpdater &U) {
  return runIslScheduleOptimizerUsingNPM(S, SAM, SAR, U, nullptr);
}

llvm::PreservedAnalyses
IslScheduleOptimizerPrinterPass::run(Scop &S, ScopAnalysisManager &SAM,
                                     ScopStandardAnalysisResults &SAR,
                                     SPMUpdater &U) {
  return runIslScheduleOptimizerUsingNPM(S, SAM, SAR, U, &OS);
}
