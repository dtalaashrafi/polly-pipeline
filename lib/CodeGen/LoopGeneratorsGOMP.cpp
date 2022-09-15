//===------ LoopGeneratorsGOMP.cpp - IR helper to create loops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create parallel loops as LLVM-IR.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/LoopGeneratorsGOMP.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Module.h"

using namespace llvm;
using namespace polly;

void ParallelLoopGeneratorGOMP::createCallSpawnThreads(Value *SubFn,
                                                       Value *SubFnParam,
                                                       Value *LB, Value *UB,
                                                       Value *Stride) {
  const std::string Name = "GOMP_parallel_loop_runtime_start";
  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {PointerType::getUnqual(FunctionType::get(
                          Builder.getVoidTy(), Builder.getInt8PtrTy(), false)),
                      Builder.getInt8PtrTy(),
                      Builder.getInt32Ty(),
                      LongType,
                      LongType,
                      LongType};

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {SubFn, SubFnParam, Builder.getInt32(PollyNumThreads),
                   LB,    UB,         Stride};

  Builder.CreateCall(F, Args);
}

void ParallelLoopGeneratorGOMP::deployParallelExecution(Function *SubFn,
                                                        Value *SubFnParam,
                                                        Value *LB, Value *UB,
                                                        Value *Stride) {
  // Tell the runtime we start a parallel loop
  createCallSpawnThreads(SubFn, SubFnParam, LB, UB, Stride);
  Builder.CreateCall(SubFn, SubFnParam);
  createCallJoinThreads();
}

// pipeline (*)
void ParallelLoopGeneratorGOMP::deployPipelineExecution(
    Function *SubFn, Value *SubFnParam, Value *LB, Value *UB, Value *Stride,
    Value *OutDepend, Value *OutIndex, Value *InDepend, Value *InIndex,
    Value *Size, Value *DependNum) {
  static int num = 0;
  const std::string Name = "make_task";
  //  + std::to_string(num);
  // const std::string Name1 = "last_task_" + std::to_string(num);
  num++;
  Function *F = M->getFunction(Name);
  // Function *F1 = M->getFunction(Name1);

  if (!F)
  // || !F1)
  {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    Type *Params[] = {PointerType::getUnqual(FunctionType::get(
                          Builder.getVoidTy(), Builder.getInt8PtrTy(), false)),
                      Builder.getInt8PtrTy(),
                      Builder.getInt64Ty(),
                      Builder.getInt32Ty(),
                      PointerType::get(Builder.getInt64Ty(), 0),
                      PointerType::get(Builder.getInt32Ty(), 0),
                      Builder.getInt32Ty(),
                      Builder.getInt32Ty()};

    // PointerType::get( Builder.getInt64Ty(),0),
    //                   PointerType::get(Builder.getInt32Ty(),0)

    //  Builder.getInt64Ty(),
    // Builder.getInt32Ty()

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
    // F1 = Function::Create(Ty, Linkage, Name1, M);
  }

  Value *Args[] = {SubFn,    SubFnParam, OutDepend, OutIndex,
                   InDepend, InIndex,    Size,      DependNum};
  Builder.CreateCall(F, Args);
}

Function *ParallelLoopGeneratorGOMP::prepareSubFnDefinition(Function *F) const {
  FunctionType *FT =
      FunctionType::get(Builder.getVoidTy(), {Builder.getInt8PtrTy()}, false);
  Function *SubFn = Function::Create(FT, Function::InternalLinkage,
                                     F->getName() + "_polly_subfn", M);
  // Name the function's arguments
  SubFn->arg_begin()->setName("polly.par.userContext");
  return SubFn;
}

// Create a subfunction of the following (preliminary) structure:
//
//    PrevBB
//       |
//       v
//    HeaderBB
//       |   _____
//       v  v    |
//   CheckNextBB  PreHeaderBB
//       |\       |
//       | \______/
//       |
//       v
//     ExitBB
//
// HeaderBB will hold allocations and loading of variables.
// CheckNextBB will check for more work.
// If there is more work to do: go to PreHeaderBB, otherwise go to ExitBB.
// PreHeaderBB loads the new boundaries (& will lead to the loop body later on).
// ExitBB marks the end of the parallel execution.
std::tuple<Value *, Function *>
ParallelLoopGeneratorGOMP::createSubFn(Value *Stride, AllocaInst *StructData,
                                       SetVector<Value *> Data,
                                       ValueMapT &Map) {
  if (PollyScheduling != OMPGeneralSchedulingType::Runtime) {
    // User tried to influence the scheduling type (currently not supported)
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the scheduling type 'runtime'.\n";
  }

  if (PollyChunkSize != 0) {
    // User tried to influence the chunk size (currently not supported)
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the default chunk size.\n";
  }

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();

  // Store the previous basic block.
  BasicBlock *PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.par.setup", SubFn);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "polly.par.exit", SubFn);
  BasicBlock *CheckNextBB =
      BasicBlock::Create(Context, "polly.par.checkNext", SubFn);
  BasicBlock *PreHeaderBB =
      BasicBlock::Create(Context, "polly.par.loadIVBounds", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(ExitBB, HeaderBB);
  DT.addNewBlock(CheckNextBB, HeaderBB);
  DT.addNewBlock(PreHeaderBB, HeaderBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  Value *LBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.LBPtr");
  Value *UBPtr = Builder.CreateAlloca(LongType, nullptr, "polly.par.UBPtr");
  Value *UserContext = Builder.CreateBitCast(
      &*SubFn->arg_begin(), StructData->getType(), "polly.par.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);
  Builder.CreateBr(CheckNextBB);

  // Add code to check if another set of iterations will be executed.
  Builder.SetInsertPoint(CheckNextBB);
  Value *Next = createCallGetWorkItem(LBPtr, UBPtr);
  Value *HasNextSchedule = Builder.CreateTrunc(
      Next, Builder.getInt1Ty(), "polly.par.hasNextScheduleBlock");
  Builder.CreateCondBr(HasNextSchedule, PreHeaderBB, ExitBB);

  // Add code to load the iv bounds for this set of iterations.
  Builder.SetInsertPoint(PreHeaderBB);
  Value *LB = Builder.CreateLoad(LongType, LBPtr, "polly.par.LB");
  Value *UB = Builder.CreateLoad(LongType, UBPtr, "polly.par.UB");

  // Subtract one as the upper bound provided by OpenMP is a < comparison
  // whereas the codegenForSequential function creates a <= comparison.
  UB = Builder.CreateSub(UB, ConstantInt::get(LongType, 1),
                         "polly.par.UBAdjusted");

  Builder.CreateBr(CheckNextBB);
  Builder.SetInsertPoint(&*--Builder.GetInsertPoint());
  BasicBlock *AfterBB;
  Value *IV =
      createLoop(LB, UB, Stride, Builder, LI, DT, AfterBB, ICmpInst::ICMP_SLE,
                 nullptr, true, /* UseGuard */ false);

  BasicBlock::iterator LoopBody = Builder.GetInsertPoint();

  // Add code to terminate this subfunction.
  Builder.SetInsertPoint(ExitBB);
  createCallCleanupThread();
  Builder.CreateRetVoid();

  Builder.SetInsertPoint(&*LoopBody);

  return std::make_tuple(IV, SubFn);
}

// pipeline (*)
std::tuple<Value *, Function *> ParallelLoopGeneratorGOMP::createSubFnPipeline(
    Value *Stride, AllocaInst *StructData, SetVector<Value *> Data,
    ValueMapT &Map) {
  if (PollyScheduling != OMPGeneralSchedulingType::Runtime) {
    // User tried to influence the scheduling type (currently not supported)
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the scheduling type 'runtime'.\n";
  }

  if (PollyChunkSize != 0) {
    // User tried to influence the chunk size (currently not supported)
    errs() << "warning: Polly's GNU OpenMP backend solely "
              "supports the default chunk size.\n";
  }

  Function *SubFn = createSubFnDefinition();
  LLVMContext &Context = SubFn->getContext();
  Value *IV;

  // Store the previous basic block.
  BasicBlock *PrevBB = Builder.GetInsertBlock();

  // Create basic blocks.
  BasicBlock *HeaderBB = BasicBlock::Create(Context, "polly.pipe.setup",
                                            SubFn); // it is added to to subfn
  BasicBlock *TaskBB = BasicBlock::Create(Context, "polly.pipe.task", SubFn);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "polly.pipe.exit", SubFn);

  DT.addNewBlock(HeaderBB, PrevBB);
  DT.addNewBlock(TaskBB, HeaderBB);
  DT.addNewBlock(ExitBB, TaskBB);

  // Fill up basic block HeaderBB.
  Builder.SetInsertPoint(HeaderBB);
  Value *UserContext = Builder.CreateBitCast(
      &*SubFn->arg_begin(), StructData->getType(), "polly.pipe.userContext");

  extractValuesFromStruct(Data, StructData->getAllocatedType(), UserContext,
                          Map);
  Builder.CreateBr(TaskBB);

  Builder.SetInsertPoint(TaskBB);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();
  // Builder.SetInsertPoint(&*LoopBody);

  Builder.SetInsertPoint((TaskBB->getTerminator()));

  return std::make_tuple(IV, SubFn);
}

// pipeline(*) // depend

std::tuple<Value *, Function *> ParallelLoopGeneratorGOMP::createGetDependFn(
    Value *LB, Value *UB, Value *Stride, SetVector<Value *> &UsedValues,
    ValueMapT &Map, BasicBlock::iterator *LoopBody) {
  Function *DependFn = createDependFnDefinition();
  LLVMContext &Context = DependFn->getContext();
  AllocaInst *StructData = storeValuesIntoStruct(UsedValues);

  // Create basic blocks.
  BasicBlock *HeaderBB =
      BasicBlock::Create(Context, "polly.pipe.depend.setup", DependFn);
  BasicBlock *ComputeBB =
      BasicBlock::Create(Context, "polly.pipe.depend.compute", DependFn);
  BasicBlock *ExitBB =
      BasicBlock::Create(Context, "polly.pipe.depend.exit", DependFn);

  Builder.SetInsertPoint(HeaderBB);
  Value *UserContext =
      Builder.CreateBitCast(&*DependFn->arg_begin(), StructData->getType(),
                            "polly.pipe.depend.userContext");
  extractValuesFromStruct(UsedValues, StructData->getAllocatedType(),
                          UserContext, Map);

  Builder.CreateBr(ComputeBB);

  Builder.SetInsertPoint(ComputeBB);
  Builder.CreateBr(ExitBB);

  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();

  // set builder ...
}

// pipeline (*) //depend
Function *ParallelLoopGeneratorGOMP::prepareDependFnDefinition(Function *F) {
  // NOTE: change InternalLinkage to ExternalLinkage in the first step.
  FunctionType *FT =
      FunctionType::get(Builder.getVoidTy(), {Builder.getInt8PtrTy()}, false);
  Function *SubFn = Function::Create(FT, Function::ExternalLinkage,
                                     F->getName() + "_polly_dependfn", M);
  // Name the function's arguments
  SubFn->arg_begin()->setName("polly.pipe.depend.userContext");
  return SubFn;
}

Value *ParallelLoopGeneratorGOMP::createCallGetWorkItem(Value *LBPtr,
                                                        Value *UBPtr) {
  const std::string Name = "GOMP_loop_runtime_next";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {LongType->getPointerTo(), LongType->getPointerTo()};
    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {LBPtr, UBPtr};
  Value *Return = Builder.CreateCall(F, Args);
  Return = Builder.CreateICmpNE(
      Return, Builder.CreateZExt(Builder.getFalse(), Return->getType()));
  return Return;
}

// pipeline (*)
Value *ParallelLoopGeneratorGOMP::createCallGetWorkItemPipeline(Value *LBPtr,
                                                                Value *UBPtr) {
  const std::string Name = "next";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;
    Type *Params[] = {LongType->getPointerTo(), LongType->getPointerTo()};
    FunctionType *Ty = FunctionType::get(Builder.getInt8Ty(), Params, false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Value *Args[] = {LBPtr, UBPtr};
  Value *Return = Builder.CreateCall(F, Args);
  Return = Builder.CreateICmpNE(
      Return, Builder.CreateZExt(Builder.getFalse(), Return->getType()));
  return Return;
}

void ParallelLoopGeneratorGOMP::createCallJoinThreads() {
  const std::string Name = "GOMP_parallel_end";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}

void ParallelLoopGeneratorGOMP::createCallCleanupThread() {
  const std::string Name = "GOMP_loop_end_nowait";

  Function *F = M->getFunction(Name);

  // If F is not available, declare it.
  if (!F) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    F = Function::Create(Ty, Linkage, Name, M);
  }

  Builder.CreateCall(F, {});
}
