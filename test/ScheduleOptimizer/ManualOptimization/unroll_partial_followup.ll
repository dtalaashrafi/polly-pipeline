; RUN: opt %loadPolly -polly-opt-isl -polly-ast -analyze < %s | FileCheck %s --match-full-lines
; RUN: opt %loadPolly -polly-opt-isl -polly-codegen -simplifycfg -S < %s | FileCheck %s --check-prefix=CODEGEN
;
; Partial unroll by a factor of 4.
;
define void @func(i32 %n, double* noalias nonnull %A) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      store double 42.0, double* %A
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for, !llvm.loop !2

exit:
  br label %return

return:
  ret void
}


!2 = distinct !{!2, !4, !5, !6}
!4 = !{!"llvm.loop.unroll.enable", i1 true}
!5 = !{!"llvm.loop.unroll.count", i4 4}
!6 = !{!"llvm.loop.unroll.followup_unrolled", !7}

!7 = distinct !{!7, !8}
!8 = !{!"llvm.loop.id", !"This-is-the-unrolled-loop"}


; CHECK-LABEL: Printing analysis 'Polly - Optimize schedule of SCoP' for region: 'for => return' in function 'func':
; CHECK:       domain: "[n] -> { Stmt_body[i0] : 0 <= i0 < n }"
; CHECKL         mark: "Loop with Metadata"
; CHECK:           schedule: "[n] -> [{ Stmt_body[i0] -> [(i0 - (i0) mod 4)] }]"
; CHECK:             sequence:
; CHECK-NEXT:        - filter: "[n] -> { Stmt_body[i0] : (i0) mod 4 = 0 }"
; CHECK-NEXT:        - filter: "[n] -> { Stmt_body[i0] : (-1 + i0) mod 4 = 0 }"
; CHECK-NEXT:        - filter: "[n] -> { Stmt_body[i0] : (2 + i0) mod 4 = 0 }"
; CHECK-NEXT:        - filter: "[n] -> { Stmt_body[i0] : (1 + i0) mod 4 = 0 }"


; CHECK-LABEL: Printing analysis 'Polly - Generate an AST of the SCoP (isl)'for => return' in function 'func':
; CHECK:       // Loop with Metadata
; CHECK-NEXT:  for (int c0 = 0; c0 < n; c0 += 4) {


; CODEGEN: br i1 %polly.loop_cond, label %polly.loop_header, label %polly.exiting, !llvm.loop ![[LOOPID:[0-9]+]]
; CODEGEN: ![[LOOPID]] = distinct !{![[LOOPID]], ![[LOOPNAME:[0-9]+]]}
; CODEGEN: ![[LOOPNAME]] = !{!"llvm.loop.id", !"This-is-the-unrolled-loop"}
