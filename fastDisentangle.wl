(* ::Package:: *)

Clear[fastDisentangle, randomComplex, tensorContract, stripTensorProduct, entanglement]

(* fastDisentangle returns a unitary tensor with dimensions
     (\[Chi]1, \[Chi]2, \[Chi]1 \[Chi]2) that approximately disentangles `A`.
   `A` must have dimensions (\[Chi]1 \[Chi]2, \[Chi]3, \[Chi]4) where either
     \[Chi]2\[LessEqual]Ceiling[\[Chi]4/Ceiling[\[Chi]1/\[Chi]3]] or \[Chi]1\[LessEqual]Ceiling[\[Chi]3/Ceiling[\[Chi]2/\[Chi]4]].
   If these ratios are integers, then \[Chi]1 \[Chi]2 <= \[Chi]3 \[Chi]4 is sufficient.
   \[Chi]1 <= \[Chi]3 and \[Chi]2 <= \[Chi]4 is also sufficient.
   example: fastDisentangle[{2,3}, randomComplex[6,5,7]] *)
fastDisentangle[{\[Chi]1_Integer,\[Chi]2_Integer}, A_?ArrayQ] /;
    ArrayDepth@A==3 && \[Chi]1 \[Chi]2==Length@A && \[Chi]1<=Dimensions[A][[2]] && \[Chi]2<=Dimensions[A][[3]] :=
    Module[{r,\[Alpha]3,\[Alpha]4,V3,V4,q,B,B\[Dagger],U,transposeQ=\[Chi]1>\[Chi]2},
    r = randomComplex@Length@A; (*1*)
    {\[Alpha]3,\[Alpha]4} = {#1[[All,1]]\[Conjugate], #3[[All,1]]}& @@ SingularValueDecomposition[r . A, 1]; (*2*)
    V3 = Last@SingularValueDecomposition[A . \[Alpha]4, \[Chi]1]; (*3*)
    V4 = Last@SingularValueDecomposition[Transpose[A,{1,3,2}] . \[Alpha]3, \[Chi]2]; (*4*)
    B  = tensorContract[A,V3,V4, {{2,4},{3,6}}]; (*5*)
    B\[Dagger] = Conjugate@Transpose[B, If[transposeQ,{3,2,1},{3,1,2}]];
    (* help ensure linearly independent rows of B\[Dagger]: *)
    B\[Dagger]+= $M\[Epsilon] Max@Abs@B\[Dagger] RC@@Dimensions@B\[Dagger];
    U  = ArrayReshape[Orthogonalize[Catenate@B\[Dagger],
             Tolerance->0, Method->"ModifiedGramSchmidt"], Dimensions@B\[Dagger]]; (*6*)
    If[transposeQ, U = Transpose@U];
    U]
fastDisentangle[{\[Chi]1_Integer,\[Chi]2_Integer}, A_?ArrayQ] /;
    ArrayDepth@A==3 && \[Chi]1 \[Chi]2==Length@A && \[Chi]1>Dimensions[A][[2]] && \[Chi]2<=Dimensions[A][[3]] :=
    Module[{\[Chi]4to3,\[Chi]4\[Prime],\[Chi]3,\[Chi]4},
    {\[Chi]3,\[Chi]4} = Rest@Dimensions@A;
    \[Chi]4to3 = \[Chi]1 / \[Chi]3 // Ceiling;
    \[Chi]4\[Prime] = \[Chi]4 / \[Chi]4to3 // Ceiling;
    Module[{V},
    V = Last@SingularValueDecomposition@Catenate@A;(*1*)
    V = ArrayReshape[PadRight[V, {\[Chi]4, \[Chi]4to3 \[Chi]4\[Prime]}], {\[Chi]4, \[Chi]4to3, \[Chi]4\[Prime]}];(*2*)
    fastDisentangle[{\[Chi]1, \[Chi]2}, Flatten[A . V, {{1},{2,3},{4}}]](*3*)
    ] /; \[Chi]2 <= \[Chi]4\[Prime]]
fastDisentangle[{\[Chi]1_Integer,\[Chi]2_Integer}, A_?ArrayQ] /;
    ArrayDepth@A==3 && \[Chi]1 \[Chi]2==Length@A && \[Chi]1<=Dimensions[A][[2]] && \[Chi]2>Dimensions[A][[3]] :=
    Transpose[fastDisentangle[{\[Chi]2,\[Chi]1}, Transpose[A, {1,3,2}]], {2,1,3}]
fastDisentangle@___ := Throw@"fastDisentangle: bad arguments"

(* random complex tensor with dimensions `ns` *)
randomComplex@ns__Integer := RandomVariate[NormalDistribution[],{ns,2}] . ({1,I}/Sqrt[2.])

(* TensorContract for the tensor product of multiple tensors *)
tensorContract[T_, s_] := TensorContract[T, s]
tensorContract[T__, s_] := stripTensorProduct@TensorContract[Inactive[TensorProduct]@T, s]
stripTensorProduct@Inactive[TensorProduct]@T__ := TensorProduct@T
stripTensorProduct@TensorTranspose[Inactive[TensorProduct]@T__, perm_] :=
    TensorTranspose[TensorProduct@T, perm]

(* entanglement entropy of U.A *)
entanglement@UA_ /; ArrayDepth@UA==4 := With[{\[Lambda]s=SingularValueList[Flatten[UA, {{1,3},{2,4}}], Tolerance->0]},
    With[{ps = \[Lambda]s^2/\[Lambda]s . \[Lambda]s}, -Total[ps Log@Clip[ps,{$MinMachineNumber,1}]]]]
