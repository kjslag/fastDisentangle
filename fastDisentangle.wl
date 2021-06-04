(* ::Package:: *)

(* ::Text:: *)
(*NOTE: This implementation does not correctly handle the Orthogonalization in step 6 of https://arxiv.org/pdf/2104.08283 for certain fine-tuned cases where there are linearly dependent vectors.*)


Clear[fastDisentangle, randomReal, randomComplex, tensorContract, stripTensorProduct, entanglement]

(* fastDisentangle returns a unitary tensor with dimensions
     (\[Chi]1, \[Chi]2, \[Chi]1 \[Chi]2) that approximately disentangles `A`.
   `A` must have dimensions (\[Chi]1 \[Chi]2, \[Chi]3, \[Chi]4) where either
     \[Chi]2\[LessEqual]Ceiling[\[Chi]4/Ceiling[\[Chi]1/\[Chi]3]] or \[Chi]1\[LessEqual]Ceiling[\[Chi]3/Ceiling[\[Chi]2/\[Chi]4]].
   If these ratios are integers, then \[Chi]1 \[Chi]2 <= \[Chi]3 \[Chi]4 is sufficient.
   \[Chi]1 <= \[Chi]3 and \[Chi]2 <= \[Chi]4 is also sufficient.
   example: fastDisentangle[{2,3}, randomComplex[6,5,7]] *)
fastDisentangle[{\[Chi]1_Integer,\[Chi]2_Integer}, A_?ArrayQ,
    transposeQ0_:Automatic] /;
    ArrayDepth@A==3 && \[Chi]1 \[Chi]2==Length@A && \[Chi]1<=Dimensions[A][[2]] && \[Chi]2<=Dimensions[A][[3]] :=
    Module[{r,\[Alpha]3,\[Alpha]4,V3,V4,q,B,B\[Dagger],U, transposeQ = Replace[transposeQ0, Automatic -> \[Chi]1>\[Chi]2],
        rand=If[Total[Abs@Im@A,\[Infinity]]==0, randomReal, randomComplex]},
    (* implementing Algorithm 1 in https://arxiv.org/pdf/2104.08283 *)
    r = rand@Length@A; (*1*)
    {\[Alpha]3,\[Alpha]4} = {#1[[All,1]]\[Conjugate], #3[[All,1]]}& @@ SingularValueDecomposition[r . A, 1]; (*2*)
    V3 = Last@SingularValueDecomposition[A . \[Alpha]4, \[Chi]1]; (*3*)
    V4 = Last@SingularValueDecomposition[Transpose[A,{1,3,2}] . \[Alpha]3, \[Chi]2]; (*4*)
    B  = tensorContract[A,V3,V4, {{2,4},{3,6}}]; (*5*)
    B\[Dagger] = Conjugate@Transpose[B, If[transposeQ,{3,2,1},{3,1,2}]];
    (* help ensure linearly independent rows of B\[Dagger]: *)
    B\[Dagger]+= $MachineEpsilon Max@Abs@B\[Dagger] rand@@Dimensions@B\[Dagger];
    (* NOTE: Orthogonalize is fast but it will not handle linearly dependant vectors the way we need to in certain cases, eg checkAnsatz[4,2,1,4,4,2].
       See eg Python or Julia implementations for correct code. *)
    U  = ArrayReshape[Orthogonalize[Catenate@B\[Dagger],
             Tolerance->0, Method->"ModifiedGramSchmidt"], Dimensions@B\[Dagger]]; (*6*)
    If[transposeQ, U = Transpose@U];
    U]
fastDisentangle[{\[Chi]1_Integer,\[Chi]2_Integer}, A_?ArrayQ] /;
    ArrayDepth@A==3 && \[Chi]1 \[Chi]2==Length@A && \[Chi]1>Dimensions[A][[2]] && \[Chi]2<=Dimensions[A][[3]] :=
    Module[{\[Chi]4to3,\[Chi]4\[Prime],\[Chi]3,\[Chi]4},
    (* implementing Appendix B in https://arxiv.org/pdf/2104.08283 *)
    {\[Chi]3,\[Chi]4} = Rest@Dimensions@A;
    \[Chi]4to3 = \[Chi]1 / \[Chi]3 // Ceiling;
    \[Chi]4\[Prime] = \[Chi]4 / \[Chi]4to3 // Ceiling;
    Module[{V},
    V = Last@SingularValueDecomposition@Catenate@A;(*1*)
    V = ArrayReshape[PadRight[V, {\[Chi]4, \[Chi]4to3 \[Chi]4\[Prime]}], {\[Chi]4, \[Chi]4to3, \[Chi]4\[Prime]}];(*2*)
    fastDisentangle[{\[Chi]1, \[Chi]2}, Flatten[A . V, {{1},{2,3},{4}}], False](*3*)
    ] /; \[Chi]2 <= \[Chi]4\[Prime]]
fastDisentangle[{\[Chi]1_Integer,\[Chi]2_Integer}, A_?ArrayQ] /;
    ArrayDepth@A==3 && \[Chi]1 \[Chi]2==Length@A && \[Chi]1<=Dimensions[A][[2]] && \[Chi]2>Dimensions[A][[3]] :=
    Transpose[fastDisentangle[{\[Chi]2,\[Chi]1}, Transpose[A, {1,3,2}]], {2,1,3}]
fastDisentangle@___ := Throw@"fastDisentangle: bad arguments"

(* random complex tensor with dimensions `ns` *)
randomReal@ns__Integer := RandomVariate[NormalDistribution[],{ns}]
randomComplex@ns__Integer := randomReal[ns,2] . ({1,I}/Sqrt[2.])

(* TensorContract for the tensor product of multiple tensors *)
tensorContract[T_, s_] := TensorContract[T, s]
tensorContract[T__, s_] := stripTensorProduct@TensorContract[Inactive[TensorProduct]@T, s]
stripTensorProduct@Inactive[TensorProduct]@T__ := TensorProduct@T
stripTensorProduct@TensorTranspose[Inactive[TensorProduct]@T__, perm_] :=
    TensorTranspose[TensorProduct@T, perm]

(* entanglement entropy of U.A, as defined in equation (10) of https://arxiv.org/pdf/2104.08283 *)
entanglement@UA_ /; ArrayDepth@UA==4 := With[{\[Lambda]s=SingularValueList[Flatten[UA, {{1,3},{2,4}}], Tolerance->0]},
    With[{ps = \[Lambda]s^2/\[Lambda]s . \[Lambda]s}, -ps . Log@Clip[ps,{$MinMachineNumber,1}]]]


(*verification code:*)

Clear[checkAnsatz, checkAnsatzRepeated]

(* check that the ansatz in equation (1) of https://arxiv.org/pdf/2104.08283 results in the minimal entanglement entropy *)
checkAnsatz[\[Chi]1_,\[Chi]2_,\[Chi]3a_,\[Chi]4b_,\[Chi]3c_,\[Chi]4c_, \[Epsilon]_:0, rand_:randomComplex] := Module[{M1,M2,M3,A,normalize=#/Norm@Flatten@#&},
    {M1,M2,M3} = normalize /@ {rand[\[Chi]1,\[Chi]3a], rand[\[Chi]2,\[Chi]4b], rand[\[Chi]3c,\[Chi]4c]};
    A = Flatten[M1\[TensorProduct]M2\[TensorProduct]M3, {{1,3},{2,5},{4,6}}];
    A = A + \[Epsilon] normalize[randomComplex@@Dimensions@A];
    entanglement[fastDisentangle[{\[Chi]1,\[Chi]2}, A] . A] - entanglement@ArrayReshape[M3,{1,1,\[Chi]3c,\[Chi]4c}]]

(* repeatedly check the ansatz *)
checkAnsatzRepeated[max\[Chi]_:9] := Module[{c=0,\[Epsilon],args,\[Chi]s,\[Chi]1,\[Chi]2,\[Chi]3a,\[Chi]4b,\[Chi]3c,\[Chi]4c,\[Chi]3,\[Chi]4,S},
    c//Dynamic//PrintTemporary;
    While[True, ++c;
        {\[Chi]1,\[Chi]2,\[Chi]3a,\[Chi]4b,\[Chi]3c,\[Chi]4c} = \[Chi]s = RandomInteger[{1,max\[Chi]}, 6];
        \[Chi]3 = \[Chi]3a \[Chi]3c;
        \[Chi]4 = \[Chi]4b \[Chi]4c;
        \[Epsilon]  = RandomChoice@{0, 10^RandomReal@{-20,-6}};
        (* Due to the Orthogonalize issue noted above, this implementation only works for \[Chi]1 <= \[Chi]3a \[And] \[Chi]2 <= \[Chi]4b. *)
        (*If[(\[Chi]1 <= \[Chi]3 \[And] \[Chi]2 <= \[Chi]4) \[Or] ((\[Chi]3c==1 \[Or] \[Chi]4c==1) \[And] (\[Chi]2 <= Ceiling[\[Chi]4/Ceiling[\[Chi]1/\[Chi]3]] \[Or] \[Chi]1 <= Ceiling[\[Chi]3/Ceiling[\[Chi]2/\[Chi]4]])),*)
        If[\[Chi]1 <= \[Chi]3a \[And] \[Chi]2 <= \[Chi]4b,
            args = Join[\[Chi]s, {\[Epsilon], RandomChoice@{randomReal,randomComplex}}];
            S = checkAnsatz@@args;
            If[S>Sqrt@Max[\[Epsilon],$MachineEpsilon], args->S // Throw]]]]
