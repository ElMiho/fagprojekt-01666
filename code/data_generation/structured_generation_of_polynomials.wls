argv = Rest @ $ScriptCommandLine;
argStart = ToExpression[argv[[1]]];
argEnd = ToExpression[argv[[2]]];
LaunchKernels[4]

listToInteger[list_, base_] := Module[
    {res = 0},
    For[i = 0, i < Length[list], i++,
        res = res + list[[i + 1]]*base^i;
    ];
    res
]

integerToList[n_, sumDeg_, base_] := Module[
    {res = {}, var = 0, k = n, i = 1},
    For[i, i <= sumDeg, i++,
        var = Mod[k, base^i];
        AppendTo[res, var/(base^(i - 1))];
        k = k - var;
    ];
    res
]

Needs["Combinatorica`"];
all = Range[-5,5];

generateFractions[list_] = list[[1]] / list[[2]];
fractionTuples = CartesianProduct[all, all];

(* Generation fraction list and clean up w.r.t. duplicates and division by 0 *)
res = generateFractions /@ fractionTuples;
res = DeleteDuplicates[res];
res = DeleteCases[res, ComplexInfinity];
fractionList = DeleteCases[res, Indeterminate];

(* No positive integers in denominator! *)
fractionListNoPositiveIntegers = DeleteCases[
    fractionList, 
    x_ /; x > 0 && MemberQ[Range[1,5], x]
    ];

(* To ensure convergence we look at P(n)/Q(n) such that deg(P) <= deg(Q(n)) - 2 *)
numeratorDegree = Range[0, 8]; 
denominatorDegree := Range[2, 10];
allPossiblePolynomialFractions := CartesianProduct[numeratorDegree, denominatorDegree];
legalPolynomialFractions := DeleteCases[
    allPossiblePolynomialFractions, 
    list_ /; list[[1]] >= list[[2]] - 1
];

maxTime = 0.1;

(* Loading scaling (better name?) factors and create a map *)
scaleFactors = Import["../data_scripts/Sample_size.mat"][1];
mapFactors[list_] := {
    IntegerPart[list[[1]]], 
    IntegerPart[list[[2]]]} -> list[[3]];
factorsMap := Association[mapFactors /@ scaleFactors];