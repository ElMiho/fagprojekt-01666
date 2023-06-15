LaunchKernels[8];
Needs["Combinatorica`"];
maxTime = 7;

numeratorDegree = Range[0, 8]; 
denominatorDegree = Range[2, 10];
allPossiblePolynomialFractions = CartesianProduct[numeratorDegree, denominatorDegree];
legalPolynomialFractions = DeleteCases[
    allPossiblePolynomialFractions, 
    list_ /; list[[1]] >= list[[2]] - 1
];

Print[legalPolynomialFractions];

all = Range[-5,5];

generateFractions[list_] := list[[1]] / list[[2]];
fractionTuples = CartesianProduct[all, all];

(* Generation fraction list and clean up w.r.t. duplicates and division by 0 *)
res = generateFractions /@ fractionTuples;
res = DeleteDuplicates[res];
res = DeleteCases[res, ComplexInfinity];
coeffList = DeleteCases[res, Indeterminate];

(* 
    yes we use the same roots for coeff,
    but we simply want to show 
    that a lot of rootsum shows up for higher degrees
*)

Print[coeffList];
numberOfPolynomials = 100;

Print["Evaluating sums"];
For[i = 1, i <= Length[legalPolynomialFractions], i++,
    numeratorDegree = legalPolynomialFractions[[i]][[1]];
    denominatorDegree = legalPolynomialFractions[[i]][[2]];

    (* numberOfPolynomials sums from each category in parallel *)

    answers = {};
    SetSharedVariable[answers];
    
    ParallelDo[
        numeratorCoeff = RandomChoice[coeffList, numeratorDegree];
        denominatorCoeff = RandomChoice[coeffList, denominatorDegree];

        numerator = n^(numeratorDegree) + Sum[
            numeratorCoeff[[i]] * n^(i-1), {i, 1, numeratorDegree}
        ];
        denominator = n^(denominatorDegree) + Sum[
            denominatorCoeff[[i]] * n^(i-1), {i, 1, denominatorDegree}
        ];

        expr = numerator / denominator;
        s = TimeConstrained[
            Sum[expr, {n, 1, Infinity}], maxTime
        ];

        AppendTo[answers, {numeratorCoeff, denominatorCoeff, s}];,
        {k, 1, numberOfPolynomials}
    ];

    Export[Directory[] <> "/coeff_sum/" <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree] <> ".txt", answers]
];

Print[answers];