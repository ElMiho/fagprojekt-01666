Needs["Combinatorica`"];
all = Range[-5,5];

generateFractions[list_] = list[[1]] / list[[2]];
fractionTuples = CartesianProduct[all, all];

res = generateFractions /@ fractionTuples;
res = DeleteDuplicates[res];
res = DeleteCases[res, ComplexInfinity];
fractionList = DeleteCases[res, Indeterminate];
fractionListNoPositiveIntegers = DeleteCases[
    fractionList, 
    x_ /; x > 0 && MemberQ[Range[1,5], x]
    ];

maxDegreeDenominator = 20;
minutes = N[maxTime * numberOfSums / 60];
hours = minutes / 60;
Print["Total max running time (minutes): " <> ToString[minutes]]
Print["Total max running time (hours): " <> ToString[hours]]

expressions = {};
answers = {};

For[i = 1, i <= numberOfSums, i++,
    denominatorDegree = RandomChoice[Range[3, maxDegreeDenominator]];
    numeratorDegree = RandomChoice[Range[0, denominatorDegree - 2]];

    denominator = RandomChoice[fractionListNoPositiveIntegers, denominatorDegree];
    numerator = RandomChoice[fractionList, numeratorDegree];
    expr = 
        Product[n - numerator[[k]], {k, 1, numeratorDegree}] /
        Product[n - denominator[[k]], {k, 1, denominatorDegree}];
    s = TimeConstrained[
        Sum[expr, {n, 1, Infinity}], maxTime
    ];

    AppendTo[expressions, {numerator, denominator}]
    AppendTo[answers, s]
    Print["Finished sum " <> ToString[i]]
]

(* Export everything! *)
Export[Directory[] <> "/data/expressions-" <> outputNaming <> ".txt", expressions]
Export[Directory[] <> "/data/answers-" <> outputNaming <> ".txt", answers]