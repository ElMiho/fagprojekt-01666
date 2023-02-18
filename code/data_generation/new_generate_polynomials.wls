Needs["Combinatorica`"];
all = Range[-5,5];

(* Generate all the fractions and integers *)

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

Print["fraction list"]
Print[fractionList]

Print["fraction list - no positive integers"]
Print[fractionListNoPositiveIntegers]

(* Polynomial generation
General idea:
    1. Pick a random integer, d, representing the degree of the denominator
    2. Pick another random integer (degree of numerator), k, which satisfies k <= d - 2
    3. Pick a corresponding amount of integers for each polynoimal
*)
numberOfSums = 10;
maxDegreeDenominator = 20;
maxTime = 7; (* seconds *)
minutes = N[7 * 10 / 60];
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
Export[Directory[] <> "/data/expressions.txt", expressions]
Export[Directory[] <> "/data/answers.txt", answers]