LaunchKernels[8];
Needs["Combinatorica`"];

maxTime = 0.1;

numeratorDegree = Range[0, 8]; 
denominatorDegree = Range[2, 10];
allPossiblePolynomialFractions = CartesianProduct[numeratorDegree, denominatorDegree];
legalPolynomialFractions = DeleteCases[
    allPossiblePolynomialFractions, 
    list_ /; list[[1]] >= list[[2]] - 1
];

Print[legalPolynomialFractions];

allRoots = Range[-10, 10];
noPositiveRoots = DeleteCases[
    allRoots,
    x_ /; x > 0
];

Print[allRoots];
Print["\n"];
Print[noPositiveRoots]

For[i = 1, i <= Length[legalPolynomialFractions], i++,
    numDegree = legalPolynomialFractions[[i]][[1]];
    denomDegree = legalPolynomialFractions[[i]][[2]];

    Print["current: "];
    Print[ToString[numDegree] <> "/" <> ToString[denomDegree]];

    For[l = 1, l <= 10, l++,
        answers = {};
        SetSharedVariable[answers];
        
        ParallelDo[
            numRoots = RandomChoice[allRoots, numDegree];
            denomRoots = RandomChoice[noPositiveRoots, denomDegree];

            expr = Product[
                n - numRoots[[p]],
                {p, 1, numDegree}
            ] / Product[
                n - denomRoots[[p]],
                {p, 1, denomDegree}
            ];
            (* Print[expr]; *)

            s = TimeConstrained[
                Sum[expr, {n, 1, Infinity}], maxTime
            ];
            AppendTo[answers, s];,
            {k, 1, 10}
        ];

        Export[
            Directory[] <> "/data_for_abortion_rate/" <> ToString[numDegree] <> "-" <> ToString[denomDegree] <> "-partition" <> ToString[l] <> ".txt", 
            answers
        ];
    ]
];