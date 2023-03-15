LaunchKernels[10];
Needs["Combinatorica`"];

(* Create list of roots *)
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

(* Generate legal polynomials *)
(* To ensure convergence we look at P(n)/Q(n) such that deg(P) <= deg(Q(n)) - 2 *)
numeratorDegree = Range[0, 8]; 
denominatorDegree := Range[2, 10];
allPossiblePolynomialFractions := CartesianProduct[numeratorDegree, denominatorDegree];
legalPolynomialFractions := DeleteCases[
    allPossiblePolynomialFractions, 
    list_ /; list[[1]] >= list[[2]] - 1
];

maxTime = 7;

f[l_] := StringSplit[l, ","];

test = {{2, 7}};
(* Random expressions *)
For[j = 1, j < 2, j++,
    numeratorDenominator = test[[j]];
    numeratorDegree = numeratorDenominator[[1]];
    denominatorDegree = numeratorDenominator[[2]];

    rawFile = StringSplit[
        Import[
        Directory[] <> "/.././sampling/random_expressions/rand_expressions-" 
            <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree]  <> ".txt"
        ],
        "\n"
    ];
    
    file := ToExpression /@ f /@ rawFile;

    answers = {};
    SetSharedVariable[answers];

    ParallelDo[
        data = file[[i]];

        numerator = Part[data, Range[1, numeratorDegree]];
        denominator = Part[data, Range[1 + numeratorDegree, numeratorDegree + denominatorDegree]];
        
        numeratorRoots = Part[fractionList, numerator];
        denominatorRoots = Part[fractionListNoPositiveIntegers, denominator];

        (* Generate expression based on lists of roots *)
        expr = Product[
            n-numeratorRoots[[k]], 
            {k, 1, numeratorDegree}
        ] / Product[
            n-denominatorRoots[[k]],
            {k, 1, denominatorDegree}
        ];
        
        (* Evaluate the sum *)
        s = TimeConstrained[
            Sum[expr, {n, 1, Infinity}], maxTime
        ];
        AppendTo[answers, s];,
        {i, 1, Length[rawFile]}
    ]

    Export[Directory[] <> "/data/random-answers-" <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree] <> ".txt", answers]
]
