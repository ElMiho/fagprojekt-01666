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

maxTime := 0.1;

f[l_] := StringSplit[l, ","];

target = {0, 2};
numeratorDenominator = target;
numeratorDegree = numeratorDenominator[[1]];
denominatorDegree = numeratorDenominator[[2]];

Print["[*] Loading files"];
rawFileNumerator = StringSplit[
    Import[
    Directory[] <> "/random_expressions_roots/rand_roots_num_expressions-" 
        <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree]  <> ".txt"
    ],
    "\n"
];

rawFileDenominator = StringSplit[
    Import[
    Directory[] <> "/random_expressions_roots/rand_roots_den_expressions-" 
        <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree]  <> ".txt"
    ],
    "\n"
];

fileNumerator := ToExpression /@ f /@ rawFileNumerator;
fileDenominator := ToExpression /@ f /@ rawFileDenominator;

Print["[*] Evaluating "]

answers = {};
k = 10;
For[i = 1, i <= k, i++,
    Print[i];
    rootsNumerator := fileNumerator[[i]];
    rootsDenominator := fileDenominator[[i]]];
    
    (* Generate expression based on lists of roots *)
    expr = Product[
        n - rootsNumerator[[p]], 
        {p, 1, numeratorDegree}
    ] / Product[
        n - rootsDenominator[[p]],
        {p, 1, denominatorDegree}
    ];
    Print[expr];
    
    (* Evaluate the sum *)
    s = TimeConstrained[
        Sum[expr, {n, 1, Infinity}], maxTime
    ];
    AppendTo[answers, {rootsNumerator, rootsDenominator, s}];
]

Export[Directory[] <> "/random-answers-random/random-answers-" <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree] <> ".txt", answers];

