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

Print["[*] Loading file"];
rawFile = StringSplit[
    Import[
    Directory[] <> "/random_expressions/rand_expressions-" 
        <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree]  <> ".txt"
    ],
    "\n"
];

file := ToExpression /@ f /@ rawFile;

Print["[*] Evaluating "]

answers = {};
k = 10;
For[i = 1, i <= 3, i++,
    Print[i];
    data := file[[i]];
    Print["wait 1"];
    numerator := Part[data, Range[1, numeratorDegree]];
    denominator := Part[data, Range[1 + numeratorDegree, numeratorDegree + denominatorDegree]];
    Print["wait 2"];

    numeratorRoots = Part[fractionList, numerator];
    Print["wait 3"];
    denominatorRoots = Part[fractionListNoPositiveIntegers, denominator];
    Print["wait 4"];
    
    (* Generate expression based on lists of roots *)
    expr = Product[
        n - numeratorRoots[[p]], 
        {p, 1, numeratorDegree}
    ] / Product[
        n - denominatorRoots[[p]],
        {p, 1, denominatorDegree}
    ];
    Print[expr];
    
    (* Evaluate the sum *)
    s = TimeConstrained[
        Sum[expr, {n, 1, Infinity}], maxTime
    ];
    AppendTo[answers, {numeratorRoots, denominatorRoots, s}];
]

Export[Directory[] <> "/random-answers-random/random-answers-" <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree] <> ".txt", answers];

