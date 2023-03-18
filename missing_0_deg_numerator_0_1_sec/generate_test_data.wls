argv = Rest @ $ScriptCommandLine;
argStart = ToExpression[argv[[1]]];
argEnd = ToExpression[argv[[2]]];
(* LaunchKernels[4] *)

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


all2 = Range[0, 8]; all3 := Range[2, 10];
res3 := CartesianProduct[all2, all3];
res4 := DeleteCases[res3, list_ /; list[[2]] <= list[[1]] + 1];
maxTime = 0.1;


For[i = argStart, i < argEnd, i++,

    expressions = {};
    answers = {};

    current = res4[[i]];
    For[k = 1, k < 3000, k++,
        numerator = RandomChoice[fractionList, current[[1]]];
        denominator = RandomChoice[fractionListNoPositiveIntegers, current[[2]]];

        expr = 
            Product[n - numerator[[j]], {j, 1, current[[1]]}] /
            Product[n - denominator[[j]], {j, 1, current[[2]]}];
        s = TimeConstrained[
            Sum[expr, {n, 1, Infinity}], maxTime
        ];
        
        AppendTo[expressions, {numerator, denominator}];
        AppendTo[answers, s];
    ]

    Export[Directory[] <> "/data/expressions-" <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]] <> ".txt", expressions]
    Export[Directory[] <> "/data/answers-" <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]] <> ".txt", answers]
]

(*
For[i = argStart, i < argEnd, i++,
    expressions = {};
    answers = {};

    current = res4[[i]];
    Do[
        numerator = RandomChoice[fractionList, current[[1]]];
        denominator = RandomChoice[fractionListNoPositiveIntegers, current[[2]]];

        expr = 
            Product[n - numerator[[j]], {j, 1, current[[1]]}] /
            Product[n - denominator[[j]], {j, 1, current[[2]]}];
        s = TimeConstrained[
            Sum[expr, {n, 1, Infinity}], maxTime
        ];
        
        AppendTo[expressions, {numerator, denominator}];
        AppendTo[answers, s];,
        {k, 10}
    ]

    Export[Directory[] <> "/data/1-3-2023-no-parallel-expressions-" <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]] <> ".txt", expressions]
    Export[Directory[] <> "/data/1-3-2023-no-parallel-answers-" <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]] <> ".txt", answers]
] // AbsoluteTiming
*)
