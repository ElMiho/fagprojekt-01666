Needs["Combinatorica`"];

argv = Rest @ $ScriptCommandLine;
argStart = ToExpression[argv[[1]]];
argEnd = ToExpression[argv[[2]]];
kernels = ToExpression[argv[[3]]];
size = ToExpression[argv[[4]]];
total = ToExpression[argv[[5]]];

LaunchKernels[kernels];

expressionFolder = "random_expressions_benchmark";
outputFolder = "random-answers-partition-benchmark";
fileNameOutput = "random-answers-";
fileNameExpression = "rand_expressions-";
maxTime = 7;
(* size = 500 *)
numberOfPartitions = Ceiling[total/size];

(* Create list of roots *)
all = Range[-5,5];

generateFractions[list_] := list[[1]] / list[[2]];
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
denominatorDegree = Range[2, 10];
allPossiblePolynomialFractions = CartesianProduct[numeratorDegree, denominatorDegree];
legalPolynomialFractions = DeleteCases[
    allPossiblePolynomialFractions,
    list_ /; list[[1]] >= list[[2]] - 1
];

Print["[*] Restricting legal poly frac using argStart and argEnd"];
legalPolynomialFractions = Part[
    legalPolynomialFractions, argStart;;argEnd;;1
];
Print[legalPolynomialFractions];

f[l_] := StringSplit[l, ","];

(* Checking with existing files *)
Print["[*] Checking with existing files"];
stringToPartition[target_][x_] := StringTake[
    x, StringLength[target]+1;;StringLength[x]-4;;1
];

allFiles = FileNames[outputFolder <> "/*"];
(* note that this assumes there is 10^6 in each file *)
Print["length of legal poly frac: " <>
   ToString[Length[legalPolynomialFractions]]];
Print["[*] Starting loop"];

For[i = 1, i <= Length[legalPolynomialFractions], i++,
    current = legalPolynomialFractions[[i]];
    currentString = outputFolder <> "/" <> fileNameOutput <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]] <> "-partition-";
    currentFiles = Select[allFiles, StringContainsQ[#, currentString] &];
    partitionK = Max[
        ToExpression /@ stringToPartition[currentString] /@ currentFiles
    ];
    start = If[partitionK < numberOfPartitions, current];
    If[partitionK < numberOfPartitions, Break[]];
];
Print["start: " <> ToString[start]];
Print["partitionK value: " <> ToString[partitionK]];
Print["(so starting with " <> ToString[partitionK + 1] <> ")"];

Print["[*] Evaluating remaining from existing files"];
rawFile = StringSplit[
    Import[
    Directory[] <> "/" <> expressionFolder <> "/" <> fileNameExpression
        <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]]  <> ".txt"
    ],
    "\n"
];

file = ToExpression /@ f /@ rawFile;
offset = size;
partitionFile = Partition[
    file, size, offset, {1, 1}, {}
];

For[k = partitionK + 1, k <= Length[partitionFile], k++,
    Print["Partition: " <> ToString[k]];
    answers = {};
    SetSharedVariable[answers];

    currentPartition = partitionFile[[k]];

    ParallelDo[
        data = currentPartition[[i]];
        numerator = Part[data, Range[1, current[[1]]]];
        denominator = Part[data, Range[1 + current[[1]], current[[1]] + current[[2]]]];

        numeratorRoots = Part[fractionList, numerator];
        denominatorRoots = Part[fractionListNoPositiveIntegers, denominator];

        (* Generate expression based on lists of roots *)
        expr = Product[
            n - numeratorRoots[[p]],
            {p, 1, current[[1]]}
        ] / Product[
            n - denominatorRoots[[p]],
            {p, 1, current[[2]]}
        ];

        (* Evaluate the sum *)
        s = TimeConstrained[
            Sum[expr, {n, 1, Infinity}], maxTime
        ];
        AppendTo[answers, {numeratorRoots, denominatorRoots, s}];,
        {i, 1, Length[currentPartition]}
    ];

    Print["Finished with " <> ToString[k]];

    Export[Directory[] <> "/" <> outputFolder <> "/" <> fileNameOutput <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]] <> "-partition-" <> ToString[k] <> ".txt", answers]
];

Print["[*] Evaluate the rest"];
test = {{0, 2}, {2, 7}};
(* Random expressions *)
startingPoint = Position[legalPolynomialFractions, start][[1]][[1]];
Print[startingPoint];
Print["(so starting with index " <> ToString[startingPoint + 1] <> " in legal poly frac)"];

target = legalPolynomialFractions;
For[j = startingPoint + 1, j <= Length[target], j++,
    Print["index: " <> ToString[j]];
    numeratorDenominator = target[[j]];
    numeratorDegree = numeratorDenominator[[1]];
    denominatorDegree = numeratorDenominator[[2]];

    rawFile = StringSplit[
        Import[
        Directory[] <> "/" <> expressionFolder <> "/" <> fileNameExpression
            <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree]  <> ".txt"
        ],
        "\n"
    ];

    file = ToExpression /@ f /@ rawFile;

    offset = size;
    partitionFile = Partition[
        file, size, offset, {1, 1}, {}
    ];

    For[k = 1, k <= Length[partitionFile], k++,
        Print["Partition: " <> ToString[k]];
        answers = {};
        SetSharedVariable[answers];

        currentPartition = partitionFile[[k]];

        ParallelDo[
            data = currentPartition[[i]];
            numerator = Part[data, Range[1, numeratorDegree]];
            denominator = Part[data, Range[1 + numeratorDegree, numeratorDegree + denominatorDegree]];

            numeratorRoots = Part[fractionList, numerator];
            denominatorRoots = Part[fractionListNoPositiveIntegers, denominator];

            (* Generate expression based on lists of roots *)
            expr = Product[
                n - numeratorRoots[[p]],
                {p, 1, numeratorDegree}
            ] / Product[
                n - denominatorRoots[[p]],
                {p, 1, denominatorDegree}
            ];

            (* Evaluate the sum *)
            s = TimeConstrained[
                Sum[expr, {n, 1, Infinity}], maxTime
            ];
            AppendTo[answers, {numeratorRoots, denominatorRoots, s}];,
            {i, 1, Length[currentPartition]}
        ];
        Print["Finished with " <> ToString[k]];

        Export[Directory[] <> "/" <> outputFolder <> "/" <> fileNameOutput <> ToString[numeratorDegree] <> "-" <> ToString[denominatorDegree] <> "-partition-" <> ToString[k] <> ".txt", answers]
    ]
];