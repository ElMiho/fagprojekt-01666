(* LaunchKernels[8]; *)
Needs["Combinatorica`"];

argv = Rest @ $ScriptCommandLine;
argStart = ToExpression[argv[[1]]];
argEnd = ToExpression[argv[[2]]];

expressionFolder = "random_expressions";
outputFolder = "random-answers-partition";
fileNameOutput = "random-answers-";
fileNameExpression = "rand_expressions-";
maxTime = 0.1;

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
size = 500; (* how many sums in parallel at a given time *)

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

    (* In case of unique sampling there is not 2000 partions (2000 * 500 = 10^6) *)
    rawFile = StringSplit[
        Import[
        Directory[] <> "/" <> expressionFolder <> "/" <> fileNameExpression 
            <> ToString[current[[1]]] <> "-" <> ToString[current[[2]]]  <> ".txt"
        ],
        "\n"
    ];

    file = ToExpression /@ f /@ rawFile;
    maxNumberOfPartitions = Ceiling[
        Length[file] / size
    ];

    start = If[partitionK < maxNumberOfPartitions, current];
    If[partitionK < maxNumberOfPartitions, Break[]];
];
Print["start: " <> ToString[start]];
Print["partitionK value: " <> ToString[partitionK]];
Print["(so starting with " <> ToString[partitionK + 1] <> ")"];

If[partitionK != -Infinity,
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
];

Print["[*] Evaluate the rest"];
test = {{0, 2}, {2, 7}};
(* Random expressions *)
startingPoint = Position[legalPolynomialFractions, start][[1]][[1]];
Print[startingPoint];
correctStartingPoint = If[partitionK == -Infinity, startingPoint, startingPoint + 1];
Print["(so starting with index " <> ToString[correctStartingPoint] <> " in legal poly frac)"];

target = legalPolynomialFractions;
For[j = correctStartingPoint, j <= Length[target], j++,
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