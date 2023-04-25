(* LaunchKernels[5] *)
argv = Rest @ $ScriptCommandLine;
folder = argv[[1]];

str = Directory[] <> "/" <> ToString[folder];
Print[str];
files = FileNames[str <> "/*"];

outputFiles = FileNames[
    ToString[folder] <> "_simplified" <> "/*"
];
Print["outputFiles"];
Print[outputFiles];

time = AbsoluteTiming[For[i = 1, i <= Length[files], i++,
    fileName = files[[i]];
    Print["file: " <> fileName];

    l = StringSplit[fileName, "/"][[-2]] <> "_simplified/" <> StringSplit[fileName, "/"][[-1]];
    Print[l];
    r = !MemberQ[outputFiles, l];
    Print[r]

    If[r,
        rawFile = StringSplit[
        Import[fileName], "\n"
        ];
        simplifiedAnswers = {};
        SetSharedVariable[simplifiedAnswers];

        actualFileName = StringSplit[fileName, "/"][[-1]];
        outputString = ToString[folder] <> "_simplified/" <> ToString[actualFileName];
        
        ParallelDo[
            line = ToExpression[rawFile[[k]]];
            answer = line[[3]];
            newAnswer = Simplify[answer];
            AppendTo[simplifiedAnswers, {line[[1]], line[[2]], newAnswer}];,
            {k, 1, Length[rawFile]}
        ];

        Print[outputString];
        Export[outputString, simplifiedAnswers];
    ];
];
];
Print["time: "];
Print[time];