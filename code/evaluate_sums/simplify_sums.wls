argv = Rest @ $ScriptCommandLine;
folder = argv[[1]];

str = Directory[] <> "/" <> ToString[folder];
Print[str];
files = FileNames[str <> "/*"];

time = AbsoluteTiming[For[i = 1, i <= Length[files], i++,
    fileName = files[[i]];
    Print["file: " <> fileName];
    rawFile = StringSplit[
        Import[fileName], "\n"
    ];
    simplifiedAnswers = {};
    SetSharedVariable[simplifiedAnswers];
    
    ParallelDo[
        line = ToExpression[rawFile[[k]]];
        answer = line[[3]];
        newAnswer = Simplify[answer];
        AppendTo[simplifiedAnswers, {line[[1]], line[[2]], newAnswer}];,
        {k, 1, Length[rawFile]}
    ];

    actualFileName = StringSplit[fileName, "/"][[-1]];
    
    outputString = ToString[folder] <> "_simplified/" <> ToString[actualFileName] <> "_simplified.txt";
    Print[outputString];
    Export[outputString, simplifiedAnswers];
];
];
Print["time: "];
Print[time];