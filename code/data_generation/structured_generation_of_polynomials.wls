argv = Rest @ $ScriptCommandLine;
argStart = ToExpression[argv[[1]]];
argEnd = ToExpression[argv[[2]]];
LaunchKernels[4]

listToInteger[list_, base_] := Module[
    {res = 0},
    For[i = 0, i < Length[list], i++,
        res = res + list[[i + 1]]*base^i;
    ];
    res
]

integerToList[n_, sumDeg_, base_] := Module[
    {res = {}, var = 0, k = n},
    For[i = 1, i <= sumDeg, i++,
        var = Mod[k, base^i];
        AppendTo[res, var/(base^(i - 1))];
        k = k - var;
    ];
    res
]
