argv = Rest @ $ScriptCommandLine;
numeratorDegree = ToExpression[argv[[1]]];
denominatorDegree = ToExpression[argv[[2]]];

numeratorRoots = {};
denominatorRoots = {};

Print["numerator: "];
For[i = 3, i < 3 + numeratorDegree, i++,
    r = ToExpression[argv[[i]]];
    AppendTo[numeratorRoots, r];
]
Print[numeratorRoots];

Print["denominator: "];
For[i = 3 + numeratorDegree, i < 3 + numeratorDegree + denominatorDegree, i++,
    r = ToExpression[argv[[i]]];
    AppendTo[denominatorRoots, r];
]
Print[denominatorRoots];

expr = Product[
    n - numeratorRoots[[p]],
    {p, 1, numeratorDegree}
] / Product[
    n - denominatorRoots[[p]],
    {p, 1, denominatorDegree}
];
Print[expr];

s = Sum[
    expr, {n, 1, Infinity}
];
Print[s];