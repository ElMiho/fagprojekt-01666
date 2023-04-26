argv = Rest @ $ScriptCommandLine;
string = argv[[1]];

res = ToExpression[
    StringSplit[string, " "]
];

(* Print[res]; *)

numeratorDegree = res[[1]];
denominatorDegree = res[[2]];

(* Print["numerator"]; *)
numerator = {};
For[i = 3, i < 3 + numeratorDegree, i++,
    (* Print[res[[i]]]; *)
    AppendTo[numerator, res[[i]]];
];

(* Print["denominator"]; *)
denominator = {};
For[i = 3 + numeratorDegree, i < 3 + numeratorDegree + denominatorDegree, i++,
    (* Print[res[[i]]]; *)
    AppendTo[denominator, res[[i]]];
];

expr = Product[
    n - numerator[[p]], {p, 1, numeratorDegree}
] / Product[
    n - denominator[[p]], {p, 1, denominatorDegree}
];

(* Print[expr]; *)

s = Sum[
    expr, {n, 1, Infinity}
];

Print[s];