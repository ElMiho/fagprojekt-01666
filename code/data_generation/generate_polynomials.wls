Needs["Combinatorica`"];

(* Generate all the numbers *)

listNegativeNumbers = Range[-5, 0];

listAll = Range[-5, 5];

numberOfPrimes = 10;

primes = Prime[Range[numberOfPrimes]];

numerator = RandomChoice[primes, 5];

denominator = SymmetricDifference[primes, numerator];

generateFractions[list_] = list[[1]] / list[[2]];

fractions = generateFractions /@ CartesianProduct[numerator, denominator];

totalListNumerator = DeleteDuplicates[Join[listAll, fractions]];

totalListDenominator = DeleteDuplicates[Join[listNegativeNumbers, fractions]];

(* Generate the polynomial representations *)

deg2Numerator = DeleteDuplicates[Sort /@ CartesianProduct[totalListNumerator, totalListNumerator]];

deg3Numerator = DeleteDuplicates[Sort /@ Flatten /@ CartesianProduct[deg2Numerator, totalListNumerator]];

deg2Denominator = DeleteDuplicates[Sort /@ CartesianProduct[totalListDenominator, totalListDenominator]];

deg3Denominator = DeleteDuplicates[Sort /@ Flatten /@ CartesianProduct[deg2Denominator, totalListDenominator]];

deg4Denominator = DeleteDuplicates[Sort /@ Flatten /@ CartesianProduct[deg3Denominator, totalListDenominator]];

Print["Generated all polynomials"]

(* Evaluate it all! 
Note that CartesianProduct is not used since there is approx 92*10^9 possibilities
so instead we pick them randomly
*)

numberOfSums = 10;

(* We will start by considering 2 deg over 4 degreee - this also ensures convergence *)

expressions = {};

answers = {};

For[i = 1, i <= numberOfSums, i++,
    polyNumerator = RandomChoice[deg2Numerator];
    polyDenominator = RandomChoice[deg4Denominator];
    Print[ToString[polyNumerator]];
    Print[ToString[polyDenominator]];
    expr = Product[n - polyNumerator[[k]], {k, 1, Length[polyNumerator]}] / Product[n - polyDenominator[[k]], {k, 1, Length[polyDenominator]}];
    s = Sum[expr, {n, 1, Infinity}];
    AppendTo[answers, s];
    AppendTo[expressions, {polyNumerator, polyDenominator}];
    Print["Finished sum " <> ToString[i]];
]

(* Export everything *)

Export[Directory[] <> "/data/expressions.txt", expressions];

Export[Directory[] <> "/data/answers.txt", answers];
