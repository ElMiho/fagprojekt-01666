import os
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
session = WolframLanguageSession()

def input_to_lists(list_of_input_tokens):
    
    idx = list_of_input_tokens.index('/')
    numerator = list_of_input_tokens[0:idx]
    denominator = list_of_input_tokens[idx+1:]

    if numerator == ['#']:
        numerator_roots = []
    else:
        numerator_roots = [int(x) for x in numerator]

    denominator_roots = [int(x) for x in denominator]

    return numerator_roots, denominator_roots

session.evaluate('''
    evaluateSum[list1_, list2_] := ToString[
        Sum[
            Product[
                n - list1[[p]], {p, 1, Length[list1]}
            ] / Product[
                n - list2[[p]], {p, 1, Length[list2]}
            ], {n, 1, Infinity}
        ], InputForm
    ]
''')
                 
evaluate_sum = session.function(wlexpr('evaluateSum'))

# Always call this when finished!
def close_session():
    session.terminate()

# poor mans test cases
if __name__ == '__main__':
    print(evaluate_sum([], [0, 0]))
    
    test_input_token = ['#', '/', '0', '0']
    numerator_roots, denominator_roots = input_to_lists(test_input_token)
    
    print(numerator_roots)
    print(denominator_roots)

    print(evaluate_sum(
        numerator_roots, denominator_roots
    ))

    close_session()