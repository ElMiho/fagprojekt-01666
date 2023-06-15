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



def evaluate_sum(numerator_roots, denominator_roots):
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
    evaluate_sum_f = session.function(wlexpr('evaluateSum'))
    
    return evaluate_sum_f(numerator_roots, denominator_roots)
    
    
    

# Always call this when finished!
def close_session():
    session.terminate()

# poor mans test cases
if __name__ == '__main__':
    print(evaluate_sum([], [0, 0]))
    
    test_input_token = ['#', '/', '0', '0']
    numerator_roots, denominator_roots = input_to_lists(test_input_token)
    
=======
        numerator_degree = 0
        numerator_roots = []
    else:
        numerator_degree = len(numerator)
        numerator_roots = numerator

    denominator_degree = len(denominator)
    denominator_roots = denominator

    return numerator_degree, denominator_degree, numerator_roots, denominator_roots

def evaluate_sum(
        numerator_degree: int, 
        denominator_degree: int, 
        numerator_roots: list, 
        denominator_roots: list) -> str:
    # just some checks
    assert len(numerator_roots) == numerator_degree
    assert len(denominator_roots) == denominator_degree
    
    numerator_roots_string = ' '.join(str(x) for x in numerator_roots)
    denominator_roots_string = ' '.join(str(x) for x in denominator_roots)

    if numerator_roots_string == '':
        roots_string = denominator_roots_string
    else:
        roots_string = numerator_roots_string + ' ' + denominator_roots_string

    res = os.popen(
        f"""
        wolframscript --file evaluate_sum_from_roots_cmd.wls "{numerator_degree} {denominator_degree} {roots_string}"
        """
    ).read()

    return res

# poor mans test cases
if __name__ == '__main__':
    print(evaluate_sum(0, 2, [], [0, 0]))
    
    test_input_token = ['#', '/', '0', '0']
    numerator_degree, denominator_degree, numerator_roots, denominator_roots = input_to_lists(test_input_token)
    
    print(numerator_degree)
    print(denominator_degree)
>>>>>>> 14a9f20fa3375541d800bf108f7cc72193f370dc
    print(numerator_roots)
    print(denominator_roots)

    print(evaluate_sum(
<<<<<<< HEAD
        numerator_roots, denominator_roots
    ))
   
    close_session()
=======
        numerator_degree, denominator_degree, numerator_roots, denominator_roots
    ))
>>>>>>> 14a9f20fa3375541d800bf108f7cc72193f370dc
