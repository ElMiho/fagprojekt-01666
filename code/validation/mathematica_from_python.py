import os

def evaluate_sum(numerator_degree: int, denominator_degree: int, numerator_roots: list, denominator_roots: list) -> str:
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

if __name__ == '__main__':
    print(evaluate_sum(0, 2, [], [0, 0]))