from sympy import I, oo, Sum, exp, pi
from sympy.abc import n
import sympy as sp
import itertools
import random
import json
import sys

# for time constrained operations
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def sum_from_roots(numerator_roots, denominator_roots, max_time):
    poly_1 = sp.prod([n - root for root in numerator_roots])
    poly_2 = sp.prod([n - root for root in denominator_roots])
    expr = poly_1 / poly_2

    try:
        with time_limit(max_time):
            return Sum(expr, (n, 1, oo)).doit()
    except TimeoutException as e:
        return "aborted"

legal_poly_fraction = [(x, y) for x in range(0, 7+1) for y in range(2, 10+1) if y >= x + 2]

roots_tuple = list(
    itertools.product(list(range(-5, 5+1)), list(range(-5, 5+1)))
)

roots = list(set([sp.Rational(x, y) for x, y in roots_tuple if y != 0]))
positive_roots = [x for x in roots if x <= 0 or x not in range(1, 5+1)]

number_of_sums_per_category = 100
max_time = 7

category = int(sys.argv[1])

for numerator_degree, denominator_degree in [legal_poly_fraction[category]]:
    print(numerator_degree, denominator_degree)
    answers = []
    for i in range(number_of_sums_per_category):
        print(i)
        numerator_roots = random.choices(roots, k=numerator_degree)
        denominator_roots = random.choices(positive_roots, k=denominator_degree)

        sum = sum_from_roots(numerator_roots, denominator_roots, max_time)
        res = str(sum)

        answers.append(
            ([str(v) for v in numerator_roots], [str(v) for v in denominator_roots], res)
        )

    with open(f"sympy-data/{numerator_degree}-{denominator_degree}.txt", "w") as f:
        json.dump(answers, f)
        f.close()

