from validation.validation_medthods import extend_sum, extend_sum2
from validation.validation_medthods import posible_degrees
from validation.validation_medthods import random_list_of_nuerator_and_denominator
from validation.validation_medthods import neural_network_validation
from model.equation_interpreter import Equation
from validation.validation_medthods import TED_of_list_postfix_eq_as_tokens
import pickle
from validation.TED import graph_from_postfix, TreeEditDistance


  
#%%


num_d = 0
den_d = 2


filename = f"validation/save_data/mul_{num_d}_{den_d}.pkl"
degree_vector = posible_degrees(10)
total_counter = [0 for _ in degree_vector]
found_distance = [[] for _ in degree_vector]


loops = 1
for _ in range(0,10): #HUSK AT SÆT OP
    vector_idx = degree_vector.index([num_d, den_d])
    if loops % 5 == 0:
        print(f"{loops} completed -- status : {len(found_distance[vector_idx])}")
        saved_variables = {'found_distance': found_distance}
        with open(filename, 'wb+') as f:
            pickle.dump(saved_variables, f)
    loops += 1
    
    
    try:
        roots_list = random_list_of_nuerator_and_denominator([-5,5], True, False, [num_d, den_d])
        nn_out = neural_network_validation(roots_list)
        boole = Equation(nn_out, "postfix").is_valid()
        if not boole:
            print("false")
            continue
        _, correct_tree = graph_from_postfix(nn_out)
    except Exception:
        print("false")
        continue
        
    extend_sum = extend_sum2(roots_list)
    
    nn_out_es = [neural_network_validation(roots) for roots in extend_sum] 
    nn_out_valid = []
    
    for i in range(0, len(nn_out_es)):
        try:
            boole = Equation(nn_out_es[i], "postfix").is_valid()
        except Exception:
            boole = False
        
        if boole:
            nn_out_valid.append(nn_out_es[i])
    
    
         
    if  len(nn_out_valid) > 1: #gider ikke have den med hvis det er kun er en
        #print(f"found multiple!!! {len(nn_out_valid)}")
        dist = 0
        count = 0
        for tokens in nn_out_valid:
            try:
                _, predicted_tree = graph_from_postfix(tokens)
                ted = TreeEditDistance().calculate(predicted_tree, correct_tree)
                dist += ted[0]
                count += 1
            except Exception:
                print("no tree")
                None
        
        if count > 1:
            found_distance[vector_idx].append(dist)
        
    loops += 1
print("done")
