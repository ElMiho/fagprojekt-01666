import numpy as np
import sympy as sp

def token_input_space(minimal_value: int, maximal_value: int, rational_values = 0):
    '''
    Function:
       Input : 
           minimal_value - lowest value in the closed bounded interval: int
           maximal_value - higest value in the closed bounded interval: int 
       Output: 
           a vector of unique ralional values, with last and second last index being "/" and "#"  

    Rational_values guide:
    0 = a vector of unique ralional values, with last and second last index being "/" and "#"  
    1 = a vector of unique int values
    2 = a vector of unique ralional values
    3 = a vector of unique ralional values, without posetive int numbers
    '''

    # number of int's int the interval
    total_numbers = abs(maximal_value - minimal_value) + 1
    
    # alocates memery to an list with all entrys as minimal_value
    A = [sp.Rational(minimal_value, 1) for _ in range(total_numbers*total_numbers)]
    
    # begins adding new rational numbers starting from index 0
    row_index = 0

    if rational_values != 1 :
        # loop to create all rational values
        for i in range(minimal_value, maximal_value + 1):
            for j in range(minimal_value, maximal_value + 1):
                
                # dont want to divide with 0
                if (0 != j):
                    
                    # saves the rational number
                    # sp.rational() reduces all the rational numbers
                    A[row_index] = sp.Rational(i, j)
                    row_index += 1
        

    elif rational_values == 1 :
        for i in range(minimal_value, maximal_value+1):
            A[row_index] = i
            row_index += 1
     
    # sorts the array uning mergesort            
    merge_sort(A, 0, len(A)-1)
    
    # creates a new array with only unique rational number form (minimal_val - 2) to maximal value
    A = create_array_with_only_uniqe_values(A) 
                
    
    if rational_values == 1 or rational_values == 0:
        # append the special tokens "/" for when divide of polys, "#" symbol for empty.
        A.append("/")
        A.append("#")
    
    elif rational_values == 3:
        for i in range(1, 6):
            A.remove(i)
        
    
    # return the uniqe array
    return A

def file_of_input_to_tokenize_expression_to_file(filepath: str, new_file_name: str):
    """
    Function:
        Creates an file in the working dicretory, 
        
        NOTICE!!! if file named newFileName.txt already exist in working dicretory it will DELETE the file!!!!

    Parameters
    ----------
    filepath : TYPE {String}
        a file path to were the expressions.txt file is located    
    
    newFileName : TYPE {String}
        a file path to the new file.

    Returns
    -------
    None.

    """
    # opens the new file
    to_file = open(new_file_name, "w")
    
    # opens the reading file
    with open(filepath) as f:
        
        # while the inner loop dont break, becouse of next line dont exist
        while True:
            
            # Reads next line
            line = f.readline()
            
            # if next line dont exist break
            if not line:
                break
            
            # prints the string to a new file
            to_file.write(str(input_string_to_tokenize_expression(line.strip())) + "\n")

    # closed both files
    f.close()
    to_file.close()


def input_string_to_tokenize_expression(string: str):
    '''
    Function:
        takes input string created in matematica, and changes it into a list using only tokens from input space.

    Parameters
    ----------
    string : TYPE {string}
        a sting that need to be translated to token space

    Returns
    -------
    outputList : TYPE {List}
        a list that has been translated to token space

    '''    
    # i use these to save time by skip to next char in the string in special casses
    skip_next = False
    skip_next2 = False
    
    # if a number is negative 
    next_number_is_negative = False
    
    # empty list 
    output_list = []

    # enumerates the string then go thogh all the chars
    for i, char in enumerate(string):
        
        # Token is the token value for a given token, uses "$" to check if a new token space has been assigen
        token = "$"
        
        # block skips next two chars in sting 
        if skip_next2:
            skip_next = True
            skip_next2 = False
            continue
            
        # block skips next char in string
        elif skip_next:
            skip_next = False
            continue
        
        # test if it is the end of the sting, then breaks the string loop
        if char == '}' and string[i + 1] == '}':
            return output_list
        
        # gets the next two chars in the string
        else :
            next_char = string[i + 1]
            next_char2 = string[i + 2]
        
        # test if we are in the beginning af the sting
        if char == '{' and next_char == '{':
            
            # test if the numerator of the polys is empty then asigns token #
            if next_char2 == '}':
                token = "#"
            
            else :
                skip_next = True
                continue
                
        # test if the next rational number is negative
        elif char == '-':
            next_number_is_negative = True
            continue
          
        # test if the char is a of type int and next char is "/"  
        elif char.isdigit() and next_char == "/":
            skip_next2 = True

            # assigns Token the rational value depending on if its negative or not
            if next_number_is_negative:
                token = -sp.Rational(int(char), int(next_char2))
            else:
                token = sp.Rational(int(char), int(next_char2))
        
        # test if char is of type int and the next char is not "/"
        elif char.isdigit() and next_char != "/":
            
            # assigns Token the int value depending on if its negative or not
            if next_number_is_negative:
                token = -sp.Rational(int(char), 1)
            else:
                token = sp.Rational(int(char), 1)
          
        # test if "}" is followed by "," becouse then its the divistion beteeen the two polys and it is token /
        elif char == "}" and next_char == ",":
            token = "/"
       
        #print to file
        if token != "$":
            output_list.append(token)
            next_number_is_negative = False
            
def merge(arr, l, m, r):

    n1 = m - l + 1
    n2 = r - m

    # create temp arrays
    L = [0] * (n1)
    R = [0] * (n2)

    # Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]

    for j in range(0, n2):
        R[j] = arr[m + 1 + j]

    # Merge the temp arrays back into arr[l..r]
    i = 0	 # Initial index of first subarray
    j = 0	 # Initial index of second subarray
    k = l	 # Initial index of merged subarray

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # Copy the remaining elements of L[], if there
    # are any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    # Copy the remaining elements of R[], if there
    # are any
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1

def merge_sort(arr, l, r):
	if l < r:

		# Same as (l+r)//2, but avoids overflow for
		# large l and h
		m = l+(r-l)//2

		# Sort first and second halves
		merge_sort(arr, l, m)
		merge_sort(arr, m+1, r)
		merge(arr, l, m, r)
        
def create_array_with_only_uniqe_values(old_array):
    
    A = [0 for _ in range(count_uniqe(old_array))]
    
    row_idx = 0
    for i in range(0, len(old_array)):
        if (old_array[i] != old_array[i-1]):
            A[row_idx] = old_array[i]
            row_idx += 1
    
   
    return A  

def count_uniqe(A):
    '''

    Parameters
    ----------
    A : array

    Returns
    -------
    count : int
        number of uniqe numbers in an array

    '''
    
    if (len(A) != 0):
        count = 1
    else:
        count = 0
        
    for i in range(1, len(A)):
        if (A[i] != A[i-1]):
            count += 1
    return count


if __name__ == "__main__":
    A = token_input_space(-5,5, 1)
    print(A)
