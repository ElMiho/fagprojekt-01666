

import numpy as np
import sympy as sp





def tokenInputSpace(minimal_value: int, maximal_value: int):
    '''
    Function:
       Input : 
           an interval of type interval
       Output: 
           a vector of unique ralional values
           note: index 0 is saved for "/" token used for poly / poly, value = minimal_value - 2
           index 1 is saved for {empty} if 1 / poly, value = minimal_value - 1
                     
    '''
    
    
    
    # number of int's int the interval
    total_numbers = abs(maximal_value - minimal_value) + 1
    
    # + 2 is to make space for / and {empty} tokens
    A = sp.ones(total_numbers*total_numbers + 2, 1)*sp.Rational(minimal_value-1, 1)

    # changes index 0 to minimal_value - 2
    A[0] = sp.Rational(minimal_value - 2, 1)
    
    # begins adding new rational numbers starting from index 2
    row_index = 2

    for i in range(minimal_value, maximal_value + 1):
        for j in range(minimal_value, maximal_value + 1):
            
            if (0 != j):
                
                # saves the rational number
                # sp.rational() reduces all the rational numbers
                A[row_index] = sp.Rational(i, j)
                row_index += 1
                
    # sorts the array uning mergesort            
    mergeSort(A, 0, len(A)-1)
    
    # creates a new array with only unique rational number form (minimal_val - 2) to maximal value
    A = createArrayWithOnlyUniqeValues(A)
    
    
    
    # return the uniqe array
    return A



def stringToTokenIndex(string: str, aTokenInputSpace: float):
        
    '''
    Function:
        translates an input string to its tokenspace

    Parameters
    ----------
    string : TYPE {string}
        a sting that need to be translated to token space
        
    aTokenInputSpace : TYPE {floats}
        a array of type float, that consist of all the values corisponding to tokens
        idx 0 of the array need to corispodn to "/" token
        idx 1 of the array need to corispond to "{empty poly}" token

    Returns
    -------
    outputString : TYPE {string}
        a strig that has translated the input sting to token space

    '''
    
    A = aTokenInputSpace
    
    # i use these to save time and to skip to next int when it sees x/y
    skip_next = False
    skip_next2 = False
    
    # if a number is negative 
    next_number_is_negative = False
    
    outputString = ""
    for i, char in enumerate(string):
        
        # Token is the token index for a given rational number, when its -1 it to ensure it dont print
        TOKEN = -1
        
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
            outputString += "\n"
            return outputString
        
        # gets the next two chars in the string
        else :
            next_char = string[i + 1]
            next_char2 = string[i + 2]
        
        
        # test if we are in the beginning af the sting
        if char == '{' and next_char == '{':
            
            # test if the numerator of the polys is empty then asigns token 1
            if next_char2 == '}':
                TOKEN = 1
            
            else :
                skip_next = True
                continue
                
            
        # test if the next rational number is negative
        elif char == '-':
            next_number_is_negative = True
            continue
          
        # test if the char is a of type int and next char is "/" becouse 
        elif char.isdigit() and next_char == "/":
            skip_next2 = True
            if next_number_is_negative:
                TOKEN = BinarySearch(A, 0, len(A)-1, -int(char), int(next_char2))
            else:
                TOKEN = BinarySearch(A, 0, len(A)-1, int(char), int(next_char2))
        
        # test if char is of type int and the next char is not "/"
        elif char.isdigit() and next_char != "/":
            
            if next_number_is_negative:
                TOKEN = BinarySearch(A, 0, len(A)-1, -int(char), 1)
            else:
                TOKEN = BinarySearch(A, 0, len(A)-1, int(char), 1)
          
        # test if "}" is followed by "," becouse then its the divistion beteeen the two polys and it is token 0
        elif char == "}" and next_char == ",":
            TOKEN = 0
       
        #print to file
        if TOKEN != -1:
            outputString += f"{TOKEN},"
            next_number_is_negative = False
        


def fileToTokenIndex(filepath: str, min_value: int, max_value: int, newFileName: str):
    """
    Function:
        Creates an file in the working dicretory
        
        NOTICE!!! if file named tokenFile.txt already exist in working dicretory it will DELETE the file!!!!

    Parameters
    ----------
    filepath : TYPE {String}
        a file path to were the expressions.txt file is located    
    
    min_value : TYPE {int}
        the minial value included in the input token space
        
    max_value : TYPE
        the maximal value included in the input token space

    Returns
    -------
    None.

    """
    
    
    
    toFile = open(newFileName, "w")
    A = tokenInputSpace(min_value, max_value)

    with open(filepath) as f:
        
        
        while True:
            
            # Reads next line
            line = f.readline()
            
            # if next line dont exist break
            if not line:
                break
            
            # prints the string to a new file
            toFile.write(stringToTokenIndex(line.strip(), A))
                    
    f.close()
    toFile.close()
    
    

def createArrayWithOnlyUniqeValues(oldArray):
    
    A = sp.zeros(countUniqe(oldArray), 1)
    
    rowIdx = 0
    for i in range(0, len(oldArray)):
        if (oldArray[i] != oldArray[i-1]):
            A[rowIdx] = oldArray[i]
            rowIdx += 1
    
   
    return A  

def countUniqe(A):
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

def mergeSort(arr, l, r):
	if l < r:

		# Same as (l+r)//2, but avoids overflow for
		# large l and h
		m = l+(r-l)//2

		# Sort first and second halves
		mergeSort(arr, l, m)
		mergeSort(arr, m+1, r)
		merge(arr, l, m, r)

def BinarySearch(A, i, j, numinator, dominator):
       
    if (j < i):
        return -1
    
    m = int((i+j)//2)
    
    if A[m] == sp.Rational(numinator, dominator):
        return m
        
    elif A[m] < sp.Rational(numinator, dominator):
        return BinarySearch(A, m+1, j, numinator, dominator)
    else:
        return BinarySearch(A, i, m-1, numinator, dominator)
        
   



def sympyWhere(A, numinator, dominator):
    rational_number = sp.Rational(numinator, dominator)
    
    for i in range(0, len(A)):
        if (A[i] == rational_number):
            return i
        
    return -1

A = tokenInputSpace(-5,5)