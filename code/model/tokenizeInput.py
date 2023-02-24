

import numpy as np
  


def tokenInputSpace(minimal_value: int, maximal_value: int):
    '''
    Function:
       Input : 
           an interval of type interval
       Output: 
           a vector of unique ralional values type float
           note: index 0 is saved for "/" token used for poly / poly, value = minimal_value - 2
           index 1 is saved for {empty} if 1 / poly, value = minimal_value - 1
                     
    '''
    
    # number of int + 2 is to make space for / and {empty} tokens
    total_numbers = abs(maximal_value - minimal_value) + 2
    
    # initilizer array of max size
    A = np.full((total_numbers*total_numbers + 1, 3), minimal_value - 1, dtype = float )
    row_index = 0
    
    for i in range(minimal_value, maximal_value + 1):
        for j in range(minimal_value, maximal_value + 1):
            
            if (0 != j):
                # saves all rational numbers (it is not need but nice to have)
                A[row_index, 0] = i
                A[row_index, 1] = j
                
                # saves the rational number as a float
                A[row_index, 2] = i/j
                row_index += 1
     
    # creates an unipe value in the list
    A[row_index, 2] = minimal_value - 2       
    
    # sorts the matrix acroding to 3 row
    A = A[A[:, 2].argsort(kind = 'mergesort')]
    
    # removes all dublicated values
    all_rational_numbers_in_interval_plus2 = np.unique(A[:, 2], axis = 0)
            
    return all_rational_numbers_in_interval_plus2



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
                TOKEN = int(np.where(A == -int(char)/int(next_char2))[0])
            else:
                TOKEN = int(np.where(A == int(char)/int(next_char2))[0])
        
        # test if char is of type int and the next char is not "/"
        elif char.isdigit() and next_char != "/":
            
            if next_number_is_negative:
                TOKEN = int(np.where(A == -int(char))[0])
            else:
                TOKEN = int(np.where(A == int(char))[0])
          
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






        
        