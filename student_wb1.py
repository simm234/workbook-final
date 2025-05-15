from approvedimports import *

def exhaustive_search_4tumblers(puzzle: CombinationProblem) -> list:
    """simple brute-force search method that tries every combination until
    it finds the answer to a 4-digit combination lock puzzle.
    """

    # check that the lock has the expected number of digits
    assert puzzle.numdecisions == 4, "this code only works for 4 digits"

    # create an empty candidate solution
    my_attempt = CandidateSolution()

# insert your code below here
    for attempt_digit1 in puzzle.value_set:   #in the for loop using the variable name attempt_digit1, attempt_digit2, attempt_digit3, attempt_digit4 
        for attempt_digit2 in puzzle.value_set: #using nested for loop for each of the variable so loop runs in all possible value
            for attempt_digit3 in puzzle.value_set:
                for attempt_digit4 in puzzle.value_set:
                    my_attempt.variable_values = [attempt_digit1, attempt_digit2, attempt_digit3, attempt_digit4]  #setting all the varibles in one particular variable 
                    try:
                        result = puzzle.evaluate(my_attempt.variable_values)  #calulating final evaluation of the values and storing in variable named result 
                        if result == 1:
                            return my_attempt.variable_values  #using if statement to return the correct result
                    except Exception: #using exception case for error handling so that the program doesnot crash
                        pass
#insert your code above here
#should never get here
    return [-1, -1, -1, -1]


import numpy as np
def get_names(namearray: np.ndarray) -> list:
    family_names = []
    # ====> insert your code below here
    for i in range(namearray.shape[0]):  # using shape[0] to get the number of rows
        family_charecters = namearray[i, -6:]  # using the variable family_charecters to get the value of the last six charecter by slicing the array
        family_join = ''.join(family_charecters)  # to join the family_charecters into a string using family_join
        family_names.append(family_join)  #not striping spaces as the problem states not to do so


     # <==== insert your code above here
    return family_names 




def check_sudoku_array(attempt: np.ndarray) -> int:
    tests_passed = 0
    slices = []  # this will be a list of numpy arrays

    # ====> insert your code below here


    # using assertion to check if array is of 2 dimention and has 9 size
    assert attempt.shape == (9, 9), "The Sudoku array must be 9x9."

    # Adding the  rows to slices
    for i in range(9):
        slices.append(attempt[i, :])  # Each row 


    # Adding the  columns to slices
    for j in range(9):
        slices.append(attempt[:, j])  # Each column


    # Adding  3x3 sub-squares to slices
    for row in range(0, 9, 3):  # Starting rows of 3x3 sub_SQUARES  
        for col in range(0, 9, 3):  # Starting columns of 3x3 sub-squares 
            slices.append(attempt[row:row+3, col:col+3].flatten())  # Flattening the 3x3 sub-square into a 1D array


    #Checking if all slices are unique or not
    for slice in slices:
        if len(np.unique(slice)) == 9:  
        # to get particular unique value from the array
            tests_passed += 1 

        # the increment of the value of tests_passed is done 

    # <==== insert your code above here
    # return count of tests passed
    return tests_passed
