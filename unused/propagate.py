'''
This program calculates the quantum state of a photon after n iterations through the Grover multiport.
B represents the phase shift, e^(i*omega*tau), obtained by propogation along path b.
C represents the diffusion and reflection, r(z, w)*e^(i*phi(z, w)), accummulated by passing through the sample path c.

The program returns the coefficients of each quantum state |b>, |c>, and |d> in terms of B and C.
'''


# add a zero column in first place
def multiply_by_b(polynomial):
    degree_of_c = len(polynomial[0])
    zero_row = [0 for _ in range(degree_of_c)]
    polynomial.insert(0, zero_row)
    return polynomial

# add a zero row in first place
def multiply_by_c(polynomial):
    for row in polynomial:
        row.insert(0, 0)
    return polynomial

# inserts zeroes at the ends to make a matrix fit specified dimensions
def pad_matrix(polynomial, rows, columns):
    degree_of_c = len(polynomial[0])
    degree_of_b = len(polynomial)

    column_expansion = columns - degree_of_c # number of new zero columns needed
    row_expansion = rows - degree_of_b # number of new zero rows needed

    for _ in range(column_expansion):
        for row in polynomial:
            row.append(0)

    zero_row = [0 for _ in range(columns)]
    for _ in range(row_expansion):
        polynomial.append(zero_row)

def add_pair_of_polynomials(polynomial1, polynomial2):
    # make sure both polynomials have the same dimensions
    degree_of_c1 = len(polynomial1[0])
    degree_of_b1 = len(polynomial1)
    degree_of_c2 = len(polynomial2[0])
    degree_of_b2 = len(polynomial2)

    rows = max(degree_of_b1, degree_of_b2)
    columns = max(degree_of_c1, degree_of_c2)

    pad_matrix(polynomial1, rows, columns)
    pad_matrix(polynomial2, rows, columns)

    # add
    sum = [[polynomial1[i][j] + polynomial2[i][j] for j in range(columns)] for i in range(rows)]
    return sum

def add_polynomials(polynomial1, polynomial2, polynomial3=False):
    first_sum = add_pair_of_polynomials(polynomial1, polynomial2) 
    if polynomial3: 
        sum = add_pair_of_polynomials(first_sum, polynomial3)
        return sum
    else:
        return first_sum

def multiply_by_scalar(polynomial, scalar=-1):
    return [[value * scalar for value in row] for row in polynomial]


# each result is a 2d array, where the row index is the power of B, 
# and column index is the power of C (both zero-indexed).
# The number stored is the coefficient of that monomial.
#
# a full result requires three of these arrays (for each of b, c, and d output ports)

def calculate_result(iterations):
    # if the photon only passes through the Grover coin once, the following defines its quantum state.
    result = {'b': [[1]], 'c': [[1]], 'd': [[1]]}
    result_d_series = [[[1]]]

    for _ in range(iterations-1):
        # propagate along respective paths
        result['b'] = multiply_by_b(result['b'])
        result['c'] = multiply_by_c(result['c'])

        # apply Grover matrix

        # store copies of results and results multiplied by -1
        opposite_of_b = multiply_by_scalar(result['b'])
        opposite_of_c = multiply_by_scalar(result['c'])
        #opposite_of_d = multiply_by_scalar(result['d'])
        result_b = result['b']
        result_c = result['c']
        #result_d = result['d']

        result['b'] = add_polynomials(opposite_of_b, result_c)
        result['c'] = add_polynomials(result_b, opposite_of_c)
        result['d'] = add_polynomials(result_b, result_c)
        result_d_series.append(result['d'])

    return result, result_d_series


success = False
while success == False:
    try:
        iterations = int(input("How many times to iterate the photon through the Grover coin? \n"))
        success = True
    except:
        print("Please input a positive integer.")
        continue

final_result, all_d_states = calculate_result(iterations)


#for path in ['d']:
    #print("Final state with respect to path " + path + ":")
    #for row in final_result[path]: print(row)

#print("All states:")
#for state in all_d_states:
#    for row in state: print(row)

sum_of_states = [[0]]
print("Sum of states:")
for state in all_d_states:
    sum_of_states = add_polynomials(sum_of_states, state)
for row in sum_of_states: print(row)