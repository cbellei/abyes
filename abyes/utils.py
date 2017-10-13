import numpy as np

def check_size(x, dim):
    if not len(x)==dim:
        raise Exception("The data should be a two-dimensional array")
    else:
        return


def print_result(result):

    msg = ''
    if result == 1:
        msg = "Result is conclusive: B variant is winner!"
    elif result == -1:
        msg = "Result is conclusive: A variant is winner!"
    elif result == 0:
        msg = "Result is conclusive: A and B variants are effectively equivalent!"
    else:
        if(type(result)==list and len(result)==2):
            print_result(result[0])
            print_result(result[1])
        else:
            msg = "Result is inconclusive."

    print(msg)
