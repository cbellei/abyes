def check_size(x, dim):
    if not len(x)==dim:
        raise Exception('The data should be a two-dimensional array')
    else:
        return


def print_result(result):

    msg = ''
    if result == 1:
        print()
        msg = 'Result is conclusive: B variant is winner!'
    elif result == -1:
        print()
        msg = '* Result is conclusive: A variant is winner!'
    elif result == 0:
        print()
        msg = '* Result is conclusive: A and B variants are effectively equivalent!'
    else:
        if(type(result)==list and len(result)==2):
            print_result(result[0])
            print_result(result[1])
        else:
            print()
            msg = 'Result is inconclusive.'

    print(msg)
    print()

def print_info(info):
    print()
    print('*** abyes ***')
    print()
    print('Method = %s' % info.method)
    print('Decision Rule = %s' % info.rule)
    if info.rule== 'rope':
        print('Alpha = %s' % info.alpha)
        print('Rope = %s' % str(info.rope))
    elif info.rule== 'loss':
        print('Threshold of Caring = %s' % info.toc)
    print('Decision Variable = %s' %info.decision_var)

