def spamupdate(w,email,truth):

    # Input:
    # w     weight vector
    # email instance vector
    # truth label
    #
    # Output:
    #
    # updated weight vector
    #
    # INSERT CODE HERE:
    w = w - 5*(email.reshape(-1,1) * truth.reshape(-1,1))


    return w
