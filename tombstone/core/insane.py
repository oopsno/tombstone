def give_up(exit_code=None):
    """
    Give up current job and shutdown your Python interpreter.

    :param exit_code: if exit_code is 0, `give_up` returns None; otherwise, call `exit` with random exit code.
    :return: None
    :rtype: None
    """
    if exit_code is None:
        import random
        exit_code = random.randint(1, 255)
    if exit_code != 0:
        exit(exit_code)
