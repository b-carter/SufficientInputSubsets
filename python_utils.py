import sys


def redirect_stdout(to_path):
    # python should close this when the script exists
    f = open(to_path, 'w')
    sys.stdout = f

def redirect_stderr(to_path):
    # python should close this when the script exists
    f = open(to_path, 'w')
    sys.stderr = f
