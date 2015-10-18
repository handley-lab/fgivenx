import sys


class ProgressBar(object):
    bar_length = 40
    i=0.0
    n=0
    message=" "
    def __init__(self,n,message=" ",bar_length=40):
        self.n=n
        self.message=message
        self.bar_length=bar_length
    def __call__(self):
        status = ""
        self.i+=1.0
        progress = self.i/self.n

        if progress >= 1:
            progress = 1
            status = "Done...             \r\n"

        block = int(round(self.bar_length*progress))
        text = "\r" + self.message + " [{0}] {1}% {2}             ".format( "#"*block + "-"*(self.bar_length-block), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()



# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def update_progress(progress,message=" "):
    barLength = 40 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r" + message + " [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

