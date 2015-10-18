import sys
import os

class ProgressBar(object):
    percent = -1
    bar_length = 40
    i=0.0
    n=0
    message=" "
    def __init__(self,n,message=" "):
        self.n=n
        self.message=message
        rows, columns = os.popen('stty size', 'r').read().split()
        self.bar_length= int(columns) - len(message) - 15


    def __call__(self):
        status = ""
        last   = False
        self.i+=1.0
        progress = self.i/self.n

        if progress >= 1:
            last = True
            progress = 1
            status = "Done\r\n\r"

        block = int(round(self.bar_length*progress))
        percent = round(progress*1000)/10
        if percent > self.percent or last:
            self.percent = percent
            text = "\r" + self.message + " [{0}] {1}% {2} ".format( "#"*block + "-"*(self.bar_length-block), percent, status)
            sys.stdout.write(text)
            sys.stdout.flush()

        if last:
            sys.stdout.write("\r")
            sys.stdout.flush()
