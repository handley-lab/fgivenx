import sys
import os

class ProgressBar(object):
    percent = -1
    i=0.0
    n=0
    message=" "
    def __init__(self,n,message=" "):
        self.n=n
        self.message=message


    def __call__(self):
        status = ""
        last   = False
        self.i+=1.0
        progress = self.i/self.n

        if progress >= 1:
            last = True
            progress = 1
            status = "Done\r\n\r"

        percent = round(progress*1000)/10

        if percent > self.percent or last:
            self.percent = percent

            rows, columns = os.popen('stty size', 'r').read().split()
            bar_length= int(columns) - len(self.message) - 15
            block = int(round(bar_length*progress))

            text = "\r" + self.message + " [{0}] {1}% {2} ".format( "#"*block + "-"*(bar_length-block), percent, status)
            sys.stdout.write(text)
            sys.stdout.flush()

        if last:
            sys.stdout.write("\r")
            sys.stdout.flush()
