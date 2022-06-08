import numpy as np


class UnavoidableBias:


    def __init__(self, R, epsilon, advers_name):

        if advers_name == 'prescient':
            D = R(epsilon/(2* (1-epsilon)))
        elif advers_name == 'oblivious':
            D= R(epsilon/(2* (1- epsilon)))

        elif advers_name == 'malicious':
            D= R(epsilon)
        self.D= D

    def return_D(self):
        return self.D
