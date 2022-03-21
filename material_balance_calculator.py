'''
Make a molar balance solver using graph theory 
(use abstraction levels initialised on the tablet, make sure to also include the species balance): 〖(f〗_tot x_a)in=(〖(f〗_tot x_a))out
'''

class SystemModel(object):

    def __init__(self, number_of_streams,number_of_nodes):
        super().__init__()
        self.number_of_streams = number_of_streams
        self.number_of_nodes = number_of_nodes

    def __str__(self):
        return 