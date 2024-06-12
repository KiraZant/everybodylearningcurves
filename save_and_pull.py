from utils import *

def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + '/graphs')
        os.makedirs(path + '/models')
        os.makedirs(path + '/results')
