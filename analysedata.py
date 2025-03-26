import cfbd
import numpy as np
import pandas as pd
import math
from api import *


def getHomeAdvantages():
    df = pd.read_excel(r'/Users/Matt/PycharmProjects/CFB Analytics/Home_Advantages.xlsx')
    return df
