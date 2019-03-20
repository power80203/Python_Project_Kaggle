#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(".."))

import config

train = pd.read_csv(config.trainset_path)