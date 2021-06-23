import numpy as np
import pandas as pd
from dataclass import Potential
from data_prepare import prepare_data
from chart_func import plot_barline

df = prepare_data()

pt_total = Potential(df, name="公立医院+社区医院")
pt_hp = Potential(df[df["医院类型"]=="公立医院"], name="公立医院")
pt_cm = Potential(df[df["医院类型"]=="社区医院"], name="社区医院")


pt_total.plot_decile_groupby("医院类型")
pt_cm.plot_decile_groupby("销售状态")