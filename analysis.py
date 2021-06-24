from chart_func import plot_pie
import numpy as np
import pandas as pd
from dataclass import Potential
from data_prepare import prepare_data

df = prepare_data()

pt_total = Potential(df, name="公立医院+社区医院")
pt_hp = Potential(df[df["医院类型"] == "公立医院"], name="公立医院")
pt_cm = Potential(df[df["医院类型"] == "社区医院"], name="社区医院")

pt_cm.plot_contrib_barline(value="终端潜力值", index="省份", aggfunc=sum)
pt_cm.plot_contrib_barline(value="终端潜力值", index="城市", aggfunc=sum, top=30)