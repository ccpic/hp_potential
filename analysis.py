from chart_func import plot_pie
import numpy as np
import pandas as pd
from dataclass import Potential
from data_prepare import prepare_data

df = prepare_data()

pt_total = Potential(df, name="公立医院+社区医院")
pt_hp = Potential(df[df["医院类型"] == "公立医院"], name="公立医院")
pt_cm = Potential(df[df["医院类型"] == "社区医院"], name="社区医院")


result = pt_total.get_pivot(value="终端潜力值", index="医院类型", column=None, aggfunc=sum)

plot_pie(
    savefile="test.png",
    sizes=result["终端潜力值"].values,
    labels=result.index,
)