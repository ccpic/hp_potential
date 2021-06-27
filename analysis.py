from numpy.core.overrides import array_function_dispatch
from numpy.lib.twodim_base import mask_indices
from chart_func import plot_pie, plot_bubble
import numpy as np
import pandas as pd
from dataclass import Potential
from data_prepare import prepare_data

df = prepare_data()

pt_total = Potential(df, name="公立医院+社区医院")
pt_hp = Potential(df[df["医院类型"] == "公立医院"], name="公立医院")
pt_cm = Potential(df[df["医院类型"] == "社区医院"], name="社区医院")

mask = (df["医院类型"] == "公立医院") & (df["销售状态"] == "有销量目标医院")
pt_hp_hassale = Potential(df.loc[mask, :], name="有量公立医院")

mask = (df["医院类型"] == "社区医院") & (df["销售状态"] == "有销量目标医院")
pt_cm_hassale = Potential(df.loc[mask, :], name="有量社区医院")

mask = (df["医院类型"] == "社区医院") & (df["省份"] == "北京")
pt_cm_bj = Potential(df.loc[mask, :], name="北京社区医院", savepath="./plots/北京/")

pt_cm_hassale.table_to_excel(index="医院名称", top=30)