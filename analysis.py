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

# pt_cm_hassale.plot_pivot_stackedbar(
#     value="医院名称", index="潜力分位", column="信立坦销售表现", 
# )

# df_potential = pt_hp_hassale.get_pivot(value="信立坦同期销量", index="潜力分位", column=None, aggfunc=sum)
# df_number = pt_hp_hassale.get_pivot("医院名称", index="潜力分位",column=None, aggfunc=len)
# df_combined = pd.concat([df_potential, df_number],axis=1)
# df_combined["终端单产"] = df_combined["信立坦同期销量"]/df_combined["医院名称"]/12/7
# print(df_combined)

pt_hp.table_to_excel("城市", top=100)