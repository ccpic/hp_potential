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
pt_hp_hassale = Potential(df.loc[mask, :], name="公立有销量目标医院")

mask = (df["医院类型"] == "社区医院") & (df["销售状态"] == "有销量目标医院")
pt_cm_hassale = Potential(df.loc[mask, :], name="社区有销量目标医院")

# pt_cm.plot_2d_bubble(
#     value_x="终端潜力值", value_y="销售代表", index="城市", top=30, log_x=True, z_scale=15
# )
pt_cm.plot_2d_bubble(
    value_x="终端潜力值",
    value_y="信立坦同期销量",
    index="省份",
    log_x=True,
    log_y=True,
    z_scale=15,
    label_limit=100,
)
