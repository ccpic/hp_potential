from numpy.core.overrides import array_function_dispatch
from numpy.lib.twodim_base import mask_indices
from chart_func import plot_pie, plot_bubble
import numpy as np
import pandas as pd
from dataclass import Potential
from data_prepare import prepare_data

df = prepare_data()

filter_name = "全国"
filter_mask = (df["省份"]!="台湾")

mask = filter_mask
pt_total = Potential(
    df.loc[mask, :], name="%s城市医院整体市场" % filter_name, savepath="./plots/%s/" % filter_name
)
mask = (df["医院类型"] == "公立医院") & filter_mask
pt_hp = Potential(
    df.loc[mask, :], name="%s等级医院" % filter_name, savepath="./plots/%s/" % filter_name
)
mask = (df["医院类型"] == "社区医院") & filter_mask
pt_cm = Potential(
    df.loc[mask, :], name="%s社区医院" % filter_name, savepath="./plots/%s/" % filter_name
)
mask = (df["医院类型"] == "社区医院") & (df["销售状态"] == "有销量目标医院") & filter_mask
pt_cm_hassale = Potential(
    df.loc[mask, :], name="%s有量社区医院" % filter_name, savepath="./plots/%s/" % filter_name
)

# 终端潜力 vs 信立坦份额 气泡图
pt_cm.plot_2d_bubble(
    value_x="销售状态",
    value_y="信立坦销售份额",
    index="城市",
    log_x=False,
    log_y=False,
    z_scale=0.0001,
    label_limit=40,
    with_reg=False,
    fmt_y="{:.0%}",
    lim_y=[0, 0.4],
)
