from numpy.core.overrides import array_function_dispatch
from numpy.lib.twodim_base import mask_indices
from chart_func import plot_pie, plot_bubble
import numpy as np
import pandas as pd
from dataclass import Potential
from data_prepare import prepare_data

df = prepare_data()

filter_name = "全国"
filter_mask = df["省份"] != "台湾"
# filter_mask = df["省份"] == "广东"

mask = filter_mask
pt_total = Potential(
    df.loc[mask, :],
    name="%s城市医院整体市场" % filter_name,
    savepath="./plots/%s/" % filter_name,
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

# mask = (df["医院类型"] == "公立医院") & (df["中医院"] == "中医院") & filter_mask
# pt_zh = Potential(
#     df.loc[mask, :],
#     name="%s中医院" % filter_name,
#     savepath="./plots/中医院/%s/" % filter_name,
# )

# mask = (df["医院类型"] == "公立医院") & (df["中医院"] == "非中医院") & filter_mask
# pt_nzh = Potential(
#     df.loc[mask, :],
#     name="%s非中医院" % filter_name,
#     savepath="./plots/非中医院/%s/" % filter_name,
# )

# mask = (
#     (df["医院类型"] == "公立医院")
#     & (df["中医院"] == "中医院")
#     & (df["销售状态"] == "有销量目标医院")
#     & filter_mask
# )

# pt_zh_hassale = Potential(
#     df.loc[mask, :],
#     name="%s有销量中医院" % filter_name,
#     savepath="./plots/中医院/%s/" % filter_name,
# )

# mask = (
#     (df["医院类型"] == "公立医院")
#     & (df["中医院"] == "非中医院")
#     & (df["销售状态"] == "有销量目标医院")
#     & filter_mask
# )

# pt_nzh_hassale = Potential(
#     df.loc[mask, :],
#     name="%s有销量非中医院" % filter_name,
#     savepath="./plots/非中医院/%s/" % filter_name,
# )

# # 终端潜力 vs 信立坦份额 气泡图
# pt_total.plot_2d_bubble(
#     value_x="销售状态",
#     value_y="信立坦销售份额",
#     index="城市",
#     log_x=False,
#     log_y=False,
#     z_scale=0.00005,
#     label_limit=50,
#     with_reg=False,
#     fmt_x="{:.0%}",
#     fmt_y="{:.0%}",
#     lim_y=[0, 0.15],
# )

pt_cm.table_to_excel(
    lst_index=[
        "省份",
        "城市",
        "等级医院内部潜力分位",
        # "社区医院内部潜力分位",
        # "等级+社区合并计算潜力分位",
        "医院名称",
        "事业部",
        "区域",
        "大区经理",
        "地区经理",
    ]
)

# # 终端潜力 vs 信立坦份额 气泡图
# pt_cm.plot_2d_bubble(
#     value_x="终端潜力值",
#     value_y="信立坦销售份额",
#     index="省份",
#     log_x=False,
#     log_y=False,
#     z_scale=0.00003,
#     label_limit=50,
#     with_reg=False,
#     fmt_x="{:.0%}",
#     fmt_y="{:.0%}",
#     lim_y=[0, 0.3],
#     xavgline=True,
#     yavgline=True,
# )

# pt_cm.plot_2d_bubble(
#     value_x="终端潜力值",
#     value_y="信立坦销售份额",
#     index="城市",
#     log_x=False,
#     log_y=False,
#     z_scale=0.00003,
#     label_limit=30,
#     with_reg=False,
#     fmt_x="{:.0%}",
#     fmt_y="{:.0%}",
#     lim_y=[0, 0.3],
#     xavgline=True,
#     yavgline=True,
# )


# # 不同医院类型潜力/指标/销售的占比饼图
# pt_hp.plot_share_pie(value="终端潜力值", index="中医院")
# pt_hp.plot_share_pie(value="医院名称", index="中医院")
# pt_hp.plot_share_pie(value="信立坦2021指标", index="中医院")
# pt_hp.plot_share_pie(value="信立坦MAT销量", index="中医院")

# pt_hp.plot_pivot_stackedbar(value="终端潜力值",
#                                index="省份",
#                                column="中医院",
#                                line_share="中医院",
#                                unit_index="百万",
#                                y1labelthreshold=10)
# pt_hp.plot_pivot_stackedbar(value="终端潜力值",
#                                index="城市",
#                                column="中医院",
#                                line_share="中医院",
#                                top=30,
#                                unit_index="百万",
#                                y1labelthreshold=10)

# pt_hp.plot_pivot_stackedbar(
#     value="医院名称", index="潜力分位", column="中医院", percentage=True, y1labelthreshold=0
# )

# pt_zh.plot_share_pie(value="终端潜力值", index="销售状态")
# pt_zh.plot_share_pie(value="医院名称", index="销售状态")
# pt_zh_hassale.plot_share_pie(value="终端潜力值", index="信立坦销售表现")
# pt_zh_hassale.plot_share_pie(value="医院名称", index="信立坦销售表现")

# pt_nzh.plot_share_pie(value="终端潜力值", index="销售状态")
# pt_nzh.plot_share_pie(value="医院名称", index="销售状态")
# pt_nzh_hassale.plot_share_pie(value="终端潜力值", index="信立坦销售表现")
# pt_nzh_hassale.plot_share_pie(value="医院名称", index="信立坦销售表现")
