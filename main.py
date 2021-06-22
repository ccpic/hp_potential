# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from dataclass import Potential
from chart_func import *

pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 5000)

df_hp = pd.read_excel(open("data.xlsx", "rb"), sheet_name="医院潜力")  # 从Excel读取大医院潜力数据
df_hp["信立泰医院名称"].fillna(df_hp["IQVIA 医院名称"], inplace=True)  # 没有信立泰名称的copy IQVIA医院名称
df_hp = df_hp.loc[:, ["信立泰医院代码", "信立泰医院名称", "省份", "城市", "区县", "终端潜力值"]]
df_hp.columns = ["医院编码", "医院名称", "省份", "城市", "区县", "终端潜力值"]
df_hp["数据源"] = "IQVIA大医院潜力201909MAT"
df_hp["医院类型"] = "公立医院"
df_hp["终端潜力值"] = df_hp["终端潜力值"] * 1.1  # 大医院潜力项目早1年半做，放大1.1倍，RAAS市场年增长率10%
print(df_hp)

df_cm = pd.read_excel(open("data.xlsx", "rb"), sheet_name="社区潜力")  # 从Excel读取社区医院潜力数据
df_cm = df_cm.loc[:, ["信立泰ID", "终端名称", "省份", "城市", "区县", "潜力值（DOT）"]]
df_cm.columns = ["医院编码", "医院名称", "省份", "城市", "区县", "终端潜力值"]
df_cm["数据源"] = "Pharbers社区医院潜力202103MAT"
df_cm["医院类型"] = "社区医院"
print(df_cm)

df_combined = pd.concat([df_hp, df_cm])

dup_rows = df_combined[df_combined.duplicated(subset=["医院编码"], keep="last")].dropna(
    subset=["医院编码"]
)  # 找出IQVIA和Pharbers数据重复的终端，keep参数=first保留IQVIA的，last保留Pharbers的
# dup_rows.to_csv("dup.csv", encoding="utf-8-sig")

df_combined = df_combined.drop(dup_rows.index)

pt = Potential(df_combined, name="大医院+社区潜力")
# df_with_decile = pt.get_decile()
# df_with_decile.to_csv("decile.csv", encoding="utf-8-sig")

pt.plot_decile_groupby("医院类型")

