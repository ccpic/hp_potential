# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd


df = pd.read_excel(
    open("三合一表5月初版.xlsx", "rb"), sheet_name="三合一表_导出版"
)  # 从Excel读取内部三合一销售数据
df["标准数量"] = df["标准数量"] * 7  # 盒数转化成片数
mask = df["产品名称"] == "信立坦"


mask_sales = mask & (df.tag.isin(["销量", "药房销量"])) & (df["填报日期"].between(202004, 202103))
df_sales = df.loc[mask_sales, :]
pivoted_sales = pd.pivot_table(data=df_sales, values="标准数量", index="目标代码", aggfunc=sum)

print(pivoted_sales)
pivoted_reps = pd.pivot_table(
    data=df_sales,
    values="销售代表",
    index="目标代码",
    aggfunc=lambda x: pd.unique([str(v) for v in x]),
)


mask_target = mask & (df.tag.isin(["指标"])) & (df["年"] == 2021)
df_target = df.loc[mask_target, :]
pivoted_target = pd.pivot_table(
    data=df_target, values="标准数量", index="目标代码", aggfunc=sum
)


df_combined = pd.concat([pivoted_sales, pivoted_reps, pivoted_target], axis=1)
df_combined.reset_index(level=0, inplace=True)
df_combined.columns = ["医院编码", "MAT销量标准片数", "销售代表", "2021指标标准片数"]
df_combined.to_csv("internal_sales.csv", encoding="utf-8-sig", index=False)
