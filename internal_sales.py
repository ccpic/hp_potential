# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd


# 非na的unique列表
def unique_non_null(s):
    return s.dropna().unique()


filename_3on1 = "三合一表10月终版"
start_month = 202011
end_month = 202110

df = pd.read_excel(
    open("%s.xlsx" % filename_3on1, "rb"), sheet_name="三合一表_导出版"
)  # 从Excel读取内部三合一销售数据
df["标准数量"] = df["标准数量"] * 7  # 盒数转化成片数
mask = df["产品名称"] == "信立坦"

mask_sales = (
    mask & (df.tag.isin(["销量", "药房销量"])) & (df["填报日期"].between(start_month, end_month))
)  
df_sales = df.loc[mask_sales, :]
df_sales["大区经理"] = df["大区"] + " " + df["大区经理"]  # 坑位号+姓名，下同
df_sales["地区经理"] = df["地区"] + " " + df["地区经理"]
df_sales["销售代表"] = df["代表岗位"] + " " + df["销售代表"]


# 单家终端销售
pivoted_sales = pd.pivot_table(data=df_sales, values="标准数量", index="目标代码", aggfunc=sum)

# 关联代表情况
pivoted_reps = pd.pivot_table(
    data=df_sales,
    values="销售代表",
    index="目标代码",
    aggfunc=lambda x: unique_non_null(x),
)

# 指标情况
mask_target = mask & (df.tag.isin(["指标"])) & (df["年"] == int(str(end_month)[:4]))
df_target = df.loc[mask_target, :]
pivoted_target = pd.pivot_table(
    data=df_target, values="标准数量", index="目标代码", aggfunc=sum
)

# 合并上方各结果
df_combined = pd.concat([pivoted_sales, pivoted_reps, pivoted_target], axis=1)
df_combined.reset_index(level=0, inplace=True)
df_combined = pd.merge(
    df_combined,
    df_sales.drop_duplicates(subset=["目标代码", "事业部", "区域", "大区经理", "地区经理"], keep="last")[
        ["目标代码", "事业部", "区域", "大区经理", "地区经理"]
    ],
    on="目标代码",
    how="left",
)

df_combined.columns = [
    "医院编码",
    "信立坦MAT销量",
    "销售代表",
    "信立坦年度指标",
    "事业部",
    "区域",
    "大区经理",
    "地区经理",
]
df_combined.to_csv("internal_sales.csv", encoding="utf-8-sig", index=False)

print(df_combined)
