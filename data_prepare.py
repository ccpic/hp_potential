# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from dataclass import Potential
from chart_func import *

pd.set_option("display.max_columns", 5000)
pd.set_option("display.width", 5000)


def prepare_data():
    """导入公立医院终端潜力数据"""
    df_hp = pd.read_excel(open("潜力数据.xlsx", "rb"), sheet_name="医院潜力")  # 从Excel读取大医院潜力数据
    df_hp["信立泰医院名称"].fillna(df_hp["IQVIA 医院名称"], inplace=True)  # 没有信立泰名称的copy IQVIA医院名称
    df_hp = df_hp.loc[:, ["信立泰医院代码", "信立泰医院名称", "省份", "城市", "区县", "终端潜力值"]]
    df_hp.columns = ["医院编码", "医院名称", "省份", "城市", "区县", "终端潜力值"]
    df_hp["数据源"] = "IQVIA大医院潜力201909MAT"
    df_hp["医院类型"] = "公立医院"
    df_hp["终端潜力值"] = df_hp["终端潜力值"] * 1.1  # 大医院潜力项目早1年半做，放大1.1倍，RAAS市场年增长率10%

    """ 导入社区医院终端潜力数据"""
    df_cm = pd.read_excel(
        open("潜力数据.xlsx", "rb"), sheet_name="社区潜力"
    )  # 从Excel读取社区医院潜力数据
    df_cm = df_cm.loc[:, ["信立泰ID", "终端名称", "省份", "城市", "区县", "潜力值（DOT）"]]
    df_cm.columns = ["医院编码", "医院名称", "省份", "城市", "区县", "终端潜力值"]
    df_cm["数据源"] = "Pharbers社区医院潜力202103MAT"
    df_cm["医院类型"] = "社区医院"

    """ 删除重复值"""
    df_combined = pd.concat([df_hp, df_cm])
    dup_rows = df_combined[df_combined.duplicated(subset=["医院编码"], keep="last")].dropna(
        subset=["医院编码"]
    )  # 找出IQVIA和Pharbers数据重复的终端，keep参数=first保留IQVIA的，last保留Pharbers的
    # dup_rows.to_csv("dup.csv", encoding="utf-8-sig")
    df_combined = df_combined.drop(dup_rows.index)  # drop重复数据

    """准备内部销售数据并merge，因内部数据较大，在internal_sales.py文件单独处理"""
    df_internal = pd.read_csv("internal_sales.csv")
    df_combined = pd.merge(
        left=df_combined, right=df_internal, how="left", on="医院编码"
    )  # left表示以潜力数据为主题匹配
    df_combined["销售状态"] = df_combined.apply(
        lambda row: "非目标医院"
        if pd.isna(row["医院编码"])
        else (
            "无销量目标医院"
            if pd.isna(row["MAT销量标准片数"]) or row["MAT销量标准片数"] == 0
            else "有销量目标医院"
        ),
        axis=1,
    )
    # 根据是否有医院编码以及是否有销量标记终端销售状态

    """计算潜力分位"""
    df_combined.sort_values(by=["终端潜力值"], inplace=True)
    df_combined["潜力分位"] = (
        np.floor(df_combined["终端潜力值"].cumsum() / df_combined["终端潜力值"].sum() * 10) + 1
    ).astype(
        "int"
    )  # 注意这里不能用pandas自带的rank或者scipy包的percentilofscore
    df_combined.sort_values(by=["终端潜力值"], ascending=False, inplace=True)

    """导出"""
    df_combined.to_csv("data.csv", encoding="utf-8-sig")

    return df_combined


if __name__ == "__main__":
    print(prepare_data())