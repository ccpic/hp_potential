import collections
from matplotlib.pyplot import xlabel
import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import pivot_table
from chart_func import *


D_SORTER = {
    "销售状态": ["有销量目标医院", "无销量目标医院", "非目标医院"],
    "医院类型": ["公立医院", "社区医院"],
    "潜力分位": list(range(10, 0, -1)),
}


class Potential(pd.DataFrame):
    @property
    def _constructor(self):
        return Potential._internal_constructor(self.__class__)

    class _internal_constructor(object):
        def __init__(self, cls):
            self.cls = cls

        def __call__(self, *args, **kwargs):
            kwargs["name"] = None
            return self.cls(*args, **kwargs)

        def _from_axes(self, *args, **kwargs):
            return self.cls._from_axes(*args, **kwargs)

    def __init__(
        self,
        data,
        name,
        savepath="./plots/",
        index=None,
        columns=None,
        dtype=None,
        copy=True,
    ):
        super(Potential, self).__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.name = name
        self.savepath = savepath

    def get_pivot(self, value, index, column, aggfunc):  # 潜力分位相关的透视表
        pivoted = pd.pivot_table(
            data=self, values=value, index=index, columns=column, aggfunc=aggfunc
        )

        pivoted = pd.DataFrame(pivoted.to_records())  # pivot table对象转为默认df
        pivoted.set_index(index, inplace=True)

        pivoted["列汇总"] = pivoted.sum(axis=1)
        pivoted.sort_values(by="列汇总", ascending=False, inplace=True)
        pivoted.drop("列汇总", axis=1, inplace=True)

        if index in D_SORTER:
            try:
                pivoted = pivoted.reindex(index=D_SORTER[index])  # 对于部分变量有固定列排序
            except KeyError:
                pass

        if column in D_SORTER:
            try:
                pivoted = pivoted.reindex(columns=D_SORTER[column])  # 对于部分变量有固定列排序
            except KeyError:
                pass

        print(pivoted)

        return pivoted

    def plot_pivot_stackedbar(
        self,
        value,
        index,
        column,
        aggfunc,
        unit_index=None,
        top=None,
        percentage=False,
        y1labelthreshold=0.015,
    ):  # 潜力分位绘图

        df = self.get_pivot(value, index, column, aggfunc)

        """是否换单位"""
        if unit_index == "百万":
            df = df * 0.000001
        elif unit_index == "万":
            df = df * 0.0001
        elif unit_index == "千":
            df = df * 0.001

        """是否只取top项，如只取top一些文本标签会变化"""
        label_prefix = "各"
        xtitle = index
        if top is not None:
            df = df.iloc[:top, :]
            label_prefix = "TOP" + str(top)
            xtitle = label_prefix + index

        """根据统计方式不同判断y轴标签及y轴显示逾限"""
        if aggfunc == len:
            ylabel = "终端数量"
        elif aggfunc == sum:
            ylabel = "潜力DOT"

        """如果换过单位在y轴标签也要体现"""
        if unit_index is not None:
            ylabel = "%s（%s）" % (ylabel, unit_index)

        """根据不同index决定x轴标签是否旋转90度"""
        if index == "潜力分位":
            xlabel_rotation = 0
        else:
            xlabel_rotation = 90

        """是否显示y轴值标签"""
        if percentage:
            y1fmt = ""
            show_total = False
        else:
            y1fmt = "{:,.0f}"
            show_total = True

        title = "%s%s%s不同%s%s" % (self.name, label_prefix, index, column, ylabel)
        plot_barline(
            df,
            savefile="%s%s.png" % (self.savepath, title),
            xlabel_rotation=xlabel_rotation,
            y1fmt=y1fmt,
            y1labelfmt="{:,.0f}",
            y1labelthreshold=y1labelthreshold,
            percentage=percentage,
            title=title,
            xtitle=xtitle,
            ytitle=ylabel,
            show_total=show_total,
        )
