import pandas as pd
import numpy as np
from chart_func import *


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

    def get_decile(self):  # 计算各终端潜力分位
        df = self.sort_values(by=["终端潜力值"])
        df["潜力分位"] = (
            np.floor(df["终端潜力值"].cumsum() / df["终端潜力值"].sum() * 10) + 1
        ).astype("int")

        df.sort_values(by=["终端潜力值"], ascending=False, inplace=True)
        return df

    def get_decile_groupby(self, columns):  # 潜力分位相关的透视表
        df = self.get_decile()
        pivoted = pd.pivot_table(
            data=df, values="医院名称", index="潜力分位", columns=columns, aggfunc=len
        )
        pivoted.sort_index(ascending=False, inplace=True)
        return pivoted

    def plot_decile_groupby(self, columns):  # 潜力分位绘图
        xlabel = "潜力分位"
        ylabel = "终端数量"
        title = "各%s不同%s%s" % (xlabel, columns, ylabel)
        plot_barline(
            self.get_decile_groupby(columns),
            savefile="%s%s%s.png" % (self.savepath, self.name, title),
            xlabel_rotation=0,
            y1fmt="",
            y1labelfmt="{:,.0f}",
            y1labelthreshold=0,
            percentage=True,
            title=title,
            xtitle="潜力分位",
            ytitle="终端数量",
        )
