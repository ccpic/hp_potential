import collections
from matplotlib.pyplot import title, xlabel
import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import pivot_table
from chart_func import *


D_SORTER = {
    "销售状态": ["有销量目标医院", "无销量目标医院", "非目标医院"],
    "医院类型": ["公立医院", "社区医院"],
    "潜力分位": list(range(10, 0, -1)),
}

D_AGGFUNC = {
    "终端潜力值": sum,
    "信立坦同期销量": sum,
    "信立坦2021指标": sum,
    "医院名称": len,
    "销售代表": lambda x: ",".join(x.astype(str)),
}

D_UNIT = {
    "十亿": 1000000000,
    "亿": 100000000,
    "百万": 1000000,
    "万": 10000,
    "千": 1000,
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
        line_share=None,
        unit_index=None,
        top=None,
        percentage=False,
        y1labelthreshold=0.015,
    ):  # 潜力分位绘图

        aggfunc = D_AGGFUNC[value]
        df_bar = self.get_pivot(value, index, column, aggfunc)

        """添加指定列的share作为折线图数据"""
        if line_share is not None:
            df_line = df_bar[line_share].fillna(0) / df_bar.sum(axis=1)
            df_line = df_line.to_frame()
            df_line.columns = [line_share]
        else:
            df_line = None

        """是否换单位"""
        if unit_index is not None:
            df_bar = df_bar / D_UNIT[unit_index]

        """是否只取top项，如只取top一些文本标签会变化"""
        label_prefix = "各"
        xtitle = index
        if top is not None:
            df_bar = df_bar.iloc[:top, :]
            if df_line is not None:
                df_line = df_line.iloc[:top, :]
            label_prefix = "TOP" + str(top)
            xtitle = label_prefix + index

        """根据统计方式不同判断y轴标签及y轴显示逾限"""
        if aggfunc == len:
            ytitle = "终端数量"
        elif aggfunc == sum:
            ytitle = "潜力DOT"

        """如果换过单位在y轴标题也要体现"""
        if unit_index is not None:
            ytitle = "%s（%s）" % (ytitle, unit_index)

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

        """图表标题"""
        title = "%s%s%s%s%s" % (
            self.name,
            label_prefix,
            index,
            "" if column is None else "不同" + column,
            ytitle,
        )

        plot_barline(
            df_bar=df_bar,
            df_line=df_line,
            savefile="%s%s柱状图.png" % (self.savepath, title.replace("\n", "")),
            xlabel_rotation=xlabel_rotation,
            y1fmt=y1fmt,
            y1labelfmt="{:,.0f}",
            y1labelthreshold=y1labelthreshold,
            percentage=percentage,
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            show_total=show_total,
        )

    def plot_share_pie(self, value, index, unit_index=None):
        aggfunc = D_AGGFUNC[value]
        df = self.get_pivot(value=value, index=index, column=None, aggfunc=aggfunc)

        """是否换单位"""
        if unit_index is not None:
            df = df / D_UNIT[unit_index]

        """根据统计方式不同判断y轴标签及y轴显示逾限"""
        if aggfunc == len:
            label = "终端数量"
        elif aggfunc == sum:
            label = "潜力DOT"

        """如果换过单位在标签也要体现"""
        if unit_index is not None:
            label = "%s（%s）" % (label, unit_index)

        """图表标题"""
        title = "%s\n%s\n%s占比" % (
            value,
            index,
            label,
        )
        plot_pie(
            savefile="%s%s饼图.png" % (self.savepath, title.replace("\n", "")),
            sizes=df[value].values,
            labels=df.index,
            title=title,
        )

    def plot_contrib_barline(
        self,
        value,
        index,
        unit_index=None,
        top=None,
    ):
        aggfunc = D_AGGFUNC[value]
        df = self.get_pivot(value=value, index=index, column=None, aggfunc=aggfunc)
        df_bar = df / df.sum()
        df_bar.columns = ["潜力贡献占比"]
        df_line = df_bar.cumsum()  # 累积贡献
        df_line.columns = ["累积贡献占比"]

        """是否换单位"""
        if unit_index is not None:
            df = df / D_UNIT[unit_index]

        """是否只取top项，如只取top一些文本标签会变化"""
        label_prefix = "各"
        xtitle = index
        if top is not None:
            df_bar = df_bar.iloc[:top, :]
            df_line = df_line.iloc[:top, :]
            label_prefix = "TOP" + str(top)
            xtitle = label_prefix + index

        """根据统计方式不同判断y轴标签及y轴显示逾限"""
        if aggfunc == len:
            ytitle = "终端数量"
        elif aggfunc == sum:
            ytitle = "潜力DOT"

        """如果换过单位在标签也要体现"""
        if unit_index is not None:
            ytitle = "%s（%s）" % (ytitle, unit_index)

        """根据不同index决定x轴标签是否旋转90度"""
        if index == "潜力分位":
            xlabel_rotation = 0
        else:
            xlabel_rotation = 90

        """图表标题"""
        title = "%s%s%s%s贡献及累积占比" % (
            self.name,
            label_prefix,
            index,
            ytitle,
        )

        plot_barline(
            df_bar=df_bar,
            df_line=df_line,
            savefile="%s%s柱状图.png" % (self.savepath, title.replace("\n", "")),
            xlabel_rotation=xlabel_rotation,
            y1fmt="{:.0%}",
            show_y1label=False,
            y1labelfmt="{:.1%}",
            y1labelthreshold=0,
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            show_total=True,
            total_fontsize=12,
        )

    def plot_2d_bubble(
        self,
        value_x,
        value_y,
        index,
        log_x=False,
        log_y=False,
        unit_index_x=None,
        unit_index_y=None,
        top=None,
        with_reg=True,
        z_scale=1,
        fmt_x="{:,.0f}",
        fmt_y="{:,.0f}",
        label_limit=30,
    ):
        aggfunc1 = D_AGGFUNC[value_x]
        df_x = self.get_pivot(
            value=value_x,
            index=index,
            column=None,
            aggfunc=aggfunc1,
        )
        """是否取对数"""
        if log_x:
            df_x = np.log(df_x)

        """是否换单位"""
        if unit_index_x is not None:
            df_x = df_x / D_UNIT[unit_index_x]

        aggfunc2 = D_AGGFUNC[value_y]
        df_y = self.get_pivot(
            value=value_y,
            index=index,
            column=None,
            aggfunc=aggfunc2,
        )

        '''如果统计销售代表，一个lambda很难完成，需要进一步处理'''
        if value_y == "销售代表":
            value_y = "销售代表人数"
            df_y[value_y] = df_y["销售代表"].apply(
                lambda x: len(list(set([y for y in x.split(",") if str(y) != "nan"])))
            ) # 将销售代表list换算成人数
            df_y.drop("销售代表", axis=1, inplace=True)

        """是否取对数"""
        if log_y:
            df_y = np.log(df_y)

        """是否换单位"""
        if unit_index_y is not None:
            df_y = df_y / D_UNIT[unit_index_y]
            
        # print(df_x,df_y)
        df_combined = pd.concat([df_x, df_y], axis=1)
        df_combined.replace([np.inf, -np.inf, np.nan], 0 , inplace=True) # 所有异常值替换为0
        
        """是否只取top项，如只取top一些文本标签会变化"""
        label_prefix = "各"
        if top is not None:
            df_combined = df_combined.iloc[:30, :]
            label_prefix = "TOP" + str(top)

        print(df_combined)
        
        """轴标题"""
        xtitle = value_x
        if unit_index_x is not None:
            xtitle = "%s（%s）" % (xtitle, unit_index_x)
        if log_x:
            xtitle = "%s %s" % (xtitle, "取对数")
        ytitle = value_y
        if unit_index_y is not None:
            ytitle = "%s（%s）" % (ytitle, unit_index_y)
        if log_y:
            ytitle = "%s %s" % (ytitle, "取对数")

        """图表标题"""
        title = "%s%s%s%s vs. %s" % (
            self.name,
            label_prefix,
            index,
            value_x,
            value_y,
        )
        
        '''绘图'''
        plot_bubble(
            savefile="%s%s气泡图.png" % (self.savepath, title.replace("\n", "")),
            x=df_combined[value_x],
            y=df_combined[value_y],
            z=df_combined[value_x],
            labels=df_combined.index,
            title=title,
            xtitle=xtitle,
            ytitle=ytitle,
            z_scale=z_scale,
            xfmt=fmt_x,
            yfmt=fmt_y,
            label_limit=label_limit,
            with_reg=with_reg,
        )
