import collections
from matplotlib.pyplot import title, xlabel
import pandas as pd
import numpy as np
from pandas.core.reshape.pivot import pivot_table
from chart_func import *
import xlsxwriter

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

    def get_table(self, index, top=None):  # 综合表现表格

        # 潜力部分
        pivoted_potential = pd.pivot_table(
            data=self,
            values="终端潜力值",
            index=index,
            columns=None,
            aggfunc=[len, sum],
            fill_value=0,
        )
        pivoted_potential = pd.DataFrame(
            pivoted_potential.to_records()
        )  # pivot table对象转为默认df
        pivoted_potential.set_index(index, inplace=True)
        # pivoted_potential.reset_index(axis=1, inplace=True)
        pivoted_potential.columns = ["终端数量", "潜力(DOT)"]
        pivoted_potential["潜力贡献"] = (
            pivoted_potential["潜力(DOT)"] / pivoted_potential["潜力(DOT)"].sum()
        )
        # 覆盖部分
        pivoted_access = pd.pivot_table(
            data=self,
            values="终端潜力值",
            index=index,
            columns="销售状态",
            aggfunc=[len, sum],
            fill_value=0,
        )
        pivoted_access = pd.DataFrame(
            pivoted_access.to_records()
        )  # pivot table对象转为默认df
        pivoted_access.set_index(index, inplace=True)
        # pivoted_access.reset_index(axis=1, inplace=True)
        pivoted_access.columns = [
            "无销量目标医院终端数量",
            "有销量目标医院终端数量",
            "非目标医院终端数量",
            "无销量目标医院潜力(DOT)",
            "有销量目标医院潜力(DOT)",
            "非目标医院潜力(DOT)",
        ]
        pivoted_access["信立坦目标覆盖终端数量"] = (
            pivoted_access["无销量目标医院终端数量"] + pivoted_access["有销量目标医院终端数量"]
        )
        pivoted_access["信立坦目标覆盖潜力(DOT %)"] = 1 - pivoted_access[
            "非目标医院潜力(DOT)"
        ] / pivoted_access.sum(axis=1)
        pivoted_access["信立坦销售覆盖终端数量"] = pivoted_access["有销量目标医院终端数量"]
        pivoted_access["信立坦销售覆盖潜力(DOT %)"] = pivoted_access[
            "有销量目标医院潜力(DOT)"
        ] / pivoted_access.sum(axis=1)

        # 内部销售部分
        pivoted_sales = pd.pivot_table(
            data=self,
            values="信立坦同期销量",
            index=index,
            columns=None,
            aggfunc=sum,
            fill_value=0,
        )
        pivoted_sales = pd.DataFrame(pivoted_sales.to_records())  # pivot table对象转为默认df
        pivoted_sales.set_index(index, inplace=True)
        pivoted_sales.columns = ["信立坦同期销量(DOT)"]
        pivoted_sales["信立坦销售贡献"] = (
            pivoted_sales["信立坦同期销量(DOT)"] / pivoted_sales["信立坦同期销量(DOT)"].sum()
        )

        # 三部分合并
        df_combined = pd.concat(
            [pivoted_potential, pivoted_access, pivoted_sales], axis=1
        )
        df_combined["信立坦所有终端份额"] = df_combined["信立坦同期销量(DOT)"] / df_combined["潜力(DOT)"]
        df_combined["信立坦目标终端份额"] = df_combined["信立坦同期销量(DOT)"] / (
            df_combined["潜力(DOT)"] * df_combined["信立坦目标覆盖潜力(DOT %)"]
        )
        df_combined["信立坦销售终端份额"] = df_combined["信立坦同期销量(DOT)"] / (
            df_combined["潜力(DOT)"] * df_combined["信立坦销售覆盖潜力(DOT %)"]
        )
        df_combined = df_combined.reindex(
            [
                "终端数量",
                "潜力(DOT)",
                "潜力贡献",
                "信立坦目标覆盖终端数量",
                "信立坦目标覆盖潜力(DOT %)",
                "信立坦销售覆盖终端数量",
                "信立坦销售覆盖潜力(DOT %)",
                "信立坦同期销量(DOT)",
                "信立坦销售贡献",
                "信立坦所有终端份额",
                "信立坦目标终端份额",
                "信立坦销售终端份额",
            ],
            axis="columns",
        )
        df_combined.sort_values(
            by=["潜力(DOT)"], ascending=False, inplace=True
        )  # 根据潜力由高到低排序

        # 只取top items，应该放在计算合计前
        if top is not None:
            df_combined = df_combined.iloc[:top, :]

        # 计算合计，部分字段不能简单相加
        df_combined.loc["合计", :] = df_combined.sum(axis=0)
        df_combined.loc["合计", "信立坦目标覆盖潜力(DOT %)"] = (
            df_combined.loc[df_combined.index != "合计", "潜力(DOT)"]
            * df_combined.loc[df_combined.index != "合计", "信立坦目标覆盖潜力(DOT %)"]
        ).sum() / df_combined.loc["合计", "潜力(DOT)"]
        df_combined.loc["合计", "信立坦销售覆盖潜力(DOT %)"] = (
            df_combined.loc[df_combined.index != "合计", "潜力(DOT)"]
            * df_combined.loc[df_combined.index != "合计", "信立坦销售覆盖潜力(DOT %)"]
        ).sum() / df_combined.loc["合计", "潜力(DOT)"]
        df_combined.loc["合计", "信立坦所有终端份额"] = (
            df_combined.loc["合计", "信立坦同期销量(DOT)"] / df_combined.loc["合计", "潜力(DOT)"]
        )
        df_combined.loc["合计", "信立坦目标终端份额"] = df_combined.loc["合计", "信立坦同期销量(DOT)"] / (
            df_combined.loc["合计", "潜力(DOT)"] * df_combined.loc["合计", "信立坦目标覆盖潜力(DOT %)"]
        )
        df_combined.loc["合计", "信立坦销售终端份额"] = df_combined.loc["合计", "信立坦同期销量(DOT)"] / (
            df_combined.loc["合计", "潜力(DOT)"] * df_combined.loc["合计", "信立坦销售覆盖潜力(DOT %)"]
        )

        df_combined.replace([np.inf, -np.inf, np.nan], 0, inplace=True)  # 所有异常值替换为0

        return df_combined

    def table_to_excel(self, index, top=None):
        df = self.get_table(index, top=top)

        # # Pandas导出
        # df.to_excel(writer, sheet_name="data")

        # 获取工作表对象
        path = "%s%s%s潜力及销售表现表格.xlsx" % (self.savepath, self.name, index)
        wbk = xlsxwriter.Workbook(path, {"nan_inf_to_errors": True})
        sht = wbk.add_worksheet()

        # 添加表格
        sht.add_table(
            first_row=0,
            first_col=0,
            last_row=df.shape[0],
            last_col=df.shape[1],
            options={
                "data": [[i for i in row] for row in df.itertuples()],
                "header_row": True,
                "first_column": True,
                "style": "Table Style Light 1",
                "columns": [{"header": c} for c in df.reset_index().columns.tolist()],
                "autofilter": 0,
            },
        )

        # 添加格式
        format_abs = wbk.add_format(
            {
                "num_format": "#,##0",
                "font_name": "Arial",
                "font_size": 10,
            }
        )
        format_share = wbk.add_format(
            {
                "num_format": "0.0%",
                "font_name": "Arial",
                "font_size": 10,
                "valign": "center",
            }
        )
        format_total_row = wbk.add_format(
            {
                "font_name": "Arial",
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#000000",
                "valign": "center",
            }
        )
        format_thick_border = wbk.add_format({"border": 5})
        # 应用格式到具体单元格
        width_default = 12
        width_wider = 20
        sht.set_column(0, 0, width_default, format_abs)  # 索引列
        sht.set_column(1, 1, width_default, format_abs)  # 终端数量
        sht.set_column(2, 2, width_wider, format_abs)  # 潜力DOT
        sht.set_column(3, 3, width_default, format_share)  # 潜力贡献
        sht.set_column(4, 4, width_default, format_abs)  # 信立坦目标终端数量
        sht.set_column(5, 5, width_default, format_share)  # 信立坦目标终端潜力覆盖(%)
        sht.set_column(6, 6, width_default, format_abs)  # 信立坦销售终端数量
        sht.set_column(7, 7, width_default, format_share)  # 信立坦销售终端潜力覆盖(%)
        sht.set_column(8, 8, width_wider, format_abs)  # 信立坦同期销量
        sht.set_column(9, 12, width_default, format_share)  # 信立坦销售贡献, 3种base的份额

        # sht.set_row(df.shape[0],None, format_total_row)

        # 添加条件格式条形图
        sht.conditional_format(  # 潜力贡献
            first_row=1,
            first_col=3,
            last_row=df.shape[0] - 1,
            last_col=3,
            options={"type": "data_bar"},
        )

        sht.conditional_format(  # 信立坦目标覆盖潜力%
            first_row=1,
            first_col=5,
            last_row=df.shape[0] - 1,
            last_col=5,
            options={"type": "data_bar", "bar_color": "#FFB628"},
        )

        sht.conditional_format(  # 信立坦目标销售覆盖潜力%
            first_row=1,
            first_col=7,
            last_row=df.shape[0] - 1,
            last_col=7,
            options={"type": "data_bar", "bar_color": "#FFB628"},
        )

        sht.conditional_format(  # 信立坦贡献
            first_row=1,
            first_col=9,
            last_row=df.shape[0] - 1,
            last_col=9,
            options={"type": "data_bar", "bar_color": "#63C384"},
        )

        sht.conditional_format(  # 最后三列份额
            first_row=1,
            first_col=10,
            last_row=df.shape[0] - 1,
            last_col=12,
            options={"type": "data_bar", "bar_color": "#B1E1C1"},
        )

        wbk.close()

    def get_pivot(self, value, index, column, aggfunc):  # 透视表
        pivoted = pd.pivot_table(
            data=self,
            values=value,
            index=index,
            columns=column,
            aggfunc=aggfunc,
            fill_value=0,
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

        # 添加指定列的share作为折线图数据
        if line_share is not None:
            df_line = df_bar[line_share].fillna(0) / df_bar.sum(axis=1)
            df_line = df_line.to_frame()
            df_line.columns = [line_share]
        else:
            df_line = None

        # 是否换单位
        if unit_index is not None:
            df_bar = df_bar / D_UNIT[unit_index]

        # 是否只取top项，如只取top一些文本标签会变化
        label_prefix = "各"
        xtitle = index
        if top is not None:
            df_bar = df_bar.iloc[:top, :]
            if df_line is not None:
                df_line = df_line.iloc[:top, :]
            label_prefix = "TOP" + str(top)
            xtitle = label_prefix + index

        # 根据统计方式不同判断y轴标签及y轴显示逾限
        if aggfunc == len:
            ytitle = "终端数量"
        elif aggfunc == sum:
            ytitle = "潜力DOT"

        # 如果换过单位在y轴标题也要体现
        if unit_index is not None:
            ytitle = "%s（%s）" % (ytitle, unit_index)

        # 根据不同index决定x轴标签是否旋转90度
        if index == "潜力分位":
            xlabel_rotation = 0
        else:
            xlabel_rotation = 90

        # 是否显示y轴值标签
        if percentage:
            y1fmt = ""
            show_total = False
        else:
            y1fmt = "{:,.0f}"
            show_total = True

        # 图表标题
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

        # 是否换单位
        if unit_index is not None:
            df = df / D_UNIT[unit_index]

        # 根据统计方式不同判断y轴标签及y轴显示逾限
        if aggfunc == len:
            label = "终端数量占比"
        elif aggfunc == sum:
            label = "%s占比" % value

        # 图表标题
        title = "%s\n不同%s\n%s" % (
            self.name,
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

        # 是否换单位
        if unit_index is not None:
            df = df / D_UNIT[unit_index]

        # 是否只取top项，如只取top一些文本标签会变化
        label_prefix = "各"
        xtitle = index
        if top is not None:
            df_bar = df_bar.iloc[:top, :]
            df_line = df_line.iloc[:top, :]
            label_prefix = "TOP" + str(top)
            xtitle = label_prefix + index

        # 根据统计方式不同判断y轴标签及y轴显示逾限
        if aggfunc == len:
            ytitle = "终端数量"
        elif aggfunc == sum:
            ytitle = "潜力DOT"

        # 如果换过单位在标签也要体现
        if unit_index is not None:
            ytitle = "%s（%s）" % (ytitle, unit_index)

        # 根据不同index决定x轴标签是否旋转90度
        if index == "潜力分位":
            xlabel_rotation = 0
        else:
            xlabel_rotation = 90

        # 图表标题
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
        # 是否取对数
        if log_x:
            df_x = np.log(df_x)

        # 是否换单位
        if unit_index_x is not None:
            df_x = df_x / D_UNIT[unit_index_x]

        aggfunc2 = D_AGGFUNC[value_y]
        df_y = self.get_pivot(
            value=value_y,
            index=index,
            column=None,
            aggfunc=aggfunc2,
        )

        # 如果统计销售代表，一个lambda很难完成，需要进一步处理
        if value_y == "销售代表":
            value_y = "销售代表人数"
            df_y[value_y] = df_y["销售代表"].apply(
                lambda x: len(list(set([y for y in x.split(",") if str(y) != "nan"])))
            )  # 将销售代表list换算成人数
            df_y.drop("销售代表", axis=1, inplace=True)

        # 是否取对数
        if log_y:
            df_y = np.log(df_y)

        # 是否换单位
        if unit_index_y is not None:
            df_y = df_y / D_UNIT[unit_index_y]

        # print(df_x,df_y)
        df_combined = pd.concat([df_x, df_y], axis=1)
        df_combined.replace([np.inf, -np.inf, np.nan], 0, inplace=True)  # 所有异常值替换为0

        # 是否只取top项，如只取top一些文本标签会变化
        label_prefix = "各"
        if top is not None:
            df_combined = df_combined.iloc[:30, :]
            label_prefix = "TOP" + str(top)

        print(df_combined)

        # 轴标题
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

        # 图表标题
        title = "%s%s%s%s vs. %s" % (
            self.name,
            label_prefix,
            index,
            value_x,
            value_y,
        )

        # 绘图
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
