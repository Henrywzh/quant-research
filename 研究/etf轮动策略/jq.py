# 克隆自聚宽文章：https://www.joinquant.com/post/65761
# 标题：[已实盘]三马v10.3- 年化151% 回撤11%
# 作者：布宽量化

# 克隆自聚宽文章：https://www.joinquant.com/post/65582
# 标题：【讨论帖】三驾马车策略-今年涨177%回撤13%五年21倍
# 作者：烟花三月ETF

# 克隆自聚宽文章：https://www.joinquant.com/post/63661
# 标题：三马v10.2 测试框架
# 作者：Cibo

"""
Cibo 三驾马车优化版
策略1：小市值策略
策略2：ETF反弹策略 (只能测试 23.9月后, 2000etf上市时间为23.9)
策略3：ETF轮动策略
策略4：白马攻防 v2.0

写在最前面
实盘相关的指引:
https://www.joinquant.com/view/community/detail/4c8dda11f3ebda5ce562c2d3375a1740?type=1
相关的策略记录指引方便回溯
策略1 特么找不到了
策略2 https://www.joinquant.com/post/61536
策略3 https://www.joinquant.com/post/62083
策略4 https://www.joinquant.com/post/61890

迭代记录
v1.0 原始策略
v2.0 改进     策略1 小市值新增行业分散, 优化策略2 基础白马为市场温度攻防白马, 优化策略3 ETF动量轮动新增降幅 5% 三日检测
v2.1    新增  换手放量检测, 优化每日持仓打印, 优化卖出后的盈亏打印, 新增子策略收益独立展示
v2.2    新增   MCAD 大盘择时, 新增实盘启用配置
v3.0    优化  策略1 小市值策略双市值筛选
v4.0 改进     策略2 白马, 使用中证2000ETF下跌反弹进行替换
v4.1    修复  止损无法卖出的bug
v5.0 改进     策略1 小市值调整为群友迷妹的优化版本(类国九), 于v4.1相比, 长周期收益中幅降低回撤大幅降低, 短周期收益小幅降低, 回撤小幅增大
v5.1    新增  成交额宽度防御检测, 对比v5.0长回测减少收益(70->63)和回撤(19->14), 降低风险
v6.0 改进     策略2 中证2000ETF策略拓展, 持仓时间调整(2->5), 增加后备选项到5个
v6.1    优化  成交额检测新增缓存避免回测太久问题, 写死 18.1.1-25.10.10触发时间
v6.2    新增  移动止损功能, 以持仓周期中最高点收益作为成本价, (不适用小市值, 更适合大波段趋势)
v6.3    优化  交易时检查是否停牌, 停牌则不进行交易, 保持日志清洁
v6.4    优化  初次运行立即运行监测机制
v7.0    测试  策略1 小市值新增营收增长率 + 审计筛选 (舍弃)
v8.0 改进     新增策略4 白马攻防v2.0
v8.1    优化  动量计算整合(ETF轮动/白马攻防)
v9.0 改进     策略3 ETF轮动新增RSRS+均价检测, 成交量检测
v9.1    修复  检查涨停打开清除时索引异常问题处理
v9.2    修复  策略3 持仓记录列表存在重复情况, 检查是进行去重处理
v9.3    新增  策略3 日内止损相关检测
v9.4    修复  清仓时变更循环体bug，浅拷贝解决
v9.5    修复  策略4 白马v2融入可按比例回测实盘
v9.6    优化  策略3 尾盘新增ETF轮动最新排名打印方便明日参考
v9.7    优化  首次运行顶背离检测过去10天
v9.8    优化  优化持仓表格打印展示持仓比例, 按策略汇总
v9.9    修复  移动止损 bug 处理, 未能动态更新最高价格
v9.10   优化  策略1,策略3 买卖拆分可独立设计运行时间
v9.11   新增  策略2 基于2023.9.28 进行资金再平衡 (2023.9.28之前汇入策略3, 之后还原)
v9.12   新增  中金股指期货打印 (弃用, 转移到研究环境)
v9.13   优化  策略3 股票池更新 (自主替换)
v10.0 改进    策略1 支持多版本小市值策略选择
v10.1   优化  基础工具相关替换, 不再具备任何子策略通用工具, 避免频繁改动影响其他策略效果
"""
import datetime
import math
import prettytable
import numpy as np
import pandas as pd
from datetime import timedelta
from jqdata import *
from jqfactor import *
from prettytable import PrettyTable

# from nredistrade import *  # 导入实盘依赖

""" ====================== 基础配置 ====================== """


def initialize(context):
    set_backtest()  # 设置回测条件
    set_params(context)  # 设置基础参数
    set_strategy_params(context)  # 设置策略参数
    # setup_redis_trade(context, 'strategy1')  # 设置实盘

    # 过滤日志
    log.set_level('order', 'error')
    # log.set_level('system', 'error')
    # log.set_level('strategy', 'error')


# 基础参数设置
def set_params(context):
    # 策略名
    """
    注意:
    1. 白马v2是半成品开发内容, 放进来希望大家去优化了. 目前收益低回撤大, cibo本人是不会考虑实盘使用
    2. 对于白马v2是否使用依旧保持 三马v4.0 版本替换掉的结论
    3. ETF反弹核心标的在23.9月才上市, 回测过去周期策略失效
    4. 本策略预设的研究周期设计为 长:18-25, 中20-25, 短24-25, 早于18的 15/17 极端行情暂不做考虑
    """
    # g.portfolio_value_proportion = [0.35, 0.1, 0.35, 0.2]  # 小市值/ETF反弹/ETF轮动/白马攻防 (实盘)
    # g.portfolio_value_proportion = [0.4, 0.2, 0.4, 0]  # 小市值/ETF反弹/ETF轮动 (实盘/短回测)
    g.portfolio_value_proportion = [0.5, 0, 0.5, 0]  # 小市值/ETF轮动 (用于长回测)
    # g.portfolio_value_proportion = [0.35, 0, 0.35, 0.3]  # 小市值/ETF轮动/白马 (用于长回测)

    # g.portfolio_value_proportion = [1, 0, 0, 0]  # 测试小市值
    # g.portfolio_value_proportion = [0, 1, 0, 0]  # 测试ETF反弹
    # g.portfolio_value_proportion = [0, 0, 1, 0]  # 测试ETF轮动
    # g.portfolio_value_proportion = [0, 0, 0, 1]  # 测试白马攻防

    g.starting_cash = context.portfolio.total_value  # 策略初始资金, 用于计算子策略收益波动曲线
    g.stock_strategy = {}  # 记录股票对应的策略, 反向映射方便检索
    g.strategy_holdings = {1: [], 2: [], 3: [], 4: []}
    # 记录策略初始的金额, 用于计算各策略收益波动曲线
    g.strategy_starting_cash = {
        1: g.starting_cash * g.portfolio_value_proportion[0],  # 小市值 初始资金
        2: g.starting_cash * g.portfolio_value_proportion[1],  # ETF反弹 初始资金
        3: g.starting_cash * g.portfolio_value_proportion[2],  # ETF轮动 初始资金
        4: g.starting_cash * g.portfolio_value_proportion[3],  # 白马攻防 初始资金
    }
    # 记录每日策略收益
    g.strategy_value_data = {}
    g.strategy_value = {
        1: g.starting_cash * g.portfolio_value_proportion[0],  # 小市值 初始资金
        2: g.starting_cash * g.portfolio_value_proportion[1],  # ETF反弹 初始资金
        3: g.starting_cash * g.portfolio_value_proportion[2],  # ETF轮动 初始资金
        4: g.starting_cash * g.portfolio_value_proportion[3],  # 白马攻防 初始资金
    }
    # 暂存一个ETF反弹的初始比例
    g.strategy_ETF_2000_proportion = g.portfolio_value_proportion[1]
    g.strategy_ETF_2000_proportion_reset = None  # 用于检测拨正
    capital_balance_2(context)  # 首次就进行一次检测


# 策略参数设置
def set_strategy_params(context):
    """ 策略1 小市值 参数 """
    g.huanshou_check = False  # 放量换手检测，Ture是日频判断是否放量，False则不然
    g.xsz_version = "v3"  # 市值选用版本 可选值: v1/v2/v3 具体逻辑自己看代码吧, 写不下
    g.enable_dynamic_stock_num = True  # 启用动态选股数量 3~6
    g.xsz_stock_num = 5  # 默认的持股数量, 启用动态后会被覆盖为 3~6
    g.yesterday_HL_list = []  # 昨日涨停股票
    g.target_list = []  # 目标持仓股票
    g.xsz_buy_etf = "512800.XSHG"  # 空仓时购买ETF
    # 止损检查
    g.run_stoploss = True  # 是否进行止损
    g.stoploss_strategy = 3  # 1为止损线止损，2为市场趋势止损, 3为联合1、2策略
    g.stoploss_limit = 0.09  # 止损线
    g.stoploss_market = 0.05  # 市场趋势止损参数
    # 顶背离检查
    g.DBL_control = True  # 小市值大盘顶背离记录（用于风险控制）
    g.dbl = []
    g.check_dbl_days = 10  # 顶背离检测窗口期长度, 窗口内不仅买入
    # 异常处理窗口期检查
    g.check_after_no_buy = False  # 检查后不再买入时间
    g.no_buy_stocks = {}  # 检查卖出的股票
    g.no_buy_after_day = 3  # 止损后不买入的时间窗口
    # 成交额宽度检查
    g.check_defense = False  # 成交额宽度检查
    g.industries = ["组20"]  # 高位防御板块
    g.defense_signal = None
    g.cnt_defense_signal = []  # 择时次数
    g.cnt_bank_signal = []  # 组20择时次数
    g.history_defense_date_list = []

    """ 策略2 ETF反弹 参数 """
    g.limit_days = 2  # 最少持仓周期
    g.n_days = 5  # 持仓周期
    g.holding_days = 0
    g.buy_list = []
    # etf池子，优先级从高到低
    g.etf_pool_2 = [
        '159536.XSHE',  # 中证2000
        '159629.XSHE',  # 中证1000
        '159922.XSHE',  # 中证500
        '159919.XSHE',  # 沪深300
        '159783.XSHE'  # 双创50
    ]

    """ 策略3 ETF轮动 参数 """
    # 策略3全局变量
    g.etf_pool_3 = [
        # 多样性市场
        "510180.XSHG",  # 上证180ETF｜成立：2006-04-13（华安）
        "513030.XSHG",  # 德国DAX ETF｜成立：2014-08-08（华安）
        "513100.XSHG",  # 纳指ETF｜成立：2013-04-25（国泰）
        "513520.XSHG",  # 日经225ETF｜成立：2019-06-12（华夏）

        # 大宗商品
        "510410.XSHG",  # 上证自然资源ETF｜成立：2012-04-10（博时）
        "518880.XSHG",  # 黄金ETF｜成立：2013-07-18（华安）
        "501018.XSHG",  # 南方原油（LOF）｜成立：2016-06-15（南方）
        "159985.XSHE",  # 豆粕期货ETF｜成立：2019-09-24（华夏）
        "511090.XSHG",  # 30年期国债ETF｜成立：2023-05-19（鹏扬）

        # 科技成长
        "159915.XSHE",  # 创业板ETF｜成立：2011-09-20（易方达）
        "588120.XSHG",  # 科创100ETF｜成立：2023-09-06（国泰）
        "512480.XSHG",  # 半导体ETF｜成立：2019-05-08（国联安）
        "159851.XSHE",  # 金融科技ETF｜成立：2021-03-04（华宝）
        '513020.XSHG',  # 港股科技ETF | 成立 2022-01-19 （国泰）
        "159637.XSHE",  # 新能源车龙头ETF｜成立：2022-08-19（东财）

        # 蓝筹高股息
        "513690.XSHG",  # 港股红利/恒生高股息ETF｜成立：2021-05-11（博时）
        "510050.XSHG",  # 上证180ETF

        # "159980.XSHE",  # 有色期货ETF｜成立：2019-10-24（大成）  # 重复了
        # "516160.XSHG",  # 新能源ETF｜成立：2021-01-22（南方） # 重复了
        # "513130.XSHG",  # 恒生科技ETF｜成立：2021-05-24（华泰柏瑞） # 重复了

        # "512710.XSHG",  # 军工龙头ETF｜成立：2019-07-23（富国）  # 太猛
        # "159692.XSHE",  # 证券ETF（东财）｜成立：2023-05-05  # 太猛
        # "515070.XSHG",  # 人工智能ETF｜成立：2019-12-09（华夏）  # 太猛

        # "515650.XSHG",  # 消费50ETF｜成立：2019-10-14（富国） # 太弱
        # "159550.XSHE",  # 互联网ETF（沪港深）｜成立：2025-06-18（东财） # 太弱
        # "512290.XSHG",  # 生物医药ETF｜成立：2019-04-18（国泰） # 太弱
    ]
    g.select_etf = None  # ETF交易传递变量
    g.m_days = 25  # 动量参考天数
    g.m_score = 5  # 动量过滤分数
    g.enable_stop_loss_by_cur_day = True  # 是否开启日内止损
    g.stoploss_limit_by_cur_day = -0.3  # 当日亏损 -3%

    """ 策略4 白马攻防 参数 """
    g.check_out_lists = []
    g.market_temperature = "warm"
    g.stock_num_2 = 5  # 目标持股数量
    g.roe = 10  # ROE权重
    g.roa = 6  # ROA权重


# 回测设置
def set_backtest():
    set_option('avoid_future_data', True)
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)

    set_slippage(FixedSlippage(0.002), type="stock")
    set_slippage(FixedSlippage(0.001), type="fund")
    cost_configs = [
        ("stock", 0.0005, 0.85 / 10000, 5),
        ("fund", 0, 0.5 / 10000, 5),
        ("mmf", 0, 0, 0)
    ]
    for asset_type, close_tax, commission, min_comm in cost_configs:
        set_order_cost(OrderCost(
            open_tax=0, close_tax=close_tax,
            open_commission=commission, close_commission=commission,
            close_today_commission=0, min_commission=min_comm
        ), type=asset_type)


""" ====================== 策略1: 小市值 ====================== """


# v1 选股模块 (双市值+行业分散)
def xsz_get_stock_list_v1(context):
    # 获取股票所属行业
    def filter_industry_stock(stock_list):
        result = get_industry(security=stock_list)
        selected_stocks = []
        industry_list = []
        for stock_code, info in result.items():
            industry_name = info['sw_l2']['industry_name']
            if industry_name not in industry_list:
                industry_list.append(industry_name)
                selected_stocks.append(stock_code)
                print(f"行业信息: {industry_name} (股票: {stock_code} {get_security_info(stock_code).display_name})")
                # 选取了 10 个不同行业的股票
                if len(industry_list) == 10:
                    break
        return selected_stocks

    initial_list = filter_stocks(context, get_index_stocks('399101.XSHE'))

    # 获取流通市值最小的50个股票
    q = query(valuation.code).filter(valuation.code.in_(initial_list)).order_by(
        valuation.circulating_market_cap.asc()).limit(50)
    initial_list = list(get_fundamentals(q).code)
    # 选取每股收益>0的股票
    # q = query(valuation.code, indicator.eps) \
    #     .filter(valuation.code.in_(initial_list)) \
    #     .filter(indicator.eps > 0) \
    #     .filter(valuation.market_cap > g.min_mv) \
    #     .filter(valuation.market_cap < g.max_mv) \
    #     .order_by(valuation.market_cap.asc())

    q = query(valuation.code).filter(valuation.code.in_(initial_list)).order_by(valuation.market_cap.asc())
    initial_list = list(get_fundamentals(q).code)
    initial_list = initial_list[:30]
    # 每个行业获取1个股票，总共获取g.stock_num个行业的股票
    final_list = filter_industry_stock(initial_list)[:g.xsz_stock_num]
    print('选出的股票:%s' % [f"{i} {get_security_info(i).display_name}" for i in final_list])
    return final_list


# v2 选股模块 (国九+roa+roe)
def xsz_get_stock_list_v2(context):
    initial_list = filter_stocks(context, get_index_stocks('399101.XSHE'))

    # 修复：正确使用聚宽基本面表查询方式
    q = query(
        valuation.code,
        valuation.market_cap,
        income.np_parent_company_owners,
        income.net_profit,
        income.operating_revenue,
        valuation.turnover_ratio
    ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(5, 50),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
        fundamentals.indicator.roe > 0.15,
        fundamentals.indicator.roa > 0.10,
    ).order_by(valuation.market_cap.asc()).limit(50)
    df = get_fundamentals(q)
    if df.empty:
        return []
    final_list = list(df.code)
    last_prices = history(1, '1d', 'close', final_list, df=False)
    # 价格过滤
    return [stock for stock in final_list if stock in context.portfolio.positions or last_prices[stock] <= 20][
           :g.xsz_stock_num]


# v3 选股模块 (国九+红利+审计)
def xsz_get_stock_list_v3(context):
    initial_list = filter_stocks(context, get_index_stocks('399101.XSHE'))

    q = query(
        valuation.code,
        valuation.market_cap,  # 总市值 circulating_market_cap/market_cap
        # income.np_parent_company_owners,  # 归属于母公司所有者的净利润
        income.net_profit,  # 净利润
        income.operating_revenue  # 营业收入
        # security_indicator.net_assets
    ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(10, 100),
        # income.np_parent_company_owners > g.np_parent_company_owners_min,
        income.operating_revenue > 1e8,
        indicator.roe > 0,
        indicator.roa > 0,
        income.net_profit > 2000000,  # 净利润大于0
    ).order_by(valuation.market_cap.asc()).limit(g.xsz_stock_num * 5)
    final_list = list(get_fundamentals(q).code)
    # 过滤审计意见
    final_list = filter_audit(context, final_list)
    # 过滤红利股
    final_list = bonus_filter(context, final_list)
    # 由于有时候选股条件苛刻，所以会没有股票入选，这时买入银华日利ETF
    if not final_list:
        print('无适合股票，买入ETF')
        return [g.xsz_buy_etf]
    # 价格过滤
    last_prices = history(1, unit='1d', field='close', security_list=final_list)
    return [s for s in final_list if s in g.strategy_holdings[1] or last_prices[s][-1] <= 50]


# 小市值早盘变量预处理
def prepare_xsz(context):
    g.trading_signal = False if context.current_dt.month in [1, 4] else True
    g.yesterday_HL_list = []
    # 获取昨日涨停列表
    if g.strategy_holdings[1]:
        df = get_price(g.strategy_holdings[1],
                       end_date=context.previous_date,
                       fields=['close', 'high_limit', 'low_limit'],
                       frequency='daily',
                       count=1,
                       panel=False,
                       fill_paused=False)
        g.yesterday_HL_list = list(df[df['close'] == df['high_limit']].code)


# 小市值卖出
def strategy_1_sell(context):
    print("-" * 45 + f"{str(context.current_dt.date())}" + "-" * 45)
    g.target_list = []
    # 近期有顶背离信号时暂停调仓（规避系统性风险）
    if g.DBL_control:
        # 首次运行检测最近10日顶背离
        if len(g.dbl) < 10:
            for i in range(9, -1, -1):
                check_dbl(context, end_days=0 - i)
    if g.DBL_control and 1 in g.dbl[-g.check_dbl_days:]:
        print(f"近{g.check_dbl_days}日检测到大盘顶背离，暂停调仓以控制风险")
        return

    # 检测空仓期
    month = context.current_dt.month
    if month in [1, 4]:
        g.trading_signal = False
    if not g.trading_signal:
        return

    # 成交额宽度检查
    if g.check_defense and g.defense_signal:
        print("触发成交额宽度检查信号，暂停调仓以控制风险")
        return

    # 动态调整选股数量
    diff = None
    if g.enable_dynamic_stock_num:
        ma_para = 10  # 设置MA参数
        today = context.previous_date
        start_date = today - timedelta(days=ma_para * 2)
        index_df = get_price('399101.XSHE', start_date=start_date, end_date=today, frequency='daily')
        index_df['ma'] = index_df['close'].rolling(window=ma_para).mean()
        last_row = index_df.iloc[-1]
        diff = last_row['close'] - last_row['ma']
        g.xsz_stock_num = 3 if diff >= 500 else \
            3 if 200 <= diff < 500 else \
                4 if -200 <= diff < 200 else \
                    5 if -500 <= diff < -200 else \
                        6
    # 选择要启用的选股版本
    g.target_list = {
                        "v1": xsz_get_stock_list_v1,
                        "v2": xsz_get_stock_list_v2,
                        "v3": xsz_get_stock_list_v3,
                    }[g.xsz_version](context)[:g.xsz_stock_num]
    print(f'小市值 {g.xsz_version} 目标持股数: {g.xsz_stock_num} [diff:{str(diff)[:6]}] 目标持仓: {g.target_list}')

    # 卖出不在目标列表中的股票（除昨日涨停股）
    sell_list = [s for s in g.strategy_holdings[1] if s not in g.target_list and s not in g.yesterday_HL_list]
    hold_list = [s for s in g.strategy_holdings[1] if s in g.target_list or s in g.yesterday_HL_list]

    if sell_list:
        hold_list and print("当前持有 %s" % ([format_stock_code(stock) for stock in hold_list]))
        sell_list and print("计划卖出 %s" % ([format_stock_code(stock) for stock in sell_list]))
    for stock in sell_list:
        close_position(stock)

    # current_data = get_current_data()
    # for stock in sell_list:
    #     current_stock_data = current_data[stock]
    #     if current_stock_data.paused:
    #         print(f"⭕ {stock} 停牌, 无法卖出")
    #     elif current_stock_data.last_price > current_stock_data.high_limit * 0.99:  # 涨幅超过 9.9% 涨停未打开
    #         print(f"⭕ {stock} 涨停, 不进行卖出")
    #     else:
    #         close_position(stock)


# 小市值买入
def strategy_1_buy(context):
    if not g.trading_signal:
        if g.xsz_buy_etf not in context.portfolio.positions:
            print("小市值清仓时期, 买入ETF")
            open_position(context, g.xsz_buy_etf, context.portfolio.total_value * g.portfolio_value_proportion[0], 1)
        return

    # 计算可用资金（策略1专用部分）
    strategy_value = context.portfolio.total_value * g.portfolio_value_proportion[0]
    current_value = sum(
        [pos.value for pos in context.portfolio.positions.values() if pos.security in g.strategy_holdings[1]])
    available_cash = max(0, strategy_value - current_value)  # 确保非负

    # 买入新标的
    buy_list = [s for s in g.target_list if s not in g.strategy_holdings[1][:]]
    if buy_list and available_cash > 0:
        cash_per_stock = available_cash / len(buy_list)
        for stock in buy_list:
            open_position(context, stock, cash_per_stock, 1)


# 清仓后次日资金可转
def close_account(context):
    if not g.trading_signal:
        if g.strategy_holdings[1] and g.xsz_buy_etf not in g.strategy_holdings[1]:
            for stock in g.strategy_holdings[1][:]:
                print(f"🤕🤕🤕 进入清仓期间 卖出 {format_stock_code(stock)}")
                close_position(stock)


# 检查昨日涨停股今日表现
def xsz_check_limit_up(context):
    # 获取当前持仓
    holdings = g.strategy_holdings[1][:]  # 只检查策略1
    if holdings:
        now_time = context.current_dt
        if g.yesterday_HL_list:
            # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
            for stock in g.yesterday_HL_list:
                current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                                         skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
                if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                    print(f"🥵🥵🥵 {format_stock_code(stock)} 涨停打开，卖出")
                    close_position(stock)
                else:
                    print(f"🤗🤗🤗 {stock} 继续涨停，继续持有")


# 止盈止损
def xsz_sell_stocks(context):
    if g.run_stoploss:
        current_positions = context.portfolio.positions
        if g.stoploss_strategy in [1, 3]:
            for stock in current_positions.keys():
                if stock in g.strategy_holdings[1]:
                    price = current_positions[stock].price
                    avg_cost = current_positions[stock].avg_cost
                    # 个股盈利止盈
                    if price >= avg_cost * 2:
                        print(f"🤑🤑🤑 收益100%止盈,卖出 {format_stock_code(stock)}")
                        close_position(stock)
                    # 个股止损
                    elif price < avg_cost * (1 - g.stoploss_limit):
                        print(f"🤬🤬🤬 收益止损,卖出 {format_stock_code(stock)}")
                        close_position(stock)
        if g.stoploss_strategy in [2, 3]:
            stock_df = get_price(security=get_index_stocks('399101.XSHE'),
                                 end_date=context.previous_date,
                                 frequency='daily',
                                 fields=['close', 'open'],
                                 count=1,
                                 panel=False)
            down_ratio = abs((stock_df['close'] / stock_df['open'] - 1).mean())
            # 市场大跌止损
            if down_ratio >= g.stoploss_market:
                print(f"🤡🤡🤡 大盘惨跌,平均降幅 {down_ratio:.2%}")
                for stock in g.strategy_holdings[1][:]:
                    close_position(stock)


""" ====================== 策略2: ETF反弹 ====================== """


# 原始中证2000策略
def zz_2000_trade(context):
    to_buy = False
    etf_index = "159536.XSHE"
    # 获取近3日的历史数据
    df = get_price(etf_index, end_date=context.previous_date, count=3, frequency='daily', fields=['high'])
    df = df.reset_index()
    if len(df) < 3:
        return

    pre3_high_max = df['high'].max()

    # 获取当前盘中实时数据
    current_data = get_current_data()
    today_open = current_data[etf_index].day_open
    today_close = current_data[etf_index].last_price

    # 策略条件判断，开盘相比最高价下跌2% & 最新价相比开盘价涨1%
    if today_open / pre3_high_max < 0.98 and today_close / today_open > 1.01:
        to_buy = True

    # 已经持仓, 检查是否继续持有
    if etf_index in context.portfolio.positions:
        position = context.portfolio.positions[etf_index]
        trade_date = position.init_time
        holding_days = len(get_trade_days(start_date=trade_date, end_date=context.current_dt)) - 1
        # 不符合却持仓超过2天, 清仓
        if not to_buy and holding_days >= 2:
            close_position(etf_index)
            print(f"卖出：{etf_index}, 持仓{holding_days}天")
    # 未持仓, 但符合条件, 进行买入
    elif to_buy:
        strategy_value = context.portfolio.total_value * g.portfolio_value_proportion[1]
        open_position(context, etf_index, strategy_value, 2)
        print(f"符合中证2000买入条件：{etf_index}")


def strategy_2_sell(context):
    cur_date = str(context.current_dt.date())
    if cur_date <= "2023-10-01":
        return

    g.buy_list = []
    sell_list = []
    sell_for_money_list = []
    # 获取近3日的历史数据
    for etf in g.etf_pool_2:
        df = get_price(etf, end_date=context.previous_date, count=4, frequency='daily', fields=['high', 'close'])
        df = df.reset_index()
        if len(df) < 4:
            return
        pre_high_max = df['high'].max()
        yestoday_close = df['close'].iloc[-1]
        # 获取当前盘中实时数据
        current_data = get_current_data()
        today_open = current_data[etf].day_open
        today_close = current_data[etf].last_price
        # 买入条件判断，开盘相比最高价下跌2% & 最新价相比开盘价涨1%
        if today_open / pre_high_max < 0.98 and today_close / today_open > 1.01:
            g.buy_list.append(etf)
        # 卖出条件判断，当前价格小于昨日收盘价
        if today_close < yestoday_close:
            sell_list.append(etf)

    # 保留最佳标的
    if g.buy_list:
        g.buy_list.sort(key=lambda x: g.etf_pool_2.index(x))
        selected_etf = g.buy_list[0]
        g.buy_list = [selected_etf]
        current_holdings = g.strategy_holdings[2]
        if current_holdings and g.etf_pool_2.index(current_holdings[0]) < g.etf_pool_2.index(selected_etf):
            # 如果有持仓，且持有的ETF不是高优先级ETF，则清仓
            sell_for_money_list.append(current_holdings[0])

    for etf in g.strategy_holdings[2]:
        position = context.portfolio.positions[etf]
        security = position.security  # 股票代码
        trade_date = position.init_time
        holding_days = len(get_trade_days(start_date=trade_date, end_date=context.current_dt)) - 1
        if (security in sell_list and holding_days >= g.limit_days) or (holding_days >= g.n_days) or \
                (security in sell_for_money_list):
            close_position(security)
            log.info(f"卖出：{security}， 持股{security} {holding_days}天")
    if not g.buy_list:
        print(f"策略2今日无反弹可购买选项")


def strategy_2_buy(context):
    cur_date = str(context.current_dt.date())
    if cur_date <= "2023-10-01":
        return

    g.buy_list = list(set(g.buy_list) - set(g.strategy_holdings[2]))
    if len(g.buy_list) > 0:
        cash = context.portfolio.total_value * g.portfolio_value_proportion[1]
        if cash < 100:
            log.warn(f'cash不足:{context.portfolio.available_cash}')
        else:
            cash = context.portfolio.total_value * g.portfolio_value_proportion[1]
            for etf in g.buy_list:
                print(f"符合策略2买入条件：{etf}")
                open_position(context, etf, cash, 2)


""" ====================== 策略3: ETF轮动 ====================== """


# 动量计算
def filter_moment_rank(stock_pool, days, ll, hh, show_print=True):
    print("计算 动量得分" + "*" * 60)

    scores_data = pd.DataFrame(index=stock_pool, columns=["annualized_returns", "r2", "score"])
    current_data = get_current_data()
    print_info = {}

    for code in stock_pool:
        try:
            hist_data = attribute_history(code, days, "1d", ["close", "high"])
            if hist_data.empty:
                continue

            prices = np.append(hist_data["close"].values, current_data[code].last_price)
            log_prices = np.log(prices)
            x_values = np.arange(len(log_prices))
            weights = np.linspace(1, 2, len(log_prices))

            slope, intercept = np.polyfit(x_values, log_prices, 1, w=weights)
            annualized_return = math.exp(slope * 250) - 1
            scores_data.loc[code, "annualized_returns"] = annualized_return

            ss_res = np.sum(weights * (log_prices - (slope * x_values + intercept)) ** 2)
            ss_tot = np.sum(weights * (log_prices - np.mean(log_prices)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            scores_data.loc[code, "r2"] = r2

            momentum_score = annualized_return * r2
            scores_data.loc[code, "score"] = momentum_score

            if min(prices[-1] / prices[-2], prices[-2] / prices[-3],
                   prices[-3] / prices[-4]) < 0.97:
                scores_data.loc[code, "score"] = 0
            print_info[code] = scores_data.loc[code, "score"]

        except Exception as e:
            print(f"计算{code}动量得分失败: {e}")
            scores_data.loc[code, "score"] = 0

    valid_etfs = scores_data[(scores_data['score'] > ll) & (scores_data['score'] < hh)] \
        .sort_values("score", ascending=False)
    rank_list = valid_etfs.index.tolist()
    if show_print and rank_list:
        for i in rank_list:
            print(f"{format_stock_code(i)} ({print_info[i]:.4f})")
    return rank_list


# 成交量过滤
def filter_volume(context, stock_list, days=7, volume_threshold=2):
    """
    :param context:
    :param stock_list: 要检测的股票
    :param days: 检测周期天数
    :param volume_threshold: 阈值
    :return:
    """
    print("ETF 异常成交量检测" + "*" * 60)

    def _get_volume_ratio(security):

        try:
            hist_data = attribute_history(security, days, '1d', ['volume'])
            if hist_data.empty or len(hist_data) < days:
                return
            avg_volume = hist_data['volume'].mean()
            df_vol = get_price(security, start_date=context.current_dt.date(), end_date=context.current_dt,
                               frequency='1m', fields=['volume'], skip_paused=False, fq='pre', panel=True,
                               fill_paused=False)
            if df_vol is None or df_vol.empty:
                return
            current_volume = df_vol['volume'].sum()
            _volume_ratio = current_volume / avg_volume
            # 检测到异常, 返回异常倍数
            if _volume_ratio > volume_threshold:
                print(f"❌ {security} 成交量较近{days}日均值 x{_volume_ratio:.2f}")
                return _volume_ratio
            print(f"✔️ {security} 成交量较近{days}日均值 x{_volume_ratio:.2f}")

        except Exception as e:
            print(f"⭕ 检查{security}成交量失败: {e}")
            return

    res = []
    for stock in stock_list:
        ratio = _get_volume_ratio(stock)
        if not ratio:
            res.append(stock)

    return res


# RSRS 均线过滤
def filter_rsrs(stock_list):
    print("ETF RSRS+均线过滤 " + "*" * 60)

    # 计算斜率
    def _get_slope(security, days=18):
        try:
            hist_data = attribute_history(security, days, '1d', ['high', 'low'])
            if hist_data.empty or len(hist_data) < days:
                return None
            slope = np.polyfit(hist_data['low'].values, hist_data['high'].values, 1)[0]
            return slope
        except Exception as e:
            print(f"计算{security} RSRS斜率失败: {e}")
            return None

    # 计算阈值
    def _get_beta(security, lookback_days=250, window=20):
        try:
            hist_data = attribute_history(security, lookback_days, '1d', ['high', 'low'])
            if hist_data.empty or len(hist_data) < lookback_days:
                return

            slope_list = []
            for i in range(len(hist_data) - window + 1):
                window_data = hist_data.iloc[i:i + window]
                low_values = window_data['low'].values
                high_values = window_data['high'].values

                if len(low_values) < window or len(high_values) < window:
                    continue
                if np.any(np.isnan(low_values)) or np.any(np.isnan(high_values)):
                    continue
                if np.any(np.isinf(low_values)) or np.any(np.isinf(high_values)):
                    continue
                if np.std(low_values) == 0 or np.std(high_values) == 0:
                    continue

                slope = np.polyfit(low_values, high_values, 1)[0]
                slope_list.append(slope)

            if len(slope_list) < 2:
                return None

            mean_slope = np.mean(slope_list)
            std_slope = np.std(slope_list)
            beta = mean_slope - 2 * std_slope
            return beta
        except Exception as e:
            print(f"计算{security} RSRS Beta失败: {e}")
            return None

    # 计算强度
    def _check_with_strength(security):
        _slope = _get_slope(security)
        _beta = _get_beta(security)
        if _slope is None or _beta is None:
            return None, 0
        _strength = (_slope - _beta) / abs(_beta) if _beta != 0 else 0
        return _slope > _beta, _strength

    # 计算均值
    def _check_above_ma(security, days=20):
        try:
            hist = attribute_history(security, days, "1d", ["close"])
            if len(hist) < days:
                return False
            current_price = get_current_data()[security].last_price
            return current_price >= hist["close"].mean()
        except Exception as e:
            print(f"计算{security} {days}日均线失败: {e}")
            return False

    res = []
    for stock in stock_list:
        stock_pass, stock_strength = _check_with_strength(stock)
        above_ma_5 = _check_above_ma(stock, 5)
        above_ma_10 = _check_above_ma(stock, 10)
        flag = "❌"
        if stock_pass:
            if stock_strength > 0.15:
                flag = "✔️"
                res.append(stock)
            elif stock_strength > 0.03 and above_ma_5:
                flag = "✔️"
                res.append(stock)
            elif above_ma_10:
                flag = "✔️"
                res.append(stock)
        print(f"{flag}  {format_stock_code(stock)} "
              f"pass:{stock_pass}  strength:{stock_strength:.2f} "
              f"ma5: {above_ma_5}  ma10: {above_ma_5}")
    return res


def get_etf_rank(context, etf_pool):
    rank_list = []
    # 过滤近3日跌幅超过5%的ETF
    current_data = get_current_data()
    print("ETF 跌幅检测 " + "*" * 60)
    for etf in etf_pool:
        df = attribute_history(etf, g.m_days, "1d", ["close", "high"])
        prices = np.append(df["close"].values, current_data[etf].last_price)
        if min(prices[-1] / prices[-2],
               prices[-2] / prices[-3],
               prices[-3] / prices[-4]) < 0.95:
            print(f"❌ {format_stock_code(etf)} 近3日跌幅超过5%, 已排除")
            continue
        # 日内止损, 距离开盘暴跌的不进行买入
        if g.enable_stop_loss_by_cur_day:
            ratio = cal_cur_to_open_ratio(etf)
            if ratio <= g.stoploss_limit_by_cur_day:
                print(f"❌ {format_stock_code(etf)} 进入跌幅达到 {ratio * 100:.2f}%, 已排除")
                continue
        print(f"✔️ {format_stock_code(etf)} 检测通过")
        rank_list.append(etf)
    # 过滤 RSRS + 均值
    rank_list = filter_rsrs(rank_list)
    # 过滤成交量异常
    rank_list = filter_volume(context, rank_list)
    # 过滤 动量得分, ( 0 ~ 5 )
    rank_list = filter_moment_rank(rank_list, g.m_days, 0, g.m_score)

    res_list = []
    # 过滤 RSI
    for etf in rank_list:
        # 过热的不买入
        if calculate_rsi(etf) < 80:
            res_list.append(etf)
    return res_list


def strategy_3_sell(context):
    # 获取动量最高的ETF
    rank_df = get_etf_rank(context, g.etf_pool_3)

    # 选不出来合适的就清仓
    if not rank_df:
        for current_etf in g.strategy_holdings[3]:
            print("👿👿👿👿👿 ETF轮动没有一个能打的, 清仓当前持仓")
            close_position(current_etf)
            g.strategy_holdings[3] = []
        return
    g.buy_etf = None
    select_etf = rank_df[0]
    current_etf = None

    # 检查当前持仓
    for asset in context.portfolio.positions:
        if asset in g.etf_pool_3:
            current_etf = asset
            break

    # 策略3专用资金
    strategy_cash = context.portfolio.total_value * g.portfolio_value_proportion[2]

    # 需要调仓的情况
    if current_etf and current_etf != select_etf:
        print(f"ETF轮动调仓: {current_etf} -> {select_etf}")
        close_position(current_etf)  # 卖掉当前的
        g.buy_etf = select_etf
        # open_position(context, g.select_etf, strategy_cash, 3)  # 买入新的
    # 首次买入或恢复持仓
    elif not current_etf and strategy_cash > 0:
        g.buy_etf = select_etf
        print(f"ETF轮动建仓: {select_etf}")
        # open_position(context, select_etf, strategy_cash, 3)  # 买入新的
    g.strategy_holdings[3] = list(set(g.strategy_holdings[3]))


def strategy_3_buy(context):
    if g.buy_etf:
        strategy_cash = context.portfolio.total_value * g.portfolio_value_proportion[2]
        open_position(context, g.buy_etf, strategy_cash, 3)  # 买入新的
    g.strategy_holdings[3] = list(set(g.strategy_holdings[3]))


""" ====================== 策略4: 白马攻防 ====================== """


def bm_adjust_position(context):
    if not g.check_out_lists:
        bm_before_market_open(context)
    buy_stocks = g.check_out_lists
    print(f"白马目标调仓: {','.join([f'{format_stock_code(i)}' for i in buy_stocks])}")
    # 卖出不在目标列表中的股票（只处理本策略持仓）
    for stock in g.strategy_holdings[4][:]:
        current_data = get_current_data()
        # 不在买入列表则卖出
        if stock not in buy_stocks:
            # 涨停无法卖出时跳过
            if current_data[stock].last_price >= current_data[stock].high_limit:
                continue
            close_position(stock)
            print(f"白马策略调出: {stock}")

    # 买入新标的
    position_count = len([s for s in context.portfolio.positions.keys()
                          if s in g.strategy_holdings[4]])
    if len(buy_stocks) > position_count:
        # 使用策略4专用资金
        value = context.portfolio.total_value * g.portfolio_value_proportion[3] / g.stock_num_2
        for stock in buy_stocks:
            if stock not in g.strategy_holdings[4]:
                if open_position(context, stock, value, 4):
                    if len(g.strategy_holdings[4]) >= g.stock_num_2:
                        break


# 市场温度判断
def cal_market_temperature(context):
    # 数据回滚两年判断市场温度
    if not hasattr(g, 'market_temperature'):
        long_index300 = list(attribute_history('000300.XSHG', 220 * 3, '1d', ('close',), df=False)['close'])
        g.market_temperature = 'cold'
        for back_day in range(220, len(long_index300)):
            index300 = long_index300[back_day - 220:back_day]
            market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))
            if market_height < 0.20:
                g.market_temperature = "cold"
            elif market_height > 0.80:
                g.market_temperature = "hot"
            elif max(index300[-60:]) / min(index300) > 1.20:
                g.market_temperature = "warm"
    # 当前一年的温度判断
    index300 = attribute_history('000300.XSHG', 220, '1d', ('close',), df=True) \
        .drop(pd.to_datetime("2024-10-08"), errors='ignore')
    index300 = index300['close'].tolist()
    market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))
    if market_height < 0.20:
        g.market_temperature = "cold"
    elif index300[-1] == min(index300):
        g.market_temperature = "cold"
    elif market_height > 0.90:
        g.market_temperature = "hot"
    elif index300[-1] == max(index300):
        g.market_temperature = "hot"
    elif max(index300[-60:]) / min(index300) > 1.20:
        g.market_temperature = "warm"


# 开盘前运行函数
def bm_before_market_open(context):
    cal_market_temperature(context)
    g.check_out_lists = []
    current_data = get_current_data()
    all_stocks = get_index_stocks("000300.XSHG")  # 以沪深300成分股味股票池进一步筛选
    # 过滤创业板、ST、停牌、当日涨停
    all_stocks = [stock for stock in all_stocks if not (
            (current_data[stock].last_price > round(
                context.portfolio.total_value * g.portfolio_value_proportion[0] * 0.95 / g.stock_num_2 / 100,
                2)) or  # 股价限高
            (current_data[stock].day_open == current_data[stock].high_limit) or  # 涨停开盘
            (current_data[stock].day_open == current_data[stock].low_limit) or  # 跌停开盘
            current_data[stock].paused or  # 停牌
            current_data[stock].is_st or  # ST
            ('ST' in current_data[stock].name) or  # ST
            ('*' in current_data[stock].name) or
            ('退' in current_data[stock].name) or
            (stock.startswith('30')) or  # 创业
            (stock.startswith('68')) or  # 科创
            (stock.startswith('8')) or  # 北交
            (stock.startswith('4'))  # 北交
    )]
    last_prices = history(1, unit='1d', field='close', security_list=all_stocks)
    all_stocks = [stock for stock in all_stocks if last_prices[stock][-1] <= 100]  # 过滤高价股

    q = None
    if g.market_temperature == "cold":
        q = query(
            valuation.code,
            indicator.roe,
            indicator.roa
        ).filter(
            valuation.pb_ratio > 0,
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 2.0,
            indicator.inc_return > 1.5,
            indicator.inc_net_profit_year_on_year > -15,
            valuation.code.in_(all_stocks)
        ).order_by(
            (indicator.roa / valuation.pb_ratio).desc()
        ).limit(
            50
        )
    elif g.market_temperature == "warm":
        q = query(
            valuation.code,
            indicator.roe,
            indicator.roa
        ).filter(
            valuation.pb_ratio > 0,
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
            indicator.inc_return > 2.0,
            indicator.inc_net_profit_year_on_year > 0,
            valuation.code.in_(all_stocks)
        ).order_by(
            (indicator.roa / valuation.pb_ratio).desc()
        ).limit(
            50
        )
    elif g.market_temperature == "hot":
        q = query(
            valuation.code,
            indicator.roe,
            indicator.roa
        ).filter(

            valuation.pb_ratio > 3,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 0.5,
            indicator.inc_return > 3.0,
            indicator.inc_net_profit_year_on_year > 20,
            valuation.code.in_(all_stocks)
        ).order_by(
            indicator.roa.desc()
        ).limit(
            50  # *10
        )

    """*****************************************************************************************"""
    df = get_fundamentals(q)
    df.index = df['code'].values

    # 按照因子给股票排序（相当于各因子平权）
    # pb_rank= df['pb_ratio'].rank(ascending=True)  # 升序排名（pb越低越好）
    roe_inv_rank = df['roe'].rank(ascending=False)  # 降序排名（roe越高越好）
    roa_inv_rank = df['roa'].rank(ascending=False)  # 降序排名（roa越高越好）

    # 应用权重计算综合得分
    df['point'] = (g.roe * roe_inv_rank + g.roa * roa_inv_rank)

    # 按得分进行排序，取指定数量的股票
    df = df.sort_values(by='point')  # [:g.buy_stock_count]

    check_out_lists = list(df.code)
    """*****************************************************************************************"""
    # 动量趋势过滤，剔除太高和太低的
    check_out_lists2 = moment_rank(check_out_lists, 25, -1.0, 10.5)
    # 顺序还是按照动量趋滤前原来的顺序
    check_out_lists = [x for x in check_out_lists if x in check_out_lists2]
    g.check_out_lists = check_out_lists[:g.stock_num_2]
    print("今日市场温度：%s" % g.market_temperature)
    print("今日白马股票池：%s" % g.check_out_lists)


# 动量计算
def moment_rank(stock_pool, days, ll, hh):
    # - 对股票近days天的收盘价取对数，进行加权线性回归（近期权重高）。
    # - 计算年化收益率（指数化斜率）和R平方（趋势强度）。
    # - 动量得分 = 年化收益率×R平方。

    def mom(_stock):
        y = np.log(attribute_history(_stock, days, '1d', ['close'], df=False)['close'])
        n = len(y)
        x = np.arange(n)
        weights = np.linspace(1, 2, n)
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        residuals = y - (slope * x + intercept)
        weighted_residuals = weights * residuals ** 2
        r_squared = 1 - (np.sum(weighted_residuals) / np.sum(weights * (y - np.mean(y)) ** 2))
        return annualized_returns * r_squared

    score_list = []
    for stock in stock_pool:
        score = mom(stock)
        score_list.append(score)
    df = pd.DataFrame(index=stock_pool, data={'score': score_list})
    df = df.sort_values(by='score', ascending=False)  # 降序
    df = df[(df['score'] > ll) & (df['score'] < hh)]
    rank_list = list(df.index)
    return rank_list


""" ====================== 辅助的定时执行函数 ====================== """


# 大盘顶背离
def check_dbl(context, market_index='399101.XSHE', end_days=0):
    """
        大盘顶背离检测：通过MACD判断市场潜在反转风险
        目的：在大盘出现顶背离（上涨乏力）时提前减仓，规避系统性下跌
    """
    # 把第一次9:31执行的给忽略掉, 第一次9:49会回溯过去10天, 避免第一次造成干扰(其实也不会, 但看日志会看的不会有困惑)
    if not g.dbl and "9:31" in str(context.current_dt.time()):
        return

    def detect_divergence():
        """检测顶背离（价格新高但MACD指标走弱，预示趋势反转）
        条件：
        1. 价格创新高（后高>前高）
        2. MACD指标未创新高（后低<前低）
        3. MACD由正转负（趋势转弱）
        4. DIF处于下降趋势（近期均值<前期均值）
        """
        fast, slow, sign = 12, 26, 9  # MACD参数
        rows = (fast + slow + sign) * 5  # 确保足够数据量（约1年）
        # 获取历史收盘价数据
        grid = attribute_history(market_index, rows + 10, fields=['close']).dropna()
        if end_days < 0:
            grid = grid.iloc[:end_days]

        if len(grid) < rows:
            print(f"{market_index} 数据不足 {rows} 天，无法检测顶背离")
            return False

        try:
            # 计算MACD指标
            grid['dif'], grid['dea'], grid['macd'] = mcad(grid.close, fast, slow, sign)

            # 寻找死叉点（MACD由正转负的时刻）
            mask = (grid['macd'] < 0) & (grid['macd'].shift(1) >= 0)
            if mask.sum() < 2:  # 需要至少2个死叉点对比
                print(f"{market_index} 死叉点不足2个，无法检测顶背离")
                return False

            # 取最近两个死叉点（前一个与当前）
            key2, key1 = mask[mask].index[-2], mask[mask].index[-1]

            # 顶背离核心条件
            price_cond = grid.close[key2] < grid.close[key1]  # 价格创新高（后高>前高）
            dif_cond = grid.dif[key2] > grid.dif[key1] > 0  # DIF未创新高（后低<前高）且为正
            macd_cond = grid.macd.iloc[-2] > 0 > grid.macd.iloc[-1]  # MACD由正转负

            # 趋势验证：DIF近期处于下降趋势（近10日均值<前10日均值）
            if len(grid['dif']) > 20:
                recent_avg = grid['dif'].iloc[-10:].mean()  # 近10日DIF均值
                prev_avg = grid['dif'].iloc[-20:-10].mean()  # 前10日DIF均值
                trend_cond = recent_avg < prev_avg
            else:
                trend_cond = False

            # print(f"{market_index} 顶背离检测: 价格创新高={price_cond}, DIF未新高={dif_cond}, "
            #       f"MACD转负={macd_cond}, DIF下降趋势={trend_cond}")
            return price_cond and dif_cond and macd_cond and trend_cond

        except Exception as e:
            print(f"{market_index} 顶背离检测错误: {e}")
            return False

    # 非小市值只计算判断, 不做仓位处理
    if market_index != '399101.XSHE':
        res = 1 if detect_divergence() else 0
        if res:
            print(f"{market_index} 触发顶背离了!!!!! 快跑 !!!!!")
        return res

    if detect_divergence():
        g.dbl.append(1)
        print(f"⚠️⚠️⚠️⚠️⚠️ 检测到{market_index}顶背离信号（价格新高但MACD走弱），清仓非涨停股票")

        # 仅保留当前涨停股（可能延续强势），清仓其他股票
        current_data = get_current_data()

        # 仅对小市值进行处理
        for stock in g.strategy_holdings[1][:]:
            # 当前未涨停的股票清仓
            if current_data[stock].last_price < current_data[stock].high_limit:
                print(f"{stock} 因大盘顶背离清仓（非涨停股）")
                close_position(stock)
    else:
        g.dbl.append(0)
        # print(f"未检测到大盘顶背离，市场趋势正常, 最近的顶背离记录: {g.dbl[-g.check_dbl_days:]}")


# 尾盘记录各个策略的收益
def make_record(context):
    positions = context.portfolio.positions
    if not positions:
        return
    current_data = get_current_data()
    g.strategy_value_data = {1: 0, 2: 0, 3: 0, 4: 0}
    # 复制一个昨天的记录进行累计
    copy_strategy_value = {
        1: g.strategy_value[1],
        2: g.strategy_value[2],
        3: g.strategy_value[3],
        4: g.strategy_value[4],
    }
    for stock, pos in positions.items():
        strategy_id = g.stock_strategy[stock]
        current_value = pos.total_amount * current_data[stock].last_price  # 当前价值
        cost_value = pos.total_amount * pos.avg_cost  # 成本价值
        pnl_value = current_value - cost_value  # 当前盈亏金额
        copy_strategy_value[strategy_id] += pnl_value  # 计算浮盈浮亏
        g.strategy_value_data[strategy_id] += current_value
    if g.portfolio_value_proportion[0]:
        record(小市值=round(copy_strategy_value[1] / g.strategy_starting_cash[1] * 100 - 100, 2))
    if g.strategy_ETF_2000_proportion:
        record(ETF反弹=round(copy_strategy_value[2] / g.strategy_starting_cash[2] * 100 - 100, 2))
    if g.portfolio_value_proportion[2]:
        record(ETF轮动=round(copy_strategy_value[3] / g.strategy_starting_cash[3] * 100 - 100, 2))
    if g.portfolio_value_proportion[3]:
        record(白马攻防=round(copy_strategy_value[4] / g.strategy_starting_cash[4] * 100 - 100, 2))

    # 收盘后再把ETF轮动的明日选股提前透漏下
    if g.portfolio_value_proportion[2]:
        print("收盘后检测下最新的ETF动量排名, 方便明天参考")
        filter_moment_rank(g.etf_pool_3, g.m_days, 0, g.m_score)


# 制表展示每日收益
def print_summary(context):
    """
    打印当前投资组合的总资产和持仓详情

    参数:
        context: 包含投资组合信息的对象。
        get_current_data: 获取当前市场数据的函数。
    """
    # 获取总资产
    total_value = round(context.portfolio.total_value, 2)

    # 获取当前持仓
    current_stocks = context.portfolio.positions
    if not current_stocks:
        # 如果没有持仓，只显示总资产
        print(f"🚤🚤🚤🚤🚤 当前总资产: {total_value} 休息ing ")
        return

    # 创建表格
    table = PrettyTable([
        " 所属策略 ",
        " 股票代码 ",
        " 股票名称 ",
        " 持仓数量 ",
        " 持仓价格 ",
        " 当前价格 ",
        " 盈亏数额 ",
        " 盈亏比例 ",
        " 股票市值 ",
        " 仓位占比 "])
    table.hrules = prettytable.ALL  # 显示所有水平线

    # # 设置对齐方式
    # table.align["所属策略    "] = "l"
    # table.align["股票代码    "] = "l"
    # table.align["股票名称  "] = "l"
    # for field in ["持仓数量  ", "持仓价格  ", "当前价格  ", "盈亏数额  ", "盈亏比例  ", "市值  ", "仓位占比  "]:
    #     table.align[field] = "r"

    # 遍历持仓股票
    total_market_value = 0  # 总市值（用于累加每只股票的市值）
    for stock in current_stocks:
        current_shares = current_stocks[stock].total_amount  # 持仓数量
        current_price = round(get_current_data()[stock].last_price, 3)  # 当前价格
        avg_cost = round(current_stocks[stock].avg_cost, 3)  # 持仓平均成本

        # 计算盈亏比例
        profit_ratio = (current_price - avg_cost) / avg_cost if avg_cost != 0 else 0
        profit_ratio_percent = f"{profit_ratio * 100:.2f}%"  # 转为百分比并保留两位小数
        profit_ratio_percent += f" {'↑' if profit_ratio > 0 else '↓'}"
        # 计算盈亏数额
        profit_amount = round((current_price - avg_cost) * current_shares, 2)

        # 计算市值
        market_value = round(current_shares * current_price, 2)
        total_market_value += market_value  # 累加总市值

        # 处理股票代码：移除后缀
        stock_code = stock.split(".")[0]  # 只保留股票代码部分

        # 添加到表格
        table.add_row([
            g.stock_strategy[stock],
            stock_code,
            format_stock_code(stock),
            current_shares,
            avg_cost,
            current_price,
            profit_amount,
            profit_ratio_percent,
            market_value,
            f"{market_value / context.portfolio.total_value * 100:.2f}%"
        ])

    # 账户总资产
    total_value = context.portfolio.total_value
    # 汇总
    if g.strategy_value_data[1]:
        table.add_row(["小市值", "", "", "", "", "", "", "", f"{g.strategy_value_data[1]:.2f}",
                       f"{g.strategy_value_data[1] / total_value * 100:.2f}%"])
    if g.strategy_value_data[2]:
        table.add_row(["ETF反弹", "", "", "", "", "", "", "", f"{g.strategy_value_data[2]:.2f}",
                       f"{g.strategy_value_data[2] / total_value * 100:.2f}%"])
    if g.strategy_value_data[3]:
        table.add_row(["ETF轮动", "", "", "", "", "", "", "", f"{g.strategy_value_data[3]:.2f}",
                       f"{g.strategy_value_data[3] / total_value * 100:.2f}%"])
    if g.strategy_value_data[4]:
        table.add_row(["白马攻防", "", "", "", "", "", "", "", f"{g.strategy_value_data[4]:.2f}",
                       f"{g.strategy_value_data[4] / total_value * 100:.2f}%"])
    table.add_row(["总市值", "", "", "", "", "", "", "", f"{total_market_value:.2f}", ""])
    table.add_row(["总资产", "", "", "", "", "", "", "", f"{total_value:.2f}", ""])

    # 打印表格
    print(f'当前总资产\n{table}')


# 小市值换手检测
def xsz_huanshou_check(context):
    huanshou(context, stock_list=g.strategy_holdings[1][:])


# ETF轮动日内止损检测
def etf_stop_loss_by_cur_day(context):
    holdings = set(g.strategy_holdings[3])
    # 检测日内亏损
    stop_loss_by_cur_day(holdings, ratio=g.stoploss_limit_by_cur_day)


""" ====================== 公共函数 ====================== """


# 封装实盘下单函数
def my_order_target_value(security, value):
    o = order_target_value(security, value)
    if o:
        if o.is_buy:
            if o.price * o.amount > 0:
                print(f"🚚🚚🚚🚚🚚 {format_stock_code(security)}  "
                      f"买价{o.price:<7.2f}  "
                      f"买量{o.amount:<7}   "
                      f"价值{o.price * o.amount:.2f}")
                return o
        else:
            if o.price * o.amount > 0:
                print(f"🚛🚛🚛🚛🚛 {format_stock_code(security)}  "
                      f"卖价{o.price:<7.2f}  "
                      f"成本{o.avg_cost:<7.2f}   "
                      f"卖量{o.amount:<7}   "
                      f"盈亏{(o.price - o.avg_cost) * o.amount:.2f}"
                      f"( {(o.price - o.avg_cost) / o.avg_cost * 100:.2f}% )")
                return o


# 开仓买入并记录策略持仓
def open_position(context, security, value, strategy_id):
    if value <= 5000:
        # print(f"买入金额{value}较小, 不进行买入")
        return
    if security in context.portfolio.positions:
        security_value = context.portfolio.positions[security].value
        if abs(value - security_value) < 5000:
            # print(f"买入差额{value - security_value}较小, 不进行买入")
            return
    order = my_order_target_value(security, value)
    if order:
        security not in g.strategy_holdings[strategy_id] and g.strategy_holdings[strategy_id].append(security)
        g.stock_strategy[security] = strategy_id
    return order


# 闭仓卖出并清空策略持仓
def close_position(security):
    order = my_order_target_value(security, 0)
    if order:
        strategy_id = g.stock_strategy[security]
        # 持仓列表移除
        security in g.strategy_holdings[strategy_id] and g.strategy_holdings[strategy_id].remove(security)
        # 计算卖出的盈亏
        pnl_value = (order.price - order.avg_cost) * order.amount
        # 每日策略总价值更新盈亏
        g.strategy_value[strategy_id] += pnl_value
    return order


# 日内止损
def stop_loss_by_cur_day(stock_list, ratio=-0.03):
    for stock in stock_list:
        cur_ratio = cal_cur_to_open_ratio(stock)
        if cur_ratio < ratio:
            print(f"{format_stock_code(stock)} 距离开盘跌幅 {cur_ratio * 100:.2f}% 清仓处理")
            close_position(stock)


""" ====================== 模块工具函数 ====================== """


# 展示优化
def format_stock_code(stock_code):
    try:
        stock_info = get_security_info(stock_code)
    except Exception:
        return f"{stock_code[:6]}"
    return f"{stock_code[:6]}({stock_info.display_name}) "


# 筛选审计意见
def filter_audit(context, code_list):
    # 获取审计意见，近三年内如果有不合格(report_type为3、4、5、7)的审计意见则返回False，否则返回True
    final_list = []
    '''
    审计意见类型编码
        类型编码 审计意见类型
        1 	     无保留
        2 	     无保留带解释性说明
        3        保留意见
        4        拒绝/无法表示意见
        5        否定意见
        6 	     未经审计
        7 	     保留带解释性说明
        10 	     经审计（不确定具体意见类型）
        11       无保留带持续经营重大不确定性
    '''
    for stock in code_list:
        previous_date = context.previous_date
        last_year = (previous_date.replace(year=previous_date.year - 3, month=1, day=1)).strftime('%Y-%m-%d')
        q = query(finance.STK_AUDIT_OPINION.code, finance.STK_AUDIT_OPINION.pub_date, finance.STK_AUDIT_OPINION) \
            .filter(finance.STK_AUDIT_OPINION.code == stock, finance.STK_AUDIT_OPINION.pub_date >= last_year)
        df = finance.run_query(q)
        values_to_check = [3, 4, 5, 7]
        if not df['opinion_type_id'].isin(values_to_check).any():
            final_list.append(stock)
    return final_list


# 获取红利列表
def bonus_filter(context, stock_list):
    year = context.previous_date.year
    start_date = datetime.datetime(year=year, month=1, day=1)
    end_date = context.previous_date
    if end_date.month in [5]:
        q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.company_name, finance.STK_XR_XD.board_plan_pub_date,
                  finance.STK_XR_XD.bonus_amount_rmb, finance.STK_XR_XD.bonus_ratio_rmb
                  ).filter(
            # finance.STK_XR_XD.bonus_type !='年度分红',
            finance.STK_XR_XD.board_plan_pub_date > start_date,
            finance.STK_XR_XD.implementation_pub_date <= end_date,
            # finance.STK_XR_XD.a_xr_date < context.previous_date,
            finance.STK_XR_XD.bonus_ratio_rmb > 0,
            finance.STK_XR_XD.code.in_(stock_list))
        expected_bonus_df = finance.run_query(q)

        if len(expected_bonus_df) > 0:
            bonus_list = expected_bonus_df['code'].unique().tolist()
            price_df = history(1, unit='1d', field='close', security_list=bonus_list, df=True, skip_paused=False,
                               fq='pre')
            price_df = price_df.T
            price_df.rename(columns={price_df.columns[0]: 'Close_now'}, inplace=True)
            price_df['code'] = price_df.index
            expected_bonus_df = pd.merge(expected_bonus_df, price_df, on=('code',), how='left')
            expected_bonus_df['bonus_ratio'] = (expected_bonus_df['bonus_ratio_rmb']) / expected_bonus_df['Close_now']
            expected_bonus_df = expected_bonus_df.sort_values(by='bonus_ratio', ascending=True)
            bonus_list = expected_bonus_df['code'].unique().tolist()
        else:
            bonus_list = []
    else:
        reprot_date = datetime.datetime(year=year - 1, month=12, day=31)
        q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.company_name, finance.STK_XR_XD.a_registration_date,
                  finance.STK_XR_XD.bonus_amount_rmb, finance.STK_XR_XD.bonus_ratio_rmb
                  ).filter(
            finance.STK_XR_XD.report_date == reprot_date,
            finance.STK_XR_XD.bonus_type == '年度分红',
            finance.STK_XR_XD.implementation_pub_date <= end_date,
            finance.STK_XR_XD.board_plan_bonusnote == '不分配不转增',
            finance.STK_XR_XD.code.in_(stock_list))

        no_year_bonus = finance.run_query(q)
        no_year_bonus_list = no_year_bonus['code'].unique().tolist()
        # 排除今年不分红的股票
        bonus_list = [code for code in stock_list if code not in no_year_bonus_list]
        bonus_list = short_by_market_cap(context, bonus_list)

    if len(bonus_list) < g.xsz_stock_num:
        bonus_list.extend([x for x in short_by_market_cap(context, stock_list) if x not in bonus_list][
                          :g.xsz_stock_num - len(bonus_list)])
    return bonus_list


# 计算RSI指标
def calculate_rsi(code, period=14):
    """计算RSI指标"""
    df = attribute_history(code, 125, '1d', ['close', ], skip_paused=True, df=True, fq='pre')
    prices = df['close'].values
    deltas = np.diff(prices)
    seed = deltas[:period + 1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        return 100
    rs = up / down
    rsi = 100. - 100. / (1. + rs)
    return rsi


#  基础过滤各种股票
def filter_stocks(context, stock_list):
    current_data = get_current_data()
    # 涨跌停和最近价格的判断
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    # 过滤标准
    filtered_stocks = []
    for stock in stock_list:
        if current_data[stock].paused:  # 停牌
            continue
        if current_data[stock].is_st:  # ST
            continue
        if '退' in current_data[stock].name:  # 退市
            continue
        if stock.startswith('30') or stock.startswith('68') or stock.startswith('8') or stock.startswith('4'):  # 市场类型
            continue
        if not (stock in context.portfolio.positions or last_prices[stock][-1] < current_data[stock].high_limit):  # 涨停
            continue
        if not (stock in context.portfolio.positions or last_prices[stock][-1] > current_data[stock].low_limit):  # 跌停
            continue
        # 次新股过滤
        start_date = get_security_info(stock).start_date
        if context.previous_date - start_date < timedelta(days=375):
            continue
        filtered_stocks.append(stock)
    return filtered_stocks


# 计算最新价格对比开盘价格的比值
def cal_cur_to_open_ratio(security):
    current_data = get_current_data()
    last_price = current_data[security].last_price
    day_open = current_data[security].day_open
    return (last_price - day_open) / day_open


# 计算MACD指标
def mcad(close, short=12, long=26, m=9):
    """计算 MACD 指标
    用于判断趋势强度和潜在反转点，由 DIF、DEA、MACD 柱组成

    参数:
        close: 收盘价序列
        short: 短期EMA周期（默认12）
        long: 长期EMA周期（默认26）
        m: 信号周期（默认9）

    返回:
        DIF: 短期EMA与长期EMA的差值
        DEA: DIF的M期EMA
        MACD: (DIF-DEA)*2（放大波动）
    """

    # 计算指数移动平均线
    def ema(series, n):
        """计算指数移动平均线（Exponential Moving Average）
        用于平滑价格波动，反映近期价格趋势，权重随时间递减

        参数:
            series: 价格序列（如收盘价）
            N: 计算周期

        返回:
            EMA序列
        """
        return pd.Series.ewm(series, span=n, min_periods=n - 1, adjust=False).mean()

    dif = ema(close, short) - ema(close, long)
    dea = ema(dif, m)
    return dif, dea, (dif - dea) * 2


# 换手检测
def huanshou(context, stock_list):
    # 换手率计算
    def huanshoulv(_stock, is_avg=False):
        if is_avg:
            # 计算平均换手率
            end_date = context.previous_date
            df_volume = get_price(_stock, end_date=end_date, frequency='daily', fields=['volume'], count=20)
            df_cap = get_valuation(_stock, end_date=end_date, fields=['circulating_cap'], count=1)
            circulating_cap = df_cap['circulating_cap'].iloc[0] if not df_cap.empty else 0
            if circulating_cap == 0:
                return 0.0
            df_volume['turnover_ratio'] = df_volume['volume'] / (circulating_cap * 10000)
            return df_volume['turnover_ratio'].mean()
        else:
            # 计算实时换手率
            date_now = context.current_dt
            df_vol = get_price(_stock, start_date=date_now.date(), end_date=date_now, frequency='1m', fields=['volume'],
                               skip_paused=False, fq='pre', panel=True, fill_paused=False)
            volume = df_vol['volume'].sum()
            date_pre = context.previous_date
            df_circulating_cap = get_valuation(_stock, end_date=date_pre, fields=['circulating_cap'], count=1)
            circulating_cap = df_circulating_cap['circulating_cap'].iloc[0] if not df_circulating_cap.empty else 0
            if circulating_cap == 0:
                return 0.0
            turnover_ratio = volume / (circulating_cap * 10000)
            return turnover_ratio

    current_data = get_current_data()
    shrink, expand = 0.003, 0.1
    # for stock in context.portfolio.positions:
    for stock in stock_list:
        if current_data[stock].paused == True:
            continue
        if current_data[stock].last_price >= current_data[stock].high_limit * 0.97:
            continue
        if context.portfolio.positions[stock].closeable_amount == 0:
            continue
        rt = huanshoulv(stock, False)
        avg = huanshoulv(stock, True)
        if avg == 0:
            continue
        r = rt / avg
        action, icon = '', ''
        if avg < 0.003:
            action, icon = '缩量', '❄️'
        elif rt > expand and r > 2:
            action, icon = '放量', '🔥'
        if action:
            print(f"{action} {format_stock_code(stock)}  换手率:{rt:.2%}  均:{avg:.2%} 倍率:x{r:.1f} {icon}")
            close_position(stock)


# 成交量宽度防御检测
def check_defense_trigger(context):
    """改进后的防御条件检查"""

    # 计算宽度
    def get_market_breadth(ma_days):
        required_days = ma_days + 10
        end_date = context.current_dt.replace(hour=14, minute=49)

        # 获取行业分类数据
        sw_l1 = get_industries('sw_l1', date=context.current_dt.date())
        industry_stocks = {}
        for idx, row in sw_l1.iterrows():
            ind_stocks = get_industry_stocks(idx, date=end_date)
            industry_stocks[row['name']] = ind_stocks  # 存储行业对应的股票列表

        # 获取所有股票
        all_stocks = []
        for stocks in industry_stocks.values():
            all_stocks.extend(stocks)
        all_stocks = list(set(all_stocks))  # 去重

        # 获取价格和成交额数据
        data = get_bars(all_stocks, end_dt=end_date, count=required_days, unit='1d',
                        fields=['date', 'close', 'volume', 'money'], include_now=True, df=True)

        # 处理价格数据：用level_1作为索引（行号），level_0作为股票代码列
        price_reset = data.reset_index()
        price_data = price_reset.pivot(index='level_1', columns='level_0', values='close')  # 按要求的透视表写法

        # 计算移动平均和站上均线的股票占比
        ma = price_data.rolling(window=ma_days).mean()
        above_ma = price_data > ma

        # 核心逻辑：按透视表处理20日成交金额，计算平均值后再分组
        # 1. 重置索引并创建成交额透视表（行=行号，列=股票代码，值=成交额）
        money_reset = data.reset_index()
        money_pivot = money_reset.pivot(index='level_1', columns='level_0', values='money')  # 成交额透视表

        recent_20d_money_pivot = money_pivot.tail(20)  # 关键：直接从透视表取最近20天

        avg_money = recent_20d_money_pivot.mean().reset_index()  # 按列求平均
        avg_money.columns = ['code', 'avg_money']  # 重命名列：股票代码、平均成交额

        # 4. 按平均成交额排序并分为20组
        avg_money = avg_money.sort_values('avg_money', ascending=False)
        # 使用qcut进行分组，处理可能的重复值
        avg_money['money_group'] = pd.qcut(avg_money['avg_money'], 20, labels=[f'组{i + 1}' for i in range(20)],
                                           duplicates='drop')

        # 5. 创建成交额分组字典（组名: 股票列表）
        money_groups = {group: group_df['code'].tolist()
                        for group, group_df in avg_money.groupby('money_group')}

        # 6. 计算每个成交额组站上均线的股票比例
        group_scores = pd.DataFrame(index=price_data.index)
        for group, stocks in money_groups.items():
            valid_stocks = list(set(above_ma.columns) & set(stocks))
            if valid_stocks:
                group_scores[group] = 100 * above_ma[valid_stocks].sum(axis=1) / len(valid_stocks)

        # 7. 计算近3天各组平均站上均线比例
        recent_group_data = group_scores[-3:].mean()
        _sorted_ma_data = recent_group_data.sort_values(ascending=False)

        # 8. 处理涨跌幅数据和每日指标
        df = data.reset_index().rename(columns={'level_0': 'symbol', 'level_1': 'index'})
        df['pct_change'] = df.groupby(['symbol'])['close'].pct_change()

        trade_days = get_trade_days(end_date=context.current_dt, count=3)
        by_date = trade_days[0]
        df = df[df.date >= by_date]

        grouped = df.groupby('date')
        _result = pd.DataFrame({
            'up_ratio': grouped['pct_change'].apply(lambda x: (x > 0).mean()),
            'down_over': grouped['pct_change'].apply(lambda x: (x <= -0.0985).sum())
        }).reset_index()
        return _sorted_ma_data, _result

    # 计算趋势指标
    def calculate_trend_indicators(index_symbol='399101.XSHE'):
        """计算趋势指标: 过去3天内只要有一天处于高位，则视为高位，避免边界问题）"""
        # 参数设置
        high_lookback = 60  # 近期高点观察窗口
        high_proximity = 0.95  # 接近高点的阈值（95%）
        check_days = 2  # 检查过去1天的状态

        end_date = context.current_dt.replace(hour=14, minute=49)

        # 获取历史数据（需要包含足够天数，用于计算过去5天的指标）
        # 为了计算过去5天的指标，需要多获取high_lookback天数据（避免边界问题）
        total_days_needed = high_lookback + 10
        data = get_bars(index_symbol, end_dt=end_date,
                        count=total_days_needed,
                        unit='1d', fields=['date', 'close', 'high', 'avg', 'volume'], include_now=True, df=True)

        data['date'] = pd.to_datetime(data['date'])

        # 计算过去每天的is_high状态
        _past_is_high_list = []

        # 遍历过去2天
        for i in range(-check_days, 0):
            # 数据切片，每次60天，不包含最后一天
            valid_data = data.iloc[:i][-high_lookback:]
            current_day_price = valid_data['close'].iloc[-1]

            # 计算当天的接近高点状态
            day_max_high = valid_data['high'].max()
            day_close_to_high = current_day_price >= (day_max_high * high_proximity)

            # 当天的is_high
            day_is_high = day_close_to_high
            _past_is_high_list.append(day_is_high)

        # 当前天的指标（最后一天）
        current_data = data[-high_lookback:]
        current_price = current_data['close'].iloc[-1]
        max_high = current_data['high'].max()
        close_to_high = current_price >= (max_high * high_proximity)

        # 将当前天加入列表，
        _past_is_high_list.append(close_to_high)

        # 新的is_high只要有一天为True，则为True
        _is_high = any(_past_is_high_list)

        return _is_high, _past_is_high_list

    # 为方便回测直接用记录的历史路标对比
    cur_date_str = str(context.current_dt.date())
    if cur_date_str <= g.history_defense_date_list[-1]:
        if cur_date_str in g.history_defense_date_list:
            g.defense_signal = True
            print("组20防御: True, 处于历史触发范围内")
        else:
            g.defense_signal = False
            print("触发防御: False, 未处于历史触发范围内")
    # 超过时间则手动计算, 用于实盘
    else:
        if g.defense_signal:
            # 如果已经进入防御板块，只要看组20有没有在前三
            sorted_ma_data, result = get_market_breadth(20)
            up_ratio = result.iloc[-3:]['up_ratio'].mean()  # 涨跌比
            avg_score = sorted_ma_data['组1']  # 宽度
            # 退出版本1：
            defense_in_top = any([ind in sorted_ma_data.index[:3] for ind in g.industries])  # 逻辑防御板块在前3
            bank_exit_signal = not defense_in_top
            # 退出版本2：宽度和涨跌比修复
            # bank_exit_signal= up_ratio>=0.5 and avg_score >=55
            g.defense_signal = not bank_exit_signal
            print(f"组20防御: {g.defense_signal} "
                  f"组1宽度:{avg_score:.1f} "
                  f"涨跌比:{up_ratio:.2f} "
                  f"组20防御次数:{sum(g.cnt_bank_signal)} "
                  f"top宽度:{sorted_ma_data.index[:5].tolist()}")
        else:
            # 判断条件
            is_high, past_is_high_list = calculate_trend_indicators()
            if is_high:  # 高位或者缩量
                # 行业强度判断
                sorted_ma_data, result = get_market_breadth(20)
                defense_in_top = any([ind in sorted_ma_data.index[:2] for ind in g.industries])  # 防御板块在前二
                # 版本2改为判断剔除防御板块后的平均宽度
                avg_score = sorted_ma_data[[ind not in g.industries for ind in sorted_ma_data.index]].mean()
                above_average = avg_score < 60  # 平均宽度低于60
                # 版本三，涨跌比均值低于50%
                up_ratio = result.iloc[-3:]['up_ratio'].mean()
                above_ratio = up_ratio < 0.5
                # 组20择时综合判断
                is_bank_defense = defense_in_top and above_average and above_ratio
                g.defense_signal = is_bank_defense
                if is_bank_defense:
                    g.cnt_bank_signal.append(is_bank_defense)
                print(f"组20防御: {is_bank_defense} "
                      f"高位:{is_high}{past_is_high_list} "
                      f"组1宽度:{avg_score:.1f} "
                      f"涨跌比:{up_ratio:.2f} "
                      f"top宽度:{sorted_ma_data.index[:5].tolist()} ")
            else:
                g.defense_signal = False
                print(f"触发防御: {g.defense_signal} 高位:{is_high}{past_is_high_list}")

    # 检测到需要防御进行空仓, 只空仓小市值的票
    now_time = context.current_dt
    if g.defense_signal:
        for stock in g.strategy_holdings[1][:]:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                                     skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            # 已涨停不清仓
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                close_position(stock)


# 资金再平衡, 2000ETF反弹策略的周期无法早于2023.10, 基于此时间进行资金平衡
def capital_balance_2(context):
    """
    2023.10 之前 ETF反弹 的仓位纳入到 ETF轮动 中
    """
    cur_date = str(context.current_dt.date())
    # 基于首次进行检测
    if cur_date < "2023-09-28" and g.strategy_ETF_2000_proportion_reset is None:
        g.portfolio_value_proportion[2] += g.strategy_ETF_2000_proportion
        g.portfolio_value_proportion[1] = 0
        g.strategy_ETF_2000_proportion_reset = False
    # 到达既定时间后进行拨正原始比例
    elif cur_date >= "2023-09-28" and g.strategy_ETF_2000_proportion_reset is False:
        # 计算ETF轮动所需要分配资金
        strategy_total_value = context.portfolio.total_value * g.strategy_ETF_2000_proportion
        # 检测ETF轮动是否有持仓, 如果有的话就要吐出来还给ETF反弹
        if g.strategy_holdings[2]:
            cur_etf = g.strategy_holdings[2]
            if context.portfolio.positions[cur_etf].closeable_amount > 0:
                o = order_value(context, cur_etf, -strategy_total_value)  # 卖出需要预留给ETF轮动的资金
                if o:
                    stock_show = f"{format_stock_code(cur_etf)}: ".ljust(20)
                    print(f"🚛🚛🚛🚛🚛 ETF反弹预留资金转移 {stock_show}  "
                          f"卖价{o.price:<7.2f}  "
                          f"成本{o.avg_cost:<7.2f}   "
                          f"卖量{o.amount:<7}   "
                          f"盈亏{(o.price - o.avg_cost) * o.amount:.2f}"
                          f"( {(o.price - o.avg_cost) / o.avg_cost * 100:.2f}% )")
        g.portfolio_value_proportion[2] -= g.strategy_ETF_2000_proportion
        g.portfolio_value_proportion[1] = g.strategy_ETF_2000_proportion
        g.strategy_ETF_2000_proportion_reset = True  # 拨正原始比例


#  根据市值排序
def short_by_market_cap(context, stock_list):
    short_q = query(
        valuation.code,
        valuation.market_cap,  # 总市值 circulating_market_cap/market_cap
    ).filter(
        valuation.code.in_(stock_list),
        valuation.day == context.previous_date,
    ).order_by(valuation.market_cap.asc())
    short_df = get_fundamentals(short_q)
    short_list = short_df['code'].unique().tolist()
    return short_list


""" ====================== 执行入口, 定时任务下发 ====================== """


def after_code_changed(context):
    unschedule_all()

    # 策略1 小市值策略
    if g.portfolio_value_proportion[0] > 0:
        run_daily(prepare_xsz, '9:05')
        # 初次检测成交额
        if g.check_defense and g.defense_signal is None:
            check_defense_trigger(context)
        # 每日开盘前检测大盘顶背离, 只针对策略1
        if g.DBL_control:
            run_daily(check_dbl, '9:31')  # 不要早于9点30, 否则会导致绘制的收益曲线无法拿到价格信息
        run_weekly(strategy_1_sell, 2, '09:40')
        run_weekly(strategy_1_buy, 2, '09:40:02')
        run_daily(xsz_sell_stocks, time='10:00')  # 止损函数
        # 换手检查
        if g.huanshou_check:
            run_daily(xsz_huanshou_check, '10:30')
        # 涨停板检查
        run_daily(xsz_check_limit_up, '14:00')
        # 成交额宽度检测
        if g.check_defense:
            run_daily(check_defense_trigger, '14:50')
        # 检测清仓并买入ETF
        run_daily(close_account, '14:50')

    # 策略2 ETF反弹策略
    if g.strategy_ETF_2000_proportion > 0:
        run_daily(capital_balance_2, '14:45')  # 基于 2023.9.28 进行资金再平衡
        run_daily(strategy_2_sell, '14:49')
        run_daily(strategy_2_buy, '14:50')

    # 策略3 ETF轮动策略
    if g.portfolio_value_proportion[2] > 0:
        run_daily(strategy_3_sell, '10:35:00')
        run_daily(strategy_3_buy, '10:35:05')
        if g.enable_stop_loss_by_cur_day:
            run_daily(etf_stop_loss_by_cur_day, '10:01')  # 日内亏损检测
            run_daily(etf_stop_loss_by_cur_day, '10:31')  # 日内亏损检测

    # 策略4 白马策略
    if g.portfolio_value_proportion[3] > 0:
        run_monthly(bm_before_market_open, 1, time='8:00')
        # run_daily(bm_before_market_open, time='15:10')
        run_monthly(bm_adjust_position, 1, time='10:40')

    # 记录各策略每日收益
    run_daily(make_record, '15:01')
    # 打印每日收益
    run_daily(print_summary, '15:02')
