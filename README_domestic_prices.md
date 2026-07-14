# 国内铜铝采购价格每日预测系统

这个系统独立于旧的国际铜铝价格回归脚本，目标是预测国内现货采购基准价。默认数据源为用户提供的长江有色网爬取结果：

- 文件：`D:/BSH实习/铜、铝采购影响分析/原材料日度价格.xlsx`
- 工作表：`ccmn_changjiang_avg_prices`
- 预测品种：`1#铜`、`A00铝`、`1#白银`、`铝合金ADC12`、`铸造铝合金锭(ZLD104)`
- SMM 和在线长江/SHFE 抓取：作为备用源，不作为默认源
- 单位：人民币/吨
- 频率：每日
- 预测窗口：未来 30 个自然日

## 默认更新

```powershell
python update_domestic_prices.py
```

默认会读取 Excel 中五个品种的日度价格，跳过空值，然后为每个品种生成未来 30 天预测。

## 备用数据源

在线长江有色/上期所源：

```powershell
python update_domestic_prices.py --source changjiang_shfe
```

该源目前只覆盖铜、铝两个品种。

```powershell
python update_domestic_prices.py --source akshare_spot
```

该备用源使用 AkShare 的生意社现货价格接口，不是长江口径，只建议在长江页面不可用时临时使用。

如果使用 SMM 导出的 CSV 或内部表跑通流程：

```powershell
python update_domestic_prices.py --source csv --spot-csv path\to\smm_spot.csv
```

CSV 至少需要日期、金属、均价三列。英文列名可用 `date,metal,price`，中文列名可用 `日期,品种,均价`。

如果有正式 SMM API，可以在 `config.yaml` 填接口路径，并设置环境变量后运行：

```powershell
$env:SMM_API_BASE_URL="https://your-authorized-smm-api.example.com"
$env:SMM_API_TOKEN="your-token"
python update_domestic_prices.py --source api
```

## 打开看板

```powershell
streamlit run app.py
```

数据库文件默认为 `domestic_procurement_prices.sqlite`。

## 看板功能

- 每个品种页展示历史价格趋势、未来 30 天预测、最新价格和近 5 日变化。
- “区间统计”可以手动选择或随机生成一段日期，计算该区间的均价、最高价、最低价和有效天数。
- “回测对比”可以选择训练区间，例如只用 2021-2022 年数据训练，再预测 2023 年以后价格，并展示预测值与真实值的 MAE、MAPE、RMSE 和逐日误差表。
