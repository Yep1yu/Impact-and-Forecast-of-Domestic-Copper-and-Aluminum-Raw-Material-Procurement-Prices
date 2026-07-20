# Streamlit Community Cloud 部署

本项目采用以下更新链路：

1. GitHub Actions 在工作日北京时间 10:30、11:30、14:30 拉取长江有色价格。
2. 更新任务在临时 SQLite 副本中重建日度和月度预测。
3. 数据库通过完整性、表结构、数据量和最近运行状态检查后，替换仓库中的正式数据库。
4. GitHub Actions 提交更新文件；Streamlit Community Cloud 检测到提交后自动同步网页。

Streamlit 只以只读方式打开 SQLite。自动抓取和模型训练不在 Streamlit 进程中运行，因此应用休眠不会中断定时更新。

## 1. 配置 GitHub Secret

自动抓取需要有效的 CCMN 登录 Cookie。打开 GitHub 仓库：

`Settings` → `Secrets and variables` → `Actions` → `New repository secret`

- Name：`CCMN_COOKIE`
- Secret：登录 `ccmn.cn` 后浏览器请求中的完整 Cookie 值

Cookie 不要写入代码、配置文件或 Streamlit Secrets。它只由 GitHub Actions 使用。CCMN Cookie 过期后，工作流会失败，需要在同一位置替换为新的值。

## 2. 手动验证更新

进入 GitHub 仓库的 `Actions` 页面，选择 `Daily price update`，点击 `Run workflow`。成功后应出现一条提交信息为 `Update daily procurement price data` 的提交。

同时在 `Settings` → `Actions` → `General` 中确认 Workflow permissions 允许 `Read and write permissions`。如果 `main` 是受保护分支，需要允许 GitHub Actions 写入，或将工作流改为通过 Pull Request 发布数据。

工作流会在发布前运行：

```powershell
python scripts/verify_dashboard_database.py domestic_procurement_prices.sqlite
```

只有验证成功的数据库才会提交。

## 3. 部署 Streamlit

在 Streamlit Community Cloud 创建应用，并填写：

- Repository：`Yep1yu/Impact-and-Forecast-of-Domestic-Copper-and-Aluminum-Raw-Material-Procurement-Prices`
- Branch：`main`
- Main file path：`app.py`
- Python：`3.11`

本方案不需要在 Streamlit 的 Secrets 页面配置 `CCMN_COOKIE`。依赖由仓库根目录的 `requirements.txt` 安装。

## 4. 故障排查

- `CCMN_COOKIE is missing`：GitHub Secret 尚未配置。
- CCMN 返回“请登录账号”：Cookie 已过期，重新登录网站并更新 GitHub Secret。
- 数据库验证失败：正式数据库不会被替换；在 Actions 日志中检查缺少的数据表或模型输出。
- 工作流成功但网页未更新：确认 Streamlit 应用绑定的是同一仓库的 `main` 分支，并在应用设置中重启一次。

请勿提交 `.streamlit/secrets.toml`、Cookie、Token 或其他凭证。
