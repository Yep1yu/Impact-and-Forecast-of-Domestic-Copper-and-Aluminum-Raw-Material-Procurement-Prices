from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "micro_industry_regression_results_v3_extended.xlsx"
OUTPUT = ROOT / "latest_copper_aluminum_model_results.png"

W, H = 741, 489
BLUE = "#063fa6"
GREEN = "#008039"
RED = "#e53935"
NEG_GREEN = "#118044"
TEXT = "#17233c"
MUTED = "#667085"
LIGHT_BORDER = "#d7e8fb"
BG = "#f6fbff"

FONT_REG = "C:/Windows/Fonts/simhei.ttf"
FONT_BOLD = "C:/Windows/Fonts/simhei.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


F = {
    "title": font(16, True),
    "h": font(8, True),
    "body": font(8),
    "body_b": font(8, True),
    "coef": font(8),
    "rank": font(7, True),
    "card_title": font(11, True),
    "big": font(13, True),
    "small": font(7),
    "tiny": font(6),
    "insight": font(8, True),
}


LABELS = {
    "LME铜库存_变化率": "LME铜库存",
    "汽车产量当期值_变化率": "汽车产量",
    "电线电缆光缆及电工器材制造PPI_变化": "电线电缆PPI",
    "新能源汽车产量当期值_变化率": "新能源汽车产量",
    "发电机组产量当期值_变化率": "发电机组产量",
    "CFTC铜非商业净多头变化": "CFTC铜净多头",
    "动力煤价格月环比": "动力煤价格",
    "CFTC铜持仓量月环比": "CFTC铜持仓量",
    "房间空气调节器产量当期值_变化率": "空调产量",
    "SHFE铜主连持仓量_变化率": "SHFE铜持仓量",
    "中国原铝产量月环比": "中国原铝产量",
    "全球原铝产量月环比": "全球原铝产量",
    "天然气价格月环比": "天然气价格",
    "汽车销量Top50厂商合计_变化率": "汽车销量Top50",
    "中国氧化铝产量月环比": "中国氧化铝产量",
    "发电量当期值_变化率": "发电量",
    "SHFE铝主连持仓量_变化率": "SHFE铝持仓量",
    "Brent原油价格月环比": "Brent原油",
    "SHFE铝仓单库存_变化率": "SHFE铝仓单",
}


SUB = {
    "LME铜库存_变化率": "LME Cu Stock",
    "汽车产量当期值_变化率": "Auto Output",
    "电线电缆光缆及电工器材制造PPI_变化": "Cable PPI",
    "新能源汽车产量当期值_变化率": "NEV Output",
    "发电机组产量当期值_变化率": "Generator",
    "CFTC铜非商业净多头变化": "CFTC Net Long",
    "动力煤价格月环比": "Coal",
    "CFTC铜持仓量月环比": "CFTC OI",
    "房间空气调节器产量当期值_变化率": "Aircon",
    "SHFE铜主连持仓量_变化率": "SHFE OI",
    "中国原铝产量月环比": "China Primary Al",
    "全球原铝产量月环比": "Global Primary Al",
    "天然气价格月环比": "Gas",
    "汽车销量Top50厂商合计_变化率": "Auto Sales",
    "中国氧化铝产量月环比": "China Alumina",
    "发电量当期值_变化率": "Power",
    "SHFE铝主连持仓量_变化率": "SHFE OI",
    "Brent原油价格月环比": "Oil",
    "SHFE铝仓单库存_变化率": "SHFE Warrant",
}


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def center_text(draw, xy, text, fnt, fill):
    x0, y0, x1, y1 = xy
    tw, th = text_size(draw, text, fnt)
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, font=fnt, fill=fill)


def ellipsed(draw, text, fnt, max_w):
    if text_size(draw, text, fnt)[0] <= max_w:
        return text
    out = text
    while out and text_size(draw, out + "...", fnt)[0] > max_w:
        out = out[:-1]
    return out + "..."


def round_rect(draw, box, r, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def draw_metal_icon(draw, x, y, theme):
    if theme == "copper":
        draw.ellipse((x, y + 3, x + 27, y + 18), fill="#a85023", outline="#7a3418")
        draw.rectangle((x + 8, y, x + 30, y + 14), fill="#d97839", outline="#99431d")
        draw.ellipse((x + 6, y, x + 21, y + 14), fill="#f1a06a", outline="#99431d")
        draw.ellipse((x + 10, y + 3, x + 17, y + 11), fill="#55210f")
    else:
        draw.polygon([(x, y + 13), (x + 22, y + 5), (x + 34, y + 11), (x + 12, y + 20)], fill="#aeb9b6", outline="#63706c")
        draw.polygon([(x + 3, y + 8), (x + 24, y), (x + 34, y + 6), (x + 13, y + 14)], fill="#dfe8e5", outline="#63706c")
        draw.line((x + 15, y + 4, x + 25, y + 9), fill="#7b8784", width=1)


def draw_panel(draw, box, title, theme, df, max_abs):
    x, y, w, h = box
    color = BLUE if theme == "copper" else GREEN
    round_rect(draw, (x, y, x + w, y + h), 6, "white", LIGHT_BORDER)
    draw.rounded_rectangle((x, y, x + w, y + 30), radius=6, fill=color)
    draw.rectangle((x, y + 24, x + w, y + 30), fill=color)
    draw_metal_icon(draw, x + 15, y + 5, theme)
    center_text(draw, (x, y + 2, x + w, y + 30), title, F["title"], "white")

    header_y = y + 38
    cols = {
        "rank": x + 15,
        "var": x + 76,
        "coef": x + 168,
        "sig": x + 282,
        "impact": x + w - 38,
    }
    for label, cx in [("排名", cols["rank"]), ("变量", cols["var"]), ("系数", cols["coef"]), ("显著性", cols["sig"]), ("对价格影响", cols["impact"])]:
        center_text(draw, (cx - 25, header_y - 4, cx + 25, header_y + 12), label, F["h"], color)

    row_top = y + 58
    row_h = 25
    axis_left = x + 204
    axis_right = x + 270
    axis_zero = (axis_left + axis_right) / 2
    half = (axis_right - axis_left) / 2
    for i, row in df.head(10).iterrows():
        yy = row_top + i * row_h
        if i % 2 == 1:
            draw.rectangle((x + 8, yy - 3, x + w - 8, yy + row_h - 4), fill="#fbfdff")
        rank_fill = color if i < 3 else "#6f7785"
        draw.ellipse((x + 10, yy + 2, x + 23, yy + 15), fill=rank_fill)
        center_text(draw, (x + 10, yy + 1, x + 23, yy + 15), str(i + 1), F["rank"], "white")
        name = LABELS.get(row["指标"], row["指标"])
        sub = SUB.get(row["指标"], "")
        draw.text((x + 33, yy - 1), ellipsed(draw, name, F["body_b"], 118), font=F["body_b"], fill=TEXT)
        if sub:
            draw.text((x + 33, yy + 9), ellipsed(draw, f"({sub})", F["tiny"], 110), font=F["tiny"], fill=MUTED)

        coef = float(row["标准化系数"])
        draw.text((x + 163, yy + 3), f"{coef:.3f}", font=F["coef"], fill=TEXT)
        bar_end = axis_zero + coef / max_abs * half
        bar_color = color if coef >= 0 else "#0b4cab" if theme == "copper" else "#008b46"
        x0, x1 = sorted([axis_zero, bar_end])
        draw.rounded_rectangle((x0, yy + 5, x1, yy + 11), radius=2, fill=bar_color)

        sig = str(row["显著性"]).strip()
        sig_fill = color if sig == "不显著" else BLUE
        center_text(draw, (x + 265, yy + 0, x + 300, yy + 15), sig, F["body_b"], sig_fill)
        direction = str(row["方向"])
        dir_fill = RED if direction == "正向" else NEG_GREEN
        center_text(draw, (x + w - 62, yy + 0, x + w - 12, yy + 15), direction, F["body_b"], dir_fill)

    axis_y = y + h - 25
    draw.line((axis_left, axis_y, axis_right, axis_y), fill="#d1d9e6", width=1)
    draw.line((axis_zero, axis_y - 137, axis_zero, axis_y + 4), fill="#cbd5e1", width=1)
    for t in [-max_abs, -max_abs / 2, 0, max_abs / 2, max_abs]:
        tx = axis_zero + t / max_abs * half
        draw.line((tx, axis_y - 3, tx, axis_y + 3), fill="#93a4b7", width=1)
        center_text(draw, (tx - 18, axis_y + 4, tx + 18, axis_y + 15), f"{t:.1f}", F["tiny"], "#5a6472")
    draw.line((axis_left, axis_y + 18, axis_zero - 6, axis_y + 18), fill=NEG_GREEN, width=2)
    draw.polygon([(axis_left, axis_y + 18), (axis_left + 4, axis_y + 15), (axis_left + 4, axis_y + 21)], fill=NEG_GREEN)
    draw.line((axis_zero + 6, axis_y + 18, axis_right, axis_y + 18), fill=RED, width=2)
    draw.polygon([(axis_right, axis_y + 18), (axis_right - 4, axis_y + 15), (axis_right - 4, axis_y + 21)], fill=RED)
    center_text(draw, (axis_left - 4, axis_y + 19, axis_zero - 4, axis_y + 30), "负向", F["tiny"], NEG_GREEN)
    center_text(draw, (axis_zero + 4, axis_y + 19, axis_right + 4, axis_y + 30), "正向", F["tiny"], RED)


def sig_label(sig, direction):
    if sig == "不显著":
        return f"不显著{direction}"
    if sig == "*":
        return f"弱显著{direction}"
    return f"显著{direction}"


def draw_top3_card(draw, box, title, theme, df):
    x, y, w, h = box
    color = BLUE if theme == "copper" else GREEN
    round_rect(draw, (x, y, x + w, y + h), 6, "#fafdff", LIGHT_BORDER)
    draw.text((x + 52, y + 8), title, font=F["card_title"], fill=color)
    draw.ellipse((x + 26, y + 7, x + 40, y + 21), fill=color)
    draw.rectangle((x + 31, y + 19, x + 35, y + 25), fill=color)
    draw.rectangle((x + 27, y + 24, x + 39, y + 27), fill=color)
    col_w = (w - 18) / 3
    medals = ["#f5a400", "#b9c0ca", "#cc6d23"]
    for i, (_, row) in enumerate(df.head(3).iterrows()):
        cx = x + 9 + i * col_w
        draw.ellipse((cx, y + 37, cx + 15, y + 52), fill=medals[i])
        center_text(draw, (cx, y + 36, cx + 15, y + 52), str(i + 1), F["rank"], "white")
        name = LABELS.get(row["指标"], row["指标"])
        draw.text((cx + 20, y + 34), ellipsed(draw, name, F["body_b"], col_w - 23), font=F["body_b"], fill=TEXT)
        draw.text((cx + 20, y + 45), ellipsed(draw, SUB.get(row["指标"], ""), F["tiny"], col_w - 23), font=F["tiny"], fill=MUTED)
        coef = float(row["标准化系数"])
        coef_fill = RED if coef >= 0 else NEG_GREEN
        draw.text((cx + 20, y + 59), f"{coef:.3f}", font=F["big"], fill=coef_fill)
        draw.text((cx + 20, y + 76), sig_label(str(row["显著性"]).strip(), str(row["方向"])), font=F["body_b"], fill=coef_fill)


def draw_insights(draw, box):
    x, y, w, h = box
    round_rect(draw, (x, y, x + w, y + h), 6, "#fafdff", LIGHT_BORDER)
    draw.ellipse((x + 21, y + 9, x + 36, y + 24), fill=BLUE)
    draw.text((x + 25, y + 8), "$", font=font(11, True), fill="white")
    draw.text((x + 48, y + 9), "核心启示", font=F["card_title"], fill=BLUE)

    draw.ellipse((x + 22, y + 41, x + 40, y + 59), fill=BLUE)
    draw.text((x + 28, y + 43), "铜", font=F["body_b"], fill="white")
    draw.text((x + 48, y + 38), "铜价受库存去化和制造端需求驱动", font=F["insight"], fill=BLUE)
    draw.text((x + 48, y + 51), "LME库存负向主导；线缆PPI、NEV与发电机组正向支撑。", font=F["small"], fill=TEXT)

    draw.ellipse((x + 22, y + 73, x + 40, y + 91), fill=GREEN)
    draw.text((x + 28, y + 75), "铝", font=F["body_b"], fill="white")
    draw.text((x + 48, y + 70), "铝价供应项权重最高，成本项显著抬升", font=F["insight"], fill=GREEN)
    draw.text((x + 48, y + 83), "原铝供应方向分化；天然气、空调和SHFE持仓贡献正向影响。", font=F["small"], fill=TEXT)


def load():
    xl = pd.ExcelFile(INPUT)
    copper = pd.read_excel(INPUT, sheet_name=xl.sheet_names[3]).sort_values("影响强度排名")
    aluminum = pd.read_excel(INPUT, sheet_name=xl.sheet_names[4]).sort_values("影响强度排名")
    return copper.reset_index(drop=True), aluminum.reset_index(drop=True)


def main():
    copper, aluminum = load()
    im = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(im)

    draw_panel(draw, (6, 6, 355, 331), "铜价模型结果", "copper", copper, 0.5)
    draw_panel(draw, (368, 6, 367, 331), "铝价模型结果", "aluminum", aluminum, 2.4)
    draw_top3_card(draw, (6, 345, 238, 113), "铜价关键驱动因素 TOP3", "copper", copper)
    draw_top3_card(draw, (250, 345, 238, 113), "铝价关键驱动因素 TOP3", "aluminum", aluminum)
    draw_insights(draw, (494, 345, 241, 113))

    note = "注：1. 系数为标准化回归系数，按绝对值排序；样本期2021-04至2025-12。2. 显著性：*** p<0.01，** p<0.05，* p<0.10。3. 正/负向表示变量上升对价格环比的影响方向。"
    draw.text((9, 470), note, font=F["tiny"], fill="#394150")
    im.convert("RGB").save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
