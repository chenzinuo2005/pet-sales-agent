"""知识库数据生成 — 调用 DeepSeek API 批量生成 5 个知识库 txt"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

DOG_BREEDS = [
    ("american_bulldog", "美国斗牛犬"),
    ("american_pit_bull_terrier", "美国比特犬"),
    ("basset_hound", "巴吉度猎犬"),
    ("beagle", "比格犬"),
    ("boxer", "拳师犬"),
    ("chihuahua", "吉娃娃"),
    ("english_cocker_spaniel", "英国可卡犬"),
    ("english_setter", "英国雪达犬"),
    ("german_shorthaired_pointer", "德国短毛指示犬"),
    ("great_pyrenees", "大白熊犬"),
    ("havanese", "哈瓦那犬"),
    ("japanese_chin", "日本狆"),
    ("keeshond", "荷兰卷尾犬"),
    ("leonberger", "莱昂贝格犬"),
    ("miniature_pinscher", "迷你品犬"),
    ("newfoundland", "纽芬兰犬"),
    ("pomeranian", "博美犬"),
    ("pug", "巴哥犬"),
    ("saint_bernard", "圣伯纳犬"),
    ("samoyed", "萨摩耶"),
    ("scottish_terrier", "苏格兰梗"),
    ("shiba_inu", "柴犬"),
    ("staffordshire_bull_terrier", "斯塔福德斗牛梗"),
    ("wheaten_terrier", "软毛麦色梗"),
    ("yorkshire_terrier", "约克夏梗"),
]

CAT_BREEDS = [
    ("abyssinian", "阿比西尼亚猫"),
    ("bengal", "孟加拉豹猫"),
    ("birman", "伯曼猫"),
    ("bombay", "孟买猫"),
    ("british_shorthair", "英国短毛猫"),
    ("egyptian_mau", "埃及猫"),
    ("maine_coon", "缅因猫"),
    ("persian", "波斯猫"),
    ("ragdoll", "布偶猫"),
    ("russian_blue", "俄罗斯蓝猫"),
    ("siamese", "暹罗猫"),
    ("sphynx", "斯芬克斯无毛猫"),
]

ALL_BREEDS = DOG_BREEDS + CAT_BREEDS
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def call_deepseek(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


def generate_file(filename: str, category: str, batch_size: int = 10):
    """按品种批量生成某个类别的知识库内容，追加写入文件"""
    filepath = os.path.join(OUTPUT_DIR, filename)

    # 清空已有文件
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {category}\n\n")

    for i in range(0, len(ALL_BREEDS), batch_size):
        batch = ALL_BREEDS[i : i + batch_size]
        breed_list = "\n".join([f"- {cn} ({en})" for en, cn in batch])

        user_prompt = f"""请为以下 {len(batch)} 个猫狗品种分别撰写"{category}"内容。

品种列表：
{breed_list}

要求：
1. 每个品种独立一个段落，以"## <中文名>"开头
2. 内容用中文，专业准确，适合宠物店客服参考
3. 每个品种写 100-150 字
4. 直接输出内容，不要额外说明"""

        print(f"  生成 {filename}: 品种 {i+1}-{min(i+batch_size, len(ALL_BREEDS))}/{len(ALL_BREEDS)} ...")
        content = call_deepseek(
            "你是一位资深宠物专家，为宠物店知识库撰写专业内容。直接输出要求的格式，不要额外废话。",
            user_prompt,
        )

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content + "\n\n")

    print(f"  [OK] {filename} done")


def main():
    tasks = [
        # ("breed_info.txt", "品种介绍"),  # already generated
        ("care_guide.txt", "饲养指南（饮食建议、运动需求、美容护理频率、训练难度、适合人群）"),
        ("pricing.txt", "价格参考（宠物级价格区间、赛级价格区间、影响价格的关键因素）"),
        ("health.txt", "常见健康问题（品种常见遗传病、易患疾病、疫苗接种要点、日常健康检查建议）"),
    ]

    for filename, category in tasks:
        print(f"\n{'='*50}")
        print(f"开始生成: {filename}")
        print(f"{'='*50}")
        generate_file(filename, category)

    # after_sales.txt 单独生成（非按品种）
    print(f"\n{'='*50}")
    print("开始生成: after_sales.txt")
    print(f"{'='*50}")
    generate_after_sales()

    print(f"\n全部完成! 文件输出到: {OUTPUT_DIR}")


def generate_after_sales():
    filepath = os.path.join(OUTPUT_DIR, "after_sales.txt")
    prompt = """请为一家名为"萌宠之家"的宠物店撰写售后政策知识库文档。

要求：
1. 涵盖以下方面：健康保证期、退换政策、疫苗驱虫服务、售后咨询渠道、常见售后问题解答
2. 语气温暖专业，体现对宠物和主人的关怀
3. 每个方面独立一段落，以"## <标题>"开头
4. 总计 800-1200 字
5. 直接输出内容，不要额外说明"""

    content = call_deepseek(
        "你是一位宠物店运营专家，为宠物店撰写售后政策文档。内容专业、温暖、实操性强。",
        prompt,
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print("  [OK] after_sales.txt done")


if __name__ == "__main__":
    main()
