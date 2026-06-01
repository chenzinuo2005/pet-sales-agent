"""Build optimized RAG dataset: merge per-breed info from raw data files into unified, retrieval-friendly format.

Usage: uv run python scripts/build_rag_data.py
Output: data/breeds_encyclopedia.txt (all 37 breeds, one self-contained section each)
"""

import re
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Raw source files ──
BREED_INFO = os.path.join(DATA_DIR, "breed_info.txt")
CARE_GUIDE = os.path.join(DATA_DIR, "care_guide.txt")
HEALTH = os.path.join(DATA_DIR, "health.txt")
PRICING = os.path.join(DATA_DIR, "pricing.txt")
OUTPUT = os.path.join(DATA_DIR, "breeds_encyclopedia.txt")

# ── 37 breeds ordered list (must match data files) ──
BREEDS_CN = [
    "美国斗牛犬", "美国比特犬", "巴吉度猎犬", "比格犬", "拳师犬",
    "吉娃娃", "英国可卡犬", "英国雪达犬", "德国短毛指示犬", "大白熊犬",
    "哈瓦那犬", "日本狆", "荷兰卷尾犬", "莱昂贝格犬", "迷你品犬",
    "纽芬兰犬", "博美犬", "巴哥犬", "圣伯纳犬", "萨摩耶",
    "苏格兰梗", "柴犬", "斯塔福德斗牛梗", "软毛麦色梗", "约克夏梗",
    "阿比西尼亚猫", "孟加拉豹猫", "伯曼猫", "孟买猫", "英国短毛猫",
    "埃及猫", "缅因猫", "波斯猫", "布偶猫", "俄罗斯蓝猫",
    "暹罗猫", "斯芬克斯无毛猫",
]

BREED_TYPES = {}
for b in BREEDS_CN:
    BREED_TYPES[b] = "猫" if "猫" in b else "犬"


def load_raw(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def extract_section(text: str, breed_cn: str) -> str:
    """Extract the paragraph about `breed_cn` from a raw data file.

    Sections start with `## breed_name` and end before the next `##`.
    """
    # Normalize: some breeds use `## Name` others `## Name\n- **英文名**...`
    pattern = rf"##\s+{re.escape(breed_cn)}\s*\n(.*?)(?=\n##\s|\Z)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return ""
    content = m.group(1).strip()
    # Remove markdown list markers and extra formatting for cleaner integration
    content = re.sub(r"^\s*-\s*\*\*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\*\*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"\*\*", "", content)
    return content


def build_encyclopedia() -> str:
    info = load_raw(BREED_INFO)
    care = load_raw(CARE_GUIDE)
    health = load_raw(HEALTH)
    pricing = load_raw(PRICING)

    header = (
        "# 萌宠之家 · 猫狗品种百科大全\n"
        "# 涵盖37个猫狗品种的完整信息：基本特征、饲养指南、健康提示、价格参考\n"
        "# 每个品种均为独立检索单元，包含品种识别与选购所需的全部要点\n\n"
    )

    sections = []
    for breed in BREEDS_CN:
        breed_type = BREED_TYPES[breed]
        info_text = extract_section(info, breed)
        care_text = extract_section(care, breed)
        health_text = extract_section(health, breed)
        price_text = extract_section(pricing, breed)

        sections.append(
            f"## {breed} | 类型: {breed_type}\n\n"
            f"### 基本特征\n{info_text}\n\n"
            f"### 饲养指南\n{care_text}\n\n"
            f"### 健康提示\n{health_text}\n\n"
            f"### 价格参考\n{price_text}\n"
        )

    return header + "\n---\n\n".join(sections)


if __name__ == "__main__":
    encyclopedia = build_encyclopedia()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(encyclopedia)
    print(f"Written {len(encyclopedia)} chars to {OUTPUT}")
    print(f"Breeds: {len(BREEDS_CN)}")
