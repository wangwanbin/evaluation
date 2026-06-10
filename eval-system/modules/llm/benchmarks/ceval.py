"""C-Eval 中文知识 Benchmark — 涵盖人文、社科、理工、医学等领域的中文选择题。

C-Eval 是一个全面的中文基础模型评测数据集，覆盖 52 个学科。
"""

from __future__ import annotations

import logging
import re

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.ceval")

# 内置 30 道中文知识题
BUILTIN_QUESTIONS = [
    # === 大学数学 ===
    {"question": "函数 f(x)=x² 在 x=2 处的导数是？", "choices": ["A) 2", "B) 4", "C) 6", "D) 8"], "answer": "B", "category": "大学数学"},
    {"question": "∫₀¹ x dx 等于多少？", "choices": ["A) 0", "B) 0.5", "C) 1", "D) 2"], "answer": "B", "category": "大学数学"},
    {"question": "矩阵 [[1,2],[3,4]] 的行列式是？", "choices": ["A) -2", "B) 2", "C) 4", "D) -4"], "answer": "A", "category": "大学数学"},

    # === 大学物理 ===
    {"question": "光的传播速度在真空中约为多少？", "choices": ["A) 3×10⁶ m/s", "B) 3×10⁸ m/s", "C) 3×10¹⁰ m/s", "D) 3×10⁴ m/s"], "answer": "B", "category": "大学物理"},
    {"question": "一杯 100°C 的开水和一杯 200g 的冰水混合后，最终温度最可能接近？", "choices": ["A) 0°C", "B) 50°C", "C) 100°C", "D) 取决于水的质量"], "answer": "D", "category": "大学物理"},
    {"question": "电阻 R₁=2Ω 和 R₂=3Ω 串联后的总电阻是多少？", "choices": ["A) 1.2Ω", "B) 5Ω", "C) 6Ω", "D) 2/3Ω"], "answer": "B", "category": "大学物理"},

    # === 中国文学 ===
    {"question": "鲁迅的《狂人日记》是中国现代文学史上第一篇什么体裁的作品？", "choices": ["A) 白话小说", "B) 文言小说", "C) 散文", "D) 诗歌"], "answer": "A", "category": "中国文学"},
    {"question": "「床前明月光」是谁的诗句？", "choices": ["A) 杜甫", "B) 李白", "C) 白居易", "D) 王维"], "answer": "B", "category": "中国文学"},
    {"question": "四大名著中《红楼梦》的作者是？", "choices": ["A) 罗贯中", "B) 吴承恩", "C) 曹雪芹", "D) 施耐庵"], "answer": "C", "category": "中国文学"},

    # === 中国历史 ===
    {"question": "秦始皇统一六国是在哪一年？", "choices": ["A) 公元前 230 年", "B) 公元前 221 年", "C) 公元前 210 年", "D) 公元前 206 年"], "answer": "B", "category": "中国历史"},
    {"question": "辛亥革命发生在哪一年？", "choices": ["A) 1900 年", "B) 1911 年", "C) 1919 年", "D) 1921 年"], "answer": "B", "category": "中国历史"},
    {"question": "唐朝的开国皇帝是谁？", "choices": ["A) 李世民", "B) 李渊", "C) 李治", "D) 李隆基"], "answer": "B", "category": "中国历史"},

    # === 法律 ===
    {"question": "我国最高人民法院是国家的什么机关？", "choices": ["A) 立法机关", "B) 行政机关", "C) 司法机关", "D) 监察机关"], "answer": "C", "category": "法律"},
    {"question": "我国宪法规定，公民有受教育的什么？", "choices": ["A) 权利", "B) 义务", "C) 权力", "D) 权利和义务"], "answer": "D", "category": "法律"},
    {"question": "民法典于哪一年正式施行？", "choices": ["A) 2019 年", "B) 2020 年", "C) 2021 年", "D) 2022 年"], "answer": "C", "category": "法律"},

    # === 计算机科学 ===
    {"question": "TCP/IP 协议中 IP 地址 IPv4 的长度是多少位？", "choices": ["A) 16 位", "B) 32 位", "C) 64 位", "D) 128 位"], "answer": "B", "category": "计算机科学"},
    {"question": "时间复杂度 O(n²) 的算法最适合描述哪种性能特征？", "choices": ["A) 常数时间", "B) 线性时间", "C) 平方时间", "D) 对数时间"], "answer": "C", "category": "计算机科学"},
    {"question": "在数据库中，用于唯一标识一条记录的字段称为？", "choices": ["A) 索引", "B) 外键", "C) 主键", "D) 约束"], "answer": "C", "category": "计算机科学"},

    # === 经济学 ===
    {"question": "通货膨胀时，中央银行通常会采取什么货币政策？", "choices": ["A) 降息", "B) 加息", "C) 增加货币发行", "D) 降低存款准备金率"], "answer": "B", "category": "经济学"},
    {"question": "需求定律表明，在其他条件不变时，价格上升会导致需求量？", "choices": ["A) 增加", "B) 减少", "C) 不变", "D) 先增后减"], "answer": "B", "category": "经济学"},
    {"question": "GDP 的中文全称是？", "choices": ["A) 国民生产总值", "B) 国内生产总值", "C) 居民消费价格指数", "D) 生产者物价指数"], "answer": "B", "category": "经济学"},

    # === 医学 ===
    {"question": "人体最大的淋巴器官是？", "choices": ["A) 胸腺", "B) 脾脏", "C) 淋巴结", "D) 扁桃体"], "answer": "B", "category": "医学"},
    {"question": "高血压的诊断标准是血压高于？", "choices": ["A) 120/80 mmHg", "B) 130/85 mmHg", "C) 140/90 mmHg", "D) 150/95 mmHg"], "answer": "C", "category": "医学"},
    {"question": "胰岛素分泌不足会导致什么疾病？", "choices": ["A) 甲状腺功能亢进", "B) 糖尿病", "C) 高血压", "D) 贫血"], "answer": "B", "category": "医学"},

    # === 哲学 ===
    {"question": "马克思主义哲学的核心观点是？", "choices": ["A) 唯心主义", "B) 辩证唯物主义", "C) 存在主义", "D) 实用主义"], "answer": "B", "category": "哲学"},
    {"question": "「量变引起质变」体现的是什么规律？", "choices": ["A) 对立统一规律", "B) 质量互变规律", "C) 否定之否定规律", "D) 因果规律"], "answer": "B", "category": "哲学"},
    {"question": "实践是检验真理的唯一标准，这个观点强调什么？", "choices": ["A) 理论的优先性", "B) 实践的优先性", "C) 认识的优先性", "D) 感觉的优先性"], "answer": "B", "category": "哲学"},

    # === 教育学 ===
    {"question": "提出「教育即生活」的教育家是？", "choices": ["A) 杜威", "B) 赫尔巴特", "C) 夸美纽斯", "D) 卢梭"], "answer": "A", "category": "教育学"},
    {"question": "因材施教原则强调的是？", "choices": ["A) 统一要求", "B) 个体差异", "C) 循序渐进", "D) 理论联系实际"], "answer": "B", "category": "教育学"},
]

FEW_SHOT_EXAMPLES = [
    {"question": "中国的首都是？", "choices": ["A) 上海", "B) 北京", "C) 广州", "D) 深圳"], "answer": "B"},
    {"question": "水的化学式是？", "choices": ["A) CO₂", "B) H₂O", "C) NaCl", "D) CH₄"], "answer": "B"},
]


class CEvalBenchmark(BaseLLMBenchmark):
    """C-Eval 中文知识评测"""

    benchmark_id = "ceval"
    benchmark_name = "C-Eval"
    category = "知识储备"

    def load_questions(self) -> list[dict]:
        questions = []
        for i, q in enumerate(BUILTIN_QUESTIONS):
            questions.append({
                "id": f"ceval_{i}",
                "question": q["question"],
                "choices": q["choices"],
                "answer": q["answer"],
                "category": q.get("category", "中文知识"),
            })
        logger.info(f"C-Eval 共加载 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        messages = [{
            "role": "system",
            "content": "你是 C-Eval 中文知识评测的答题专家。请回答以下中文知识选择题，只输出选项字母（A/B/C/D）。"
        }]
        if few_shot:
            for ex in FEW_SHOT_EXAMPLES[:2]:
                choices_str = "\n".join(ex["choices"])
                messages.append({"role": "user", "content": f"题目：{ex['question']}\n{choices_str}"})
                messages.append({"role": "assistant", "content": ex["answer"]})
        choices_str = "\n".join(question["choices"])
        messages.append({
            "role": "user",
            "content": f"题目：{question['question']}\n{choices_str}\n请只输出正确答案的选项字母。"
        })
        return messages

    def extract_answer(self, text: str) -> str:
        text = text.strip().upper()
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1)
        match = re.search(r'(?:答案[是为]?)\s*([A-D])', text)
        if match:
            return match.group(1)
        if text in "ABCD":
            return text
        for c in text:
            if c in "ABCD":
                return c
        return ""

    def check_answer(self, extracted: str, reference: str) -> bool:
        return extracted.strip().upper() == reference.strip().upper()


benchmark = CEvalBenchmark()
