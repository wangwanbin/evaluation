"""Needle-in-a-Haystack（大海捞针）长文本理解 Benchmark。

测试模型在长文本中检索和定位特定信息的能力。
动态生成长文本，不依赖外部数据。
"""

from __future__ import annotations

import logging
import random
import re
from typing import Optional

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.needle")

# 内置测试场景
BUILTIN_SCENARIOS = [
    {
        "needle": "爱因斯坦出生于德国的乌尔姆市",
        "haystack_context": "物理学发展",
        "question": "根据上文，爱因斯坦出生在哪里？",
        "answer": "德国的乌尔姆市",
        "position": "middle",
    },
    {
        "needle": "Python 语言由吉多·范罗苏姆于 1991 年首次发布",
        "haystack_context": "编程语言历史",
        "question": "根据上文，Python 语言是由谁在什么时候首次发布的？",
        "answer": "吉多·范罗苏姆于 1991 年",
        "position": "early",
    },
    {
        "needle": "珠穆朗玛峰的海拔为 8848.86 米",
        "haystack_context": "世界地理",
        "question": "根据上文，珠穆朗玛峰的海拔是多少？",
        "answer": "8848.86 米",
        "position": "late",
    },
    {
        "needle": "量子纠缠是指两个或多个粒子之间存在一种特殊的关联，测量其中一个会立即影响另一个",
        "haystack_context": "量子物理",
        "question": "根据上文，什么是量子纠缠？",
        "answer": "两个或多个粒子之间存在一种特殊的关联，测量其中一个会立即影响另一个",
        "position": "middle",
    },
    {
        "needle": "2024 年夏季奥运会在法国巴黎举行",
        "haystack_context": "体育赛事",
        "question": "根据上文，2024 年夏季奥运会在哪里举行？",
        "answer": "法国巴黎",
        "position": "various",
    },
]

# 填充文本（用于生成长文本）
FILLER_SENTENCES = [
    "历史长河中，人类文明不断进步和发展，各个领域都取得了令人瞩目的成就。",
    "科学技术的发展极大地改变了人们的生活方式，提高了生产效率和生活质量。",
    "教育是立国之本，一个国家的发展离不开对教育的重视和投入。",
    "文化多样性是人类社会的重要特征，不同文明之间的交流互鉴推动了世界发展。",
    "经济全球化使得各国之间的联系更加紧密，合作共赢成为时代主题。",
    "环境保护越来越受到人们的重视，可持续发展理念深入人心。",
    "人工智能技术的发展正在深刻改变各个行业的运行模式。",
    "互联网的普及使得信息传播速度大大加快，人们获取知识的途径更加多元。",
    "创新是一个国家发展的不竭动力，科技进步需要持续的基础研究投入。",
    "社会公平正义是人类追求的共同理想，法治建设是保障公平的重要基础。",
    "医疗健康领域的发展直接关系到人民的福祉，预防医学越来越受到重视。",
    "粮食安全是国家安全的重要基础，现代农业技术为粮食增产提供了保障。",
    "新能源技术的研究和应用对于应对气候变化具有重要意义。",
    "城市化的快速发展带来了机遇与挑战，智慧城市建设成为新趋势。",
    "量子计算、生物技术、新材料等前沿科技正在开辟新的发展空间。",
    "心理健康问题日益受到社会关注，建立完善的心理健康服务体系十分重要。",
    "数字经济的发展为传统产业转型升级提供了新的动力和机遇。",
    "太空探索拓展了人类认知的边界，航天技术的进步推动着科学发现。",
    "水资源保护是全球面临的共同挑战，节约用水需要每个人的参与。",
    "终身学习的理念越来越深入人心，知识更新速度加快要求人们不断学习。",
    "在全球化的背景下，跨文化交流能力变得越来越重要。",
    "人口老龄化是许多国家面临的社会问题，养老服务体系需要不断完善。",
    "青少年教育关系到国家的未来，素质教育改革正在深入推进。",
    "数据安全和个人隐私保护在数字化时代显得尤为重要。",
    "交通运输基础设施的完善为经济发展提供了有力支撑。",
    "生物多样性保护对于维持生态系统平衡具有不可替代的作用。",
    "食品安全关系到每个人的身体健康，需要建立严格的监管体系。",
    "就业是最大的民生，职业教育和技能培训在促进就业中发挥着重要作用。",
    "住房问题关系到人民的基本生活需求，房地产市场健康发展十分重要。",
    "社区治理创新是基层社会治理现代化的重要组成部分。",
]


def _build_long_text(needle: str, context: str, position: str = "middle",
                     target_length: int = 3000) -> str:
    """构建包含目标信息的长文本"""
    parts = []

    # 填充前文
    filler_count = 10
    if position == "early":
        # needle 在前 1/4 处
        for _ in range(filler_count // 2):
            parts.append(random.choice(FILLER_SENTENCES))
        parts.append(f"在讨论{context}时，有一个重要的信息：{needle}。")
        for _ in range(filler_count * 3):
            parts.append(random.choice(FILLER_SENTENCES))
    elif position == "late":
        # needle 在后 1/4 处
        for _ in range(filler_count * 3):
            parts.append(random.choice(FILLER_SENTENCES))
        parts.append(f"值得注意的是，在{context}领域，{needle}。")
        for _ in range(filler_count // 2):
            parts.append(random.choice(FILLER_SENTENCES))
    else:
        # needle 在中间
        for _ in range(filler_count * 2):
            parts.append(random.choice(FILLER_SENTENCES))
        parts.append(f"在{context}方面，需要特别指出的是，{needle}。")
        for _ in range(filler_count * 2):
            parts.append(random.choice(FILLER_SENTENCES))

    text = "".join(parts)

    # 调整长度
    while len(text) < target_length:
        text += random.choice(FILLER_SENTENCES)

    return text[:target_length]


class NeedleInHaystackBenchmark(BaseLLMBenchmark):
    """大海捞针长文本理解评测"""

    benchmark_id = "needle"
    benchmark_name = "Needle-in-a-Haystack"
    category = "长文本理解"

    def __init__(self):
        super().__init__()
        random.seed(42)  # 固定种子确保可复现

    def load_questions(self) -> list[dict]:
        """生成测试题（5 种场景 × 3 种位置 = 15 题）"""
        questions = []
        idx = 0
        for scenario in BUILTIN_SCENARIOS:
            for pos in ["early", "middle", "late"]:
                long_text = _build_long_text(
                    needle=scenario["needle"],
                    context=scenario["haystack_context"],
                    position=pos,
                )
                questions.append({
                    "id": f"needle_{idx}",
                    "long_text": long_text,
                    "question": scenario["question"],
                    "needle": scenario["needle"],
                    "answer": scenario["answer"],
                    "position": pos,
                    "text_length": len(long_text),
                    "category": "长文本理解",
                })
                idx += 1
        logger.info(f"Needle-in-a-Haystack 共生成 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建包含长文本的 prompt"""
        messages = [
            {
                "role": "system",
                "content": "你是一个长文本阅读理解助手。请仔细阅读以下文本，然后回答相关问题。回答应简洁准确。"
            },
            {
                "role": "user",
                "content": (
                    f"请阅读以下文本：\n\n"
                    f"{question['long_text']}\n\n"
                    f"---\n\n"
                    f"问题：{question['question']}\n\n"
                    f"请根据上文内容回答问题。"
                )
            }
        ]
        return messages

    def extract_answer(self, text: str) -> str:
        """提取回答"""
        text = text.strip()
        return text

    def check_answer(self, extracted: str, reference: str) -> bool:
        """判断是否包含关键信息"""
        ref_lower = reference.strip().lower()
        ext_lower = extracted.strip().lower()

        # 检查是否包含参考答案中的关键词
        ref_words = ref_lower.replace('？', '').replace('，', ',').replace('。', '.').split()
        key_parts = [p for p in ref_words if len(p) >= 3]

        match_count = sum(1 for p in key_parts if p in ext_lower)
        threshold = max(1, len(key_parts) * 0.5)
        return match_count >= threshold


benchmark = NeedleInHaystackBenchmark()
