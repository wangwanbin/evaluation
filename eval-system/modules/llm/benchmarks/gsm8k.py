"""GSM8K (小学数学 8K) Benchmark — 小学数学应用题评测。

内置 40 道不同难度和运算类型的数学题。
完整数据从 datasets/llm/gsm8k 加载。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.gsm8k")

# GSM8K few-shot 样例（4 个典型题）
GSM8K_FEW_SHOT = [
    {
        "question": "树丛里有 15 棵树。园丁今天又种了一些，现在有 21 棵树。园丁今天种了几棵树？",
        "steps": "原来有 15 棵树。后来又种了一些，现在有 21 棵。所以种了 21 - 15 = 6 棵。",
        "answer": "6"
    },
    {
        "question": "停车场有 3 辆车，又来了 2 辆，现在有几辆？",
        "steps": "原来有 3 辆车。又来了 2 辆。3 + 2 = 5。",
        "answer": "5"
    },
    {
        "question": "小红有 32 颗巧克力，她姐姐有 42 颗。她们吃了 35 颗后，还剩多少颗？",
        "steps": "原来共有 32 + 42 = 74 颗。吃了 35 颗后，还剩 74 - 35 = 39 颗。",
        "answer": "39"
    },
    {
        "question": "小明有 20 根棒棒糖。他给了小华一些后，还剩下 12 根。小明给了小华几根？",
        "steps": "小明原本有 20 根，给了小华一些后剩 12 根。所以给了 20 - 12 = 8 根。",
        "answer": "8"
    },
]

# 内置 40 道数学题
BUILTIN_QUESTIONS = [
    # === 加减法（简单） ===
    {"question": "小红有 16 张贴纸，她送给朋友 7 张，小红还剩多少张贴纸？", "answer": "9"},
    {"question": "一个农场有 12 只鸡和 8 只鸭，卖掉 5 只鸡后，农场还有几只鸡？", "answer": "7"},
    {"question": "小明有 24 颗糖果，他想要平均分给 6 个朋友，每个朋友能得到几颗？", "answer": "4"},
    {"question": "商店的苹果每斤 3 元，小芳买了 5 斤，需要付多少钱？", "answer": "15"},
    {"question": "学校有 45 名学生，其中 18 名是男生，女生有多少名？", "answer": "27"},
    {"question": "一本书有 86 页，小华第一天读了 23 页，第二天读了 35 页，还剩多少页没读？", "answer": "28"},
    {"question": "公交车上原有 34 人，到站后下去了 12 人，又上来了 8 人，现在车上有多少人？", "answer": "30"},
    {"question": "妈妈买了 3 打鸡蛋（1 打 = 12 个），用了 15 个做蛋糕，还剩多少个鸡蛋？", "answer": "21"},
    {"question": "一根绳子长 2 米，用去了 45 厘米，还剩多少厘米？", "answer": "155"},
    {"question": "商店原来有 50 支笔，上午卖出 18 支，下午又进了 25 支，现在商店有多少支笔？", "answer": "57"},

    # === 乘除法（中等） ===
    {"question": "一个长方形长 8 厘米，宽 5 厘米，它的周长是多少厘米？", "answer": "26"},
    {"question": "一个正方形的边长是 6 厘米，它的面积是多少平方厘米？", "answer": "36"},
    {"question": "某车间有 4 条生产线，每条线每天生产 125 个零件，该车间每天共生产多少个零件？", "answer": "500"},
    {"question": "一本书的价格是 36 元，一套 8 本共多少钱？", "answer": "288"},
    {"question": "有 120 个苹果，每箱装 15 个，可以装满几箱？", "answer": "8"},
    {"question": "一列火车 3 小时行驶了 360 公里，它的平均速度是多少公里/小时？", "answer": "120"},
    {"question": "学校买了 240 本练习本，平均分给 6 个年级，每个年级能分到多少本？", "answer": "40"},
    {"question": "一个长方形的面积是 72 平方米，宽是 6 米，长是多少米？", "answer": "12"},
    {"question": "小明每天存 5 元钱，多少天能存够 150 元？", "answer": "30"},
    {"question": "一箱牛奶有 24 盒，一周内喝掉了 3/4，喝了多少盒？", "answer": "18"},

    # === 多步运算（较难） ===
    {"question": "小明有 50 元，买了一个 28 元的书包和 3 支每支 4 元的笔，还剩多少钱？", "answer": "10"},
    {"question": "一个长方形花坛长 12 米，宽 8 米。如果沿着花坛走 3 圈，共走了多少米？", "answer": "120"},
    {"question": "一本故事书 120 页，小花第一天看了全书的 1/4，第二天看了 35 页，还有多少页没看？", "answer": "55"},
    {"question": "甲乙两地相距 300 公里。一辆汽车从甲地出发，前 2 小时行驶了 120 公里，剩下的路程需要 3 小时开完，平均每小时要开多少公里？", "answer": "60"},
    {"question": "某商店促销，买 3 送 1。小明需要 12 个杯子，他需要付几个杯子的钱？", "answer": "9"},
    {"question": "一个水池装有进水管和出水管。单开进水管 6 小时注满，单开出水管 9 小时排空。两管同时开，多少小时能注满？", "answer": "18"},
    {"question": "树上有一些鸟。飞走了 8 只后，又飞来 5 只，现在树上有 23 只。原来树上有多少只？", "answer": "26"},
    {"question": "一个班级有 48 名学生，其中 1/3 是女生。后来又转来了 4 名女生，现在女生占总人数的几分之几？", "answer": "3/7"},
    {"question": "某商品原价 200 元，先打八折，再降价 10 元，现在售价多少元？", "answer": "150"},
    {"question": "甲有 40 元，乙的钱是甲的一半多 5 元，丙的钱是乙的 2 倍少 10 元，三人共有多少钱？", "answer": "115"},

    # === 实际应用（综合） ===
    {"question": "一个游泳池长 50 米，小明游了 4 个来回，共游了多少米？", "answer": "400"},
    {"question": "妈妈买了 2.5 公斤苹果，每公斤 6.8 元，付了 50 元，应找回多少元？", "answer": "33"},
    {"question": "一辆车每百公里耗油 8 升，每升汽油 7.5 元，行驶 350 公里需要多少油钱？", "answer": "210"},
    {"question": "修一条 600 米的路，前 4 天每天修 75 米，剩下的要在 3 天内修完，平均每天要修多少米？", "answer": "100"},
    {"question": "甲乙两人从相距 24 公里的两地相向而行，甲的速度是 5 公里/小时，乙的速度是 3 公里/小时，几小时后相遇？", "answer": "3"},
    {"question": "一个水池盛满了水，第一天用去 1/4，第二天用去剩下的 2/3，还剩下 10 吨。原来有多少吨水？", "answer": "40"},
    {"question": "从 1 到 100 的所有整数中，数字 9 出现了多少次？", "answer": "20"},
    {"question": "一根竹竿插入水池中，入泥部分长 0.5 米，露出水面的部分比入泥部分长 0.8 米，水深是露出水面部分的 1.5 倍，竹竿全长多少米？", "answer": "3.25"},
    {"question": "一批零件有 240 个，王师傅每小时做 12 个，李师傅每小时做 18 个。两人合作，几小时能完成？", "answer": "8"},
    {"question": "某商场将一种商品按标价的九折出售仍可获利 20%。若该商品的进价为 240 元，标价是多少元？", "answer": "320"},
]


class GSM8KBenchmark(BaseLLMBenchmark):
    """GSM8K 数学推理评测"""

    benchmark_id = "gsm8k"
    benchmark_name = "GSM8K"
    category = "数学推理"

    def __init__(self):
        super().__init__()
        self._questions: list[dict] = []
        self._data_path = Path(self.config.results_dir).parent.parent / "datasets" / "llm" / "gsm8k"

    def load_questions(self) -> list[dict]:
        """加载 GSM8K 题目（优先从文件，回退到内置 40 题）"""
        questions = []

        if self._data_path.exists():
            questions = self._load_from_files()

        if not questions:
            logger.info("未找到数据集文件，使用内置评测数据（40 题）")
            questions = [
                {**q, "id": f"gsm8k_{i}"}
                for i, q in enumerate(BUILTIN_QUESTIONS)
            ]

        logger.info(f"GSM8K 共加载 {len(questions)} 道题目")
        return questions

    def _load_from_files(self) -> list[dict]:
        """从 JSON 文件加载题目"""
        questions = []
        try:
            for f in sorted(self._data_path.glob("*.json")):
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for i, item in enumerate(data):
                    item["id"] = f"gsm8k_{len(questions)}"
                    questions.append(item)
        except Exception as e:
            logger.warning(f"加载 GSM8K 数据集失败: {e}")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建带 CoT 的 prompt（中文）"""
        messages = []

        if few_shot:
            messages.append({
                "role": "system",
                "content": "你是数学解题助手。请逐步推理并解答以下数学问题。在最后一行输出「答案是 X」，其中 X 是最终数字答案。"
            })

            for ex in GSM8K_FEW_SHOT[:4]:
                messages.append({"role": "user", "content": ex["question"]})
                messages.append({
                    "role": "assistant",
                    "content": f"{ex['steps']} 答案是 {ex['answer']}。"
                })
        else:
            messages.append({
                "role": "system",
                "content": "你是数学解题助手。请逐步推理，最终输出「答案是 X」。"
            })

        messages.append({"role": "user", "content": question["question"]})
        return messages

    def extract_answer(self, text: str) -> str:
        """从模型输出中提取最终数字答案"""
        patterns = [
            r'(?:答案是?|answer\s+is)\s*(-?\d+(?:\.?\d+)?(?:/\d+)?)',
            r'(?:答案是?|answer\s+is)\s*(-?\d+(?:\.?\d+)?)',
            r'(-?\d+(?:\.?\d+)?)\s*$',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        numbers = re.findall(r'-?\d+(?:\.?\d+)?', text)
        return numbers[-1] if numbers else text.strip()

    def check_answer(self, extracted: str, reference: str) -> bool:
        """判断数字答案是否一致"""
        try:
            ext_clean = extracted.strip().rstrip('.')
            ref_clean = reference.strip().rstrip('.')

            if ext_clean == ref_clean:
                return True

            ext_num = float(ext_clean) if '/' not in ext_clean else eval(ext_clean)
            ref_num = float(ref_clean) if '/' not in ref_clean else eval(ref_clean)
            return abs(ext_num - ref_num) < 0.01

        except (ValueError, ZeroDivisionError, SyntaxError):
            return extracted.strip() == reference.strip()


benchmark = GSM8KBenchmark()
