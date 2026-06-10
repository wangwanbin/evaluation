"""HellaSwag 常识推理 Benchmark — 从四个选项中选出最合理的句子结尾。

测试模型的常识推理能力，基于日常场景判断最符合逻辑的后续。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.hellaswag")

# 内置 15 道常识推理题
BUILTIN_QUESTIONS = [
    {
        "context": "一个男人走进理发店，坐在椅子上。理发师围上围布，拿起剪刀。",
        "endings": [
            "理发师开始剪头发",
            "理发师开始做饭",
            "男人开始游泳",
            "理发师开始修车"
        ],
        "answer": "A"
    },
    {
        "context": "下雨了，小明没有带伞。他站在商店门口避雨。",
        "endings": [
            "小明在商店门口等雨停",
            "小明开始在雨中跳舞",
            "小明把商店的招牌拆下来当伞",
            "小明打电话叫出租车去游泳馆"
        ],
        "answer": "A"
    },
    {
        "context": "妈妈在厨房里准备晚餐。她打开冰箱，拿出鸡蛋和蔬菜。",
        "endings": [
            "妈妈开始洗菜切菜",
            "妈妈把鸡蛋扔到窗外",
            "妈妈把冰箱推到客厅",
            "妈妈开始组装电脑"
        ],
        "answer": "A"
    },
    {
        "context": "同学们在操场上体育课，老师拿出一个足球。",
        "endings": [
            "同学们开始踢足球",
            "同学们开始写作业",
            "同学们把足球吃掉",
            "同学们用足球打乒乓球"
        ],
        "answer": "A"
    },
    {
        "context": "图书馆里非常安静，大家都在看书。一个小孩子突然大声哭起来。",
        "endings": [
            "家长赶紧把孩子抱出去安抚",
            "大家一起跟着哭",
            "图书管理员开始唱歌",
            "所有人把书扔掉"
        ],
        "answer": "A"
    },
    {
        "context": "医生检查完病人后，开了一张处方。病人拿着处方走出了诊室。",
        "endings": [
            "病人去药房取药",
            "病人去超市买零食",
            "病人把处方折成纸飞机",
            "病人回诊室要求开演唱会"
        ],
        "answer": "A"
    },
    {
        "context": "冬天到了，湖面结了一层厚厚的冰。几个孩子看到了非常开心。",
        "endings": [
            "孩子们在冰面上滑冰玩耍",
            "孩子们往湖里倒热水游泳",
            "孩子们把冰面敲碎种花",
            "孩子们在冰面上晒被子"
        ],
        "answer": "A"
    },
    {
        "context": "考试快要开始了，小李发现忘带铅笔。他向旁边的同学借。",
        "endings": [
            "同学借了一支铅笔给小李",
            "同学把课桌拆了给小李",
            "开始用粉笔答题",
            "用指甲在试卷上刻答案"
        ],
        "answer": "A"
    },
    {
        "context": "在自助餐厅，小王拿了一个盘子，走到取餐区。",
        "endings": [
            "小王用夹子把食物夹到盘子里",
            "小王把盘子顶在头上跳舞",
            "小王把盘子摔在地上",
            "小王用盘子当帽子戴"
        ],
        "answer": "A"
    },
    {
        "context": "一位老人在公交车上站着，没有空座位。",
        "endings": [
            "一名年轻人站起来给老人让座",
            "年轻人把老人推出车门",
            "司机让所有人下车",
            "乘客开始赛跑抢座位"
        ],
        "answer": "A"
    },
    {
        "context": "厨师发现厨房着火了，火苗从锅里蹿起。",
        "endings": [
            "厨师迅速盖上锅盖并关火",
            "厨师往锅里倒汽油",
            "厨师用风扇对着火吹",
            "厨师对着火唱歌"
        ],
        "answer": "A"
    },
    {
        "context": "小华的手机突然没电了，他需要联系家人来接他。",
        "endings": [
            "小华借别人的手机打电话",
            "小华把手机拆开充电",
            "对着手机大喊让家人听到",
            "把手机扔到空中让信号变好"
        ],
        "answer": "A"
    },
    {
        "context": "邮递员拿着一封信走到一家门口，按响了门铃。",
        "endings": [
            "主人开门出来取信",
            "主人把门拆了",
            "主人从窗户跳出来",
            "主人叫邮递员翻墙进来"
        ],
        "answer": "A"
    },
    {
        "context": "周末早上，阳光照进窗户。闹钟响了，小美睁开眼睛。",
        "endings": [
            "小美起床穿衣服准备开始新的一天",
            "小美把闹钟吃掉继续睡觉",
            "小美把窗户拆了扔出去",
            "小美在床上翻跟头"
        ],
        "answer": "A"
    },
    {
        "context": "篮球比赛中，对方球员投篮不中，球弹筐而出。",
        "endings": [
            "我方球员跳起抢到篮板球",
            "裁判宣布对方得分",
            "球员们开始拔河",
            "观众跑进球场抢球"
        ],
        "answer": "A"
    },
]


class HellaSwagBenchmark(BaseLLMBenchmark):
    """HellaSwag 常识推理评测"""

    benchmark_id = "hellaswag"
    benchmark_name = "HellaSwag"
    category = "常识推理"

    def load_questions(self) -> list[dict]:
        """加载常识推理题（15 题）"""
        questions = []
        for i, q in enumerate(BUILTIN_QUESTIONS):
            questions.append({
                "id": f"hellaswag_{i}",
                "context": q["context"],
                "endings": q["endings"],
                "answer": q["answer"],
                "category": "常识推理",
            })
        logger.info(f"HellaSwag 共加载 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建 prompt（中文）"""
        context = question["context"]
        endings = question["endings"]
        choices_str = "\n".join(endings)

        messages = [
            {
                "role": "system",
                "content": "你是常识推理专家。请根据日常逻辑，从四个选项中选择最合理的后续，只输出选项字母（A/B/C/D）。"
            },
            {
                "role": "user",
                "content": f"情境：{context}\n\n请选择最合理的后续：\n{choices_str}\n\n请只输出最合理选项的字母。"
            }
        ]
        return messages

    def extract_answer(self, text: str) -> str:
        """提取答案字母"""
        text = text.strip().upper()

        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1)

        if text in ["A", "B", "C", "D"]:
            return text

        for c in text:
            if c in "ABCD":
                return c
        return ""

    def check_answer(self, extracted: str, reference: str) -> bool:
        return extracted.strip().upper() == reference.strip().upper()


benchmark = HellaSwagBenchmark()
