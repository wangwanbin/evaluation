"""PIQA (Physical Interaction: Question Answering) 物理常识推理 Benchmark。

测试模型对日常物理交互的理解，需要判断完成特定目标最合理的方式。
"""

from __future__ import annotations

import logging
import re

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.piqa")

# 内置 15 道物理常识题
BUILTIN_QUESTIONS = [
    {
        "goal": "打开一个拧得很紧的瓶盖",
        "choices": ["A) 用毛巾包住瓶盖用力拧", "B) 把瓶子放进冰箱加热", "C) 用锤子敲碎瓶盖", "D) 把瓶子放在桌上等它自己开"],
        "answer": "A",
    },
    {
        "goal": "将墙上的一幅画挂正",
        "choices": ["A) 用螺丝刀拧紧画框背后的挂钩", "B) 用胶水把画粘在墙上", "C) 用钉子把画的四角钉死", "D) 把画靠在墙边"],
        "answer": "A",
    },
    {
        "goal": "快速冷却一杯热水",
        "choices": ["A) 把杯子放在冰箱里", "B) 把杯子放在阳光下晒", "C) 把杯子放进微波炉加热", "D) 给杯子盖上盖子"],
        "answer": "A",
    },
    {
        "goal": "从书架上取下一本够不到的书",
        "choices": ["A) 搬梯子或凳子垫高去拿", "B) 把书架推倒让书掉下来", "C) 用扫帚把书捅下来", "D) 用力摇晃书架让书滑落"],
        "answer": "A",
    },
    {
        "goal": "清理地板上的碎玻璃",
        "choices": ["A) 戴上手套用扫帚和簸箕清扫", "B) 用吸尘器直接吸", "C) 用手直接捡", "D) 用水冲走"],
        "answer": "A",
    },
    {
        "goal": "让一把生锈的锁恢复正常使用",
        "choices": ["A) 往锁芯里滴润滑油并反复转动", "B) 用锤子把锁砸开", "C) 把锁放进水里泡", "D) 用火烤锁芯"],
        "answer": "A",
    },
    {
        "goal": "用一根吸管喝饮料时，吸管底部破了",
        "choices": ["A) 把吸管的破口处剪掉再喝", "B) 用胶带把破口缠紧继续喝", "C) 把吸管折起来堵住破口", "D) 倒过来用另一端喝"],
        "answer": "A",
    },
    {
        "goal": "切蛋糕时让每一块大小均匀",
        "choices": ["A) 先切出中心线再等分切割", "B) 从边上开始一片片切", "C) 用力一刀切成两半再切小块", "D) 用手掰开"],
        "answer": "A",
    },
    {
        "goal": "把一颗松动的螺丝拧紧",
        "choices": ["A) 使用螺丝刀顺时针旋转螺丝", "B) 逆时针旋转螺丝刀", "C) 用钳子夹住螺丝往外拔", "D) 用锤子把螺丝敲进去"],
        "answer": "A",
    },
    {
        "goal": "测量一个不规则的石头体积",
        "choices": ["A) 放入量筒中用排水法测量", "B) 用尺子量长宽高计算", "C) 用天平称重后估算", "D) 把石头切开再量"],
        "answer": "A",
    },
    {
        "goal": "延长鲜花的保鲜时间",
        "choices": ["A) 斜剪花茎末端放入清水中", "B) 把花放在阳光下暴晒", "C) 把花放进冰箱冷冻室", "D) 用吹风机把花吹干"],
        "answer": "A",
    },
    {
        "goal": "用传统方法生火（钻木取火）",
        "choices": ["A) 快速持续摩擦木头产生热量", "B) 慢慢旋转木棍", "C) 把木头放在水中浸泡", "D) 用舌头舔木棍让其变干"],
        "answer": "A",
    },
    {
        "goal": "在一个没有开瓶器的情况下打开红酒瓶",
        "choices": ["A) 用螺丝钉拧入软木塞后用钳子拔出", "B) 用锤子把瓶口敲碎", "C) 用嘴咬住软木塞用力拉", "D) 把瓶子倒过来用力拍瓶底"],
        "answer": "A",
    },
    {
        "goal": "冬天防止玻璃窗结冰",
        "choices": ["A) 在窗玻璃内侧涂抹肥皂水", "B) 打开窗户让冷空气进来", "C) 用热水浇窗户", "D) 在窗户上贴满报纸遮光"],
        "answer": "A",
    },
    {
        "goal": "把一段铜线从中间切断",
        "choices": ["A) 用钢丝钳夹住铜线用力剪断", "B) 用手把铜线折断", "C) 用打火机烧断铜线", "D) 用牙咬断"],
        "answer": "A",
    },
]


class PIQABenchmark(BaseLLMBenchmark):
    """PIQA 物理常识推理评测"""

    benchmark_id = "piqa"
    benchmark_name = "PIQA"
    category = "常识推理"

    def load_questions(self) -> list[dict]:
        questions = []
        for i, q in enumerate(BUILTIN_QUESTIONS):
            questions.append({
                "id": f"piqa_{i}",
                "goal": q["goal"],
                "choices": q["choices"],
                "answer": q["answer"],
                "category": "物理常识",
            })
        logger.info(f"PIQA 共加载 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        choices_str = "\n".join(question["choices"])
        messages = [
            {
                "role": "system",
                "content": "你是物理常识推理专家。请根据日常物理知识，选择完成以下目标最合理的方式，只输出选项字母（A/B/C/D）。"
            },
            {
                "role": "user",
                "content": f"目标：{question['goal']}\n请选择最合理的方式：\n{choices_str}\n\n请只输出最合理选项的字母。"
            }
        ]
        return messages

    def extract_answer(self, text: str) -> str:
        text = text.strip().upper()
        match = re.search(r'\b([A-D])\b', text)
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


benchmark = PIQABenchmark()
