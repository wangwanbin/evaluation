"""MMLU (大规模多任务语言理解) Benchmark。

覆盖 57 个学科，分为人文、社科、理工、医学四大类。
内置 15 个学科的样例数据用于演示，完整数据从 datasets/llm/mmlu 加载。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from core.config import load_config
from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.mmlu")

# ============================================================
# MMLU 57 学科列表（按大类分组）
# ============================================================
MMLU_SUBJECTS = [
    # 理工类
    "abstract_algebra", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    # 生物医学类
    "anatomy", "clinical_knowledge", "college_biology", "college_medicine",
    "human_aging", "human_sexuality", "medical_genetics",
    "nutrition", "professional_medicine", "professional_psychology", "virology",
    # 社科类
    "business_ethics", "econometrics", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "high_school_psychology",
    "management", "marketing", "public_relations", "security_studies",
    "sociology", "us_foreign_policy",
    # 人文类
    "formal_logic", "global_facts", "high_school_european_history",
    "high_school_us_history", "high_school_world_history",
    "international_law", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory",
    "professional_accounting", "professional_law", "world_religions",
    # 交叉类
    "astronomy", "college_chemistry", "high_school_biology",
    "high_school_chemistry", "miscellaneous",
]

MMLU_SUBJECTS_CN = {
    # 理工类
    "abstract_algebra": "抽象代数", "college_computer_science": "大学计算机科学",
    "college_mathematics": "大学数学", "college_physics": "大学物理",
    "computer_security": "计算机安全", "conceptual_physics": "概念物理",
    "electrical_engineering": "电气工程", "elementary_mathematics": "初等数学",
    "high_school_computer_science": "高中计算机科学",
    "high_school_mathematics": "高中数学", "high_school_physics": "高中物理",
    "high_school_statistics": "高中统计学", "machine_learning": "机器学习",
    # 生物医学类
    "anatomy": "解剖学", "clinical_knowledge": "临床知识",
    "college_biology": "大学生物学", "college_medicine": "大学医学",
    "human_aging": "人类衰老", "human_sexuality": "人类性学",
    "medical_genetics": "医学遗传学", "nutrition": "营养学",
    "professional_medicine": "专业医学", "professional_psychology": "专业心理学",
    "virology": "病毒学",
    # 社科类
    "business_ethics": "商业伦理", "econometrics": "计量经济学",
    "high_school_geography": "高中地理",
    "high_school_government_and_politics": "高中政府与政治",
    "high_school_macroeconomics": "高中宏观经济",
    "high_school_microeconomics": "高中微观经济",
    "high_school_psychology": "高中心理学",
    "management": "管理学", "marketing": "市场营销",
    "public_relations": "公共关系", "security_studies": "安全研究",
    "sociology": "社会学", "us_foreign_policy": "美国外交政策",
    # 人文类
    "formal_logic": "形式逻辑", "global_facts": "全球事实",
    "high_school_european_history": "高中欧洲史",
    "high_school_us_history": "高中美国史",
    "high_school_world_history": "高中世界史",
    "international_law": "国际法", "jurisprudence": "法理学",
    "logical_fallacies": "逻辑谬误", "moral_disputes": "道德争议",
    "moral_scenarios": "道德场景", "philosophy": "哲学",
    "prehistory": "史前学", "professional_accounting": "专业会计",
    "professional_law": "专业法律", "world_religions": "世界宗教",
    # 交叉类
    "astronomy": "天文学", "college_chemistry": "大学化学",
    "high_school_biology": "高中生物", "high_school_chemistry": "高中化学",
    "miscellaneous": "杂项",
}

# ============================================================
# 内置 Few-shot 样例（每科 3 题，用于构建 prompt）
# ============================================================
BUILTIN_FEW_SHOT: dict[str, list[dict]] = {
    "abstract_algebra": [
        {"question": "有限域 GF(7) 中，3 的乘法逆元是多少？", "choices": ["A) 2", "B) 3", "C) 5", "D) 6"], "answer": "C"},
        {"question": "以下哪个集合在矩阵乘法下构成群？", "choices": ["A) 所有 2×2 矩阵", "B) 所有行列式非零的 2×2 矩阵", "C) 所有对称矩阵", "D) 所有对角矩阵"], "answer": "B"},
        {"question": "Z_6 中元素 4 的阶（加法阶）是多少？", "choices": ["A) 6", "B) 4", "C) 3", "D) 2"], "answer": "C"},
    ],
    "college_physics": [
        {"question": "一个物体从静止开始以 2m/s² 的加速度匀加速运动，5 秒后的速度是多少？", "choices": ["A) 5 m/s", "B) 10 m/s", "C) 15 m/s", "D) 20 m/s"], "answer": "B"},
        {"question": "光的波长 500nm，频率约为多少？", "choices": ["A) 3×10¹² Hz", "B) 6×10¹⁴ Hz", "C) 3×10¹⁴ Hz", "D) 6×10¹² Hz"], "answer": "B"},
        {"question": "理想气体状态方程是？", "choices": ["A) PV = nRT", "B) PV = nkT", "C) PV² = nRT", "D) P²V = nRT"], "answer": "A"},
    ],
    "machine_learning": [
        {"question": "以下哪个是监督学习算法？", "choices": ["A) K-means", "B) PCA", "C) 决策树", "D) Apriori"], "answer": "C"},
        {"question": "SVM 的中文全称是？", "choices": ["A) 支持向量机", "B) 系统虚拟内存", "C) 序列值映射", "D) 统计方差模型"], "answer": "A"},
        {"question": "分类问题中最常用的评估指标是？", "choices": ["A) 均方误差", "B) 准确率", "C) R²", "D) 轮廓系数"], "answer": "B"},
    ],
}

# ============================================================
# 内置评测题目（15 个学科 × 5 题 = 75 题）
# ============================================================
BUILTIN_QUESTIONS: dict[str, list[dict]] = {

    # === 理工类 ===
    "abstract_algebra": [
        {"question": "在群 Z₁₂（加法）中，元素 5 的阶是多少？", "choices": ["A) 12", "B) 5", "C) 6", "D) 4"], "answer": "A"},
        {"question": "以下哪个是域？", "choices": ["A) Z", "B) Z₅", "C) Z₆", "D) Z[x]"], "answer": "B"},
        {"question": "Z_n 是域当且仅当 n 是：", "choices": ["A) 素数", "B) 合数", "C) 任意整数", "D) 偶数"], "answer": "A"},
        {"question": "域的特征一定是什么？", "choices": ["A) 合数", "B) 素数或 0", "C) 偶数", "D) 奇数"], "answer": "B"},
        {"question": "以下哪个是域 Q(√2) 在 Q 上的一组基？", "choices": ["A) {1, √2}", "B) {1}", "C) {√2}", "D) {1, 2}"], "answer": "A"},
    ],
    "college_physics": [
        {"question": "牛顿第一定律也被称为：", "choices": ["A) 惯性定律", "B) 加速度定律", "C) 作用反作用定律", "D) 万有引力定律"], "answer": "A"},
        {"question": "一个 2kg 的物体以 3m/s 运动，其动能为：", "choices": ["A) 3J", "B) 6J", "C) 9J", "D) 12J"], "answer": "C"},
        {"question": "光的本质在量子力学中被描述为：", "choices": ["A) 波", "B) 粒子", "C) 波粒二象性", "D) 场"], "answer": "C"},
        {"question": "绝对零度相当于多少摄氏度？", "choices": ["A) -100°C", "B) -273.15°C", "C) 0°C", "D) -373.15°C"], "answer": "B"},
        {"question": "电磁波谱中波长最长的是：", "choices": ["A) 紫外线", "B) X射线", "C) 无线电波", "D) 红外线"], "answer": "C"},
    ],
    "machine_learning": [
        {"question": "过拟合是指模型：", "choices": ["A) 训练误差小测试误差大", "B) 训练误差大测试误差小", "C) 两者误差都大", "D) 两者误差都小"], "answer": "A"},
        {"question": "以下哪种方法常用于防止过拟合？", "choices": ["A) 正则化", "B) 增加学习率", "C) 减少数据量", "D) 增加模型层数"], "answer": "A"},
        {"question": "BP 神经网络中的 BP 是指：", "choices": ["A) 反向传播", "B) 正向传播", "C) 批量处理", "D) 基函数"], "answer": "A"},
        {"question": "以下哪个不是集成学习方法？", "choices": ["A) Random Forest", "B) XGBoost", "C) K-means", "D) AdaBoost"], "answer": "C"},
        {"question": "GAN 由哪两部分组成？", "choices": ["A) 生成器和判别器", "B) 编码器和解码器", "C) 卷积和池化", "D) 注意力和全连接"], "answer": "A"},
    ],
    "computer_security": [
        {"question": "以下哪种是对称加密算法？", "choices": ["A) RSA", "B) AES", "C) ECC", "D) Diffie-Hellman"], "answer": "B"},
        {"question": "SQL 注入攻击属于哪类安全威胁？", "choices": ["A) 注入攻击", "B) 中间人攻击", "C) DoS 攻击", "D) 暴力破解"], "answer": "A"},
        {"question": "HTTPS 使用的默认端口是：", "choices": ["A) 80", "B) 443", "C) 22", "D) 8080"], "answer": "B"},
        {"question": "以下哪种认证方式安全性最高？", "choices": ["A) 密码", "B) 短信验证码", "C) 指纹 + 密码", "D) 手势图案"], "answer": "C"},
        {"question": "XSS 攻击的中文全称是：", "choices": ["A) 跨站脚本攻击", "B) 跨站请求伪造", "C) 缓冲区溢出", "D) 拒绝服务攻击"], "answer": "A"},
    ],

    # === 生物医学类 ===
    "anatomy": [
        {"question": "人体上臂的骨骼是：", "choices": ["A) 股骨", "B) 胫骨", "C) 肱骨", "D) 桡骨"], "answer": "C"},
        {"question": "人体心脏有几个腔室？", "choices": ["A) 2 个", "B) 3 个", "C) 4 个", "D) 5 个"], "answer": "C"},
        {"question": "人体最大的器官是：", "choices": ["A) 肝脏", "B) 皮肤", "C) 大脑", "D) 心脏"], "answer": "B"},
        {"question": "人体中最小的骨骼位于：", "choices": ["A) 手指", "B) 耳朵", "C) 鼻子", "D) 脚趾"], "answer": "B"},
        {"question": "人体有多少对肋骨？", "choices": ["A) 10 对", "B) 12 对", "C) 14 对", "D) 8 对"], "answer": "B"},
    ],
    "clinical_knowledge": [
        {"question": "人体正常血压范围（收缩压/舒张压）是：", "choices": ["A) 90/60 mmHg", "B) 120/80 mmHg", "C) 140/90 mmHg", "D) 160/100 mmHg"], "answer": "B"},
        {"question": "糖尿病的典型症状不包括：", "choices": ["A) 多饮", "B) 多尿", "C) 多食", "D) 多睡"], "answer": "D"},
        {"question": "以下哪种抗生素用于治疗细菌感染？", "choices": ["A) 阿莫西林", "B) 利巴韦林", "C) 奥司他韦", "D) 氟康唑"], "answer": "A"},
        {"question": "人体正常体温（腋下）约为：", "choices": ["A) 35.0°C", "B) 36.5°C", "C) 37.5°C", "D) 38.0°C"], "answer": "B"},
        {"question": "以下哪项不是高血压的危险因素？", "choices": ["A) 高盐饮食", "B) 肥胖", "C) 规律运动", "D) 吸烟"], "answer": "C"},
    ],

    # === 社科类 ===
    "high_school_government_and_politics": [
        {"question": "中国特色社会主义进入新时代的主要矛盾是：", "choices": ["A) 人民日益增长的美好生活需要和不平衡不充分的发展之间的矛盾", "B) 人民日益增长的物质文化需要同落后的社会生产之间的矛盾", "C) 生产力与生产关系的矛盾", "D) 经济基础与上层建筑的矛盾"], "answer": "A"},
        {"question": "我国的根本政治制度是：", "choices": ["A) 人民代表大会制度", "B) 中国共产党领导的多党合作制度", "C) 民族区域自治制度", "D) 基层群众自治制度"], "answer": "A"},
        {"question": "宪法规定国家的最高权力机关是：", "choices": ["A) 国务院", "B) 全国人民代表大会", "C) 最高人民法院", "D) 中央军委"], "answer": "B"},
        {"question": "社会主义核心价值观在个人层面的要求是：", "choices": ["A) 爱国、敬业、诚信、友善", "B) 富强、民主、文明、和谐", "C) 自由、平等、公正、法治", "D) 爱国、守法、明礼、诚信"], "answer": "A"},
        {"question": "我国的根本大法是：", "choices": ["A) 刑法", "B) 宪法", "C) 民法典", "D) 行政法"], "answer": "B"},
    ],
    "sociology": [
        {"question": "社会学研究的核心对象是：", "choices": ["A) 社会群体和结构", "B) 个体心理", "C) 生物进化", "D) 语言符号"], "answer": "A"},
        {"question": "'社会化' 指的是：", "choices": ["A) 个人学习社会规范和价值的过程", "B) 国有企业改革", "C) 社会福利制度", "D) 社会分层"], "answer": "A"},
        {"question": "以下哪个不是社会分层的主要维度？", "choices": ["A) 收入", "B) 教育", "C) 身高", "D) 职业"], "answer": "C"},
        {"question": "核心家庭通常包括：", "choices": ["A) 夫妻和子女", "B) 夫妻和双方父母", "C) 三代同堂", "D) 单身"], "answer": "A"},
        {"question": "文化堕距理论由哪位社会学家提出？", "choices": ["A) 涂尔干", "B) 奥格本", "C) 韦伯", "D) 马克思"], "answer": "B"},
    ],

    # === 人文类 ===
    "philosophy": [
        {"question": "我思故我在（Cogito ergo sum）是谁提出的？", "choices": ["A) 柏拉图", "B) 笛卡尔", "C) 康德", "D) 尼采"], "answer": "B"},
        {"question": "康德的三大批判不包括以下哪一本？", "choices": ["A) 纯粹理性批判", "B) 实践理性批判", "C) 判断力批判", "D) 存在与虚无"], "answer": "D"},
        {"question": "柏拉图的理念论认为：", "choices": ["A) 理念是真实世界的原型", "B) 物质是唯一真实的", "C) 知识来源于经验", "D) 上帝已死"], "answer": "A"},
        {"question": "存在主义的核心观点是：", "choices": ["A) 存在先于本质", "B) 本质先于存在", "C) 物质决定意识", "D) 绝对精神"], "answer": "A"},
        {"question": "功利主义的主要代表人物是：", "choices": ["A) 边沁和密尔", "B) 康德和黑格尔", "C) 萨特和加缪", "D) 罗尔斯和诺齐克"], "answer": "A"},
    ],
    "formal_logic": [
        {"question": "如果 p→q 为真，且 p 为真，则可以推出：", "choices": ["A) q 为真", "B) q 为假", "C) q 不确定", "D) ¬q 为真"], "answer": "A"},
        {"question": "以下哪个是有效的论证形式？", "choices": ["A) 肯定前件（Modus Ponens）", "B) 否定前件", "C) 肯定后件", "D) 否定后件"], "answer": "A"},
        {"question": "p ∧ q 为真当且仅当：", "choices": ["A) p 和 q 都为真", "B) p 或 q 为真", "C) p 和 q 都为假", "D) p 真 q 假"], "answer": "A"},
        {"question": "合取范式（CNF）是：", "choices": ["A) 子句的合取", "B) 子句的析取", "C) 文字的合取", "D) 文字的析取"], "answer": "A"},
        {"question": "量词 ∀xP(x) 的否定是：", "choices": ["A) ∃x¬P(x)", "B) ∀x¬P(x)", "C) ¬∃xP(x)", "D) ∃xP(¬x)"], "answer": "A"},
    ],
    "world_religions": [
        {"question": "佛教的创始人是谁？", "choices": ["A) 释迦牟尼", "B) 老子", "C) 孔子", "D) 穆罕默德"], "answer": "A"},
        {"question": "伊斯兰教的经典是：", "choices": ["A) 圣经", "B) 古兰经", "C) 吠陀", "D) 道德经"], "answer": "B"},
        {"question": "基督教的核心教义是：", "choices": ["A) 三位一体", "B) 轮回转世", "C) 无为而治", "D) 四谛"], "answer": "A"},
        {"question": "道教的创始人是：", "choices": ["A) 老子", "B) 庄子", "C) 张道陵", "D) 孔子"], "answer": "C"},
        {"question": "犹太教的圣地位于：", "choices": ["A) 耶路撒冷", "B) 麦加", "C) 瓦拉纳西", "D) 拉萨"], "answer": "A"},
    ],

    # === 交叉类 ===
    "astronomy": [
        {"question": "太阳系中最大的行星是：", "choices": ["A) 土星", "B) 木星", "C) 天王星", "D) 海王星"], "answer": "B"},
        {"question": "银河系属于哪种星系？", "choices": ["A) 旋涡星系", "B) 椭圆星系", "C) 不规则星系", "D) 透镜星系"], "answer": "A"},
        {"question": "光年是什么单位？", "choices": ["A) 距离单位", "B) 时间单位", "C) 速度单位", "D) 光度单位"], "answer": "A"},
        {"question": "红移现象说明了什么？", "choices": ["A) 宇宙在膨胀", "B) 宇宙在收缩", "C) 恒星在变亮", "D) 银河系在旋转"], "answer": "A"},
        {"question": "月球的公转周期约为：", "choices": ["A) 7 天", "B) 15 天", "C) 27.3 天", "D) 365 天"], "answer": "C"},
    ],
}

# 学科 → 大类的映射
SUBJECT_CATEGORY = {
    # 理工
    "abstract_algebra": "理工", "college_physics": "理工",
    "computer_security": "理工", "machine_learning": "理工",
    "college_computer_science": "理工", "college_mathematics": "理工",
    "conceptual_physics": "理工", "electrical_engineering": "理工",
    "elementary_mathematics": "理工", "high_school_computer_science": "理工",
    "high_school_mathematics": "理工", "high_school_physics": "理工",
    "high_school_statistics": "理工", "college_chemistry": "理工",
    # 生物医学
    "anatomy": "生物医学", "clinical_knowledge": "生物医学",
    "college_biology": "生物医学", "college_medicine": "生物医学",
    "human_aging": "生物医学", "human_sexuality": "生物医学",
    "medical_genetics": "生物医学", "nutrition": "生物医学",
    "professional_medicine": "生物医学", "professional_psychology": "生物医学",
    "virology": "生物医学", "high_school_biology": "生物医学",
    # 社科
    "sociology": "社科", "high_school_government_and_politics": "社科",
    "business_ethics": "社科", "econometrics": "社科",
    "high_school_geography": "社科", "high_school_macroeconomics": "社科",
    "high_school_microeconomics": "社科", "high_school_psychology": "社科",
    "management": "社科", "marketing": "社科",
    "public_relations": "社科", "security_studies": "社科",
    "us_foreign_policy": "社科",
    # 人文
    "philosophy": "人文", "formal_logic": "人文",
    "global_facts": "人文", "high_school_european_history": "人文",
    "high_school_us_history": "人文", "high_school_world_history": "人文",
    "international_law": "人文", "jurisprudence": "人文",
    "logical_fallacies": "人文", "moral_disputes": "人文",
    "moral_scenarios": "人文", "prehistory": "人文",
    "professional_accounting": "人文", "professional_law": "人文",
    "world_religions": "人文",
    # 交叉
    "astronomy": "交叉科学", "miscellaneous": "交叉科学",
}


class MMLUBenchmark(BaseLLMBenchmark):
    """MMLU 评测实现"""

    benchmark_id = "mmlu"
    benchmark_name = "MMLU"
    category = "知识储备"

    def __init__(self):
        super().__init__()
        self._questions: list[dict] = []
        self._data_path = Path(self.config.results_dir).parent.parent / "datasets" / "llm" / "mmlu"

    def load_questions(self) -> list[dict]:
        """加载 MMLU 题目（优先从文件加载，回退到内置数据 75 题）"""
        questions = []

        # 尝试从数据集文件加载
        if self._data_path.exists():
            questions = self._load_from_files()

        # 回退到内置样例数据
        if not questions:
            logger.info("未找到数据集文件，使用内置评测数据（15 学科 × 5 题 = 75 题）")
            questions = self._load_builtin()

        logger.info(f"MMLU 共加载 {len(questions)} 道题目")
        return questions

    def _load_from_files(self) -> list[dict]:
        """从数据集目录加载题目 JSON"""
        questions = []
        try:
            for subj_file in sorted(self._data_path.glob("*.json")):
                with open(subj_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                subject = subj_file.stem
                for item in data:
                    item["category"] = SUBJECT_CATEGORY.get(subject, subject)
                    item["subject"] = subject
                    item["id"] = f"{subject}_{len(questions)}"
                    questions.append(item)
        except Exception as e:
            logger.warning(f"加载数据集文件失败: {e}")
            return []
        return questions

    def _load_builtin(self) -> list[dict]:
        """加载内置样例题目（75 题）"""
        questions = []
        for subject in sorted(BUILTIN_QUESTIONS.keys()):
            for q in BUILTIN_QUESTIONS[subject]:
                questions.append({
                    "id": f"{subject}_{len(questions)}",
                    "question": q["question"],
                    "choices": q["choices"],
                    "answer": q["answer"],
                    "category": SUBJECT_CATEGORY.get(subject, "其他"),
                    "subject": subject,
                })
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建 MMLU 提示词（中文 prompt）"""
        messages = []
        subject = question.get("subject", "")
        subject_cn = MMLU_SUBJECTS_CN.get(subject, subject)

        if few_shot:
            messages.append({
                "role": "system",
                "content": f"你是 MMLU（大规模多任务语言理解）{subject_cn}领域的答题专家。请回答以下多选题，只输出选项字母（A/B/C/D）。"
            })

            # 添加 few-shot 样例
            examples = BUILTIN_FEW_SHOT.get(subject, [])
            for ex in examples[:3]:
                choices_str = "\n".join(ex["choices"])
                messages.append({
                    "role": "user",
                    "content": f"题目：{ex['question']}\n{choices_str}"
                })
                messages.append({
                    "role": "assistant",
                    "content": ex["answer"]
                })
        else:
            messages.append({
                "role": "system",
                "content": "你是 MMLU 答题专家。请直接输出选项字母（A/B/C/D）。"
            })

        # 添加当前题目
        choices = question.get("choices", [])
        choices_str = "\n".join(choices)
        content = f"题目：{question['question']}\n{choices_str}\n请只输出正确答案的选项字母。"
        messages.append({"role": "user", "content": content})

        return messages

    def extract_answer(self, text: str) -> str:
        """从模型输出中提取答案字母 (A/B/C/D)"""
        text = text.strip().upper()

        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1)

        match = re.search(r'(?:答案[是为]?|answer[:\s]*)\s*([A-D])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        if text in ["A", "B", "C", "D"]:
            return text

        for c in text:
            if c in "ABCD":
                return c

        return text[:1] if text else ""

    def check_answer(self, extracted: str, reference: str) -> bool:
        """判断答案是否正确"""
        return extracted.strip().upper() == reference.strip().upper()


benchmark = MMLUBenchmark()
