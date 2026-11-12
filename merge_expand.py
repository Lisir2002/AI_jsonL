# -*- coding: utf-8 -*-
"""合并首批与新增样本 -> 去重 -> 打乱 -> 重写 JSONL + 更新数据说明。"""
import json, os, sys, random, collections, hashlib

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from new_recognition import NEW_REC
from new_dialogues import NEW_DIA
from new_tone_samples import NEW_TONE_DIA

TAXONOMY = ["喜悦","悲伤","愤怒","恐惧","焦虑","惊讶","厌恶","平静","委屈","孤独","愧疚","期待","感动","疲惫"]
JSONL = os.path.join(BASE, "情感AI混合训练集.jsonl")
README = os.path.join(BASE, "数据说明.md")
SEED = 2026

def safe_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    os.sync()
    back = open(path, encoding="utf-8").read()
    assert back == text, "write-back mismatch: " + path
    return hashlib.md5(back.encode()).hexdigest()

# 1) 读首批，抽取两类 system 提示词（保证与既有一致）
existing = [json.loads(l) for l in open(JSONL, encoding="utf-8") if l.strip()]
REC_SYS = next(m["messages"][0]["content"] for m in existing if "情感分析专家" in m["messages"][0]["content"])
DIA_SYS = next(m["messages"][0]["content"] for m in existing if "情感陪伴助手" in m["messages"][0]["content"])

# 2) 构造新增样本并校验
new_samples = []
for text, emo, inten, trig in NEW_REC:
    assert emo in TAXONOMY and 1 <= inten <= 5, (text, emo, inten)
    out = json.dumps({"emotion": emo, "intensity": inten, "trigger": trig}, ensure_ascii=False)
    new_samples.append([{"role":"system","content":REC_SYS},{"role":"user","content":text},{"role":"assistant","content":out}])
for turns in NEW_DIA + NEW_TONE_DIA:
    assert turns[0][0]=="user" and turns[-1][0]=="assistant"
    for i in range(len(turns)-1):
        assert turns[i][0]!=turns[i+1][0]
    new_samples.append([{"role":"system","content":DIA_SYS}]+[{"role":r,"content":c} for r,c in turns])

# 3) 去重（按非 system 消息内容签名），保留首批在前
def sig(msgs):
    return json.dumps([(m["role"],m["content"]) for m in msgs if m["role"]!="system"], ensure_ascii=False)
seen, combined, dup = set(), [], 0
for m in existing:
    k = sig(m["messages"])
    if k in seen: dup += 1; continue
    seen.add(k); combined.append(m["messages"])
for msgs in new_samples:
    k = sig(msgs)
    if k in seen: dup += 1; continue
    seen.add(k); combined.append(msgs)

# 4) 打乱
random.Random(SEED).shuffle(combined)

# 5) 写 JSONL
body = "\n".join(json.dumps({"messages":ms}, ensure_ascii=False) for ms in combined) + "\n"
md5_jsonl = safe_write(JSONL, body)

# 6) 统计（从最终样本独立计算）
n_rec=n_dia=single=multi=0
label_dist=collections.Counter(); inten_dist=collections.Counter()
for ms in combined:
    sysc = ms[0]["content"]
    if "情感分析专家" in sysc:
        n_rec+=1
        a=json.loads(ms[-1]["content"]); label_dist[a["emotion"]]+=1; inten_dist[a["intensity"]]+=1
    else:
        n_dia+=1
        uturns=sum(1 for m in ms if m["role"]=="user")
        if uturns==1: single+=1
        else: multi+=1

# 7) 校验最终文件
lines=open(JSONL,encoding="utf-8").read().splitlines()
assert len(lines)==len(combined)
for l in lines:
    o=json.loads(l); assert set(o)=={"messages"} and o["messages"][0]["role"]=="system"

label_rows="\n".join(f"| {k} | {label_dist.get(k,0)} |" for k in TAXONOMY)
inten_rows="\n".join(f"| {i} | {inten_dist.get(i,0)} |" for i in range(1,6))
readme=f"""# 情感AI混合训练集 · 数据说明

## 一、概述
面向「情感AI大模型」的混合 SFT（监督微调）训练集，覆盖两类能力：
1. **情绪识别/标注**：输入一段文本 → 输出结构化情绪分析（JSON）。
2. **共情对话/情感陪伴**：输入用户的情绪倾诉 → 输出温暖、共情的回应（含单轮与多轮）。

两类样本通过**不同的 system 提示词**区分任务，模型可据 system 学会任务条件化行为。
本版本已在首批 80 条基础上**扩充**，新增样本覆盖更多场景与表达风格，并去重、重新打乱。

## 二、规模与构成
- 总样本数：**{len(combined)} 条**（首批 80 + 新增 {len(new_samples)}，去重剔除 {dup} 条）
- 情绪识别：**{n_rec} 条**
- 共情对话：**{n_dia} 条**（单轮 {single} 条，多轮 {multi} 条，多轮为 2–3 个用户回合）
- 已用固定随机种子（seed={SEED}）打乱，两类任务交错分布，可直接用于训练。

## 三、文件格式
标准 JSONL，每行一个 JSON 对象，仅含 `messages` 字段（最大化兼容各类微调框架）：
```json
{{"messages": [
  {{"role": "system", "content": "任务/角色设定"}},
  {{"role": "user", "content": "用户输入"}},
  {{"role": "assistant", "content": "目标输出"}}
]}}
```
多轮对话在 user / assistant 之间交替，最后一条为 assistant。

## 四、任务一：情绪识别
- **标签体系（14 类）**：喜悦、悲伤、愤怒、恐惧、焦虑、惊讶、厌恶、平静、委屈、孤独、愧疚、期待、感动、疲惫。
- **输出格式**（assistant 内容为 JSON 字符串）：
  `{{"emotion": "标签", "intensity": 1-5, "trigger": "简短触发原因"}}`
- 覆盖不同强度（1–5）、口语/书面/反讽/含蓄/网络用语等多种表达。

### 情绪标签分布
| 标签 | 条数 |
|---|---|
{label_rows}

### 强度分布
| 强度 | 条数 |
|---|---|
{inten_rows}

## 五、任务二：共情对话
- **回应原则**：先接纳并确认感受 → 再温和陪伴/启发/引导；不说教、不评判、不否定感受、不急于给方案；尊重对方节奏。
- **场景覆盖**：职场（裁员/背锅/骚扰/产后边缘化）、失恋/异地恋、学业/考试、容貌焦虑、网暴、失眠、孤独/空巢、亲子/育儿分歧、催生/催婚、经济压力、中年危机、哀伤（亲人/宠物离世）、照护耗竭、离婚重建、迷茫，以及**积极情绪分享**（上岸/求婚/康复/创业好转/小确幸等）。
- **安全示范**：包含心理危机、职场骚扰边界、照护者耗竭、亲人确诊等场景的**安全范式回应**——表达关心与在意、鼓励联系信任的人与专业心理/医疗/法律支持、不做诊断、不评判、不鼓励任何伤害行为、不受害者有罪论。可用于训练模型的安全边界。
- **语气词/口语化风格（专项）**：见 `new_tone_samples.py` 与《语气词使用规范.md》。语气词按情绪与场合选择、自然点缀、不必每句都加；覆盖近音词辨析（哦/噢/喔、叭/吧、呀/啊、呢/呐、啦/咯、嘛/吗、嗯系、呗/诶），跨情绪适配，并含严肃/哀伤/危机场景的克制示范（不加俏皮语气词）与『有的句子带、有的不带』的自然节奏。

## 六、使用建议
- 兼容 LLaMA-Factory、ms-swift、Axolotl、OpenAI fine-tuning 等主流 SFT 流程（messages 格式）。
- 如需按任务拆分/加权采样，可依据 system 提示词内容过滤（情绪识别 vs 情感陪伴）。
- 建议训练前人工抽检共情质量与安全回应；可按同一范式继续扩充到更大规模。

## 七、已知边界
- 情绪识别采用**单一主情绪**标注（真实文本可能含混合情绪），强度为 1–5 主观量表。
- 危机/安全场景回应中未写死具体热线号码，以避免信息过时；落地时可按部署地区补充官方心理援助渠道。

## 八、复现与扩充
- **复现**：先运行 `gen_emotion_data.py` 生成首批 80 条，再运行 `merge_expand.py` 合并 `new_recognition.py` / `new_dialogues.py` / `new_tone_samples.py` 中的新增样本，得到本数据集（脚本输出的文档名为 `数据说明.md`，内容即本 README）。
- **扩充**：在 `new_recognition.py` / `new_dialogues.py` / `new_tone_samples.py` 中按相同结构追加样本，重跑 `merge_expand.py` 即可（自动去重、打乱、重算分布）。
"""
md5_readme = safe_write(README, readme)

print("总样本:", len(combined), "| 识别:", n_rec, "| 对话:", n_dia, "(单轮", single, "多轮", multi, ")")
print("新增:", len(new_samples), "| 去重剔除:", dup)
print("标签分布:", dict(label_dist))
print("强度分布:", dict(sorted(inten_dist.items())))
print("JSONL md5:", md5_jsonl, "| README md5:", md5_readme)
print("校验通过：行数一致、每行合法JSON、仅含messages")