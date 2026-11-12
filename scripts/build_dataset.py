# -*- coding: utf-8 -*-
"""总装：手工精选核心 + 模板库程序化扩写 -> 去重 -> 分类写入文件夹 -> 生成索引。
目标总量 5000：情绪识别2500 + 共情对话单轮1500 + 共情对话多轮750 + 语气词专项250。"""
import json, os, sys, random, collections, ast, re, hashlib

BASE = os.path.dirname(os.path.abspath(__file__))          # v2/
PARENT = os.path.dirname(BASE)                              # 情感AI训练数据/
sys.path.insert(0, BASE); sys.path.insert(0, PARENT)
from banks_recognition import SLOTS, FRAMES
from banks_dialogue import DSLOTS, DIA_THEMES
from tone_bank import TONE_SLOTS, TONE_FRAMES
from tone_contrastive import CONTRASTIVE
import new_recognition, new_dialogues, new_tone_samples

DSLOTS.setdefault("亲近的人", ["爷爷","奶奶","爸爸","妈妈","外公","外婆","一位挚友"])
TAXONOMY = ["喜悦","悲伤","愤怒","恐惧","焦虑","惊讶","厌恶","平静","委屈","孤独","愧疚","期待","感动","疲惫"]
THEMES = list(DIA_THEMES.keys())
SEED = 20261113
rng = random.Random(SEED)
OUT = os.path.join(BASE, "dataset")
REC_TOTAL, SINGLE_TOTAL, MULTI_TOTAL = 2500, 1500, 750
TONE_GEN = {"情绪场合适配":50, "克制与严肃":30, "日常软萌":32}

# ---------- 读取手工精选核心 ----------
def extract_consts(path):
    tree = ast.parse(open(path, encoding="utf-8").read()); out={}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("recognition","dialogues","REC_SYS","DIA_SYS"):
                    out[t.id]=ast.literal_eval(node.value)
    return out
GEN = next(p for p in (os.path.join(BASE,"gen_emotion_data.py"), os.path.join(PARENT,"gen_emotion_data.py")) if os.path.exists(p))
_g = extract_consts(GEN)
base_rec, base_dia = _g["recognition"], _g["dialogues"]
REC_SYS, DIA_SYS = _g["REC_SYS"], _g["DIA_SYS"]

def rec_msgs(text,emo,inten,trig):
    out=json.dumps({"emotion":emo,"intensity":inten,"trigger":trig},ensure_ascii=False)
    return [{"role":"system","content":REC_SYS},{"role":"user","content":text},{"role":"assistant","content":out}]
def dia_msgs(turns):
    return [{"role":"system","content":DIA_SYS}]+[{"role":r,"content":c} for r,c in turns]
def sig(msgs):
    return json.dumps([(m["role"],m["content"]) for m in msgs if m["role"]!="system"],ensure_ascii=False)

# ---------- 槽位填充 ----------
def fill(tpl, slots):
    names=re.findall(r"\{(\w+)\}", tpl)
    chosen={n: rng.choice(slots[n]) for n in set(names)}
    for n,v in chosen.items(): tpl=tpl.replace("{"+n+"}", v)
    return tpl
def fill_pair(t,g,slots):
    names=set(re.findall(r"\{(\w+)\}", t))|set(re.findall(r"\{(\w+)\}", g))
    chosen={n: rng.choice(slots[n]) for n in names}
    for n,v in chosen.items(): t=t.replace("{"+n+"}",v); g=g.replace("{"+n+"}",v)
    return t,g

# ---------- 主题分类(用于手工精选对话) ----------
THEME_KW={
 "哀伤丧失":["走了","去世","离世","回了汪星","遗物","最后一面","思念","失去他","病不太好","确诊"],
 "积极分享":["升职","考过","脱单","学会","康复","中奖","求婚","盈利","上岸","被表扬","开花","结果","驾照","五公里","献血","乔迁","新家","表白成功"],
 "健康担忧":["体检","失眠","手术","麻醉","胸口","抑郁","生病","健康","医生","复查","结节","病"],
 "职场压力":["领导","同事","加班","方案","裁员","辞职","上班","工作","客户","甩锅","背锅","边缘化","产假","岗位","面试","加薪","绩效"],
 "感情婚恋":["分手","失恋","异地","表白","暗恋","催婚","催生","结婚","老公","对象","感情","容貌","胖","瘦","裙子","喜欢他","婚姻"],
 "学业考试":["考试","复习","答辩","考研","论文","导师","成绩","学习","厌学","上台","发言","考公","培训","毕业"],
 "家庭亲子":["爸妈","父母","孩子","育儿","妈妈","爸爸","外婆","奶奶","亲子","家长"],
 "经济压力":["房贷","车贷","工资","开销","负债","月光","经济","欠钱","存不下","账单","盈利"],
 "人际社交":["朋友","闺蜜","兄弟","网暴","社恐","聚会","误解","疏远","插队","背后说","借钱","合群"],
 "孤独迷茫":["孤独","迷茫","一个人","没朋友","空巢","想家","没人懂","三十","一事无成","生日","没人记得","外地"],
 "自我成长":["自我怀疑","不够好","没用","自律","坚持","成长","舒适区","敏感","想太多","自信","方向"],
 "日常陪伴":["早安","晚安","无聊","聊天","晚霞","奶茶","猫","狗","天气","云","泡面","火锅","可爱","蛋糕"],
}
def classify_theme(turns):
    text=" ".join(c for r,c in turns)
    for th in ["哀伤丧失","积极分享"]:
        if any(k in text for k in THEME_KW[th]): return th
    best,bn="日常陪伴",0
    for th,kws in THEME_KW.items():
        n=sum(1 for k in kws if k in text)
        if n>bn: best,bn=th,n
    return best

# ---------- 1) 情绪识别 ----------
rec_by_emo=collections.defaultdict(list); rec_seen=set()
def add_rec(text,emo,inten,trig):
    if text in rec_seen: return False
    rec_seen.add(text); rec_by_emo[emo].append(rec_msgs(text,emo,inten,trig)); return True
for text,emo,inten,trig in base_rec+list(new_recognition.NEW_REC):
    add_rec(text,emo,inten,trig)
# 目标每类数量
per=REC_TOTAL//14; rem=REC_TOTAL%14
targets={e:per+(1 if i<rem else 0) for i,e in enumerate(TAXONOMY)}
for emo in TAXONOMY:
    frames=FRAMES[emo]; need=targets[emo]-len(rec_by_emo[emo]); attempts=0
    while need>0 and attempts<200000:
        attempts+=1
        t,g,ib=rng.choice(frames)
        text,trig=fill_pair(t,g,SLOTS)
        inten=max(1,min(5,ib+rng.choice([-1,0,0,0,1])))
        if add_rec(text,emo,inten,trig): need-=1

# ---------- 2) 共情对话 ----------
single_by_theme=collections.defaultdict(list); multi_by_theme=collections.defaultdict(list)
dseen=set()
def add_dia(turns,theme,multi):
    k=sig(dia_msgs(turns))
    if k in dseen: return False
    dseen.add(k)
    (multi_by_theme if multi else single_by_theme)[theme].append(dia_msgs(turns)); return True
for turns in base_dia+list(new_dialogues.NEW_DIA):
    th=classify_theme(turns); add_dia(turns,th,len(turns)>2)
def gen_dialogue(multi):
    theme=rng.choice(THEMES); sit=rng.choice(DIA_THEMES[theme]["situations"])
    u=fill(rng.choice(sit["u"]),DSLOTS); a=fill(rng.choice(sit["a"]),DSLOTS)
    turns=[("user",u),("assistant",a)]
    if multi:
        fu=fill(rng.choice(sit["fu"]),DSLOTS); fa=fill(rng.choice(sit["fa"]),DSLOTS)
        turns+=[("user",fu),("assistant",fa)]
    return turns,theme
def fill_dialogue(target,by_theme,multi):
    total=sum(len(v) for v in by_theme.values()); attempts=0
    while total<target and attempts<400000:
        attempts+=1
        turns,theme=gen_dialogue(multi)
        if add_dia(turns,theme,multi): total+=1
fill_dialogue(SINGLE_TOTAL,single_by_theme,False)
fill_dialogue(MULTI_TOTAL,multi_by_theme,True)

# ---------- 3) 语气词专项（纳入全局对话去重 dseen）----------
tone_contrastive=[]
for c in CONTRASTIVE:
    m=dia_msgs([("user",c["u"]),("assistant",c["a"])]); k=sig(m)
    if k not in dseen: dseen.add(k); tone_contrastive.append(m)
tone_seed=[]
for t in new_tone_samples.NEW_TONE_DIA:
    m=dia_msgs(t); k=sig(m)
    if k not in dseen: dseen.add(k); tone_seed.append(m)
tone_slots={**DSLOTS,**TONE_SLOTS}
tone_gen={k:[] for k in TONE_GEN}
for sub,target in TONE_GEN.items():
    frames=TONE_FRAMES[sub]; attempts=0
    while len(tone_gen[sub])<target and attempts<100000:
        attempts+=1
        u,a=rng.choice(frames); u=fill(u,tone_slots); a=fill(a,tone_slots)
        m=dia_msgs([("user",u),("assistant",a)]); k=sig(m)
        if k in dseen: continue
        dseen.add(k); tone_gen[sub].append(m)
# 全局去重后补足到 5000（用单轮对话补齐）
cur=(sum(len(v) for v in rec_by_emo.values())+sum(len(v) for v in single_by_theme.values())
     +sum(len(v) for v in multi_by_theme.values())+len(tone_contrastive)+len(tone_seed)
     +sum(len(v) for v in tone_gen.values()))
deficit=5000-cur
if deficit>0:
    fill_dialogue(SINGLE_TOTAL+deficit, single_by_theme, False)

# ---------- 写文件 ----------
def write_jsonl(relpath,msgs_list):
    p=os.path.join(OUT,relpath); os.makedirs(os.path.dirname(p),exist_ok=True)
    body="\n".join(json.dumps({"messages":m},ensure_ascii=False) for m in msgs_list)+"\n"
    tmp=p+".tmp"; open(tmp,"w",encoding="utf-8").write(body); os.replace(tmp,p)
    return len(msgs_list)
manifest=[]; total=0
for emo in TAXONOMY:
    n=write_jsonl(f"data/01_情绪识别/{emo}.jsonl",rec_by_emo[emo]); total+=n
    manifest.append({"path":f"data/01_情绪识别/{emo}.jsonl","类别":"情绪识别","情绪":emo,"条数":n})
for th in THEMES:
    n=write_jsonl(f"data/02_共情对话_单轮/{th}.jsonl",single_by_theme[th]); total+=n
    manifest.append({"path":f"data/02_共情对话_单轮/{th}.jsonl","类别":"共情对话·单轮","主题":th,"条数":n})
for th in THEMES:
    n=write_jsonl(f"data/03_共情对话_多轮/{th}.jsonl",multi_by_theme[th]); total+=n
    manifest.append({"path":f"data/03_共情对话_多轮/{th}.jsonl","类别":"共情对话·多轮","主题":th,"条数":n})
tone_files=[("近音词对照",tone_contrastive),("精选语气词样本",tone_seed),
            ("情绪场合适配",tone_gen["情绪场合适配"]),("克制与严肃",tone_gen["克制与严肃"]),("日常软萌",tone_gen["日常软萌"])]
for name,lst in tone_files:
    n=write_jsonl(f"data/04_语气词专项/{name}.jsonl",lst); total+=n
    manifest.append({"path":f"data/04_语气词专项/{name}.jsonl","类别":"语气词专项","子类":name,"条数":n})

# ---------- 索引 ----------
rec_n=sum(len(v) for v in rec_by_emo.values()); s_n=sum(len(v) for v in single_by_theme.values()); m_n=sum(len(v) for v in multi_by_theme.values())
tone_n=len(tone_contrastive)+len(tone_seed)+sum(len(v) for v in tone_gen.values())
def tbl(items,key):
    return "\n".join(f"| {it[key]} | {it['条数']} | `{it['path']}` |" for it in items)
rec_items=[m for m in manifest if m["类别"]=="情绪识别"]
s_items=[m for m in manifest if m["类别"]=="共情对话·单轮"]
m_items=[m for m in manifest if m["类别"]=="共情对话·多轮"]
t_items=[m for m in manifest if m["类别"]=="语气词专项"]
readme=f"""# 情感AI混合训练集 · 总索引

面向「情感AI大模型」的 SFT 训练集，**总量 {total} 条**，按任务与场景分门别类存放于 `data/`，
每行一个标准 JSONL（仅含 `messages` 字段），可直接用于 LLaMA-Factory / ms-swift / Axolotl / OpenAI fine-tuning。

## 一、目录结构
```
data/
├── 01_情绪识别/          # 文本→情绪JSON标注，按14类情绪分文件（{rec_n}条）
├── 02_共情对话_单轮/     # 单轮共情回应，按12主题分文件（{s_n}条）
├── 03_共情对话_多轮/     # 多轮共情陪伴，按12主题分文件（{m_n}条）
└── 04_语气词专项/        # 语气词近音辨析/情绪适配/克制/软萌（{tone_n}条）
docs/                     # 语气词使用规范、数据说明
scripts/                  # 生成与合并脚本（可复现/继续扩充）
manifest.json             # 机器可读索引（文件→类别/条数）
```

## 二、构成概览
| 类别 | 条数 |
|---|---|
| 情绪识别 | {rec_n} |
| 共情对话·单轮 | {s_n} |
| 共情对话·多轮 | {m_n} |
| 语气词专项 | {tone_n} |
| **合计** | **{total}** |

## 三、情绪识别（{rec_n}）
标签体系14类；输出 `{{"emotion","intensity"(1-5),"trigger"}}`。
| 情绪 | 条数 | 文件 |
|---|---|---|
{tbl(rec_items,"情绪")}

## 四、共情对话·单轮（{s_n}）
| 主题 | 条数 | 文件 |
|---|---|---|
{tbl(s_items,"主题")}

## 五、共情对话·多轮（{m_n}）
| 主题 | 条数 | 文件 |
|---|---|---|
{tbl(m_items,"主题")}

## 六、语气词专项（{tone_n}）
遵循《docs/语气词使用规范.md》：近音词按情绪选对、自然点缀不必每句都加、严肃/哀伤/危机场景保持稳重。
| 子类 | 条数 | 文件 |
|---|---|---|
{tbl(t_items,"子类")}

## 七、数据构成与质量说明
- **手工精选核心**：首批及历次手工编写样本（情绪识别152、共情对话100、语气词56）+ 语气词近音对照{len(tone_contrastive)}条，质量基准。
- **程序化扩写**：其余由富模板库（多框架×大槽位）组合扩写，固定随机种子（seed={SEED}）生成并去重，用于扩充规模与场景覆盖。
- 两类任务以不同 system 提示词区分；共情 system 已注入语气词总原则。
- 建议训练前人工抽检共情质量与安全回应；可按需对某类别加权采样或继续扩充。

## 八、复现与扩充
- 复现：运行 `scripts/build_dataset.py`（依赖 `scripts/` 下模板库与精选样本模块）。
- 扩充：在对应模板库（banks_recognition / banks_dialogue / tone_bank）追加框架或槽位，重跑即可；自动去重、分类、重建索引。
"""
tmp=os.path.join(OUT,"README.md.tmp"); open(tmp,"w",encoding="utf-8").write(readme); os.replace(tmp,os.path.join(OUT,"README.md"))
json.dump({"总量":total,"seed":SEED,"文件":manifest},open(os.path.join(OUT,"manifest.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

print("总量:",total,"| 识别:",rec_n,"| 单轮:",s_n,"| 多轮:",m_n,"| 语气词:",tone_n)
print("情绪识别每类:",{e:len(rec_by_emo[e]) for e in TAXONOMY})
print("单轮主题:",{t:len(single_by_theme[t]) for t in THEMES})
print("多轮主题:",{t:len(multi_by_theme[t]) for t in THEMES})
print("语气词:",{k:len(v) for k,v in tone_gen.items()},"+对照",len(tone_contrastive),"+精选",len(tone_seed))
print("输出目录:",OUT)