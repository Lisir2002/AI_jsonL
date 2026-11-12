# -*- coding: utf-8 -*-
"""总装（手写优先版）：逐条手写样本优先入类，模板仅补各类别缺口。
手写与模板分存于各类别下 手写/ 与 模板/ 子文件夹，便于逐条审核与只加载手写训练。
目标：情绪识别2500 + 共情对话单轮1500 + 多轮750 + 语气词249 = 5000，逐步全部替换为手写。"""
import json, os, sys, random, collections, ast, re, glob, importlib.util

BASE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(BASE)
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
OUT = os.environ.get("DATASET_OUT") or (os.path.dirname(BASE) if os.path.basename(BASE)=="scripts" else os.path.join(BASE, "dataset"))
REC_TOTAL, SINGLE_TOTAL, MULTI_TOTAL, TONE_TOTAL = 2500, 1501, 750, 249

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

THEME_KW={
 "哀伤丧失":["走了","去世","离世","回了汪星","遗物","最后一面","思念","失去他","病不太好","确诊"],
 "积极分享":["升职","考过","脱单","学会","康复","中奖","求婚","盈利","上岸","被表扬","开花","结果","驾照","五公里","献血","乔迁","新家","表白成功"],
 "健康担忧":["体检","失眠","手术","麻醉","胸口","抑郁","生病","健康","医生","复查","结节","病"],
 "职场压力":["领导","同事","加班","方案","裁员","辞职","上班","工作","客户","甩锅","背锅","边缘化","产假","岗位","面试","加薪","绩效","PUA","画大饼","返聘"],
 "感情婚恋":["分手","失恋","异地","表白","暗恋","催婚","催生","结婚","老公","对象","感情","容貌","胖","瘦","裙子","喜欢他","婚姻","恐婚","见家长","相亲","婚礼","已读不回","伴娘"],
 "学业考试":["考试","复习","答辩","考研","论文","导师","成绩","学习","厌学","上台","发言","考公","培训","毕业","高考"],
 "家庭亲子":["爸妈","父母","孩子","育儿","妈妈","爸爸","外婆","奶奶","亲子","家长","婆媳","辅导作业","叛逆","二胎","哥哥"],
 "经济压力":["房贷","车贷","工资","开销","负债","月光","经济","欠钱","存不下","账单","盈利","装修","副业"],
 "人际社交":["朋友","闺蜜","兄弟","网暴","社恐","聚会","误解","疏远","插队","背后说","借钱","合群","室友","合租","邻居"],
 "孤独迷茫":["孤独","迷茫","一个人","没朋友","空巢","想家","没人懂","三十","一事无成","生日","没人记得","外地","独居","跨年"],
 "自我成长":["自我怀疑","不够好","没用","自律","坚持","成长","舒适区","敏感","想太多","自信","方向","拖延","学历","口音","冒充","脱发","暴食","反弹"],
 "日常陪伴":["早安","晚安","无聊","聊天","晚霞","奶茶","猫","狗","天气","云","泡面","火锅","可爱","蛋糕","多肉","搬家","认床","末班车"],
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

# ---------- 收集全部手写存量（种子 + 对照 + 所有手写批次）----------
hw_rec = list(base_rec) + list(new_recognition.NEW_REC)
hw_dia = [list(t) for t in base_dia] + [list(t) for t in new_dialogues.NEW_DIA]
hw_tone_seed = [list(t) for t in new_tone_samples.NEW_TONE_DIA]
hw_tone_batch = []
batch_files = sorted(glob.glob(os.path.join(BASE,"handwritten_batch*.py")))
for f in batch_files:
    spec=importlib.util.spec_from_file_location("hw_"+os.path.basename(f)[:-3], f)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    hw_rec += list(getattr(mod,"HAND_REC",[]))
    hw_dia += [list(t) for t in getattr(mod,"HAND_DIA",[])]
    hw_tone_batch += [list(t) for t in getattr(mod,"HAND_TONE",[])]

# ---------- 情绪识别：手写优先 ----------
rec_hw=collections.defaultdict(list); rec_tpl=collections.defaultdict(list); rec_seen=set()
for text,emo,inten,trig in hw_rec:
    if text in rec_seen: continue
    rec_seen.add(text); rec_hw[emo].append(rec_msgs(text,emo,inten,trig))
rec_hw_n=sum(len(v) for v in rec_hw.values())
per=REC_TOTAL//14; rem=REC_TOTAL%14
targets={e:per+(1 if i<rem else 0) for i,e in enumerate(TAXONOMY)}
for emo in TAXONOMY:
    need=targets[emo]-len(rec_hw[emo]); frames=FRAMES[emo]; attempts=0
    while need>0 and attempts<200000:
        attempts+=1
        t,g,ib=rng.choice(frames); text,trig=fill_pair(t,g,SLOTS)
        inten=max(1,min(5,ib+rng.choice([-1,0,0,0,1])))
        if text in rec_seen: continue
        rec_seen.add(text); rec_tpl[emo].append(rec_msgs(text,emo,inten,trig)); need-=1

# ---------- 共情对话：手写优先 ----------
s_hw=collections.defaultdict(list); m_hw=collections.defaultdict(list)
s_tpl=collections.defaultdict(list); m_tpl=collections.defaultdict(list); dseen=set()
def add_hw_dia(turns):
    m=dia_msgs(turns); k=sig(m)
    if k in dseen: return
    dseen.add(k); th=classify_theme(turns)
    (m_hw if len(turns)>2 else s_hw)[th].append(m)
for turns in hw_dia: add_hw_dia(turns)
s_hw_n=sum(len(v) for v in s_hw.values()); m_hw_n=sum(len(v) for v in m_hw.values())
def gen_turns(multi):
    theme=rng.choice(THEMES); sit=rng.choice(DIA_THEMES[theme]["situations"])
    u=fill(rng.choice(sit["u"]),DSLOTS); a=fill(rng.choice(sit["a"]),DSLOTS)
    turns=[("user",u),("assistant",a)]
    if multi:
        turns+=[("user",fill(rng.choice(sit["fu"]),DSLOTS)),("assistant",fill(rng.choice(sit["fa"]),DSLOTS))]
    return turns,theme
def fill_tpl(target,hw_map,tpl_map,multi):
    total=sum(len(v) for v in hw_map.values()); attempts=0
    while total<target and attempts<400000:
        attempts+=1
        turns,theme=gen_turns(multi); m=dia_msgs(turns); k=sig(m)
        if k in dseen: continue
        dseen.add(k); tpl_map[theme].append(m); total+=1
fill_tpl(SINGLE_TOTAL,s_hw,s_tpl,False)
fill_tpl(MULTI_TOTAL,m_hw,m_tpl,True)

# ---------- 语气词：手写优先 ----------
tone_contrastive=[]; tone_other=[]
for c in CONTRASTIVE:
    m=dia_msgs([("user",c["u"]),("assistant",c["a"])]); k=sig(m)
    if k not in dseen: dseen.add(k); tone_contrastive.append(m)
for turns in hw_tone_seed+hw_tone_batch:
    m=dia_msgs(turns); k=sig(m)
    if k not in dseen: dseen.add(k); tone_other.append(m)
tone_hw_n=len(tone_contrastive)+len(tone_other)
tone_slots={**DSLOTS,**TONE_SLOTS}; tone_tpl=[]; attempts=0
while tone_hw_n+len(tone_tpl)<TONE_TOTAL and attempts<100000:
    attempts+=1
    sub=rng.choice(list(TONE_FRAMES)); u,a=rng.choice(TONE_FRAMES[sub])
    u=fill(u,tone_slots); a=fill(a,tone_slots); m=dia_msgs([("user",u),("assistant",a)]); k=sig(m)
    if k in dseen: continue
    dseen.add(k); tone_tpl.append(m)

# ---------- 写文件（手写/模板 分开）----------
def write_jsonl(relpath,lst):
    p=os.path.join(OUT,relpath); os.makedirs(os.path.dirname(p),exist_ok=True)
    body="\n".join(json.dumps({"messages":m},ensure_ascii=False) for m in lst)+"\n" if lst else ""
    tmp=p+".tmp"; open(tmp,"w",encoding="utf-8").write(body); os.replace(tmp,p)
    return len(lst)
manifest=[]; total=0
def emit(cat,sub,key,lst,src):
    global total
    n=write_jsonl(f"data/{cat}/{sub}/{key}.jsonl",lst); total+=n
    manifest.append({"path":f"data/{cat}/{sub}/{key}.jsonl","类别":cat,"子类":key,"来源":src,"条数":n})
for emo in TAXONOMY:
    emit("01_情绪识别","手写",emo,rec_hw[emo],"手写"); emit("01_情绪识别","模板",emo,rec_tpl[emo],"模板")
for th in THEMES:
    emit("02_共情对话_单轮","手写",th,s_hw[th],"手写"); emit("02_共情对话_单轮","模板",th,s_tpl[th],"模板")
for th in THEMES:
    emit("03_共情对话_多轮","手写",th,m_hw[th],"手写"); emit("03_共情对话_多轮","模板",th,m_tpl[th],"模板")
emit("04_语气词专项","手写","近音词对照",tone_contrastive,"手写")
emit("04_语气词专项","手写","精选与批次",tone_other,"手写")
emit("04_语气词专项","模板","补充",tone_tpl,"模板")

# ---------- 进度台账 + 索引 ----------
rec_n=sum(len(v) for v in rec_hw.values())+sum(len(v) for v in rec_tpl.values())
s_n=sum(len(v) for v in s_hw.values())+sum(len(v) for v in s_tpl.values())
m_n=sum(len(v) for v in m_hw.values())+sum(len(v) for v in m_tpl.values())
tone_n=tone_hw_n+len(tone_tpl)
rec_tpl_n=sum(len(v) for v in rec_tpl.values()); s_tpl_n=sum(len(v) for v in s_tpl.values()); m_tpl_n=sum(len(v) for v in m_tpl.values())
hw_all=rec_hw_n+s_hw_n+m_hw_n+tone_hw_n; tpl_all=rec_tpl_n+s_tpl_n+m_tpl_n+len(tone_tpl)
def pct(a,b): return f"{a/b*100:.1f}%" if b else "—"
prog=f"""# 手写替换进度台账

目标：把 5000 条全部替换为**逐条手写**（模板仅暂补缺口，逐步清零）。
- 手写批次源文件：`scripts/handwritten_batch*.py`（可逐条审核）；已接入 {len(batch_files)} 个批次。
- 各类别下 `手写/` 为逐条手写、`模板/` 为待替换的模板扩写。只想要手写质量可仅加载各 `手写/` 目录。

| 类别 | 目标 | 已手写 | 模板剩余 | 手写占比 |
|---|---|---|---|---|
| 情绪识别 | {REC_TOTAL} | {rec_hw_n} | {rec_tpl_n} | {pct(rec_hw_n,rec_n)} |
| 共情对话·单轮 | {SINGLE_TOTAL} | {s_hw_n} | {s_tpl_n} | {pct(s_hw_n,s_n)} |
| 共情对话·多轮 | {MULTI_TOTAL} | {m_hw_n} | {m_tpl_n} | {pct(m_hw_n,m_n)} |
| 语气词专项 | {TONE_TOTAL} | {tone_hw_n} | {len(tone_tpl)} | {pct(tone_hw_n,tone_n)} |
| **合计** | **5000** | **{hw_all}** | **{tpl_all}** | **{pct(hw_all,total)}** |

> 剩余约 **{tpl_all}** 条模板待手写替换。手写按批次持续累积，重跑 `build_dataset.py` 自动并入并重算进度。
"""
tmp=os.path.join(OUT,"进度.md.tmp"); open(tmp,"w",encoding="utf-8").write(prog); os.replace(tmp,os.path.join(OUT,"进度.md"))

def tbl(cat,sub,key):
    items=[m for m in manifest if m["类别"]==cat and m["来源"]==sub]
    return "\n".join(f"| {it['子类']} | {it['条数']} | `{it['path']}` |" for it in items)
readme=f"""# 情感AI混合训练集 · 总索引（手写替换进行中）

面向「情感AI大模型」的 SFT 训练集，**总量 {total} 条**，其中**逐条手写 {hw_all} 条**、模板暂补 {tpl_all} 条（逐步替换为手写，见 `进度.md`）。
每行标准 JSONL（仅含 `messages`），兼容 LLaMA-Factory / ms-swift / Axolotl / OpenAI fine-tuning。

## 一、目录结构
```
data/<类别>/手写/   # 逐条手写（质量优先，可单独加载训练/审核）
data/<类别>/模板/   # 模板扩写，待手写替换（逐步清零）
docs/               # 语气词使用规范、数据说明
scripts/            # 生成脚本 + 手写批次源文件 handwritten_batch*.py
进度.md             # 手写替换进度台账
manifest.json       # 机器可读索引（含每文件 来源=手写/模板）
```

## 二、构成概览
| 类别 | 总数 | 已手写 | 模板剩余 |
|---|---|---|---|
| 情绪识别 | {rec_n} | {rec_hw_n} | {rec_tpl_n} |
| 共情对话·单轮 | {s_n} | {s_hw_n} | {s_tpl_n} |
| 共情对话·多轮 | {m_n} | {m_hw_n} | {m_tpl_n} |
| 语气词专项 | {tone_n} | {tone_hw_n} | {len(tone_tpl)} |
| **合计** | **{total}** | **{hw_all}** | **{tpl_all}** |

## 三、情绪识别（手写 {rec_hw_n} / 模板 {rec_tpl_n}）
标签14类；输出 `{{"emotion","intensity"(1-5),"trigger"}}`。
| 情绪 | 手写 | 模板 |
|---|---|---|
""" + "\n".join(f"| {e} | {len(rec_hw[e])} | {len(rec_tpl[e])} |" for e in TAXONOMY) + f"""

## 四、共情对话·单轮（手写 {s_hw_n} / 模板 {s_tpl_n}）
| 主题 | 手写 | 模板 |
|---|---|---|
""" + "\n".join(f"| {t} | {len(s_hw[t])} | {len(s_tpl[t])} |" for t in THEMES) + f"""

## 五、共情对话·多轮（手写 {m_hw_n} / 模板 {m_tpl_n}）
| 主题 | 手写 | 模板 |
|---|---|---|
""" + "\n".join(f"| {t} | {len(m_hw[t])} | {len(m_tpl[t])} |" for t in THEMES) + f"""

## 六、语气词专项（手写 {tone_hw_n} / 模板 {len(tone_tpl)}）
遵循《docs/语气词使用规范.md》。手写含近音词对照{len(tone_contrastive)}条 + 精选与批次{len(tone_other)}条。

## 七、审核与训练
- **逐条审核手写**：见 `scripts/handwritten_batch*.py`（源文件）或各 `data/*/手写/`。
- **只用手写训练**：仅加载各 `data/*/手写/` 目录。
- 严肃/哀伤/危机场景保持稳重、不用俏皮语气词；system 已注入语气词总原则。

## 八、复现与继续手写
- 复现：运行 `scripts/build_dataset.py`，自动汇总全部手写批次、模板补缺口、重算进度。
- 继续手写：新增 `scripts/handwritten_batchN.py`（HAND_REC/HAND_DIA/HAND_TONE），重跑即并入，模板缺口自动减少。
"""
tmp=os.path.join(OUT,"README.md.tmp"); open(tmp,"w",encoding="utf-8").write(readme); os.replace(tmp,os.path.join(OUT,"README.md"))
json.dump({"总量":total,"手写":hw_all,"模板":tpl_all,"seed":SEED,"手写批次":len(batch_files),"文件":manifest},open(os.path.join(OUT,"manifest.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)

print("总量:",total,"| 手写:",hw_all,"| 模板:",tpl_all,"| 手写占比:",pct(hw_all,total))
print("识别 手写/模板:",rec_hw_n,"/",rec_tpl_n,"| 单轮:",s_hw_n,"/",s_tpl_n,"| 多轮:",m_hw_n,"/",m_tpl_n,"| 语气词:",tone_hw_n,"/",len(tone_tpl))
print("手写批次文件:",[os.path.basename(f) for f in batch_files])
print("输出目录:",OUT)