# -*- coding: utf-8 -*-
"""网文创作训练数据组装脚本。
读取 novel_batch1~5.py，按 9 类任务生成 messages 格式 JSONL + README + manifest。
校验：0 重复、0 英文字母、分类计数。输出到 OUT 目录。
"""
import os, sys, json, re, collections, importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dataset")

SYS = ("你是一位资深网络小说创作助手，精通玄幻、仙侠、都市、重生、系统流、言情、历史、科幻、悬疑、游戏等题材。"
       "你遵循网文创作规律：开篇黄金三章（快速入冲突、亮金手指、立人设、留钩子，忌大段设定堆砌与回忆杀）；"
       "爽点先抑后扬、节奏张弛有度；人设强标签加反差感；金手指有规则、限制与代价；反派智商在线；"
       "章末设断章钩子；行文以动作对话推进、短段落、展示而非告知。请根据用户的创作任务给出高质量示范。")

CATS = ["开篇黄金三章","续写","爽点打脸","人设","金手指系统","钩子断章","大纲设定","标题简介","技法诊断改写"]

def load_all():
    items = []
    for n in range(1, 6):
        m = importlib.import_module(f"novel_batch{n}")
        for it in m.NOVEL_BATCH:
            items.append(it)
    return items

def to_msg(it):
    return {"messages":[
        {"role":"system","content":SYS},
        {"role":"user","content":it["instruction"]},
        {"role":"assistant","content":it["output"]},
    ]}

def sig(it):
    return json.dumps([it["instruction"], it["output"]], ensure_ascii=False)

def main():
    items = load_all()
    # 校验
    seen=set(); dups=[]; eng=[]; badcat=[]
    for it in items:
        if it["cat"] not in CATS: badcat.append(it["cat"])
        s=sig(it)
        if s in seen: dups.append(it["instruction"][:20])
        seen.add(s)
        for k in ("instruction","output"):
            for w in re.findall(r'[A-Za-z]+', it[k]): eng.append((it["cat"],w))
    print("总条数:", len(items))
    print("重复:", len(dups), dups[:5])
    print("英文残留:", eng if eng else "无")
    print("非法分类:", badcat if badcat else "无")
    cnt=collections.Counter(it["cat"] for it in items)
    genre=collections.Counter(it["genre"] for it in items)
    print("分类计数:", dict(cnt))
    print("题材计数:", dict(genre))
    assert not dups and not eng and not badcat, "校验未通过"

    os.makedirs(os.path.join(OUT,"data"), exist_ok=True)
    os.makedirs(os.path.join(OUT,"scripts"), exist_ok=True)
    # 分类写 JSONL
    manifest={"total":len(items),"categories":{},"genres":dict(genre),"format":"messages(system/user/assistant)","sys_prompt":SYS}
    for i,cat in enumerate(CATS,1):
        sub=[it for it in items if it["cat"]==cat]
        fn=f"{i:02d}_{cat}.jsonl"
        path=os.path.join(OUT,"data",fn)
        with open(path,"w",encoding="utf-8") as f:
            for it in sub:
                f.write(json.dumps(to_msg(it),ensure_ascii=False)+"\n")
        manifest["categories"][cat]={"file":f"data/{fn}","count":len(sub)}
    # manifest
    with open(os.path.join(OUT,"manifest.json"),"w",encoding="utf-8") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)
    # README
    lines=["# 网文创作训练数据","",
        f"共 **{len(items)} 条**高质量手写示范，覆盖 9 类创作任务、{len(genre)} 种题材。",
        "格式：标准 JSONL，每行仅含 `messages`（system/user/assistant），兼容 LLaMA-Factory / ms-swift / Axolotl / OpenAI fine-tuning。","",
        "## 分类索引","", "| 类别 | 条数 | 文件 |","|---|---|---|"]
    for cat in CATS:
        c=manifest["categories"][cat]
        lines.append(f"| {cat} | {c['count']} | `{c['file']}` |")
    lines+=["","## 题材分布","", "| 题材 | 条数 |","|---|---|"]
    for g,n in genre.most_common():
        lines.append(f"| {g} | {n} |")
    lines+=["","## 设计说明","",
        "- 每条为「创作任务 → 高质量示范」指令对，output 为可直接学习的创作示范。",
        "- 开篇遵循黄金三章（快入冲突/亮金手指/立人设/留钩子）；爽点先抑后扬；人设强标签+反差；金手指有规则与代价；反派智商在线；章末断章钩子。",
        "- 技法诊断改写类采用「先诊断问题 → 再给改写/改进」结构，教模型识别并修正常见写作毛病。",
        "- 全部逐条手写，0 重复、0 英文借词。","",
        "## 训练建议","",
        "- 可整体加载，也可按分类/题材筛选子集（如只训开篇、只训某题材）。",
        "- system 提示已内置网文创作规律，训练时保留以稳定风格。","",
        "## 生成","", "`python3 scripts/build_novel.py <输出目录>` 可重新组装。源数据在 `scripts/novel_batch1~5.py`。"]
    with open(os.path.join(OUT,"README.md"),"w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    # 拷贝源脚本
    import shutil
    for n in range(1,6):
        shutil.copy(os.path.join(HERE,f"novel_batch{n}.py"), os.path.join(OUT,"scripts",f"novel_batch{n}.py"))
    shutil.copy(os.path.join(HERE,"build_novel.py"), os.path.join(OUT,"scripts","build_novel.py"))
    print("输出目录:", OUT)
    print("文件数:", sum(len(fs) for _,_,fs in os.walk(OUT)))

if __name__=="__main__":
    main()