from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def setup():
    Path('outputs/figures').mkdir(parents=True,exist_ok=True); sns.set_theme(style='whitegrid',context='notebook')
def bar(values,path,title,ylabel):
    setup(); plt.figure(figsize=(7,4)); plt.bar(values.index.astype(str),values.values); plt.title(title); plt.ylabel(ylabel); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(path,dpi=150); plt.close()
