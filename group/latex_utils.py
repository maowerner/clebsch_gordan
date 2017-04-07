"""Utils for latex related things."""

try:
    import sympy
    usesympy=True
except ImportError:
    usesympy=False

def start_document(packages=["amsmath", "booktabs"]):
    print("\\documentclass[10pt,a4paper]{article}")
    for p in packages:
        print("\\usepackage{%s}" % p)
    print("\\begin{document}")

def end_document():
    print("\\end{document}")

def latexify(arg):
    if usesympy:
        tmp = sympy.nsimplify(arg)
        tmp1 = sympy.simplify(tmp)
        tmp = str(sympy.latex(tmp1,fold_short_frac=True))
    else:
        tmp = str(arg)
    return tmp

def hrules(booktabs=True):
    if booktabs:
        trule = "\\toprule"
        mrule = "\\midrule"
        brule = "\\bottomrule"
    else:
        trule = "\\hline"
        mrule = trule
        brule = trule
    return trule, mrule, brule

def start_table(align="ll|cc|l"):
    print("\\begin{table}")
    print("\\begin{tabular}{%s}" % align)

def end_table(label=None, caption=None):
    if caption is not None:
        print("\\caption{%s}" % caption)
    if label is not None:
        print("\\label{%s}" % label)
    print("\\end{tabular}")
    print("\\end{table}")

