import matplotlib.pyplot as plt

# obtained the values through output of report_stats
xs = [0.499184210757, 0.0190360552577, 0.00253638945399, 0.0, 0.0104520260469]
ys = [0.504363496467, 0.981513353717, 0.976309555146, 0.0, 0.014983700164]
texts = [-1, 0, 0.2, 1, 2]

plt.scatter(xs, ys)
plt.plot(xs, ys)
for i, txt in enumerate(texts):
    plt.annotate(txt, (xs[i], ys[i]))
