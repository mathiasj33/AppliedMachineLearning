import matplotlib.pyplot as plt

# obtained the values through output of report_stats
xs = [0.07144504149, 0.00266481423647, 0.00341493171596, 0.144148062099, 0.141988190758]
ys = [0.504363496467, 0.981513353717, 0.976309555146, 0.0, 0.014983700164]
texts = [-1, 0, 0.2, 1, 2]

plt.scatter(xs, ys)
plt.plot(xs, ys)
for i, txt in enumerate(texts):
    plt.annotate(txt, (xs[i], ys[i]))
