import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

xs = [0.0714946601559, 0.00266481423647, 0.003, 0.144028393552, 0.144028393552]
ys = [0.504019276328, 0.981513353717, 0.976, 0.00083017798206, 0.00083017798206]
texts = [-1, 0, 0.2, 1, 2]

plt.scatter(xs, ys)
for i, txt in enumerate(texts):
    plt.annotate(txt, (xs[i], ys[i]))
