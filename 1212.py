import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, TextBox, Button
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import random


# МОЙ МЕТОД
def my_clustering(cities, K):
    cities_arr = np.array(cities)
    if K <= 0 or len(cities_arr) == 0:
        return np.array([]), np.array([])

    centers = [cities_arr[0]]
    for _ in range(1, K):
        max_min_dist = -1
        best_city = None
        for city in cities_arr:
            if any(np.array_equal(city, c) for c in centers):
                continue
            min_dist = min(np.linalg.norm(city - c) for c in centers)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_city = city
        if best_city is not None:
            centers.append(best_city)
        else:
            break
    centers = np.array(centers)

    labels = []
    for city in cities_arr:
        dists = [np.linalg.norm(city - c) for c in centers]
        labels.append(np.argmin(dists))
    return np.array(labels), centers


# K-MEANS
def my_kmeans(cities, K, max_iters=100):
    cities_arr = np.array(cities)
    centers = np.array(random.sample(cities, K))
    for it in range(max_iters):
        labels = np.array([np.argmin([np.linalg.norm(c - cent) for cent in centers]) for c in cities_arr])
        new_centers = []
        for i in range(K):
            cluster = cities_arr[labels == i]
            if len(cluster) > 0:
                new_centers.append(np.mean(cluster, axis=0))
            else:
                new_centers.append(centers[i])
        new_centers = np.array(new_centers)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return labels, centers, it + 1


# SKLEARN
def sklearn_kmeans(cities, K):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(np.array(cities))
    return kmeans.labels_, kmeans.cluster_centers_, kmeans.n_iter_


#  ТЕСТЫ
def generate_random_cities(n):
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


short_names = ['Тест1', 'Тест2', 'Тест3', 'Тест4', 'Тест5', 'Тест6', 'Тест7', 'Тест8', 'Тест9', 'Тест10']

full_names = {
    'Тест1': 'Случайные города',
    'Тест2': 'Три разнесенных кластера',
    'Тест3': 'Один кластер',
    'Тест4': 'Все точки одинаковы',
    'Тест5': 'Пустой кластер',
    'Тест6': 'Пересекающиеся кластеры',
    'Тест7': 'Кластеры разной плотности',
    'Тест8': 'Кластеры + шум',
    'Тест9': 'Один выброс',
    'Тест10': '3 сгенерированных кластера'
}

test_data = {
    'Тест1': None,
    'Тест2': [(2, 2), (2, 3), (3, 2), (7, 7), (8, 7), (7, 8), (12, 12), (13, 12), (12, 13)],
    'Тест3': [(5, 5), (5.5, 5.5), (4.5, 4.5), (5, 4.5), (4.5, 5)],
    'Тест4': [(5, 5)] * 30,
    'Тест5': [(0, 0), (1, 0), (0, 1), (1, 1), (10, 10)],
    'Тест6': [(2, 2), (3, 2), (2, 3), (3, 3), (2.5, 2.5), (3.5, 2.5), (2.5, 3.5), (3.5, 3.5)],
    'Тест7': [(0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1), (8, 8), (9, 9), (8, 9), (9, 8)],
    'Тест8': [(0, 0), (1, 0), (0, 1), (1, 1), (5, 5), (6, 5), (5, 6), (6, 6), (2, 2), (7, 2)],
    'Тест9': [(0, 0)] * 8 + [(100, 100)],
    'Тест10': [tuple(pt) for pt in make_blobs(n_samples=80, centers=3, random_state=42)[0]]
}

#  ОКНО
fig = plt.figure(figsize=(14, 7))
fig.subplots_adjust(left=0.1, bottom=0.12, right=0.98, top=0.92)


ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)
plt.subplots_adjust(left=0.2)

# МЕНЮ
rax = plt.axes([0.03, 0.25, 0.08, 0.60])
radio = RadioButtons(rax, short_names)

# Поле для N
ax_n = plt.axes([0.03, 0.18, 0.08, 0.05])
text_n = TextBox(ax_n, 'N =', initial='100')

# Поле для K
ax_k = plt.axes([0.03, 0.11, 0.08, 0.05])
text_k = TextBox(ax_k, 'K =', initial='3')


ax_btn = plt.axes([0.03, 0.04, 0.08, 0.05])
btn = Button(ax_btn, 'ОБНОВИТЬ', color='#3498DB', hovercolor='#2980B9')


for label in radio.labels:
    label.set_fontsize(9)

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']


def update(event):
    test_short = radio.value_selected
    test_full = full_names[test_short]

    if test_short == 'Тест1':
        n = int(text_n.text)
        points = np.array(generate_random_cities(n))
        title = f'{test_short}: {test_full} | N = {n}'
    else:
        points = np.array(test_data[test_short])
        title = f'{test_short}: {test_full}'

    k = int(text_k.text)
    if k > len(points):
        k = len(points)
    if k < 1:
        k = 1

    labels1, centers1 = my_clustering(points, k)
    labels2, centers2, it2 = my_kmeans(points.tolist(), k)
    labels3, centers3, it3 = sklearn_kmeans(points.tolist(), k)

    ax1.clear()
    ax2.clear()
    ax3.clear()

    for i in range(k):
        mask1 = labels1 == i
        mask2 = labels2 == i
        mask3 = labels3 == i

        if np.any(mask1):
            ax1.scatter(points[mask1, 0], points[mask1, 1], c=colors[i % len(colors)], s=30, alpha=0.7)
        if np.any(mask2):
            ax2.scatter(points[mask2, 0], points[mask2, 1], c=colors[i % len(colors)], s=30, alpha=0.7)
        if np.any(mask3):
            ax3.scatter(points[mask3, 0], points[mask3, 1], c=colors[i % len(colors)], s=30, alpha=0.7)

    if len(centers1) > 0:
        ax1.scatter(centers1[:, 0], centers1[:, 1], marker='X', c='black', s=100, linewidths=2)
    if len(centers2) > 0:
        ax2.scatter(centers2[:, 0], centers2[:, 1], marker='X', c='black', s=100, linewidths=2)
    if len(centers3) > 0:
        ax3.scatter(centers3[:, 0], centers3[:, 1], marker='X', c='black', s=100, linewidths=2)

    ax1.set_title('Мой метод', fontsize=11, fontweight='bold')
    ax2.set_title(f'Мой K-means | {it2} итер.', fontsize=11, fontweight='bold')
    ax3.set_title(f'Sklearn K-means | {it3} итер.', fontsize=11, fontweight='bold')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.2)

    fig.suptitle(f'{title} | K = {k}', fontsize=13, fontweight='bold')
    fig.canvas.draw_idle()


def toggle_n(event):
    if radio.value_selected == 'Тест1':
        text_n.ax.set_visible(True)
    else:
        text_n.ax.set_visible(False)
    fig.canvas.draw_idle()


radio.on_clicked(toggle_n)
btn.on_clicked(update)

text_n.ax.set_visible(True)
update(None)

plt.show()