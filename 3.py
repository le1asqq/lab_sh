import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from collections import Counter
data = [
    ['Яблоко', 7, 7, 'Фрукт'],
    ['Салат', 2, 5, 'Овощ'],
    ['Бекон', 1, 2, 'Протеин'],
    ['Банан', 9, 1, 'Фрукт'],
    ['Орехи', 1, 5, 'Протеин'],
    ['Рыба', 1, 1, 'Протеин'],
    ['Сыр', 1, 1, 'Протеин'],
    ['Виноград', 8, 1, 'Фрукт'],
    ['Морковь', 2, 8, 'Овощ'],
    ['Апельсин', 6, 1, 'Фрукт'],
]
X = np.array([[d[1], d[2]] for d in data])
y = np.array([d[3] for d in data])
class CustomKNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    def fit(self, X, y):#запоминает таблицу с продуктами
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    def predict(self, X):#берет новый продукт смотрит(считает по евкликодову расстоянию)
        #на самых похожих на него  и опред класс(какой класс встречается чаще)
        # если 1.1.1 то выбирает который ближе(бибилотечный выбирает первого в списке)
        X = np.array(X)
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_idx]
            vote_counts = Counter(k_labels)
            max_votes = max(vote_counts.values())
            winners = [label for label, count in vote_counts.items() if count == max_votes]
            if len(winners) > 1:
                winner_distances = []
                for winner in winners:
                    winner_mask = self.y_train == winner
                    min_dist = np.min(distances[winner_mask])
                    winner_distances.append((winner, min_dist))
                winner_distances.sort(key=lambda x: x[1])
                predictions.append(winner_distances[0][0])
            else:
                predictions.append(winners[0])
        return np.array(predictions)
    def predict_proba(self, X):#вероятность того что это
        X = np.array(X)
        probas = []
        unique_classes = np.unique(self.y_train)
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_idx = np.argsort(distances)[:self.n_neighbors]
            k_labels = self.y_train[k_idx]
            proba = [np.sum(k_labels == c) / self.n_neighbors for c in unique_classes]
            probas.append(proba)
        return np.array(probas)
test_products = [
    ['Томат', 4, 4],
    ['Мороженое', 8, 2],
    ['Картофель фри', 1, 9]
]
test_points = [[4, 4], [8, 2], [1, 9]]
test_labels = ['Томат', 'Мороженое', 'Картофель фри']
def draw_test_points(ax, points, labels):
    for i, (sweet, crunch) in enumerate(points):
        star_label = 'Тестовый объект' if i == 0 else None
        ax.scatter(sweet, crunch, marker='*', s=250, c='black', edgecolors='white', linewidths=2, label=star_label)
        ax.annotate(labels[i], (sweet + 0.2, crunch + 0.2), fontsize=9)
print("Сравнение k=3")
my_knn = CustomKNNClassifier(n_neighbors=3)
my_knn.fit(X, y)
sk_knn = KNeighborsClassifier(n_neighbors=3)
sk_knn.fit(X, y)
for name, sweet, crunch in test_products:
    my_pred = my_knn.predict([[sweet, crunch]])[0]
    sk_pred = sk_knn.predict([[sweet, crunch]])[0]
    print(f"{name:15} свой: {my_pred:10} sklearn: {sk_pred}")
print("\nВероятность выпадения:")
for name, sweet, crunch in test_products:
        proba = my_knn.predict_proba([[sweet, crunch]])[0]
        print(f"\n{name}:")
        for i, cls in enumerate(np.unique(y)):
            print(f"  {cls}: {proba[i] * 100:.1f}%")
def plot_decision_boundary(X, y, model, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    unique = np.unique(y)
    color_map = {label: i for i, label in enumerate(unique)}
    Z_num = np.array([color_map[z] for z in Z])
    Z_num = Z_num.reshape(xx.shape)
    colors_list = ['red', 'green', 'blue', 'orange', 'purple']
    ax.contourf(xx, yy, Z_num, alpha=0.3, levels=np.arange(len(unique) + 1) - 0.5, colors=colors_list[:len(unique)])
    for label in unique:
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_list[color_map[label]], s=80, label=label)
    ax.set_xlabel('Сладость')
    ax.set_ylabel('Хруст')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_decision_boundary(X, y, my_knn, 'Разделяющая поверхность (свой k-NN, k=3)', axes[0])
plot_decision_boundary(X, y, sk_knn, 'Разделяющая поверхность (sklearn k-NN, k=3)', axes[1])
draw_test_points(axes[0], test_points, test_labels)
draw_test_points(axes[1], test_points, test_labels)
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()
new_class_data = [
    ['Пирожное', 8, 3, 'Десерт'],
    ['Конфеты', 8, 2, 'Десерт'],
    ['Печенье', 7, 6, 'Десерт'],
]
X_new = np.vstack([X, [[d[1], d[2]] for d in new_class_data]])
y_new = np.hstack([y, [d[3] for d in new_class_data]])
my_knn_new = CustomKNNClassifier(n_neighbors=3)
my_knn_new.fit(X_new, y_new)
sk_knn_new = KNeighborsClassifier(n_neighbors=3)
sk_knn_new.fit(X_new, y_new)
print("\nСравнение после добавления новго класса")
for name, sweet, crunch in test_products:
    my_pred = my_knn_new.predict([[sweet, crunch]])[0]
    sk_pred = sk_knn_new.predict([[sweet, crunch]])[0]
    print(f"{name:15}  свой: {my_pred:10} sklearn: {sk_pred}")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_decision_boundary(X_new, y_new, my_knn_new, 'С новым классом (свой k-NN)', axes[0])
plot_decision_boundary(X_new, y_new, sk_knn_new, 'С новым классом  (sklearn)', axes[1])

draw_test_points(axes[0], test_points, test_labels)
draw_test_points(axes[1], test_points, test_labels)

axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.show()
loo = LeaveOneOut()
correct_my = 0
correct_sk = 0
start_my = time.time()
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    my_knn_loo = CustomKNNClassifier(n_neighbors=3)
    my_knn_loo.fit(X_train, y_train)
    if my_knn_loo.predict(X_test)[0] == y_test[0]:
        correct_my += 1
time_my = time.time() - start_my
start_sk = time.time()
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sk_knn_loo = KNeighborsClassifier(n_neighbors=3)
    sk_knn_loo.fit(X_train, y_train)
    if sk_knn_loo.predict(X_test)[0] == y_test[0]:
        correct_sk += 1
time_sk = time.time() - start_sk
print(f"точность свой k-NN: {correct_my}/{len(X)} = {correct_my / len(X):.2f} (время: {time_my:.4f}с)")
print(f"точность sklearn:   {correct_sk}/{len(X)} = {correct_sk / len(X):.2f} (время: {time_sk:.4f}с)")
from sklearn.metrics import confusion_matrix
y_true = []
y_pred = []
for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    my_knn_loo = CustomKNNClassifier(n_neighbors=3)
    my_knn_loo.fit(X_train, y_train)
    y_true.append(y_test[0])
    y_pred.append(my_knn_loo.predict(X_test)[0])
cm = confusion_matrix(y_true, y_pred, labels=['Овощ', 'Протеин', 'Фрукт'])
print("\nМатрица ошибок:")
print("            Предсказано")
print("            Овощ  Протеин  Фрукт")
print(f"Овощ         {cm[0][0]:3}     {cm[0][1]:3}       {cm[0][2]:3}")
print(f"Протеин      {cm[1][0]:3}     {cm[1][1]:3}       {cm[1][2]:3}")
print(f"Фрукт        {cm[2][0]:3}     {cm[2][1]:3}       {cm[2][2]:3}")
print("\nТест на ничью (белый ящик)")
tie_point = np.array([[4, 4]])
my_pred = my_knn.predict(tie_point)[0]
sk_pred = sk_knn.predict(tie_point)[0]
print(f"Точка (4,4)  свой: {my_pred}, sklearn: {sk_pred}")
print("ничья, выбор ближайшего")
print("\nТест на выбросы (чёрный ящик)")
X_outlier = np.vstack([X, [[100, 100]]])
y_outlier = np.hstack([y, 'Выброс'])
my_knn_out = CustomKNNClassifier(n_neighbors=3)
my_knn_out.fit(X_outlier, y_outlier)
print("Классификация яблока после добавления выброса:", my_knn_out.predict([[7, 7]])[0])
print("Выброс (100,100) :", my_knn_out.predict([[100, 100]])[0])