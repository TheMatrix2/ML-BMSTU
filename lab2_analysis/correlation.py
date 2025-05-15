import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Загрузка данных
df = pd.read_csv('heart.csv')

# Просмотр первых 5 строк
print("Первые 5 строк датасета:")
print(df.head())

# Базовая информация о датасете
print("\nОсновная информация о датасете:")
print(df.info())

# Статистическое описание
print("\nСтатистическое описание датасета:")
print(df.describe())

# Проверка пропущенных значений
print("\nКоличество пропущенных значений по каждому столбцу:")
print(df.isnull().sum())

# Определение типов данных для каждого признака
print("\nТипы признаков в датасете:")
data_types = {
    'Количественный непрерывный': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
    'Количественный дискретный': ['ca'],
    'Категориальный (бинарный)': ['sex', 'fbs', 'exang', 'target'],
    'Категориальный/ранговый': ['cp', 'restecg', 'slope', 'thal']
}

for type_name, features in data_types.items():
    print(f"{type_name}: {', '.join(features)}")

# Проверка распределения количественных признаков на нормальность
print("\nПроверка на нормальность распределения (тест Шапиро-Уилка):")
for feature in data_types['Количественный непрерывный']:
    stat, p_value = stats.shapiro(df[feature])
    print(f"{feature}: p-value = {p_value:.6f} ({'нормальное' if p_value > 0.05 else 'не нормальное'} распределение)")

# 1. Корреляция Пирсона для количественных непрерывных признаков
print("\n1. Корреляция Пирсона между количественными непрерывными признаками:")
continuous_features = data_types['Количественный непрерывный']
pearson_corr = df[continuous_features].corr(method='pearson')
print(pearson_corr)

# Визуализация корреляции Пирсона
plt.figure(figsize=(10, 8))
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция Пирсона между количественными непрерывными признаками')
plt.tight_layout()
plt.savefig('pearson_correlation_heatmap.png')
plt.show()

# 2. Корреляция Спирмена для всех количественных признаков (не требует нормальности)
print("\n2. Корреляция Спирмена для всех количественных признаков:")
quant_features = data_types['Количественный непрерывный'] + data_types['Количественный дискретный']
spearman_corr = df[quant_features].corr(method='spearman')
print(spearman_corr)

# Визуализация корреляции Спирмена
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция Спирмена между количественными признаками')
plt.tight_layout()
plt.savefig('spearman_correlation_heatmap.png')

# 3. Корреляция Кендалла для всех количественных признаков
print("\n3. Корреляция Кендалла для всех количественных признаков:")
kendall_corr = df[quant_features].corr(method='kendall')
print(kendall_corr)

# 4. Point-biserial корреляция между количественными и бинарными признаками
print("\n4. Point-biserial корреляция между количественными и целевой переменной:")
for quant in quant_features:
    r, p = stats.pointbiserialr(df[quant], df['target'])
    print(f"{quant} vs target: r = {r:.4f}, p-value = {p:.6f}")

# 5. Хи-квадрат для категориальных признаков
print("\n5. Хи-квадрат и V Крамера для категориальных признаков с целевой переменной:")
for cat in data_types['Категориальный (бинарный)'] + data_types['Категориальный/ранговый']:
    if cat == 'target':
        continue

    contingency_table = pd.crosstab(df[cat], df['target'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))

    print(f"{cat} vs target: Chi2 = {chi2:.4f}, p-value = {p:.6f}, V Крамера = {v:.4f}")

    # Вывод таблицы сопряженности
    print(f"Таблица сопряженности {cat} и target:")
    print(contingency_table)
    print()

# 6. Корреляционное отношение (Эта-квадрат) для нелинейных связей
print("\n6. Корреляционное отношение для каждого количественного признака с целевой переменной:")


def correlation_ratio(x, y):
    """Вычисление корреляционного отношения"""
    y_unique = np.unique(y)
    x_means = [np.mean(x[y == val]) for val in y_unique]
    x_mean = np.mean(x)

    ss_total = np.sum((x - x_mean) ** 2)
    ss_between = np.sum([(np.sum(y == val) * (mean - x_mean) ** 2) for val, mean in zip(y_unique, x_means)])

    return np.sqrt(ss_between / ss_total) if ss_total != 0 else 0


for quant in quant_features:
    eta = correlation_ratio(df[quant].values, df['target'].values)
    print(f"{quant} vs target: η = {eta:.4f}")

# 7. Визуализация отношений между признаками и целевой переменной
print("\n7. Визуализация отношений между признаками и целевой переменной:")

# Для количественных признаков с целевой переменной - ящик с усами
plt.figure(figsize=(15, 10))
for i, quant in enumerate(quant_features):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='target', y=quant, data=df)
    plt.title(f'{quant} vs target')
plt.tight_layout()
plt.savefig('boxplots.png')

# Тепловая карта для всех признаков
plt.figure(figsize=(15, 12))
all_corr = df.corr(method='spearman')
sns.heatmap(all_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция Спирмена между всеми признаками')
plt.tight_layout()
plt.savefig('all_correlation_heatmap.png')

# 8. Определение наиболее важных признаков для машинного обучения
print("\n8. Наиболее важные признаки для прогнозирования сердечных заболеваний:")

# По абсолютному значению корреляции с целевой переменной
target_correlations = {}

# Корреляция Спирмена для количественных признаков
for quant in quant_features:
    corr, p = stats.spearmanr(df[quant], df['target'])
    target_correlations[quant] = abs(corr)

# Point-biserial для категориальных бинарных признаков
for cat in data_types['Категориальный (бинарный)']:
    if cat != 'target':
        corr, p = stats.pointbiserialr(df[cat], df['target'])
        target_correlations[cat] = abs(corr)

# Для категориальных/ранговых используем V Крамера
for cat in data_types['Категориальный/ранговый']:
    contingency_table = pd.crosstab(df[cat], df['target'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    v = np.sqrt(chi2 / (n * min(contingency_table.shape) - 1))
    target_correlations[cat] = v

# Сортировка признаков по степени связи с целевой переменной
sorted_features = sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)
print("Признаки, отсортированные по степени связи с целевой переменной:")
for feature, value in sorted_features:
    print(f"{feature}: {value:.4f}")

# 9. Пример применения знаний о корреляции для выбора признаков
print("\n9. Пример выбора признаков для модели на основе корреляционного анализа:")

# Выбираем топ-5 признаков по важности
top_features = [f for f, _ in sorted_features[:5]]
print(f"Выбранные признаки для модели: {top_features}")

# Визуализация корреляций между выбранными признаками
plt.figure(figsize=(10, 8))
top_features_corr = df[top_features + ['target']].corr(method='spearman')
sns.heatmap(top_features_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляция между топ-5 признаками и целевой переменной')
plt.tight_layout()
plt.savefig('top_features_correlation.png')

# 10. Краткие выводы
print("\n10. Краткие выводы по результатам корреляционного анализа:")
print("- Для выявления линейных связей между количественными непрерывными признаками использовался коэффициент Пирсона")
print(
    "- Для оценки монотонных связей между количественными признаками без предположения о нормальности распределения использовались коэффициенты Спирмена и Кендалла")
print(
    "- Для анализа связи между категориальными признаками и целевой переменной использовались критерий хи-квадрат и V Крамера")
print(
    "- Для оценки силы связи между количественными и бинарными признаками использовался коэффициент точечно-бисериальной корреляции")
print("- Для нелинейных связей было рассчитано корреляционное отношение")
print(
    f"- Наиболее информативными признаками для прогнозирования сердечных заболеваний являются: {', '.join(top_features)}")