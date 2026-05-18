# Coffee Grind Fineness — Project Analysis & Roadmap

**Дата аудита:** май 2026  
**Ветка / контекст:** `Conformal_regression`, baseline `convnext_small` → `data/runs/convnext_small/run_001`  
**Источники:** `src/`, `data/runs/`, `README.md`, `CONCEPTS.md`, `conformal_report.json`

---

## 1. Executive summary

Проект представляет собой **зрелый исследовательский пайплайн** для регрессии fineness по фото зёрен: групповой сплит без утечки, сегментация фона, несколько бэкбонов, FGSM-adversarial training, сетка экспериментов (`run_all_experiments.py`) и отдельный модуль **split conformal prediction**.

**Сильная сторона:** на текущем `run_001` точечная модель уже даёт **test MAE ≈ 3.2** (шкала Fineness из CSV, 0–100) — это порядка **8–10%** относительно типичного разброса меток (~40 ± 15).

**Главный разрыв с «идеалом» из README:** обучение `run_001` **не совпадает** с рекомендованным рецептом сетки (`AdamW`, cosine LR, Huber, `lr=3e-4`, **100 эпох**). Использованы дефолты: Adam, ReduceLROnPlateau, MSE, `lr=1e-3`, **30 эпох**.

**Главный инсайт по uncertainty:** conformal-слой **работает статистически корректно (консервативно)**, но **практически малоинформативен** — симметричный интервал **одинаковой ширины** (~17.2 Fineness при α=0.1) при MAE ~3.2, то есть ширина ≈ **5.3× MAE**. Это следствие метода (глобальный квантиль |y−ŷ|), а не только «плохой модели».

---

## 2. Текущий статус

### 2.1. Данные и протокол оценки

| Аспект | Факт | Оценка |
|--------|------|--------|
| Объём после QC | ~607 изображений, **~162 grain-группы** (4 кадра/зерно) | Мало для deep learning «с нуля»; достаточно для fine-tune |
| Сплит | Group-aware (`split_labels.py`, seed=42), фиксированные `train/val/test.csv` | **Хорошо** — защита от leakage по зерну |
| Train / val / test | 489 / 60 / 58 изображений; Fineness mean ≈ 45.8 / 41.2 / **38.2** | **Сдвиг распределения** test (ниже среднее, уже std) |
| Вход модели | Сегментированные 1080² → `CenterCrop(224)` + dataset-specific normalize | Теряется контекст кадра; маска завязана на «коричневый» цвет |
| Дубликаты в батче | 4 почти идентичных ракурса одного зерна | Эффективный размер выборки **<< 607** |

**Вывод:** метрики на уровне **изображений** завышают уверенность; для науки и продукта нужна **оценка на уровне grain-group** (одно предсказание на 4 кадра или агрегация).

### 2.2. Результаты `run_001` (ConvNeXt-Small)

**Конфигурация** (`config.json`):

- `unfreeze_after: 15`, `adversarial: true`, `adv_epsilon: 0.01`, `adv_weight: 0.5`
- `loss: mse`, `optimizer: adam`, `scheduler: plateau`, `lr: 0.001`, `epochs: 30`
- Без online/precomputed augmentation, segmented images

**Кривая обучения** (`metrics.csv`, MAE в нормализованной шкале ×100 в логах):

| Эпоха | Train MAE | Val MAE | Комментарий |
|-------|-----------|---------|-------------|
| 15 (до unfreeze) | 7.4 | 8.4 | Стабильное снижение |
| **16** (unfreeze) | **16.5** | **15.0** | Резкий скачок — типичный шок при разморозке backbone |
| 26 | 2.9 | **3.9** | Лучшие val-эпохи |
| 30 | 2.7 | **3.8** | Финальный чекпоинт `epoch_030.pt` |

**Точечные метрики:**

- **Val** (scatter `predictions_epoch_030.csv`, n=60): MAE ≈ **3.77**, RMSE ≈ **4.70**, max error ≈ **11.1**
- **Test** (`conformal_report.json`, тот же вес): MAE ≈ **3.24**, RMSE ≈ **4.05**

Test MAE **ниже** val — возможно удачная выборка test (n=58, 16 групп) и сдвиг распределения, а не «переобучение на test». Требуется **group-level** и **bootstrap** доверительные интервалы для MAE.

### 2.3. Сравнение с «идеалом» из README / сетки экспериментов

В репозитории **нет** локального `TRAINING_PLAN.md` с итоговой таблицей MAE по всем моделям; эталон задаётся текстом README и `run_all_experiments.py`:

```text
BASE = train.py --epochs 100 --adamw --cosine-lr --huber --huber-delta 0.03 --lr 3e-4
ConvNeXt-Small best recipe: + --unfreeze-after 15 --adversarial --adv-epsilon 0.02 --adv-weight 0.4
```

| Параметр | `run_001` (факт) | README / grid «ideal» |
|----------|------------------|------------------------|
| Эпохи | 30 | 100 |
| Loss | MSE | Huber (δ=0.03) |
| Optimizer | Adam | AdamW |
| Scheduler | Plateau | Cosine |
| LR | 1e-3 | 3e-4 |
| Adv ε / weight | 0.01 / 0.5 | 0.02 / 0.4 |
| Augmentation | нет | опционально (для ConvNeXt в README — **без** aug в лучшем рецепте) |

**Оценка близости к идеалу:** по **архитектуре и стратегии unfreeze+adversarial** — близко; по **оптимизации и длительности обучения** — **далеко** (оценочно использовано ~30% рекомендованного бюджета и другой loss landscape). Ожидаемый запас: **ещё 10–25%** снижения MAE без смены модели — гипотеза, требует одного прогона по рецепту README.

`CONCEPTS.md` дополнительно фиксирует: **агрессивная аугментация на этом датасете хуже умеренной в ~5×** — для помола важна **текстура зерна**, а не «картинка вообще».

### 2.4. Bottlenecks в коде и инфраструктуре

1. **Чекпоинты:** сохранение только каждые 5 эпох; **нет `best.pt` по val MAE** — conformal и отчёты по умолчанию берут **последнюю** эпоху, не лучшую (epoch 26 val MAE 3.94 vs epoch 30 3.80 — разница небольшая, но принцип важен).

2. **Сегментация** (`region_extraction.py`): маска по фиксированному коричневому вектору + порог относительно mean — чувствительна к **балансу белого, экспозиции, сорту зерна**. Ошибки маски → шум в признаках fineness.

3. **CenterCrop(224)** без multi-scale / crop по маске — игнорируется пространственная структура кучи зёрен на 1080².

4. **Conformal calibration на val:** val участвовал в **early stopping / выборе эпох** косвенно (мониторинг); для строгой CP-гарантии нужен **третий hold-out** (calibration-only).

5. **Коррелированные кадры:** train_loader shuffle не устраняет зависимость внутри группы; MAE по 58 test-изображениям ≠ 58 независимых наблюдений.

6. **Отсутствие group-metrics в `train.py` / conformal** — только image-level CSV.

---

## 3. Анализ Conformal Prediction

### 3.1. Что реализовано

- **Метод:** symmetric **split conformal** с score \(R_i = |y_i - \hat f(x_i)|\), один глобальный half-width \(q\) на всех test-точках (`src/conformal/split.py`).
- **Калибровка:** validation (n=60 изображений ≈ **15 grain-групп**).
- **Оценка:** test (n=58).
- **Отчёт:** Fineness-шкала (×100), внутренний блок `internal_normalized_0_1` для отладки.

### 3.2. Цифры `conformal_report.json` (α = 0.1, 90% номинал)

| Метрика | Значение | Интерпретация |
|---------|----------|----------------|
| Test MAE (baseline point) | **3.24** | Сильная точечная модель для узкой задачи |
| Empirical coverage | **96.6%** | Выше номинала 90% → **консервативно** |
| Mean interval width | **17.20** | Очень широко относительно ошибки |
| Median width | **17.20** | = mean → **константная** ширина для всех x |
| Calibration half-width q | **8.60** | ±8.6 Fineness вокруг ŷ |
| Width / MAE | **~5.3×** | Интервал слабо полезен для «точного» UI |

**Alpha-sweep (test):** при снижении номинала coverage падает быстрее ширины; при α=0.2 coverage **82.8%** при номинале 80% — уже ближе к калибровке, но всё ещё широкие полосы.

### 3.3. Насколько метод «эффективен»?

**Плюсы (для исследования и compliance):**

- Простота, воспроизводимость, **distribution-free** гарантия при exchangeability.
- Модуль отделён от `train.py` — не ломает обучение.
- Покрытие ≥ номинала — **не недооцениваем** риск (важно для безопасности рекомендаций помола).

**Минусы (для продукта и оператора):**

- **Одинаковая ширина** для лёгких и сложных кадров — не отражает гетероскедастичность (размытие, плохая маска, редкие fineness).
- Ширина доминирует над **размахом меток test** (min 8.9, max 72.3, std ≈ 14.7): интервал ±8.6 вокруг ŷ часто перекрывает **половину шкалы** — плохая **resolution** uncertainty.
- Малое **n_calib** (60 точек, ~15 групп) → дискретный квантиль, скачки ширины при смене α.
- **Val reuse** для калибровки и ранней остановки — завышает доверие к coverage на paper-level.

### 3.4. Стоит ли переходить на CQR (Conformalized Quantile Regression)?

| Критерий | Split conformal (сейчас) | CQR |
|----------|--------------------------|-----|
| Адаптивная ширина по x | Нет | **Да** |
| Нужно переобучение | Нет | **Да** (нижний/верхний квантиль или multi-quantile head) |
| Данные | Достаточно | **На грани** (~130 групп) — риск переобучения квантилей |
| Интерпретируемость | Высокая | Средняя |
| Ожидаемый эффект на width | Baseline | Часто **−20–40%** mean width при том же coverage *если* ошибка гетероскедастична |

**Рекомендация:**

1. **Краткосрочно** — оставить split CP, но:
   - выделить **calibration split** из train (20–25% групп);
   - отчитывать **group-level** coverage и **условное** покрытие по бинам fineness;
   - добавить **normalized width** = width / (local label range или pred).

2. **Среднесрочно** — пилот **CQR** на ConvNeXt с двумя головами (q10, q90) + conformal correction на calib; сравнить mean width при фиксированном coverage.

3. **Jackknife+ / CV+** — только если появится стабильный baseline и нужна теория при малых n; вычислительно тяжелее, выигрыш на n≈60 сомнителен.

**Вердикт:** переход на CQR **оправдан**, но **после** (a) лучшего point-predictor по README-рецепту, (b) отдельного calib-holdout, (c) group-metrics. Иначе CQR будет калибровать шум сегментации, а не истинную aleatoric uncertainty помола.

---

## 4. Предложения по улучшению (технический стек)

### 4.1. Данные

| Идея | Обоснование для помола кофе |
|------|---------------------------|
| **Геометрическая aug без photometric** на segmented (`--no-photometric` в precompute) | Сохраняет цвет/текстуру зерна; увеличивает разнообразие ракурса без «ломания» признака fineness |
| **Не** включать тяжёлый JPEG/blur/perspective online | Подтверждено `CONCEPTS.md` и недавним смягчением `AugmentationConfig` |
| **Crop по bounding box маски** вместо center 224 | Фокус на зёрнах, меньше пустого фона после сегментации |
| Пересчёт **dataset mean/std** после смены пайплайна | Текущие stats привязаны к segmented train |
| **Group-level labels audit** | 4 кадра с одним fineness — проверить аннотационный шум (одна ли настройка мельницы?) |
| Улучшение маски (HSV/LAB + морфология или лёгкий U-Net) | Снижает outliers, которые раздувают conformal q |

### 4.2. Модели

| Модель | Когда имеет смысл |
|--------|-------------------|
| **ConvNeXt-Small** (текущий лидер) | Оставить primary; добить hyperparams README |
| **EfficientNet-B0** | README: сильна с aug+adv; быстрые эксперименты |
| **ResNet-152** | Больше ёмкость; риск OOM, нужен bs=16 |
| **ViT / ViT-Large** | В сетке часто OOM/нестабильны на этом n — **низкий приоритет** |
| **Ensemble** (ConvNeXt + EfficientNet) | Может снизить MAE и **сузить** conformal residuals |

Для помола важны **локальные текстуры** → CNN/ConvNeXt предпочтительнее ViT при <1k независимых групп.

### 4.3. Loss functions

| Loss | Рекомендация |
|------|--------------|
| **MSE** (сейчас) | Чувствителен к выбросам (max val error ~11); хорош для гладких предсказаний |
| **Huber** (δ=0.03 в grid) | **Первый switch** — робастность к редким плохим кадрам/маскам без потери дифференцируемости |
| **L1 / MAE loss** | Ещё более робастен; может слегка ухудшить RMSE, улучшить медиану ошибки |
| **Conformal** | Не заменяет loss; измеряет uncertainty поверх любого loss |

Для **сужения интервалов** важнее улучшить **|y−ŷ|** на calibration, чем менять только conformal wrapper.

### 4.4. Обучение и оценка (код)

- Сохранять **`best.pt`** по val MAE (group-aggregated).
- Логировать **test** только один раз в конце (сейчас test не в train loop — хорошо).
- `run_conformal.py`: опция `--checkpoint best.pt`, `--calib-split holdout`.
- Метрики: **MAE_group**, **coverage_group**, **ECE**-style calibration curve (nominal vs empirical по α).

---

## 5. Roadmap развития

### 5.1. Short-term (1–2 недели, low-hanging fruit)

| # | Задача | Ожидаемый эффект |
|---|--------|------------------|
| S1 | Переобучить ConvNeXt по **полному README-рецепту** (100 ep, Huber, AdamW, cosine, adv 0.02/0.4) | ↓ MAE, ↓ conformal width |
| S2 | **`best.pt` по val MAE** + conformal на лучшем весе | Стабильнее test / CP |
| S3 | **Group-level evaluation** скрипт (агрегация 4 кадров: mean ŷ, одно y) | Честная оценка |
| S4 | Выделить **calibration.csv** (15–20% групп из train), val только для early stop | Корректнее coverage |
| S5 | Перезапустить `run_conformal.py` с отчётом width/MAE **по группам** | Прозрачность uncertainty |
| S6 | Документировать в репо **фактические** MAE лучших run'ов (таблица в MD из `metrics.csv`) | Восполнить отсутствующий TRAINING_PLAN |

### 5.2. Mid-term (1–2 месяца, архитектура и обучение)

| # | Задача | Ожидаемый эффект |
|---|--------|------------------|
| M1 | **CQR pilot** (dual quantile head + conformal calibration) | Адаптивная ширина, ↓ mean width |
| M2 | Crop-by-mask + лёгкая geom precompute aug | ↑ effective data, лучше текстура |
| M3 | Сравнение **EfficientNet-B0** и **ConvNeXt** по одному протоколу | Выбор production-модели |
| M4 | Улучшение сегментации (морфология / обучаемый сегментатор) | ↓ outliers, стабильнее CP |
| M5 | **Conditional coverage** (по бинам fineness, по sharpness) | Понимание, где модель «не уверена» |
| M6 | Интеграция uncertainty в API/UI (interval + point) | Продуктовая ценность |

### 5.3. Long-term (вне текущего scope, для полноты)

- Active learning на зёрнах с широкими интервалами.
- Multi-task: fineness + particle size distribution (если появятся метки).
- On-device calibration по новой камере/освещению (conformal re-calibration без полного retrain).

---

## 6. Чек-лист — что сделать следующим

### Обучение и метрики

- [ ] Запустить ConvNeXt-Small:  
  `uv run python src/train.py --model convnext_small --unfreeze-after 15 --adamw --cosine-lr --huber --huber-delta 0.03 --lr 3e-4 --epochs 100 --adversarial --adv-epsilon 0.02 --adv-weight 0.4`
- [ ] Добавить/использовать сохранение **лучшего** чекпоинта по val MAE.
- [ ] Собрать таблицу: val MAE, test MAE (group + image), лучшая эпоха.

### Conformal

- [ ] Создать holdout **calibration** split (не val).
- [ ] Перезапустить:  
  `uv run python src/run_conformal.py --run-dir data/runs/convnext_small/run_001 --alpha 0.10`
- [ ] Зафиксировать: coverage, mean width, width/MAE, доля test вне интервала **по группам**.
- [ ] Решить go/no-go на **CQR** по критерию: mean width ↓ ≥25% при coverage ∈ [0.88, 0.92].

### Данные

- [ ] Проверить качество маски на 20 случайных raw→segmentation (визуальный аудит).
- [ ] Опционально: precompute geom-aug segmented (`--no-photometric`) и один прогон `--augmented-data`.

### Документация / процесс

- [ ] Восстановить или создать **`TRAINING_PLAN.md`** с итоговой таблицей экспериментов.
- [ ] PR по ветке `Conformal_regression` с ссылкой на `src/conformal/README.md` и этот roadmap.

---

## 7. Приложение — ключевые артефакты

| Путь | Содержание |
|------|------------|
| `data/runs/convnext_small/run_001/config.json` | Гиперпараметры baseline |
| `data/runs/convnext_small/run_001/metrics.csv` | 30 эпох train/val MSE/MAE |
| `data/runs/convnext_small/run_001/predictions_epoch_030.csv` | Val predictions (MAE ≈ 3.77) |
| `data/runs/conformal_eval/conformal_report.json` | Test MAE 3.24, coverage 96.6%, width 17.2 |
| `src/conformal/README.md` | Операционная документация CP |
| `CONCEPTS.md` | Group split, aug, overfitting |

---

*Документ подготовлен по состоянию репозитория на момент аудита. Численные цели «идеала» следует обновить после прогона полной сетки `run_all_experiments.py` и заполнения TRAINING_PLAN.*
