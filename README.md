# AOI плат: одна верхняя mono-камера IMX183 + Computar 12 mm

Прототип CAD-driven AOI для Basler acA5472-17um / Hikrobot MV-CE200-10UM. Python + OpenCV, без нейросети.
Отчёты по камерам — в `basler-aca5472-17um-report/` и `hikrobot-mv-ce200-10um-report/`.

## Метод

Классическая схема промышленных AOI (Omron/Koh Young/Viscom для 2D-этапа) и
variation model из MVTec HALCON, собранная на OpenCV:

1. **Выравнивание.** Фидуциалы → преобразование в канонический кадр платы (фиксированные px/мм).
   2 фидуциала = подобие (сдвиг/поворот/масштаб), 3 = аффинное, ≥4 = гомография.
   Затем ECC-уточнение по среднему golden и выравнивание низкочастотной освещённости.
2. **Окна из CAD.** Centroid (KiCad `.pos`, Altium Pick&Place, любой CSV) + библиотека корпусов →
   окно на каждую позицию. Центр окна сдвигается на **параллакс** высоты детали: верх корпуса
   высотой h на расстоянии r от оси смещается на `r·h/(WD−h)`.
3. **Golden 20–30 плат.** По каждой позиции: среднее и σ на пиксель в системе координат корпуса,
   распределения NCC и фотометрической разницы корпуса (MAD). Пороги берутся из статистики, не руками.
4. **Проверка позиции:**
   - поиск корпуса шаблоном (NCC) в окне ±0.8 мм, ±20° → сдвиг, поворот;
   - корпус делится на зоны: **внутренняя** (маркировка, тон партии) проверяется только медианой
     с допуском, **рамка** (выводы, кромки) — MAD после компенсации сдвига яркости. Так новая партия
     с другим оттенком и маркировкой не даёт ложных MISMATCH;
   - наличие = NCC **и** MAD рамки **и** тон внутренней зоны; только NCC не работает: он инвариантен
     к контрасту, и голые площадки с зазором «похожи» на чип;
   - полярность: golden-метка (pin 1, полоса диода, «−» электролита) ищется во всех 4 (квадратные корпуса)
     или 2 ориентациях; запасной тест — перевёрнутый шаблон целиком;
   - сдвиг больше допуска (`shift_tol_frac`·ширина, не более `shift_tol_max_mm`) при уверенном совпадении — SHIFT;
   - пайка/посторонние: отклонение за **огибающую** golden (dilate/erode среднего на `edge_tol_mm`)
     в единицах σ; плюс доля отклоняющихся пикселей в зоне выводов (`joint_frac`) — непропай;
   - сколы/царапины на верхней грани — толстые или очень сильные отклонения во внутренней зоне.
5. **Плата вне деталей:** тот же variation model по всей плате с маской деталей —
   посторонние предметы, шарики, замыкания/обрывы дорожек.
6. **Отчёт:** PASS/FAIL по позициям, кропы живое | эталон | карта отклонений, CSV/JSON.

## Быстрый старт (без стенда)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m aoi optics --wd 100 150 200            # поле, мкм/px, ГРИП, параллакс
.venv/bin/python -m aoi synth --out demo                    # синтетические golden + good/bad
.venv/bin/python -m aoi train   -c demo/config.json --images demo/golden --golden demo/golden_model
.venv/bin/python -m aoi inspect -c demo/config.json --images demo/test --golden demo/golden_model --out demo/reports
open demo/reports/bad/report.html
.venv/bin/python -m unittest tests.test_pipeline
```

## На стенде

1. Оптика: f/4, фокус и диафрагма законтрены, экспозиция фиксированная, auto выключено.
   Свет — кольцо/купол; коаксиал на поле 126 мм с углом ±29° не работает.
2. `cp config.example.json config.json`, в нём `camera.backend` = `basler` (нужен `pypylon`) или
   `hikrobot` (MVS SDK, `mvs_import_path` на `Samples/Python/MvImport`).
   `optics.wd_mm` = расстояние от платы до передней линзы + 27 мм (главная плоскость V1226-MPZ).
3. Снять кадр: `python -m aoi capture -c config.json`, затем
   `python -m aoi fiducials -c config.json captures/<файл>.png` → вписать `approx_px` фидуциалов.
4. Проверить `cad_y_up` / `cad_rot_ccw` на первой golden: рамки в `overview.jpg` должны лечь на детали.
   Корпуса не из встроенной библиотеки — в `packages.json` (пример `packages.example.json`),
   зону метки проверить по жёлтой рамке на кропе эталона.
5. 20–30 заведомо хороших плат: `capture --n 25 --prefix golden --out golden_raw`, затем `train`.
6. Поток: `python -m aoi run -c config.json` (снять + проверить + отчёт).

## Тестовый стенд (синтетика, близкая к реальной плате)

`aoi/bench/` — процедурные платы 64–80 мм с маской, дорожками, полигонами, переходными, HASL/ENIG,
20 типов корпусов (0402…1206, SOD/SOT, SOIC/TSSOP/TQFP/LQFP/QFN, электролиты, тантал, LED, кварц, дроссель),
параллакс высоких деталей, расфокус, виньетирование, поза и перспектива камеры, шум сенсора.
Две партии компонентов (A — golden, B — другие оттенки, маркировка и дата-коды). 15 типов дефектов
с разметкой: missing, shift, rotate, polarity, rot90, wrong, tombstone, billboard, damaged, no_solder,
excess, bridge, ball, foreign, marking.

```bash
.venv/bin/python -m aoi bench --tag my_change                 # ~2.5 мин: 4 дизайна, 89 дефектов
.venv/bin/python -m aoi bench --tag holdout --seed 7          # другие дизайны — проверка на переобучение
.venv/bin/python -m aoi bench --tag try --override '{"var_k": 4}'
.venv/bin/python -m aoi.bench.analyze bench/runs/<прогон>      # разбор: пропуски и ложные с метриками
```

Каждый прогон пишет `bench/runs/<время>_<tag>/index.html` (дашборд с кропами пропусков и ложных),
отчёты по каждой плате и дописывает строку в `bench/RESULTS.md`.

Текущий результат (v6, подробности в `bench/RESULTS.md`): обнаружение 93.3%, верный класс 87.6%
(84.3% на отложенных дизайнах), брак как PASS — 0, ложные позиции 0.6–1.4% (baseline v0: 76.4% / 59.6%,
61% ложных на новой партии).

## Калибровка камеры (ChArUco)

```bash
python -m aoi calib-target --squares 12 8 --square-mm 8 --marker-mm 6 --out calib_target.png
# печать 100 %, клетку измерить штангенциркулем; 15–25 кадров по всему полю, наклон до ~10–15°
# (ГРИП на f/4 — единицы мм), последний кадр — мишень плашмя на плоскости платы
python -m aoi capture -c config.json --n 20 --prefix calib --out calib_raw
python -m aoi calibrate --images calib_raw --square-mm 7.98 --out calib/camera.npz
```

Пишет `camera.npz` (K, D — подхватывается через `optics.undistort_npz`) и `camera.json`: RMS
перепроецирования (норма < 0.3 px), ошибки по кадрам, WD до главной плоскости и px/мм по плоскому кадру —
готовые значения для `optics.wd_mm` и `board.raw_px_per_mm`. Проверено на синтетике (`tests/test_calib.py`):
fx/cx/cy и k1 восстанавливаются с точностью < 0.5 %, WD ±0.7 мм.

## Нейросетевой детектор компонентов (эксперимент, `aoi/nn/`)

YOLO11s на 11 классов: resistor, capacitor, electrolytic, ic, transistor, diode, led, inductor, crystal,
footprint (пустое посадочное место), other. Обучается на открытых датасетах (Roboflow circuit-elements,
smd-components, переведены в mono) и на размеченных кадрах синтетического стенда (та же камера, 43 px/мм).

```bash
pip install ultralytics huggingface_hub   # + torch с CUDA
python -c "from huggingface_hub import snapshot_download as d; d('LibreYOLO/circuit-elements', repo_type='dataset', local_dir='data/raw/circuit-elements'); d('gatilin/rf100-vl', repo_type='dataset', local_dir='data/raw/rf100-vl', allow_patterns=['smd-components/*'])"
python -m aoi.nn.prep_public data/raw/circuit-elements data/raw/rf100-vl/smd-components --out data/yolo/public
python -m aoi.nn.synth_export --out data/yolo/synth
python -m aoi.nn.train --data data/yolo/public data/yolo/synth --name det_v0
python -m aoi.nn.evaluate runs/nn/det_v0/weights/best.pt     # mAP по источникам + целые кадры стенда
```

На линии: `python -m aoi detect -c config.json --nn best.pt` — только детектор, без CAD (файлы или кадр
с камеры, масштаб приводится к 43 px/мм по `raw_px_per_mm`); `inspect`/`run` с `--nn best.pt` — второе
мнение по каждой CAD-позиции: `NN_EMPTY` (место пустое) и `NN_CLASS` (не тот тип детали). По умолчанию
только пометки в отчёте; `inspect.nn_veto = true` делает их браком.

Результат det_v0 (40 эпох, RTX 3080, ~65 мин; веса и графики в `nn_results/det_v0/`): синтетика стенда
mAP50 0.93; на целых кадрах 6 новых дизайнов найдено 100% деталей, класс верный 98.7–100%, отсутствующая
деталь замечена в 91.7%, подмена R↔C — 100%, 0.36 с на кадр. Публичные фото — mAP50 0.38: IC/LED/пустые места
хорошо, мелкие R/C/диоды плохо (крошечные на снимках целых плат). Для реальной камеры нужно дообучение на
своих кадрах; пустых мест в синтетике мало (набор `--missing 6` для следующей версии).

## Границы

- Бэкенды камер написаны по pylon/MVS API, но на железе не проверены.
- `edge_tol_mm` ≥ 2–3σ разброса установки и < половины зазора между выводами.
  При 23 мкм/px (WD 100) мост на шаге 0.8 мм ловится; на 45 мкм/px — уже нет.
- Нет: объём галтели, поднятый вывод, BGA/QFN под корпусом, цвет, OCR (следующий этап).
- По стенду слабые места: сколы на светлых выводах чипов (нужен косой свет), подмена ИС того же корпуса
  (только OCR), класс дефектов SOT-23 и избытка припоя; ложные позиции ~1% — цель для линии <0.2%.
