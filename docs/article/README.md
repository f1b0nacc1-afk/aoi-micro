# Article draft (LaTeX)

Информационная база / черновик статьи по **aoi-micro**.

## Сборка

```bash
cd docs/article
pdflatex main.tex
pdflatex main.tex   # второй проход — оглавление
```

Или: `latexmk -pdf main.tex`

## Содержимое

| Файл | Назначение |
|---|---|
| `main.tex` | текст: мотивация, методы, стенд, результаты classical + NN, ограничения |
| `figures/` | копии из `nn_results/det_v0/` и preview стенда |

Когда будут требования редакции — меняем преамбулу/`\documentclass`, переносим разделы.
