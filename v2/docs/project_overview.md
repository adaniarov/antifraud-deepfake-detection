# v2/docs/project_overview.md

## Project summary
Каталог `v2/` — трек **Core-датасета** для той же бакалаврской работы по обнаружению LLM-сгенерированного текста в антифрод-контексте. От корневого (legacy) пайплайна отличается **ужесточённой чистотой**: в Core не попадают широкие spam-пулы как fraud-классы, сценарии human ↔ LLM **согласованы** по `scenario_family`, добавлены отдельные fraud-email семейства (phishing vs advance-fee scam) и единая SMS fraud policy на двух источниках.

## Goal
Собрать Core, обучить и сравнить детекторы (classical + transformer), оценить **обобщение на held-out generator** (Claude-family, test-only) и задокументировать ограничения.

## Main components
1. Подготовка human-side по `dataset_design_final.md` (очистка, фильтры, подтипы, аудиты ham).
2. Генерация LLM-side: 5 prompt families + готовая пара HC3 Finance.
3. Симметричный preprocessing и сборка train/val/test без утечки holdout.
4. Baselines: классические модели и трансформеры.
5. Отчётность по партициям и срезам осей.
6. По мере необходимости — дополнительные stress-test источники **вне** Core (не смешивать с основной оценкой без явного решения).

## Research question
Какие методы дают лучший компромисс между качеством in-domain, переносимостью на **невидимый генератор** и устойчивостью к артефактам источника при **Core** постановке задачи?

## Main task
Бинарная классификация:
- `0` = human
- `1` = LLM

Сопутствующие оси для контроля смещений: `fraudness`, `channel`, `scenario_family`, `time_band`, `length_bin`, `source_family`, `origin_model`.

## Domain (Core scope)
- Phishing email (Nazario anchor).
- Advance-fee / 419-style scam email (Nigerian Fraud anchor) — **отдельно** от phishing.
- Децептивные fraud SMS (SmishTank + фильтрованный Mendeley smishing) по единой подтиповой политике.
- Legitimate email и SMS (ham-корпуса).
- Финансовый QA control slice (HC3 Finance human / chatgpt).

Явно **вне** Core (см. §3.4 дизайна): legacy T3/T5/T6, broad generic spam как fraud, отдельные вспомогательные корпуса — только auxiliary / transfer, не как часть итогового Core.

## Evaluation focus
- **Главные метрики** (в т.ч. в тексте ВКР) — **отдельно** для **val**, **test_non_claude** (`core_eval_slice=test_seen`, сопоставимый human vs **seen** LLM), **test_claude_only** (holdout Claude, stress-test). См. `v2/docs/project_status.md` (раздел про протокол метрик).
- **test_full** (union test_seen ∪ test_claude_only) — только **дополнительный** агрегированный срез; не подменять им единственную «главную» цифру качества.
- Разрезы по `scenario_family`, `channel`, `fraudness`.
- В ВКР явно: main comparable pool **balanced human vs seen_llm**; Claude — отдельный holdout; Core — **multi-genre** бенчмарк; **temporal bias** и ограничения генераторов; **SMS fraud** data-limited (см. `project_status.md`, `dataset_design_final.md` §7.2).

## Key deliverables
- Замороженный Core manifest и статистика.
- Воспроизводимый preprocessing и генерация.
- Baseline-результаты на Core (отдельно от legacy v1).
- Текст диссертации с явным разделением **Core** vs дополнительные эксперименты.

## Relation to repo root
Корневой `docs/` и `data/final/` относятся к **предыдущей** сборке и baseline; они полезны как методологический прецедент, но **не** являются источником истины для размеров и метрик Core v2 до явной заморозки v2.
