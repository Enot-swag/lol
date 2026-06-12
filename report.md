 Отчёт по проекту кредитного скоринга

## Цель
Построить модель прогнозирования дефолта по кредиту на основе датасета "Give Me Some Credit".

## Данные
- 150 000 записей, 11 признаков.
- Пропуски заполнены медианой/модой, выбросы обработаны IQR.

## Модели
| Модель          | Val ROC-AUC | Val Avg Precision |
|----------------|-------------|-------------------|
| Logistic Reg    | 0.832       | 0.68              |
| Random Forest   | 0.851       | 0.71              |
| LightGBM        | 0.868       | 0.74              |
| XGBoost         | 0.865       | 0.73              |
| MLP (PyTorch)   | 0.855       | 0.72              |

Лучшая: LightGBM, сохранена в `models/`.

## API
- `/predict` POST – возвращает вероятность и класс.
- `/health` GET – проверка состояния.
- `/metrics` – метрики Prometheus.

## Наблюдаемость
- JSON-логи всех запросов.
- Счётчики и гистограммы времени ответа.

## Запуск
```bash
docker-compose up --build
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '@sample.json'