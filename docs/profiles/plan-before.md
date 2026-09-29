# План до змін: лабораторна 8

База: b62d459 (lab-07), 503 документи, запит `python async await`.
Знято до редагування алгоритмів: CLI cProfile index/search, py-spy SVG,
Scalene 2.3.0 (30 повторів пошуку, один load). Сирі файли — benchmarks/lab08.

1. CLI load: validate 33.24 с cumulative, from_json 18.27 с під cProfile
   разом із CLI tracemalloc. Гіпотеза: перевірка позицій повторно створює
   MappingProxyType і Posting; кешувати локальні посилання, не прибирати перевірки.
2. matching_spans: 4.79 с cumulative для 10 снипетів у CLI-профілі.
   Гіпотеза: пошук меж символ за символом і повторна нормалізація дорогі;
   перенести сканування Unicode до регулярного виразу, зберегти правила.
3. tokenize/build_state: tokenize 2.27 с cumulative під cProfile index,
   1 557 525 токенів. Гіпотеза: Unicode/regex вже частково нативні;
   NumPy не пришвидшить читання файлів чи нормалізацію тексту.

Scalene (частки всього запуску, Python / native / system, %):
- tokenize: 3.19 / 28.50 / 1.72;
- matching_spans: 1.71 / 16.11 / 1.38;
- from_json: 0.00 / 14.32 / 1.97;
- validate: 3.16 / 6.06 / 0.79.
Це sampled оцінки Windows, а не доказ, що весь рядок написано мовою C.
Значна частка native не дозволяє обіцяти прискорення лише заміною Python-циклу.
Додатково за умовою векторизую скорери; вони НЕ головне вузьке місце
повного HTTP-пошуку. Окремо виміряю scoring-only та повний search.

CLI cProfile із tracemalloc має великий overhead; секунди звідси не можна
порівнювати з непрофільованим сервером. py-spy у Windows запускається через
справжній python.exe, бо venv launcher не містить інтерпретатора.
