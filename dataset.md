# Данные
https://disk.yandex.ru/d/OwgxFdiQ0JiE9w

# Описание
## Songsterr


### Для сбора данных с сайта сонгстер был написан [скрапер](https://github.com/nairary/Project-77-DL-in-audio-processing/tree/main/Dataset/Songster)


- Songsterr.zip - датасет на 474 трека состоящий из:
- songs_data.json - метаданные по аудио-файлам
- donwloads_audio - скачанные с ютуб аудио
- downloads_midi - скачанные с songster midi - файлы
## Hooktheory
- Hooktheory.json.gz - Содержит весь датасет из 50 часов треков в упрощенном виде без аудиофайлов. Для каждого трека есть название его midi версии, ссылки на ютуб для аудио версии, а также разметка для выравнивания миди/аудио
- Hooktheory_Train_MIDI.tar.gz - 80% от всего датасета midi для обучения модели
- Hooktheory_Valid_MIDI.tar.gz - 10% от всего датасета midi для валидации модели
- Hooktheory_Test_MIDI.tar.gz  - 10% от всего датасета midi для тестирования модели
