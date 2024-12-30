import os
import json
import pretty_midi

# Ключевые слова для распознавания вокала
CURRENT_DIRECTORY = os.getcwd()
VOCAL_KEYWORDS = ["vocal", "vocals", "voice", "voices"]
LEAD_KEYWORD = "lead"

def load_match_scores(match_scores_path):
    """
    Загружает match-scores.json, имеющий структуру:
    {
      "TRWMHMP128EF34293F": {
        "c3da6699f64da3db8e523cbbaa80f384": 0.73212455227411,
        "d8392424ea57a0fe6f65447680924d37": 0.747619664919494
      },
      "TRWOLRE128F427D710": {
        "728c3dbdc9b47142dc8f725c6805c259": 0.672055071290478,
        ...
      },
      ...
    }
    Возвращает словарь вида:
    {
      "TRWMHMP128EF34293F": "d8392424ea57a0fe6f65447680924d37",
      "TRWOLRE128F427D710": "468be2f5dd31a1ba444b8018d8e8c7ad",
      ...
    }
    где каждому MSD ID сопоставлен MD5, у которого score максимален.
    """
    with open(match_scores_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    best_md5_for_id = {}
    for msd_id, md5_dict in data.items():
        best_md5 = None
        best_score = float('-inf')
        for md5_val, score in md5_dict.items():
            if score > best_score:
                best_score = score
                best_md5 = md5_val
        best_md5_for_id[msd_id] = best_md5

    return best_md5_for_id


def get_subfolders_from_msd_id(msd_id: str):
    """
    Из MSD ID, например 'TRWOLRE128F427D710', извлекает:
      - Первые три буквы/символа после 'TR' (W, O, L).
      - Возвращает список ['W', 'O', 'L'].
    """
    # Допустим, MSD ID всегда начинается с 'TR'
    sub = msd_id[2:5]  # три символа после 'TR'
    return list(sub)   # например, 'WOL' -> ['W','O','L']


def build_midi_path(lmd_aligned_root, msd_id, md5):
    """
    Генерирует путь к MIDI-файлу внутри lmd_aligned.
    Например, если msd_id = 'TRWOLRE128F427D710' и md5='468be2f5dd31a1ba444b8018d8e8c7ad', то:
      - папка = lmd_aligned / W / O / L / TRWOLRE128F427D710
      - файл  = 468be2f5dd31a1ba444b8018d8e8c7ad.mid
    """
    subfolders = get_subfolders_from_msd_id(msd_id)  # ['W','O','L']
    return os.path.join(
        lmd_aligned_root,
        *subfolders,         # W\O\L
        msd_id,              # TRWOLRE128F427D710
        f"{md5}.mid"         # 468be2f5dd31a1ba444b8018d8e8c7ad.mid
    )


def replicate_path_in_vocals(midi_path, lmd_aligned_root, lmd_aligned_vocals_root):
    """
    Превращает путь, начинающийся с lmd_aligned_root, в путь, начинающийся с lmd_aligned_vocals_root.
    Например:
      D:\...\lmd_aligned\W\O\L\TRWOLRE128F427D710\468be2f5dd31a1ba444b8018d8e8c7ad.mid
    станет
      D:\...\lmd_aligned_vocals\W\O\L\TRWOLRE128F427D710\468be2f5dd31a1ba444b8018d8e8c7ad.mid
    """
    # Отрежем часть пути, совпадающую с lmd_aligned_root
    rel_path = os.path.relpath(midi_path, start=lmd_aligned_root)
    # Заменим корень
    return os.path.join(lmd_aligned_vocals_root, rel_path)


def find_vocal_instrument(pm: pretty_midi.PrettyMIDI):
    """
    Ищет среди pm.instruments ту, которая:
      1) Имеет непустое name (название дорожки).
      2) В name есть "vocal|vocals|voice|voices", иначе ищем "lead".
      3) Если несколько подходят, берём ту, где больше всего нот.
    Если вообще ни одна не подходит, вернём None.
    """
    candidates = []
    for inst in pm.instruments:
        if not inst.name:  # нет названия - пропускаем
            continue
        name_lower = inst.name.lower()

        # Проверяем "vocal|vocals|voice|voices"
        is_vocal = any(kw in name_lower for kw in VOCAL_KEYWORDS)
        if not is_vocal:  
            # Если нет слов 'vocal...', смотрим 'lead'
            if LEAD_KEYWORD in name_lower:
                is_vocal = True

        if is_vocal:
            # Посчитаем кол-во нот, где velocity>0
            note_count = sum(1 for n in inst.notes if n.velocity > 0)
            candidates.append((inst, note_count))

    if not candidates:
        return None

    # Сортируем по кол-ву нот (убывающий порядок)
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_inst = candidates[0][0]  # тот, у кого больше всего нот
    return best_inst


def create_vocal_pretty_midi(original_pm: pretty_midi.PrettyMIDI, vocal_inst: pretty_midi.Instrument):
    """
    Создаёт новый PrettyMIDI, где только одна дорожка (vocal_inst).
    Копируем resolution, и (опционально) при большом желании можем вручную
    перенести основные темпы, но не лезем во внутренние поля.
    """
    # 1) Создаём новый объект с той же resolution
    new_pm = pretty_midi.PrettyMIDI(resolution=original_pm.resolution)

    # 2) (Необязательно) Переносим какие-то глобальные свойства, например, первый темп
    # tempos, times = original_pm.get_tempo_changes()
    # if len(tempos) > 0:
    #     # Можно просто установить начальный темп:
    #     new_pm._set_tempo(tempos[0], 0)  # Но это тоже приватный метод, возможно не всегда доступен

    # 3) Создаём инструмент
    new_inst = pretty_midi.Instrument(
        program=vocal_inst.program,
        is_drum=vocal_inst.is_drum,
        name=vocal_inst.name
    )
    new_inst.notes = vocal_inst.notes[:]
    new_inst.control_changes = vocal_inst.control_changes[:]
    new_pm.instruments.append(new_inst)

    return new_pm



def process_vocals(
    match_scores_path, 
    lmd_aligned_root, 
    lmd_aligned_vocals_root
):
    """
    1) Считываем match-scores.json.
    2) Для каждого MSD ID выбираем MD5 с максимальным score.
    3) Строим путь <lmd_aligned_root>/<subfolders>/<MSD_ID>/<md5>.mid
    4) С помощью pretty_midi парсим .mid:
       - Если нет track.name => пропускаем
       - Если нет вокальных ключевых слов => пропускаем
       - Если несколько вокальных треков => берём с макс. кол-вом нот.
    5) Создаём новый MIDI (только вокальная дорожка).
    6) Сохраняем зеркально в <lmd_aligned_vocals_root>/<subfolders>/<MSD_ID>/<md5>.mid
    """

    # Шаг 1: загружаем словарь "MSD_ID -> (Best) MD5"
    best_md5_map = load_match_scores(match_scores_path)

    for msd_id, best_md5 in best_md5_map.items():
        if not best_md5:
            # не нашли ни одного md5
            continue

        # Шаг 3: строим путь к .mid
        midi_path = build_midi_path(lmd_aligned_root, msd_id, best_md5)
        if not os.path.isfile(midi_path):
            print(f"[SKIP] Файл не найден: {midi_path}")
            continue

        # Шаг 4: парсим MIDI и ищем вокальную дорожку
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"[SKIP] Ошибка чтения '{midi_path}': {e}")
            continue

        vocal_inst = find_vocal_instrument(pm)
        if vocal_inst is None:
            print(f"[SKIP] Нет вокальной дорожки в '{midi_path}'")
            continue

        # Шаг 5: создаём MIDI, где только эта дорожка
        new_pm = create_vocal_pretty_midi(pm, vocal_inst)

        # Шаг 6: сохраняем в зеркальную структуру
        out_midi_path = replicate_path_in_vocals(midi_path, lmd_aligned_root, lmd_aligned_vocals_root)
        os.makedirs(os.path.dirname(out_midi_path), exist_ok=True)
        new_pm.write(out_midi_path)

        print(f"[OK]  {midi_path} -> {out_midi_path}")


# Пример использования:
if __name__ == "__main__":
    MATCH_SCORES_JSON = os.path.join(CURRENT_DIRECTORY, "match-scores.json")

    # Папки:
    LMD_ALIGNED_ROOT = os.path.join(CURRENT_DIRECTORY, "lmd_aligned")
    LMD_ALIGNED_VOCALS_ROOT = os.path.join(CURRENT_DIRECTORY, "lmd_aligned_vocals")

    process_vocals(MATCH_SCORES_JSON, LMD_ALIGNED_ROOT, LMD_ALIGNED_VOCALS_ROOT)
