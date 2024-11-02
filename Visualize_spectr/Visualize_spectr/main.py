import numpy as np
import librosa
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI()

# Подключаем статику (для HTML и других файлов)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def main():
    return FileResponse('static/index.html')

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Загружаем аудиофайл и считываем его
    audio_data, sr = librosa.load(io.BytesIO(await file.read()), sr=None, duration=30)

    # Генерация объединенного графика спектрограммы и звуковой волны
    combined_plot_json = generate_combined_plot(audio_data, sr)

    # Заглушка для обработки MIDI нот
    midi_notes = "MIDI Notes placeholder"

    return {
        "combined_plot": combined_plot_json,
        "midi_notes": midi_notes
    }

def generate_combined_plot(audio_data, sr):
    # Параметры для CQT
    fmin = librosa.note_to_hz('C1')  # Начальная нота
    bins_per_octave = 12*3             # Количество бинов на октаву
    n_bins = 7 * bins_per_octave     # Общее количество бинов (например, 7 октав)
    hop_length = int(sr * 0.011)  # около 11 мс
    # Генерация CQT спектрограммы с указанными параметрами
    CQT = np.abs(librosa.cqt(audio_data, sr=sr, fmin=fmin,
                             n_bins=n_bins, bins_per_octave=bins_per_octave))
    CQT_db = librosa.amplitude_to_db(CQT, ref=np.max)

    # Построение временной оси для waveform
    times = np.arange(len(audio_data)) / sr  # Приведение к секундам вручную

    # Генерация временной оси для CQT
    CQT_times = librosa.times_like(CQT, sr=sr)

    # Получение частот и соответствующих им нот
    frequencies = librosa.cqt_frequencies(n_bins=CQT.shape[0], fmin=fmin,
                                          bins_per_octave=bins_per_octave)
    notes = librosa.hz_to_note(frequencies, octave=True)

    # Используем make_subplots для создания графика с двумя подграфиками
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Waveform', 'CQT Spectrogram'))

    # Добавляем waveform на верхний подграфик
    fig.add_trace(go.Scatter(x=times, y=audio_data, mode='lines', name='Waveform'), row=1, col=1)

    # Добавляем спектрограмму на нижний подграфик с нотами на оси Y
    fig.add_trace(go.Heatmap(z=CQT_db, x=CQT_times, y=notes,
                             colorscale='Viridis', name='Spectrogram'), row=2, col=1)

    # Настройки графика
    fig.update_layout(height=1000, title_text="Waveform and CQT Spectrogram with Notes", showlegend=False)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Notes", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    # Возвращаем график в виде JSON
    return fig.to_json()




if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
