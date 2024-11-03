import os
import re
from subprocess import Popen, DEVNULL, PIPE

import numpy as np
import pygame

if os.name == "nt":
    from ctypes import windll
    if hasattr(windll, "shcore"):
        windll.shcore.SetProcessDpiAwareness(1)

SIZE = (720, 720)
DOT_COLOR = (0, 255, 0)
GRID_COLOR = (40, 40, 0)
BG_COLOR = (16, 16, 16)
audio_file = r"24.wav"


class FFMPEG_AudioReader:
    def __init__(self, filename, buffersize):
        self.filename = filename

        cmd = ["ffmpeg", "-hide_banner", "-i", filename]
        popen_params = {"bufsize": 10 ** 5, "stdout": PIPE,
                        "stderr": PIPE, "stdin": DEVNULL}

        proc = Popen(cmd, **popen_params)

        self.duration = 0
        self.sample_rate = 48000
        for line in proc.communicate()[1].decode("utf8", errors="ignore").splitlines()[1:]:
            if line.startswith("  Duration:"):
                time_raw_string = line.split("Duration: ")[-1]
                match_duration = re.search(r"(\d\d:\d\d:\d\d.\d\d)", time_raw_string)
                time = match_duration.group(1)
                if isinstance(time, str):
                    time = [float(part.replace(",", ".")) for part in time.split(":")]
                self.duration = sum(mult * part for mult, part in zip((1, 60, 3600), reversed(time)))

            try:
                self.sample_rate = int(re.search(r" (\d+) Hz", line).group(1))
            except (AttributeError, ValueError):
                pass

        proc.terminate()
        del proc

        self.bufsize = max(int(self.sample_rate * self.duration) + 1, buffersize)
        self.buffer = None
        self.buffer_start_frame = 1
        cmd = ["ffmpeg",
               "-i", self.filename,
               "-vn",
               "-loglevel", "error",
               "-f", "s32le",
               "-c:a", "pcm_s32le",
               "-ar", f"{self.sample_rate:d}",
               "-ac", "2",
               "-"]
        popen_params = {"bufsize": self.bufsize, "stdout": PIPE,
                        "stderr": PIPE, "stdin": DEVNULL, }

        self.proc = Popen(cmd, **popen_params)

        self.pos = 0
        self.buffer_around(1)

    def buffer_around(self, frame_number):
        new_buffer_start = max(0, frame_number - self.bufsize // 2)

        self.pos = new_buffer_start

        chunk_size = int(round(self.bufsize))
        s = self.proc.stdout.read(chunk_size * 8)
        result = (np.frombuffer(s, dtype=np.int32) / 2147483648).reshape((-1, 2))

        pad = np.zeros((chunk_size - len(result), 2), dtype=result.dtype)
        self.buffer = np.concatenate((result, pad))
        self.pos += chunk_size

        self.buffer_start_frame = new_buffer_start


class AudioFileClip:
    def __init__(self, filename):
        self.reader = FFMPEG_AudioReader(filename, buffersize=96000)
        self.sample_rate = self.reader.sample_rate

    def iter_chunks(self, chunk_size=None, quantize=False):
        total_size = int(self.sample_rate * self.reader.duration)

        nchunks = total_size // chunk_size + 1

        positions = np.linspace(0, total_size, nchunks + 1, endpoint=True, dtype=int)

        for i in range(nchunks):
            timings = (1.0 / self.sample_rate) * np.arange(positions[i], positions[i + 1])
            yield self.to_soundarray(timings, quantize=quantize, buffer_size=chunk_size)

    def to_soundarray(self, tt=None, quantize=False, buffer_size=96000):
        if tt is None:
            max_duration = 1 * buffer_size / self.sample_rate
            if self.reader.duration > max_duration:
                return np.vstack(tuple(self.iter_chunks(chunk_size=buffer_size, quantize=quantize)))

        in_time = (tt >= 0) & (tt < self.reader.duration)

        result = np.zeros((len(tt), 2))
        indices = np.round((self.sample_rate * tt)).astype(int)[in_time] - self.reader.buffer_start_frame
        result[in_time] = self.reader.buffer[indices]
        return result


def get_grid(size) -> np.ndarray:
    grd = pygame.Surface(size)
    grd.fill(BG_COLOR)

    for horizontal in range(10):
        pygame.draw.line(grd, GRID_COLOR, (horizontal * SIZE[0] / 10, 0), (horizontal * SIZE[0] / 10, SIZE[0]))
    for vertical in range(10):
        pygame.draw.line(grd, GRID_COLOR, (0, vertical * SIZE[1] / 10), (SIZE[0], vertical * SIZE[1] / 10))

    pygame.draw.line(grd, GRID_COLOR, (SIZE[0] / 2, 0), (SIZE[0] / 2, SIZE[0]), 3)
    pygame.draw.line(grd, GRID_COLOR, (0, SIZE[1] / 2), (SIZE[0], SIZE[1] / 2), 3)

    for horizontal in range(100):
        pygame.draw.line(grd, GRID_COLOR, (horizontal * SIZE[0] / 100, SIZE[1] / 2 - 3),
                         (horizontal * SIZE[0] / 100, SIZE[1] / 2 + 3))
    for vertical in range(100):
        pygame.draw.line(grd, GRID_COLOR, (SIZE[0] / 2 - 3, vertical * SIZE[1] / 100),
                         (SIZE[0] / 2 + 3, vertical * SIZE[1] / 100))
    return pygame.surfarray.pixels2d(grd)


file = AudioFileClip(audio_file)
sr = file.sample_rate
audio = file.to_soundarray()

READ_TIME = 20  # milliseconds
READ_LENGTH = int(READ_TIME / 1000 * sr)

pygame.init()

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Oscilloscope", "示波器")
grid = get_grid(SIZE)

pygame.mixer.music.load(audio_file)
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(loops=1)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    pygame.surfarray.blit_array(screen, grid)

    end = int(pygame.mixer_music.get_pos() / 1000 * sr)
    frames = audio[end - READ_LENGTH:end]
    for left, right in frames:
        x = int(right * SIZE[0] / 2) + SIZE[0] // 2
        y = int(-left * SIZE[1] / 2) + SIZE[1] // 2
        screen.set_at((y, x), DOT_COLOR)

    pygame.display.flip()
