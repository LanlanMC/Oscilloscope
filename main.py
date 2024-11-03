import os
import re
import subprocess

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
    def __init__(self, filename, buffersize, fps=44100):
        self.filename = filename
        self.nbytes = 4
        self.fps = fps
        self.format = "s32le"
        self.codec = "pcm_s32le"

        cmd = ["ffmpeg", "-hide_banner", "-i", filename]

        popen_params = {
            "bufsize": 10 ** 5,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
        }

        proc = subprocess.Popen(cmd, **popen_params)
        (output, error) = proc.communicate()

        self.duration = 0
        self.sample_rate = "unknown"
        for line in error.decode("utf8", errors="ignore").splitlines()[1:]:
            if line.startswith("  Duration:"):
                time_raw_string = line.split("Duration: ")[-1]
                match_duration = re.search(r"([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", time_raw_string)
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

        self.buffersize = min(int(self.fps * self.duration) + 1, buffersize)
        self.buffer = None
        self.buffer_startframe = 1
        cmd = ["ffmpeg",
               "-i", self.filename,
               "-vn",
               "-loglevel", "error",
               "-f", self.format,
               "-c:a", self.codec,
               "-ar", f"{self.fps:d}",
               "-ac", "2",
               "-"]
        popen_params = {"bufsize": self.buffersize, "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE, "stdin": subprocess.DEVNULL, }

        self.proc = subprocess.Popen(cmd, **popen_params)

        self.pos = 0
        self.buffer_around(1)

    def read_chunk(self, chunksize):
        chunksize = int(round(chunksize))
        s = self.proc.stdout.read(2 * chunksize * self.nbytes)
        data_type = {1: "int8", 2: "int16", 4: "int32"}[self.nbytes]
        result = np.frombuffer(s, dtype=data_type)
        result = ((1.0 * result / 2 ** (8 * self.nbytes - 1))
                  .reshape((int(len(result) / 2), 2)))

        pad = np.zeros((chunksize - len(result), 2), dtype=result.dtype)
        result = np.concatenate((result, pad))
        self.pos = self.pos + chunksize
        return result

    def get_frame(self, tt):
        in_time = (tt >= 0) & (tt < self.duration)

        frames = np.round((self.fps * tt)).astype(int)[in_time]
        fr_min, fr_max = frames.min(), frames.max()

        if not (0 <= (fr_max - self.buffer_startframe) < len(self.buffer)):
            self.buffer_around(fr_max)

        result = np.zeros((len(tt), 2))
        indices = frames - self.buffer_startframe
        try:
            result[in_time] = self.buffer[indices]
            return result
        except IndexError:
            indices[indices >= len(self.buffer)] = len(self.buffer) - 1
            result[in_time] = self.buffer[indices]
            return result

    def buffer_around(self, frame_number):
        new_bufferstart = max(0, frame_number - self.buffersize // 2)

        if self.buffer is not None:
            current_f_end = self.buffer_startframe + self.buffersize
            conserved = current_f_end - new_bufferstart
            chunksize = self.buffersize - conserved
            array = self.read_chunk(chunksize)
            self.buffer = np.vstack([self.buffer[-conserved:], array])
        else:
            self.pos = new_bufferstart
            self.buffer = self.read_chunk(self.buffersize)

        self.buffer_startframe = new_bufferstart

    def __del__(self):
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait()
            self.proc = None


class AudioFileClip:
    def __init__(self, filename, fps=48000):
        self.reader = FFMPEG_AudioReader(filename, buffersize=20_0000, fps=fps)
        self.fps = fps

    def iter_chunks(self, chunksize=None, quantize=False):
        total_size = int(self.fps * self.reader.duration)

        nchunks = total_size // chunksize + 1

        positions = np.linspace(0, total_size, nchunks + 1, endpoint=True, dtype=int)

        for i in range(nchunks):
            timings = (1.0 / self.fps) * np.arange(positions[i], positions[i + 1])
            yield self.to_soundarray(timings, quantize=quantize, buffersize=chunksize)

    def to_soundarray(self, tt=None, quantize=False, buffersize=50000):
        if tt is None:
            max_duration = 1 * buffersize / self.fps
            if self.reader.duration > max_duration:
                return np.vstack(tuple(self.iter_chunks(chunksize=buffersize, quantize=quantize)))
        snd_array = self.reader.get_frame(tt)

        return snd_array


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
sr = file.fps
audio = file.to_soundarray()

READ_LENGTH = 1024

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
