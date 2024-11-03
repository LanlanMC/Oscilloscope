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


# _ = tk.Tk()
# _.geometry("1x1+512+512")
# _.iconify()
# audio_file = askopenfilename(filetypes=(("波形文件", ".wav"),))
# _.destroy()
# if not os.path.isfile(audio_file):
#     exit()

class FFmpegInfosParser:
    def __init__(self, infos, filename, fps_source="fps"):
        self.infos = infos
        self.filename = filename
        self.fps_source = fps_source
        self.duration_tag_separator = "Duration: "

        self._reset_state()

    def _reset_state(self):
        self._inside_file_metadata = False

        self._inside_output = False

        self._default_stream_found = False

        self._current_input_file = {"streams": []}
        self._current_stream = None
        self._current_chapter = None

        self.result = {
            "video_found": False,
            "audio_found": False,
            "metadata": {},
            "inputs": [],
        }

        self._last_metadata_field_added = None

    def parse(self):
        input_chapters = []

        for line in self.infos.splitlines()[1:]:
            if line.startswith("  Duration:"):
                self._inside_file_metadata = False
                if self.duration_tag_separator == "Duration: ":
                    self.result["duration"] = self.parse_duration(line)

                # parse global bitrate (in kb/s)
                bitrate_match = re.search(r"bitrate: (\d+) kb/s", line)
                self.result["bitrate"] = int(bitrate_match.group(1)) if bitrate_match else None

                # parse start time (in seconds)
                start_match = re.search(r"start: (\d+\.?\d+)", line)
                self.result["start"] = float(start_match.group(1)) if start_match else None
            elif line.lstrip().startswith("Stream "):
                main_info_match = re.search(
                    r"^Stream\s#(\d+):(\d+)(?:\[\w+])?\(?(\w+)?\)?:\s(\w+):",
                    line.lstrip()
                )
                input_number, stream_number, language, stream_type = main_info_match.groups()
                input_number = int(input_number)
                stream_number = int(stream_number)
                stream_type_lower = stream_type.lower()

                self._current_stream = {
                    "input_number": input_number,
                    "stream_number": stream_number,
                    "stream_type": stream_type_lower,
                    "language": language,
                    "default": not self._default_stream_found or line.endswith("(default)")
                }
                self._default_stream_found = True

                # for default streams, set their numbers globally, so it's
                # easy to get without iterating all
                if self._current_stream["default"]:
                    self.result[f"default_{stream_type_lower}_input_number"] = input_number
                    self.result[f"default_{stream_type_lower}_stream_number"] = stream_number

                if "input_number" not in self._current_input_file:
                    self._current_input_file["input_number"] = input_number
                try:
                    global_data, stream_data = self.parse_data_by_stream_type(stream_type, line)
                except NotImplementedError:
                    pass
                else:
                    self.result.update(global_data)
                    self._current_stream.update(stream_data)

        if self._current_input_file:
            self._current_input_file["streams"].append(self._current_stream)
            self.result["inputs"].append(self._current_input_file)

        self.result["video_n_frames"] = 1
        self.result["video_duration"] = None

        result = self.result

        # reset state of the parser
        self._reset_state()

        return result

    def parse_data_by_stream_type(self, stream_type, line):
        """Parses data from "Stream ... {stream_type}" line."""
        try:
            return {
                "Audio": self.parse_audio_stream_data,
                "Video": lambda _line: ({}, {}),
                "Data": lambda _line: ({}, {})
            }[stream_type](line)
        except KeyError:
            raise NotImplementedError(f"{stream_type} stream parsing is not supported and will be ignored")

    def parse_audio_stream_data(self, line):
        """Parses data from "Stream ... Audio" line."""
        global_data, stream_data = ({"audio_found": True}, {})
        try:
            stream_data["fps"] = int(re.search(r" (\d+) Hz", line).group(1))
        except (AttributeError, ValueError):
            stream_data["fps"] = "unknown"
        match_audio_bitrate = re.search(r"(\d+) kb/s", line)
        stream_data["bitrate"] = (
            int(match_audio_bitrate.group(1)) if match_audio_bitrate else None
        )
        if self._current_stream["default"]:
            global_data["audio_fps"] = stream_data["fps"]
            global_data["audio_bitrate"] = stream_data["bitrate"]
        return global_data, stream_data

    def parse_duration(self, line):
        """Parse the duration from the line that outputs the duration of
        the container.
        """
        try:
            time_raw_string = line.split(self.duration_tag_separator)[-1]
            match_duration = re.search(
                r"([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])",
                time_raw_string,
            )

            def convert_to_seconds(time):
                factors = (1, 60, 3600)

                if isinstance(time, str):
                    time = [float(part.replace(",", ".")) for part in time.split(":")]

                if not isinstance(time, (tuple, list)):
                    return time

                return sum(mult * part for mult, part in zip(factors, reversed(time)))

            return convert_to_seconds(match_duration.group(1))
        except Exception:
            raise IOError(f"""Failed to read the duration of file '{self.filename}'.
Here are the file infos returned by ffmpeg:

{self.infos}""")


def ffmpeg_parse_infos(filename, check_duration=True, fps_source="fps"):
    # Open the file in a pipe, read output
    cmd = ["ffmpeg", "-hide_banner", "-i", filename]

    popen_params = {
        "bufsize": 10 ** 5,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL,
    }

    proc = subprocess.Popen(cmd, **popen_params)
    (output, error) = proc.communicate()
    infos = error.decode("utf8", errors="ignore")

    proc.terminate()
    del proc

    try:
        return FFmpegInfosParser(infos, filename, fps_source=fps_source).parse()
    except Exception as exc:
        if os.path.isdir(filename):
            raise IsADirectoryError(f"'{filename}' is a directory")
        elif not os.path.exists(filename):
            raise FileNotFoundError(f"'{filename}' not found")
        raise IOError(f"Error passing `ffmpeg -i` command output:\n\n{infos}") from exc


class FFMPEG_AudioReader:
    def __init__(self, filename, buffersize, fps=44100, nbytes=2, nchannels=2):
        self.filename = filename
        self.nbytes = nbytes
        self.fps = fps
        self.format = "s%dle" % (8 * nbytes)
        self.codec = "pcm_s%dle" % (8 * nbytes)
        self.nchannels = nchannels
        infos = ffmpeg_parse_infos(filename)
        self.duration = infos["duration"]
        self.bitrate = infos["audio_bitrate"]
        self.infos = infos
        self.proc = None

        self.n_frames = int(self.fps * self.duration)
        self.buffersize = min(self.n_frames + 1, buffersize)
        self.buffer = None
        self.buffer_startframe = 1
        self.initialize()
        self.buffer_around(1)

    def initialize(self, start_time=0):
        """Opens the file, creates the pipe."""
        self.close()  # if any previous instance was still running

        cmd = (
                ["ffmpeg"]
                + ["-i", self.filename, "-vn"]
                + [
                    "-loglevel", "error",
                    "-f", self.format,
                    "-c:a", self.codec,
                    "-ar", f"{self.fps:d}",
                    "-ac", "2",
                    "-"
                ]
        )

        popen_params = {"bufsize": self.buffersize, "stdout": subprocess.PIPE,
                        "stderr": subprocess.PIPE, "stdin": subprocess.DEVNULL,}

        self.proc = subprocess.Popen(cmd, **popen_params)

        self.pos = np.round(self.fps * start_time)

    def read_chunk(self, chunksize):
        chunksize = int(round(chunksize))
        s = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
        data_type = {1: "int8", 2: "int16", 4: "int32"}[self.nbytes]
        result = np.frombuffer(s, dtype=data_type)
        result = (1.0 * result / 2 ** (8 * self.nbytes - 1)).reshape(
            (int(len(result) / self.nchannels), self.nchannels)
        )

        pad = np.zeros((chunksize - len(result), self.nchannels), dtype=result.dtype)
        result = np.concatenate([result, pad])
        self.pos = self.pos + chunksize
        return result

    def get_frame(self, tt):
        in_time = (tt >= 0) & (tt < self.duration)

        frames = np.round((self.fps * tt)).astype(int)[in_time]
        fr_min, fr_max = frames.min(), frames.max()

        if not (0 <= (fr_max - self.buffer_startframe) < len(self.buffer)):
            self.buffer_around(fr_max)

        try:
            result = np.zeros((len(tt), self.nchannels))
            indices = frames - self.buffer_startframe
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

    def close(self):
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait()
            self.proc = None

    __del__ = close


class AudioFileClip:
    def __init__(self, filename, fps=48000):
        self.reader = FFMPEG_AudioReader(filename, buffersize=20_0000, fps=fps, nbytes=4)
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
