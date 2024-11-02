import re
import subprocess
import os

import numpy as np
import pygame

import tkinter as tk
from tkinter.filedialog import askopenfilename

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
    def __init__(
            self,
            infos,
            filename,
            fps_source="fps",
            check_duration=True,
            decode_file=False,
    ):
        self.infos = infos
        self.filename = filename
        self.check_duration = check_duration
        self.fps_source = fps_source
        self.duration_tag_separator = "time=" if decode_file else "Duration: "

        self._reset_state()

    def _reset_state(self):
        """Reinitializes the state of the parser. Used internally at
        initialization and at the end of the parsing process.
        """
        # could be 2 possible types of metadata:
        #   - file_metadata: Metadata of the container. Here are the tags set
        #     by the user using `-metadata` ffmpeg option
        #   - stream_metadata: Metadata for each stream of the container.
        self._inside_file_metadata = False

        # this state is needed if `duration_tag_separator == "time="` because
        # execution of ffmpeg decoding the whole file using `-f null -` appends
        # to the output the blocks "Stream mapping:" and "Output:", which
        # should be ignored
        self._inside_output = False

        # flag which indicates that a default stream has not been found yet
        self._default_stream_found = False

        # current input file, stream and chapter, which will be built at runtime
        self._current_input_file = {"streams": []}
        self._current_stream = None
        self._current_chapter = None

        # resulting data of the parsing process
        self.result = {
            "video_found": False,
            "audio_found": False,
            "metadata": {},
            "inputs": [],
        }

        # keep the value of latest metadata value parsed so we can build
        # at next lines a multiline metadata value
        self._last_metadata_field_added = None

    def parse(self):
        """Parses the information returned by FFmpeg in stderr executing their binary
        for a file with ``-i`` option and returns a dictionary with all data needed
        by MoviePy.
        """
        # chapters by input file
        input_chapters = []

        for line in self.infos.splitlines()[1:]:
            if (
                    self.duration_tag_separator == "time="
                    and self.check_duration
                    and "time=" in line
            ):
                # parse duration using file decodification
                self.result["duration"] = self.parse_duration(line)
            elif self._inside_output or line[0] != " ":
                if self.duration_tag_separator == "time=" and not self._inside_output:
                    self._inside_output = True
                # skip lines like "At least one output file must be specified"
            elif not self._inside_file_metadata and line.startswith("  Metadata:"):
                # enter "  Metadata:" group
                self._inside_file_metadata = True
            elif line.startswith("  Duration:"):
                # exit "  Metadata:" group
                self._inside_file_metadata = False
                if self.check_duration and self.duration_tag_separator == "Duration: ":
                    self.result["duration"] = self.parse_duration(line)

                # parse global bitrate (in kb/s)
                bitrate_match = re.search(r"bitrate: (\d+) kb/s", line)
                self.result["bitrate"] = (
                    int(bitrate_match.group(1)) if bitrate_match else None
                )

                # parse start time (in seconds)
                start_match = re.search(r"start: (\d+\.?\d+)", line)
                self.result["start"] = (
                    float(start_match.group(1)) if start_match else None
                )
            elif self._inside_file_metadata:
                # file metadata line
                field, value = self.parse_metadata_field_value(line)

                # multiline metadata value parsing
                if field == "":
                    field = self._last_metadata_field_added
                    value = self.result["metadata"][field] + "\n" + value
                else:
                    self._last_metadata_field_added = field
                self.result["metadata"][field] = value
            elif line.lstrip().startswith("Stream "):
                # exit stream "    Metadata:"
                if self._current_stream:
                    self._current_input_file["streams"].append(self._current_stream)

                # get input number, stream number, language and type
                main_info_match = re.search(
                    r"^Stream\s#(\d+):(\d+)(?:\[\w+])?\(?(\w+)?\)?:\s(\w+):",
                    line.lstrip(),
                )
                (
                    input_number,
                    stream_number,
                    language,
                    stream_type,
                ) = main_info_match.groups()
                input_number = int(input_number)
                stream_number = int(stream_number)
                stream_type_lower = stream_type.lower()

                if language == "und":
                    language = None

                # start builiding the current stream
                self._current_stream = {
                    "input_number": input_number,
                    "stream_number": stream_number,
                    "stream_type": stream_type_lower,
                    "language": language,
                    "default": not self._default_stream_found
                               or line.endswith("(default)"),
                }
                self._default_stream_found = True

                # for default streams, set their numbers globally, so it's
                # easy to get without iterating all
                if self._current_stream["default"]:
                    self.result[f"default_{stream_type_lower}_input_number"] = (
                        input_number
                    )
                    self.result[f"default_{stream_type_lower}_stream_number"] = (
                        stream_number
                    )

                # exit chapter
                if self._current_chapter:
                    input_chapters[input_number].append(self._current_chapter)
                    self._current_chapter = None

                if "input_number" not in self._current_input_file:
                    # first input file
                    self._current_input_file["input_number"] = input_number
                elif self._current_input_file["input_number"] != input_number:
                    # new input file

                    # include their chapters if there are for this input file
                    if len(input_chapters) >= input_number + 1:
                        self._current_input_file["chapters"] = input_chapters[
                            input_number
                        ]

                    # add new input file to self.result
                    self.result["inputs"].append(self._current_input_file)
                    self._current_input_file = {"input_number": input_number}

                # parse relevant data by stream type
                try:
                    global_data, stream_data = self.parse_data_by_stream_type(
                        stream_type, line
                    )
                except NotImplementedError as exc:
                    pass
                else:
                    self.result.update(global_data)
                    self._current_stream.update(stream_data)
            elif line.startswith("    Metadata:"):
                # enter group "    Metadata:"
                continue
            elif self._current_stream:
                # stream metadata line
                if "metadata" not in self._current_stream:
                    self._current_stream["metadata"] = {}

                field, value = self.parse_metadata_field_value(line)

                if self._current_stream["stream_type"] == "video":
                    field, value = self.video_metadata_type_casting(field, value)
                    if field == "rotate":
                        self.result["video_rotation"] = value

                # multiline metadata value parsing
                if field == "":
                    field = self._last_metadata_field_added
                    value = self._current_stream["metadata"][field] + "\n" + value
                else:
                    self._last_metadata_field_added = field
                self._current_stream["metadata"][field] = value
            elif line.startswith("    Chapter"):
                # Chapter data line
                if self._current_chapter:
                    # there is a previews chapter?
                    if len(input_chapters) < self._current_chapter["input_number"] + 1:
                        input_chapters.append([])
                    # include in the chapters by input matrix
                    input_chapters[self._current_chapter["input_number"]].append(
                        self._current_chapter
                    )

                # extract chapter data
                chapter_data_match = re.search(
                    r"^ {4}Chapter #(\d+):(\d+): start (\d+\.?\d+?), end (\d+\.?\d+?)",
                    line,
                )
                input_number, chapter_number, start, end = chapter_data_match.groups()

                # start building the chapter
                self._current_chapter = {
                    "input_number": int(input_number),
                    "chapter_number": int(chapter_number),
                    "start": float(start),
                    "end": float(end),
                }
            elif self._current_chapter:
                # inside chapter metadata
                if "metadata" not in self._current_chapter:
                    self._current_chapter["metadata"] = {}
                field, value = self.parse_metadata_field_value(line)

                # multiline metadata value parsing
                if field == "":
                    field = self._last_metadata_field_added
                    value = self._current_chapter["metadata"][field] + "\n" + value
                else:
                    self._last_metadata_field_added = field
                self._current_chapter["metadata"][field] = value

        # last input file, must be included in self.result
        if self._current_input_file:
            self._current_input_file["streams"].append(self._current_stream)
            # include their chapters, if there are
            if len(input_chapters) == self._current_input_file["input_number"] + 1:
                self._current_input_file["chapters"] = input_chapters[
                    self._current_input_file["input_number"]
                ]
            self.result["inputs"].append(self._current_input_file)

        # some video duration utilities
        if self.result["video_found"] and self.check_duration:
            self.result["video_n_frames"] = int(
                self.result["duration"] * self.result["video_fps"]
            )
            self.result["video_duration"] = self.result["duration"]
        else:
            self.result["video_n_frames"] = 1
            self.result["video_duration"] = None
        # We could have also recomputed duration from the number of frames, as follows:
        # >>> result['video_duration'] = result['video_n_frames'] / result['video_fps']

        # not default audio found, assume first audio stream is the default
        if self.result["audio_found"] and not self.result.get("audio_bitrate"):
            self.result["audio_bitrate"] = None
            for streams_input in self.result["inputs"]:
                for stream in streams_input["streams"]:
                    if stream["stream_type"] == "audio" and stream.get("bitrate"):
                        self.result["audio_bitrate"] = stream["bitrate"]
                        break

                if self.result["audio_bitrate"] is not None:
                    break

        result = self.result

        # reset state of the parser
        self._reset_state()

        return result

    def parse_data_by_stream_type(self, stream_type, line):
        """Parses data from "Stream ... {stream_type}" line."""
        try:
            return {
                "Audio": self.parse_audio_stream_data,
                "Video": self.parse_video_stream_data,
                "Data": lambda _line: ({}, {}),
            }[stream_type](line)
        except KeyError:
            raise NotImplementedError(
                f"{stream_type} stream parsing is not supported by moviepy and"
                " will be ignored"
            )

    def parse_audio_stream_data(self, line):
        """Parses data from "Stream ... Audio" line."""
        global_data, stream_data = ({"audio_found": True}, {})
        try:
            stream_data["fps"] = int(re.search(r" (\d+) Hz", line).group(1))
        except (AttributeError, ValueError):
            # AttributeError: 'NoneType' object has no attribute 'group'
            # ValueError: invalid literal for int() with base 10: '<string>'
            stream_data["fps"] = "unknown"
        match_audio_bitrate = re.search(r"(\d+) kb/s", line)
        stream_data["bitrate"] = (
            int(match_audio_bitrate.group(1)) if match_audio_bitrate else None
        )
        if self._current_stream["default"]:
            global_data["audio_fps"] = stream_data["fps"]
            global_data["audio_bitrate"] = stream_data["bitrate"]
        return (global_data, stream_data)

    def parse_video_stream_data(self, line):
        """Parses data from "Stream ... Video" line."""
        global_data, stream_data = ({"video_found": True}, {})

        try:
            match_video_size = re.search(r" (\d+)x(\d+)[,\s]", line)
            if match_video_size:
                # size, of the form 460x320 (w x h)
                stream_data["size"] = [int(num) for num in match_video_size.groups()]
        except Exception:
            raise IOError(
                (
                    "MoviePy error: failed to read video dimensions in"
                    " file '%s'.\nHere are the file infos returned by"
                    "ffmpeg:\n\n%s"
                )
                % (self.filename, self.infos)
            )

        match_bitrate = re.search(r"(\d+) kb/s", line)
        stream_data["bitrate"] = int(match_bitrate.group(1)) if match_bitrate else None

        # Get the frame rate. Sometimes it's 'tbr', sometimes 'fps', sometimes
        # tbc, and sometimes tbc/2...
        # Current policy: Trust fps first, then tbr unless fps_source is
        # specified as 'tbr' in which case try tbr then fps

        # If result is near from x*1000/1001 where x is 23,24,25,50,
        # replace by x*1000/1001 (very common case for the fps).

        if self.fps_source == "fps":
            try:
                fps = self.parse_fps(line)
            except (AttributeError, ValueError):
                fps = self.parse_tbr(line)
        elif self.fps_source == "tbr":
            try:
                fps = self.parse_tbr(line)
            except (AttributeError, ValueError):
                fps = self.parse_fps(line)
        else:
            raise ValueError(
                ("fps source '%s' not supported parsing the video '%s'")
                % (self.fps_source, self.filename)
            )

        # It is known that a fps of 24 is often written as 24000/1001
        # but then ffmpeg nicely rounds it to 23.98, which we hate.
        coef = 1000.0 / 1001.0
        for x in [23, 24, 25, 30, 50]:
            if (fps != x) and abs(fps - x * coef) < 0.01:
                fps = x * coef
        stream_data["fps"] = fps

        if self._current_stream["default"] or "video_size" not in self.result:
            global_data["video_size"] = stream_data.get("size", None)
        if self._current_stream["default"] or "video_bitrate" not in self.result:
            global_data["video_bitrate"] = stream_data.get("bitrate", None)
        if self._current_stream["default"] or "video_fps" not in self.result:
            global_data["video_fps"] = stream_data["fps"]

        return (global_data, stream_data)

    def parse_fps(self, line):
        """Parses number of FPS from a line of the ``ffmpeg -i`` command output."""
        return float(re.search(r" (\d+.?\d*) fps", line).group(1))

    def parse_tbr(self, line):
        """Parses number of TBS from a line of the ``ffmpeg -i`` command output."""
        s_tbr = re.search(r" (\d+.?\d*k?) tbr", line).group(1)

        # Sometimes comes as e.g. 12k. We need to replace that with 12000.
        if s_tbr[-1] == "k":
            tbr = float(s_tbr[:-1]) * 1000
        else:
            tbr = float(s_tbr)
        return tbr

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
            raise IOError(
                (
                    "MoviePy error: failed to read the duration of file '%s'.\n"
                    "Here are the file infos returned by ffmpeg:\n\n%s"
                )
                % (self.filename, self.infos)
            )

    def parse_metadata_field_value(
            self,
            line,
    ):
        """Returns a tuple with a metadata field-value pair given a ffmpeg `-i`
        command output line.
        """
        raw_field, raw_value = line.split(":", 1)
        return (raw_field.strip(" "), raw_value.strip(" "))

    def video_metadata_type_casting(self, field, value):
        """Cast needed video metadata fields to other types than the default str."""
        if field == "rotate":
            return (field, float(value))
        return (field, value)


def ffmpeg_parse_infos(
        filename,
        check_duration=True,
        fps_source="fps",
        decode_file=False,
        print_infos=False,
):
    # Open the file in a pipe, read output
    cmd = ["ffmpeg", "-hide_banner", "-i", filename]
    if decode_file:
        cmd.extend(["-f", "null", "-"])

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

    if print_infos:
        # print the whole info text returned by FFMPEG
        print(infos)

    try:
        return FFmpegInfosParser(
            infos,
            filename,
            fps_source=fps_source,
            check_duration=check_duration,
            decode_file=decode_file,
        ).parse()
    except Exception as exc:
        if os.path.isdir(filename):
            raise IsADirectoryError(f"'{filename}' is a directory")
        elif not os.path.exists(filename):
            raise FileNotFoundError(f"'{filename}' not found")
        raise IOError(f"Error passing `ffmpeg -i` command output:\n\n{infos}") from exc


class FFMPEG_AudioReader:
    def __init__(self, filename, buffersize, decode_file=False, fps=44100, nbytes=2, nchannels=2):
        self.filename = filename
        self.nbytes = nbytes
        self.fps = fps
        self.format = "s%dle" % (8 * nbytes)
        self.codec = "pcm_s%dle" % (8 * nbytes)
        self.nchannels = nchannels
        infos = ffmpeg_parse_infos(filename, decode_file=decode_file)
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
        self.close()  # if any

        if start_time != 0:
            offset = min(1, start_time)
            i_arg = [
                "-ss",
                "%.05f" % (start_time - offset),
                "-i",
                self.filename,
                "-vn",
                "-ss",
                "%.05f" % offset,
            ]
        else:
            i_arg = ["-i", self.filename, "-vn"]

        cmd = (
                ["ffmpeg"]
                + i_arg
                + [
                    "-loglevel",
                    "error",
                    "-f",
                    self.format,
                    "-acodec",
                    self.codec,
                    "-ar",
                    "%d" % self.fps,
                    "-ac",
                    "%d" % self.nchannels,
                    "-",
                ]
        )

        popen_params = {
            "bufsize": self.buffersize,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
        }

        self.proc = subprocess.Popen(cmd, **popen_params)

        self.pos = np.round(self.fps * start_time)

    def skip_chunk(self, chunksize):
        """TODO: add documentation"""
        _ = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
        self.proc.stdout.flush()
        self.pos = self.pos + chunksize

    def read_chunk(self, chunksize):
        """TODO: add documentation"""
        # chunksize is not being autoconverted from float to int
        chunksize = int(round(chunksize))
        s = self.proc.stdout.read(self.nchannels * chunksize * self.nbytes)
        data_type = {1: "int8", 2: "int16", 4: "int32"}[self.nbytes]
        if hasattr(np, "frombuffer"):
            result = np.frombuffer(s, dtype=data_type)
        else:
            result = np.fromstring(s, dtype=data_type)
        result = (1.0 * result / 2 ** (8 * self.nbytes - 1)).reshape(
            (int(len(result) / self.nchannels), self.nchannels)
        )

        # Pad the read chunk with zeros when there isn't enough audio
        # left to read, so the buffer is always at full length.
        pad = np.zeros((chunksize - len(result), self.nchannels), dtype=result.dtype)
        result = np.concatenate([result, pad])
        # self.proc.stdout.flush()
        self.pos = self.pos + chunksize
        return result

    def seek(self, pos):
        """
        Reads a frame at time t. Note for coders: getting an arbitrary
        frame in the video with ffmpeg can be painfully slow if some
        decoding has to be done. This function tries to avoid fectching
        arbitrary frames whenever possible, by moving between adjacent
        frames.
        """
        if (pos < self.pos) or (pos > (self.pos + 1000000)):
            t = 1.0 * pos / self.fps
            self.initialize(t)
        elif pos > self.pos:
            # print pos
            self.skip_chunk(pos - self.pos)
        # last case standing: pos = current pos
        self.pos = pos

    def get_frame(self, tt):
        """TODO: add documentation"""
        if isinstance(tt, np.ndarray):
            # lazy implementation, but should not cause problems in
            # 99.99 %  of the cases

            # elements of t that are actually in the range of the
            # audio file.
            in_time = (tt >= 0) & (tt < self.duration)

            # Check that the requested time is in the valid range
            if not in_time.any():
                raise IOError(
                    "Error in file %s, " % (self.filename)
                    + "Accessing time t=%.02f-%.02f seconds, " % (tt[0], tt[-1])
                    + "with clip duration=%f seconds, " % self.duration
                )

            # The np.round in the next line is super-important.
            # Removing it results in artifacts in the noise.
            frames = np.round((self.fps * tt)).astype(int)[in_time]
            fr_min, fr_max = frames.min(), frames.max()

            if not (0 <= (fr_min - self.buffer_startframe) < len(self.buffer)):
                self.buffer_around(fr_min)
            elif not (0 <= (fr_max - self.buffer_startframe) < len(self.buffer)):
                self.buffer_around(fr_max)

            try:
                result = np.zeros((len(tt), self.nchannels))
                indices = frames - self.buffer_startframe
                result[in_time] = self.buffer[indices]
                return result

            except IndexError:
                # repeat the last frame instead
                indices[indices >= len(self.buffer)] = len(self.buffer) - 1
                result[in_time] = self.buffer[indices]
                return result

        else:
            ind = int(self.fps * tt)
            if ind < 0 or ind > self.n_frames:  # out of time: return 0
                return np.zeros(self.nchannels)

            if not (0 <= (ind - self.buffer_startframe) < len(self.buffer)):
                # out of the buffer: recenter the buffer
                self.buffer_around(ind)

            # read the frame in the buffer
            return self.buffer[ind - self.buffer_startframe]

    def buffer_around(self, frame_number):
        """
        Fills the buffer with frames, centered on ``frame_number``
        if possible
        """
        # start-frame for the buffer
        new_bufferstart = max(0, frame_number - self.buffersize // 2)

        if self.buffer is not None:
            current_f_end = self.buffer_startframe + self.buffersize
            if new_bufferstart < current_f_end < new_bufferstart + self.buffersize:
                # We already have part of what must be read
                conserved = current_f_end - new_bufferstart
                chunksize = self.buffersize - conserved
                array = self.read_chunk(chunksize)
                self.buffer = np.vstack([self.buffer[-conserved:], array])
            else:
                self.seek(new_bufferstart)
                self.buffer = self.read_chunk(self.buffersize)
        else:
            self.seek(new_bufferstart)
            self.buffer = self.read_chunk(self.buffersize)

        self.buffer_startframe = new_bufferstart

    def close(self):
        """Closes the reader, terminating the subprocess if is still alive."""
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.stdout.close()
                self.proc.stderr.close()
                self.proc.wait()
            self.proc = None

    def __del__(self):
        # If the garbage collector comes, make sure the subprocess is terminated.
        self.close()


class AudioClip:
    def get_frame(self, t):
        """Gets a numpy array representing the RGB picture of the clip,
        or (mono or stereo) value for a sound clip, at time ``t``.

        Parameters
        ----------

        t : float or tuple or str
          Moment of the clip whose frame will be returned.
        """
        # Coming soon: smart error handling for debugging at this point
        if self.memoize:
            if t == self.memoized_t:
                return self.memoized_frame
            else:
                frame = self.make_frame(t)
                self.memoized_t = t
                self.memoized_frame = frame
                return frame
        else:
            # print(t)
            return self.make_frame(t)

    def iter_frames(self, fps=None, with_times=False, logger=None, dtype=None):
        for frame_index in logger.iter_bar(frame_index=np.arange(0, int(self.duration * fps))):
            t = frame_index / fps

            frame = self.get_frame(t)
            if (dtype is not None) and (frame.dtype != dtype):
                frame = frame.astype(dtype)
            if with_times:
                yield t, frame
            else:
                yield frame

    def __init__(self, make_frame=None, duration=None, fps=None):
        self.start = 0
        self.end = None
        self.duration = None

        self.memoize = False
        self.memoized_t = None
        self.memoized_frame = None

        if fps is not None:
            self.fps = fps

        if make_frame is not None:
            self.make_frame = make_frame
            frame0 = self.get_frame(0)
            if hasattr(frame0, "__iter__"):
                self.nchannels = len(list(frame0))
            else:
                self.nchannels = 1
        if duration is not None:
            self.duration = duration
            self.end = duration

    def iter_chunks(
            self,
            chunksize=None,
            chunk_duration=None,
            fps=None,
            quantize=False,
            nbytes=2
    ):
        if fps is None:
            fps = self.fps
        if chunk_duration is not None:
            chunksize = int(chunk_duration * fps)

        total_size = int(fps * self.duration)

        nchunks = total_size // chunksize + 1

        positions = np.linspace(0, total_size, nchunks + 1, endpoint=True, dtype=int)

        for i in range(nchunks):
            size = positions[i + 1] - positions[i]
            assert size <= chunksize
            timings = (1.0 / fps) * np.arange(positions[i], positions[i + 1])
            yield self.to_soundarray(
                timings, nbytes=nbytes, quantize=quantize, fps=fps, buffersize=chunksize
            )

    def to_soundarray(self, tt=None, fps=None, quantize=False, nbytes=2, buffersize=50000):
        if tt is None:
            if fps is None:
                fps = self.fps

            max_duration = 1 * buffersize / fps
            if self.duration > max_duration:
                stacker = np.vstack if self.nchannels == 2 else np.hstack
                return stacker(tuple(self.iter_chunks(chunksize=buffersize, fps=fps, quantize=quantize, nbytes=2)))
            else:
                tt = np.arange(0, self.duration, 1.0 / fps)
        snd_array = self.get_frame(tt)

        if quantize:
            snd_array = np.maximum(-0.99, np.minimum(0.99, snd_array))
            inttype = {1: "int8", 2: "int16", 4: "int32"}[nbytes]
            snd_array = (2 ** (8 * nbytes - 1) * snd_array).astype(inttype)

        return snd_array

    def max_volume(self, stereo=False, chunksize=50000, logger=None):
        """Returns the maximum volume level of the clip."""
        # max volume separated by channels if ``stereo`` and not mono
        stereo = stereo and self.nchannels > 1

        # zero for each channel
        maxi = np.zeros(self.nchannels)
        for chunk in self.iter_chunks(chunksize=chunksize):
            maxi = np.maximum(maxi, abs(chunk).max(axis=0))

        # if mono returns float, otherwise array of volumes by channel
        return maxi if stereo else maxi[0]


class AudioFileClip(AudioClip):
    def __init__(self, filename, decode_file=False, buffersize=200000, nbytes=2, fps=44100):
        AudioClip.__init__(self)

        self.filename = filename
        self.reader = FFMPEG_AudioReader(filename, buffersize=buffersize, decode_file=decode_file, fps=fps,
                                         nbytes=nbytes)
        self.fps = fps
        self.duration = self.reader.duration
        self.end = self.reader.duration

        self.make_frame = lambda t: self.reader.get_frame(t)
        self.nchannels = self.reader.nchannels

    def close(self):
        """Close the internal reader."""
        if self.reader:
            self.reader.close()
            self.reader = None


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


def get_audio_info(path: str) -> dict[int, int]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cmd = ["ffmpeg", "-i", path]
    popen_params = {"bufsize": 1000, "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE, "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000
    line_audio: str = [line for line in subprocess.Popen(cmd, **popen_params).communicate()[1].decode().splitlines()
                       if " Audio: " in line][0].strip()[20:].split(", ")

    audio_info = {
        "sr": int(line_audio[1][:-3]),
        "channels": int(line_audio[2].replace("stereo", "2").replace(" channels", ""))
    }

    print(f"采样率:", audio_info["sr"])
    return audio_info


audio = AudioFileClip(audio_file)
sr = audio.fps
channels = audio.nchannels
audio = audio.to_soundarray()
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
