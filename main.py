from ctypes import windll
import wave
import struct
import pygame
import os

import tkinter as tk
from tkinter.filedialog import askopenfilename

windll.shcore.SetProcessDpiAwareness(1)

SIZE = (720, 720)
DOT_COLOR = (0, 255, 0)
GRID_COLOR = (40, 40, 0)
BG_COLOR = (16, 16, 16)
FPS = 73.2
frame_time = 1/FPS * 1000

_ = tk.Tk()
_.geometry("1x1+256+256")
_.iconify()
audio_file = askopenfilename(filetypes=(("波形文件", ".wav"),))
if not os.path.isfile(audio_file) and not audio_file.endswith(".wav"):
    exit()
try:
    wro: wave.Wave_read = wave.open(audio_file)
except wave.Error as e:
    print(f"出错: {e}")
    exit()
sample_rate = wro.getframerate()
READ_LENGTH = int(wro.getframerate() / FPS)

pygame.init()

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Oscilloscope", "示波器")

clock = pygame.time.Clock()


def get_grid(size) -> pygame.Surface:
    grd = pygame.Surface(size)
    grd = grd.convert_alpha()
    grd.set_alpha(128)
    grd.fill(BG_COLOR)

    for x in range(10):
        pygame.draw.line(grd, GRID_COLOR, (x * SIZE[0] / 10, 0), (x * SIZE[0] / 10, SIZE[0]))

    for y in range(10):
        pygame.draw.line(grd, GRID_COLOR, (0, y * SIZE[1] / 10), (SIZE[0], y * SIZE[1] / 10))

    pygame.draw.line(grd, GRID_COLOR, (SIZE[0] / 2, 0), (SIZE[0] / 2, SIZE[0]), 3)
    pygame.draw.line(grd, GRID_COLOR, (0, SIZE[1] / 2), (SIZE[0], SIZE[1] / 2), 3)

    for x in range(100):
        pygame.draw.line(grd, GRID_COLOR, (x * SIZE[0] / 100, SIZE[1] / 2 - 3), (x * SIZE[0] / 100, SIZE[1] / 2 + 3))

    for y in range(100):
        pygame.draw.line(grd, GRID_COLOR, (SIZE[0] / 2 - 3, y * SIZE[1] / 100), (SIZE[0] / 2 + 3, y * SIZE[1] / 100))

    return grd


grid = get_grid(SIZE)

pygame.mixer.music.load(audio_file)
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(loops=1)

font = pygame.font.SysFont("Microsoft YaHei UI", 18)
latency = 0.0


while True:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if pygame.mixer.music.get_pos() % 333 < frame_time:
        latency = pygame.mixer.music.get_pos() - wro.tell() * 1000 / sample_rate
        if abs(latency) > 100:
            wro.setpos(pygame.mixer.music.get_pos() * sample_rate // 1000)
    frames = wro.readframes(READ_LENGTH)

    screen.fill(BG_COLOR)
    screen.blit(grid, grid.get_rect())

    for i in range(0, READ_LENGTH, 1):
        r = struct.unpack('hh', frames[i * 4:i * 4 + 4])
        x = int(r[1] * SIZE[0] / 65536) + SIZE[0] // 2
        y = int(-r[0] * SIZE[1] / 65536) + SIZE[1] // 2
        screen.set_at((y, x), DOT_COLOR)
    screen.blit(font.render(f"音像延迟: {latency:.1f}ms", True, (225, 225, 225)
                            if abs(latency) < 75 else (255, 100, 100)), (0, 0))

    pygame.display.flip()

