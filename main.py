import wave
import struct
import os
from ctypes import windll

import tkinter as tk
from tkinter.filedialog import askopenfilename

import pygame

windll.shcore.SetProcessDpiAwareness(1)

SIZE = (720, 720)
DOT_COLOR = (0, 255, 0)
GRID_COLOR = (40, 40, 0)
BG_COLOR = (16, 16, 16)
FPS = int(input("FPS> "))

_ = tk.Tk()
_.geometry("1x1+512+512")
audio_file = askopenfilename(filetypes=(("波形文件", ".wav"),), parent=_, title="选取音频")
_.destroy()
if not os.path.isfile(audio_file) and not audio_file.endswith(".wav"):
    exit()
try:
    wro: wave.Wave_read = wave.open(audio_file)
except wave.Error as e:
    print(f"出错: {e}")
    exit()
sample_rate = wro.getframerate()
read_length = int(wro.getframerate() / FPS)

pygame.init()

screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Oscilloscope", "示波器")

clock = pygame.time.Clock()


grid = pygame.Surface(SIZE)
grid.fill(BG_COLOR)

for horizontal in range(10):
    pygame.draw.line(grid, GRID_COLOR, (horizontal * SIZE[0] / 10, 0), (horizontal * SIZE[0] / 10, SIZE[0]))

for vertical in range(10):
    pygame.draw.line(grid, GRID_COLOR, (0, vertical * SIZE[1] / 10), (SIZE[0], vertical * SIZE[1] / 10))

pygame.draw.line(grid, GRID_COLOR, (SIZE[0] / 2, 0), (SIZE[0] / 2, SIZE[0]), 3)
pygame.draw.line(grid, GRID_COLOR, (0, SIZE[1] / 2), (SIZE[0], SIZE[1] / 2), 3)

for horizontal in range(100):
    pygame.draw.line(grid, GRID_COLOR, (horizontal * SIZE[0] / 100, SIZE[1] / 2 - 3),
                     (horizontal * SIZE[0] / 100, SIZE[1] / 2 + 3))

for vertical in range(100):
    pygame.draw.line(grid, GRID_COLOR, (SIZE[0] / 2 - 3, vertical * SIZE[1] / 100),
                     (SIZE[0] / 2 + 3, vertical * SIZE[1] / 100))

grid = pygame.surfarray.pixels2d(grid)

pygame.mixer.music.load(audio_file)
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(loops=1)

font = pygame.font.SysFont("Microsoft YaHei UI", 18)
latency = 0.0


try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        latency = pygame.mixer.music.get_pos() - wro.tell() * 1000 / sample_rate
        fps = 1 / (1 / FPS - max(min(latency / 15000, 0.00075), -0.00075))
        read_length = int(wro.getframerate() / FPS)
        if abs(latency) > 64:  # latency correction
            wro.setpos(pygame.mixer.music.get_pos() * sample_rate // 1000)
        frames = wro.readframes(read_length)

        pygame.surfarray.blit_array(screen, grid)

        for left, right in struct.iter_unpack("hh", frames):
            x = int(right * SIZE[0] / 65536) + SIZE[0] // 2
            y = int(-left * SIZE[1] / 65536) + SIZE[1] // 2
            screen.set_at((y, x), DOT_COLOR)

        screen.blit(font.render(f"音像延迟: {latency:.1f}ms", True, (225, 225, 225)
                                if abs(latency) < 48 else (255, 80, 80)), (0, 0))

        pygame.display.flip()
        clock.tick(FPS)
except wave.Error:
    pass
