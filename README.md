# Oscilloscope
Tis is a simple oscilloscope program written in Python using the [PyGame](https://www.pygame.org/), [NumPy](https://numpy.org/), and [FFMpeg](https://www.ffmpeg.org/).

----
### Installation
To install the required packages, run the following command in your terminal:

`pip install -r requirements.txt`

Also, make sure you have FFMpeg installed on your system, which can be downloaded from [here](https://www.ffmpeg.org/download.html).

I recommend using Python 3.10 because this is the version I used to develop this program.

----
### Usage
To run the program, simply run the `oscilloscope.py` file. Any audio format supported by FFMpeg can be used as input.
And you don't need to worry about the bit depth, the program will automatically convert it to 32 bit to ensure
 Hi-Res audio is supported. Sample rate also doesn't matter, the program will take good care of that.
