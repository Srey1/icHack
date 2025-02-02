from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.all import vflip, hflip, rotate

clip = VideoFileClip('icHack/BusDetector/stock_footage/IMG_8451.MOV')

# Rotate the video 90 degrees clockwise
rotated_clip = clip.rotate(90)

# Save the rotated video
rotated_clip.write_videofile('icHack/BusDetector/flipped_stock_footage/IMG_8451.MOV', codec='libx264')
