for i in *.mp4; do
ffmpeg -i "$i" -vf "mpdecimate, scale=640:-2" -an -vsync vfr -vcodec libx264 "fixed/${i%.*}_fix.mp4"; 
done
for i in *.avi; do
ffmpeg -i "$i" -vf "mpdecimate, scale=640:-2" -an -vsync vfr -vcodec libx264 "fixed/${i%.*}_fix.mp4"; 
done
