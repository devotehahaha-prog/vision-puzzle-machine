"""Render the actual screen drawing code to a PNG without touching hardware."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manual_page import paint, ManualPageState
from start_page import paint_button, theme_colors

image = Image.new("RGB", (800,480))
canvas = ImageDraw.Draw(image)
font_path = next(p for p in [Path("C:/Windows/Fonts/msyh.ttc"),Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")] if p.exists())


class Draw:
    def clear(self, color): canvas.rectangle((0,0,799,479), fill=color)
    def fill_rect(self,x,y,w,h,color):
        if w > 0 and h > 0: canvas.rectangle((x,y,x+w-1,y+h-1),fill=color)


class Text:
    def measure(self,value,size):
        font=ImageFont.truetype(str(font_path),size)
        return (int(font.getlength(value)),size)
    def text(self,x,y,value,size=20,color=(255,255,255),bg=None):
        canvas.text((x,y),value,font=ImageFont.truetype(str(font_path),size),fill=color,anchor="lt")


devices={"grbl":{"connected":True,"pending":False,"machine":"Idle","MPos":(0,-42.5,-0.1),"message":"已连接"},
         "aux":{"connected":True,"pending":False,"armed":True,"speed":0,"magnet":False,"message":"已连接",
                "motion_enabled":True,"motion_busy":False,"zeroed":True,"r_x100":4500}}
paint(Draw(),Text(),theme_colors(lambda *c:c),ManualPageState(),devices,paint_button)
out=Path(sys.argv[1] if len(sys.argv)>1 else "manual_control_preview.png")
image.save(out)
print(out.resolve())
