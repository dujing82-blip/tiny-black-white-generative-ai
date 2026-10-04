"""
Generate a simple black-and-white pixel-art dataset for the class.

Five classes:
    heart, face, robot, tree, spaceship

Each image is 16 x 16 pixels.
0 = black
1 = white

We generate many slightly different examples by changing position,
width, height, and small object details. This gives the neural network
real variation to learn instead of 1,000 exact copies of one picture.
"""

from pathlib import Path
import csv
import numpy as np
from PIL import Image

IMAGE_SIZE = 16
OBJECTS = ["heart", "face", "robot", "tree", "spaceship"]
IMAGES_PER_OBJECT = 2000
ROOT = Path("pixel_dataset")
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def blank():
    """Create a 16x16 black image."""
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

def pixel(img, x, y):
    """Safely turn one pixel white."""
    if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
        img[y, x] = 255

def heart(rng):
    img=blank(); dx=int(rng.integers(-2,3)); dy=int(rng.integers(-2,3))
    # A heart described by rows of white pixels.
    rows=[(5,6),(4,8),(3,10),(3,10),(4,8),(5,6),(6,4),(7,2)]
    # Randomly make the heart a little wider or narrower.
    extra=int(rng.integers(-1,2))
    for y,(x0,x1) in enumerate(rows, start=3+dy):
        for x in range(x0+dx-extra, x1+dx+extra):
            pixel(img,x,y)
    return img

def face(rng):
    img=blank(); dx=int(rng.integers(-1,2)); dy=int(rng.integers(-1,2))
    # Random face radius: 5 or 6 pixels.
    r=int(rng.integers(5,7)); cx,cy=8+dx,8+dy
    for y in range(16):
        for x in range(16):
            d=(x-cx)**2+(y-cy)**2
            if (r-1)**2 <= d <= r**2+3:
                pixel(img,x,y)
    # Eyes can be slightly closer or farther apart.
    eye=int(rng.integers(2,4))
    pixel(img,cx-eye,cy-2); pixel(img,cx+eye,cy-2)
    # Simple smiling mouth.
    pixel(img,cx-2,cy+2); pixel(img,cx+2,cy+2)
    for x in range(cx-1,cx+2): pixel(img,x,cy+3)
    return img

def robot(rng):
    img=blank(); dx=int(rng.integers(-2,3)); dy=int(rng.integers(-1,2))
    w=int(rng.integers(7,10)); h=int(rng.integers(7,10))
    x0=8-w//2+dx; y0=4+dy
    # Rectangular robot head/body outline.
    for x in range(x0,x0+w):
        pixel(img,x,y0); pixel(img,x,y0+h-1)
    for y in range(y0,y0+h):
        pixel(img,x0,y); pixel(img,x0+w-1,y)
    # Eyes.
    pixel(img,x0+2,y0+2); pixel(img,x0+w-3,y0+2)
    # Mouth.
    for x in range(x0+2,x0+w-2): pixel(img,x,y0+h-3)
    # Antenna is sometimes one pixel taller.
    antenna=int(rng.integers(1,3))
    for k in range(1,antenna+1): pixel(img,x0+w//2,y0-k)
    # Legs.
    pixel(img,x0+2,y0+h); pixel(img,x0+w-3,y0+h)
    return img

def tree(rng):
    img=blank(); dx=int(rng.integers(-2,3)); dy=int(rng.integers(-1,2))
    cx=8+dx; top=2+dy
    # Random canopy width makes different trees.
    max_half=int(rng.integers(3,6))
    for k in range(9):
        half=min(max_half,1+k//2)
        for x in range(cx-half,cx+half+1):
            pixel(img,x,top+k)
    # Trunk.
    for y in range(top+9,min(16,top+14)):
        pixel(img,cx,y)
        if rng.random()>0.35: pixel(img,cx+1,y)
    return img

def spaceship(rng):
    img=blank(); dx=int(rng.integers(-2,3)); dy=int(rng.integers(-1,2))
    cx=8+dx; top=2+dy
    height=int(rng.integers(10,13))
    # Pointed body.
    for k in range(height):
        half=min(3, k//2, max(0,(height-k)//3+1))
        for x in range(cx-half,cx+half+1):
            pixel(img,x,top+k)
    # Window.
    pixel(img,cx,top+4)
    # Side fins.
    pixel(img,cx-4,top+height-3); pixel(img,cx+4,top+height-3)
    pixel(img,cx-3,top+height-2); pixel(img,cx+3,top+height-2)
    # Flames vary slightly.
    pixel(img,cx-1,top+height); pixel(img,cx+1,top+height)
    if rng.random()>0.5: pixel(img,cx,top+height+1)
    return img

DRAW = {"heart":heart, "face":face, "robot":robot, "tree":tree, "spaceship":spaceship}

rows=[]
index=0
for object_id, name in enumerate(OBJECTS):
    for example in range(IMAGES_PER_OBJECT):
        # A deterministic seed means everyone generates the same dataset.
        rng=np.random.default_rng(100000 + index)
        img=DRAW[name](rng)
        filename=f"{index:05d}.png"
        Image.fromarray(img, mode="L").save(IMAGE_DIR/filename)
        rows.append([f"images/{filename}", name, object_id])
        index += 1

with open(ROOT/"labels.csv","w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["filename","word","label"])
    writer.writerows(rows)

print(f"Done! Created {len(rows)} images.")
print(f"{IMAGES_PER_OBJECT} examples x {len(OBJECTS)} words = {len(rows)} total images.")
