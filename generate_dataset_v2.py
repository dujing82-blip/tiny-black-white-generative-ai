"""
V2 dataset: 10,000 black-and-white 16x16 pixel-art images.
Five objects x 2,000 examples each.

Compared with V1, each object has more meaningful shape variation.
The goal is to give the VAE a richer distribution to learn while keeping
the visual world extremely simple for a beginner lesson.
"""
from pathlib import Path
import csv, numpy as np
from PIL import Image

N=16
OBJECTS=["heart","face","robot","tree","spaceship"]
PER_CLASS=2000
ROOT=Path("pixel_dataset_v2"); IMG=ROOT/"images"; IMG.mkdir(parents=True,exist_ok=True)

def blank(): return np.zeros((N,N),np.uint8)
def p(a,x,y):
    if 0<=x<N and 0<=y<N: a[y,x]=255

def heart(r):
    a=blank(); dx=int(r.integers(-2,3)); dy=int(r.integers(-2,3)); scale=float(r.uniform(.85,1.15))
    # Filled heart from an implicit heart equation; scale/position vary.
    for y in range(N):
      for x in range(N):
        xx=(x-(7.5+dx))/(4.8*scale); yy=-(y-(7.5+dy))/(4.8*scale)
        if (xx*xx+yy*yy-1)**3-xx*xx*yy**3 <= 0: p(a,x,y)
    return a

def face(r):
    a=blank(); dx=int(r.integers(-1,2));dy=int(r.integers(-1,2));rad=int(r.integers(5,7));cx=8+dx;cy=8+dy
    for y in range(N):
      for x in range(N):
        d=((x-cx)**2+(y-cy)**2)**.5
        if rad-.75<=d<=rad+.45:p(a,x,y)
    eye=int(r.integers(2,4)); ey=cy-2+int(r.integers(-1,2))
    p(a,cx-eye,ey);p(a,cx+eye,ey)
    mouth_y=cy+3
    p(a,cx-2,mouth_y-1);p(a,cx+2,mouth_y-1)
    for x in range(cx-1,cx+2):p(a,x,mouth_y)
    return a

def robot(r):
    a=blank();dx=int(r.integers(-2,3));dy=int(r.integers(-1,2))
    w=int(r.integers(7,11));h=int(r.integers(7,10));x0=8-w//2+dx;y0=4+dy
    for x in range(x0,x0+w):p(a,x,y0);p(a,x,y0+h-1)
    for y in range(y0,y0+h):p(a,x0,y);p(a,x0+w-1,y)
    eye_gap=max(2,w//3);p(a,x0+eye_gap,y0+2);p(a,x0+w-1-eye_gap,y0+2)
    # Different mouth styles.
    if r.random()<.5:
      for x in range(x0+2,x0+w-2):p(a,x,y0+h-3)
    else:
      p(a,x0+w//2-1,y0+h-3);p(a,x0+w//2,y0+h-3);p(a,x0+w//2+1,y0+h-3)
    # Antenna and optional arms.
    for k in range(1,int(r.integers(2,4))):p(a,x0+w//2,y0-k)
    if r.random()<.7:
      p(a,x0-1,y0+4);p(a,x0+w,y0+4)
    p(a,x0+2,y0+h);p(a,x0+w-3,y0+h)
    return a

def tree(r):
    a=blank();dx=int(r.integers(-2,3));dy=int(r.integers(-1,2));cx=8+dx;top=1+dy
    maxhalf=int(r.integers(3,6)); layers=int(r.integers(8,11))
    for k in range(layers):
      # Triangular canopy with slight width variation.
      half=min(maxhalf,1+k//2)
      if r.random()<.15: half=max(1,half-1)
      for x in range(cx-half,cx+half+1):p(a,x,top+k)
    trunk_top=top+layers
    for y in range(trunk_top,min(N,trunk_top+5)):
      p(a,cx,y)
      if r.random()<.75:p(a,cx+1,y)
    return a

def spaceship(r):
    a=blank();dx=int(r.integers(-2,3));dy=int(r.integers(-1,2));cx=8+dx;top=1+dy
    h=int(r.integers(10,13));wide=int(r.integers(2,4))
    for k in range(h):
      half=min(wide,max(0,k//2),max(0,(h-k)//3+1))
      for x in range(cx-half,cx+half+1):p(a,x,top+k)
    # Window position and fins vary.
    p(a,cx,top+int(r.integers(3,6)))
    fin_y=top+h-3
    p(a,cx-wide-1,fin_y);p(a,cx+wide+1,fin_y)
    p(a,cx-wide,fin_y+1);p(a,cx+wide,fin_y+1)
    p(a,cx-1,top+h);p(a,cx+1,top+h)
    if r.random()<.6:p(a,cx,top+h+1)
    return a

DRAW={"heart":heart,"face":face,"robot":robot,"tree":tree,"spaceship":spaceship}
rows=[];idx=0
for label,name in enumerate(OBJECTS):
  for _ in range(PER_CLASS):
    rng=np.random.default_rng(200000+idx)
    im=DRAW[name](rng)
    fn=f"{idx:05d}.png";Image.fromarray(im,mode="L").save(IMG/fn)
    rows.append([f"images/{fn}",name,label]);idx+=1
with open(ROOT/"labels.csv","w",newline="") as f:
  w=csv.writer(f);w.writerow(["filename","word","label"]);w.writerows(rows)
print(f"Created {len(rows):,} images = {PER_CLASS:,} examples x {len(OBJECTS)} words.")
