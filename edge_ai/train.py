"""Train on licensed paired person/garment/target triplets, JSONL manifest."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.utils.data import Dataset,DataLoader
from edge_ai.model import EdgeVTON
H,W=384,256
CATEGORIES={"tops":0,"bottoms":1,"one-pieces":2}
def image(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert("RGB").resize((W,H),Image.Resampling.BILINEAR)
        return torch.from_numpy(np.array(im,dtype=np.float32)/255).permute(2,0,1)
class Pairs(Dataset):
    def __init__(self,manifest):
        self.root=Path(manifest).resolve().parent
        self.rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
        if not self.rows: raise ValueError("Empty manifest")
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        cat=torch.zeros(3,H,W); cat[CATEGORIES[r["category"]]]=1
        return torch.cat((image(self.root/r["person"]),image(self.root/r["garment"]),cat)),image(self.root/r["target"])
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True); p.add_argument("--val",required=True)
    p.add_argument("--output",required=True); p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1: p.error("positive epochs and batch size required")
    torch.manual_seed(42)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=EdgeVTON().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    train=DataLoader(Pairs(a.train),batch_size=a.batch_size,shuffle=True)
    val=DataLoader(Pairs(a.val),batch_size=a.batch_size)
    best=float("inf")
    for epoch in range(a.epochs):
        model.train()
        for x,y in train:
            opt.zero_grad(); loss=torch.nn.functional.l1_loss(model(x.to(device)),y.to(device))
            loss.backward(); opt.step()
        model.eval(); total=0; n=0
        with torch.inference_mode():
            for x,y in val:
                total+=torch.nn.functional.l1_loss(model(x.to(device)),y.to(device)).item()*len(x); n+=len(x)
        score=total/n
        print(epoch,score)
        if score<best:
            best=score; torch.save(model.state_dict(),a.output)
if __name__=="__main__": main()
