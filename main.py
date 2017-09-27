from model import *
from trainer import Trainer

fe = FrontEnd()
d = D()
g = G()
d_mag = D_Mag()

for i in [fe, d, g, d_mag]:
    i.cuda()
    i.apply(weights_init)

trainer = Trainer(g, fe, d, d_mag)
trainer.train()
