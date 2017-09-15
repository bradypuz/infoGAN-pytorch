from model import *
from trainer import Trainer

fe = FrontEnd()
d = D()
g = G()

for i in [fe, d, g]:
    i.cuda()
    i.apply(weights_init)

trainer = Trainer(g, fe, d)
trainer.train()
