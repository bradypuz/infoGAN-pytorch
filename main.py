from model import *
from trainer import Trainer

fe = FrontEnd()
d = D()
g = G()
d_mag = D_Mag()
d_mag_fc = D_Mag_FC()
d_blur = D_blur()

for i in [fe, d, g, d_mag, d_mag_fc, d_blur]:
    i.cuda()
    i.apply(weights_init)

trainer = Trainer(g, fe, d, d_mag, d_mag_fc, d_blur)
trainer.train()
