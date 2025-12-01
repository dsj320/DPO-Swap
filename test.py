     import wandb, numpy as np
     wandb.init(project="Face_Swapping", name="debug-upload")
     
     wandb.log({"debug/img": wandb.Image(np.zeros((64,64,3), dtype=np.uint8))})