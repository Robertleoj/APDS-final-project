
import torch
from utils import load_checkpoint, dice_all
from datasets import Preprocessor_2p5D, get_split_indices
from glob import glob




class Evaluator:
    def __init__(self, config, net, device='cpu'):

        self.config = config
        self.net = net.to(device)
        self.eval_batch_size = config['eval_batch_size']
        self.slice_size = config['n_slices']
        self.device = device
        self.preprocessor = Preprocessor_2p5D(config)
        

    def evaluate(self, vol):
        predictions = []

        # add an empty prediction to predictions
        predictions.append(self.__make_empty())

        total_slices = 0
        print(f"Total slices = {total_slices}", end='')

        for batch in self.__make_batches(vol):
            
            with torch.no_grad():
                out = self.net(batch)
                pred = out.argmax(1)
                predictions.append(pred.cpu())

            total_slices += batch.shape[0]

            print(f"\rTotal slices = {total_slices}", end='')

        print()

        # add an empty prediction to predictions
        predictions.append(self.__make_empty())

        
        return torch.concat(predictions, dim=0)


    def evaluate_checkpoint(self, ckpt_iters):
        self.net = load_checkpoint(
            self.net, 
            epoch=0, 
            iter=ckpt_iters,
            ckpts_path=self.config['checkpoint_dir'],
            device=self.device
        )

        split_indices = get_split_indices(self.config['split_path'])
        val_indices = split_indices[1]

        dice_scores = {}

        for idx in val_indices:
            vol, seg = self.preprocessor.get_scan_no_slice(idx)
            model_out = self.evaluate(vol)

            dice_score = dice_all(seg, model_out, self.config['mask_classes'])

            dice_scores[idx] = dice_score
            print(f"{idx=} {dice_score=}")

        return dice_scores

    def evaluate_all_checkpoints(self):
        dice_all = {}
        checkpoint_dir = self.config['checkpoint_dir']

        get_iter = lambda p: int(p.split('_')[-1].split('.')[0])

        all_ckpt_iters = list(map(
            get_iter,
            glob(f'{checkpoint_dir}/*.ckpt')
        ))

        for itr in sorted(all_ckpt_iters):
            print(f"Evaluating checkpoint at {itr} iters")
            dice_all[itr] = self.evaluate_checkpoint(itr)

        return dice_all

        
    def __make_batches(self, vol):

        all_slices = []

        vol_depth = vol.shape[0]
        slices_each_side = self.slice_size // 2

        for i in range(slices_each_side, vol_depth - slices_each_side):
            all_slices.append(
                vol[i - slices_each_side: i + slices_each_side + 1, :, :]
                .unsqueeze(0)
            )

        
        n_slices = len(all_slices)

        idx = 0
        while idx < n_slices:
            
            end_idx = min(idx + self.eval_batch_size, n_slices - 1)

            batch = torch.concat(all_slices[idx: end_idx + 1], dim=0)

            yield batch.to(self.device)

            idx = end_idx + 1
        
    
    def __make_empty(self):
        either_side = self.config['n_slices'] // 2
        empty = torch.zeros((either_side, 512, 512), dtype=torch.long)
        
        return empty
    
    

