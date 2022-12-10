
import torch
from utils import load_checkpoint, dice_all, postprocess
from datasets import Preprocessor_2p5D, get_split_indices
from glob import glob





class Evaluator:
    def __init__(self, config, net, device='cpu', print_func=None):

        if print_func is None:
            self.print_func = print
        else:
            self.print_func = print_func

        self.config = config
        self.net = net.to(device)
        self.eval_batch_size = config['eval_batch_size']
        self.slice_size = config['n_slices']
        self.device = device
        self.preprocessor = Preprocessor_2p5D(config)
        

    def evaluate(self, vol):

        self.net.eval()
        predictions = []

        # add an empty prediction to predictions
        predictions.append(self.__make_empty())

        total_slices = 0

        print(f"Total slices = {vol.shape[0]}")

        if self.print_func is print:
            self.print_func(f"Processed slices = {total_slices}", end='')
        else:
            self.print_func(f"Processed slices = {total_slices}")

        for batch in self.__make_batches(vol):
            
            with torch.no_grad():
                out = self.net(batch)
                pred = out.argmax(1)
                predictions.append(pred.cpu())

            total_slices += batch.shape[0]
            
            if self.print_func is print:
                self.print_func(f"\rProcessed slices = {total_slices}", end='')
            else:
                self.print_func(f"Processed slices = {total_slices}")

        print()

        # add an empty prediction to predictions
        predictions.append(self.__make_empty())

        
        return torch.concat(predictions, dim=0)


    def evaluate_checkpoint(self, 
        ckpt_iters, 
        apply_postprocess=True,
        which_split='val'
    ):

        assert which_split in ('train', 'val', 'test')

        self.net = load_checkpoint(
            self.net, 
            epoch=0, 
            iter=ckpt_iters,
            ckpts_path=self.config['checkpoint_dir'],
            device=self.device,
        )

        split_indices = get_split_indices(self.config['split_path'])

        if which_split == 'val':
            idx = 1

        if which_split == 'train':
            idx = 0
        
        if which_split == 'test':
            idx = 2
        

        val_indices = split_indices[idx]

        dice_scores = {}

        for idx in val_indices:
            vol, seg = self.preprocessor.get_scan_no_slice(idx)
            model_out = self.evaluate(vol)

            if apply_postprocess:
                model_out = postprocess(model_out)

            dice_score = dice_all(seg, model_out, self.config['mask_classes'])

            dice_scores[idx] = dice_score
            self.print_func(f"{idx=} {dice_score=}")

        return dice_scores

    def evaluate_all_checkpoints(self, 
        apply_postprocess=True,
        which_split='val'
    ):

        self.print_func("evaluate all checkpoints")
        dice_all = {}
        checkpoint_dir = self.config['checkpoint_dir']

        get_iter = lambda p: int(p.split('_')[-1].split('.')[0])

        self.print_func("obtaining all checkpoint iterations")
        all_ckpt_iters = list(map(
            get_iter,
            glob(f'{checkpoint_dir}/*.ckpt')
        ))

        self.print_func("Starting main for loop")
        
        all_ckpt_iters = sorted(all_ckpt_iters)

        itr = all_ckpt_iters[0]

        while itr in all_ckpt_iters:
            self.print_func(f"Evaluating checkpoint at {itr} iters")
            dice_all[itr] = self.evaluate_checkpoint(
                itr,
                apply_postprocess,
                which_split=which_split
            )
            itr += 3000

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
    
    

