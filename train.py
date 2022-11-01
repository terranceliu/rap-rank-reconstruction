import os
import torch
from src.algo.nondp import IterativeAlgoNonDP
from src.utils.general import get_data_onehot

from utils import get_args, get_base_dir, get_data, get_qm, get_model, update_model_init, get_errors

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = get_args()
base_dir = get_base_dir(args)

data, data_split = get_data(args, return_both_splits=True)
if args.init_split:
    if args.K == -1:
        args.K = len(data_split)
    oh_split = get_data_onehot(data_split)

query_manager = get_qm(args, data, device)
true_answers = query_manager.get_answers(data)

model_base_dir = os.path.join('out', base_dir, 'runs/seed_{}')
for seed in range(args.num_models):
    model_save_dir = model_base_dir.format(seed)
    model_dict = get_model(args.synth, query_manager, args.K, device, seed)
    if args.init_split:
        update_model_init(model_dict['G'], oh_split, seed)

    algo = IterativeAlgoNonDP(model_dict['G'], args.T,
                              default_dir=model_save_dir, verbose=True, seed=seed,
                              lr=model_dict['lr'], eta_min=model_dict['eta_min'],
                              max_idxs=args.max_idxs, max_iters=1,
                              sample_by_error=True, log_freq=1,
                              )

    if os.path.exists(os.path.join(model_save_dir, 'last.pt')):
        continue
    algo.fit(true_answers)
    algo.save('last.pt')

    syn_answers = algo.G.get_qm_answers()
    errors = get_errors(true_answers, syn_answers)
    print(errors)