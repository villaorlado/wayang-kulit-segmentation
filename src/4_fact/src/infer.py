import argparse
import json
import torch
from .models.blocks import FACT 
from .utils.dataset import create_test_dataset, DataLoader
from .utils import utils
from .utils.evaluate import Checkpoint, Video
from .utils.train_tools import save_results
from .configs.utils import int2float_check, _get_var
from tqdm import tqdm
from yacs.config import CfgNode


def read_args_json_file(args_json_file):

    # Read the json file
    with open(args_json_file, "r") as f:
        config_dict = json.load(f)
    json_cfg = CfgNode(config_dict)
    set_cfgs = json_cfg.aux.set_cfgs

    # Define the new cfg file
    cfg = CfgNode(json_cfg)

    # preprocess set_cfgs to convert int2float
    L = len(set_cfgs) if set_cfgs else 0
    new_set_cfgs = []
    for i in range(L//2):
        k = set_cfgs[i*2]
        v = set_cfgs[i*2+1]

        if not isinstance(k, list):
                k = [k]
        for k_ in k:
            tgt = _get_var(cfg, k_.split('.'))
            v_ = int2float_check(v, tgt)
            new_set_cfgs.extend([k_, v_])

    # update cfg
    cfg.merge_from_list(new_set_cfgs)

    return cfg


def eval_video(cfg_path:str,
               thumbnails_dir:str,
               mapping_path:str,
               weights_path:str,
               output_json_path:str):
    """
    Performs the inference on the thumbnails of every video in the target directory

    [Arguments]
        args_json_path: path to the json file containing the arguments
        weights_path: path to the model weights

        output_json_path: path to the json file where the predictions will be saved

        thumbnails_npy_path: path to the thumbnails npy file
        mapping_path: path to the class mapping file
    """

    # Read config file
    cfg = read_args_json_file(cfg_path)
    test_dataset = create_test_dataset(cfg, thumbnails_dir, mapping_path)

    # Load Model
    model = FACT(cfg, test_dataset.input_dimension, test_dataset.nclasses)

    # Read Model weights
    weights = torch.load(weights_path, map_location='cpu')
    if 'frame_pe.pe' in weights:
        del weights['frame_pe.pe']
    model.load_state_dict(weights, strict=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Dataloaders
    test_loader = DataLoader(test_dataset, 1, shuffle=False)

    # Get predictions
    all_results = {}
    for vname, batch_seq, train_label_list, _ in tqdm(test_loader):

        # Perform the predictions
        seq_list = [s.to(device) for s in batch_seq]
        train_label_list = [s.to(device) for s in train_label_list]
        video_saves = model(seq_list, train_label_list)

        print(f"Predicted {vname[0]}")
        # Save the predictions
        local_results = {}
        pred_array = []
        for pred in video_saves[0]['pred']:
            pred_array.append(int(pred))
        local_results["pred"] = pred_array
        all_results[vname[0]] = local_results

    # Writing to output_file
    json_object = json.dumps(all_results, indent=4)
    with open(f"{output_json_path}", "w") as outfile:
        outfile.write(json_object)
    print(f"Saved predictions to {output_json_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference on the thumbnails")
    parser.add_argument("--cfg_path", type=str, required=True, help="path to the json file containing the arguments")
    parser.add_argument("--thumbnails_dir", type=str, required=True, help="path to the thumbnails directory")
    parser.add_argument("--mapping_path", type=str, required=True, help="path to the class mapping file")
    parser.add_argument("--weights_path", type=str, required=True, help="path to the model weights")
    parser.add_argument("--results_output_path", type=str, required=True, help="path to output the results file")

    args = parser.parse_args()

    eval_video(args.cfg_path, args.thumbnails_dir, args.mapping_path, args.weights_path, args.results_output_path)