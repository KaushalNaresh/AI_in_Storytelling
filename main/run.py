import argparse
import torch.nn.functional as F
from transformers import GPT2Config
from tqdm import trange
import torch
from transformers import GPT2LMHeadModel
import logging
from transformers import GPT2Tokenizer
import numpy as np
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s-%(message)s',datefmt = '%m/%d/%Y',level = logging.INFO)
trn_mdl = {'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)}
logger = logging.getLogger(__name__)
max_val = int(998)  
def seed(args):
    torch.manual_seed(args.dataval)
    np.random.seed(args.dataval)
    if args.gpu_val > 0:
        torch.cuda.manual_seed_all(args.dataval)
def kpfilter(prob_val, tk=0, tp=0.0, fil=-float('Inf')):
    tk = min(tk, prob_val.size(-1))  
    if tk > 0:
        indices_to_remove = prob_val < torch.topk(prob_val, tk)[0][..., -1, None]
        prob_val[indices_to_remove] = fil
    if tp > 0.0:
        prob_val_s, index_val = torch.sort(prob_val, descending=True)
        sump = torch.cumsum(F.softmax(prob_val_s, dim=-1), dim=-1)
        del_val = sump > tp
        del_val[..., 0] = 0
        del_val[..., 1:] = del_val[..., :-1].clone()
        indices_to_remove = del_val.scatter(dim=1, index=index_val, src=del_val)
        prob_val[indices_to_remove] = fil
    return prob_val
def seqsample(outp, tot_seq, point, tot_count=1, tmp=1, tk=0, tp=0.0, no_rep=1.0,dev_type='cpu'):
    point = torch.tensor(point, dtype=torch.long, device=dev_type)
    point = point.unsqueeze(0).repeat(tot_count, 1)
    res_received = point
    with torch.no_grad():
        for z in trange(tot_seq):
            inval = {'input_ids': res_received}
            res = outp(**inval)
            prob_val_prov = res[0][:, -1, :] / (tmp if tmp > 0 else 1.)
            for i in range(tot_count):
                for _ in set(res_received[i].tolist()):
                    prob_val_prov[i, _] /= no_rep
            prob_val_arranged = kpfilter(prob_val_prov, tk=tk, tp=tp)
            if tmp == 0:
                gen_val = torch.argmax(prob_val_arranged, dim=-1).unsqueeze(-1)
            else:
                gen_val = torch.multinomial(F.softmax(prob_val_arranged, dim=-1), num_samples=1)
            res_received = torch.cat((res_received, gen_val), dim=1)
    return res_received
#Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", default="output", type=str)
    parser.add_argument("--model", default="gpt2", type=str)
    parser.add_argument("--text",  default="", type=str)
    parser.add_argument("--tot_seq",default=200,type=int)
    parser.add_argument("--Total_samples", default=1,type=int)
    parser.add_argument("--textpad", default="", type=str)
    parser.add_argument("--tmp", default=0.8,type=float)
    parser.add_argument("--no_rep", default=1.0, type=float)
    parser.add_argument("--tp", default=0.9, type=float)
    parser.add_argument("--gpu_cpu", action='store_true')
    parser.add_argument("--tk",  default=0, type=int)
    parser.add_argument('--dataval', default=42, type=str)
    parser.add_argument('--cut_gen', default=None, type=str)
    args = parser.parse_args()
    args.dev_type = torch.device("cuda" if torch.cuda.is_available() and not args.gpu_cpu else "cpu")
    args.gpu_val = torch.cuda.device_count()
    seed(args)
    args.model = args.model.lower()
    model_class, tokenizer_class = trn_mdl[args.model]
    tokenizer = tokenizer_class.from_pretrained(args.path_to_model)
    outp = model_class.from_pretrained(args.path_to_model)
    outp.to(args.dev_type)
    outp.eval()
    if args.tot_seq < 0 and outp.config.max_position_embeddings > 0:
        args.tot_seq = outp.config.max_position_embeddings
    elif 0 < outp.config.max_position_embeddings < args.tot_seq:
        args.tot_seq = outp.config.max_position_embeddings 
    elif args.tot_seq < 0:
        args.tot_seq = max_val  
    logger.info(args)
    if args.model in ["ctrl"]:
        if args.tmp > 0.7:
            logger.info('works effectively with lower k and sampling types.')
    while True:
        txt_data = args.text if args.text else input("Model prompt >>> ")
        tknsvals = tokenizer.encode(txt_data, add_special_tokens=False)
        if args.model == "ctrl":
            if not any(tknsvals[0] == i for i in tokenizer.control_codes.values()):
                logger.info("WARNING! Risk of a bad result...")
        results = seqsample(
            outp=outp,
            point=tknsvals,
            tot_count=args.Total_samples,
            tot_seq=args.tot_seq,
            tmp=args.tmp,
            tk=args.tk,
            tp=args.tp,
            no_rep=args.no_rep,
            dev_type=args.dev_type)
        results = results[:, len(tknsvals):].tolist()
        for r in results:
            obtained = tokenizer.decode(r, clean_up_tokenization_spaces=True)
            obtained = obtained[: obtained.find(args.cut_gen) if args.cut_gen else None]
            print(obtained)
        if args.text:
            break
    return obtained
if __name__ == '__main__':
    main()
