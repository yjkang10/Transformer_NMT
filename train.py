import argparse
import json
import os
import time
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

from criterion import LabelSmoothing
from model import Transformer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config/config.json')
    parser.add_argument("--vocab_pkl", type=str, default="data/preprocessed/dictionary.pkl")
    parser.add_argument("--train_path", type=str, default="data/preprocessed/train_")
    parser.add_argument("--test_path", type=str, default="data/preprocessed/test_")
    parser.add_argument("--output_dir", type=str, default="saved_model")
    parser.add_argument("--device", type=str, default='gpu')
    args = parser.parse_args()

    device = torch.device('cuda') if args.device is 'gpu' else 'cpu'
    print("device: ", device)

    with open(args.config_file) as f:
        config = json.load(f)

    log_dir = args.output_dir 
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tb_writer = SummaryWriter(log_dir)

    batch_size = config['train']['batch_size']
    step_batch = config['train']['step_batch']
    max_epoch = config['train']['max_epoch']
    eval_interval = config['train']['eval_interval']
    warmup = config['train']['warmup']
    beta1 = config['train']['beta1']
    beta2 = config['train']['beta2']
    smoothing = config['train']['smoothing']
    
    n_layers = config['model']['n_layers']
    max_sent = config['model']['n_position']
    d_model = config['model']['d_model']
    d_ff = config['model']['d_ff']
    n_head = config['model']['n_head']
    dropout_p = config['model']['dropout_p']
    d_k = d_model // n_head
    

    with open(args.vocab_pkl, 'rb') as fr:
        word_to_id, id_to_word = pickle.load(fr)
    V = len(word_to_id)
    print("vocab length: ", V)

    pad_idx = word_to_id['<pad>']
    print("padding index: ", pad_idx)

    with open(args.train_path + "src.pkl", 'rb') as fr:
        src, _ = pickle.load(fr)
    with open(args.train_path + "trg.pkl", 'rb') as fr:
        trg_in, trg_out, _ = pickle.load(fr)

    with open(args.test_path + "src.pkl", 'rb') as fr:
        test_src, _ = pickle.load(fr)
    with open(args.test_path + "trg.pkl", 'rb') as fr:
        _, test_trg_out, _ = pickle.load(fr)

    train = TensorDataset(src, trg_in, trg_out)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)

    test = TensorDataset(test_src[:2], test_trg_out[:2])
    test_loader = DataLoader(test, batch_size=2, shuffle=False)

    model = Transformer(V, embedding_dim=d_model, max_sent=max_sent, dropout_p=dropout_p, 
                        n_layers=n_layers, d_model=d_model, d_k=d_k, d_ff=d_ff, 
                        n_head=n_head, device=device, pad_idx=pad_idx).to(device)
    criterion = LabelSmoothing(smoothing, V, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=1e-9)
    scaler = amp.GradScaler()


    # ckpt = torch.load('saved_model/model_5.ckpt')
    # model.load_state_dict(ckpt['model_state_dict'])
    # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # scaler.load_state_dict(ckpt['scalar_state_dict'])


    start = time.time()
    total_loss = 0
    stack = 0
    step_num = 0

    for epoch in range(max_epoch):
        print("*"*20 + "Epoch: {}/{}".format(epoch+1, max_epoch) + "*"*20)
        

        for src, trg_in, trg_out in tqdm(train_loader):
            src_mask = (src != pad_idx).unsqueeze(-2).to(device)
            src = src.to(device)
            trg_in = trg_in.to(device)
            trg_out = trg_out.to(device)
            
            with amp.autocast():
                out = model.forward(src, trg_in)
                loss = criterion(out.view(-1, V), trg_out.view(-1))
                loss /= step_batch
            
            scaler.scale(loss).backward()
            total_loss += loss.item()
            stack += 1

            if step_num == 100000: 
                break
            
            if stack > 1 and stack % step_batch == 1:
                step_num += 1
                optimizer.param_groups[0]['lr'] = d_model ** (-0.5) * np.minimum(step_num ** (-0.5), step_num * (warmup ** (-1.5)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                tb_writer.add_scalar('loss/step', total_loss, step_num)
                tb_writer.add_scalar('lr/step', optimizer.param_groups[0]['lr'], step_num)
                tb_writer.flush()
                total_loss = 0
                
            if stack % eval_interval == 1:
                elapsed = (time.time() - start)/60
                print("Step: %d | Loss: %f | Time: %f[min]" %(step_num, loss, elapsed))
                start = time.time()

            else:
                continue   
        
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "step_num": step_num}, args.output_dir+f"/model_{epoch+1}.ckpt")
        print("Model is saved!")

 
        predictions = []
        
        references = []
        ref_temp =''

        for src, trg_out in tqdm(test_loader):
            with torch.no_grad():

                trg_in = torch.zeros_like(trg_out)
                trg_in[:, 0] = torch.tensor([1]).to(torch.int64)  # <bos> : 1

                src_mask = (src != pad_idx).unsqueeze(-2).to(device)
                src = src.to(device)
                trg_in = trg_in.to(device)
                trg_out = trg_out.to(device)
                
                batch = src.size(0)
                result = torch.empty(batch, max_sent)

                for i in range(max_sent):
                    out = model.forward(src, trg_in)
                    pred = torch.max(out, dim=-1)[1]
                    if i != max_sent-1:
                        trg_in[:, i+1] = pred[:, i]
                    result[:, i] = pred[:, i].to('cpu')

                for batch in range(1):
                    prediction = []
                    temp = ''
                    for idx in result[batch]:
                        word = id_to_word[int(idx)]
                        if '@@' in word:
                            temp = word[:-2]
                            continue
                        if temp:
                            if word != word.lower():
                                word = temp + ' ' + word
                            else:
                                word = temp + word
                            temp = ''
                        if word == '<eos>':
                            continue
                        prediction.append(word)
                    sentence = ' '.join(prediction)
                    predictions.append(sentence + '\n')

                    reference = []
                    ref_temp = ''
                    for idx in trg_out[batch]:
                        word = id_to_word[int(idx)]
                        if '@@' in word:
                            ref_temp = word[:-2]
                            continue
                        if ref_temp:
                            if word != word.lower():
                                word = ref_temp + ' ' + word
                            else:
                                word = ref_temp + word
                            ref_temp = ''
                        if word == '<eos>':
                            break
                        reference.append(word)
                    gold_sentence = ' '.join(reference)
                    references.append(gold_sentence + '\n')
                    
                            
        print('Prediction' + '-'*60)
        print(predictions)
        print('Refernce' + '-'*60)
        print(references)
