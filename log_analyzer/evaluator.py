import torch
import numpy as np

def evaluate_model(batch_num, batch, test_day, trainer, jagged, outfile = None, verbose = False): 
    # TODO: outfile is None since I think it is not necessary. but there is a functionality for outfile, since we might need it in future.
    #       If we don't use outfile, then we can remove batch_num as well.
    cuda = torch.cuda.is_available()
    trainer.model.eval()
    
    X, Y, L, M = trainer.split_batch(batch)
    
    token_losses = trainer.compute_loss(X, Y, lengths=L, mask=M)

    if outfile is not None:
        for line, sec, day, usr, red, loss in zip(batch['line'].flatten().tolist(),
                                                batch['second'].flatten().tolist(),
                                                batch['day'].flatten().tolist(),
                                                batch['user'].flatten().tolist(),
                                                batch['red'].flatten().tolist(),
                                                token_losses.flatten().tolist()):
            outfile.write('%s %s %s %s %s %s %r\n' % (batch_num, line, sec, day, usr, red, loss))

    if verbose:
        print(f"{X.shape[0]}, {batch['line'][0]}, {batch['second'][0]} fixed {test_day} {loss}") 
        # TODO: I don't think this print line, but I decided to keep it since removing a line is always easier than adding a line.
        #       Also, In the original code, there was {data.index} which seems to be an accumulated sum of batch sizes. 
        #       I don't think we need {data.index}. but... I added it to to-do since we might need to do it in future.
    return None