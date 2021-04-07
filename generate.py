import os
import sys
import time
import copy
import shutil
import random

import torch
import numpy as np
from tqdm import tqdm

import config
import utils


##### Parse CmdLine Arguments #####
args, unparsed = config.get_args()
cwd = os.getcwd()
print(args)


device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)




##### Build Model #####
if args.model.lower() == 'cain_encdec':
    from model.cain_encdec import CAIN_EncDec
    print('Building model: CAIN_EncDec')
    model = CAIN_EncDec(depth=args.depth, start_filts=32)
elif args.model.lower() == 'cain':
    from model.cain import CAIN
    print("Building model: CAIN")
    model = CAIN(depth=args.depth)
elif args.model.lower() == 'cain_noca':
    from model.cain_noca import CAIN_NoCA
    print("Building model: CAIN_NoCA")
    model = CAIN_NoCA(depth=args.depth)
else:
    raise NotImplementedError("Unknown model!")
# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)
#print(model)

print('# of parameters: %d' % sum(p.numel() for p in model.parameters()))


# If resume, load checkpoint: model
if args.resume:
    #utils.load_checkpoint(args, model, optimizer=None)
    checkpoint = torch.load('pretrained_cain.pth')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint



def test(args, epoch):
    print('Evaluating for epoch = %d' % epoch)
    losses, psnrs, ssims, lpips = utils.init_meters(args.loss)
    ##### Load Dataset #####
    test_loader = utils.load_dataset(
        args.dataset, args.data_root, args.batch_size, args.test_batch_size, args.num_workers, img_fmt=args.img_fmt)
    model.eval()

    t = time.time()
    save_folder = 'test%03d' % epoch
    save_folder = os.path.join(save_folder, args.dataset)
    with torch.no_grad():
        for i, (images, imgpaths) in enumerate(tqdm(test_loader)):

            # Build input batch
            im1, im2, gt = images[0].to(device), images[2].to(device), images[1].to(device)

            # Forward
            out, _ = model(im1, im2)
            
            # Evaluate metrics
            utils.eval_metrics(out, gt, psnrs, ssims, lpips)

            print(imgpaths[1][-1])
            print("Loss: %f, PSNR: %f, SSIM: %f, LPIPS: %f" %
                      (losses['total'].val, psnrs.val, ssims.val, lpips.val))

            savepath = os.path.join('checkpoint', args.exp_name, save_folder)

            for b in range(images[0].size(0)):
                paths = imgpaths[1][b].split('/')
                fp = os.path.join(savepath, paths[-3], paths[-2])
                if not os.path.exists(fp):
                    os.makedirs(fp)
                # remove '.png' extension
                fp = os.path.join(fp, paths[-1][:-4])
                utils.save_image(out[b], "%s.png" % fp)
                    
            # Print progress
            print('im_processed: {:d}/{:d} {:.3f}s   \r'.format(i + 1, len(test_loader), time.time() - t))
    print("Average PSNR: {}".format(psnrs.avg))
    print("Average SSIM: {}".format(ssims.avg))
    return


""" Entry Point """
def main(args):

    num_iter = 1 # x2**num_iter interpolation
    for _ in range(num_iter):
        
        # run test
        test(args, args.start_epoch)


if __name__ == "__main__":
    main(args)
