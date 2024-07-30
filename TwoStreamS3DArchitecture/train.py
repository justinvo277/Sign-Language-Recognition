import os
import torch
import argparse


from tqdm import tqdm
from dataset.dataloader import VideoDataset
from twostream.s3d_twostream import TwoStreamS3D
from utils import EarlyStopping, training_loop, validation_loop, count_parameters

parser = argparse.ArgumentParser(description="Setting Path")

parser.add_argument("--data_folder_path", type=str, help="Enter your data folder path", default="/home/hieu/dev/SLData/SLR/WLASL100_train")
parser.add_argument("--pretrained", type=str, help="Enter your pretrained path", default=None)
parser.add_argument("--model_save", type=str, help="Enter your save path", default=None)
parser.add_argument("--folder_log_path", type=str, help="Enter your folder log path", default="/home/hieu/dev/SLData/SLR/iciit/TwoStreamS3DArchitecture/log")
parser.add_argument("--folder_save_weight", type=str, help="Enter your folder save weight path", default="/home/hieu/dev/SLData/SLR/iciit/TwoStreamS3DArchitecture/checkpoint")

parser.add_argument("--data_name", type=str, help="Enter name of dataset", default="WLASL100")
parser.add_argument("--num_frames", type=int, help="Enter number frame of video", default=32)
parser.add_argument("--image_size", type=int, help="Enter image size of frame", default=224)

parser.add_argument("--num_classes", type=int, help="Enter number of classification", default=100)
parser.add_argument("--num_epochs", type=int, help="Enter number of epoch for training", default=500)
parser.add_argument("--batch_size", type=int, help="Enter batch size of a iteration", default=10)
parser.add_argument("--lr", type=float, help="Enter learning rate for trainning", default=1e-5)
parser.add_argument("--max_viz", type=int, help="Enter max_viz for validation phase", default=10)

args = parser.parse_args()



if __name__ == "__main__":

    #Check cuda;
    DEVICE = None
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    #Dataloder train with balance sample each class for train phase;
    dataset_train = VideoDataset(args.data_folder_path, args.num_frames, phase="train")
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, prefetch_factor=2, pin_memory=True, shuffle=True)
    print("LOADING DATASET TO TRAIN !!!")
    dataloader_train_lst = []
    loop_train = tqdm(dataloader_train, leave=True)
    for batch_idx, (rgb_frames, kp_frames, labels) in enumerate(loop_train):
        dataloader_train_lst.append((rgb_frames, kp_frames, labels))

        #Dataloder for validation and test phase;
    dataset_test = VideoDataset(args.data_folder_path, args.num_frames, phase="test")
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=4, prefetch_factor=2, pin_memory=True, shuffle=True)
    print("LOADING DATASET TO TEST !!!")
    dataloader_test_lst = []
    loop_test = tqdm(dataloader_test, leave=True)
    for batch_idx, (rgb_frames, kp_frames, labels) in enumerate(loop_test):
        dataloader_test_lst.append((rgb_frames, kp_frames, labels))


    model = TwoStreamS3D(num_classes=args.num_classes)
    if args.pretrained != None:
        print("LOAD PRETRINED")
        model.load_state_dict(torch.load(args.pretrained))
    model = model.to(DEVICE)

    # Loss and optimizer;
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0, amsgrad=False)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    name_weight = args.data_name + "_early_stopping.pth"
    early_stopping_path = os.path.join(args.folder_save_weight, name_weight)
    early_stopping = EarlyStopping(patience=10, verbose=True, path=early_stopping_path)

    print(f"Train on device: {DEVICE}")
    print("\n")
    print(f"Train on dataset: {args.data_name} dataset")
    print(f"Samples in train datase: {len(dataloader_train) * args.batch_size}")
    print(f"Samples in test datase: {len(dataloader_test) * args.batch_size}")
    print("\n")

    print("Model Detail")
    total_parameters = count_parameters(model)
    print(f"Total parameters in the model is: {round(total_parameters / 1000000, 4)}M parameters")
    print("\n")

    print(f"Number of epochs: {args.num_epochs}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("\n")

    for epoch in range(1, args.num_epochs + 1):

        print(f"EPOCH [{epoch}/{args.num_epochs}]: ")
        print("Trainning Phase")
        res_train = training_loop(model=model, dataloader=dataloader_train_lst, 
        optimizer=optimizer, scheduler=scheduler, loss_fn=criterion,device=DEVICE, log_path=args.folder_log_path, 
        epoch=epoch)
        print(f"Loss training phase is {res_train['loss']}, accuracy is {res_train['accuracy']}")

        if epoch % args.max_viz == 0:
            print("Validation Phase")
            res_val = validation_loop(model=model, dataloader=dataloader_test_lst, loss_fn=criterion, image_size=args.image_size, device=DEVICE)
            print(f"Loss training phase is {res_val['loss']}, accuracy is {res_val['accuracy']}")
            early_stopping(res_val['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        # print("\n")

    print("Testing Phase")
    res_test = validation_loop(model=model, dataloader=dataloader_test_lst, loss_fn=criterion, image_size=args.image_size, device=DEVICE)
    print(f"Loss training phase is {res_test['loss']}, accuracy is {res_test['accuracy']}")
    weight = args.dataset_name + "_best.pth"
    weight_path = os.path.join(args.folder_save_weight, weight)
    torch.save(model.state_dict(), weight_path)
    print("Finished Training Model !!")