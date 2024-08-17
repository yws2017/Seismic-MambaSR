import argparse
import os
import time
from math import log10
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor,ToPILImage
from model import MambaIR
from utils import read_h5,save_h5,frequency_distance,ssim,cal_psnr,normal



def main(opt):
    UPSCALE_FACTOR = opt.upscale_factor
    TEST_DATA_PATH = opt.test_data_path
    TEST_TARGET_PATH = opt.test_target_path
    MODEL_PATH = opt.model_path
    SAVE_PATH = opt.save_path
    if not os.path.exists(SAVE_PATH + "/predicted/"):
        os.makedirs(SAVE_PATH + "/predicted/")
    model = MambaIR()
    model.load_state_dict(torch.load(MODEL_PATH,map_location = "cuda:0"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    file_list = os.listdir(TEST_DATA_PATH)
    to_tensor = ToTensor()

    for file in file_list:
        data = read_h5(os.path.join(TEST_DATA_PATH,file))
        data_max = np.max(data)
        data_min = np.min(data)
        # data_normal = (data - data_min) / (data_max - data_min) * 2 -1
        data_normal = (data - 5) / 5
        data_normal = to_tensor(data_normal).type(torch.FloatTensor)
        input = torch.unsqueeze(data_normal,dim = 0)
        input = input.to(device)
        out = model(input).detach().cpu()
        out = np.squeeze(out.numpy(),axis = (0,1))
        out = out * 5 + 5
        # out = (out + 1) / 2 * (data_max - data_min) + data_min
        save_h5(out,SAVE_PATH + "/predicted/" + file)
    cal_metric(sr_path = SAVE_PATH + "/predicted/",hr_path = TEST_TARGET_PATH,save_path = SAVE_PATH)



def cal_metric(sr_path,hr_path,save_path):
    sr_files = os.listdir(sr_path)
    metric = {"psnr":[],"ssim":[],"fd":[]}
    for file in sr_files:
        sr = read_h5(os.path.join(sr_path,file))
        hr = read_h5(os.path.join(hr_path,file))
        fd = frequency_distance(sr, hr, dt=0.002)
        psnr = cal_psnr(sr,hr)
        ssim_ = ssim(sr,hr).numpy()
        metric["psnr"].append(psnr)
        metric["ssim"].append(ssim_)
        metric["fd"].append(fd)
    data_frame_metric = pd.DataFrame(data = {"psnr":metric["psnr"],"ssim":metric["ssim"],"fd":metric["fd"]},
                                     index = sr_files
                                     )
    data_frame_metric.to_csv(save_path + "/result_of_files.csv",index_label = "FileName")
    with open(save_path + "/result_test.txt","w") as f:
        f.writelines("*" * 20 + "\n")
        f.writelines("psnr:    " + str(sum(metric["psnr"]) / len(sr_files)) + "\n")
        f.writelines("ssim:    " + str(sum(metric["ssim"]) / len(sr_files)) + "\n")
        f.writelines("fd:    " + str(sum(metric["fd"]) / len(sr_files)) + "\n")
        f.writelines("*" * 20)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Test The Model Performance")
    parser.add_argument("--upscale_factor",default = 2,type = int,help = "super resolution upscale factor")
    parser.add_argument("--test_data_path",default = r"../data/real/low",type = str,help = "path of test low seismic images with noise")
    parser.add_argument("--test_target_path",default = "../data/real/high",type = str,help = "path of test high seismic images without noise")
    parser.add_argument("--model_path",default = r"../result/SRF_2/model/netG_epoch_80.pth",type = str,help = "pre_trained model use for test")
    parser.add_argument("--save_path",default = r"../result/SRF_2",type = str,help = "save the test result")
    opt = parser.parse_args()
    main(opt)

# import argparse
# import os
# import time
# from math import log10
# import pandas
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# from torch.autograd import Variable
# from torchvision.transforms import ToTensor, ToPILImage
# from model import MambaIR
# from utils import read_h5, save_h5, frequency_distance, ssim, cal_psnr, normal


# def main(opt):
#     UPSCALE_FACTOR = opt.upscale_factor
#     TEST_DATA_PATH = opt.test_data_path
#     TEST_TARGET_PATH = opt.test_target_path
#     MODEL_PATH = opt.model_path
#     SAVE_PATH = opt.save_path
#     WINDOW_SIZE = opt.window_size
#     STRIDE = opt.stride

#     if not os.path.exists(SAVE_PATH + "/predicted/"):
#         os.makedirs(SAVE_PATH + "/predicted/")
    
#     model = MambaIR()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location="cuda:0"))
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     file_list = os.listdir(TEST_DATA_PATH)
#     to_tensor = ToTensor()

#     for file in file_list:
#         data = read_h5(os.path.join(TEST_DATA_PATH, file))
#         data_max = np.max(data)
#         data_min = np.min(data)
#         data_normal = (data - data_min) / (data_max - data_min)
#         data_normal = to_tensor(data_normal).type(torch.FloatTensor).to(device)
        
#         out = sliding_window_prediction(data_normal, model, WINDOW_SIZE, STRIDE, device)
        
#         out = out * (data_max - data_min) + data_min
#         save_h5(out.cpu().numpy(), SAVE_PATH + "/predicted/" + file)
    
#     cal_metric(sr_path=SAVE_PATH + "/predicted/", hr_path=TEST_TARGET_PATH, save_path=SAVE_PATH)


# def sliding_window_prediction(data, model, window_size, stride, device):
#     _, h, w = data.shape
#     output = torch.zeros((h * 2, w * 2)).to(device)  # Assuming 2x upscale factor
#     count_map = torch.zeros((h * 2, w * 2)).to(device)
    
#     for i in range(0, h - window_size + 1, stride):
#         for j in range(0, w - window_size + 1, stride):
#             input_window = data[:, i:i + window_size, j:j + window_size]
#             input_window = torch.unsqueeze(input_window, dim=0).to(device)
#             output_window = model(input_window).squeeze().detach()
            
#             output[i*2:i*2 + window_size*2, j*2:j*2 + window_size*2] += output_window
#             count_map[i*2:i*2 + window_size*2, j*2:j*2 + window_size*2] += 1
    
#     output /= count_map
#     return output


# def cal_metric(sr_path, hr_path, save_path):
#     sr_files = os.listdir(sr_path)
#     metric = {"psnr": [], "ssim": [], "fd": []}
#     for file in sr_files:
#         sr = read_h5(os.path.join(sr_path, file))
#         hr = read_h5(os.path.join(hr_path, file))
#         fd = frequency_distance(sr, hr, dt=0.002)
#         psnr = cal_psnr(sr, hr)
#         ssim_ = ssim(sr, hr).numpy()
#         metric["psnr"].append(psnr)
#         metric["ssim"].append(ssim_)
#         metric["fd"].append(fd)
    
#     data_frame_metric = pd.DataFrame(data={"psnr": metric["psnr"], "ssim": metric["ssim"], "fd": metric["fd"]}, index=sr_files)
#     data_frame_metric.to_csv(save_path + "/result_of_files.csv", index_label="FileName")
    
#     with open(save_path + "/result_test.txt", "w") as f:
#         f.writelines("*" * 20 + "\n")
#         f.writelines("psnr:    " + str(sum(metric["psnr"]) / len(sr_files)) + "\n")
#         f.writelines("ssim:    " + str(sum(metric["ssim"]) / len(sr_files)) + "\n")
#         f.writelines("fd:    " + str(sum(metric["fd"]) / len(sr_files)) + "\n")
#         f.writelines("*" * 20)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Test The Model Performance")
#     parser.add_argument("--upscale_factor", default=2, type=int, help="super resolution upscale factor")
#     parser.add_argument("--test_data_path", default=r"../data/real/low", type=str, help="path of test low seismic images with noise")
#     parser.add_argument("--test_target_path", default="../data/real/high", type=str, help="path of test high seismic images without noise")
#     parser.add_argument("--model_path", default=r"../result/SRF_2/model/netG_bestmodel.pth", type=str, help="pre_trained model use for test")
#     parser.add_argument("--save_path", default=r"../result/SRF_2", type=str, help="save the test result")
#     parser.add_argument("--window_size", default=128, type=int, help="size of the sliding window")
#     parser.add_argument("--stride", default=16, type=int, help="stride of the sliding window")
    
#     opt = parser.parse_args()
#     main(opt)