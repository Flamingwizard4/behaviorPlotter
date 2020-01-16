import os, re, dropbox, subprocess, pdb, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


##
#code for finding top 3 accuracy/confidence predictions of 3D CNN
#and displaying them per individual per behavior in a gif/video
#ideally should be 10 x 7 x 3 for the 10 categories of behavior, the seven individuals, and three examples of each behavior
#
#go to each trial's AllClusterData.csv
#find top three test predictions for resnet18 for each behavior for each trial
#download corresponding directory from dropbox if not already downloaded
#
#find clip in directory and convert it to frames, keeping track of minimum frames
#trim all clips to minimum frames (UNNECESSARY, ALL HAVE 120)
#
#create num_frame number of identical matplotlib 10 x 21 plots with corresponding frames shown and labels included
#order behaviors/trials
#clean up gif frames
#convert into gif/video
##


cur_dir = os.getcwd()

data_dir = os.path.join(cur_dir, 'ClusterData')

'''
test_cluster = pd.read_csv(os.path.join(data_dir, "CV10_3", "AllClusterData.csv"))

test_cluster_tests = test_cluster[test_cluster["modelAll_18_type"] == "Tests"]

test_cluster_tests_sorted = test_cluster_tests.sort_values(by = "modelAll_18_conf", ascending = False)

test_n = int(test_cluster["N"].iloc[0])
'''



actions = ['s', 'o', 't', 'm', 'x', 'p', 'c', 'f', 'd', 'b'] #test_cluster["ManualLabel"].dropna().unique()

#print(actions_ordered)

#test_n + r"([0-9]|_)*"

'''grabbing videos
for trial in os.listdir(data_dir):

    data = pd.read_csv(os.path.join(data_dir, trial, "AllClusterData.csv"))

    #only keep Tests
    data = data[data["modelAll_18_type"] == "Tests"]
    #data.to_csv(os.path.join(data_dir, trial, "TestClusterData.csv"))

    #grab top three of each action
    #print(trial, '\n')
    for act in actions:
        if not os.path.exists(os.path.join(data_dir, trial, act)):
            os.mkdir(os.path.join(data_dir, trial, act))
        datachunk = data[data["ManualLabel"] == act]
        #print(act, ': ', len(datachunk))
        datachunk_sorted = datachunk.sort_values(by = "modelAll_18_conf", ascending = False)
        for i in range(1):
            id = int(datachunk_sorted["LID"].iloc[i])
            vid_num = datachunk_sorted["videoID"].iloc[i]
            #print(id, vid_num)
            for vid in dbx.files_list_folder(os.path.join(dbx_dir, trial, vid_num)).entries:
                if re.match(str(id), vid.name):
                    #print("found")
                    try:
                        subprocess.check_call(['rclone', 'copyto', "cichlidVideo:" + os.path.join(dbx_dir[8:], trial, vid_num, vid.name), os.path.join(data_dir, trial, act, vid.name)])
                    except subprocess.CalledProcessError:
                        print(id, " not found.")
'''

''' extracting frames (VERSION NORM)
for trial in os.listdir(data_dir):
    #print(trial)
    for action in actions:
        #print(action)

        act_dir = os.path.join(data_dir, trial, action)

        if not os.path.exists(os.path.join(act_dir, "NormFrames")):
            os.mkdir(os.path.join(act_dir, "NormFrames"))
        else:
            shutil.rmtree(os.path.join(act_dir, "NormFrames"))
            os.mkdir(os.path.join(act_dir, "NormFrames"))

        for file in os.listdir(os.path.join(act_dir)):
            if os.path.isdir(os.path.join(act_dir, file)):
                #print("Frames: ", file)
                pass
            else:
                #print("Video :", file)
                vidcap = cv2.VideoCapture(os.path.join(act_dir, file))
                success, image = vidcap.read()
                print(success)
                #pdb.set_trace()
                if(success):
                    image = np.asarray(Image.fromarray(image).convert('L')) #normalize
                #pdb.set_trace()
                count = 0
                success = True
                while success:
                    cv2.imwrite(os.path.join(act_dir, "NormFrames", "frame%d.jpg" % count), image)     # save frame as JPEG file
                    success, image = vidcap.read()
                    print("Read frame %d: " % count, success)
                    if(success):
                        image = np.asarray(Image.fromarray(image).convert('L')) #cv2.normalize(image, None, 0, 255, norm_type = cv2.NORM_MINMAX) #image / np.linalg.norm(image)
                    count += 1
                break
#'''

num_frame = 120 #all clips have same length already

'''determines minimum number of frames
for trial in os.listdir(data_dir):
    #print(trial)
    for action in actions:
        #print(action)
        frame_dir = os.path.join(data_dir, trial, action, "Frames")
        count = 0
        for frame in os.listdir(frame_dir):
            count += 1
        print("Count: ", count)
        if count < num_frame:
            num_frame = count
'''

actions_full = ["Quiver", "Other", "Feed Spit", "Feed Multiple", "Reflection", "Bower Spit", "Bower Scoop", "Feed Scoop", "Sand Drop", "Bower Multiple"]
#actions_ordered = actions_full[6, 8, 1, 2, 9, 4, 3, 0, 7, 5]
actions_ordered = ["Feed Scoop", "Feed Spit", "Feed Multiple", "Bower Scoop", "Bower Spit", "Bower Multiple", "Quiver", "Sand Drop", "Other", "Reflection"]
acts_ordered = ["f", "t", "m", "c", "p", "b", "s", "d", "o", "x"]
actions_reverse = actions_ordered.copy()
actions_reverse.reverse()
#print(actions_reverse)
trials_ordered = ["CV10_3", "TI2_4", "TI3_3", "MC6_5", "MC16_2", "MCxCVF1_12a_1", "MCxCVF1_12b_1"]

''' #build plots
for f in range(num_frame): #frames
    index = 1

    plt.figure(figsize = (9, 11.5))
    plt.subplots_adjust(wspace = 0.1, hspace = 0.125)
    plt.suptitle("Cichlid Behavior Classification")
    #plt.use_sticky_edges could fix margins
    plt.margins(0.01)
    for y in range(len(actions)):
        plt.figtext(0.0135, 0.84 - 0.0775*y, actions_ordered[y])

    plt.figtext(0.1455, 0.9, "Pit (CV)") #CV10_3
    plt.figtext(0.2625, 0.9, "Pit (TI)") #TI2_4
    plt.figtext(0.3725, 0.9, "Pit (TI)") #TI3_3
    plt.figtext(0.4685, 0.9, "Castle (MC)") #MC6_5
    plt.figtext(0.5825, 0.9, "Castle (MC)") #MC16_2
    plt.figtext(0.6925, 0.9, "F1 (MCxCV)") #MCxCVF1_12a_1
    plt.figtext(0.805, 0.9, "F1 (MCxCV)") #MCxCVF1_12b_1

    for a in acts_ordered: #rows
        for t in trials_ordered: #cols
            frame_dir = os.path.join(data_dir, t, a, "NormFrames")
            plt.subplot(10, 7, index)
            plt.axis("off")
            for frame in os.listdir(frame_dir):
                if frame == ("frame%d.jpg" % f):
                    #print("Found frame # ", f)
                    img = Image.open(os.path.join(frame_dir, frame))
                    plt.imshow(img, cmap = 'gray')
                    break
            index += 1
    plt.savefig(os.path.join(cur_dir, "Plot_NormFrames", 'plot%d.png' % f))
    plt.close()
'''
#how to convert to gif (run on command line)
#convert -delay 3 'plot%d.png[0-119]' output_norm.gif
#move output_norm.gif to outer directory and cd into it
#gifsicle -b -O2 --colors 256 output_norm.gif


##
#LEFT TO DO:
#convert to video
#make 2x5
##
