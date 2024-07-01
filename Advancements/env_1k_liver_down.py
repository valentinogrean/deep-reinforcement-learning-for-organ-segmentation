import numpy as np
import matplotlib.pyplot as plt
import nibabel
import random
import time
import pickle

from skimage import draw
from skimage.transform import resize

##############################################################################################
patients = list()
images = {}
segmentations = {}

def generate_patient_number():
    return random.randrange(1,10)

def get_random_slice(patient_number):
    patient_slices = patients[patient_number]
    return patient_slices[np.random.choice(range(len(patient_slices)))]

def generate_slices_with_organ():
    patients.append([])
    for i in range(9):
        organ_slices = list()

        filename_image = 'D:\\datasets\\abdomenCT-1k\\Patient_' + str(i+1).zfill(2)+ '\\segmentation.nii.gz'
        image = nibabel.load(filename_image).get_fdata()
        shape = image.shape

        for j in range(shape[2]):
            segmentation_slice = image[:,:,j] 
            if (1 in segmentation_slice):
                images[(i+1, j)] = get_CT_data(i+1 , j)
                segmentations[(i+1, j)] = get_full_segmentation(i+1 , j)

                x_start = 0
                x_end = 0
                ground_truth_slice = segmentations[(i+1, j)]
                for z in range(10, 230):
                    ground_truth_line = ground_truth_slice[z,:]
                    if ground_truth_line.max() == 1:
                        if x_start == 0:
                            x_start = z

                            y_start = 0
                            y_end = 0
                            for tz in range(10, 230):
                                if ground_truth_line[tz] == 1:
                                    if y_start == 0:
                                        y_start = tz
                                    y_end = tz
                        x_end = z
                if x_start > 0:
                    organ_slices.append((j, x_start, x_end, y_start, y_end))
        patients.append(organ_slices)
    with open("pickles\\patients_1k_organ1", "wb") as fp:   #Pickling
        pickle.dump(patients, fp)
    with open("pickles\\images_1k_organ1", "wb") as fp:   #Pickling
        pickle.dump(images, fp)
    with open("pickles\\segmentations_1k_organ1", "wb") as fp:   #Pickling
        pickle.dump(segmentations, fp)

def get_CT_data(patient_number, slice_number):

    filename_image = 'D:\\datasets\\abdomenCT-1k\\Patient_' + str(patient_number).zfill(2)+ '\\imaging.nii.gz'
    image = nibabel.load(filename_image).get_fdata()

    image_slice = image[:,:,slice_number]
    image_slice = image_slice[..., np.newaxis]
    image_slice = image_slice.astype(int)

    image_slice_clipped = np.clip(image_slice, -500, 1000)
    image_slice_resized = resize(image_slice_clipped, (256, 256), order=0, preserve_range=True, anti_aliasing=False)
    image_slice_normalized = preprocessing(image_slice_resized, False)

    #plt.imshow(image_slice_normalized)
    #plt.show()

    return np.array(image_slice_normalized, dtype=np.float32)

def get_full_segmentation(patient_number, slice_number):

    filename_image = 'D:\\datasets\\abdomenCT-1k\\Patient_' + str(patient_number).zfill(2)+ '\\segmentation.nii.gz'
    image = nibabel.load(filename_image).get_fdata()

    segmentation_slice = image[:,:,slice_number]
    #segmentation_slice = segmentation_slice[..., np.newaxis]

    segmentation_slice[segmentation_slice == 1] = 1
    segmentation_slice[segmentation_slice == 2] = 0
    segmentation_slice[segmentation_slice == 3] = 0
    segmentation_slice[segmentation_slice == 4] = 0
    segmentation_slice[segmentation_slice == 5] = 0
    segmentation_slice[segmentation_slice == 6] = 0

    segmentation_slice_resized = resize(segmentation_slice, (256, 256), order=0, preserve_range=True, anti_aliasing=False)

    return np.array(segmentation_slice_resized, dtype=np.float32)

def preprocessing(image, z_score=True):
    # Perform z-score normalization
    if z_score:
        # Compute mean and standard deviation
        mean = np.mean(image)
        std = np.std(image)
        # Scaling
        image_normalized = (image - mean) / std
    # Perform scaling normalization between [0,1]
    else:
        # Identify minimum and maximum
        max_value = np.max(image)
        min_value = np.min(image)
        # Scaling
        image_normalized = ((image - min_value) / (max_value - min_value)) * 0.7
    # Update the sample with the normalized image
    return image_normalized

##############################################################################################

cursor_size = 25
cursor_value = 0.8
organ_segmentation_value = 1

class CTEnv1k_Liver_Down:
    def __init__(self, render=False, final_render = False, random_slice=True, patient_number=0, slice_number=0):

        if random_slice == True:
            #generate_slices_with_organ()
            with open("pickles\\patients_1k_organ1", "rb") as fp:   # Unpickling
                self.patients = pickle.load(fp)
            with open("pickles\\images_1k_organ1", "rb") as fp:   # Unpickling
                self.images = pickle.load(fp)
            with open("pickles\\segmentations_1k_organ1", "rb") as fp:   # Unpickling
                self.segmentations = pickle.load(fp)

        self.renderImages = render
        self.finalRender = final_render
        self.random_slice = random_slice
        self.slice_number = slice_number
        self.patient_number = patient_number

        if self.renderImages == True or self.finalRender == True:
            plt.ion()
            self.figure, self.axarr = plt.subplots(3,1) 

        if self.random_slice == True:
            self.patient_number = 1

        # data
        self.own_segmentation = np.zeros((256, 256, 1), dtype=np.float32)

        # instance attributes
        self.min_x = None
        self.max_x = None
        self.current_x = None
        self.new_min_start = 60
        self.new_range_start = 140
        self.new_min_end = 80
        self.new_range_end = 140

        self.done = False

        self.reset()

    def get_random_slice(self, patient_number):
        patient_slices = self.patients[patient_number]
        return patient_slices[np.random.choice(range(len(patient_slices)))]

    def reset(self):
        self.current_step = 0
        self.done = False
        self.y1_start = None
        self.y1_end = None

        if self.random_slice == True:
            self.patient_number += 1
            if self.patient_number == 10:
                self.patient_number = 1
            self.slice_number, self.min_x, self.max_x, self.y_ground_truth_start_upper, self.y_ground_truth_end_upper = self.get_random_slice(self.patient_number)
            self.ground_truth_segmentation = np.array(self.segmentations[(self.patient_number, self.slice_number)], dtype=np.float32)
            self.image = np.array(self.images[(self.patient_number, self.slice_number)], dtype=np.float32)
        else:
            self.image = get_CT_data(self.patient_number, self.slice_number)
            self.ground_truth_segmentation = get_full_segmentation(self.patient_number, self.slice_number)
            x_start = 0
            x_end = 0
            ground_truth_slice = self.ground_truth_segmentation
            for z in range(10, 230):
                ground_truth_line = ground_truth_slice[z,:]
                if ground_truth_line.max() == 1:
                    if x_start == 0:
                        x_start = z
                    x_end = z
            self.min_x = x_start
            self.max_x = x_end
            
            
        y_start = 0
        y_end = 0
        ground_truth_line = self.ground_truth_segmentation[self.max_x,:]
        for tz in range(10, 230):
            if ground_truth_line[tz] == 1:
                if y_start == 0:
                    y_start = tz
                y_end = tz
        
        self.current_x = self.max_x
        self.y_ground_truth_start = y_start
        self.y_ground_truth_end = y_end

        #y_start = 0
        #y_end = 0
        #ground_truth_line = self.ground_truth_segmentation[self.min_x,:]
        #for tz in range(10, 230):
        #    if ground_truth_line[tz] == 1:
        #        if y_start == 0:
        #            y_start = tz
        #        y_end = tz
        #self.y_ground_truth_start_upper = y_start
        #self.y_ground_truth_end_upper = y_end
        
        self.own_segmentation = np.zeros((256, 256, 1), dtype=np.float32)

        rr, cc = draw.line(self.max_x, self.y_ground_truth_start, self.max_x, self.y_ground_truth_end)
        self.own_segmentation[rr, cc] = organ_segmentation_value
        #rr, cc = draw.line(self.min_x, self.y_ground_truth_start_upper, self.min_x, self.y_ground_truth_end_upper)
        #self.own_segmentation[rr, cc] = organ_segmentation_value

        self.y1_start = self.y_ground_truth_start
        self.y1_end = self.y_ground_truth_end

        #render
        if self.renderImages == True:
            self.render()

        return self.prepare_state_slice(), self.slice_number, self.min_x, self.max_x

    def step(self, action):
        reward = 0

        start = action[0]
        #end = action[1]

        old_x = self.current_x
        self.current_x = self.current_x - 3
        if self.current_x < self.min_x:
            self.current_x = self.min_x

        ground_truth_line = self.ground_truth_segmentation[self.current_x,:]

        #calculate line
        start_value = int((start * self.new_range_start) + self.new_min_start)
        #end_value = int((end * self.new_range_end) + self.new_min_end)

        #rewards
        ground_truth_start = 0
        ground_truth_end = 0
        for i in range(40, 220):
            if ground_truth_line[i] == 1:
                if ground_truth_start == 0:
                    ground_truth_start = i
                ground_truth_end = i

        if ground_truth_start == 0:
            ground_truth_start = self.y1_start
            ground_truth_end = self.y1_end

        if np.abs(ground_truth_start - start_value) <= 1:
            reward += 0.5
        else:
            reward += (1.9 / np.abs(ground_truth_start - start_value)) * 0.5
        #if np.abs(ground_truth_end - end_value) <= 1:
        #    reward += 0.5
        #else:
        #    reward += (1.9 / np.abs(ground_truth_end - end_value)) * 0.5

        #if start_value > end_value:
        #    reward -= 0.4

        #update own segmentation - x + 3
        rr, cc = draw.line(old_x, self.y1_start, self.current_x, start_value)
        self.own_segmentation[rr, cc] = organ_segmentation_value
        #rr, cc = draw.line(old_x, self.y1_end, self.current_x, end_value)
        #self.own_segmentation[rr, cc] = organ_segmentation_value
        self.y1_start = start_value
        #self.y1_end = end_value

        # is episode finished?
        if self.current_x <= self.min_x:
            self.done = True
            if self.finalRender == True:
                self.render()

        #render
        if self.renderImages == True:
            self.render()

        # conform to the Gym API
        return self.prepare_state_slice(), reward, self.done, None

    def render(self):
        self.axarr[0].imshow(self.image)
        self.axarr[1].imshow(self.ground_truth_segmentation)
        self.axarr[2].imshow(self.prepare_state_slice()[0,:,:])
    
        # drawing updated values
        self.figure.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()
 
        time.sleep(0.1)

    def prepare_state_slice(self):

        state_slice = np.array(self.image, dtype=np.float32)

        state_slice = np.where(self.own_segmentation != 0, self.own_segmentation, state_slice)

        ceva = state_slice[:,:,0]
        ceva = ceva[np.newaxis, ...]

        return ceva

###########################################################################################################################