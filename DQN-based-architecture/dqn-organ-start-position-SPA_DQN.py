import numpy as np
import tensorflow as tf
import datetime
import nibabel
import matplotlib.pyplot as plt
import time
import argparse
import skimage.feature
import random
import math

from tensorflow import keras
from keras import layers
from skimage import draw
from collections import deque
from skimage.transform import resize

# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

##############################################################################################
#pre-processing and testing in isolation code
##############################################################################################

patients = list()
images = {}
segmentations = {}
full_segmentations = {}

def generate_patient_number():
    return random.randrange(1,11)

def get_random_slice_number(patient_number):
    patient_slices = patients[patient_number]
    return patient_slices[np.random.choice(range(len(patient_slices)))]

def generate_slices_with_organ():
    patients.append([])
    for i in range(10):
        organ_slices = list()

        filename_image = 'D:\\datasets\\multi-organ-segthor-train\\Patient_' + str(i+1).zfill(2)+ '\\segmentation.nii.gz'
        image = nibabel.load(filename_image).get_fdata()
        shape = image.shape

        for j in range(shape[2]):
            segmentation_slice = image[:,:,j] 
            if (2 in segmentation_slice):
               organ_slices.append(j)
               images[(i+1, j)] = get_CT_data(i+1 , j)
               segmentations[(i+1, j)] = get_segmentation_data(i+1 , j)
        patients.append(organ_slices)

def get_CT_data(patient_number, slice_number):

    filename_image = 'D:\\datasets\\multi-organ-segthor-train\\Patient_' + str(patient_number).zfill(2)+ '\\imaging.nii.gz'
    image = nibabel.load(filename_image).get_fdata()

    image_slice = image[:,:,slice_number]
    image_slice = image_slice[..., np.newaxis]
    image_slice = image_slice.astype(int)

    image_slice_clipped = np.clip(image_slice, -500, 1000)
    image_slice_resized = resize(image_slice_clipped, (256, 256), order=0, preserve_range=True, anti_aliasing=False)
    image_slice_normalized = preprocessing(image_slice_resized, False)

    return np.array(image_slice_normalized, dtype=np.float32)


def get_segmentation_data(patient_number, slice_number):

    filename_image = 'D:\\datasets\\multi-organ-segthor-train\\Patient_' + str(patient_number).zfill(2)+ '\\segmentation.nii.gz'
    image = nibabel.load(filename_image).get_fdata()

    segmentation_slice = image[:,:,slice_number]
    #segmentation_slice = segmentation_slice[..., np.newaxis]

    segmentation_slice[segmentation_slice == 1] = 0
    segmentation_slice[segmentation_slice == 2] = 0.2
    segmentation_slice[segmentation_slice == 3] = 0
    segmentation_slice[segmentation_slice == 4] = 0
    segmentation_slice[segmentation_slice == 5] = 0
    segmentation_slice[segmentation_slice == 6] = 0

    segmentation_slice_resized = resize(segmentation_slice, (256, 256), order=0, preserve_range=True, anti_aliasing=False)

    edges = skimage.feature.canny(
        image=segmentation_slice_resized,
        sigma=2,
        low_threshold=0,
        high_threshold=0.1,
    )
    edges = edges[..., np.newaxis]

    return np.array(edges, dtype=np.float32)

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
#ENV
##############################################################################################

cursor_size = 25
cursor_value = 0.8
organ_segmentation_value = 1

class CTEnv:
    def __init__(self, render=False, final_render = False, random_slice=True, patient_number=0, slice_number=0):

        self.renderImages = render
        self.finalRender = final_render
        self.random_slice = random_slice
        self.slice_number = slice_number
        self.patient_number = patient_number

        if self.renderImages == True or self.finalRender == True:
            plt.ion()
            self.figure, self.axarr = plt.subplots(2,1) 

        # data
        self.own_segmentation = np.zeros((256, 256, 1), dtype=np.float32)
        self.action_space = [0, 1, 2, 3]

        # instance attributes
        self.x = None
        self.y = None
        self.x_target = None
        self.y_target = None

        self.max_steps_per_episode = 75
        self.current_step = 0
        self.done = False

        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False

        if self.random_slice == True:
            self.patient_number = generate_patient_number()
            self.slice_number = get_random_slice_number(self.patient_number)
            self.ground_truth_segmentation = np.array(segmentations[(self.patient_number, self.slice_number)], dtype=np.float32)
            self.image = np.array(images[(self.patient_number, self.slice_number)], dtype=np.float32)
        else:
            self.image = get_CT_data(self.patient_number, self.slice_number)
            self.ground_truth_segmentation = get_segmentation_data(self.patient_number, self.slice_number)
        
        self.x = 100
        self.y = 100
        self.calculate_initial_position()

        #render
        if self.renderImages == True:
            self.render()

        return self.prepare_state_slice()

    def calculate_initial_position(self):
        for i in range(255):
            for j  in range(255):
                if self.ground_truth_segmentation[i,j] == organ_segmentation_value:
                    self.x_target = i
                    self.y_target = j
                    break
            if self.ground_truth_segmentation[i,j] == organ_segmentation_value:
                break

    def distance(self, otherX, otherY):
        # find distance to another point
        return math.sqrt((self.x_target - otherX)**2 + (self.y_target - otherY)**2)

    def step(self, action):
        assert action in self.action_space

        # move
        if action == 0:
            self.x += 1
        elif action == 1: 
            self.y += 1
        elif action == 2: 
            self.x -= 1
        elif action == 3: 
            self.y -= 1

        reward = self.distance(self.x, self.y) / -100

        # out of bounds 
        if self.x > 250-cursor_size:
            reward = -0.6
            self.x = 250-cursor_size
        if self.y > 250-cursor_size:
            reward = -0.6
            self.y = 250-cursor_size
        if self.x < cursor_size + 5:
            reward = -0.6
            self.x = cursor_size + 5
        if self.y < cursor_size + 5: # out of bounds
            reward = -0.6
            self.y = cursor_size + 5

        # is episode finished?
        if self.distance(self.x, self.y) < 2:
            reward = 10
            self.done = True
            if self.finalRender == True:
                self.render()
        else:
            self.current_step += 1
            if self.current_step > self.max_steps_per_episode:
                self.done = True
                if self.finalRender == True:
                    self.render()

        #render
        if self.renderImages == True:
            self.render()

        # conform to the Gym API
        return self.prepare_state_slice(), reward, self.done, None

    def render(self):
        self.axarr[0].imshow(self.ground_truth_segmentation)
        self.axarr[1].imshow(self.prepare_state_slice()[:,:,0])
    
        # drawing updated values
        self.figure.canvas.draw()
    
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.figure.canvas.flush_events()
 
        time.sleep(0.1)

    def prepare_state_slice(self):

        state_slice = np.array(self.image, dtype=np.float32)
        rr, cc = draw.line(0, self.y, cursor_size, self.y)
        state_slice[rr, cc] = 0.8
        rr, cc = draw.line(255, self.y, 255-cursor_size, self.y)
        state_slice[rr, cc] = 0.8
        rr, cc = draw.line(self.x, 0, self.x, cursor_size)
        state_slice[rr, cc] = 0.8
        rr, cc = draw.line(self.x, 255, self.x, 255-cursor_size)
        state_slice[rr, cc] = 0.8

        return state_slice

###########################################################################################################################
#MODEL
###########################################################################################################################

num_actions = 4

def create_q_model():
    inputs = layers.Input(shape=(256, 256, 1, )) 
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same") (inputs)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same") (x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3),  activation="relu", padding="same") (x)
    x = layers.Conv2D(128, (3, 3),  activation="relu", padding="same") (x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, (3, 3),  activation="relu", padding="same") (x)
    x = layers.Conv2D(256, (3, 3),  activation="relu", padding="same") (x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)  
    
    x = layers.Conv2D(512, (3, 3),  activation="relu") (x)
    x = layers.Conv2D(512, (3, 3),  activation="relu") (x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x) 
    
    #x = layers.Flatten()(x)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(2048,  activation="relu") (x)
    x = layers.Dense(1024,  activation="relu") (x)
    action = layers.Dense(num_actions, activation="linear")(x)
    
    return keras.Model(inputs=inputs, outputs=action)

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()


#################################################################################################################################
#HYPERPARAMTERS
#################################################################################################################################

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon_random_frames = 1000
epsilon_greedy_frames = 500000.0
max_memory_length = 195000
update_after_actions = 4
update_target_network = 10000
loss_function = keras.losses.Huber()

##############################################################################################################################
#MAIN DQN LOOP - train
##############################################################################################################################

if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
    args = parser.parse_args()

    #logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = './logs_dqn_segmentation/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    #generate slices that contain organs
    generate_slices_with_organ()

    episode_reward_history.append(-30)
    episode_reward_history.append(-30)

    # create env
    env = CTEnv(render=False, final_render = False)
    best_episode_reward = -10000
    best_running_reward = -10000

    while True:  # Run until solved
        state = env.reset()
        episode_reward = 0
        done = False

        while True:
            frame_count += 1

            # Use epsilon-greedy for exploration
            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                # Take random action
                action = np.random.choice(num_actions)
            else:
                # Predict action Q-values
                # From environment state
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()

            # Decay probability of taking random action
            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)

            state_for_history = np.array(state, np.float32)
            # Apply the sampled action in our environment
            state_next, reward, done, _ = env.step(action)
            state_next_for_history = np.array(state_next, np.float32)

            episode_reward += reward

            # Save actions and states in replay buffer
            action_history.append(action)
            state_history.append(state_for_history)
            done_history.append(done)
            rewards_history.append(reward)
            state = state_next

            # Update every fourth frame and once batch size is over 32
            if frame_count % update_after_actions == 0 and len(done_history) > batch_size + 1:
                # Get indices of samples for replay buffers
                indices = np.random.choice(range(len(done_history) - 1), size=batch_size)
                indices_next = [x+1 for x in indices]

                # Using list comprehension to sample from replay buffer
                state_sample = np.array([state_history[i] for i in indices], dtype=np.float32)                
                state_next_sample =np.array([state_history[i] for i in indices_next], dtype=np.float32)

                rewards_sample = [rewards_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor(
                    [float(done_history[i]) for i in indices]
                )
                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                #future_rewards = model_target.predict(state_next_sample, verbose=0)
                state_next_tensor = tf.convert_to_tensor(state_next_sample)
                future_rewards = model_target(state_next_tensor, training=False)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(
                    future_rewards, axis=1
                )
                # If final frame set the last value to -1
                #updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, num_actions)

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = model(state_sample)
                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)
                # Backpropagation
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if frame_count % update_target_network == 0:
                # update the the target network with new weights
                model_target.set_weights(model.get_weights())
                model_target.save_weights("model_dqn_segm/target_weights.h5")
                # Log details
                template = "running reward: {:.2f} at episode {}, frame count {}"
                print(template.format(running_reward, episode_count, frame_count))

            # Limit the state and reward history
            if len(rewards_history) > max_memory_length:
                del rewards_history[:1]
                del state_history[:1]
                del action_history[:1]
                del done_history[:1]

            if done:
                break
        # Update running reward to check condition for solving
        episode_reward_history.append(episode_reward)
        if len(episode_reward_history) > 100:
            del episode_reward_history[:1]
        running_reward = np.mean(episode_reward_history)

        with summary_writer.as_default():
            tf.summary.scalar("Episode reward", episode_reward, step=episode_count)
            tf.summary.scalar("Running reward - mean of last 100 episodes", running_reward, step=episode_count)
            summary_writer.flush()

        episode_count += 1

        if(episode_reward > best_episode_reward):
            model.save_weights("model_dqn_segm/best_episode_weights.h5")
            best_episode_reward = episode_reward

        if(running_reward > best_running_reward):
            model.save_weights("model_dqn_segm/best_running_weights.h5")
            best_running_reward = running_reward

        if running_reward > 1000:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            break