#import torchvision.datasets.SBDataset as sbd
from utils.models import *
from utils.tools import *

import torch
import torch.nn.functional as F


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from collections import namedtuple
from itertools import count
from PIL import Image
import torch.optim as optim
import cv2 as cv
from torch.autograd import Variable

try:
    if 'google.colab' in str(get_ipython()):
        from google.colab import drive
        drive.mount('/content/gdrive')
        LOAD = True
        SAVE_MODEL_PATH = '/content/gdrive/MyDrive/models/' + 'q_network'
    else:
        LOAD = False
        SAVE_MODEL_PATH = "./models/q_network"
except NameError:
        LOAD = False
        SAVE_MODEL_PATH = "./models/q_network"





use_cuda = True
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
if use_cuda:
    criterion = nn.MSELoss().cuda()   
else:
    criterion = nn.MSELoss()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
           # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))  #  numbers here need to be adjusted in future
])

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class Agent():
    def __init__(self, classe, alpha=0.2, nu=0.3, threshold=0.5, num_episodes=15, load=False ):
        self.BATCH_SIZE = 32
        self.GAMMA = 0.999
        self.EPS = 1
        self.TARGET_UPDATE = 1
        self.save_path = SAVE_MODEL_PATH
        screen_height, screen_width = 224, 224
        self.n_actions = 9
        self.classe = classe

        self.feature_extractor = FeatureExtractor()
        if not load:
            self.policy_net = DQN(screen_height, screen_width, self.n_actions)
        else:
            self.policy_net = self.load_network()
        self.target_net = DQN(screen_height, screen_width, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        if use_cuda:
          self.feature_extractor = self.feature_extractor.cuda()
          self.target_net = self.target_net.cuda()
          self.policy_net = self.policy_net.cuda()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=1e-6)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.episode_durations = []
        
        self.alpha = alpha # €[0, 1]  Scaling factor
        self.nu = nu # Reward of Trigger
        self.threshold = threshold
        self.actions_history = []
        self.num_episodes = num_episodes
        self.actions_history += [[100]*9]*20

    def save_network(self):
        torch.save(self.policy_net, self.save_path+"_"+self.classe)
        print('Saved')

    def load_network(self):
        if not use_cuda:
            return torch.load(self.save_path+"_"+self.classe, map_location=torch.device('cpu'))
        return torch.load(self.save_path+"_"+self.classe)

    def intersection_over_union(self, box1, box2):
        # ymin, xmin, ymax, xmax = box
        # x_min, x_max, y_min, y_max 
        x11, x21, y11, y21 = box1
        x12, x22, y12, y22 = box2
        
        yi1 = max(y11, y12)
        xi1 = max(x11, x12)
        yi2 = min(y21, y22)
        xi2 = min(x21, x22)
        inter_area = max(((xi2 - xi1) * (yi2 - yi1)), 0)
        # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
        box1_area = (x21 - x11) * (y21 - y11)
        box2_area = (x22 - x12) * (y22 - y12)
        union_area = box1_area + box2_area - inter_area
        # compute the IoU
        iou = inter_area / union_area
        return iou

    def compute_reward(self, actual_state, previous_state, ground_truth):
        res = self.intersection_over_union(actual_state, ground_truth) - self.intersection_over_union(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1
      
    def rewrap(self, coord):
        return min(max(coord,0), 224)
      
    def compute_trigger_reward(self, actual_state, ground_truth):
        res = self.intersection_over_union(actual_state, ground_truth)
        if res>self.threshold:
            return -self.nu
        return self.nu

    def get_best_next_action(self, actions, ground_truth):
        max_reward = -99
        best_action = -1
        actual_equivalent_coord = self.calculate_position_box(actions)
        for i in range(0, 9):
            copy_actions = actions.copy()
            copy_actions.append(i)
            new_equivalent_coord = self.calculate_position_box(copy_actions)
            if i!=0:
                reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord,  ground_truth)
            if reward>max_reward:
                max_reward = reward
                best_action = i

        return best_action

    def do_action(self, image, action, xmin=0, xmax=224, ymin=0, ymax=224):      
        r = action
        alpha_h = self.alpha * (  ymax - ymin )
        alpha_w = self.alpha * (  xmax - xmin )
        #print(r)
        if r == 0:
            pass
        if r == 1:
            xmin += alpha_w
        if r == 2:
            xmax -= alpha_w
        if r == 3:
            ymin += alpha_h
        if r == 4:
            ymax -= alpha_h
        if r == 5:
            xmin -= alpha_w
        if r == 6:
            xmax += alpha_w
        if r == 7:
            ymin -= alpha_h
        if r == 8:
            ymax += alpha_h
        return [self.rewrap(xmin), self.rewrap(xmax), self.rewrap(ymin), self.rewrap(ymax)]

    def select_action(self, state, actions, ground_truth):
        sample = random.random()
        eps_threshold = self.EPS
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()
        else:
          #return np.random.randint(0,9)
            return self.get_best_next_action(actions, ground_truth)

    def select_action_model(self, state):
        with torch.no_grad():
                if use_cuda:
                    inpu = Variable(state).cuda()
                else:
                    inpu = Variable(state)
                qval = self.policy_net(inpu)
                _, predicted = torch.max(qval.data,1)
                action = predicted[0] # + 1
                try:
                  return action.cpu().numpy()[0]
                except:
                  return action.cpu().numpy()


    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states), 
                                         volatile=True).type(Tensor)
        
        state_batch = Variable(torch.cat(batch.state)).type(Tensor)
        if use_cuda:
            state_batch = state_batch.cuda()
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1)).type(LongTensor)
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1)).type(Tensor)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1).type(Tensor)) 

        if use_cuda:
            non_final_next_states = non_final_next_states.cuda()
        d = self.target_net(non_final_next_states) # Eventually replace with policy_net
        next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)

        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute  loss
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    
    def compose_state(self, image, dtype=FloatTensor):
      image_feature = self.get_features(image, dtype)
      image_feature = image_feature.view(1,-1)
      history_flatten = self.actions_history.view(1,-1).type(dtype)
      state = torch.cat((image_feature, history_flatten), 1)
      return state
    
    def get_features(self, image, dtype=FloatTensor):
      global transform
      #image = transform(image)
      image = image.view(1,*image.shape)
      image = Variable(image).type(dtype)
      if use_cuda:
          image = image.cuda()
      feature = self.feature_extractor(image)
      return feature.data

    
    def update_history(self, action):
      action_vector = torch.zeros(9)
      action_vector[action] = 1
      size_history_vector = len(torch.nonzero(self.actions_history))
      if size_history_vector < 9:
          self.actions_history[size_history_vector][action] = 1
      else:
          for i in range(8,0,-1):
              self.actions_history[i][:] = self.actions_history[i-1][:]
          self.actions_history[0][:] = action_vector[:] 
      return self.actions_history

    def calculate_position_box(self, actions, xmin=0, xmax=224, ymin=0, ymax=224):
      alpha_h = self.alpha * (  ymax - ymin )
      alpha_w = self.alpha * (  xmax - xmin )
      real_x_min, real_x_max, real_y_min, real_y_max = 0, 224, 0, 224
      for r in actions:
        if r == 1:
          real_x_min += alpha_w
        if r == 2:
          real_x_max -= alpha_w
        if r == 3:
          real_y_min += alpha_h
        if r == 4:
          real_y_max -= alpha_h
        if r == 5:
          real_x_min -= alpha_w
        if r == 6:
          real_x_max += alpha_w
        if r == 7:
          real_y_min -= alpha_h
        if r == 8:
          real_y_max += alpha_h
        real_x_min, real_x_max, real_y_min, real_y_max = int(self.rewrap(real_x_min)), int(self.rewrap(real_x_max)), int(self.rewrap(real_y_min)), int(self.rewrap(real_y_max))
      return [real_x_min, real_x_max, real_y_min, real_y_max]
    
    def predict_image(self, image, plot=False):
        xmin = 0
        xmax = 224
        ymin = 0
        ymax = 224

        done = False
        all_actions = []
        self.actions_history = torch.ones((9,9))
        state = self.compose_state(image)
        original_coordinates = [xmin, xmax, ymin, ymax]
        original_image = image.clone()
        new_image = image
        while not done:
            action = self.select_action_model(state)
            all_actions.append(action)
            if action == 0:
                next_state = None
                new_equivalent_coord = self.calculate_position_box(all_actions)
                done = True

            else:
                self.actions_history = self.update_history(action)
                new_x_min, new_x_max, new_y_min, new_y_max = self.do_action(image, action,  xmin, xmax, ymin, ymax)
                
                new_coordinates = [new_x_min, new_x_max, new_y_min, new_y_max]
                new_equivalent_coord = self.calculate_position_box(all_actions)
                actual_equivalent_coord = self.calculate_position_box(all_actions[:-2])
                
                
                new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                new_image = transform(new_image)
                
                
                show_new_bdbox(original_image, new_equivalent_coord, color='b')
                
                
                fig,ax = plt.subplots(1)
                ax.imshow(new_image.transpose(0, 2).transpose(0, 1))
                plt.show()
                
                
                next_state = self.compose_state(new_image)
                xmin, xmax, ymin, ymax = new_x_min, new_x_max, new_y_min, new_y_max
            

            # Move to the next state
            state = next_state
            actual_coordinates = new_coordinates
            image = new_image


    def train(self, train_loader):
        xmin = 0
        xmax = 224
        ymin = 0
        ymax = 224

        skipped = 0
        treated = 0
        for i_episode in range(self.num_episodes):
            
            for index in  range(len(train_loader)):
                try:
                    #clear_output()
                    os.system('cls')
                    print("Class : "+str(self.classe))
                    print("EPS = "+str(self.EPS))
                    print("\n\nEpisode : ("+str(i_episode)+"/"+str(self.num_episodes)+")")
                    print("Image ("+str(index)+"/"+str(len(train_loader))+")")
                    print("Treated "+str((treated/(treated+skipped+1))*100     )+"%")
                    image, [oxmin, oxmax, oymin, oymax] = extract(index, train_loader)
                    #image = image.transpose(0, 2).transpose(0, 1)

                    labels = [oxmin, oxmax, oymin, oymax]
                    actual_coordinates = [xmin, xmax, ymin, ymax]
                    original_image = image.clone()
                    ground_truth = labels
            
                    all_actions = []
            
                    # Initialize the environment and state
                    self.actions_history = torch.ones((9,9))
                    state = self.compose_state(image)
                    original_coordinates = [xmin, xmax, ymin, ymax]
                    new_image = image
                    done = False
                    t = 0
                    actual_coordinates = original_coordinates
                    new_coordinates = original_coordinates
                    treated += 1
                    while not done:
                        action = self.select_action(state, all_actions, ground_truth)
                        all_actions.append(action)
                        if action == 0:
                            next_state = None
                            new_equivalent_coord = self.calculate_position_box(all_actions)
                            reward = self.compute_trigger_reward(new_equivalent_coord,  ground_truth)
                            done = True

                        else:
                            self.actions_history = self.update_history(action)
                            new_x_min, new_x_max, new_y_min, new_y_max = self.do_action(image, action,  xmin, xmax, ymin, ymax)
                            
                            new_coordinates = [new_x_min, new_x_max, new_y_min, new_y_max]
                            new_equivalent_coord = self.calculate_position_box(all_actions)
                            actual_equivalent_coord = self.calculate_position_box(all_actions[:-2])
                            
                            
                            new_image = original_image[:, int(new_equivalent_coord[2]):int(new_equivalent_coord[3]), int(new_equivalent_coord[0]):int(new_equivalent_coord[1])]
                            new_image = transform(new_image)
                            
                            """
                            show_new_bdbox(original_image, ground_truth, color='r')
                            show_new_bdbox(original_image, new_equivalent_coord, color='b')
                            
                            
                            fig,ax = plt.subplots(1)
                            ax.imshow(new_image.transpose(0, 2).transpose(0, 1))
                            plt.show()
                            """
                            
                            next_state = self.compose_state(new_image)
                            reward = self.compute_reward(new_equivalent_coord, actual_equivalent_coord, ground_truth)
                            xmin, xmax, ymin, ymax = new_x_min, new_x_max, new_y_min, new_y_max
                        
                        
                        self.memory.push(state, int(action), next_state, reward)

                        # Move to the next state
                        state = next_state
                        actual_coordinates = new_coordinates
                        image = new_image
                        # Perform one step of the optimization (on the target network)
                        self.optimize_model()
                        
                except ValueError:
                    skipped += 1
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if i_episode<5:
                self.EPS -= 0.18
            self.save_network()
            

            print('Complete')
