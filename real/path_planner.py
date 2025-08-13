import numpy as np

import torch
import torch.nn as nn

from robot import robot

import time

# === SIREN trajectory model ===
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=5.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights(is_first)
    
    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class TrajectoryPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(11, 64, is_first=True),
            SineLayer(64, 64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def forward(self, t, q0, qT):
        B, T = t.shape[:2]
        q0_expand = q0.unsqueeze(1).expand(-1, T, -1)
        qT_expand = qT.unsqueeze(1).expand(-1, T, -1)
        inp = torch.cat([t, q0_expand, qT_expand], dim=-1)
        return self.net(inp.view(-1, 11)).view(B, T, 5)

class path_planner:
    def __init__(self, robot, model_path):
        self.robot = robot
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model instance first
        self.model = TrajectoryPINN().to(self.device)
        
        # Load the state dictionary
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        
        self.path = None
        self.path_index = None
        self.time = None

    def generate_path(self, current_pos, target_pos, time):
        '''
        Takes the current position and the target position in the
        form of two 5x1 arrays in radians and the time as a float
        in order to assign the path from the model.
        '''
        self.time = time
        
        # Converting time to array and reshaping for the model
        time_array = np.linspace(0, 1, 200)  # Note: normalized to [0,1] as in training
        time_tensor = torch.tensor(time_array, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)
        
        # Convert positions to tensors
        current_pos_tensor = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
        target_pos_tensor = torch.tensor(target_pos, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            path = self.model(time_tensor, current_pos_tensor.unsqueeze(0), target_pos_tensor.unsqueeze(0))
            
        self.path = path.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
        self.path_index = 0
        
        #print(f"Generated path shape: {self.path.shape}")

    def update_path(self):
        if self.path_index >= 200:
            print('Path Completed')
            return
        
        curr_pos = self.path[self.path_index, :]

        positions = [curr_pos[i] for i in range(len(curr_pos))]

        servos = ['shoulder_pan',
                  'shoulder_lift',
                  'elbow_flex',
                  'wrist_flex',
                  'wrist_roll']
        
        self.robot.move_joints(servos, positions, radians = True)
        time.sleep(self.time / 200)

        self.path_index += 1

    def play_path(self):
        for _ in range(200):
            self.update_path()