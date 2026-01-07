import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import math
from collections import OrderedDict
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T
# from DriveTransformer.team_code.pid_controller import DecouplePIDController
from DriveTransformer.team_code.pure_pursuit_controller import PurePursuitController
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import (get_dist_info, init_dist, load_checkpoint,
                        wrap_fp16_model)
from mmcv.datasets.pipelines import Compose
from mmcv.parallel.collate import collate as  mm_collate_to_batch_form
from mmcv.core.bbox import get_box_type
from team_code.planner import RoutePlanner
from pyquaternion import Quaternion
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import random

# Optimize matplotlib for faster rendering
plt.ioff()  # Turn off interactive mode
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['agg.path.chunksize'] = 10000

IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)


def get_entry_point():
    return 'DriveTransformerAgentDiffusion_Small_MLP_W_ES'


class DriveTransformerAgentDiffusion_Small_MLP_W_ES(autonomous_agent.AutonomousAgent):
    """
    Drive TransformerAgentDiffusion Agent
    """
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        # self.controller = DecouplePIDController(speed_k_p=2.0, speed_k_i=0.8, speed_k_d=1.5, steer_k_p=1.5, steer_k_i=0.2, steer_k_d=0.2)
        self.controller = PurePursuitController(lookahead_distance=4.0, wheelbase=2.89, max_throttle=1.0,
                                                brake_speed=0.2, brake_ratio=0.8, speed_KP=5.0, speed_KI=0.5,
                                                speed_KD=1.0, speed_n=20, clip_delta=0.25, dt=0.2)
        self.config_path = path_to_conf_file.split('+')[0]
        self.ckpt_path = path_to_conf_file.split('+')[1]
        if IS_BENCH2DRIVE:
            self.save_name = path_to_conf_file.split('+')[-1]
        else:
            self.config_path = path_to_conf_file
            self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.device = "cuda"
        cfg = Config.fromfile(self.config_path)
        self.cameras = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
        #remap path
        if hasattr(cfg, 'plugin'):
            if cfg.plugin:
                import importlib
                if hasattr(cfg, 'plugin_dir'):
                    plugin_dir = cfg.plugin_dir
                    _module_dir = os.path.dirname(plugin_dir)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]
                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)  
        self.model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        # load checkpoint
        if self.ckpt_path != "None":
            ckpt = torch.load(self.ckpt_path)
            ckpt = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            for key, value in ckpt.items():
                new_key = key.replace("model.","").replace("._orig_mod", "")
                new_state_dict[new_key] = value
            print(self.model.load_state_dict(new_state_dict, strict = False))
        wrap_fp16_model(self.model)
        self.model.to(self.device)
        self.model.eval()

        self.test_pipeline = []
        self.past_ego_pos_cache = []
        self.cache_lenth = 20
        # pipeline
        for test_pipeline in cfg.test_pipeline:
            if test_pipeline["type"] not in ['LoadMultiViewImageFromFiles','LoadAnnotations3D', "CustomObjectRangeFilter", "CustomObjectNameFilter", "TrajPreprocess"]:
                self.test_pipeline.append(test_pipeline)
            if test_pipeline["type"] == "CustomFormatBundle3D":
                test_pipeline["collect_keys"] = ['lidar2img', 'cam_intrinsic','timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape']
            if test_pipeline["type"] == "CustomCollect3D":
                test_pipeline["keys"] = ['img', 'ego_his_trajs', 'ego_lcf_feat', 'ego_fut_cmd', 'prev_exists', 'index', 'lidar2img', 'cam_intrinsic', 'timestamp', 'ego_pose', 'ego_pose_inv', 'pad_shape']
        self.test_pipeline = Compose(self.test_pipeline)
        self.save_path = None
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.lat_ref, self.lon_ref = 42.0, 2.0
        self.pid_metadata = {}
        self.prev_control_cache = []
        self.prev_control_list = []
        self.step_time_avg = []
        
        # Agent tracking in ego coordinates
        self.tracked_agents = {}  # dict: agent_id -> {'positions': [(step, x_ego, y_ego, ego_pose)], 'world_positions': [(step, x_world, y_world)], 'last_seen': step}
        self.next_agent_id = 0
        self.tracking_max_distance = 3.0  # meters - max distance to associate detections across frames
        self.tracking_max_age = 20  # frames - remove tracks not seen for this many frames
        self.ego_pose_history = {}  # dict: step -> ego_pose (for transforming historical positions)
        
        # Forecasting method: 'moving_average' or 'linear_fit'
        self.forecast_method = 'moving_average'  # Change this to switch methods

        # string = pathlib.Path(os.environ['ROUTES']).stem + '_'
        string = self.save_name
        self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
        self.save_path.mkdir(parents=True, exist_ok=False)

        (self.save_path / 'combined').mkdir()
        (self.save_path / 'meta').mkdir()

        # transform from lidar to image coordinates
        self.lidar2img = {
        'CAM_FRONT':np.array([[ 1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -9.52000000e+02],
                              [ 0.00000000e+00,  4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                              [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -1.19000000e+00],
                              [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[ 6.03961325e-14,  1.39475744e+03,  0.00000000e+00, -9.20539908e+02],
                                   [-3.68618420e+02,  2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                   [-8.19152044e-01,  5.73576436e-01,  0.00000000e+00, -8.29094072e-01],
                                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[ 1.31064327e+03, -4.77035138e+02,  0.00000000e+00,-4.06010608e+02],
                                    [ 3.68618420e+02,  2.58109396e+02, -1.14251841e+03,-6.47296750e+02],
                                    [ 8.19152044e-01,  5.73576436e-01,  0.00000000e+00,-8.29094072e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[-5.60166031e+02, -8.00000000e+02,  0.00000000e+00, -1.28800000e+03],
                            [ 5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                            [ 1.22464680e-16, -1.00000000e+00,  0.00000000e+00, -1.61000000e+00],
                            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
        'CAM_BACK_LEFT':np.array([[-1.14251841e+03,  8.00000000e+02,  0.00000000e+00, -6.84385123e+02],
                                  [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                  [-9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]),
  
        'CAM_BACK_RIGHT': np.array([[ 3.60989788e+02, -1.34723223e+03,  0.00000000e+00, -1.04238127e+02],
                                    [ 4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                    [ 9.39692621e-01, -3.42020143e-01,  0.00000000e+00, -4.92889531e-01],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        }
        # transform from lidar to camera coordinates
        self.lidar2cam = {
        'CAM_FRONT':np.array([[ 1.  ,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -1.  , -0.24],
                              [ 0.  ,  1.  ,  0.  , -1.19],
                              [ 0.  ,  0.  ,  0.  ,  1.  ]]),
        'CAM_FRONT_LEFT':np.array([[ 0.57357644,  0.81915204,  0.  , -0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [-0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_FRONT_RIGHT':np.array([[ 0.57357644, -0.81915204, 0.  ,  0.22517331],
                                   [ 0.        ,  0.        , -1.  , -0.24      ],
                                   [ 0.81915204,  0.57357644,  0.  , -0.82909407],
                                   [ 0.        ,  0.        ,  0.  ,  1.        ]]),
        'CAM_BACK':np.array([[-1. ,  0.,  0.,  0.  ],
                             [ 0. ,  0., -1., -0.24],
                             [ 0. , -1.,  0., -1.61],
                             [ 0. ,  0.,  0.,  1.  ]]),
     
        'CAM_BACK_LEFT':np.array([[-0.34202014,  0.93969262,  0.  , -0.25388956],
                                  [ 0.        ,  0.        , -1.  , -0.24      ],
                                  [-0.93969262, -0.34202014,  0.  , -0.49288953],
                                  [ 0.        ,  0.        ,  0.  ,  1.        ]]),
  
        'CAM_BACK_RIGHT':np.array([[-0.34202014, -0.93969262,  0.  ,  0.25388956],
                                  [ 0.        ,  0.         , -1.  , -0.24      ],
                                  [ 0.93969262, -0.34202014 ,  0.  , -0.49288953],
                                  [ 0.        ,  0.         ,  0.  ,  1.        ]])
        }
        
        # camera intrinsics
        self.cam_intrinsics = {
        'CAM_FRONT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                              [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                   [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_FRONT_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                    [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
        'CAM_BACK':np.array([[560.16603057,   0.        , 800.        ,   0.        ],
                             [  0.        , 560.16603057, 450.        ,   0.        ],
                             [  0.        ,   0.        ,   1.        ,   0.        ],
                             [  0.        ,   0.        ,   0.        ,   1.        ]]),
     
        'CAM_BACK_LEFT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
  
        'CAM_BACK_RIGHT':np.array([[1.14251841e+03, 0.00000000e+00, 8.00000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 1.14251841e+03, 4.50000000e+02, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        }
        
        self.lidar2ego = np.array([[ 0. ,  1. ,  0. , -0.39],
                                   [-1. ,  0. ,  0. ,  0.  ],
                                   [ 0. ,  0. ,  1. ,  1.84],
                                   [ 0. ,  0. ,  0. ,  1.  ]])
        topdown_extrinsics =  np.array([[1.0, 0.0, 0.0, 0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 50.0], [0.0, 0.0, 0.0, 1.0]])
        topdown_intrinsics = np.array([[548.993771650447, 0.0, 256.0, 0], [0.0, 548.993771650447, 256.0, 0], [0.0, 0.0, 1.0, 0], [0, 0, 0, 1.0]])
        self.coor2topdown = topdown_intrinsics @ topdown_extrinsics
        
        self.all_sensors =  {
                # camera rgb
                'CAM_FRONT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                'CAM_FRONT_LEFT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                'CAM_FRONT_RIGHT':{
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                'CAM_BACK':{
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                'CAM_BACK_LEFT':{
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                'CAM_BACK_RIGHT':{
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                'IMU':{
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                'GPS':{
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                # speed
                'SPEED':{
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                'bev': {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }
        }
   
    def _init(self):
        # get gps reference point
        try:
            locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
            lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0
            def equations(vars):
                x, y = vars
                eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
                eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
                return [eq1, eq2]
            initial_guess = [0, 0]
            solution = fsolve(equations, initial_guess)
            self.lat_ref, self.lon_ref = solution[0], solution[1]
        except Exception as e:
            print(e, flush=True)
            self.lat_ref, self.lon_ref = 0, 0
        # route planner
        self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._route_planner.set_route(self._plan_gps_HACK, True)
        self._command_planner = RoutePlanner(7.5, 25.0, 257, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
        self._command_planner.set_route(self._global_plan, True)
        self.initialized = True



        # determine if we should save the data
        # self.should_save_data = random.random() < 0.4
        self.should_save_data = True

        # diffusion-es replanning frequency control
        self.replan_every_n_steps = 10  # Replan every N steps
        self.last_replan_step = -1  # Track when we last replanned
        self.cached_trajectory = None  # Cache the planned trajectory
  
    def sensors(self):
        sensors = []
        select_sensor_names = self.cameras + ['IMU','GPS','SPEED']
        if IS_BENCH2DRIVE:
            select_sensor_names.append('bev')
        for key in select_sensor_names:
            sensors.append(self.all_sensors[key])
        return sensors

    def tick(self, input_data):
        self.step += 1
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
        imgs = {}
        for cam in self.cameras:
            img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            imgs[cam] = img

        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]
  
        pos = self.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        far_node, far_command = self._command_planner.run_step(pos)

        if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        result = {
                'imgs': imgs,
                'gps': gps,
                'pos':pos,
                'speed': speed,
                'compass': compass,
                'bev': bev,
                'acceleration':acceleration,
                'angular_velocity':angular_velocity,
                'command_near':near_command,
                'command_near_xy':near_node,
                'command_far':far_command,
                'command_far_xy':far_node,    
                }
        return result
    
    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        tick_data = self.tick(input_data)

        results = {}
        results['lidar2img'] = []
        results['lidar2cam'] = []
        results['cam_intrinsic'] = []
        results['img'] = []
        results['folder'] = ' '
        results['scene_token'] = ' '  
        results['frame_idx'] = 0
        results['timestamp'] = np.array(self.step / 20)
        results['box_type_3d'], _ = get_box_type('LiDAR')
        results['index'] = self.step
        results['prev_exists'] = (self.step > 1) 
        for cam in self.cameras: 
            results['lidar2img'].append(self.lidar2img[cam])
            results['lidar2cam'].append(self.lidar2cam[cam])
            results['cam_intrinsic'].append(self.cam_intrinsics[cam])
            results['img'].append(tick_data['imgs'][cam])
        results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
        results['lidar2cam'] = np.stack(results['lidar2cam'],axis=0)
  
        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
        ego_theta = -raw_theta + np.pi/2
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
        # can bus
        can_bus = np.zeros(18)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180 
        results['can_bus'] = can_bus
        results['aug_config'] = {'resize': 0.66, 'resize_dims': (1056, 594), 'crop': (0, 210, 1056, 594), 'flip': False, 'rotate': 0, 'rotate_3d': 0}
        # ego_lcf_feat
        ego_lcf_feat = np.zeros(9)
        ego_lcf_feat[0] = tick_data['speed']
        ego_lcf_feat[2:4] = can_bus[10:12].copy()
        ego_lcf_feat[4] = can_bus[15]
        ego_lcf_feat[5] = 4.89238167
        ego_lcf_feat[6] = 1.83671331
        ego_lcf_feat[7] = tick_data['speed']
        ego_lcf_feat[8] = 0 if len(self.prev_control_cache) < 2 else self.prev_control_cache[0].steer
        results['ego_lcf_feat'] = ego_lcf_feat
        # command
        command = np.zeros(140)
        command[0:6] = self.command2hot(tick_data['command_far'])
        command[70:76] = self.command2hot(tick_data['command_near'])
        theta_to_lidar = raw_theta
        command_near_xy = np.array([tick_data['command_near_xy'][0]-can_bus[0],-tick_data['command_near_xy'][1]-can_bus[1]])
        command_far_xy = np.array([tick_data['command_far_xy'][0]-can_bus[0],-tick_data['command_far_xy'][1]-can_bus[1]])  
        rotation_matrix = np.array([[np.cos(theta_to_lidar),-np.sin(theta_to_lidar)],[np.sin(theta_to_lidar),np.cos(theta_to_lidar)]])
        local_command_near_xy = rotation_matrix @ command_near_xy
        local_command_far_xy = rotation_matrix @ command_far_xy
        command[6:70] = self.pos2posemb(local_command_far_xy)
        command[76:140] = self.pos2posemb(local_command_near_xy)
        results['ego_fut_cmd'] = command
        # ego position
        ego2world = np.eye(4)
        ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world[0:2,3] = can_bus[0:2]
        lidar2global = ego2world @ self.lidar2ego
        results['l2g_r_mat'] = lidar2global[0:3,0:3]
        results['l2g_t'] = lidar2global[0:3,3]
        current_pose = lidar2global
        current_pose_inv = self.invert_pose(current_pose)
        results['ego_pose'] = current_pose
        results['ego_pose_inv'] = current_pose_inv
        # Store current ego pose for agent tracking transformations
        self.ego_pose_history[self.step] = current_pose.copy()   
        # ego past trajectory
        past_pose_1 = self.past_ego_pos_cache[-10] if len(self.past_ego_pos_cache) >= 10 else lidar2global
        past_pose_2 = self.past_ego_pos_cache[0] if len(self.past_ego_pos_cache) == 20 else lidar2global   
        past2current_1 = current_pose_inv @ past_pose_1
        past2current_2 = current_pose_inv @ past_pose_2
        past2current_1_xy = past2current_1[0:2,3]
        past2current_2_xy = past2current_2[0:2,3]
        ego_his_trajs = np.zeros((2,2))
        ego_his_trajs[0] = past2current_1_xy - past2current_2_xy
        ego_his_trajs[1] = -past2current_1_xy
        results['ego_his_trajs'] = ego_his_trajs
        if len(self.past_ego_pos_cache)==20:
            self.past_ego_pos_cache.pop(0)
        self.past_ego_pos_cache.append(current_pose)
        if self.step%2 == 1:
            return self.prev_control
        stacked_imgs = np.stack(results['img'],axis=-1)
        results['img_shape'] = stacked_imgs.shape
        results['ori_shape'] = stacked_imgs.shape
        results['pad_shape'] = stacked_imgs.shape
        # data pipeline
        results = self.test_pipeline(results)      
        input_data_batch = mm_collate_to_batch_form([results], samples_per_gpu=1)
        for key, data in input_data_batch.items():
            if key != 'img_metas':
                if isinstance(data,torch.Tensor):
                    input_data_batch[key] = data.to(self.device)
                    if input_data_batch[key].dtype==torch.float64:
                        input_data_batch[key] = input_data_batch[key].to(torch.float32)
                elif isinstance(data,list):
                    if torch.is_tensor(data[0]):
                        input_data_batch[key][0] = input_data_batch[key][0].to(self.device)
                        if input_data_batch[key][0].dtype==torch.float64:
                            input_data_batch[key][0] = input_data_batch[key][0].to(torch.float32)
        step_start_time = time.time()
        # model inference
        # with autocast(dtype=torch.float16):  # Force fp16 instead of bf16
        output_data_batch = self.model(input_data_batch, return_loss=False, rescale=True)

        # output_data_batch as predictions for the whole batch
        # dict_keys(['boxes_3d', 'scores_3d', 'labels_3d', 'trajs_3d', 'map_boxes_3d', 'map_scores_3d', 'map_labels_3d', 'map_pts_3d', 'ego_fut_cmd', 'ego_fut_preds_fix_time', 'ego_fut_preds_fix_dist'])
        # output_data_batch[0]['trajs_3d']: shape of (100, 6, 12)
        # output_data_batch[0]['labels_3d'] : shape of (100,) discrete for each class
        # output_data_batch[0]['scores_3d'] : shape of (100,) from 0 to 1
        # output_data_batch[0]['map_labels_3d']: shape of (33,)
        # output_data_batch[0]['map_scores_3d']: shape of (33,) from 0 to 1
        # output_data_batch[0]['map_pts_3d']: shape of (33, 20, 2)
        # output_data_batch[0]['ego_fut_preds_fix_time']: shape of (1,1,30,2)
        # output_data_batch[0]['ego_fut_preds_fix_dist']: shape of (1,1,20,2)

        # ========================================================================
        # DIFFUSION SAMPLING: Replace fixed trajectories with diffusion-generated ones
        # ========================================================================
        # Verify diffusion is enabled
        if not hasattr(self.model.pts_bbox_head, 'use_diffusion_loss'):
            raise RuntimeError("Model does not have use_diffusion_loss attribute. Wrong model loaded?")
        if not self.model.pts_bbox_head.use_diffusion_loss:
            raise RuntimeError("use_diffusion_loss is False. This agent requires diffusion to be enabled.")
        if not hasattr(self.model.pts_bbox_head, 'intermediate_ego_query'):
            raise RuntimeError("Model does not have intermediate_ego_query. Model may not be storing intermediate queries.")
        if self.model.pts_bbox_head.intermediate_ego_query is None:
            raise RuntimeError("intermediate_ego_query is None. Forward pass did not populate intermediate queries.")

        # Get ego tokens from intermediate queries (stored during forward pass)
        intermediate_ego_query = self.model.pts_bbox_head.intermediate_ego_query  # [L+1, B, N_ego, D]

        num_layers = intermediate_ego_query.shape[0]
        batch_size = intermediate_ego_query.shape[1]

        # Extract ego tokens (one per decoder layer)
        # Each ego_query_tokens_at_layer is [B, N_ego, D], we take the first token [B, D]
        ego_tokens = []
        for layer_idx in range(num_layers):
            ego_query_tokens_at_layer = intermediate_ego_query[layer_idx]  # [B, N_ego, D]
            # Extract the ego token (assuming it's a single token, take the first one)
            ego_token = ego_query_tokens_at_layer[:, 0, :]  # [B, D]
            ego_tokens.append(ego_token)

        # Get drivable area map and vehicle info for reward function
        # Compute these once before the reward function to avoid repeated computation
        drivable_info = self.get_drivable_area_map(grid_size=200, grid_resolution=0.2)
        drivable_map_for_reward = drivable_info['drivable_map']
        grid_resolution_for_reward = drivable_info['grid_resolution']
        vehicles_info_for_reward = self.get_vehicles_info()

        # Define reward function for diffusion-es that avoids collisions and stays in drivable area
        def reward_fn(trajectories):
            """
            Reward function that avoids collisions and stays in drivable area.
            Emphasizes earlier timesteps more heavily.
            trajectories: [B*num_particles, num_traj_tokens * 2] in unnormalized absolute coordinates
            Returns: rewards [B*num_particles] (higher is better)
            """
            # Trajectories are in unnormalized absolute coordinates, reshape to [B*N, num_traj_tokens, 2]
            traj_reshaped = trajectories.reshape(trajectories.shape[0], -1, 2)
            num_particles = traj_reshaped.shape[0]
            num_timesteps = traj_reshaped.shape[1]

            # Convert trajectories to numpy for processing
            if isinstance(traj_reshaped, torch.Tensor):
                traj_np = traj_reshaped.cpu().numpy()
            else:
                traj_np = traj_reshaped

            # Compute drivability scores for all waypoints [num_particles, num_timesteps]
            drivable_scores = self.get_waypoints_in_drivable_area(
                traj_np, drivable_map_for_reward, grid_resolution_for_reward
            )
            if isinstance(drivable_scores, torch.Tensor):
                drivable_scores = drivable_scores.cpu().numpy()

            # Compute collision scores for all waypoints [num_particles, num_timesteps]
            if len(vehicles_info_for_reward['vehicle_ids']) > 0:
                agent_bbx = vehicles_info_for_reward['bbox_corners_ego']
                agent_vel = vehicles_info_for_reward['ego_velocities']
                collision_scores = self.get_collision_points(
                    torch.from_numpy(traj_np).float() if not isinstance(traj_reshaped, torch.Tensor) else traj_reshaped,
                    agent_bbx, agent_vel, dt=0.2, safety_margin=1.0  # Increased safety margin
                )
                if isinstance(collision_scores, torch.Tensor):
                    collision_scores = collision_scores.cpu().numpy()
            else:
                collision_scores = np.zeros((num_particles, num_timesteps))

            # Create time-based weights: exponential decay emphasizing earlier timesteps
            # Earlier timesteps get higher weight (more important to get right)
            time_weights = np.exp(-0.1 * np.arange(num_timesteps))  # Exponential decay
            time_weights = time_weights / time_weights.sum()  # Normalize to sum to 1

            # Compute rewards for each trajectory
            rewards = np.zeros(num_particles, dtype=np.float32)
            for i in range(num_particles):
                # Drivability reward: weighted sum of drivable scores (0-1 per timestep)
                # Higher = more time in drivable area
                drivability_reward = (drivable_scores[i] * time_weights).sum()

                # Collision penalty: weighted sum of collision scores (0 or 1 per timestep)
                # Lower = fewer collisions
                collision_penalty = (collision_scores[i] * time_weights).sum()

                # Forward progress reward: encourage moving forward (positive y)
                forward_progress = (traj_np[i, :, 1] * time_weights).sum()  # y = forward direction

                # Straightness reward: penalize excessive lateral movement
                lateral_deviation = np.abs(traj_np[i, :, 0] * time_weights).sum()  # x = left direction

                # Combine rewards with weights
                reward = (
                    10.0 * drivability_reward +      # Stay in drivable area (0-10)
                    -100.0 * collision_penalty +     # Avoid collisions (large penalty)
                    0.0 * forward_progress +         # Make forward progress
                    -0.0 * lateral_deviation         # Minimize lateral deviation
                )

                rewards[i] = reward

            # Convert back to torch tensor
            rewards_tensor = torch.from_numpy(rewards).to(trajectories.device)
            return rewards_tensor

        # Check if we need to replan or can reuse cached trajectory
        steps_since_last_replan = self.step - self.last_replan_step
        should_replan = (self.cached_trajectory is None or
                        steps_since_last_replan >= self.replan_every_n_steps)

        if should_replan:
            # Sample trajectory using diffusion-es with reward guidance
            # Note: sample_trajectory_diffusion_es now returns a dict with unnormalized trajectories
            start_time = time.time()
            diffusion_es_outputs = self.model.pts_bbox_head.sample_trajectory_diffusion_es(
                ego_tokens=ego_tokens,
                reward_fn=reward_fn,
                batch_size=batch_size,
                num_iterations=3,  # Number of refine-and-select iterations
                num_inference_steps=50,  # Denoising steps per iteration
                num_particles=32,  # Population size
                top_k=8,  # Select top 8 particles each iteration
                renoise_ratio=0.8,  # Renoise to 80% through denoising
                eta=1.0,  # Non-Deterministic sampling
                clip_denoised=False,
                device=intermediate_ego_query.device
            )
            end_time = time.time()
            print(f"Diffusion-ES sampling completed in {end_time - start_time:.2} seconds.", flush=True)

            # Extract best trajectory from diffusion output
            # best_trajectory is on CPU as [B, num_traj_tokens, 2], we need to extract batch 0
            sampled_traj_absolute = diffusion_es_outputs['best_trajectory'][0]  # [num_traj_tokens, 2]

            # Convert to numpy if it's a tensor
            if isinstance(sampled_traj_absolute, torch.Tensor):
                sampled_traj_absolute_np = sampled_traj_absolute.numpy()
            else:
                sampled_traj_absolute_np = sampled_traj_absolute

            # Calculate truncation index based on displacement threshold
            truncation_idx = len(sampled_traj_absolute_np)  # Default: use full trajectory
            max_displacement = 5.0  # meters - max allowed displacement between consecutive waypoints

            for i in range(1, len(sampled_traj_absolute_np)):
                displacement = np.linalg.norm(sampled_traj_absolute_np[i] - sampled_traj_absolute_np[i-1])
                if displacement > max_displacement:
                    truncation_idx = i
                    break

            # Truncate trajectory if displacement between consecutive points is too large
            ego_traj_fix_time_truncated = sampled_traj_absolute_np[:truncation_idx]

            # Ensure we have at least 2 waypoints after truncation
            if len(ego_traj_fix_time_truncated) < 2:
                ego_traj_fix_time_truncated = np.array([[0.0, 0.0], [0.5, 0.0]])

            # Cache the trajectory and update replan step
            self.cached_trajectory = ego_traj_fix_time_truncated.copy()
            self.last_replan_step = self.step
        else:
            # Reuse cached trajectory
            ego_traj_fix_time_truncated = self.cached_trajectory
            print(f"Reusing cached trajectory (step {steps_since_last_replan}/{self.replan_every_n_steps})", flush=True)

        # Update output_data_batch so visualization shows truncated trajectory
        # Model outputs [left, forward] - no swap needed
        truncated_tensor = torch.from_numpy(ego_traj_fix_time_truncated).float()


        if self.step <= 20:
            steer, throttle, brake = 0.0, 0.0, 1.0
        else:
            # Controller expects numpy array [N, 2]
            steer, throttle, brake, metadata = self.controller.control_pid(ego_traj_fix_time_truncated, tick_data['speed'], ego_traj_fix_time_truncated[-1])

        # breakpoint()
        self.step_time_avg.append(float(time.time()-step_start_time))
        if len(self.step_time_avg)==20:
            self.step_time_avg.pop(0)
        # Control was already computed above before visualization (lines 602-612)
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['speed'] = float(tick_data['speed'])

        self.prev_control = control
        if len(self.prev_control_cache)==2:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)

        if self.step % 20 == 0 and self.should_save_data:
            start_time = time.time()
            self.save(tick_data, diffusion_es_outputs, draw_traj=True)
            end_time = time.time()
            print(f"Visualization saved in {end_time-start_time:.2f} seconds.", flush=True)


        return control
    
    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose
    
    def command2hot(self,command,max_dim=6):
        if command < 0:
            command = 4
        command -= 1
        cmd_one_hot = np.zeros(max_dim)
        cmd_one_hot[command] = 1
        return cmd_one_hot
    
    def pos2posemb(self,pos, num_pos_feats=32, temperature=10000):
        scale = 2 * np.pi
        pos = pos * scale
        dim_t = np.arange(num_pos_feats, dtype=np.float32)
        dim_t = temperature ** (2 * (dim_t//2) / num_pos_feats)
        pos_tmp = pos[..., None] / dim_t
        posemb = np.stack((np.sin(pos_tmp[..., 0::2]), np.cos(pos_tmp[..., 1::2])), axis=-1)
        return posemb.reshape(-1)
    
    def ego_to_bev_pixels(self, ego_coords):
        """
        Convert ego coordinates [left, forward] to BEV pixel coordinates [u, v]

        Args:
            ego_coords: numpy array of shape [N, 2] where columns are [left, forward]

        Returns:
            numpy array of shape [N, 2] with pixel coordinates [u, v]
        """
        N = ego_coords.shape[0]
        # Create homogeneous coordinates: [x, y, z, 1]
        # In ego frame: x=forward, y=left, z=0 (on ground plane)
        # Model outputs [left, forward], so we need to swap
        homogeneous = np.zeros((N, 4))
        homogeneous[:, 0] = ego_coords[:, 1]  # x = forward
        homogeneous[:, 1] = ego_coords[:, 0]  # y = left
        homogeneous[:, 2] = 0.0  # z = 0 (ground plane)
        homogeneous[:, 3] = 1.0

        # Project to BEV pixel coordinates using coor2topdown matrix
        pixel_coords = (self.coor2topdown @ homogeneous.T).T  # [N, 4]
        # Normalize by w coordinate
        pixels = pixel_coords[:, :2] / pixel_coords[:, 2:3]

        return pixels

    def _rect_corners(self, center, length, width, yaw):
        """
        Return 4 corner points of an oriented rectangle centered at `center`.

        Args:
            center: (2,) numpy array - center position in ego frame [left, forward]
            length: float - length of rectangle (along forward direction)
            width: float - width of rectangle (along left direction)
            yaw: float - rotation angle in radians

        Returns:
            (4, 2) numpy array of corner points in ego frame [left, forward]
        """
        hl = length / 2.0
        hw = width / 2.0
        # Local corners: forward-left coordinate system
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        c = math.cos(yaw)
        s = math.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        world_pts = (R @ local.T).T + center.reshape(1, 2)
        return world_pts


    def get_waypoints_in_drivable_area(self, 
                                        ego_trajectories, 
                                        drivable_map,
                                        grid_resolution):

        """
        For each waypoint in ego_trajectories, determine if it lies within drivable area
        Args:
            ego_trajectories: [B*num_particles, num_traj_tokens, 2] in unnormalized absolute coordinates
            drivable_map: 2D numpy array, 1=drivable, 0=non-drivable
            grid_resolution: float, meters per pixel
        Returns:
            drivable_values: [B*num_particles, num_traj_tokens] float values from drivable map at each waypoint with bilinear interpolation
            0 = non-drivable, 1 = drivable
        """
        
        # assume ego_trajectories: [B*num_particles, num_traj_tokens, 2] in unnormalized absolute coordinates
        # assume agent_bbx is a list of length M, each element is (4, 2) numpy array of bbox corners in ego frame
        # assume agent_vel is a list of length M, each element is (2,) numpy array of velocity in ego frame
        # drivable_map: 2D numpy array, 1=drivable, 0=non-drivable
        # grid_resolution: float, meters per pixel

        # for x is right, y is forward in ego frame
        # drivable map (0,0) is behind vehicle, to the left
        # drivable_map[i, j] is col i, row j

        if not isinstance(drivable_map, torch.Tensor):
            drivable_map = torch.from_numpy(drivable_map).float()

        # first need to shift ego trajectories to drivable map pixel coordinates
        map_size = drivable_map.shape[0]
        ego_trajectories_pixels = ego_trajectories / grid_resolution + map_size / 2.0

        # do an interpolation to get drivable values at trajectory points
        if not isinstance(ego_trajectories_pixels, torch.Tensor):
            ego_trajectories_pixels = torch.from_numpy(ego_trajectories_pixels).float().to(drivable_map.device)
        
        x = ego_trajectories_pixels[:, :, 0]
        y = ego_trajectories_pixels[:, :, 1]

        x_0 = torch.floor(x).long()
        x_1 = x_0 + 1
        y_0 = torch.floor(y).long()
        y_1 = y_0 + 1

        x_0 = torch.clamp(x_0, 0, map_size - 1)
        x_1 = torch.clamp(x_1, 0, map_size - 1)
        y_0 = torch.clamp(y_0, 0, map_size - 1)
        y_1 = torch.clamp(y_1, 0, map_size - 1)

        Ia = drivable_map[x_0, y_0]
        Ib = drivable_map[x_0, y_1]
        Ic = drivable_map[x_1, y_0]
        Id = drivable_map[x_1, y_1]

        wa = (x_1.float() - x) * (y_1.float() - y)
        wb = (x_1.float() - x) * (y - y_0.float())
        wc = (x - x_0.float()) * (y_1.float() - y)
        wd = (x - x_0.float()) * (y - y_0.float())

        drivable_values = wa * Ia + wb * Ib + wc * Ic + wd * Id  # [B*num_particles, num_traj_tokens]

        return drivable_values # [B*num_particles, num_traj_tokens]

    def get_collision_points(self,
                             ego_trajectories,
                             agent_bbx,
                             agent_vel,
                             dt=0.2,
                             safety_margin=0.5):
        """
        For each waypoint in ego_trajectories, determine if it collides with any agent bounding box
        Args:
            ego_trajectories: [B*num_particles, num_traj_tokens, 2] in ego frame coordinates [left, forward]
            agent_bbx: list of length M, each element is (4, 2) numpy array of bbox corners in ego frame [left, forward]
            agent_vel: list of length M, each element is (2,) numpy array of velocity in ego frame [left, forward]
            dt: float, time difference between trajectory waypoints
            safety_margin: float, minimum distance tolerance in meters (default: 0.5m)
        Returns:
            collision_values: [B*num_particles, num_traj_tokens] float values, 1.0 if collision, 0.0 if no collision
        """

        if not isinstance(ego_trajectories, torch.Tensor):
            ego_trajectories = torch.from_numpy(ego_trajectories).float()

        batch_size = ego_trajectories.shape[0]
        num_timesteps = ego_trajectories.shape[1]

        # If no agents, return zeros
        if len(agent_bbx) == 0:
            return torch.zeros(batch_size, num_timesteps, device=ego_trajectories.device)

        # Ego vehicle dimensions (length x width in meters)
        # Coordinates are [left, forward]
        # Add safety margin to ego dimensions for conservative collision detection
        ego_half_width = (2.0 + safety_margin) / 2.0   # half-width in left direction
        ego_half_length = (4.5 + safety_margin) / 2.0  # half-length in forward direction

        collision_values = torch.zeros(batch_size, num_timesteps, device=ego_trajectories.device)

        # Convert ego trajectories to numpy for computation
        ego_traj_np = ego_trajectories.cpu().numpy()  # (B, T, 2)

        # For each agent
        for agent_idx in range(len(agent_bbx)):
            bbox_corners = np.array(agent_bbx[agent_idx])  # (4, 2) in [left, forward] format
            agent_velocity = np.array(agent_vel[agent_idx])  # (2,) in [left, forward] format

            # Get agent bbox center and oriented axes
            bbox_center = bbox_corners.mean(axis=0)  # (2,)

            # Compute edges from bbox corners (assuming corners are ordered)
            edge1 = bbox_corners[1] - bbox_corners[0]  # One side
            edge2 = bbox_corners[3] - bbox_corners[0]  # Adjacent side

            # Get edge lengths and normalized directions
            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)

            # Normalized axis directions for the agent's oriented box
            axis1 = edge1 / len1  # Unit vector along edge1
            axis2 = edge2 / len2  # Unit vector along edge2

            # Half-lengths along each edge (add safety margin to agent dimensions)
            half_len1 = len1 / 2.0 + safety_margin / 2.0
            half_len2 = len2 / 2.0 + safety_margin / 2.0

            # For each timestep, compute future agent position
            for t in range(num_timesteps):
                # Future center of agent at timestep t (constant velocity model)
                future_center = bbox_center + agent_velocity * dt * (t + 1)  # (2,)

                # Get all ego positions at this timestep: (B, 2)
                ego_positions = ego_traj_np[:, t, :]  # (B, 2)

                # Check collision for each trajectory using Separating Axis Theorem
                # Test 4 axes: agent's 2 axes + ego's 2 axes (axis-aligned)
                for batch_idx in range(batch_size):
                    ego_pos = ego_positions[batch_idx]  # (2,) in [left, forward]

                    # Vector from agent center to ego center
                    diff = ego_pos - future_center  # (2,)

                    collision = True

                    # Test agent's axis 1
                    proj_diff = np.abs(np.dot(diff, axis1))
                    # Ego box projected onto agent's axis1: sum of absolute projections
                    ego_proj = ego_half_width * np.abs(axis1[0]) + ego_half_length * np.abs(axis1[1])
                    if proj_diff > half_len1 + ego_proj:
                        collision = False
                        continue

                    # Test agent's axis 2
                    proj_diff = np.abs(np.dot(diff, axis2))
                    ego_proj = ego_half_width * np.abs(axis2[0]) + ego_half_length * np.abs(axis2[1])
                    if proj_diff > half_len2 + ego_proj:
                        collision = False
                        continue

                    # Test ego's axis 1 (left direction: [1, 0])
                    proj_diff = np.abs(diff[0])  # Projection onto [1, 0]
                    agent_proj = half_len1 * np.abs(axis1[0]) + half_len2 * np.abs(axis2[0])
                    if proj_diff > ego_half_width + agent_proj:
                        collision = False
                        continue

                    # Test ego's axis 2 (forward direction: [0, 1])
                    proj_diff = np.abs(diff[1])  # Projection onto [0, 1]
                    agent_proj = half_len1 * np.abs(axis1[1]) + half_len2 * np.abs(axis2[1])
                    if proj_diff > ego_half_length + agent_proj:
                        collision = False
                        continue

                    # No separating axis found -> collision!
                    if collision:
                        collision_values[batch_idx, t] = 1.0

        return collision_values
        
            
        





    def save(self, tick_data, diffusion_es_outputs, draw_traj=False):
        frame = self.step // 10

        bev_frame = tick_data['bev'].copy()

        # Get vehicle information for bounding box visualization
        vehicles_info = self.get_vehicles_info()

        # Get drivable area map from CARLA
        start_time = time.time()
        drivable_area_info = self.get_drivable_area_map(grid_size=200, grid_resolution=0.2)
        end_time = time.time()
        print(f"Drivable area map computed in {end_time-start_time:.2f} seconds.", flush=True)  

        if draw_traj and diffusion_es_outputs is not None:
            # Determine number of subplots: Front Image + BEV + Best Trajectory + 1 per iteration
            num_iterations = len(diffusion_es_outputs.get('iterations', []))
            num_subplots = 4 + num_iterations  # Front Image, BEV, Best Trajectory, + iterations + one for drivable area

            # Arrange in grid: calculate rows and columns
            # Aim for roughly square grid, prefer more columns than rows
            num_cols = int(np.ceil(np.sqrt(num_subplots * 1.5)))  # 1.5 aspect ratio preference
            num_rows = int(np.ceil(num_subplots / num_cols))

            # Create figure with subplots in a grid
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
            axes = axes.flatten() if num_subplots > 1 else [axes]

            # Hide unused subplots
            for idx in range(num_subplots, len(axes)):
                axes[idx].axis('off')

            ax_idx = 0

            # Subplot 0: Front camera image
            front_img = tick_data['imgs']['CAM_FRONT']
            axes[ax_idx].imshow(cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB))
            axes[ax_idx].set_title(f'Front Camera - Step {self.step}')
            axes[ax_idx].axis('off')
            ax_idx += 1

            # Subplot 1: BEV image only
            axes[ax_idx].imshow(bev_frame)
            axes[ax_idx].set_title(f'BEV Image - Step {self.step}')
            axes[ax_idx].axis('off')
            ax_idx += 1

            drivable_map = drivable_area_info['drivable_map']
            grid_resolution = drivable_area_info['grid_resolution']
            grid_size = drivable_map.shape[0]
            map_extent = [
                -grid_size * grid_resolution / 2,
                grid_size * grid_resolution / 2,
                -grid_size * grid_resolution / 2,
                grid_size * grid_resolution / 2,
            ]

            # Subplot 2: Best trajectory with drivable area overlay
            best_traj = diffusion_es_outputs['best_trajectory'][0]  # [num_traj_tokens, 2]

            # Convert to numpy if needed
            if isinstance(best_traj, torch.Tensor):
                best_traj_np = best_traj.cpu().numpy()
            else:
                best_traj_np = best_traj

            # Overlay drivable area map: 1=white (drivable), 0=black (non-drivable)
            axes[ax_idx].imshow(
                drivable_map.T,
                origin='lower',
                cmap='gray',
                vmin=0.0,
                vmax=1.0,
                alpha=0.5,
                extent=map_extent,
                zorder=1
            )

            # set background color to black (outside map extent)
            axes[ax_idx].set_facecolor('black')

            # Plot best trajectory in ego coordinates (forward, left)
            axes[ax_idx].plot(best_traj_np[:, 0], best_traj_np[:, 1],
                             'r-', linewidth=3, label='Best Trajectory', alpha=0.8)
            axes[ax_idx].plot(best_traj_np[0, 0], best_traj_np[0, 1],
                             'go', markersize=10, label='Start', zorder=5)
            axes[ax_idx].plot(best_traj_np[-1, 0], best_traj_np[-1, 1],
                             'r*', markersize=15, label='End', zorder=5)
            axes[ax_idx].plot(0, 0, 'yo', markersize=12, label='Ego Vehicle', zorder=10)

            # Draw ego vehicle bounding box
            ego_bbox = self.hero_actor.bounding_box
            ego_extent = ego_bbox.extent
            ego_length = ego_extent.x * 2.0
            ego_width = ego_extent.y * 2.0
            ego_center = np.array([0.0, 0.0])  # Ego vehicle is at origin in ego frame
            ego_corners = self._rect_corners(ego_center, ego_length, ego_width, np.pi/2)  # Add 90 deg rotation
            ego_poly = Polygon(ego_corners, closed=True, facecolor='red', edgecolor='red',
                              linewidth=2.0, alpha=0.6, zorder=15)
            axes[ax_idx].add_patch(ego_poly)

            # Draw vehicle bounding boxes using precomputed corners
            for i in range(len(vehicles_info['vehicle_ids'])):
                corners = vehicles_info['bbox_corners_ego'][i]
                poly = Polygon(corners, closed=True, facecolor='cyan', edgecolor='cyan',
                              linewidth=1.5, alpha=0.5)
                axes[ax_idx].add_patch(poly)

            # Draw velocity vectors for all vehicles
            for i in range(len(vehicles_info['vehicle_ids'])):
                ego_pos = vehicles_info['ego_positions'][i]
                ego_vel = vehicles_info['ego_velocities'][i]
                # Draw arrow from vehicle position in direction of velocity
                axes[ax_idx].arrow(ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1],
                                  head_width=0.5, head_length=0.5, fc='blue', ec='blue',
                                  alpha=0.7, linewidth=2, zorder=20)

            axes[ax_idx].set_xlabel('Left (m)')
            axes[ax_idx].set_ylabel('Forward (m)')
            axes[ax_idx].set_title(f'Best Trajectory - Step {self.step}')
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].set_xlim(-20, 20)
            axes[ax_idx].set_ylim(-20, 20)
            axes[ax_idx].set_aspect('equal', adjustable='box')
            axes[ax_idx].legend(loc='upper right')
            ax_idx += 1

            # One subplot per iteration showing all particles and selected top-k
            if 'iterations' in diffusion_es_outputs:
                for iter_idx, iteration_data in enumerate(diffusion_es_outputs['iterations']):
                    ax = axes[ax_idx]
                    ax.set_facecolor('black')
                    ax.imshow(
                        drivable_map.T,
                        origin='lower',
                        cmap='gray',
                        vmin=0.0,
                        vmax=1.0,
                        alpha=0.5,
                        extent=map_extent,
                        zorder=0
                    )

                    population = iteration_data['population'][0]  # [num_particles, num_traj_tokens, 2]
                    if isinstance(population, torch.Tensor):
                        population = population.cpu().numpy()
                    # Plot all particles from previous iteration (yellow/gold)
                    # For first iteration, use initial population
                    if population is not None:
                        for i in range(population.shape[0]):
                            ax.plot(population[i, :, 0], population[i, :, 1],
                                   'gold', linewidth=0.5, alpha=0.5, zorder=1)

                    # Plot top-k selected trajectories (blue/green)
                    top_k_trajs = iteration_data['top_k_trajectories'][0]  # [top_k, num_traj_tokens, 2]
                    if isinstance(top_k_trajs, torch.Tensor):
                        top_k_trajs_np = top_k_trajs.cpu().numpy()
                    else:
                        top_k_trajs_np = top_k_trajs


                    for k in range(top_k_trajs_np.shape[0]):
                        label = 'Selected Top-K' if k == 0 else None
                        ax.plot(top_k_trajs_np[k, :, 0], top_k_trajs_np[k, :, 1],
                               'b-', linewidth=2, alpha=1, label=label, zorder=10)

                    # Mark ego vehicle
                    ax.plot(0, 0, 'yo', markersize=12, label='Ego Vehicle', zorder=100)

                    # Draw ego vehicle bounding box
                    ego_bbox = self.hero_actor.bounding_box
                    ego_extent = ego_bbox.extent
                    ego_length = ego_extent.x * 2.0
                    ego_width = ego_extent.y * 2.0
                    ego_center = np.array([0.0, 0.0])  # Ego vehicle is at origin in ego frame
                    ego_corners = self._rect_corners(ego_center, ego_length, ego_width, np.pi/2)  # Add 90 deg rotation
                    ego_poly = Polygon(ego_corners, closed=True, facecolor='red', edgecolor='red',
                                      linewidth=2.0, alpha=0.6, zorder=15)
                    ax.add_patch(ego_poly)

                    # Draw vehicle bounding boxes using precomputed corners
                    for i in range(len(vehicles_info['vehicle_ids'])):
                        corners = vehicles_info['bbox_corners_ego'][i]
                        poly = Polygon(corners, closed=True, facecolor='cyan', edgecolor='cyan',
                                      linewidth=1.5, alpha=0.5)
                        ax.add_patch(poly)

                    # Draw velocity vectors for all vehicles
                    for i in range(len(vehicles_info['vehicle_ids'])):
                        ego_pos = vehicles_info['ego_positions'][i]
                        ego_vel = vehicles_info['ego_velocities'][i]
                        # Draw arrow from vehicle position in direction of velocity
                        ax.arrow(ego_pos[0], ego_pos[1], ego_vel[0], ego_vel[1],
                                head_width=0.5, head_length=0.5, fc='blue', ec='blue',
                                alpha=0.7, linewidth=2, zorder=20)

                    ax.set_xlabel('Left (m)')
                    ax.set_ylabel('Forward (m)')
                    ax.set_title(f'Iteration {iter_idx + 1}')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(-20, 20)
                    ax.set_ylim(-20, 20)
                    ax.set_aspect('equal', adjustable='box')
                    ax.legend(loc='upper right', fontsize=8)

                    ax_idx += 1

            # One plot for checking if proposed initial trajectories are in drivable area
            ax = axes[ax_idx]
            ax.set_facecolor('black')
            ax.imshow(
                drivable_map.T,
                origin='lower',
                cmap='gray',
                vmin=0.0,
                vmax=1.0,
                alpha=0.5,
                extent=map_extent,
                zorder=0
            )
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            # Plot initial proposed trajectories
            initial_trajs = diffusion_es_outputs['iterations'][0]['population'][0]  # [num_particles, num_traj_tokens, 2]
            if isinstance(initial_trajs, torch.Tensor):
                initial_trajs_np = initial_trajs.cpu().numpy()
            else:
                initial_trajs_np = initial_trajs
            
            # all funciton to see if each waypoint is in drivable area
            waypoint_evals = self.get_waypoints_in_drivable_area(
                initial_trajs_np,
                drivable_map,
                grid_resolution
            )

            # Compute collision values for all initial trajectories
            if len(vehicles_info['vehicle_ids']) > 0:
                agent_bbx = vehicles_info['bbox_corners_ego']
                agent_vel = vehicles_info['ego_velocities']
                initial_trajs_tensor = torch.from_numpy(initial_trajs_np).float() if not isinstance(initial_trajs, torch.Tensor) else initial_trajs
                collision_evals = self.get_collision_points(initial_trajs_tensor, agent_bbx, agent_vel, dt=0.2)
                collision_evals = collision_evals.cpu().numpy()  # [num_particles, num_traj_tokens]

            for i in range(initial_trajs_np.shape[0]):
                # plot blue for each waypoint that is drivable, red if not
                traj = initial_trajs_np[i]
                colors = []
                for t in range(traj.shape[0]):
                    if waypoint_evals[i, t] > 0.5:
                        colors.append('blue')
                    else:
                        colors.append('red')
                for t in range(traj.shape[0]-1):
                    ax.plot(traj[t:t+2, 0], traj[t:t+2, 1], color=colors[t], linewidth=1.0, alpha=0.7)

                # Mark collision points on this trajectory with X markers
                if len(vehicles_info['vehicle_ids']) > 0:
                    collision_indices = np.where(collision_evals[i] > 0.5)[0]
                    if len(collision_indices) > 0:
                        ax.scatter(traj[collision_indices, 0],
                                 traj[collision_indices, 1],
                                 c='orange', s=50, marker='x', linewidths=2,
                                 alpha=0.9, zorder=10)

            # Draw vehicle bounding boxes
            for i in range(len(vehicles_info['vehicle_ids'])):
                corners = vehicles_info['bbox_corners_ego'][i]
                poly = Polygon(corners, closed=True, facecolor='cyan', edgecolor='cyan',
                              linewidth=1.5, alpha=0.5, zorder=5)
                ax.add_patch(poly)

            # Draw ego vehicle
            ax.plot(0, 0, 'yo', markersize=12, label='Ego Vehicle', zorder=100)
            ax.set_title('Initial Proposals: Blue=Drivable, Red=Non-Drivable, Orange X=Collision')
            ax.set_xlabel('Left (m)')
            ax.set_ylabel('Forward (m)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)

            
            # Save the figure
            if self.save_path is not None:
                bev_viz_dir = self.save_path / 'bev_viz'
                bev_viz_dir.mkdir(parents=True, exist_ok=True)
                save_path = bev_viz_dir / f'bev_traj_{frame:04d}.png'
                plt.savefig(str(save_path), dpi=100)

            plt.close(fig)

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    def gps_to_location(self, gps):
        EARTH_RADIUS_EQUA = 6378137.0
        # gps content: numpy array: [lat, lon, alt]
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    def get_vehicles_info(self):
        """
        Get information about all vehicles in the scene (excluding the ego vehicle).

        Returns:
            dict: {
                'world_positions': numpy array of shape (num_vehicles, 3) - world coordinates (x, y, z)
                'world_velocities': numpy array of shape (num_vehicles, 3) - world frame velocities (x, y, z)
                'speeds': numpy array of shape (num_vehicles,) - speed in m/s
                'yaws': numpy array of shape (num_vehicles,) - yaw angles in degrees
                'pitches': numpy array of shape (num_vehicles,) - pitch angles in degrees
                'rolls': numpy array of shape (num_vehicles,) - roll angles in degrees
                'bbox_extents': numpy array of shape (num_vehicles, 3) - bounding box extents (x, y, z)
                'vehicle_ids': list of carla.Actor IDs
                'vehicle_types': list of vehicle type strings
                'ego_positions': numpy array of shape (num_vehicles, 2) - positions in ego frame (x forward, y left)
                'ego_velocities': numpy array of shape (num_vehicles, 2) - velocities in ego frame (x forward, y left)
                'bbox_corners_ego': list of numpy arrays - each array is (4, 2) containing corner positions in ego frame
            }
        """
        # Get ego vehicle transform to create transformation matrices
        gt_transform = self.hero_actor.get_transform()
        gt_x = gt_transform.location.x
        gt_y = gt_transform.location.y
        gt_theta_deg = gt_transform.rotation.yaw
        gt_theta = -math.radians(gt_theta_deg) + np.pi/2

        # Create ego to world and world to ego transformation matrices
        ego_to_world = np.array([
            [np.cos(-gt_theta), -np.sin(-gt_theta), gt_x],
            [np.sin(-gt_theta), np.cos(-gt_theta), gt_y],
            [0, 0, 1]
        ])
        world_to_ego = np.linalg.inv(ego_to_world)
        rotation_matrix = world_to_ego[:2, :2]

        # Get all vehicles in the world
        world = self.hero_actor.get_world()
        all_vehicles = world.get_actors().filter('vehicle.*')

        # Initialize lists to collect vehicle data
        world_positions = []
        world_velocities = []
        speeds = []
        yaws = []
        pitches = []
        rolls = []
        bbox_extents = []
        vehicle_ids = []
        vehicle_types = []
        ego_positions = []
        ego_velocities = []
        bbox_corners_ego = []

        for vehicle in all_vehicles:
            # Skip the ego vehicle itself
            if vehicle.id == self.hero_actor.id:
                continue

            # Get vehicle transform (position and orientation)
            veh_transform = vehicle.get_transform()
            location = veh_transform.location
            rotation = veh_transform.rotation

            # World position
            world_pos = np.array([location.x, location.y, location.z])
            world_positions.append(world_pos)

            # World velocity
            velocity = vehicle.get_velocity()
            world_vel = np.array([velocity.x, velocity.y, velocity.z])
            world_velocities.append(world_vel)

            # Speed (magnitude of velocity)
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speeds.append(speed)

            # Orientation (yaw, pitch, roll in degrees)
            yaws.append(rotation.yaw)
            pitches.append(rotation.pitch)
            rolls.append(rotation.roll)

            # Bounding box extents
            bbox = vehicle.bounding_box
            extent = bbox.extent
            bbox_extents.append([extent.x, extent.y, extent.z])

            # Vehicle ID and type
            vehicle_ids.append(vehicle.id)
            vehicle_types.append(vehicle.type_id)

            # Transform position to ego frame
            veh_world_pos_homogeneous = np.array([location.x, location.y, 1])
            veh_ego_pos = world_to_ego @ veh_world_pos_homogeneous
            # Negate x axis to match ego frame convention (x forward, y left)
            ego_pos = veh_ego_pos[:2] * np.array([-1, 1])
            ego_positions.append(ego_pos)

            # Transform velocity to ego frame
            veh_vel_world = np.array([velocity.x, velocity.y])
            veh_vel_ego = rotation_matrix @ veh_vel_world
            # Negate x axis to match ego frame convention
            ego_velocities.append(veh_vel_ego * np.array([-1, 1]))

            # Compute bounding box corners in ego frame
            vehicle_yaw_deg = rotation.yaw
            vehicle_yaw_rad = math.radians(vehicle_yaw_deg)
            ego_yaw_rad = math.radians(gt_theta_deg)
            relative_yaw = vehicle_yaw_rad - ego_yaw_rad
            # Adjust for ego frame coordinate flip (x forward = negated) and add 90 degrees
            relative_yaw = -relative_yaw + np.pi/2

            # Compute corners using bounding box dimensions
            vehicle_length = extent.x * 2.0
            vehicle_width = extent.y * 2.0
            corners = self._rect_corners(ego_pos, vehicle_length, vehicle_width, relative_yaw)
            bbox_corners_ego.append(corners)

        return {
            'world_positions': np.array(world_positions) if len(world_positions) > 0 else np.zeros((0, 3)),
            'world_velocities': np.array(world_velocities) if len(world_velocities) > 0 else np.zeros((0, 3)),
            'speeds': np.array(speeds) if len(speeds) > 0 else np.zeros(0),
            'yaws': np.array(yaws) if len(yaws) > 0 else np.zeros(0),
            'pitches': np.array(pitches) if len(pitches) > 0 else np.zeros(0),
            'rolls': np.array(rolls) if len(rolls) > 0 else np.zeros(0),
            'bbox_extents': np.array(bbox_extents) if len(bbox_extents) > 0 else np.zeros((0, 3)),
            'vehicle_ids': vehicle_ids,
            'vehicle_types': vehicle_types,
            'ego_positions': np.array(ego_positions) if len(ego_positions) > 0 else np.zeros((0, 2)),
            'ego_velocities': np.array(ego_velocities) if len(ego_velocities) > 0 else np.zeros((0, 2)),
            'bbox_corners_ego': bbox_corners_ego
        }

    def get_drivable_area_map(self, grid_size=200, grid_resolution=0.2, lookahead_distance=50.0):
        """
        Generate a binary drivable area map from CARLA using waypoint queries.
        Optimized by computing on a coarse 50x50 grid and resizing to requested size.

        Args:
            grid_size: int - size of the output grid (grid_size x grid_size), typically 200
            grid_resolution: float - resolution in meters per pixel for the output grid
            lookahead_distance: float - how far to query waypoints ahead (unused currently)

        Returns:
            dict: {
                'drivable_map': numpy array of shape (grid_size, grid_size) - binary map (1=drivable, 0=not drivable)
                'ego_location_carla': carla.Location - ego vehicle location from CARLA
                'ego_x_carla': float - ego x position in world frame from CARLA
                'ego_y_carla': float - ego y position in world frame from CARLA
                'ego_yaw_carla': float - ego yaw angle in degrees from CARLA
                'grid_center': tuple (x, y) - grid center in world coordinates
                'grid_extent': tuple (min_x, max_x, min_y, max_y) - grid boundaries in world coordinates
            }
        """
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        # Get ego vehicle location and transform from CARLA
        ego_transform = self.hero_actor.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation

        # Extract ego position from CARLA
        ego_x_carla = ego_location.x
        ego_y_carla = ego_location.y
        ego_z_carla = ego_location.z
        ego_yaw_carla = ego_rotation.yaw

        # Get CARLA map
        carla_map = CarlaDataProvider.get_map()
        if carla_map is None:
            carla_map = self.hero_actor.get_world().get_map()

        # Optimization: Compute on a coarse 50x50 grid then resize to requested grid_size
        # This reduces CARLA queries from 40,000 (200x200) to 2,500 (50x50) - 16x speedup!
        coarse_grid_size = 100
        coarse_resolution = 0.2  # 100 * 0.2 = 20m coverage (±10m from center)

        # Define grid boundaries (centered on ego vehicle) using coarse resolution
        grid_width = coarse_grid_size * coarse_resolution
        half_width = grid_width / 2.0

        min_x = ego_x_carla - half_width
        max_x = ego_x_carla + half_width
        min_y = ego_y_carla - half_width
        max_y = ego_y_carla + half_width

        # Initialize coarse binary drivable area map
        coarse_map = np.zeros((coarse_grid_size, coarse_grid_size), dtype=np.uint8)

        # Create transformation from ego frame to world frame
        # Ego frame: x=left, y=forward (as per the trajectory plots)
        # We need to convert ego frame grid to world coordinates accounting for rotation
        ego_yaw_rad = math.radians(ego_yaw_carla)
        cos_yaw = math.cos(ego_yaw_rad)
        sin_yaw = math.sin(ego_yaw_rad)

        # Sample coarse grid points in ego frame and check if they are on drivable lanes
        for i in range(coarse_grid_size):
            for j in range(coarse_grid_size):
                # Grid coordinates in ego frame (centered at 0,0)
                # Rotate 180 deg from counter-clockwise (which was j→x, i→y): negate both
                ego_x = -(j - coarse_grid_size / 2.0 + 0.5) * coarse_resolution  # left (negated from j)
                ego_y = -(i - coarse_grid_size / 2.0 + 0.5) * coarse_resolution  # forward (negated from i)

                # Transform from ego frame to world coordinates
                # Ego frame convention: x=left, y=forward
                # Need to rotate and translate to world frame
                # Account for the coordinate flip: ego_x (left) = -world_left, ego_y (forward) = world_forward
                world_x = ego_x_carla - ego_x * cos_yaw + ego_y * sin_yaw
                world_y = ego_y_carla - ego_x * sin_yaw - ego_y * cos_yaw

                # Create CARLA location
                query_location = carla.Location(x=world_x, y=world_y, z=ego_z_carla)

                # Query waypoint at this location
                waypoint = carla_map.get_waypoint(query_location, project_to_road=True,
                                                  lane_type=carla.LaneType.Driving)

                # Check if waypoint exists and is close to query location
                if waypoint is not None:
                    wp_loc = waypoint.transform.location
                    distance = math.sqrt((wp_loc.x - world_x)**2 + (wp_loc.y - world_y)**2)

                    # If waypoint is close enough, mark as drivable
                    # Use lane width as threshold
                    if distance <= waypoint.lane_width / 2.0:
                        coarse_map[i, j] = 1

        # Resize coarse map to requested grid_size using bilinear interpolation
        # This preserves binary values while providing smoother boundaries
        if grid_size != coarse_grid_size:
            drivable_map = cv2.resize(coarse_map, (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
        else:
            drivable_map = coarse_map

        return {
            'drivable_map': drivable_map,
            'ego_location_carla': ego_location,
            'ego_x_carla': ego_x_carla,
            'ego_y_carla': ego_y_carla,
            'ego_yaw_carla': ego_yaw_carla,
            'grid_center': (ego_x_carla, ego_y_carla),
            'grid_extent': (min_x, max_x, min_y, max_y),
            'grid_resolution': grid_resolution
        }
