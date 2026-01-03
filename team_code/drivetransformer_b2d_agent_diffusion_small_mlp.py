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
from matplotlib.patches import Circle

SAVE_PATH = "Drivetransformer_Logs" #os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)


def get_entry_point():
    return 'DriveTransformerAgentDiffusion_Small_MLP'


class DriveTransformerAgentDiffusion_Small_MLP(autonomous_agent.AutonomousAgent):
    """
    Drive TransformerAgentDiffusion Agent
    """
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        # self.controller = DecouplePIDController(speed_k_p=2.0, speed_k_i=0.8, speed_k_d=1.5, steer_k_p=1.5, steer_k_i=0.2, steer_k_d=0.2)
        self.controller = PurePursuitController(lookahead_distance=4.0, wheelbase=2.89, max_throttle=0.75,
                                                brake_speed=0.4, brake_ratio=1.0, speed_KP=5.0, speed_KI=0.5,
                                                speed_KD=1.0, speed_n=40, clip_delta=0.25)
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
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += self.save_name
            print("SAVE Result to ", string)
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

        # Sample trajectory from pure noise using diffusion
        sampled_traj = self.model.pts_bbox_head.sample_trajectory_from_noise(
            ego_tokens=ego_tokens,
            batch_size=batch_size,
            num_inference_steps=50,  # Adjust for speed/quality tradeoff
            eta=0.0,  # Deterministic sampling
            clip_denoised=False,
            device=intermediate_ego_query.device
        )

        # sampled_traj shape: [B, num_traj_tokens * 2] in normalized differential space
        # Unnormalize to get absolute coordinates
        sampled_traj_absolute = self.model.pts_bbox_head.diffusion_head.unnormalize_trajectory(sampled_traj)
        # sampled_traj_absolute shape: [B, num_traj_tokens, 2]

        # Replace ego_fut_preds_fix_time with diffusion-sampled trajectory
        # Need to match shape: (B, 1, 30, 2) where 1 is the mode dimension
        output_data_batch[0]['ego_fut_preds_fix_time'] = sampled_traj_absolute.unsqueeze(1)  # Add mode dim

        print(f"[DIFFUSION] Sampled trajectory shape: {sampled_traj_absolute.shape}")
        # ========================================================================

        # Ego trajectory mode selection (only 1 mode available for ego)
        selected_mode = 0
        
        # Track agents in ego coordinates across frames
        # Pass current ego pose for storing with detections
        # Extract the actual numpy array from DataContainer if needed
        current_ego_pose = results['ego_pose']
        if hasattr(current_ego_pose, 'data'):
            current_ego_pose = current_ego_pose.data
        # Convert to numpy if it's a tensor
        if torch.is_tensor(current_ego_pose):
            current_ego_pose = current_ego_pose.cpu().numpy()
        
        # Use raw (unsmoothed) ego pose for tracking to avoid lag
        self._update_agent_tracking(output_data_batch[0], current_ego_pose)
        
        # Visualize agent predictions with their best trajectory modes
        if 'agent_traj_cls_scores' in output_data_batch[0] and self.step % 10 == 0:
            # Get agent data
            agent_boxes = output_data_batch[0]['boxes_3d'].tensor.cpu().numpy()  # (N, 9) [x, y, z, w, l, h, yaw, vx, vy]
            agent_scores = output_data_batch[0]['scores_3d'].cpu().numpy()  # (N,)
            agent_labels = output_data_batch[0]['labels_3d'].cpu().numpy()  # (N,)
            agent_trajs = output_data_batch[0]['trajs_3d'].cpu().numpy()  # (N, 6, 12)
            agent_traj_cls_scores = output_data_batch[0]['agent_traj_cls_scores'].cpu().numpy()  # (N, 6) or (1, N, 6)
            
            # Handle both (N, 6) and (1, N, 6) shapes
            if agent_traj_cls_scores.ndim == 3:
                agent_traj_cls_scores = agent_traj_cls_scores[0]  # (N, 6)

            # Get best mode for each agent
            best_modes = np.argmax(agent_traj_cls_scores, axis=1)  # (N,)
            
            # Helper function to convert ego coordinates to BEV pixel coordinates
            def ego_to_pixel(coords_xy):
                """Convert ego frame (x, y) to BEV pixel coordinates using coor2topdown projection.
                coords_xy: (N, 2) array of [x, y] in ego frame (lidar coordinates: x forward, y left)
                Returns: (N, 2) array of [px, py] pixel coordinates
                """
                # Swap x,y to match the projection convention (y, x) as done in save() method
                coords_swapped = coords_xy[:, [1, 0]]  
                # Add z=0 and homogeneous coordinate
                coords_3d = np.concatenate([coords_swapped, np.zeros((len(coords_xy), 1)), np.ones((len(coords_xy), 1))], axis=-1)
                # Project to pixel coordinates
                pixel_coords = np.dot(self.coor2topdown, coords_3d.T).T
                # Normalize by homogeneous coordinate
                pixel_coords[:, :2] /= pixel_coords[:, 2:3]
                # Flip Y coordinate so forward (positive X in ego) points up in the image
                # BEV camera looks down, so we need to invert Y to make forward point up
                pixel_coords[:, 1] = 512 - pixel_coords[:, 1]
                # Swap X and Y for plotting (so X is forward/up, Y is left/right)
                pixel_coords = pixel_coords[:, [1, 0]]
                return pixel_coords[:, :2]
            
            # Create visualization with BEV and front camera side by side
            fig, (ax_bev, ax_front) = plt.subplots(1, 2, figsize=(24, 12))

            # LEFT SUBPLOT: BEV with agent predictions
            # Display BEV image as background (512x512 pixels)
            # Flip image vertically so forward points up
            bev_img = np.flipud(tick_data['bev'])
            ax_bev.imshow(bev_img, extent=[0, 512, 0, 512], origin='lower', alpha=1.0)

            ax_bev.set_xlim(-50, 562)
            ax_bev.set_ylim(-50, 562)
            ax_bev.set_aspect('equal')
            ax_bev.grid(True, alpha=0.3, color='white', linewidth=0.5)
            ax_bev.set_xlabel('X (left, pixels)', fontsize=12, color='white')
            ax_bev.set_ylabel('Y (forward, pixels)', fontsize=12, color='white')
            ax_bev.set_title(f'BEV with Agent Predictions (Step {self.step})', fontsize=14, color='white')
            ax_bev.tick_params(colors='white')

            # RIGHT SUBPLOT: Front camera
            ax_front.imshow(tick_data['imgs']['CAM_FRONT'])
            ax_front.axis('off')
            ax_front.set_title('Front Camera', fontsize=14, color='white')

            # Enable clipping to prevent labels from being cut off
            plt.rcParams['text.usetex'] = False
            fig.tight_layout(pad=2.0)

            # Set black background for entire figure
            fig.patch.set_facecolor('black')
            ax_bev.set_facecolor('black')
            ax_front.set_facecolor('black')
            
            # Plot ego vehicle at origin (convert 0,0 in ego frame to pixels)
            ego_pixel = ego_to_pixel(np.array([[0, 0]]))[0]
            ego_circle = Circle(ego_pixel, 10, color='lime', alpha=0.7, linewidth=2, fill=False, label='Ego Vehicle')
            ax_bev.add_patch(ego_circle)
            ax_bev.plot(ego_pixel[0], ego_pixel[1], 'g*', markersize=20, markeredgecolor='white', markeredgewidth=1)
            
            # Plot lane centerlines if available
            if 'map_pts_3d' in output_data_batch[0]:
                map_pts = output_data_batch[0]['map_pts_3d'].cpu().numpy()  # (M, 20, 2)
                map_scores = output_data_batch[0]['map_scores_3d'].cpu().numpy()  # (M,)
                map_labels = output_data_batch[0]['map_labels_3d'].cpu().numpy()  # (M,)
                
                # Define colors for different map element types
                map_colors = {
                    0: 'yellow',      # divider
                    1: 'white',       # boundary
                    2: 'cyan',        # ped_crossing
                }
                
                # Plot each lane with score > 0.3
                for i in range(len(map_pts)):
                    if map_scores[i] < 0.1:
                        continue
                    
                    lane_pts_ego = map_pts[i]  # (20, 2) in ego frame
                    lane_pixels = ego_to_pixel(lane_pts_ego)
                    
                    label_idx = int(map_labels[i])
                    color = map_colors.get(label_idx, 'gray')

                    # Plot lane as connected line
                    ax_bev.plot(lane_pixels[:, 0], lane_pixels[:, 1], '-', color=color,
                           linewidth=2, alpha=0.7, linestyle='--')
            
            # Color map for different agent classes
            class_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'yellow', 'blue', 'magenta']
            
            # First, plot tracked agent histories and forecasted trajectories
            # Get current ego pose for transformation
            current_ego_pose = results['ego_pose']
            if hasattr(current_ego_pose, 'data'):
                current_ego_pose = current_ego_pose.data
            # Convert to numpy if it's a tensor
            if torch.is_tensor(current_ego_pose):
                current_ego_pose = current_ego_pose.cpu().numpy()
            
            current_ego_pose_inv = self.invert_pose(current_ego_pose)
            
            # Plot each agent with score > 0.3
            for i in range(len(agent_boxes)):
                if agent_scores[i] < 0.3:
                    continue
                    
                x, y, z, w, l, h, yaw = agent_boxes[i, 0], agent_boxes[i, 1], agent_boxes[i, 2], agent_boxes[i, 3], agent_boxes[i, 4], agent_boxes[i, 5], agent_boxes[i, 6]
                vx, vy = agent_boxes[i, 7], agent_boxes[i, 8]  # Velocity in ego frame
                label_idx = int(agent_labels[i])
                color = class_colors[label_idx % len(class_colors)]
                best_mode = best_modes[i]
                
                # Draw bounding box in ego frame
                # Box corners in local frame (width=w, length=l)
                corners_local = np.array([
                    [l/2, w/2],    # front-left
                    [l/2, -w/2],   # front-right
                    [-l/2, -w/2],  # rear-right
                    [-l/2, w/2],   # rear-left
                    [l/2, w/2]     # close the box
                ])
                
                # Rotation matrix for yaw (add 90 degree offset)
                yaw_corrected = yaw + np.pi/2
                cos_yaw = np.cos(yaw_corrected)
                sin_yaw = np.sin(yaw_corrected)
                rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                
                # Rotate and translate corners to ego frame
                corners_ego = (rot_matrix @ corners_local.T).T + np.array([x, y])
                
                # Convert corners to pixels
                corners_pixels = ego_to_pixel(corners_ego)

                # Draw bounding box
                ax_bev.plot(corners_pixels[:, 0], corners_pixels[:, 1], '-', color=color, linewidth=2, alpha=0.8)
                
                # Plot predicted velocity as arrow from agent center
                agent_pixel = ego_to_pixel(np.array([[x, y]]))[0]
                scale = 10.0  # Scale velocity for visualization (1 m/s = 10 pixels)
                vel_end_pos = np.array([[x + vx * scale, y + vy * scale]])
                vel_end_pixel = ego_to_pixel(vel_end_pos)[0]

                # Draw velocity arrow
                ax_bev.arrow(agent_pixel[0], agent_pixel[1],
                        vel_end_pixel[0] - agent_pixel[0],
                        vel_end_pixel[1] - agent_pixel[1],
                        head_width=5, head_length=8, fc='cyan', ec='white',
                        linewidth=1.5, alpha=0.8, length_includes_head=True)
            
            # Plot ego predicted trajectories
            # Use the EXACT same processing as the save() method for BEV visualization
            if 'ego_fut_preds_fix_time' in output_data_batch[0] and 'ego_fut_preds_fix_dist' in output_data_batch[0]:
                # Get raw trajectories from model (same as control code)
                ego_traj_fix_time_raw = output_data_batch[0]['ego_fut_preds_fix_time'][0, selected_mode, :, [1, 0]].float().cpu().numpy()
                angles = output_data_batch[0]['ego_fut_preds_fix_dist'][0, selected_mode, :, 0].float().cpu().numpy()
                ego_traj_fix_dist_raw = np.arange(1, 21, dtype=np.float64).reshape(-1, 1).repeat(2, 1)
                ego_traj_fix_dist_raw[:, 0] *= np.cos(angles)
                ego_traj_fix_dist_raw[:, 1] *= np.sin(angles)
                
                # Process fix_time trajectory (same as save() method)
                ego_fut_preds_fix_time_vis = ego_traj_fix_time_raw[:, [1, 0]]
                ego_fut_preds_fix_time_vis = np.concatenate([ego_fut_preds_fix_time_vis, np.zeros((ego_fut_preds_fix_time_vis.shape[0], 1)), np.ones((ego_fut_preds_fix_time_vis.shape[0], 1))], axis=-1)
                ego_fut_preds_fix_time_vis = np.dot(self.coor2topdown, ego_fut_preds_fix_time_vis.T).T
                ego_fut_preds_fix_time_vis[:, :2] /= ego_fut_preds_fix_time_vis[:, 2:3]
                ego_fut_preds_fix_time_vis = np.nan_to_num(ego_fut_preds_fix_time_vis)
                ax_bev.plot(ego_fut_preds_fix_time_vis[:, 0], 512 - ego_fut_preds_fix_time_vis[:, 1], 'o-', color='red',
                       linewidth=2.5, markersize=4, alpha=0.9, label='Ego Traj (Fixed Time)')

                # Process fix_dist trajectory (same as save() method)
                # ego_fut_preds_fix_dist_vis = ego_traj_fix_dist_raw[:, [1, 0]]
                # ego_fut_preds_fix_dist_vis = np.concatenate([ego_fut_preds_fix_dist_vis, np.zeros((ego_fut_preds_fix_dist_vis.shape[0], 1)), np.ones((ego_fut_preds_fix_dist_vis.shape[0], 1))], axis=-1)
                # ego_fut_preds_fix_dist_vis = np.dot(self.coor2topdown, ego_fut_preds_fix_dist_vis.T).T
                # ego_fut_preds_fix_dist_vis[:, :2] /= ego_fut_preds_fix_dist_vis[:, 2:3]
                # ego_fut_preds_fix_dist_vis = np.nan_to_num(ego_fut_preds_fix_dist_vis)
                # ax_bev.plot(ego_fut_preds_fix_dist_vis[:, 0], 512 - ego_fut_preds_fix_dist_vis[:, 1], 's-', color='blue',
                #        linewidth=2.5, markersize=4, alpha=0.9, label='Ego Traj (Fixed Dist)')

            ax_bev.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')

            # Save combined visualization
            plt.tight_layout()
            plt.savefig(self.save_path / 'combined' / ('%04d.png' % self.step), dpi=100, bbox_inches='tight', facecolor='black')
            plt.close(fig)

        # breakpoint()
        self.step_time_avg.append(float(time.time()-step_start_time))
        if len(self.step_time_avg)==20:
            # print("Model Avg Step Time:", np.mean(self.step_time_avg))
            self.step_time_avg.pop(0)
        all_out_truck = None
        ego_traj_cls_scores = None
        angles = output_data_batch[0]['ego_fut_preds_fix_dist'][0,selected_mode,:,0].float().cpu().numpy()
        # for trajectories with fixed distance, the output is the angle with y-axis in lidar coordinate system. 
        # get the x, y coordinate with the disance and angle.
        ego_traj_fix_dist = np.arange(1,21,dtype=np.float64).reshape(-1,1).repeat(2,1) 
        ego_traj_fix_dist[:,0] *= np.cos(angles)
        ego_traj_fix_dist[:,1] *= np.sin(angles)
        ego_traj_fix_time = output_data_batch[0]['ego_fut_preds_fix_time'][0,selected_mode,:,[1,0]].float().cpu().numpy() 
        if self.step <= 20: # waiting for scenerio initialization (cars are more likely to disappear suddenly in this period)
            steer, throttle, brake = 0.0, 0.0, 1.0
        else:
            # steer, throttle, brake = self.controller.step(ego_traj_fix_time, ego_traj_fix_dist, tick_data['speed']) # PID controller
            # Pure Pursuit controller: truncate trajectory to first 10 waypoints
            truncated_traj = ego_traj_fix_time[:10]
            steer, throttle, brake, metadata = self.controller.control_pid(truncated_traj, tick_data['speed'], truncated_traj[-1])

        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['speed'] = float(tick_data['speed'])
        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data, ego_traj_fix_time.copy(), ego_traj_fix_dist.copy(), output_data_batch[0], draw_traj=True)
        self.prev_control = control
        if len(self.prev_control_cache)==2:
            self.prev_control_cache.pop(0)
        self.prev_control_cache.append(control)
        
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
    
    def save(self, tick_data, ego_fut_preds_fix_time, ego_fut_preds_fix_dist, agent_data, draw_traj=False):
        frame = self.step //10

        # Draw agent bounding boxes and trajectories on BEV
        if 'boxes_3d' in agent_data and 'agent_traj_cls_scores' in agent_data:
            agent_boxes = agent_data['boxes_3d'].tensor.cpu().numpy()
            agent_scores = agent_data['scores_3d'].cpu().numpy()
            agent_labels = agent_data['labels_3d'].cpu().numpy()
            agent_trajs = agent_data['trajs_3d'].cpu().numpy()  # (N, 6, 12)
            agent_traj_cls_scores = agent_data['agent_traj_cls_scores'].cpu().numpy()

            # Handle both (N, 6) and (1, N, 6) shapes
            if agent_traj_cls_scores.ndim == 3:
                agent_traj_cls_scores = agent_traj_cls_scores[0]

            best_modes = np.argmax(agent_traj_cls_scores, axis=1)

            # Color map for different agent classes (BGR format for OpenCV)
            class_colors_bgr = [
                (0, 0, 255),      # red
                (0, 165, 255),    # orange
                (128, 0, 128),    # purple
                (42, 42, 165),    # brown
                (203, 192, 255),  # pink
                (128, 128, 128),  # gray
                (255, 255, 0),    # cyan
                (0, 255, 255),    # yellow
                (255, 0, 0),      # blue
                (255, 0, 255)     # magenta
            ]

            # Draw each agent
            for i in range(len(agent_boxes)):
                if agent_scores[i] < 0.3:
                    continue

                x, y, z, w, l, h, yaw = agent_boxes[i, :7]
                label_idx = int(agent_labels[i])
                color_bgr = class_colors_bgr[label_idx % len(class_colors_bgr)]
                best_mode = best_modes[i]

                # Get agent trajectory for best mode (12, 2) - 12 timesteps, (x,y)
                agent_traj = agent_trajs[i, best_mode, :]  # (12,) - alternating x,y
                agent_traj_xy = agent_traj.reshape(-1, 2)  # (6, 2)

                # Draw bounding box corners (same as matplotlib version)
                corners_local = np.array([
                    [l/2, w/2],    # front-left
                    [l/2, -w/2],   # front-right
                    [-l/2, -w/2],  # rear-right
                    [-l/2, w/2],   # rear-left
                    [l/2, w/2]     # close the box
                ])

                # Rotation matrix for yaw (add 90 degree offset, same as matplotlib)
                yaw_corrected = yaw + np.pi/2
                cos_yaw = np.cos(yaw_corrected)
                sin_yaw = np.sin(yaw_corrected)
                rot_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

                # Rotate and translate corners to ego frame
                corners_ego = (rot_matrix @ corners_local.T).T + np.array([x, y])

                # Convert to BEV pixel coordinates (matching ego_to_pixel function exactly)
                corners_swapped = corners_ego[:, [1, 0]]  # Swap x,y
                corners_3d = np.concatenate([corners_swapped, np.zeros((len(corners_ego), 1)), np.ones((len(corners_ego), 1))], axis=-1)
                corners_pixels = np.dot(self.coor2topdown, corners_3d.T).T
                corners_pixels[:, :2] /= corners_pixels[:, 2:3]
                # Flip Y coordinate so forward (positive X in ego) points up
                corners_pixels[:, 1] = 512 - corners_pixels[:, 1]
                # Swap X and Y for plotting (so X is forward/up, Y is left/right)
                corners_pixels = corners_pixels[:, [1, 0]]
                corners_pixels = np.nan_to_num(corners_pixels[:, :2]).astype(np.int32)

                # Draw box
                cv2.polylines(tick_data['bev'], [corners_pixels], False, color_bgr, 2)

                # Draw velocity arrow (matching matplotlib version with ego_to_pixel)
                vx, vy = agent_boxes[i, 7], agent_boxes[i, 8]  # Velocity in ego frame

                # Convert agent center to pixels (using ego_to_pixel logic)
                agent_center = np.array([[x, y]])
                agent_center_swapped = agent_center[:, [1, 0]]
                agent_center_3d = np.concatenate([agent_center_swapped, np.zeros((1, 1)), np.ones((1, 1))], axis=-1)
                agent_center_pixel = np.dot(self.coor2topdown, agent_center_3d.T).T
                agent_center_pixel[:, :2] /= agent_center_pixel[:, 2:3]
                # Flip Y
                agent_center_pixel[:, 1] = 512 - agent_center_pixel[:, 1]
                # Swap X and Y
                agent_center_pixel = agent_center_pixel[:, [1, 0]]
                agent_center_pixel = np.nan_to_num(agent_center_pixel[:, :2]).astype(np.int32)[0]

                # Scale velocity for visualization (1 m/s = 10 pixels in ego frame)
                scale = 10.0
                vel_end = np.array([[x + vx * scale, y + vy * scale]])
                vel_end_swapped = vel_end[:, [1, 0]]
                vel_end_3d = np.concatenate([vel_end_swapped, np.zeros((1, 1)), np.ones((1, 1))], axis=-1)
                vel_end_pixel = np.dot(self.coor2topdown, vel_end_3d.T).T
                vel_end_pixel[:, :2] /= vel_end_pixel[:, 2:3]
                # Flip Y
                vel_end_pixel[:, 1] = 512 - vel_end_pixel[:, 1]
                # Swap X and Y
                vel_end_pixel = vel_end_pixel[:, [1, 0]]
                vel_end_pixel = np.nan_to_num(vel_end_pixel[:, :2]).astype(np.int32)[0]

                # Draw velocity arrow (cyan fill with white edge, matching matplotlib)
                # Create a copy to draw arrow with transparency effect
                overlay = tick_data['bev'].copy()

                # Draw arrow line in cyan
                cv2.arrowedLine(overlay,
                              tuple(agent_center_pixel),
                              tuple(vel_end_pixel),
                              (255, 255, 0),  # Cyan in BGR
                              2, tipLength=0.25)

                # Draw white outline for better visibility
                cv2.arrowedLine(tick_data['bev'],
                              tuple(agent_center_pixel),
                              tuple(vel_end_pixel),
                              (255, 255, 255),  # White in BGR
                              3, tipLength=0.25)

                # Blend the cyan arrow on top
                cv2.arrowedLine(tick_data['bev'],
                              tuple(agent_center_pixel),
                              tuple(vel_end_pixel),
                              (255, 255, 0),  # Cyan in BGR
                              2, tipLength=0.25)

        # Combined image is now saved via matplotlib in run_step (not here)
        # front_img = tick_data['imgs']['CAM_FRONT']
        # bev_img = tick_data['bev']
        #
        # # Get dimensions
        # front_h, front_w = front_img.shape[:2]
        # bev_h, bev_w = bev_img.shape[:2]
        #
        # # Resize BEV to match front camera height
        # bev_resized = cv2.resize(bev_img, (int(bev_w * front_h / bev_h), front_h))
        #
        # # Concatenate horizontally (BEV on left, front on right)
        # combined = np.concatenate([bev_resized, front_img], axis=1)
        #
        # # Save combined image only
        # Image.fromarray(combined).save(self.save_path / 'combined' / ('%04d.png' % frame))
        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

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
    
    def _update_agent_tracking(self, detection_data, current_ego_pose):
        """
        Track agents across frames in ego coordinates using Hungarian matching.
        
        Args:
            detection_data: dict with keys 'boxes_3d', 'scores_3d', 'labels_3d'
            current_ego_pose: 4x4 transformation matrix from ego to world at current timestep
        """
        if 'boxes_3d' not in detection_data or 'scores_3d' not in detection_data:
            return
        
        agent_boxes = detection_data['boxes_3d'].tensor.cpu().numpy()  # (N, 9)
        agent_scores = detection_data['scores_3d'].cpu().numpy()  # (N,)
        
        # Filter agents by score threshold
        valid_mask = agent_scores > 0.3
        valid_boxes = agent_boxes[valid_mask]
        valid_scores = agent_scores[valid_mask]
        
        if len(valid_boxes) == 0:
            # No detections, age out old tracks
            self._age_out_tracks()
            return
        
        # Extract positions (x, y in ego frame)
        current_positions = valid_boxes[:, :2]  # (N, 2)
        
        # Get active tracks (not too old)
        active_track_ids = [tid for tid, track in self.tracked_agents.items() 
                           if self.step - track['last_seen'] < self.tracking_max_age]
        
        if len(active_track_ids) == 0:
            # No existing tracks, create new ones for all detections
            for i, pos in enumerate(current_positions):
                track_id = self.next_agent_id
                self.next_agent_id += 1
                # Convert ego position to world position
                pos_ego_homogeneous = np.array([pos[0], pos[1], 0, 1])
                pos_world = current_ego_pose @ pos_ego_homogeneous
                self.tracked_agents[track_id] = {
                    'positions': [(self.step, pos[0], pos[1], current_ego_pose.copy())],
                    'world_positions': [(self.step, pos_world[0], pos_world[1])],
                    'last_seen': self.step,
                    'score': valid_scores[i]
                }
        else:
            # Match current detections to existing tracks using distance
            # Build cost matrix (distance between each detection and each track's last position)
            track_last_positions = []
            for tid in active_track_ids:
                last_pos = self.tracked_agents[tid]['positions'][-1]
                track_last_positions.append([last_pos[1], last_pos[2]])  # [x, y]
            
            track_last_positions = np.array(track_last_positions)  # (M, 2)
            
            # Compute pairwise distances (N detections x M tracks)
            from scipy.spatial.distance import cdist
            cost_matrix = cdist(current_positions, track_last_positions)  # (N, M)
            
            # Simple greedy matching (could use Hungarian algorithm for better results)
            matched_detections = set()
            matched_tracks = set()
            matches = []  # [(detection_idx, track_id)]
            
            # Sort by distance and greedily match
            flat_indices = np.argsort(cost_matrix.ravel())
            for flat_idx in flat_indices:
                det_idx = flat_idx // len(active_track_ids)
                track_idx = flat_idx % len(active_track_ids)
                
                if det_idx in matched_detections or track_idx in matched_tracks:
                    continue
                
                distance = cost_matrix[det_idx, track_idx]
                if distance < self.tracking_max_distance:
                    track_id = active_track_ids[track_idx]
                    matches.append((det_idx, track_id))
                    matched_detections.add(det_idx)
                    matched_tracks.add(track_idx)
            
            # Update matched tracks
            for det_idx, track_id in matches:
                pos = current_positions[det_idx]
                # Convert ego position to world position
                pos_ego_homogeneous = np.array([pos[0], pos[1], 0, 1])
                pos_world = current_ego_pose @ pos_ego_homogeneous
                self.tracked_agents[track_id]['positions'].append((self.step, pos[0], pos[1], current_ego_pose.copy()))
                self.tracked_agents[track_id]['world_positions'].append((self.step, pos_world[0], pos_world[1]))
                self.tracked_agents[track_id]['last_seen'] = self.step
                self.tracked_agents[track_id]['score'] = valid_scores[det_idx]
                
                # Limit history length to prevent memory growth
                if len(self.tracked_agents[track_id]['positions']) > 100:
                    self.tracked_agents[track_id]['positions'] = self.tracked_agents[track_id]['positions'][-100:]
                    self.tracked_agents[track_id]['world_positions'] = self.tracked_agents[track_id]['world_positions'][-100:]
            
            # Create new tracks for unmatched detections
            for det_idx in range(len(current_positions)):
                if det_idx not in matched_detections:
                    pos = current_positions[det_idx]
                    track_id = self.next_agent_id
                    self.next_agent_id += 1
                    # Convert ego position to world position
                    pos_ego_homogeneous = np.array([pos[0], pos[1], 0, 1])
                    pos_world = current_ego_pose @ pos_ego_homogeneous
                    self.tracked_agents[track_id] = {
                        'positions': [(self.step, pos[0], pos[1], current_ego_pose.copy())],
                        'world_positions': [(self.step, pos_world[0], pos_world[1])],
                        'last_seen': self.step,
                        'score': valid_scores[det_idx]
                    }
        
        # Age out old tracks
        self._age_out_tracks()
    
    def _age_out_tracks(self):
        """Remove tracks that haven't been seen recently."""
        tracks_to_remove = []
        for track_id, track in self.tracked_agents.items():
            if self.step - track['last_seen'] >= self.tracking_max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_agents[track_id]
    
    def get_agent_trajectories_ego_frame(self, min_length=5):
        """
        Get all tracked agent trajectories in ego coordinates.
        
        Args:
            min_length: minimum number of positions required for a trajectory
            
        Returns:
            dict: track_id -> {'positions': [(step, x, y), ...], 'last_seen': step}
        """
        return {tid: track for tid, track in self.tracked_agents.items() 
                if len(track['positions']) >= min_length}
    
    
    
    
    
