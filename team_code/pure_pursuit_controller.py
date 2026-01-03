from collections import deque
import numpy as np
import math

class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative

class PurePursuitController(object):
    
    def __init__(self, lookahead_distance=4.0, wheelbase=2.89, max_throttle=0.75, 
                 brake_speed=0.4, brake_ratio=1.0, speed_KP=5.0, speed_KI=0.5, 
                 speed_KD=1.0, speed_n=40, clip_delta=0.25):
        """
        Pure Pursuit controller for path following.
        
        Args:
            lookahead_distance: Distance ahead to look for target point
            wheelbase: Vehicle wheelbase length in meters (for steering calculation)
            max_throttle: Maximum throttle value
            brake_speed: Speed threshold for braking
            brake_ratio: Ratio for braking condition
            speed_KP, speed_KI, speed_KD, speed_n: PID parameters for speed control
            clip_delta: Maximum delta for speed control
        """
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        
        # Speed controller uses PID
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
    
    def control_pid(self, waypoints, speed, target, use_last_to_aim=False):
        """
        Predicts vehicle control with Pure Pursuit controller.
        
        Args:
            waypoints: Array of waypoints in ego frame (x forward, y left)
            speed: Current vehicle speed
            target: Target waypoint (not used in pure pursuit but kept for signature compatibility)
            use_last_to_aim: If True, use last waypoint direction (kept for signature compatibility)
        
        Returns:
            steer, throttle, brake, metadata
        """
        # Calculate desired speed from waypoints
        num_pairs = len(waypoints) - 1
        desired_speed = 0
        for i in range(num_pairs):
            desired_speed += np.linalg.norm(
                waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs
        
        # Find lookahead point using Pure Pursuit algorithm
        lookahead_point = None
        best_dist_diff = float('inf')
        
        for i in range(len(waypoints)):
            wp = waypoints[i]
            distance = np.linalg.norm(wp)
            
            # Find waypoint closest to lookahead distance
            dist_diff = abs(distance - self.lookahead_distance)
            if dist_diff < best_dist_diff:
                best_dist_diff = dist_diff
                lookahead_point = wp
        
        # If no good lookahead point found, use the farthest waypoint
        if lookahead_point is None:
            lookahead_point = waypoints[-1]
        
        # Pure Pursuit steering calculation
        # Convert lookahead point to vehicle frame (y forward, x right)
        x = lookahead_point[0]  # right
        y = lookahead_point[1]  # forward
        
        # Calculate lookahead distance (L)
        L = np.linalg.norm(lookahead_point)
        
        # Pure pursuit steering angle: delta = atan(2 * wheelbase * x / L^2)
        # Using x (lateral offset) instead of y since x is the lateral direction
        if L > 0.01:  # Avoid division by zero
            steering_angle = math.atan2(2.0 * self.wheelbase * x, L * L)
        else:
            steering_angle = 0.0
        
        # Normalize steering to [-1, 1] range
        # Assume max steering angle is ~70 degrees (1.22 radians)
        max_steering_angle = 1.22
        steer = np.clip(steering_angle / max_steering_angle, -1.0, 1.0)
        
        # Speed control (same as PID controller)
        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio
        
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0
        
        # Calculate angles for metadata (for compatibility)
        aim = lookahead_point
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        
        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[min(3, len(waypoints)-1)].astype(np.float64)),
            'wp_3': tuple(waypoints[min(2, len(waypoints)-1)].astype(np.float64)),
            'wp_2': tuple(waypoints[min(1, len(waypoints)-1)].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'lookahead_distance': float(L),
            'steering_angle_rad': float(steering_angle),
            'delta': float(delta.astype(np.float64)),
        }
        
        return steer, throttle, brake, metadata