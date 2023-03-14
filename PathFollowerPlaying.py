from UnitAlg import Vector3, Quaternion, Transform, Ray, CoordinateFrame, Circle
CoordinateFrame.set(CoordinateFrame.ROS)

from typing import Union, List, Tuple


import math
import matplotlib.pyplot as plt
#import Timer
import time

#inject the clamp and abs functions into the poorly designed math library
def clamp(value:float, min_value:float, max_value:float) -> float:
	return min(max(value, min_value), max_value)
math.clamp = clamp
def abs(value:float) -> float:
	return math.fabs(value)
math.abs = abs

# A class for a 2D robot with differential drive
class Robot():
	def __init__(self, x:float, y:float, heading:float, wheel_separation:float, wheel_radius:float) -> None:
		self.position = Vector3([x,y,0])
		self.heading = heading
		self.wheel_separation = wheel_separation
		self.wheel_radius = wheel_radius
		self.left_wheel_rpm = 0
		self.right_wheel_rpm = 0
		self.velocity = Vector3([0,0,0])
		self.angular_velocity = 0 #radians per second
		self.transform = Transform(self.position, Quaternion.from_euler(0,0,self.heading))
	
	@property
	def pose() -> Ray:
		return Ray(self.position, self.heading)
	
	def update(self, dt:float) -> None:
		#Update the position and heading
		self.position += self.velocity * dt
		self.heading += self.angular_velocity * dt
		#Update the transform
		self.transform = Transform(self.position, Quaternion.from_euler(0,0,self.heading))
	
	def get_wheel_velocities(self) -> (float, float):
		#Calculate the wheel velocities
		left_velocity = self.left_wheel_rpm * 2 * math.pi / 60
		right_velocity = self.right_wheel_rpm * 2 * math.pi / 60
		return (left_velocity, right_velocity)
		
	def set_wheel_rpms(self, left_rpm:float, right_rpm:float) -> None:
		self.left_wheel_rpm = left_rpm
		self.right_wheel_rpm = right_rpm
		#Calculate the velocity and angular velocity
		left_velocity, right_velocity = self.get_wheel_velocities()
		self.velocity = self.transform.forward* (left_velocity + right_velocity) / 2
		self.angular_velocity = (right_velocity - left_velocity) / self.wheel_separation
		
	def get_wheel_rpms(self) -> (float, float):
		return (self.left_wheel_rpm, self.right_wheel_rpm)
	
	def set_desired_speed(self, desired_speed:float, desired_angular_velocity:float) -> None:
		#Calculate the desired wheel rpms
		desired_left_rpm = (desired_speed - desired_angular_velocity * self.wheel_separation / 2) / self.wheel_radius * 60 / (2 * math.pi)
		desired_right_rpm = (desired_speed + desired_angular_velocity * self.wheel_separation / 2) / self.wheel_radius * 60 / (2 * math.pi)
		
		#Set the wheel rpms
		self.set_wheel_rpms(desired_left_rpm, desired_right_rpm)
	
	def get_max_speeds_for_arc(self, radius:float) -> (float, float):
		"""
		Calculates the maximum speed and angular velocity for a given arc radius,
		given the max acceleration and max angular acceleration of the robot.
		
		Args:
			radius: The radius of the arc to follow. Positive is clockwise, negative is counter-clockwise.
		
		Returns:
			A tuple containing the maximum speed and angular velocity for the given arc radius.
		"""
		#if the radius is very close to 0, return max angular velocity and 0 speed:
		if math.abs(radius) < 0.0001:
			return (0, self.max_angular_velocity)
			
		#Else, Assume for a moment that the speed is 1
		speed = 1
		
		#Calculate the angular velocity needed to follow the arc
		angular_velocity = speed / radius
		
		#Clamp both the speed and angular velocity to the max values, while maintaining the ratio:
		ratio = angular_velocity / speed
		if speed > self.max_velocity:
			speed = self.max_velocity
			angular_velocity = speed * ratio
		if math.abs(angular_velocity) > self.max_angular_velocity:
			angular_velocity = self.max_angular_velocity
			speed = angular_velocity / ratio		
		return (speed, angular_velocity)

def calc_2arc_joining_path(robot:Ray, target:Ray) -> Tuple[Vector3, Vector3, float]:
	"""
	Calculates the path for the robot to follow from its current pose to the target pose.
	
	Args:
		robot: The current pose of the robot.
		target: The target pose of the robot.
	Returns:
		A tuple containing the center of the first arc, the center of the second arc, and the radius of the arcs.
	"""
	
	#Get the vector from the robot's position to the target ray's origin
	robot_to_target = target.origin - robot.origin
	
	#Calculate the perpendicular direction to the robot's heading and facing towards the target
	robot_perpendicular = Vector3.cross(robot.direction, Vector3.up)
	if Vector3.dot(robot_perpendicular, robot_to_target) < 0:
		robot_perpendicular = -robot_perpendicular
	
	#Calculate the perpendicular direction to the target ray and facing towards the robot
	target_perpendicular = Vector3.cross(target.direction, Vector3.up)
	if Vector3.dot(target_perpendicular, robot_to_target) > 0:
		target_perpendicular = -target_perpendicular
	
	q, w, _ = target_perpendicular #Tp
	a, s, _ = robot_perpendicular  #Rp
	z, x, _ = target.origin        #T
	d, f, _ = robot.origin         #R

	i = a**2 - 2*a*q + q**2 + s**2 - 2*s*w + w**2-4
	# if i != 0: -- Will never be a problem because a or s will always be non-zero
	j = -2*a*d + 2*d*q + 2*a*z - 2*z*q - 2*s*f + 2*f*w + 2*s*x - 2*x*w
	k = math.sqrt((-j)**2 - 4*i*(d**2 - 2*d*z + z**2 + f**2 - 2*x*f + x**2))

	r = (j - k) / (2*i)
	if r>=0:
		return r
	r = (j + k) / (2*i)
	return r
		
# A path follower for a 2D robot with differential drive
class PathFollower():
	class CurrentPathSegment():
		def __init__(self, start:Vector3, end:Vector3) -> None:
			self.start = start
			self.end = end
			self.direction = (end - start).normalized()
			self.length = Vector3.distance(start, end)
			self.distance = 0
			self.remaining_distance = self.length
			self.done = False
	
	def __init__(self, robot:Robot, path:list, look_ahead_distance:float, max_velocity:float, max_acceleration:float, max_angular_velocity:float, max_angular_acceleration:float) -> None:
		self.robot = robot
		self.path = path
		self.look_ahead_distance = look_ahead_distance
		self.max_velocity = max_velocity
		self.max_acceleration = max_acceleration
		self.max_angular_velocity = max_angular_velocity
		self.max_angular_acceleration = max_angular_acceleration
		
		self.current_path_segment = None
		self.current_path_segment_index = 0
	
	@staticmethod
	def get_local_path_plan(robot:Robot, target_ray:Ray) -> Tuple[Circle, Ray, Circle]:
		"""
		Find 2 arcs who's radii are equal and that are tangent to eachother, where one arc
		is tangent to the robot's heading, and the other is tangent to the target ray. The
		center of the first arc is on the line perpendicular to the robot's heading that
		passes through the robot's position. The center of the second arc is on the line
		perpendicular to the target ray that passes through the target ray's origin.
		"""
		calc_2arc_joining_path(robot.pose, target_ray)
	
	def update(self, dt:float) -> None:
		#Update the current path segment
		if self.current_path_segment == None:
			self.current_path_segment = self.CurrentPathSegment(self.path[self.current_path_segment_index], self.path[self.current_path_segment_index+1])
		while True:
			#Find the closest point on the current path segment
			ray = Ray(self.current_path_segment.start, self.current_path_segment.direction)
			closest_point = ray.closest_point(self.robot.position)
			#Find the distance to the closest point
			distance = Vector3.distance(closest_point, self.current_path_segment.start)
			#Update the current path segment
			self.current_path_segment.distance = distance
			if Vector3.dot(closest_point - self.current_path_segment.start, self.current_path_segment.direction) < 0:
				self.current_path_segment.remaining_distance = self.current_path_segment.length + distance
			else:
				self.current_path_segment.remaining_distance = self.current_path_segment.length - distance
			#Check if the current path segment is done
			if self.current_path_segment.remaining_distance < self.look_ahead_distance:
				self.current_path_segment.done = True
			#Check if the current path segment is done
			if self.current_path_segment.done:
				#Check if the current path segment is the last path segment
				if self.current_path_segment_index == len(self.path) - 2:
					#Stop the robot if it is now past the last path segment's end
					if self.current_path_segment.remaining_distance < 0:
						self.robot.set_desired_speed(0, 0)
						return
					break
				else:
					#Update the current path segment index
					self.current_path_segment_index += 1
					#Update the current path segment
					self.current_path_segment = self.CurrentPathSegment(self.path[self.current_path_segment_index], self.path[self.current_path_segment_index+1])
			else:
				break
		# Create a point at the look ahead distance along the current path segment, handling if the robot is behind the current path segment
		look_ahead_point = self.current_path_segment.start + self.current_path_segment.direction * math.clamp(self.current_path_segment.distance + self.look_ahead_distance, 0, self.current_path_segment.length)
		
		#Using Pure Pursuit:
		#Calculate the desired velocity towards the look ahead point
		desired_velocity = (look_ahead_point - self.robot.position).normalized() * self.max_velocity
		#Calculate the desired angular velocity so that the robot is facing the look ahead point, but taking into account the direction of the current path segment
		
		desired_angular_velocity = Vector3.signed_angle(self.robot.transform.forward, desired_velocity, Vector3.up)
		#Clamp the desired angular velocity to the maximum angular velocity, keeping the unclamped value for calculating the desired linear speed
		unclamped_desired_angular_velocity = desired_angular_velocity
		desired_angular_velocity = math.clamp(desired_angular_velocity, -self.max_angular_velocity, self.max_angular_velocity)
		#Calculate the desired linear speed
		desired_speed = desired_velocity.magnitude
		#Clamp the desired linear speed to the maximum velocity
		desired_speed = math.clamp(desired_speed, 0, self.max_velocity)
		#Calculate the desired linear speed based on how much the angular velocity was clamped
		desired_speed *= math.clamp(1 - math.abs(unclamped_desired_angular_velocity / self.max_angular_velocity), 0, 1)
		#Clamp the desired linear speed, taking into account the max acceleration, based on the distance to the path segment's end point, if we are on the last path segment
		if self.current_path_segment_index == len(self.path) - 2:
			desired_speed = math.clamp(desired_speed, 0, math.sqrt(2 * self.max_acceleration * self.current_path_segment.remaining_distance))
		
		#Calculate the desired acceleration clamped to the maximum acceleration
		desired_acceleration = math.clamp(Vector3.distance(desired_velocity, self.robot.velocity), -self.max_acceleration, self.max_acceleration)
		#Calculate the desired angular acceleration clamped to the maximum angular acceleration
		desired_angular_acceleration = math.clamp(desired_angular_velocity - self.robot.angular_velocity, -self.max_angular_acceleration, self.max_angular_acceleration)
		
		# Set the desired speed and angular velocity
		self.robot.set_desired_speed(desired_speed, desired_angular_velocity)
		

# A plot for 2D data
class Plot():
	def __init__(self) -> None:
		self.lines = []
		self.figure = plt.figure()
		self.axes = self.figure.add_subplot(111)
		self.axes.set_aspect('equal')
		self.axes.set_xlim(-1, 1)
		self.axes.set_ylim(-1, 1)
		self.axes.grid()
	def add_line(self, name:str, x_label:str, y_label:str) -> None:
		self.lines.append({"name":name, "x_label":x_label, "y_label":y_label, "x_data":[], "y_data":[]})
	def add_data(self, name:str, x_data:float, y_data:float) -> None:
		for line in self.lines:
			if line["name"] == name:
				line["x_data"].append(x_data)
				line["y_data"].append(y_data)
	def update(self) -> None:
		self.axes.clear()
		self.axes.set_xlim(-.1, 1.5)
		self.axes.set_ylim(-.1, 1.5)
		self.axes.grid()
		for line in self.lines:
			self.axes.plot(line["x_data"], line["y_data"], label=line["name"])
		self.axes.legend()
		plt.pause(0.001)

if __name__ == '__main__':
	_R = Vector3(10.145619,8.244399,0)
	R = Vector3(13.0006178,7.489313,0)
	
	T = Vector3(19.765736, 3.835348, 0)
	T_= Vector3(29.998669, 8.076674, 0)
	
	calc_2arc_joining_path(
		Ray(R, (R-_R).normalized()),
		Ray(T, (T_-T).normalized())
	) #~3.48
	
	_R = Vector3(10,13,0)
	R = Vector3(16.9578,14,0)
	
	T = Vector3(20, 8, 0)
	T_= Vector3(26, 6, 0)
	
	calc_2arc_joining_path(
		Ray(R, (R-_R).normalized()),
		Ray(T, (T_-T).normalized())
	) #~2.00
	
	calc_2arc_joining_path(
		Ray(T, (_R-R).normalized()),
		Ray(R, (T-T_).normalized())
	) #~2.00
	
	#Create a robot
	robot = Robot(0,0,0,0.5,0.1)
	
	#Create a path
	path = []
	path.append(Vector3(0,0,0))
	path.append(Vector3(1,0,0))
	path.append(Vector3(1,1,0))
	path.append(Vector3(0,1,0))
	path.append(Vector3(0,0,0))
	
	#Create a path follower
	#Setup the path follower with the robot, path, look ahead distance, max velocity, 
	# max acceleration, max angular velocity, max angular acceleration
	path_follower = PathFollower(robot, path, 0.1, 0.5,0.5, 1.5,40)
	
	#path_follower.get_local_path_plan(robot, Ray(Vector3(1,1,0), Vector3(1,.2,0).normalized()))
	
	#Create a plot
	plot = Plot()
	plot.add_line("Robot Position", "x", "y")
	plot.add_line("Path", "x", "y")
	plot.add_line("Look Ahead Point", "x", "y")
	#plot.add_line("Robot Velocity", "x", "y")
	#plot.add_line("Robot Heading", "x", "y")
	#plot.add_line("Robot Angular Velocity", "x", "y")
	
	dt = 0.01
	#Run the simulation
	while True:
		#Update the robot
		robot.update(dt)
		#Update the path follower
		path_follower.update(dt)
		#Update the plot
		plot.add_data("Robot Position", robot.position.x, robot.position.y)
		plot.add_data("Path", path_follower.current_path_segment.start.x, path_follower.current_path_segment.start.y)
		plot.add_data("Look Ahead Point", path_follower.current_path_segment.start.x + path_follower.current_path_segment.direction.x * (path_follower.current_path_segment.distance + path_follower.look_ahead_distance), path_follower.current_path_segment.start.y + path_follower.current_path_segment.direction.y * (path_follower.current_path_segment.distance + path_follower.look_ahead_distance))
		#plot.add_data("Robot Velocity", robot.velocity.x, robot.velocity.y)
		#plot.add_data("Robot Heading", robot.position.x + math.cos(robot.heading), robot.position.y + math.sin(robot.heading))
		#plot.add_data("Robot Angular Velocity", robot.position.x + math.cos(robot.heading + robot.angular_velocity), robot.position.y + math.sin(robot.heading + robot.angular_velocity))
		plot.update()
		
		time.sleep(dt)