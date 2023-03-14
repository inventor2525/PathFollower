from UnitAlg import Vector3, Quaternion, Transform, Ray, CoordinateFrame
CoordinateFrame.set(CoordinateFrame.ROS)

import math
import matplotlib.pyplot as plt
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