from UnitAlg import Vector3, Quaternion, Transform, Ray, CoordinateFrame, Circle
from UnitAlg.Range import Range
CoordinateFrame.set(CoordinateFrame.ROS)

from typing import Union, List, Tuple

import math
import random
import matplotlib.pyplot as plt
import numpy as np
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
	def __init__(
			self, x:float, y:float, heading:float,
			wheel_separation:float, wheel_radius:float,
			max_speed:float, max_acceleration:float,
			max_angular_velocity:float, max_angular_acceleration:float
		) -> None:
		self.position = Vector3([x,y,0])
		self.heading = heading
		self.wheel_separation = wheel_separation
		self.wheel_radius = wheel_radius
		
		self.max_speed = max_speed
		self.max_acceleration = max_acceleration
		self.max_angular_velocity = max_angular_velocity
		self.max_angular_acceleration = max_angular_acceleration
		
		self.left_wheel_rpm = 0
		self.right_wheel_rpm = 0
		self.speed = 0 #meters per second
		self.angular_velocity = 0 #radians per second
		self.transform = Transform(self.position, Quaternion.from_euler(0,0,self.heading))
		self.set_speed = 0
		self.set_angular_velocity = 0
		
	@property
	def pose(self) -> Ray:
		return Ray(self.position, self.transform.forward)
	
	def update(self, dt:float) -> None:
		#Update the position and heading
		#speed/radius = angular velocity
		if abs(self.angular_velocity)<.0001:
			self.position += self.transform.forward * (self.speed * dt)
		else:
			#Calculate the radius of the circle
			radius = self.speed / self.angular_velocity
			#Calculate the center of the circle
			center = self.position - self.transform.right * radius
			#Calculate the new position
			self.position = center + Quaternion.from_angle_axis(self.angular_velocity * dt, Vector3.up)*(self.position - center)
		self.heading += self.angular_velocity * dt
		#Update the transform
		self.transform = Transform(self.position, Quaternion.from_euler(0,0,self.heading))
	
	def get_wheel_velocities(self) -> (float, float):
		#Calculate the wheel velocities
		c = 2*self.wheel_radius * math.pi
		left_velocity = self.left_wheel_rpm/60 * c
		right_velocity = self.right_wheel_rpm/60 * c
		return (left_velocity, right_velocity)
		
	def set_wheel_rpms(self, left_rpm:float, right_rpm:float) -> None:
		self.left_wheel_rpm = left_rpm
		self.right_wheel_rpm = right_rpm
		#Calculate the velocity and angular velocity
		left_velocity, right_velocity = self.get_wheel_velocities()
		self.speed = (left_velocity + right_velocity) / 2
		self.angular_velocity = (right_velocity - left_velocity) / self.wheel_separation
		
	def get_wheel_rpms(self) -> (float, float):
		return (self.left_wheel_rpm, self.right_wheel_rpm)
	
	def set_desired_speed(self, speed:float, angular_velocity:float) -> None:
		self.set_speed = speed
		self.set_angular_velocity = angular_velocity
		
		#Calculate the desired wheel rpms
		c = 2*self.wheel_radius * math.pi
		desired_left_rpm = (speed + angular_velocity * self.wheel_separation / 2) / c * 60
		desired_right_rpm = (speed - angular_velocity * self.wheel_separation / 2) / c * 60
		
		#Set the wheel rpms
		self.set_wheel_rpms(desired_left_rpm, desired_right_rpm)
	
	def get_max_speeds_for_arc(self, radius:float, dt:float) -> (float, float):
		"""
		Calculates the maximum speed and angular velocity for a given arc radius,
		given the max acceleration and max angular acceleration of the robot.
		
		Args:
			radius: The radius of the arc to follow. Positive is clockwise, negative is counter-clockwise.
			dt: The time step to use for calculating the maximum speed and angular velocity.
			
		Returns:
			A tuple containing the maximum speed and angular velocity for the given arc radius.
		"""
		max_delta_speed = self.max_acceleration * dt
		max_delta_ang_vel = self.max_angular_acceleration * dt
		
		#if the radius is very close to 0, return max angular velocity and 0 speed where possible:
		if math.abs(radius) < 0.0001:
			return (
				max(0, self.set_speed-max_delta_speed),
				min(self.max_angular_velocity + max_delta_ang_vel, self.max_angular_velocity)
			)
		
		#Get the needed ratio of angular velocity to speed
		ratio = (1 / radius) #angular_velocity = speed * ratio
		
		#Get the ranges of speeds and angular velocities that can
		#be achieved while maintaining the constraints of the robot:
		speed_range = Range(0, self.max_speed).overlapping(Range.from_center_delta(self.set_speed, max_delta_speed))
		av_range = Range(-self.max_angular_velocity, self.max_angular_velocity).overlapping(Range.from_center_delta(self.set_angular_velocity, max_delta_ang_vel))
		
		#Convert the ranges to the other variable's units:
		speed_range_in_av = speed_range * ratio
		av_range_in_speed = av_range / abs(ratio)
		
		#Take advantage of the overlapping functions clamping to the first range regardless
		#of if the second range overlaps it or not to find speeds that maintain the constraints
		#as much as possible while still getting close to the desired turn radius:
		speed = speed_range.overlapping(av_range_in_speed).clamp(self.set_speed+max_delta_speed)
		a_vel = av_range.overlapping(speed_range_in_av).clamp( speed*ratio )
		return (speed, a_vel)
		
def calc_2arc_joining_path(robot:Ray, target:Ray, left_first=True) -> Tuple[Vector3, Vector3, float]:
	"""
	Finds 2 arcs who's radii are equal and that are tangent to each other, where one arc
	is tangent to the robot's heading, and the other is tangent to the target ray. The
	center of the first arc is on the line perpendicular to the robot's heading that
	passes through the robot's position. The center of the second arc is on the line
	perpendicular to the target ray that passes through the target ray's origin.
	
	Args:
		robot: The current pose of the robot.
		target: The target pose of the robot.
	Returns:
		A tuple containing the directions to the centers of the arcs. First from the robot
		pose, second from the target pose, and the radius of the arcs.
	"""
	
	#Get the vector from the robot's position to the target ray's origin
	robot_to_target = target.origin - robot.origin
	
	#Calculate a perpendicular direction to the robot's heading
	robot_perpendicular = Vector3.cross(robot.direction, Vector3.up)
	
	#Calculate a perpendicular direction to the target ray
	target_perpendicular = Vector3.cross(target.direction, Vector3.up)
	
	if left_first:
		robot_perpendicular = -robot_perpendicular
	else:
		target_perpendicular = -target_perpendicular
		
	q, w, _ = target_perpendicular #Tp
	a, s, _ = robot_perpendicular  #Rp
	z, x, _ = target.origin        #T
	d, f, _ = robot.origin         #R
	#law of sines is:
	#A=B, a=b, so: (a/sin(A)) = (c/sin(C))
	i = a**2 - 2*a*q + q**2 + s**2 - 2*s*w + w**2-4
	if abs(i) < 0.001:
		#The arcs are parallel, so use a special case calculation using law 
		#of sines:  (a/sin(A)) = (b/sin(B)) = (c/sin(C)) where A=B, a=b, 
		#C is the sweep angle of the arc, a is the radius of the arc, and 
		#c is the distance between robot and target / 2 (otherwise known as
		#the distance from the robot to the point 'M'):
		c = robot_to_target.magnitude/2
		pi_2_A = Vector3.angle(robot.direction, robot_to_target.normalized)
		if pi_2_A < 0.001:
			#The robot is already pointing at the target, so just go straight
			#to the target
			return robot_perpendicular, target_perpendicular, 1000 #1km radius
		A = math.pi/2 - pi_2_A
		C = math.pi - 2*A
		radius = c*math.sin(A) / math.sin(C)
		return robot_perpendicular, target_perpendicular, radius
		
	j = -2*a*d + 2*d*q + 2*a*z - 2*z*q - 2*s*f + 2*f*w + 2*s*x - 2*x*w
	k = math.sqrt((-j)**2 - 4*i*(d**2 - 2*d*z + z**2 + f**2 - 2*x*f + x**2))

	r = (j - k) / (2*i)
	if r<0:
		r = (j + k) / (2*i)
	return robot_perpendicular, target_perpendicular, r

class LocalPlan():
	def __init__(self) -> None:
		self.current_distance = 0
		
	def add_distance(self, distance:float) -> None:
		"""
		Accumulates the distance traveled along the plan.
		"""
		self.current_distance += distance
	
	def length(self) -> float:
		"""
		Returns the length of the plan.
		"""
		return 0 #override this method in subclasses
	
	def is_done(self) -> bool:
		"""
		Returns true if the plan is complete.
		"""
		return self.current_distance >= self.length()/4*3
		
	def get_point_t(self, t:float) -> Vector3:
		"""
		Returns the point on the plan at the given [0,1] time.
		"""
		return Vector3.zero #override this method in subclasses
		
	def get_average_turning_radius(self, speed:float, dt:float) -> float:
		"""
		Returns the average turning radius of the plan for the given time period.
		"""
		return 0 #override this method in subclasses
		
class TwoArcLocalPlan(LocalPlan):
	"""
	A local plan that uses 2 arcs to join the robot's current pose to the target pose.
	
	Note: This plan does not work for turning around. Any plan that involves turning
	more than 90 degrees will not work.
	"""
	def __init__(self, robot:Ray, target:Ray, left_first=True):
		super().__init__()
		print(f"Robot: {robot}")
		print(f"Target: {target}")
		
		self.R = robot.origin
		self.T = target.origin
		
		print(f"""lp = TwoArcLocalPlan(
		Ray(
			Vector3{robot.origin},
			Vector3{robot.direction}
		), 
		Ray(
			Vector3{target.origin},
			Vector3{target.direction}
		)
	)""")
		#Calculate the radius of the arcs and the directions to the centers of the arcs
		self.Rp, self.Tp, r = calc_2arc_joining_path(robot, target, left_first=left_first)
		
		#Calculate the center of the first arc
		self.Rc = robot.origin + self.Rp*r
		
		#Calculate the center of the second arc
		self.Tc = target.origin + self.Tp*r
		
		#Calculate the point the robot should move to the second arc
		self.M = (self.Rc + self.Tc)/2
		
		#Calculate how far the robot should move along the first arc
		circumference = 2*math.pi*r
		self.sweep1 = Vector3.signed_angle2(-self.Rp, (self.M-self.Rc).normalized, Vector3.up if left_first else Vector3.down)
		self.distance1 = self.sweep1 * circumference / (2*math.pi)
		
		#Calculate how far the robot should move along the second arc
		self.sweep2 = Vector3.signed_angle2(-self.Tp, (self.M-self.Tc).normalized, Vector3.up if left_first else Vector3.down)
		self.distance2 = self.sweep2 * circumference / (2*math.pi)
		
		#Find the signed radii of the arc radii (positive is clockwise, negative is counter-clockwise)
		self.radius1 = r if Vector3.dot(self.Rp, Vector3.cross(robot.direction, Vector3.up)) > 0 else -r
		self.radius2 = -self.radius1
		
		#The total distance the robot will have to travel to follow the plan
		self.total_distance = self.distance1 + self.distance2
	
	@classmethod
	def shortest_path(cls, robot:Ray, target:Ray) -> 'TwoArcLocalPlan':
		"""
		Returns the shortest path between the robot and the target.
		"""
		plan1 = cls(robot, target)
		plan2 = cls(robot, target, left_first=False)
		return plan1 if plan1.length() < plan2.length() else plan2
	
	def length(self) -> float:
		return self.total_distance
		
	#check if on arc 2:
	def is_on_arc2(self, distance:float) -> bool:
		return distance > self.distance1
	
	def get_point_t(self, t:float) -> Vector3:
		"""
		Returns the point on the plan at the given [0,1] time.
		"""
		dist = t*self.total_distance
		if dist <= self.distance1:
			return Quaternion.from_angle_axis(
				dist/self.distance1*self.sweep1,
				Vector3.up if self.radius1<=0 else Vector3.down
			) * (self.R-self.Rc) + self.Rc
		else:
			return Quaternion.from_angle_axis(
				self.sweep2-(dist-self.distance1)/self.distance2*self.sweep2,
				Vector3.up if self.radius2>=0 else Vector3.down
			) * (self.T-self.Tc) + self.Tc
	
	#calculate the average turning radius the robot will have to follow the plan for dt more seconds
	def get_average_turning_radius(self, speed:float, dt:float) -> float:
		travel_distance = speed*dt
		final_distance = self.current_distance + travel_distance
		if final_distance > self.distance1:
			if self.current_distance > self.distance1:
				return self.radius2
			else:
				#find the weighted average of the two radii based on how far the robot has traveled along the path this frame
				
				#tan(pi*(t+0.5)/2) is used because on the range [0,1] it goes from 1 to -1 through an asymptote at 0.5
				#which is the same as for the radius which flips from positive to negative
				d1 = self.distance1 - self.current_distance
				t = d1 / travel_distance #percentage of the distance traveled this frame that was on the first arc
				return self.radius1 * math.tan(math.pi*(t+0.5)/2)
		else:
			return self.radius1
		
# A path follower for a 2D robot with differential drive
class PathFollower():
	class CurrentPathSegment():
		def __init__(self, start:Vector3, end:Vector3) -> None:
			self.start = start
			self.end = end
			self.direction = (end - start).normalized
			self.length = Vector3.distance(start, end)
			self.distance = 0
			self.remaining_distance = self.length
			self.done = False
	
	def __init__(self, robot:Robot, path:list, look_ahead_distance:float) -> None:
		self.robot = robot
		self.path = path
		self.look_ahead_distance = look_ahead_distance
				
		self.current_path_segment = None
		self.current_path_segment_index = 0
		
		self.local_plan : LocalPlan = None
		
		self.steps_since_last_local_plan = 0
		
	def get_new_local_plan(self) -> LocalPlan:
		self.steps_since_last_local_plan = 0
		
		if self.current_path_segment_index >= len(self.path)-1:	#If there are no more path segments
			return None
		#Update the current path segment
		if self.current_path_segment == None:
			self.current_path_segment = self.CurrentPathSegment(self.path[self.current_path_segment_index], self.path[self.current_path_segment_index+1])
		while True:
			if self.current_path_segment.length > 0:
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
			else:
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
			
		if self.current_path_segment.length == 0:
			return None
		
		# Create a point at the look ahead distance along the current path segment, handling if the robot is behind the current path segment
		look_ahead_point = self.current_path_segment.start + self.current_path_segment.direction * math.clamp(self.current_path_segment.distance + self.look_ahead_distance, 0, self.current_path_segment.length)
		
		return TwoArcLocalPlan.shortest_path(self.robot.pose, Ray(look_ahead_point, self.current_path_segment.direction))
				
	def update(self, dt:float) -> None:
		if self.local_plan == None:
			self.local_plan = self.get_new_local_plan()
			if self.local_plan == None:
				self.robot.set_desired_speed(0, 0)
				return
				
		#Check if we need a new local plan:
		self.local_plan.add_distance(self.robot.set_speed * dt)
		if self.local_plan.is_done() or self.steps_since_last_local_plan > 2:
			self.local_plan = self.get_new_local_plan()
		
		if self.local_plan == None:
			self.robot.set_desired_speed(0, 0)
			return
			
		# Get the max speeds for the current local plan
		radius = self.local_plan.get_average_turning_radius(self.robot.set_speed, dt)
		speed, angular_velocity = robot.get_max_speeds_for_arc(radius, dt)
		
		# Set the desired speed and angular velocity
		self.robot.set_desired_speed(speed, angular_velocity)
		
		self.steps_since_last_local_plan += 1
		

# A plot for 2D data
class Plot():
	def __init__(self) -> None:
		self.lines = []
		self.figure = plt.figure()
		self.axes = self.figure.add_subplot(111)
		self.axes.set_aspect('equal')
		self.axes.set_xlim(-.1, 1.5)
		self.axes.set_ylim(-.1, 1.5)
		self.axes.grid()
	def add_line(self, name:str, x_label:str, y_label:str) -> None:
		self.lines.append({"name":name, "x_label":x_label, "y_label":y_label, "x_data":[], "y_data":[]})
	def add_data(self, name:str, x_data:float, y_data:float) -> None:
		for line in self.lines:
			if line["name"] == name:
				line["x_data"].append(x_data)
				line["y_data"].append(y_data)
	def set_data(self, name:str, x_data:List[float], y_data:List[float]) -> None:
		for line in self.lines:
			if line["name"] == name:
				line["x_data"] = x_data
				line["y_data"] = y_data
	def update(self) -> None:
		x_limits = self.axes.get_xlim()
		y_limits = self.axes.get_ylim()
		
		self.axes.clear()
		self.axes.set_xlim(*x_limits)
		self.axes.set_ylim(*y_limits)
		
		self.axes.grid()
		for line in self.lines:
			self.axes.plot(line["x_data"], line["y_data"], label=line["name"])
		self.axes.legend()
		plt.pause(0.001)

if __name__ == '__main__':
	# lp = TwoArcLocalPlan(Ray(Vector3(0,0,0), Vector3(1,0,0)), Ray(Vector3(1,1,0), Vector3(1,0,0)))
	#These following cases have drawings in the cad file to check the math of, but they were written before :
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.9983534364339008, 0.14833575666316184, 0.0),
			Vector3(-0.37017975759868804, 0.9289601428824472, 0.0)
		), 
		Ray(
			Vector3(1.0, 0.24833575666316185, 0.0),
			Vector3(0.0, 1.0, 0.0)
		)
	)
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(1.023213764055606, 0.023529924652657047, 0.0),
			Vector3(0.7316892390990255, 0.6816383626137024, 0.0)
		), 
		Ray(
			Vector3(1.0, 0.12352992465265705, 0.0),
			Vector3(0.0, 1.0, 0.0)
		)
	)
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.9099999999996267, -5.867989354413456e-07, 0.0),
			Vector3(0.999999999999332, -1.1558653557561457e-06, 0.0)
		), 
		Ray(
			Vector3(1.0, 0.10000058679893545, 0.0),
			Vector3(0.0, 1.0, 0.0)
		)
	)
	
	
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.10406410021404966, 0.019179956398343023, 0.0),
			Vector3(0.9539973643458035, 0.2998149909882096, 0.0)
		), 
		Ray(
			Vector3(0.19714012373471418, 0.057249066140327785, 0.0),
			Vector3(0.9603268728373183, 0.27887684971416504, 0.0)
		)
	)
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.2775529249124851, 0.06014021545405121, 0.0),
			Vector3(0.9272698377290306, 0.3743937072627921, 0.0)
		), 
		Ray(
			Vector3(0.3681060110931993, 0.1068971905693085, 0.0),
			Vector3(0.9603268728373183, 0.27887684971416504, 0.0)
		)
	)
	
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(2.6705152044506866, -3.881345361696637, 0.0),
			Vector3(0.9787290224843935, -0.20515725808935814, 0.0)
		), 
		Ray(
			Vector3(1.074226606929373, -0.4562923738114699, 0.0),
			Vector3(0.7278699075061423, 0.6857152453803255, 0.0)
		)
	)
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.0978821985913021, -0.01817220482208759, 0.0),
			Vector3(0.9728698029950091, -0.23135329351546338, 0.0)
		), 
		Ray(
			Vector3(0.19519002900219762, -0.04135605493611016, 0.0),
			Vector3(0.9782828203359017, -0.20727451226726762, 0.0)
		)
	)
	
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.23742075732895263, 0.7414288393156969, 0.0),
			Vector3(0.8842857628952637, 0.46694613130503765, 0.0)
		), 
		Ray(
			Vector3(0.3258452514703535, 0.7881397802886814, 0.0),
			Vector3(0.8799880384505118, 0.47499584438605413, 0.0)
		)
	)
	
	#Not in cad file:
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.4718680492853997, 0.32264860610453283, 0.0),
			Vector3(0.8222193152059611, 0.5691707983569082, 0.0)
		), 
		Ray(
			Vector3(0.5540753243831122, 0.37958916119846386, 0.0),
			Vector3(0.8249705835842306, 0.5651756684613148, 0.0)
		),
		False
	)
	lp = TwoArcLocalPlan(
		Ray(
			Vector3(0.4718680492853997, 0.32264860610453283, 0.0),
			Vector3(0.8222193152059611, 0.5691707983569082, 0.0)
		), 
		Ray(
			Vector3(0.5540753243831122, 0.37958916119846386, 0.0),
			Vector3(0.8249705835842306, 0.5651756684613148, 0.0)
		),
		True
	)
	points = []
	for i in np.arange(0, 1, 0.1):
		points.append(lp.get_point_t(i))
		
	_R = Vector3(10.145619,8.244399,0)
	R = Vector3(13.0006178,7.489313,0)
	
	T = Vector3(19.765736, 3.835348, 0)
	T_= Vector3(29.998669, 8.076674, 0)
	
	calc_2arc_joining_path(
		Ray(R, (R-_R).normalized),
		Ray(T, (T_-T).normalized)
	) #~3.48
	
	_R = Vector3(10,13,0)
	R = Vector3(16.9578,14,0)
	
	T = Vector3(20, 8, 0)
	T_= Vector3(26, 6, 0)
	
	calc_2arc_joining_path(
		Ray(R, (R-_R).normalized),
		Ray(T, (T_-T).normalized)
	) #~2.00
	
	calc_2arc_joining_path(
		Ray(T, (_R-R).normalized),
		Ray(R, (T-T_).normalized)
	) #~2.00
	
	#Create a robot
	robot = Robot(
		x=0, y=0, heading=0, #pose
		wheel_separation=0.5, wheel_radius=0.1, #wheels
		max_speed=.5, max_acceleration=2, #speed
		max_angular_velocity=1.5, max_angular_acceleration=40 #angular speed
	)
	
	#Create a path
	path = []
	d = 1
	# path.append(Vector3(0,0,0))
	# path.append(Vector3(d,0,0))
	# path.append(Vector3(d,d,0))
	# path.append(Vector3(0,d,0))
	# path.append(Vector3(0,0,0))
	
	p0 = Vector3(0,0,0)
	path.append(p0)
	
	d0 = Vector3(1,0,0)
	for i in range(0,100):
		d1 = Quaternion.from_angle_axis(random.uniform(-math.pi/2,math.pi/2), Vector3.up) * d0
		rd = random.uniform(0.1*d, d)
		p1 = p0 + d1 * rd
		path.append(p1)
		p0 = p1
		d0 = d1
	# plot.set_data("Local Plan", [p.x for p in points], [p.y for p in points])
	
	#Create a path follower
	#Setup the path follower with the robot, path, look ahead distance, max velocity, 
	# max acceleration, max angular velocity, max angular acceleration
	path_follower = PathFollower(robot, path, 0.1)
	
	#path_follower.get_local_path_plan(robot, Ray(Vector3(1,1,0), Vector3(1,.2,0).normalized))
	
	#Create a plot
	plot = Plot()
	plot.add_line("Robot Position", "x", "y")
	plot.add_line("Path", "x", "y")
	plot.add_line("Look Ahead Point", "x", "y")
	plot.add_line("Local Plan", "x", "y")
	#plot.add_line("Robot Velocity", "x", "y")
	#plot.add_line("Robot Heading", "x", "y")
	#plot.add_line("Robot Angular Velocity", "x", "y")
	
	step = 0
	dt = 0.1
	#Run the simulation
	while True:
		step += 1
		#Update the path follower
		path_follower.update(dt)
		#Update the robot
		robot.update(dt)
		#Update the plot
		plot.add_data("Robot Position", robot.position.x, robot.position.y)
		plot.add_data("Path", path_follower.current_path_segment.start.x, path_follower.current_path_segment.start.y)
		plot.add_data("Look Ahead Point", path_follower.current_path_segment.start.x + path_follower.current_path_segment.direction.x * (path_follower.current_path_segment.distance + path_follower.look_ahead_distance), path_follower.current_path_segment.start.y + path_follower.current_path_segment.direction.y * (path_follower.current_path_segment.distance + path_follower.look_ahead_distance))
		#plot.add_data("Robot Velocity", robot.velocity.x, robot.velocity.y)
		#plot.add_data("Robot Heading", robot.position.x + math.cos(robot.heading), robot.position.y + math.sin(robot.heading))
		#plot.add_data("Robot Angular Velocity", robot.position.x + math.cos(robot.heading + robot.angular_velocity), robot.position.y + math.sin(robot.heading + robot.angular_velocity))
		local_plan_raster = []
		try:
			p0 = None
			d0 = None
			for i in np.arange(0, 1, 0.02):
				rp = path_follower.local_plan.get_point_t(i)
				
				if p0 is not None:
					d1 = Vector3.distance(p0,rp)
					if d0 is not None and abs(d0-d1)>1.5*d0:
						print("ERROR! Discontinuity detected")
					d0 = d1
				
				local_plan_raster.append(rp)
				p0 = rp
			plot.set_data("Local Plan", [p.x for p in local_plan_raster], [p.y for p in local_plan_raster])
		except:
			pass
		plot.update()
		print(step)
		# time.sleep(2)