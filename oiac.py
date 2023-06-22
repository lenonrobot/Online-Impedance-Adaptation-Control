"""
Xiong, X., & Fang, C. (2023). An Online Impedance Adaptation Controller for Decoding Skill Intelligence. 
Biomimetic Intelligence and Robotics, 3(2), [100100]. https://doi.org/10.1016/j.birob.2023.100100

"""
import numpy as np
import numpy.linalg as la

class ada_imp_con( ):
	"""Online impedance adaptation"""
	def __init__(self, dof):

		self.DOF = dof# degree of freedom of a robot arm

		self.k_mat = np.mat(np.zeros((self.DOF, self.DOF)))#stiffness parameter matrix
		self.b_mat = np.mat(np.zeros((self.DOF, self.DOF)))#damping parameter matrix
		self.ff_tau_mat = np.mat(np.zeros((self.DOF, 1)))

		self.q = np.mat(np.zeros((self.DOF, 1)))#real joint angle matrix
		self.q_d = np.mat(np.zeros((self.DOF, 1)))#desired joint angle matrix

		self.dq = np.mat(np.zeros((self.DOF, 1)))#real joint velocity matrix
		self.dq_d = np.mat(np.zeros((self.DOF, 1)))#desired joint velocity matrix
		self.a = 0.2
		self.b = 5.0
		self.k = 0.05


	def update_impedance(self, q, q_d, dq, dq_d):#tune stiffness and damping matrices, see Eq.(2)
		#copy inputs
		self.q = np.mat(np.copy(q)).T#real joint angle matrix
		self.q_d = np.mat(np.copy(q_d)).T#desired joint angle matrix
		self.dq = np.mat(np.copy(dq)).T#real joint velocity matrix
		self.dq_d = np.mat(np.copy(dq_d)).T#desired joint velocity
		#update stiffness K and damping B
		self.k_mat = (self.gen_track_err() * self.gen_pos_err().T)/self.gen_for_factor()
		self.b_mat = (self.gen_track_err() * self.gen_vel_err().T)/self.gen_for_factor()

		return self.k_mat, self.b_mat

	def gen_pos_err(self):#position error, see Eq. (1)
		return (self.q - self.q_d)

	def gen_vel_err(self):#velocity error, see Eq. (1)
		return (self.dq - self.dq_d)

	def gen_track_err(self):#tracking error, see Eq. (3)
		return (self.k * self.gen_vel_err() + self.gen_pos_err())

	def gen_ad_factor(self):#adaptation scalar, see Eq. (3)
		return self.a/(1.0 + self.b * la.norm(self.gen_track_err()) * la.norm(self.gen_track_err()))

"""
#Pseudocode

if __name__ == "__main__":
	self.ada_imp = aic.ada_imp_con(dof) # degree of freedom of robot arm

	while(t<t_max):
		...
		#get robot arm joint angles and velocity
  		#NOTE THAT the online impedance adaptation control requires the feedforward control tau_ff, e.g., 
    		#(1) linear (tracking) error-based control, e.g., tau_ff = a * e, a is the scalar constant, e  is the error between desired and real joint position.
		#(2)data-driven learning or opmtimization of feedforward joint torque(s) tau_ff

		self.ada_imp.(q, q_d, dq, dq_d) #undate stiffness (self.k_mat) and damping (self.b_mat) matrices 

		tau_fb =  self.k_mat* (q_d-q) + self.k_mat*(dq_d-dq) #compute feedback joint torque(s)

		tau = tau_ff + tau_fb #compute total joint torques

		...
"""
