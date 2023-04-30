import numpy as np
from scipy.special import erf
from scipy.integrate import quad, fixed_quad, trapz, simps


class CalculateFiringRate:
  def __init__(self):
    self.tau_rp = 0.002 # refractory period (s) 2 ms
    self.tau = 0.05 # synaptic time constant (s) 50 ms
    self.V_r = 0.0 # reset potential (mV)
    self.mu_0 = 20 # mean firing rate (Hz)
    self.sigma_0 = 1 # standard deviation of firing rate (Hz)
    self.theta = 20.0 # threshold potential (mV)
    self.pi_sqrt = np.sqrt(np.pi)
    
  # Define the integrand
  def integrand(self, u):
    return np.exp(u**2) * (1 + erf(u))

  # fixed_quad method
  def fixed_quad(self, a, b):
    # Define the number of quadrature points
    num_points = 200

    # Approximate the integral using fixed_quad
    integral, error = fixed_quad(self.integrand, a, b, n=num_points)

    return integral


  # trapz method
  def trapz(self, a, b):
    # Define the number of points for trapezoidal rule
    num_points = 200

    # Evaluate the integrand at the integration points
    x = np.linspace(a, b, num_points)
    y = self.integrand(x)

    # Approximate the integral using the trapezoidal rule
    integral = trapz(y, x)

    return integral
  

  # simps method
  def simps(self, a, b):
    # Define the number of points for Simpson's rule
    num_points = 201

    # Evaluate the integrand at the integration points
    x = np.linspace(a, b, num_points)
    y = self.integrand(x)

    # Approximate the integral using Simpson's rule
    integral = simps(y, x)

    return integral
      

  def calculate_nu_0(self, mu_0, sigma_0):
    # Define integration limits
    a = (self.V_r - mu_0) / sigma_0
    b = (self.theta - mu_0) / sigma_0

    # Calculate the integral using fixed_quad
    integral = self.fixed_quad(a, b)

    # Calculate nu_0
    nu_0 = 1 / (self.tau_rp + self.tau * self.pi_sqrt * integral)
    return nu_0



mu_values = (16, 18, 20, 22, 24, 26, 28, 30)
sigma_values = (2, 3, 4, 5, 6)

calculation = CalculateFiringRate()

for sigma in sigma_values:
  mu_firing_rates = []

  for mu in mu_values:
    calculation.calculate_nu_0(mu, sigma)
    mu_firing_rates.append(calculation.calculate_nu_0(mu, sigma))
  
  print("\nsigma: ", sigma, "firing rates:")
  print(*mu_firing_rates, sep=", ")





