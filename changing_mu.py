import nest
import nest.voltage_trace
import numpy as np
import matplotlib.pyplot as plt

class Simulation:
  def __init__(self):
    # Set up simulation parameters
    self.num_neurons = 10000
    self.rate_poisson = 800  # Hz
    self.sim_time = 4000.0    # ms
    self.E_L = 0.0           # resting potential, mV
    self.V_reset = 0.0       # reset potential, mV
    self.V_th = 20.0         # spike threshold, mV
    self.tau_m = 50.0        # membrane time constant, ms
    self.j = 0.03 * (self.V_th - self.E_L)     # synaptic strength, mV
    self.spike_recorder = None

  def run_simulation(self):
    print("\nRunning simulation...")
    spike_recorder = nest.Create("spike_recorder")

    # Set parameters for neurons, poisson generators and synapses
    neuron_params = {
      "E_L": self.E_L,
      "V_reset": self.V_reset,
      "V_th": self.V_th,
      "tau_m": self.tau_m
    }
    poisson_params = {"rate": self.rate_poisson}
    syn_params = {"weight": self.j}

    # Create neuron population and Poisson generators
    neurons = nest.Create("iaf_psc_delta", self.num_neurons)
    poisson = nest.Create("poisson_generator", self.num_neurons, params=poisson_params)

    # Configure neuron and synaptic parameters
    nest.SetStatus(neurons, params=neuron_params)
    nest.Connect(neurons, spike_recorder)
    nest.Connect(poisson, neurons, "one_to_one", syn_spec=syn_params)

    # Run simulation
    nest.Simulate(self.sim_time)

    self.spike_recorder = spike_recorder

  # returns the firing rates of the population for the current simulation
  def firing_rate(self):
    t_start = 2500.0
    t_end = self.sim_time
    bin_size = 10.0
    num_bins = int(self.sim_time / bin_size)
    spike_counts = np.zeros(num_bins)
    firing_rates = np.zeros(num_bins)

    spike_times = self.spike_recorder.get("events")["times"]
    
    filter_times = np.logical_and(spike_times >= t_start, spike_times < t_end)
    num_spikes = np.count_nonzero(filter_times)

    for i in range(num_bins):
      # Determine time window for current bin
      t_start = i * bin_size
      t_end = (i + 1) * bin_size

      # Count number of spikes within current time window
      filter_times = np.logical_and(spike_times >= t_start, spike_times < t_end)
      num_spikes = np.count_nonzero(filter_times)

      # Store number of spikes in current bin
      spike_counts[i] = num_spikes

    for i in range(num_bins):
      # Calculate firing rate as fraction of neurons firing within time window
      firing_rates[i] = spike_counts[i] / (self.num_neurons * bin_size)
      # convert firing rate to Hz
      firing_rates[i] = firing_rates[i] * 1000
    return firing_rates



# sigma2 is precalculated using benchmark values
sigma2 = 14400

# Create simulation object
sim = Simulation()

# mu values from 16 to 30 in steps of 2
mu_values = np.arange(4, 31, 2)
mean_firing_rates = []

for mu in mu_values:
  # convert to correct units for simulation
  mu = mu * 1000

  # calculate new poissson rate and j for current mu
  sim.rate_poisson = mu ** 2 / (sim.tau_m * sigma2)
  sim.j = sigma2 / mu
  sim.run_simulation()
  
  # Calculate mean firing rate as the mean of spikes starting from 500ms
  current_firing_rates = sim.firing_rate()
  mean_firing_rate = np.mean(current_firing_rates[50:])
  mean_firing_rates.append(mean_firing_rate)

  # print current firing rate for current mu
  print("\n\n\n------------------------------------")
  print("mu:", mu, "\nmean firing rate:", mean_firing_rate)
  print("------------------------------------\n\n\n")
  nest.ResetKernel()

# plot mean firing rate vs mu
plt.plot(mu_values, mean_firing_rates)
plt.title("Mean firing rate vs mu")
plt.xlabel("mu")
plt.ylabel("Mean Firing Rate (Hz)")
plt.show()

