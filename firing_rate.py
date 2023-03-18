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

    # recording devices
    self.voltmeter = nest.Create("voltmeter")
    self.spike_recorder = nest.Create("spike_recorder")

  def run_simulation(self):

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
    nest.Connect(neurons, self.spike_recorder)
    nest.Connect(poisson, neurons, "one_to_one", syn_spec=syn_params)
    nest.Connect(self.voltmeter, neurons)

    # Run simulation
    nest.Simulate(self.sim_time)

  # displays the mean firing rate of the population
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
    mean_firing_rate = np.mean(firing_rates[40:])

    # Calculate mean firing rate as the mean of spike counts starting from 400ms
    print("------------------------------------")
    print("mean firing rate:", np.mean(mean_firing_rate))
    print("------------------------------------")
    plt.plot(np.arange(num_bins) * bin_size, firing_rates)
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.show()
    # return mean_firing_rate

  
# Run simulation
sim = Simulation()
sim.run_simulation()
sim.firing_rate()




        