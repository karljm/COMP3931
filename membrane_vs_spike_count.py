import nest
import nest.voltage_trace
import matplotlib.pyplot as plt

class Simulation:
  def __init__(self):
    # Set up simulation parameters
    self.num_neurons = 10000
    self.rate_poisson = 800  # Hz
    self.sim_time = 200.0    # ms
    self.E_L = 0.0           # resting potential, mV
    self.V_reset = 0.0       # reset potential, mV
    self.V_th = 20.0         # spike threshold, mV
    self.tau_m = 50.0        # membrane time constant, ms
    self.j = 0.03 * (self.V_th - self.E_L)     # synaptic strength, mV

  def run_simulation(self):
    # Set up recording devices
    voltmeter = nest.Create("voltmeter")
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
    nest.Connect(voltmeter, neurons)

    # Run simulation
    nest.Simulate(self.sim_time)

    # Get membrane potentials from the voltmeter and extract the final values
    membrane_potentials = nest.GetStatus(voltmeter, "events")[0]["V_m"]
    final_membrane_potentials = membrane_potentials[-self.num_neurons:]

    # Plot the histogram of the final membrane potentials
    plt.hist(final_membrane_potentials, bins=50)
    plt.title("Membrane Potential vs Spike Count")
    plt.xlabel("Membrane Potential (mV)")
    plt.ylabel("Spike Count")
    plt.show()
  
# Run simulation
sim = Simulation()
sim.run_simulation()




        