import numpy as np
import matplotlib.pyplot as plt


def Hodgkin_Huxley(V=-65,m=0.05,h=0.6,n=0.32):
    # Hodgkin-Huxley neuron parameters
    C_m = 1.  # Membrane capacitance
    g_Na = 120.0  # Sodium (Na) conductance
    g_K = 36.0  # Potassium (K) conductance
    g_L = 0.3  # Leak conductance
    E_Na = 50.0  # Sodium (Na) reversal potential
    E_K = -77.0  # Potassium (K) reversal potential
    E_L = -54.387  # Leak reversal potential
    dt = 0.01  # Time step for simulation
    timesteps = 10000  # Number of time steps

    # Initial values
    V = -65.0  # Initial membrane potential (mV)
    m = 0.05  # Initial sodium activation gating variable
    h = 0.6  # Initial sodium inactivation gating variable
    n = 0.32  # Initial potassium activation gating variable

    # Stimulation current
    I_inj = np.zeros(timesteps)
    I_inj[1000:4000] = 10.0  # Inject current between steps 1000 and 4000

    # Arrays to store results
    V_arr = np.zeros(timesteps)
    m_arr = np.zeros(timesteps)
    h_arr = np.zeros(timesteps)
    n_arr = np.zeros(timesteps)

    # Hodgkin-Huxley model simulation
    for i in range(timesteps):
        # Hodgkin-Huxley equations
        alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

        m += dt * (alpha_m * (1.0 - m) - beta_m * m)
        h += dt * (alpha_h * (1.0 - h) - beta_h * h)
        n += dt * (alpha_n * (1.0 - n) - beta_n * n)

        g_Na_t = g_Na * m**3 * h
        g_K_t = g_K * n**4
        g_L_t = g_L

        I_Na = g_Na_t * (V - E_Na)
        I_K = g_K_t * (V - E_K)
        I_L = g_L_t * (V - E_L)

        I_total = I_inj[i] - I_Na - I_K - I_L
        V += (I_total / C_m) * dt

        # Store results
        V_arr[i] = V
        m_arr[i] = m
        h_arr[i] = h
        n_arr[i] = n

    # Plot membrane potential and gating variables
    time = np.arange(0, timesteps * dt, dt)
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(time, V_arr, 'b')
    plt.title('Membrane Potential (mV)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')

    plt.subplot(2, 1, 2)
    plt.plot(time, m_arr, 'r', label='m')
    plt.plot(time, h_arr, 'g', label='h')
    plt.plot(time, n_arr, 'c', label='n')
    plt.title('Gating Variables')
    plt.xlabel('Time (ms)')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.show()


Hodgkin_Huxley(V=0.65)
