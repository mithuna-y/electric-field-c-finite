import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Set up the grid of points
x_range = 10
y_range = 10
points = x_range + 1
x = np.linspace(0, x_range, points)
y = np.linspace(0, y_range, points)
X, Y = np.meshgrid(x, y)

# number_of_particles
number_of_particles = 1

# speed of light
c = 5

# velocity of the charged particle
velocity_x = 0
velocity_y = 1
FREQUENCY = 0.01

# scale of the vectors
scale = 1

# maximum length of each particle's electric field contribution
max_vector_length = 1

# Set up the figure
fig, ax = plt.subplots()

# Set the number of time steps
n_steps = 100

# Set the time step
dt = 1

# Set up the plot
ax.set_xlim(0, x_range)
ax.set_ylim(0, y_range)
quiver = ax.quiver(X, Y, np.zeros((points, points)), np.zeros((points, points)), scale=1)

# Set up the data
x_electric_field = np.zeros((points, points, n_steps))
y_electric_field = np.zeros((points, points, n_steps))


# path of the particle

def particle_x_position(time):
    global x_range
    return x_range / 2 + 0.1


def particle_y_position(start, time):
    return oscillation(start, time)


def oscillation(start, time):
    global FREQUENCY
    return start + np.sin(FREQUENCY * time)


def sudden_acceleration(time):
    time_for_acceleration = 10
    if time < time_for_acceleration:
        return -10 + time * velocity_y
    if time_for_acceleration < time < time_for_acceleration + 1:
        return -10 + velocity_y * time_for_acceleration + 0.05 * time ** 2
    else:
        return -10 + velocity_y * time_for_acceleration + 0.05 * 1 + velocity_y * time


def particles(time):
    return [(particle_x_position(time), particle_y_position(2, time)) for _ in
            range(number_of_particles)]
    # return [(particle_x_position(time), particle_y_position(y_range/number_of_particles * n, time))
    # for n in range(number_of_particles)]


# scatter plot for the position of the particle/s
particles_x_0 = [x for (x, y) in particles(0)]
particles_y_0 = [y for (x, y) in particles(0)]
scat = ax.scatter(particles_x_0, particles_y_0, c='r', s=50)

# text
text = ax.text(0.5, 0.5, '', transform=ax.transAxes)

# electric constants
k = 3
q = 1

# making a function to cap the length of the vectors
def cap(length):
    global max_vector_length
    return min(length, max_vector_length)


vec_cap = np.vectorize(cap)


def update_electric_field(x_field, y_field, now_step):
    t_now = now_step * dt
    # go through each future time slice t_future. If x, y such that distance to where the charge was at is greater
    # than c*(t_future-t_now) then update the electric field using x_position and y_position as the positions of the
    # particle in the calculation. If x,y less than the required distance, don't update.

    for future_step in range(now_step + 1, n_steps):
        t_future = future_step * dt
        for x in range(0, x_range):
            for y in range(0, y_range):
                electric_field_vector = np.array([0.0, 0.0])
                for particle in particles(t_now):
                    x_position = particle[0]
                    y_position = particle[1]
                    distance_from_charge = np.sqrt((x - x_position) ** 2 + (y - y_position) ** 2)
                    t_signal_reaches = t_now + distance_from_charge / c
                    if t_future > t_signal_reaches:
                        electric_field_strength = cap(k * q / (distance_from_charge ** 2))
                        x_unit_vector = (x_position - x) / distance_from_charge
                        y_unit_vector = (y_position - y) / distance_from_charge
                        electric_field_vector += electric_field_strength * np.array([x_unit_vector, y_unit_vector])
                x_field[y, x, future_step] = scale * electric_field_vector[0]
                y_field[y, x, future_step] = scale * electric_field_vector[1]
    return x_field, y_field


def update(now_step, x_electric_field, y_electric_field):
    t_now = now_step * dt

    particles_x = [x for (x, y) in particles(t_now)]
    particles_y = [y for (x, y) in particles(t_now)]
    # Update the position of the dot

    scat.set_offsets(np.c_[particles_x, particles_y])

    # text
    text.set_text('Time step: {}'.format(t_now))

    for future_step in range(now_step + 1, n_steps):
        t_future = future_step * dt
        for x in range(0, x_range):
            for y in range(0, y_range):
                electric_field_vector = np.array([0.0, 0.0])
                for particle in particles(t_now):
                    x_position = particle[0]
                    y_position = particle[1]
                    distance_from_charge = np.sqrt((x - x_position) ** 2 + (y - y_position) ** 2)
                    t_signal_reaches = t_now + distance_from_charge / c
                    if t_future > t_signal_reaches:
                        electric_field_strength = cap(k * q / (distance_from_charge ** 2))
                        x_unit_vector = (x_position - x) / distance_from_charge
                        y_unit_vector = (y_position - y) / distance_from_charge
                        electric_field_vector += electric_field_strength * np.array([x_unit_vector, y_unit_vector])
                x_electric_field[y, x, future_step] = scale * electric_field_vector[0]
                y_electric_field[y, x, future_step] = scale * electric_field_vector[1]

    # update the electric field

    # x_electric_field, y_electric_field = update_electric_field(x_electric_field, y_electric_field, now_step)

    # Update the vector plot data
    quiver.set_UVC(x_electric_field[:, :, now_step], y_electric_field[:, :, now_step])

    # Return the artists set
    return quiver,


# Animate the plot
ani = animation.FuncAnimation(fig, update, fargs=(x_electric_field, y_electric_field), frames=n_steps, interval=1,
                              repeat=False)

plt.show()
