import numpy as np
from sailenv.agent import Agent
import time
import matplotlib.pyplot as plt
from scipy.stats import sem, t

confidence = 0.95

total = 100

sizes = [
    (100, 75),
    (200, 150),
    (400, 300),
    (600, 450),
    (800, 600),
    (1000, 750),
    (1200, 900)
]

def total_size(frame):
    sizes = frame["sizes"]
    total = sizes["main"] + sizes["object"] + sizes["category"] + sizes["flow"] + sizes["depth"]
    print(sizes)
    return total

if __name__ == "__main__":

    gzip_sets = [False, True]
    get_frame_time_data_list = [[], []]
    get_frame_size_data_list = [[], []]
    for gzip, get_frame_time_data, get_frame_size_data in zip(gzip_sets, get_frame_time_data_list, get_frame_size_data_list):
        for size in sizes:
            print("Generating agent...")
            agent = Agent(flow_frame_active=True, object_frame_active=True, main_frame_active=True,
                          category_frame_active=True, depth_frame_active=True, width=size[0], height=size[1],
                          host="localhost", port=8085, gzip=gzip)
            print(f"Registering agent on server ({size[0]}, {size[1]})...")
            agent.register()
            print(f"Agent registered with ID: {agent.id} and gzip {gzip}")

            try:
                print("Press ESC to close")
                optical_flow = lve.OpticalFlowCV()
                get_frame_list = []
                frame_size_list = []

                i = 0
                while i < total:
                    start_get_frame = time.time()
                    frame = agent.get_frame()
                    step_get_frame = time.time() - start_get_frame

                    print(f"Frame {i}/{total}")

                    if i != 0:
                        get_frame_list.append(step_get_frame)
                        frame_size_list.append(total_size(frame))
                    i += 1


                def mean_with_ci(data):
                    mean = np.mean(data)
                    n = len(data)
                    std_err = sem(data)
                    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
                    return mean, h


                get_frame_time_data.append(mean_with_ci(get_frame_list))
                get_frame_size_data.append(mean_with_ci(frame_size_list))

            finally:
                agent.delete()

    y_axis = [f"{w}x{h}" for w, h in sizes]


    def get_data_and_ci(data_tuple_list):
        data_list = [val for val, ci in data_tuple_list]
        ci_list = [ci for val, ci in data_tuple_list]
        return data_list, ci_list


    a = plt.figure(1)
    data_list, ci_list = get_data_and_ci(get_frame_time_data_list[0])
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"gzip={gzip_sets[0]}")
    plt.ylabel(f"time to elaborate and transmit all frames in s")
    data_list, ci_list = get_data_and_ci(get_frame_time_data_list[1])
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"gzip={gzip_sets[1]}")
    plt.legend()
    a.show()

    def rescale(data_list, scale):
        scaled = np.array(data_list) / scale
        return scaled.tolist()

    b = plt.figure(2)
    data_list, ci_list = get_data_and_ci(rescale(get_frame_size_data_list[0], 1e6))
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"gzip={gzip_sets[0]}")
    plt.ylabel(f"total size of all frame views in MB")
    data_list, ci_list = get_data_and_ci(rescale(get_frame_size_data_list[1], 1e6))
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"gzip={gzip_sets[1]}")
    plt.legend()
    b.show()

    c, axs = plt.subplots(2)
    data_list, ci_list = get_data_and_ci(rescale(get_frame_size_data_list[0], 1e6))
    axs[0].errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"gzip={gzip_sets[0]}")
    axs[0].set_ylabel(f"size in MB, gzip: {gzip_sets[0]}")
    axs[0].legend()
    data_list, ci_list = get_data_and_ci(rescale(get_frame_size_data_list[1], 1e6))
    axs[1].errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"gzip={gzip_sets[1]}")
    axs[1].set_ylabel(f"size in MB, gzip: {gzip_sets[1]}")
    axs[1].legend()
    c.show()


    input()