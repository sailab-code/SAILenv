import numpy as np
from sailenv.agent import Agent
import time
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import seaborn as sns
from opticalflow_cv import OpticalFlowCV


FLOWNET_FLAG = True



total = 100
scene = 0
sizes = [
    (240, 180),
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 960)
]

# sizes = sizes[:2]

def collect_times_unity(size):
    print("Unity only of timings...")
    print("Generating agent...")
    agent = Agent(flow_frame_active=True, object_frame_active=False, main_frame_active=False,
                  category_frame_active=False, depth_frame_active=False, width=size[0], height=size[1],
                  host="localhost", port=8085, gzip=False)
    print(f"Registering agent on server ({size[0]}, {size[1]})...")
    agent.register()
    agent.change_scene(agent.scenes[scene])
    print(f"Agent registered with ID: {agent.id}")

    try:
        get_flow_times = []
        i = 0

        while i < total:
            start_get_frame = time.time()
            # this will measure the time to obtain only optical flow
            frame = agent.get_frame()
            step_get_frame = time.time() - start_get_frame

            print(f"Frame {i}/{total}")

            if i != 0:
                get_flow_times.append(step_get_frame)
            i += 1

        return get_flow_times
    finally:
        agent.delete()


def collect_times_unity_plus_main(size):
    print("Unity of + main timings...")
    print("Generating agent...")
    agent = Agent(flow_frame_active=True, object_frame_active=False, main_frame_active=True,
                  category_frame_active=False, depth_frame_active=False, width=size[0], height=size[1],
                  host="localhost", port=8085, gzip=False)
    print(f"Registering agent on server ({size[0]}, {size[1]})...")
    agent.register()
    agent.change_scene(agent.scenes[scene])
    print(f"Agent registered with ID: {agent.id}")

    try:
        get_flow_times = []
        i = 0

        while i < total:
            start_get_frame = time.time()
            # this will measure the time to obtain only optical flow
            frame = agent.get_frame()
            step_get_frame = time.time() - start_get_frame

            print(f"Frame {i}/{total}")

            if i != 0:
                get_flow_times.append(step_get_frame)
            i += 1

        return get_flow_times
    finally:
        agent.delete()


def collect_times_cv(size):
    print("OpenCV timings...")
    print("Generating agent...")
    agent = Agent(flow_frame_active=False, object_frame_active=False, main_frame_active=True,
                  category_frame_active=False, depth_frame_active=False, width=size[0], height=size[1],
                  host="localhost", port=8085, gzip=False)
    print(f"Registering agent on server ({size[0]}, {size[1]})...")
    agent.register()
    agent.change_scene(agent.scenes[scene])
    print(f"Agent registered with ID: {agent.id}")

    try:
        get_frame_times = []
        cv_flow_times = []
        optical_flow = OpticalFlowCV()
        i = 0

        while i < total:
            start_get_frame = time.time()
            frame = agent.get_frame()
            step_get_frame = time.time() - start_get_frame

            start_cv_flow = time.time()
            flow = optical_flow(frame["main"])
            step_cv_flow = time.time() - start_cv_flow



            print(f"Frame {i}/{total}")

            if i != 0:
                get_frame_times.append(step_get_frame)
                cv_flow_times.append(step_cv_flow)
            i += 1

        return get_frame_times, cv_flow_times
    finally:
        agent.delete()



def collect_times_flownet(size):
    from lite_flow_utils import FlowNetLiteWrapper
    print("FlowNet timings...")
    print("Generating agent...")
    agent = Agent(flow_frame_active=False, object_frame_active=False, main_frame_active=True,
                  category_frame_active=False, depth_frame_active=False, width=size[0], height=size[1],
                  host="localhost", port=8085, gzip=False)
    print(f"Registering agent on server ({size[0]}, {size[1]})...")
    agent.register()
    agent.change_scene(agent.scenes[scene])
    print(f"Agent registered with ID: {agent.id}")
    flownetlite = FlowNetLiteWrapper()
    print("Loaded FlownetLite model...")

    try:
        get_frame_times = []
        cv_flow_times = []
        # optical_flow = OpticalFlowCV()
        i = 0

        while i < total:
            start_get_frame = time.time()
            frame = agent.get_frame()
            step_get_frame = time.time() - start_get_frame

            start_cv_flow = time.time()
            flow = flownetlite(frame["main"])
            step_cv_flow = time.time() - start_cv_flow



            print(f"Frame {i}/{total}")

            if i != 0:
                get_frame_times.append(step_get_frame)
                cv_flow_times.append(step_cv_flow)
            i += 1

        return get_frame_times, cv_flow_times
    finally:
        agent.delete()


confidence = 0.95
def mean_with_ci(data):
    mean = np.mean(data)
    n = len(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return mean, h

def get_data_and_ci(data_tuple_list):
    data_list = [val for val, ci in data_tuple_list]
    ci_list = [ci for val, ci in data_tuple_list]
    return data_list, ci_list

if __name__ == '__main__':

    unity_flow_time_per_size = []
    unity_flow_plus_main_per_size = []
    cv_flow_time_per_size = []
    main_frame_time_per_size = []
    flownet_time_per_size = []
    flownet_main_time_per_size = []

    for size in sizes:
        # unity_flow_times = collect_times_unity(size)
        unity_flow_main_times = collect_times_unity_plus_main(size)
        get_frame_times, cv_flow_times = collect_times_cv(size)
        get_frame_times_net, cv_flow_times_net = collect_times_flownet(size)

        # unity_flow_time_per_size.append(mean_with_ci(unity_flow_times))
        unity_flow_plus_main_per_size.append(mean_with_ci(unity_flow_main_times))
        cv_flow_time_per_size.append(mean_with_ci(cv_flow_times))
        main_frame_time_per_size.append(mean_with_ci(get_frame_times))
        if FLOWNET_FLAG:
            flownet_time_per_size.append(mean_with_ci(cv_flow_times_net))
            flownet_main_time_per_size.append(mean_with_ci(get_frame_times_net))

    y_axis = [f"{w}x{h}" for w, h in sizes]

    cv_flow_get_frame_time_per_size = [
        (flow[0] + frame[0], flow[1] + frame[1]) for flow, frame in zip(cv_flow_time_per_size, main_frame_time_per_size)
    ]
    if FLOWNET_FLAG:
        flownet_cv_flow_get_frame_time_per_size = [
            (flow[0] + frame[0], flow[1] + frame[1]) for flow, frame in zip(flownet_time_per_size, flownet_main_time_per_size)
        ]

    a = plt.figure(1)
    plt.ylabel(f"time to obtain Optical Flow")
    # data_list, ci_list = get_data_and_ci(unity_flow_time_per_size)
    # plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Unity")
    data_list, ci_list = get_data_and_ci(unity_flow_plus_main_per_size)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Unity + main frame")
    # data_list, ci_list = get_data_and_ci(cv_flow_time_per_size)
    # plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Open-CV")
    data_list, ci_list = get_data_and_ci(cv_flow_get_frame_time_per_size)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Open-CV + get image overhead")
    if FLOWNET_FLAG:
        data_list, ci_list = get_data_and_ci(flownet_cv_flow_get_frame_time_per_size)
        plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Flownet + get image overhead")
    plt.legend()
    a.show()



