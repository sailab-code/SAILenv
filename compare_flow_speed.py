import numpy as np
from sailenv.agent import Agent
import time
import matplotlib.pyplot as plt
from scipy.stats import sem, t
import seaborn as sns
from opticalflow_cv import OpticalFlowCV
import pandas as pd

sns.set_style("white")

FLOWNET_FLAG = True


class Dataframe_Wrap:
    def __init__(self, columns):
        self.columns = columns
        self.df = pd.DataFrame(columns=columns)  # class dataframe

    def appending(self, data):
        self.df = self.df.append(data, ignore_index=True)

    def to_csv(self, name, sep='\t', encoding='utf-8'):
        self.df.to_csv(name, sep, encoding)


total = 10
scene = 0
sizes = [
    (240, 180),
    (320, 240),
    # (640, 480),
    # (800, 600),
    # (1024, 768),
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


def collect_times_unity_plus_main(size, df_resolutions):
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
                data = pd.Series(["SAILenv", f"({size[0]}x{size[1]})", step_get_frame, size[0]], index=df_resolutions.columns)
                df_resolutions.appending(data)
            i += 1

        return get_flow_times
    finally:
        agent.delete()


def collect_times_cv(size, df_resolutions):
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
                data = pd.Series(["OpenCV", f"({size[0]}x{size[1]})", step_get_frame + step_cv_flow, size[0]], index=df_resolutions.columns)
                df_resolutions.appending(data)
            i += 1

        return get_frame_times, cv_flow_times
    finally:
        agent.delete()


def collect_times_flownet(size, df_resolutions):
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

    try:
        get_frame_times = []
        flownet_times = []
        flownetlite = FlowNetLiteWrapper(device="cuda:0", compare_flow=True)
        print("Loaded FlownetLite model...")
        i = 0

        while i < total:
            start_get_frame = time.time()
            frame = agent.get_frame()
            step_get_frame = time.time() - start_get_frame
            start_cv_flow = time.time()
            flow, to_cuda_time = flownetlite(frame["main"])
            step_flownet_time = time.time() - start_cv_flow - to_cuda_time

            print(f"Frame {i}/{total}")

            if i != 0:
                get_frame_times.append(step_get_frame)
                flownet_times.append(step_flownet_time)
                data = pd.Series(["LiteFlowNet", f"({size[0]}x{size[1]})", step_get_frame + step_flownet_time, size[0]], index=df_resolutions.columns)
                df_resolutions.appending(data)
            i += 1

        return get_frame_times, flownet_times
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

    # unity_flow_time_per_size = []
    #
    # unity_flow_plus_main_per_size = []
    # unity_mean_std = []
    #
    # cv_flow_time_per_size = []
    # main_frame_time_per_size = []
    # opencv_mean_std = []
    #
    # flownet_time_per_size = []
    # flownet_main_time_per_size = []
    # flownet_mean_std = []

    columns = ["Method", "Resolution", "Seconds", "width"]
    df_resolutions = Dataframe_Wrap(columns=columns)  # class dataframe

    for size in sizes:

        # computing time  and  collecting statistics
        ################################
        # unity total time
        unity_flow_main_times = collect_times_unity_plus_main(size, df_resolutions)
        # unity_flow_plus_main_per_size.append(mean_with_ci(unity_flow_main_times))
        # unity_mean_std.append([np.mean(unity_flow_main_times), np.std(unity_flow_main_times)])

        # opencv get frame + farneback
        get_frame_times, cv_flow_times = collect_times_cv(size, df_resolutions)
        # cv_flow_time_per_size.append(mean_with_ci(cv_flow_times))
        # main_frame_time_per_size.append(mean_with_ci(get_frame_times))
        # opencv_mean_std.append([np.mean(get_frame_times + cv_flow_times), np.std(get_frame_times + cv_flow_times)])

        if FLOWNET_FLAG:
            # flownet get frame + propagation
            get_frame_times_net, cv_flow_times_net = collect_times_flownet(size, df_resolutions)
            # flownet_time_per_size.append(mean_with_ci(cv_flow_times_net))
            # flownet_main_time_per_size.append(mean_with_ci(get_frame_times_net))
            #
            # flownet_mean_std.append(
            #     [np.mean(get_frame_times_net + cv_flow_times_net), np.std(get_frame_times_net + cv_flow_times_net)])

    # y_axis = [f"{w}x{h}" for w, h in sizes]
    #
    # cv_flow_get_frame_time_per_size = [
    #     (flow[0] + frame[0], flow[1] + frame[1]) for flow, frame in zip(cv_flow_time_per_size, main_frame_time_per_size)
    # ]
    # if FLOWNET_FLAG:
    #     flownet_cv_flow_get_frame_time_per_size = [
    #         (flow[0] + frame[0], flow[1] + frame[1]) for flow, frame in
    #         zip(flownet_time_per_size, flownet_main_time_per_size)
    #     ]
    #
    # a = plt.figure(1)
    #
    # plt.ylabel(f"time to obtain Optical Flow")
    # # data_list, ci_list = get_data_and_ci(unity_flow_time_per_size)
    # # plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Unity")
    # data_list, ci_list = get_data_and_ci(unity_flow_plus_main_per_size)
    # plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Unity + main frame")
    # # data_list, ci_list = get_data_and_ci(cv_flow_time_per_size)
    # # plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Open-CV")
    # data_list, ci_list = get_data_and_ci(cv_flow_get_frame_time_per_size)
    # plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Open-CV + get image overhead")
    # if FLOWNET_FLAG:
    #     data_list, ci_list = get_data_and_ci(flownet_cv_flow_get_frame_time_per_size)
    #     plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label=f"Flownet + get image overhead")
    # plt.legend()
    # a.show()
    #
    # print("Ciao")
    df_resolutions.df.sort_values(['width', "Method"], inplace=True, ascending=True)

    sns.lineplot('Resolution', 'Seconds', hue="Method",  style="Method", data=df_resolutions.df, ci=95, sort=False, markers=True)

    plt.savefig("temp_comparison.pdf", bbox_inches='tight')
    plt.show()