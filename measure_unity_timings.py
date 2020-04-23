import numpy as np
import cv2
from agent import Agent
import time
import matplotlib.pyplot as plt
from scipy.stats import sem, t

confidence = 0.95

total = 20

sizes = [
    (100, 75),
    (200, 150),
    (400, 300),
    (600, 450),
    (800, 600),
    (1000, 750),
    (1200, 900)
]

if __name__ == "__main__":

    main_data = []
    object_data = []
    category_data = []
    flow_data = []
    depth_data = []
    cv_flow_data = []
    total_data = []
    http_data = []

    for size in sizes:
        print("Generating agent...")
        agent = Agent(flow_frame_active=True, object_frame_active=False, main_frame_active=True,
                      category_frame_active=True, depth_frame_active=False, width=size[0], height=size[1],
                      host="127.0.0.1", port=8081)
        cv_flow_active = False
        print(f"Registering agent on server ({size[0]}, {size[1]})...")
        agent.register()
        print(f"Agent registered with ID: {agent.id}")

        try:
            print("Press ESC to close")
            optical_flow = lve.OpticalFlowCV()
            main_time_list = []
            object_time_list = []
            category_time_list = []
            flow_time_list = []
            depth_time_list = []
            cv_flow_time_list = []
            total_time_list = []
            http_time_list = []

            i = 0
            while i < total:
                frame = agent.get_frame()
                timings = frame["timings"]

                cv_step_time = 0
                # compute opencv timings
                if cv_flow_active:
                    cv_start_time = time.time()
                    optical_flow(frame["main"])
                    cv_step_time = time.time() - cv_start_time

                print(f"Frame {i}/{total}")

                if i != 0:
                    main_time_list.append(timings["Main"])
                    category_time_list.append(timings["Category"])
                    object_time_list.append(timings["Object"])
                    flow_time_list.append(timings["Optical"])
                    depth_time_list.append(timings["Depth"])
                    total_time = timings["Main"] \
                                + timings["Category"] \
                                + timings["Object"] \
                                + timings["Optical"] \
                                + timings["Depth"]
                    http_time = timings["Http"] - total_time
                    total_time_list.append(total_time)
                    http_time_list.append(http_time)
                    cv_flow_time_list.append(cv_step_time)
                i += 1


            def mean_with_ci(data):
                mean = np.mean(data)
                n = len(data)
                std_err = sem(data)
                h = std_err * t.ppf((1 + confidence) / 2, n - 1)
                return mean, h


            main_data.append(mean_with_ci(main_time_list))
            category_data.append(mean_with_ci(category_time_list))
            object_data.append(mean_with_ci(object_time_list))
            flow_data.append(mean_with_ci(flow_time_list))
            depth_data.append(mean_with_ci(depth_time_list))
            cv_flow_data.append(mean_with_ci(cv_flow_time_list))
            total_data.append(mean_with_ci(total_time_list))
            http_data.append(mean_with_ci(http_time_list))

        finally:
            agent.delete()

    y_axis = [f"{w}x{h}" for w, h in sizes]


    def get_data_and_ci(data_tuple_list):
        data_list = [val for val, ci in data_tuple_list]
        ci_list = [ci for val, ci in data_tuple_list]
        return data_list, ci_list


    a = plt.figure(1)
    data_list, ci_list = get_data_and_ci(main_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list)
    plt.ylabel("time to elaborate main frame")
    a.show()

    b = plt.figure(2)
    data_list, ci_list = get_data_and_ci(category_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list)
    plt.ylabel("time to elaborate category frame")
    b.show()

    c = plt.figure(3)
    data_list, ci_list = get_data_and_ci(object_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list)
    plt.ylabel("time to elaborate object frame")
    c.show()

    d = plt.figure(4)
    plt.ylabel("time to elaborate flow frame")
    data_list, ci_list = get_data_and_ci(flow_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label="unity flow")
    data_list, ci_list = get_data_and_ci(cv_flow_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label="cv flow")
    plt.legend(loc='upper left')
    d.show()

    e = plt.figure(5)
    data_list, ci_list = get_data_and_ci(depth_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list)
    plt.ylabel("time to elaborate depth frame")
    e.show()

    d = plt.figure(4)
    plt.ylabel("HTTP Overhead vs Frame generation")
    data_list, ci_list = get_data_and_ci(http_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label="http overhead")
    data_list, ci_list = get_data_and_ci(total_data)
    plt.errorbar(y=data_list, x=y_axis, yerr=ci_list, label="frame generation")
    plt.legend(loc='upper left')
    d.show()

    input()
