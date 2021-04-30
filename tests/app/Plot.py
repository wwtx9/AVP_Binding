import matplotlib.pyplot as plt
import numpy as np
colors = np.random.randint(0, 255, (1000, 3))

def plotLocError(y_data, exid):
    #y_data = np.array(y_data)
    n = len(y_data)
    x = range(1, n + 1)

    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Steps")
    plt.ylabel("Localization Error")
    # plt.yscale('log')
    plt.title("Exp {0}: Position Norm Error".format(exid))
    plt.show()
    fig.savefig('locerror.png', dpi=fig.dpi)

def plotAllKeyPointsLocError(Kpsloc_error_array, exid):
    n = len(Kpsloc_error_array)
    # color = np.random.randint(0, 255, (n, 3))
    count = 0
    fig = plt.figure()
    for i in range(n):
        non_zero_idex_i = np.where(Kpsloc_error_array[i] == 0)[0]
        non_zero_idex_i = non_zero_idex_i[0]
        if non_zero_idex_i > 10:
            count += 1
            x = range(1, non_zero_idex_i + 1)
            plt.plot(x, Kpsloc_error_array[i, 0:non_zero_idex_i],   linestyle="--", marker="*", linewidth=1.0)
            if Kpsloc_error_array[i, 0] > 1.5:
                print(i)
            # if count == 4:
            #     break

    plt.xlabel("Steps")
    plt.ylabel("Localization Error")
    plt.title("Exp {0}: Localization Error".format(exid))
    plt.show()
    # fig.savefig('Localizationerror.png', dpi=fig.dpi)

def plotAllKeyPointsTrace(kpsTrace_array, exid):
    n = len(kpsTrace_array)

    fig = plt.figure()
    count = 0
    for i in range(n):
        non_zero_idex_i = np.where(kpsTrace_array[i] == 0)[0]
        non_zero_idex_i = non_zero_idex_i[0]
        if non_zero_idex_i > 10:
            x = range(1, non_zero_idex_i+1)
            count += 1
            plt.plot(x, kpsTrace_array[i, 0:non_zero_idex_i],  linestyle="--", marker="*", linewidth=1.0)
            # if count == 4:
            #     break
    print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.xlabel("Steps")
    plt.ylabel("trace")
    plt.yscale('log')
    plt.title("Exp {0}: trace".format(exid))
    plt.show()
    # fig.savefig('Trace.png', dpi=fig.dpi)

def plotTrace(y_data, exid):
    # y_data = np.array(y_data)
    n = len(y_data)
    x = range(1, n + 1)

    fig = plt.figure()
    plt.plot(x, y_data, color="g", linestyle="--", marker="*", linewidth=1.0)
    # plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.xlabel("Steps")
    plt.ylabel("trace")
    plt.yscale('log')
    plt.title("Exp {0}: trace".format(exid))
    plt.show()
    fig.savefig('Trace.png', dpi=fig.dpi)
