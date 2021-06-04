# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pywt
import matplotlib.pyplot as plt
import numpy as np


def calc_baseline(signal):
        """
        Calculate the baseline of signal.
        Args:
            signal (numpy 1d array): signal whose baseline should be calculated
        Returns:
            baseline (numpy 1d array with same size as signal): baseline of the signal
        """
        ssds = np.zeros((3))

        cur_lp = np.copy(signal)
        iterations = 0
        while True:
            # Decompose 1 level
            lp, hp = pywt.dwt(cur_lp, "db4")

            # Shift and calculate the energy of detail/high pass coefficient
            ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

            # Check if we are in the local minimum of energy function of high-pass signal
            if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
                break

            cur_lp = lp[:]
            iterations += 1

        # Reconstruct the baseline from this level low pass signal up to the original length
        baseline = cur_lp[:]
        for _ in range(iterations):
            baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

        return baseline[: len(signal)]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    # Read input csv file from physionet
    #f = open("/home/luizcrod/venv_dsp/samples/1.csv", "r")
    f = open("/home/luizcrod/venv_dsp/samples/heart.dat", "r")
    lines = f.readlines()
    f.close()

    # Discard the first two lines because of header. Takes either column 1 or 2 from each lines (different signal lead)
    #signal = np.zeros((len(lines) - 2))
    signal = np.zeros((len(lines) ))
    for i in range(len(signal)):
        #print(float(lines[i + 2].split(",")[2]))
        print(float(lines[i]))
        signal[i] = float(lines[i])

    baseline = calc_baseline(signal)

    # Remove baseline from orgianl signal
    ecg_out = signal - baseline

    plt.subplot(2, 1, 1)
    plt.plot(signal, "b-", label="signal")
    plt.plot(baseline, "r-", label="baseline")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(ecg_out, "b-", label="signal - baseline")
    plt.legend()
    plt.show()

# End of Code
