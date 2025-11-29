"""
Visualization of trial-averaged time-frequency transform, per-trial PSDs,
trial-averaged PSD and corresponding aperiodic FOOOF fit, as well as
trial-averaged time-frequency transform following the subtraction of
the aperiodic activity for channels C3 and C4 (or equivalent). 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.style import use
from os.path import join

use("default")


def plot_sub_tfs(
    subject,
    labels,
    tfs,
    psds,
    thresholds,
    channels,
    tf_time,
    freqs,
    band_search_ranges,
    band,
    tf_method,
    savepath,
    screen_res=[1920, 972],
    dpi=96,
    plot_format="pdf",
):
    """
    Create and save figure of the trial-averaged time-frequency maps for
    C3 and C4, the aperiodic fits for each channel and trial, the trial-averaged
    PSDs and the trial-averaged time-frequency maps after subtracting the
    aperiodic activity.

    Parameters
    ----------
    subject: int
             Index of subject for data analysis.
    labels: list
            Labels corresponding to each motor imagery class.
    tfs: 4D numpy array (trial, channel, frequency, time)
         Time frequency tensor for a given subject.
    psds: 3D numpy array (trial, channel, frequency)
          Power spectrum tensor for a given subject.
    thresholds: 2D numpy array (channel, frequency)
                Per-channel aperiodic fit in linear space for a
                given subject.
    channels: two-element list
              List containing the indices of two channels for
              visualization.
    tf_time: 1D numpy array
             Time axis corresponding to the recordings.
    freqs: 1D numpy array
           Frequency axis corresponding to the time-frequency anaysis.
    band_search_ranges: list
                        Extended beta band used during burst extraction
                        for each channel.
    band: str
          String indicating the band used for the plot.
    tf_method: str {"wavelets", "superlets"}
               String indicating the algorithm used for performing
               the time-frequency decomposition.
    Savepath: str
              Parent directory that contains all results.
    screen_res: two-element list, optional
                Number of pixels for specifying the figure size in
                conjunction with "dpi".
                Defaults to [1920, 972].
    dpi: int, optional
         Number of dots per inch for specifying the figure size in
         conjunction with "screen_res".
         Defaults to 96.
    plot_format: str {"pdf", "png"}, optional
                 File format. Prefer "pdf" for editing with vector graphics
                 applications, or "png" for less space usage and better
                 integration with presentations.
                 Defaults to "pdf".
    """

    # Clusters of channels.
    trial_clusters = [
        np.where(labels == np.unique(labels)[0])[0].tolist(),
        np.where(labels == np.unique(labels)[1])[0].tolist(),
    ]

    # Figure initialization.
    fig = plt.figure(
        constrained_layout=True,
        figsize=(screen_res[0] / dpi, screen_res[1] / dpi),
        dpi=dpi,
    )
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.05, left=0.05, right=0.95)
    gs0 = gs[0].subgridspec(
        4, 5, hspace=0.05, wspace=0.05, width_ratios=[2.45, 2.45, 2.45, 2.45, 0.2]
    )

    ax00 = fig.add_subplot(gs0[0, 0])
    ax01 = fig.add_subplot(gs0[0, 1])
    ax02 = fig.add_subplot(gs0[0, 2])
    ax03 = fig.add_subplot(gs0[0, 3])
    axes_tfs = (ax00, ax01, ax02, ax03)
    ax04 = fig.add_subplot(gs0[0, 4])

    ax10 = fig.add_subplot(gs0[1, 0])
    ax11 = fig.add_subplot(gs0[1, 1])
    ax12 = fig.add_subplot(gs0[1, 2])
    ax13 = fig.add_subplot(gs0[1, 3])
    axes_psds = (ax10, ax11, ax12, ax13)

    ax20 = fig.add_subplot(gs0[2, 0])
    ax21 = fig.add_subplot(gs0[2, 1])
    ax22 = fig.add_subplot(gs0[2, 2])
    ax23 = fig.add_subplot(gs0[2, 3])
    axes_fits = (ax20, ax21, ax22, ax23)

    ax30 = fig.add_subplot(gs0[3, 0])
    ax31 = fig.add_subplot(gs0[3, 1])
    ax32 = fig.add_subplot(gs0[3, 2])
    ax33 = fig.add_subplot(gs0[3, 3])
    axes_clean = (ax30, ax31, ax32, ax33)

    ax34 = fig.add_subplot(gs0[3, 4])

    fig_name = join(
        savepath,
        "sub_{}/tf_{}_{}_band.{}".format(subject, tf_method, band, plot_format),
    )

    # Data.
    c3 = channels[0]
    c4 = channels[1]

    # Trial-averaged time-frequency maps of C3 and C4.
    av_tf_l_c3 = np.mean(
        tfs[trial_clusters[0], :, :, :][:, c3, band_search_ranges[c3], :], axis=0
    )
    av_tf_l_c4 = np.mean(
        tfs[trial_clusters[0], :, :, :][:, c4, band_search_ranges[c4], :], axis=0
    )
    av_tf_r_c3 = np.mean(
        tfs[trial_clusters[1], :, :, :][:, c3, band_search_ranges[c3], :], axis=0
    )
    av_tf_r_c4 = np.mean(
        tfs[trial_clusters[1], :, :, :][:, c4, band_search_ranges[c4], :], axis=0
    )
    av_tfs = (av_tf_l_c3, av_tf_l_c4, av_tf_r_c3, av_tf_r_c4)

    # PSDs of C3 and C4 per trial.
    psds_l_c3 = psds[trial_clusters[0], :, :][:, c3, band_search_ranges[c3]]
    psds_l_c4 = psds[trial_clusters[0], :, :][:, c4, band_search_ranges[c4]]
    psds_r_c3 = psds[trial_clusters[1], :, :][:, c3, band_search_ranges[c3]]
    psds_r_c4 = psds[trial_clusters[1], :, :][:, c4, band_search_ranges[c4]]
    all_psds = (psds_l_c3, psds_l_c4, psds_r_c3, psds_r_c4)

    # Aperiodic activity of C3 and C4 fitted on trial-averaged data.
    fits_c3 = thresholds[c3]
    fits_c4 = thresholds[c4]
    all_fits = (fits_c3, fits_c4, fits_c3, fits_c4)

    # Trial-averaged time-frequency maps of C3 and C4 after aperiodic activity removal.
    clean_tf_l_c3 = av_tf_l_c3 - np.tile(fits_c3.reshape(-1, 1), av_tf_l_c3.shape[-1])
    clean_tf_l_c4 = av_tf_l_c4 - np.tile(fits_c4.reshape(-1, 1), av_tf_l_c4.shape[-1])
    clean_tf_r_c3 = av_tf_r_c3 - np.tile(fits_c3.reshape(-1, 1), av_tf_r_c3.shape[-1])
    clean_tf_r_c4 = av_tf_r_c4 - np.tile(fits_c4.reshape(-1, 1), av_tf_r_c4.shape[-1])
    clean_tfs = (clean_tf_l_c3, clean_tf_l_c4, clean_tf_r_c3, clean_tf_r_c4)

    # Plots.
    tfs_lims = np.zeros((2, 4))
    for j, av_tf in enumerate(av_tfs):
        tfs_lims[:, j] = [np.min(av_tf), np.max(av_tf)]
    tfs_titles = [
        "C3, Left hand trials",
        "C4, Left hand trials",
        "C3, Right hand trials",
        "C4, Right hand trials",
    ]

    # Row 1: average tf martices.
    for j, (ax, av_tf) in enumerate(zip(axes_tfs, av_tfs)):
        if j == 0:
            im0 = ax.imshow(
                av_tf,
                aspect="auto",
                origin="lower",
                vmin=np.min(tfs_lims[0, :]),
                vmax=np.max(tfs_lims[1, :]),
                extent=(
                    tf_time[0],
                    tf_time[-1],
                    freqs[band_search_ranges[c3]][0],
                    freqs[band_search_ranges[c3]][-1],
                ),
            )
        elif j == 2:
            ax.imshow(
                av_tf,
                aspect="auto",
                origin="lower",
                vmin=np.min(tfs_lims[0, :]),
                vmax=np.max(tfs_lims[1, :]),
                extent=(
                    tf_time[0],
                    tf_time[-1],
                    freqs[band_search_ranges[c3]][0],
                    freqs[band_search_ranges[c3]][-1],
                ),
            )
        else:
            ax.imshow(
                av_tf,
                aspect="auto",
                origin="lower",
                vmin=np.min(tfs_lims[0, :]),
                vmax=np.max(tfs_lims[1, :]),
                extent=(
                    tf_time[0],
                    tf_time[-1],
                    freqs[band_search_ranges[c4]][0],
                    freqs[band_search_ranges[c4]][-1],
                ),
            )

        ax.set_xlabel("Time (s)", fontsize=14)
        if j == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=14)
        else:
            ax.set_yticklabels([])
        ax.set_title(tfs_titles[j], fontsize=14)

    # Row 2: PSD of each trial and channel.
    power_lims = np.zeros((2, 2, 4))
    for j, (ax, psd) in enumerate(zip(axes_psds, all_psds)):
        if j == 0 or j == 2:
            for ps in psd:
                ax.plot(freqs[band_search_ranges[c3]], np.log10(ps).T)
        else:
            for ps in psd:
                ax.plot(freqs[band_search_ranges[c4]], np.log10(ps).T)

        power_lims[0, :, j] = ax.get_ylim()

        if j == 0:
            ax.set_ylabel("log(Power) (a.u.)", fontsize=14)
        else:
            ax.set_yticklabels([])
        ax.set_xticklabels([])

    # Row 3: aperiodic fits per channel (trials' average).
    for j, (ax, fit, psd) in enumerate(zip(axes_fits, all_fits, all_psds)):
        if j == 0 or j == 2:
            ax.plot(
                freqs[band_search_ranges[c3]], np.log10(fit).T, linewidth="2", c="b"
            )
            ax.plot(
                freqs[band_search_ranges[c3]],
                np.log10(np.mean(psd, axis=0)).T,
                linewidth="2",
                c="k",
            )
        elif j == 1:
            ax.plot(
                freqs[band_search_ranges[c4]], np.log10(fit).T, linewidth="2", c="b"
            )
            ax.plot(
                freqs[band_search_ranges[c4]],
                np.log10(np.mean(psd, axis=0)).T,
                linewidth="2",
                c="k",
            )
        else:
            ax.plot(
                freqs[band_search_ranges[c4]],
                np.log10(fit).T,
                linewidth="2",
                c="b",
                label="aperiodic fit",
            )
            ax.plot(
                freqs[band_search_ranges[c4]],
                np.log10(np.mean(psd, axis=0)).T,
                linewidth="2",
                c="k",
                label="average PSD",
            )
            ax.legend(frameon=False)

        power_lims[1, :, j] = ax.get_ylim()

        ax.set_xlabel("Frequency (Hz)", fontsize=14)
        if j == 0:
            ax.set_ylabel("log(Power) (a.u.)", fontsize=14)
        else:
            ax.set_yticklabels([])

    for ax_psd, ax_fit in zip(axes_psds, axes_fits):
        ax_psd.set_ylim([np.min(power_lims[:, 0, :]), np.max(power_lims[:, 1, :])])
        ax_fit.set_ylim([np.min(power_lims[:, 0, :]), np.max(power_lims[:, 1, :])])

    # Row 4: average tf martices after aperiodic activity subtraction.
    clean_tfs_lims = np.zeros((2, 4))
    for j, clean_tf in enumerate(clean_tfs):
        clean_tfs_lims[:, j] = [np.min(clean_tf), np.max(clean_tf)]

    for j, (ax, clean_tf) in enumerate(zip(axes_clean, clean_tfs)):
        if j == 0:
            im1 = ax.imshow(
                clean_tf,
                aspect="auto",
                origin="lower",
                vmin=np.min(clean_tfs_lims[0, :]),
                vmax=np.max(clean_tfs_lims[1, :]),
                extent=(
                    tf_time[0],
                    tf_time[-1],
                    freqs[band_search_ranges[c3]][0],
                    freqs[band_search_ranges[c3]][-1],
                ),
            )
        elif j == 2:
            ax.imshow(
                clean_tf,
                aspect="auto",
                origin="lower",
                vmin=np.min(clean_tfs_lims[0, :]),
                vmax=np.max(clean_tfs_lims[1, :]),
                extent=(
                    tf_time[0],
                    tf_time[-1],
                    freqs[band_search_ranges[c3]][0],
                    freqs[band_search_ranges[c3]][-1],
                ),
            )
        else:
            ax.imshow(
                clean_tf,
                aspect="auto",
                origin="lower",
                vmin=np.min(clean_tfs_lims[0, :]),
                vmax=np.max(clean_tfs_lims[1, :]),
                extent=(
                    tf_time[0],
                    tf_time[-1],
                    freqs[band_search_ranges[c4]][0],
                    freqs[band_search_ranges[c4]][-1],
                ),
            )

        ax.set_xlabel("Time (s)", fontsize=14)
        if j == 0:
            ax.set_ylabel("Frequency (Hz)", fontsize=14)
        else:
            ax.set_yticklabels([])

    # Saving.
    plt.colorbar(im0, cax=ax04, label="Power (a.u.)")
    plt.colorbar(im1, cax=ax34, label="Periodic activity power (a.u.)")
    plt.savefig(fig_name, dpi=dpi, facecolor="w", edgecolor="w")
