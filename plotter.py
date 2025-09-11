import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 40,
    'legend.edgecolor': 'white',
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 15,
    'xtick.minor.size': 10,
    'ytick.major.size': 15,
    'ytick.minor.size': 10,
    'xtick.major.width': 3,
    'xtick.minor.width': 3,
    'ytick.major.width': 3,
    'ytick.minor.width': 3,
    'axes.linewidth': 3,
    'figure.max_open_warning': 200,
    'lines.linewidth': 5
})

class Plotter:
    def __init__(self, print_dir='', end_name='',sector=0):
        self.print_dir = print_dir
        self.end_name = end_name
        self.sector=sector

    def plot_hits(self, all_hits, track_hits, predictions=None, event_idx=0, noth=False):
        """
        all_hits: 2D numpy array (36, 112)
        track_hits: 2D numpy array (36, 112)
        predictions: optional 2D numpy array (36, 112)
        """
        ncols = 3 if predictions is not None else 2
        fig, axs = plt.subplots(1, ncols, figsize=(20 * ncols, 20), tight_layout=True)

        predstr=""

        if ncols == 2:
            ax_all, ax_track = axs
        else:
            ax_all, ax_track, ax_pred = axs

        im1 = ax_all.imshow(all_hits, origin='lower', aspect='auto',
                            extent=[1, 112, 1, 36], cmap='Blues', vmin=0, vmax=1)
        ax_all.set_title(f"All DC::tdc Hits (Sector {self.sector}, Event {event_idx})")
        fig.colorbar(im1, ax=ax_all, fraction=0.046, pad=0.04)

        im2 = ax_track.imshow(track_hits, origin='lower', aspect='auto',
                              extent=[1, 112, 1, 36], cmap='Blues', vmin=0, vmax=1)
        ax_track.set_title("TBHits (Track)")
        fig.colorbar(im2, ax=ax_track, fraction=0.046, pad=0.04)

        if predictions is not None:
            im3 = ax_pred.imshow(predictions, origin='lower', aspect='auto',
                                 extent=[1, 112, 1, 36], cmap='Blues', vmin=0, vmax=1)
            ax_pred.set_title("Predictions")
            fig.colorbar(im3, ax=ax_pred, fraction=0.046, pad=0.04)
            predstr="_wpred"

        for ax in axs:
            ax.set_xlabel("Wire (1–112)")
            ax.set_ylabel("Layer (1–36)")

        if noth==True:
            predstr=predstr+'_noThreshold'

        outname = f"{self.print_dir}sector{self.sector}_event{event_idx}{self.end_name}{predstr}.png"
        plt.savefig(outname)
        plt.close(fig)

    def plotTrainLoss(self,tracker):
        # train_losses = [x["train_loss"].item() for x in trainer.logged_metrics_history]
        # val_losses   = [x["val_loss"].item() for x in trainer.logged_metrics_history]
        train_losses=tracker.train_losses
        val_losses=tracker.val_losses

        plt.figure(figsize=(20,20))
        plt.plot(train_losses, label='Train',color='royalblue')
        plt.plot(val_losses, label='Test',color='firebrick') 
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        outname = f"{self.print_dir}loss_sector{self.sector}{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_efficiency_background(self, thresholds, signal_eff, background_rej):
        plt.figure(figsize=(20,20))
        plt.axhline(0.95, color="dimgrey", linestyle="--")
        plt.axhline(0.9, color="grey", linestyle="--")
        plt.axhline(1.0, color="black", linestyle="--")
        plt.scatter(thresholds, signal_eff, label='Signal Efficiency', color='royalblue',s=250)
        plt.scatter(thresholds, background_rej, label='Background Rejection', color='firebrick',s=250)
        plt.xlabel("Threshold")
        plt.ylabel("Metrics")
        plt.title("Metrics vs Threshold")
        plt.ylim(0.7,1.1)
        plt.legend()
        plt.grid(True)
        outname = f"{self.print_dir}mets_sector{self.sector}{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

    def plot_metric_vs_param(self,results, param_name, metric_idx, ylabel, default_params):
        """metric_idx: 1=prediction rate, 2=signal_eff, 3=background_rej"""

        plt.figure(figsize=(20, 20))

        xs, ys, ys2 = [], [], []
        xsb, ysb, ysb2 =[], [], []
        yname = ylabel
        xsnamex=''
        xsnamey=''

        for params, (thr, rate, eff, rej) in results.items():
            params_dict = dict(params)

            # only include runs where all other params equal defaults
            valid = True
            for k, v in default_params.items():
                if k == param_name:
                    continue
                if params_dict[k] != v:
                    valid = False
                    break
                  
            if not valid:
                continue
              
            # collect the x and y values
            if param_name!="kernel_size":
              xs.append(params_dict[param_name])
            else:
              xs.append(params_dict[param_name][0])
              xsb.append(params_dict[param_name][1])
              xsnamex=' (X)'
              xsnamey=' (Y)'


            if metric_idx == 1:
                ys.append(rate)   # already a scalar
            elif metric_idx == 2:
                yname = 'Metrics'
                ys.append(eff)
                ys2.append(rej)
                ysb.append(eff)
                ysb2.append(rej)
            elif metric_idx == 3:
                ys.append(rej)
                ysb.append(rej)

        # remove duplicates while keeping first occurrence
        def remove_duplicates(x_vals, y_vals, y2_vals):
            seen = {}
            for i, x in enumerate(x_vals):
                if x not in seen:
                    seen[x] = i
            idxs = sorted(seen.values())
            x_new = [x_vals[i] for i in idxs]
            y_new = [y_vals[i] for i in idxs]
            # only slice y2_vals if it's not None AND has same length as x_vals
            if len(y2_vals) == len(x_vals):
                y2_new = [y2_vals[i] for i in idxs]
            else:
                y2_new = []
            return x_new, y_new, y2_new
        
        #have duplicates due to kernel size, bit annoying dealing with a tuple
        if len(xsb)!=0:
            xs, ys, ys2 = remove_duplicates(xs, ys, ys2)
            xsb, ysb, ysb2 = remove_duplicates(xsb, ys, ys2)

        plt.scatter(xs, ys, label=ylabel, color='royalblue', s=750)

        if len(ys2) != 0:
            plt.scatter(xs, ys2, label='Background Rejection', color='firebrick', s=750)
            plt.axhline(0.95, color="dimgrey", linestyle="--")
            plt.axhline(0.9, color="grey", linestyle="--")
            plt.axhline(1.0, color="black", linestyle="--")
            plt.ylim(0.7, 1.1)
            plt.legend()

        plt.xlabel(param_name+xsnamex)
        plt.ylabel(yname)
        plt.title(f"{yname} vs {param_name}")
        plt.grid(True)

        outname = f"{self.print_dir}{param_name}_{yname.replace(' ', '_')}_sector{self.sector}{self.end_name}.png"
        plt.savefig(outname)
        plt.close()

        if len(xsb)!=0:
          plt.figure(figsize=(20, 20))
          plt.scatter(xsb, ysb, label=ylabel, color='royalblue', s=750)

          if len(ys2) != 0:
              plt.scatter(xsb, ysb2, label='Background Rejection', color='firebrick', s=750)
              plt.axhline(0.95, color="dimgrey", linestyle="--")
              plt.axhline(0.9, color="grey", linestyle="--")
              plt.axhline(1.0, color="black", linestyle="--")
              plt.ylim(0.7, 1.1)
              plt.legend()

          plt.xlabel(param_name+xsnamey)
          plt.ylabel(yname)
          plt.title(f"{yname} vs {param_name}")
          plt.grid(True)

          outname = f"{self.print_dir}{param_name}y_{yname.replace(' ', '_')}_sector{self.sector}{self.end_name}.png"
          plt.savefig(outname)
          plt.close()