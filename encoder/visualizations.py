from datetime import datetime
from time import perf_counter as timer
import numpy as np
import umap
import visdom
from encoder.data_objects.speaker_verification_dataset import SpeakerVerificationDataset

colormap = np.array([
    [76, 255, 0],
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
], dtype=float) / 255

class Visualizations:
    def __init__(self, env_name=None, update_every=10, server="http://localhost", disabled=False):
        
        self.last_update_timestamp = timer()
        self.update_every = update_every
        self.step_times = []
        self.losses = []
        self.eers = []
        print("Visualizations will update every %d steps." % update_every)

        self.disabled = disabled
        if self.disabled:
            return

        
        now_str = datetime.now().strftime("%d-%m %Hh%M")
        self.env_name = now_str if env_name is None else f"{env_name} ({now_str})"

        try:
            self.vis = visdom.Visdom(server, env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("Visdom server not found. Please run 'visdom' from the CLI.")
        
        
        self.loss_win = None
        self.eer_win = None
        self.implementation_win = None
        self.projection_win = None
        self.implementation_string = ""

    def log_params(self):
        if self.disabled:
            return
        from encoder import params_data, params_model
        param_text = "<b>Model parameters</b>:<br>"
        for pname in (p for p in dir(params_model) if not p.startswith("__")):
            pvalue = getattr(params_model, pname)
            param_text += f"\t{pname}: {pvalue}<br>"
        param_text += "<b>Data parameters</b>:<br>"
        for pname in (p for p in dir(params_data) if not p.startswith("__")):
            pvalue = getattr(params_data, pname)
            param_text += f"\t{pname}: {pvalue}<br>"
        self.vis.text(param_text, opts={"title": "Parameters"})

    def log_dataset(self, dataset: SpeakerVerificationDataset):
        if self.disabled:
            return
        ds_text = f"<b>Number of Speakers</b>: {len(dataset.speakers)}<br>" + dataset.get_logs().replace("\n", "<br>")
        self.vis.text(ds_text, opts={"title": "Dataset"})

    def log_implementation(self, params):
        if self.disabled:
            return
        impl_text = ""
        for key, val in params.items():
            impl_text += f"<b>{key}</b>: {val}<br>"
        self.implementation_string = impl_text
        self.implementation_win = self.vis.text(impl_text, opts={"title": "Training Implementation"})

    def update(self, loss, eer, step):
        
        current_time = timer()
        self.step_times.append(1000 * (current_time - self.last_update_timestamp))
        self.last_update_timestamp = current_time
        self.losses.append(loss)
        self.eers.append(eer)
        print(".", end="")

        
        if step % self.update_every != 0:
            return
        time_info = "Step time: mean %5dms, std %5dms" % (int(np.mean(self.step_times)), int(np.std(self.step_times)))
        print("\nStep %6d   Loss: %.4f   EER: %.4f   %s" % (step, np.mean(self.losses), np.mean(self.eers), time_info))
        if not self.disabled:
            self.loss_win = self.vis.line([np.mean(self.losses)], [step],
                                           win=self.loss_win,
                                           update="append" if self.loss_win else None,
                                           opts={"legend": ["Avg. Loss"],
                                                 "xlabel": "Step",
                                                 "ylabel": "Loss",
                                                 "title": "Loss"})
            self.eer_win = self.vis.line([np.mean(self.eers)], [step],
                                          win=self.eer_win,
                                          update="append" if self.eer_win else None,
                                          opts={"legend": ["Avg. EER"],
                                                "xlabel": "Step",
                                                "ylabel": "EER",
                                                "title": "Equal Error Rate"})
            if self.implementation_win is not None:
                self.vis.text(self.implementation_string + f"<b>{time_info}</b>",
                              win=self.implementation_win,
                              opts={"title": "Training Implementation"})
        self.losses.clear()
        self.eers.clear()
        self.step_times.clear()

    def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None, max_speakers=10):
        import matplotlib.pyplot as plt
        max_speakers = min(max_speakers, len(colormap))
        embeds = embeds[:max_speakers * utterances_per_speaker]
        n_speakers = len(embeds) // utterances_per_speaker
        gt_labels = np.repeat(np.arange(n_speakers), utterances_per_speaker)
        colors = [colormap[i] for i in gt_labels]
        reducer = umap.UMAP()
        proj = reducer.fit_transform(embeds)
        plt.scatter(proj[:, 0], proj[:, 1], c=colors)
        plt.gca().set_aspect("equal", "datalim")
        plt.title("UMAP Projection (Step %d)" % step)
        if not self.disabled:
            self.projection_win = self.vis.matplot(plt, win=self.projection_win)
        if out_fpath is not None:
            plt.savefig(out_fpath)
        plt.clf()

    def save(self):
        if not self.disabled:
            self.vis.save([self.env_name])
