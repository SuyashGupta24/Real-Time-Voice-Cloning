from encoder.data_objects.random_cycler import randcycle
from encoder.data_objects.utterance import Utterance
from pathlib import Path

class Speaker:
    def __init__(self, root: Path):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None

    def _load_utterances(self):
        with self.root.joinpath("_sources.txt").open("r") as f:
            entries = [line.split(",") for line in f]
        entries = {fname: wpath for fname, wpath in entries}
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in entries.items()]
        self.utterance_cycler = randcycle(self.utterances)

    def random_partial(self, count, n_frames):
        """
        Retrieves a set of distinct partial utterances in a random sequence.
        Each complete utterance is sampled such that all are used at least once every two cycles.
        
        Parameters:
            count: Number of unique partials to fetch.
            n_frames: Frame length for each partial utterance.
            
        Returns:
            A list of tuples, each starting with an Utterance object followed by its partial data.
        """
        if self.utterances is None:
            self._load_utterances()

        selected = self.utterance_cycler.sample(count)
        return [(u,) + u.random_partial(n_frames) for u in selected]
